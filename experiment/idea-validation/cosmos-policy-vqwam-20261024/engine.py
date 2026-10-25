"""Cosmos adapter for packed MicroscalingVQ weights and decoded-BF16 Linear GEMM."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import types
from enum import Enum


FORMAT = "cosmos_vqwam_deployment_v1"
ARM = "ours_vqwam"
EXPECTED_TARGETS = 280
EXPECTED_PARITY_QUERIES = 80  # 40 frozen cases x 2 sampler seeds.
_TREE_REF = object()


def _compute_guard(root):
    job_id, nodefile = os.environ.get("PBS_JOBID"), os.environ.get("PBS_NODEFILE")
    host = socket.gethostname().split(".")[0]
    if not job_id or not nodefile or "login" in host.lower():
        raise RuntimeError("Cosmos VQ engine requires an allocated PBS compute node")
    nodes_path = Path(nodefile)
    if not nodes_path.is_file():
        raise RuntimeError(f"PBS_NODEFILE is missing: {nodes_path}")
    nodes = {line.split(".")[0] for line in nodes_path.read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError(f"Current host {host} is not listed in PBS_NODEFILE")
    if os.environ.get("ROOT") and Path(os.environ["ROOT"]).resolve() != Path(root).resolve():
        raise RuntimeError("Runtime root differs from the PBS ROOT environment")


def _protocol_value(p):
    if isinstance(p, (str, Path)):
        value = json.loads(Path(p).read_text(encoding="utf-8"))
    else:
        value = p
    if not isinstance(value, dict):
        raise TypeError("p must be a protocol mapping or JSON path")
    return value.get("protocol")


def _target_rows(modules):
    return [{"name": name,
             "shape": [int(module.out_features), int(module.in_features)],
             "weight_numel": int(module.out_features) * int(module.in_features)}
            for name, module in sorted(modules.items())]


def _tensor_version(value):
    try:
        return value._version
    except RuntimeError:
        return None


def _tensor_storage_bytes(values):
    seen, total = set(), 0

    def visit(value):
        nonlocal total
        if hasattr(value, "untyped_storage"):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(values)
    return total


def _safe_bank_file(value):
    if not isinstance(value, str):
        raise ValueError("bank file path must be a relative string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe bank file path: {value!r}")
    return path


def validate_manifest(manifest, modules, expected_protocol=None):
    """Validate the metadata-only part of the frozen 280-Linear handoff."""
    if manifest.get("format") != FORMAT:
        raise ValueError(f"Expected deployment format {FORMAT!r}")
    if manifest.get("completed_updates") != 8:
        raise ValueError("Expected the frozen eight-update MicroscalingVQ artifact")
    if not isinstance(manifest.get("source_job"), str) or not manifest["source_job"]:
        raise ValueError("Artifact source_job is required")
    if expected_protocol is not None and manifest.get("protocol") != expected_protocol:
        raise ValueError("Artifact protocol does not match the runtime protocol")
    if manifest.get("rotation_file") != "rotation.pt":
        raise ValueError("Expected the shared rotation.pt payload")

    rows = _target_rows(modules)
    if len(rows) != EXPECTED_TARGETS:
        raise ValueError(f"Cosmos target scope changed: expected {EXPECTED_TARGETS}, got {len(rows)}")
    if manifest.get("target_modules") != rows:
        raise ValueError("Artifact target_modules differ from the sorted runtime target Linears")
    entries = manifest.get("modules")
    if not isinstance(entries, list) or len(entries) != len(rows):
        raise ValueError("Artifact modules must list every target bank in canonical order")
    for index, (entry, row) in enumerate(zip(entries, rows)):
        if (not isinstance(entry, dict) or entry.get("name") != row["name"]
                or entry.get("shape") != row["shape"]
                or entry.get("file") != f"bank/{index:04d}.pt"):
            raise ValueError(f"Artifact bank manifest mismatch at index {index}")
        _safe_bank_file(entry["file"])
    return rows


def _fastwam_helpers(p):
    """Load the frozen helper snapshot copied by the PBS job or the local legacy tree."""
    project = Path(__file__).resolve().parents[3]
    protocol = json.loads(Path(p).read_text(encoding="utf-8")) if isinstance(p, (str, Path)) else p
    engine_value = protocol.get("source_paths", {}).get("engine") if isinstance(protocol, dict) else None
    engine_source = Path(engine_value) if engine_value else None
    local_source = project / "experiment" / "idea-validation" / "fastwam-vq-r128-c80-c-decoder-buffer"
    candidates = []
    for entry in sys.path:
        base = Path(entry)
        if base.name == "engine_sources" and base.is_dir():
            candidates.append(base)
    candidates.extend((Path(__file__).resolve().parent, engine_source, local_source))
    candidates.extend(Path(entry) for entry in sys.path)
    required = ("fused_input.py", "layer_runtime.py", "fused_vq.py", "decoder_runtime.py")
    selected = next((path.resolve() for path in candidates if path is not None and path.is_dir()
                     and all((path / name).is_file() for name in required)), None)
    if selected is None:
        raise FileNotFoundError("No frozen engine source directory contains " + ", ".join(required))

    def load(name):
        source = selected / f"{name}.py"
        qualified = f"_cosmos_vqwam_{name}"
        module = sys.modules.get(qualified)
        if module is not None:
            if Path(module.__file__).resolve() != source:
                raise RuntimeError(f"Cached {name} module points outside the selected snapshot")
            return module
        spec = importlib.util.spec_from_file_location(qualified, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load frozen helper {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        prior_path = sys.path[:]
        try:
            spec.loader.exec_module(module)
        finally:
            # The legacy layer_runtime adds its adjacent tuning folder to sys.path.
            # Keep the current Cosmos root/runtime resolution ahead of that folder.
            sys.path[:] = prior_path
        return module

    return tuple(load(name) for name in ("fused_input", "layer_runtime", "fused_vq", "decoder_runtime"))


def _micro_codebook(runtime):
    module_name = type(runtime.controller).__module__
    quantization = sys.modules.get(module_name) or importlib.import_module(module_name)
    helpers = getattr(quantization, "_helpers", None)
    if not callable(helpers):
        raise RuntimeError("Cosmos controller does not expose its frozen micro_codebook helper")
    return helpers()[1]


def _module_device(modules, torch):
    if any(module.weight is None for module in modules.values()):
        raise ValueError("Target Linear weights must be present before packed-arm installation")
    devices = {module.weight.device for module in modules.values()}
    if len(devices) != 1:
        raise ValueError(f"All target Linears must share one device, got {devices}")
    device = next(iter(devices))
    if device.type != "cuda":
        raise RuntimeError("Cosmos VQ install requires CUDA-resident runtime Linears")
    return device


class _Workspace:
    def __init__(self, torch, device, weight_elements):
        self.torch, self.device = torch, device
        self.weight_elements = weight_elements
        self.scratch = torch.empty(weight_elements + (weight_elements & 1),
                                   dtype=torch.bfloat16, device=device)
        self.a4 = torch.empty(0, dtype=torch.bfloat16, device=device)
        self.activation_elements = 0

    def ensure(self, elements):
        torch = self.torch
        self.activation_elements = max(self.activation_elements, elements)
        required = max(self.weight_elements, 4 * self.activation_elements)
        required += required & 1
        if self.scratch.numel() < required:
            self.scratch = torch.empty(required, dtype=torch.bfloat16, device=self.device)
        if self.a4.numel() < self.activation_elements:
            self.a4 = torch.empty(self.activation_elements, dtype=torch.bfloat16, device=self.device)
        fp32 = self.scratch.view(torch.float32)
        if fp32.numel() < 2 * elements:
            raise RuntimeError("Shared BF16 scratch cannot hold both FP32 H/U activation buffers")
        return fp32[:elements], fp32[elements:2 * elements], self.a4[:elements]


class _Banks:
    def __init__(self, runtime, modules, encodings, transforms, rotation, kernels):
        import torch

        fused_input, layer_runtime, fused_vq, decoder_runtime = kernels
        self.runtime, self.modules = runtime, modules
        self.fused_input, self.layer_runtime, self.fused_vq = fused_input, layer_runtime, fused_vq
        self.decoder_runtime = decoder_runtime
        self.encodings, self.transforms, self.prepared = {}, {}, {}
        self.decoder_decisions, self.decoder_receipt = {}, None
        self.rotation = rotation
        self.device = rotation.device
        self.calls, self.decoded_calls = Counter(), Counter()
        self.seen = set()
        self.max_weight_elements = 0
        for name, module in modules.items():
            encoding = encodings[name]
            prepared = fused_vq.prepare(encoding)
            if (prepared.rows, prepared.width) != (module.out_features, module.in_features):
                raise ValueError(f"Packed shape mismatch at {name}")
            transform = transforms[name]
            width = module.in_features
            if (width < 1 or width > 14336 or transform["input_dim"] != width
                    or transform["block"] != 128 or not transform["hadamard"]
                    or transform["scale"].dtype != torch.float32
                    or transform["signs"].dtype not in (torch.int8, torch.float32)
                    or transform["scale"].numel() != width
                    or transform["signs"].numel() != width):
                raise ValueError(f"Invalid FP32 Smooth/H128 transform at {name}")
            self.encodings[name], self.transforms[name], self.prepared[name] = encoding, transform, prepared
            self.max_weight_elements = max(self.max_weight_elements, prepared.rows * prepared.width)
            if module.bias is not None and (module.bias.device != self.device
                    or module.bias.dtype != torch.bfloat16
                    or tuple(module.bias.shape) != (module.out_features,)):
                raise ValueError(f"Expected a BF16 CUDA bias at {name}")
        self.torch = torch
        self.default_workspace = _Workspace(torch, self.device, self.max_weight_elements)
        self.workspace = self.default_workspace
        self.graph_workspaces = {}
        self.original_forwards = {name: module.forward for name, module in modules.items()}
        for name, module in modules.items():
            module.forward = self.forward(name, module)

    def activate_graph_workspace(self, key):
        workspace = self.graph_workspaces.get(key)
        if workspace is None:
            workspace = _Workspace(self.torch, self.device, self.max_weight_elements)
            self.graph_workspaces[key] = workspace
        self.workspace = workspace
        return workspace

    def _operand(self, x, name):
        import torch
        values = x.contiguous()
        width, elements = int(values.shape[-1]), values.numel()
        if width != self.transforms[name]["input_dim"] or elements == 0:
            raise ValueError(f"Activation width or row count mismatch at {name}")
        hflat, zflat, operand_flat = self.workspace.ensure(elements)
        h, z = hflat.view(values.shape), zflat.view(values.shape)
        operand = operand_flat.view(values.shape)
        rows = elements // width
        transform = self.transforms[name]
        full_blocks = width // 128
        starts, widths, inverses = self.fused_input._tail_segments(width)
        if full_blocks:
            self.fused_input._smooth_h_kernel[(rows * full_blocks,)](
                values, transform["scale"], transform["signs"], h, width,
                0, 128, full_blocks, 1.0 / (128 ** 0.5), 128,
                num_warps=4, enable_fp_fusion=False,
            )
        for start, segment_width, inverse in zip(starts, widths, inverses):
            self.fused_input._smooth_h_kernel[(rows,)](
                values, transform["scale"], transform["signs"], h, width,
                start, segment_width, 1, inverse, 128,
                num_warps=4, enable_fp_fusion=False,
            )
        full = full_blocks * 128
        if full:
            h_head = h[..., :full].reshape(*h.shape[:-1], full_blocks, 128)
            z_head = z[..., :full].reshape(*z.shape[:-1], full_blocks, 128)
            with torch.autocast(device_type="cuda", enabled=False):
                torch.matmul(h_head, self.rotation, out=z_head)
            if full < width:
                z[..., full:].copy_(h[..., full:])
        else:
            z.copy_(h)
        block = self.fused_vq.triton.next_power_of_2(width)
        self.fused_input._a4_kernel[(rows,)](
            z, operand, width, block,
            num_warps=8 if block >= 8192 else 4, enable_fp_fusion=False,
        )
        return operand

    def forward(self, name, module):
        import torch
        import torch.nn.functional as functional

        prepared = self.prepared[name]

        @torch.no_grad()
        def run(value):
            self.seen.add(name)
            self.calls[name] += 1
            operand = self._operand(value, name)
            weight = self.decode(prepared, self.workspace.scratch)
            self.decoded_calls[name] += 1
            with torch.autocast(device_type="cuda", enabled=False):
                return functional.linear(operand, weight, module.bias)

        return run

    def decode(self, prepared, scratch):
        shape = (int(prepared.rows), int(prepared.width))
        selected = self.decoder_decisions.get(shape, "reference")
        if selected == "reference":
            self.decoder_runtime._launch_reference(prepared, scratch, self.fused_vq._decode_kernel)
        else:
            self.decoder_runtime._launch_candidate(
                prepared, scratch, self.decoder_runtime._CONFIGS[selected])
        return scratch[:prepared.rows * prepared.width].view(prepared.rows, prepared.width)


def _cross_kv_name(name):
    parts = tuple(part.lower() for part in name.replace("/", ".").split(".") if part)
    cross = any("cross" in part and "attn" in part for part in parts)
    projection = parts[-1] if parts else ""
    return cross and projection in {"k", "v", "to_k", "to_v", "k_proj", "v_proj", "key", "value"}


class _CrossKVCache:
    """Query-local K/V reuse, enabled only after repeated exact-input checks."""
    def __init__(self, modules):
        self.modules = {name: module for name, module in modules.items() if _cross_kv_name(name)}
        self.slots, self.active, self.graph_capture_active = {}, False, False
        self.query_id = 0
        self.denoise_calls = 0
        self.hits = Counter()
        for name, module in self.modules.items():
            original = module.forward
            slot = {"original": original, "input": None, "version": None, "snapshot": None,
                    "buffer": None, "buffers": {}, "valid": False, "cacheable": True,
                    "matches": 0, "calls": 0}
            self.slots[name] = slot

            def cached(value, *args, _name=name, _slot=slot, **kwargs):
                return self._forward(_name, _slot, value, args, kwargs)

            module.forward = cached

    @contextmanager
    def capture_scope(self):
        if self.graph_capture_active:
            raise RuntimeError("Cosmos VQ graph-capture scope is already active")
        self.graph_capture_active = True
        try:
            yield
        finally:
            self.graph_capture_active = False

    def _forward(self, name, slot, value, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if not self.active or not slot["cacheable"]:
            return slot["original"](value, *args, **kwargs)
        if self.graph_capture_active:
            if slot["valid"] and slot["matches"] > 0:
                slot["calls"] += 1
                self.hits[name] += 1
                return slot["buffer"]
            return slot["original"](value, *args, **kwargs)
        slot["calls"] += 1
        if not slot["valid"]:
            result = slot["original"](value, *args, **kwargs)
            slot["input"] = value
            slot["version"] = _tensor_version(value)
            slot["snapshot"] = (value.detach().clone() if slot["version"] is None else None)
            shape_key = (tuple(result.shape), result.dtype, result.device)
            buffer = slot["buffers"].get(shape_key)
            if buffer is None:
                buffer = result.detach().clone(memory_format=__import__("torch").preserve_format)
                slot["buffers"][shape_key] = buffer
            else:
                buffer.copy_(result)
            slot["buffer"] = buffer
            slot["valid"] = True
            return slot["buffer"]
        if self._same_input(value, slot):
            slot["matches"] += 1
            self.hits[name] += 1
            return slot["buffer"]
        slot["cacheable"] = False
        return slot["original"](value, *args, **kwargs)

    @staticmethod
    def _same_input(value, slot):
        previous = slot["input"]
        if not hasattr(value, "shape") or not hasattr(previous, "shape"):
            return value is previous
        if (value.shape != previous.shape or value.dtype != previous.dtype
                or value.device != previous.device):
            return False
        snapshot = slot["snapshot"]
        if snapshot is not None:
            return bool(value.equal(snapshot))
        previous_version = slot["version"]
        if _tensor_version(previous) != previous_version:
            return False
        if value.data_ptr() == previous.data_ptr():
            return _tensor_version(value) == previous_version
        return bool(value.equal(previous))

    def begin_query(self):
        if self.active:
            raise RuntimeError("Cosmos VQ K/V cache query is already active")
        self.query_id += 1
        self.denoise_calls = 0
        self.active = True
        for slot in self.slots.values():
            slot.update(valid=False, cacheable=True, matches=0, calls=0,
                        input=None, version=None, snapshot=None)

    def invalidate(self):
        if self.active:
            raise RuntimeError("Cannot invalidate Cosmos VQ K/V cache during an active query")
        for slot in self.slots.values():
            slot.update(valid=False, cacheable=True, matches=0, calls=0,
                        input=None, version=None, snapshot=None)

    def end_query(self):
        self.active = False
        for slot in self.slots.values():
            slot.update(input=None, version=None, snapshot=None)

    def stable(self):
        seen = [slot for slot in self.slots.values() if slot["calls"] > 0]
        return all(not slot["cacheable"] or slot["matches"] > 0 for slot in seen)

    def signature(self):
        return tuple((name, "cached" if slot["cacheable"] and slot["matches"] else "dynamic")
                     for name, slot in sorted(self.slots.items()))

    def metadata(self):
        buffers = [buffer for slot in self.slots.values() for buffer in slot["buffers"].values()]
        return {"enabled": bool(self.modules), "candidate_modules": len(self.modules),
                "query_id": self.query_id, "active": self.active,
                "confirmed_hits": sum(self.hits.values()),
                "buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in buffers),
                "dynamic_modules": sorted(name for name, slot in self.slots.items()
                                          if not slot["cacheable"])}


def _tree_fields(value):
    try:
        fields = vars(value)
    except (TypeError, AttributeError):
        return None
    return fields if fields else None


def _stable_tree_value(torch, value):
    if (value is None or isinstance(value, (Enum, type, bool, int, float, complex, str, bytes))):
        return True
    for name in ("dtype", "device", "layout", "memory_format", "qscheme"):
        candidate = getattr(torch, name, None)
        if isinstance(candidate, type) and isinstance(value, candidate):
            return True
    return False


def _tree_kind(torch, value):
    if isinstance(value, torch.Tensor):
        return "tensor", None
    if _stable_tree_value(torch, value):
        return "leaf", None
    if isinstance(value, dict):
        return "dict", None
    if isinstance(value, (tuple, list)):
        return "sequence", None
    fields = _tree_fields(value)
    return ("object", fields) if fields is not None else ("opaque", None)


def _clone_tree(torch, value):
    # Seed tensor/opaque leaves, then let deepcopy's memo preserve shared and cyclic containers.
    import copy

    memo, seen = {}, set()

    def seed(item):
        kind, fields = _tree_kind(torch, item)
        identity = id(item)
        if kind == "tensor":
            if identity not in memo:
                memo[identity] = item.clone(memory_format=torch.preserve_format)
        elif kind == "leaf" or kind == "opaque":
            memo[identity] = item
        elif identity not in seen:
            seen.add(identity)
            if kind == "dict":
                for key, child in item.items():
                    memo[id(key)] = key
                    seed(child)
            elif kind == "sequence":
                for child in item:
                    seed(child)
            else:
                for child in fields.values():
                    seed(child)

    seed(value)
    return copy.deepcopy(value, memo)


def _copy_tree(torch, target, source, _seen=None):
    if _seen is None:
        _seen = ({}, {})
    target_seen, source_seen = _seen
    target_kind, target_fields = _tree_kind(torch, target)
    source_kind, source_fields = _tree_kind(torch, source)
    if target_kind != source_kind:
        raise ValueError("CUDA graph input tree kind changed")
    if target_kind == "tensor":
        if (target.shape, target.dtype, target.device) != (source.shape, source.dtype, source.device):
            raise ValueError("CUDA graph input tensor shape, dtype, or device changed")
        left_id, right_id = id(target), id(source)
        if left_id in target_seen or right_id in source_seen:
            if target_seen.get(left_id) != right_id or source_seen.get(right_id) != left_id:
                raise ValueError("CUDA graph input tensor alias topology changed")
            return
        target_seen[left_id], source_seen[right_id] = right_id, left_id
        if target.data_ptr() != source.data_ptr():
            target.copy_(source)
        return
    if target_kind == "leaf":
        if type(target) is not type(source) or not _stable_values_equal(target, source):
            raise ValueError("CUDA graph input scalar changed")
        return
    if target_kind == "opaque":
        if target is not source:
            raise ValueError("CUDA graph input opaque object identity changed")
        return

    left_id, right_id = id(target), id(source)
    if left_id in target_seen or right_id in source_seen:
        if target_seen.get(left_id) != right_id or source_seen.get(right_id) != left_id:
            raise ValueError("CUDA graph input object alias topology changed")
        return
    target_seen[left_id], source_seen[right_id] = right_id, left_id
    if type(target) is not type(source):
        raise ValueError("CUDA graph input object type changed")
    if target_kind == "dict":
        if target.keys() != source.keys():
            raise ValueError("CUDA graph input mapping changed")
        for key in target:
            _copy_tree(torch, target[key], source[key], _seen)
    elif target_kind == "sequence":
        if len(target) != len(source):
            raise ValueError("CUDA graph input sequence changed")
        for left, right in zip(target, source):
            _copy_tree(torch, left, right, _seen)
    else:
        if target_fields.keys() != source_fields.keys():
            raise ValueError("CUDA graph input object changed")
        for key in target_fields:
            _copy_tree(torch, getattr(target, key), getattr(source, key), _seen)


def _stable_values_equal(value, snapshot):
    if value is snapshot:
        return True
    try:
        return bool(value == snapshot)
    except (TypeError, ValueError, RuntimeError):
        return False


def _signature(torch, value, _seen=None):
    if _seen is None:
        _seen = {}
    kind, fields = _tree_kind(torch, value)
    if kind == "tensor":
        identity = id(value)
        if identity in _seen:
            return ("tensor_ref", _seen[identity])
        token = len(_seen)
        _seen[identity] = token
        return ("tensor", token, tuple(value.shape), str(value.dtype), str(value.device))
    if kind == "leaf":
        return (type(value).__module__, type(value).__qualname__, value)
    if kind == "opaque":
        return (type(value).__module__, type(value).__qualname__, "identity", id(value))
    identity = id(value)
    if identity in _seen:
        return (_TREE_REF, _seen[identity])
    token = len(_seen)
    _seen[identity] = token
    if kind == "dict":
        return ("dict", type(value).__module__, type(value).__qualname__, token,
                tuple((key, _signature(torch, item, _seen)) for key, item in value.items()))
    if kind == "sequence":
        return ("sequence", type(value).__module__, type(value).__qualname__, token,
                tuple(_signature(torch, item, _seen) for item in value))
    return ("object", type(value).__module__, type(value).__qualname__, token,
            tuple((key, _signature(torch, item, _seen)) for key, item in fields.items()))


class _DenoiseGraphs:
    def __init__(self, runtime, banks, cache):
        import torch

        self.torch, self.runtime, self.banks, self.cache = torch, runtime, banks, cache
        self.model = runtime.model
        self.original_denoise = self.model.denoise
        self.original_net_forward = self.model.net.forward
        self.original_generate = self.model.generate_samples_from_batch
        self.records, self.calls = {}, Counter()
        self.condition_ref = None
        self.condition_version = None
        self.capture_errors = []
        self.graph_status = "pending_graph_validation"
        self.graph_enabled = False

        @contextmanager
        def query_scope():
            owned = not self.cache.active
            if owned:
                self.runtime.begin_query()
            try:
                yield
            finally:
                if owned:
                    self.runtime.end_query()

        def generate(data_batch, *args, **kwargs):
            with query_scope():
                return self.original_generate(data_batch, *args, **kwargs)

        def denoise(_model, x, sigma, condition, *args, **kwargs):
            self.cache.denoise_calls += 1
            self.calls["denoise_calls"] += 1
            self.graph_enabled = False
            unchanged = True
            if self.condition_ref is None:
                self.condition_ref = condition
                self.condition_version = self._versions(condition)
            elif not self._same_tree(condition, self.condition_ref, self.condition_version):
                self.calls["condition_changes"] += 1
                unchanged = False
            self.graph_enabled = (unchanged and self.cache.denoise_calls >= 3
                                  and (not self.cache.modules or self.cache.stable()))
            try:
                # Cosmos reads a CUDA condition Tensor in a Python branch before the DiT.
                return self.original_denoise(x, sigma, condition, *args, **kwargs)
            finally:
                self.graph_enabled = False

        def net_forward(*args, **kwargs):
            if not self.graph_enabled:
                return self.original_net_forward(*args, **kwargs)
            key = (self.cache.signature(), _signature(self.torch, (args, kwargs)))
            record = self.records.get(key)
            if record is None:
                try:
                    record = self._capture(key, args, kwargs)
                except Exception as exc:
                    self.capture_errors.append(f"{type(exc).__name__}: {exc}")
                    self.graph_status = "pending_graph_validation"
                    print(f"COSMOS_GRAPH_CAPTURE_ERROR {type(exc).__name__}: {exc}", flush=True)
                    raise RuntimeError("Cosmos CUDA graph capture failed; stopping validation") from exc
            else:
                self.banks.workspace = record["workspace"]
                _copy_tree(self.torch, record["inputs"], (args, kwargs))
                record["graph"].replay()
                self.calls["graph_replays"] += 1
            return _clone_tree(self.torch, record["output"])

        self.model.generate_samples_from_batch = generate
        self.model.denoise = types.MethodType(denoise, self.model)
        self.model.net.forward = net_forward

    @staticmethod
    def _versions(value, _seen=None):
        torch = importlib.import_module("torch")
        if _seen is None:
            _seen = {}
        kind, fields = _tree_kind(torch, value)
        if kind == "tensor":
            version = _tensor_version(value)
            snapshot = value.detach().clone() if version is None else None
            return (value.data_ptr(), version, snapshot)
        if kind == "leaf":
            return ("leaf", type(value), value)
        if kind == "opaque":
            return ("opaque", type(value), id(value))
        identity = id(value)
        if identity in _seen:
            return (_TREE_REF, _seen[identity])
        token = len(_seen)
        _seen[identity] = token
        if kind == "dict":
            return ("dict", type(value), token,
                    tuple((key, _DenoiseGraphs._versions(item, _seen)) for key, item in value.items()))
        if kind == "sequence":
            return ("sequence", type(value), token,
                    tuple(_DenoiseGraphs._versions(item, _seen) for item in value))
        return ("object", type(value), token,
                tuple((key, _DenoiseGraphs._versions(item, _seen)) for key, item in fields.items()))

    @staticmethod
    def _same_tree(value, reference, versions, _seen=None):
        torch = importlib.import_module("torch")
        if _seen is None:
            _seen = ({}, {})
        value_seen, reference_seen = _seen
        kind, fields = _tree_kind(torch, reference)
        value_kind, value_fields = _tree_kind(torch, value)
        if kind == "tensor":
            if not isinstance(value, torch.Tensor) or (value.shape, value.dtype, value.device) != (
                    reference.shape, reference.dtype, reference.device):
                return False
            ptr, version, snapshot = versions
            if snapshot is not None:
                return bool(value.equal(snapshot))
            if value.data_ptr() == ptr:
                return _tensor_version(value) == version
            if _tensor_version(reference) != version:
                return False
            return bool(value.equal(reference))
        if kind == "leaf":
            return (value_kind == "leaf" and type(value) is versions[1]
                    and _stable_values_equal(value, versions[2]))
        if kind == "opaque":
            return (value_kind == "opaque" and type(value) is versions[1]
                    and id(reference) == versions[2] and value is reference)
        if isinstance(versions, tuple) and len(versions) == 2 and versions[0] is _TREE_REF:
            token = versions[1]
            return (value_kind == kind and type(value) is type(reference)
                    and value_seen.get(id(value)) == token
                    and reference_seen.get(id(reference)) == token)
        if value_kind != kind or type(value) is not versions[1]:
            return False
        if not isinstance(versions, tuple) or len(versions) != 4 or versions[0] != kind:
            return False
        token = versions[2]
        value_id, reference_id = id(value), id(reference)
        if (value_id in value_seen or reference_id in reference_seen
                or token in value_seen.values() or token in reference_seen.values()):
            return False
        value_seen[value_id], reference_seen[reference_id] = token, token
        children = versions[3]
        if kind == "dict":
            expected_keys = {key for key, _ in children}
            return (value.keys() == reference.keys() and set(value.keys()) == expected_keys
                    and all(_DenoiseGraphs._same_tree(value[key], reference[key], old, _seen)
                            for key, old in children))
        if kind == "sequence":
            return (type(value) is type(reference) and len(value) == len(reference)
                    and len(value) == len(children)
                    and all(_DenoiseGraphs._same_tree(part, old_value, old_version, _seen)
                            for part, (old_value, old_version) in zip(value, zip(reference, children))))
        expected_keys = tuple(key for key, _ in children)
        return (value_fields.keys() == fields.keys() and set(value_fields) == set(expected_keys)
                and all(_DenoiseGraphs._same_tree(getattr(value, key), getattr(reference, key), old, _seen)
                        for key, old in children))

    def _capture(self, key, args, kwargs):
        torch = self.torch
        workspace = self.banks.activate_graph_workspace(key)
        actual = (args, kwargs)
        static = _clone_tree(torch, actual)
        x = kwargs["x_B_C_T_H_W"]
        side = torch.cuda.Stream(device=x.device)
        side.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(side):
            with self.cache.capture_scope():
                self.original_net_forward(*static[0], **static[1])
        torch.cuda.current_stream(x.device).wait_stream(side)
        torch.cuda.synchronize(x.device)
        graph = torch.cuda.CUDAGraph()
        with self.cache.capture_scope():
            with torch.cuda.graph(graph):
                output = self.original_net_forward(*static[0], **static[1])
        _copy_tree(torch, static, actual)
        graph.replay()
        record = {"graph": graph, "inputs": static, "output": output, "workspace": workspace}
        self.records[key] = record
        self.calls["graph_captures"] += 1
        return record

    def metadata(self):
        return {"status": self.graph_status, "boundary": "Cosmos model.net.forward one DiT; native denoise and sampler eager",
                "branch_graphs": len(self.records), "denoise_calls": self.calls["denoise_calls"],
                "graph_captures": self.calls["graph_captures"], "graph_replays": self.calls["graph_replays"],
                "condition_changes": self.calls["condition_changes"], "capture_errors": list(self.capture_errors),
                "single_request_single_stream": True}

    def restore(self):
        self.model.generate_samples_from_batch = self.original_generate
        self.model.denoise = self.original_denoise
        self.model.net.forward = self.original_net_forward


class Engine:
    def __init__(self, runtime, banks, cache, graphs, originals, controller,
                 original_metadata, manifest, artifact_dir):
        self.runtime, self.banks, self.cache, self.graphs = runtime, banks, cache, graphs
        self.originals, self.controller = originals, controller
        self.original_metadata = original_metadata
        self.manifest, self.artifact_dir = manifest, artifact_dir
        self.parity_receipt = None
        unquantized_state = runtime.model.net.state_dict()
        self.unquantized_dit_tensor_bytes = _tensor_storage_bytes(unquantized_state)
        self.deployment_tensor_bytes = _tensor_storage_bytes(
            (unquantized_state, banks.encodings, banks.transforms, banks.rotation))
        self.original_runtime_hooks = {
            name: (name in runtime.__dict__, getattr(runtime, name, None))
            for name in ("begin_query", "end_query", "invalidate_query_cache")
        }
        self.restored = False
        self._install_query_hooks()

    def _install_query_hooks(self):
        runtime, cache, banks, graphs = self.runtime, self.cache, self.banks, self.graphs
        old_begin = self.original_runtime_hooks["begin_query"][1]
        old_end = self.original_runtime_hooks["end_query"][1]
        old_invalidate = self.original_runtime_hooks["invalidate_query_cache"][1]

        def begin_query():
            if callable(old_begin):
                old_begin()
            if not cache.active:
                cache.begin_query()
                banks.workspace = banks.default_workspace
                graphs.condition_ref = graphs.condition_version = None

        def end_query():
            try:
                if callable(old_end):
                    old_end()
            finally:
                if cache.active:
                    cache.end_query()
                graphs.condition_ref = graphs.condition_version = None
                runtime.install_metadata = self.metadata()

        def invalidate_query_cache():
            if callable(old_invalidate):
                old_invalidate()
            cache.invalidate()

        runtime.begin_query = begin_query
        runtime.end_query = end_query
        runtime.invalidate_query_cache = invalidate_query_cache

    def metadata(self):
        tuning = self.banks.decoder_receipt
        graph_metadata = self.graphs.metadata()
        parity_validated = (self.parity_receipt is not None
                            and graph_metadata.get("status") == "validated_full80"
                            and graph_metadata.get("capture_errors") == []
                            and type(graph_metadata.get("graph_replays")) is int
                            and graph_metadata["graph_replays"] > 0)
        return {"arm": ARM, "artifact_format": FORMAT, "target_module_count": len(self.banks.modules),
                "source_job": self.manifest["source_job"],
                "completed_updates": self.manifest["completed_updates"],
                "protocol": self.manifest["protocol"], "artifact_dir": str(self.artifact_dir),
                "target_weight_numel": sum(int(module.out_features) * int(module.in_features)
                                            for module in self.banks.modules.values()),
                "packed_storage_bytes": self.deployment_tensor_bytes,
                "packed_storage_unit": "bytes",
                "packed_storage_scope": "whole DiT tensor payload including packed targets, Smooth/H128 transforms, and shared U; excludes scratch and file-container overhead",
                "unquantized_dit_tensor_bytes": self.unquantized_dit_tensor_bytes,
                "packed_gpu_resident": True, "dense_target_weights": False,
                "native_vq_kernel": False, "native_integer_gemm": False,
                "native_int4_kernel_used": False,
                "layer_decode": True,
                "weight_decode": ("shape-tuned Triton VQ decoder into shared BF16 scratch"
                                  if tuning and tuning.get("status") == "passed"
                                  else "reference Triton VQ decoder into shared BF16 scratch"),
                "decoder_tuning_status": (tuning["status"] if tuning else "pending_Cosmos_shape_GPU_preflight"),
                "decoder_tuning_receipt": (tuning.get("receipt_path") if tuning else None),
                "decoder_unique_shapes": (len(tuning.get("shapes", [])) if tuning else None),
                "linear": "BF16 activation by decoded BF16 weight through PyTorch cuBLAS Linear",
                "arithmetic": "original FP32 Smooth/H128/U/A4, output quantized to BF16",
                "execution_scope": ("native Cosmos sampler/denoise with single-request/single-stream DiT graph; per-Linear decoded-BF16 "
                                    "weights and cuBLAS Linear; CUDA graph validated on the full 80-query "
                                    "action/future parity panel" if parity_validated else
                                    "native Cosmos sampler/denoise with single-request/single-stream DiT graph; per-Linear decoded-BF16 "
                                    "weights and cuBLAS Linear; CUDA graph capture pending GPU validation"),
                "activation_workspace": self._workspace_metadata(),
                "condition_cache": self.cache.metadata(), "cuda_graph": graph_metadata,
                "parity_status": ("validated_full80" if parity_validated
                                  else "pending_real_GPU_reference_comparison")}

    def mark_validated(self, receipt):
        """Accept full-panel parity only after this process has replayed a clean CUDA graph."""
        receipt = validate_full80_parity_receipt(receipt, self.manifest)
        graph_metadata = self.graphs.metadata()
        if (graph_metadata.get("capture_errors") != []
                or type(graph_metadata.get("graph_replays")) is not int
                or graph_metadata["graph_replays"] <= 0):
            raise ValueError("Full-panel receipt requires a clean CUDA graph replay in this process")
        self.graphs.graph_status = "validated_full80"
        self.parity_receipt = receipt
        self.runtime.install_metadata = self.metadata()
        return self.runtime.install_metadata

    def _workspace_metadata(self):
        ws = self.banks.workspace
        return {"scratch_bytes": int(ws.scratch.numel() * ws.scratch.element_size()),
                "activation_a4_bytes": int(ws.a4.numel() * ws.a4.element_size()),
                "h_u_alias_weight_scratch": True,
                "a4_independent_buffer": ws.a4.untyped_storage().data_ptr() != ws.scratch.untyped_storage().data_ptr(),
                "graph_workspace_count": len(self.banks.graph_workspaces)}

    def restore(self):
        if self.restored:
            return
        import torch

        self.graphs.restore()
        for name, (was_instance_value, original) in self.original_runtime_hooks.items():
            if was_instance_value:
                setattr(self.runtime, name, original)
            else:
                self.runtime.__dict__.pop(name, None)
        for name, module in self.banks.modules.items():
            weight = self.controller._base_weights_cpu[name].to(device=self.banks.device,
                                                               dtype=torch.bfloat16)
            module.register_parameter("weight", torch.nn.Parameter(
                weight, requires_grad=self.originals[name]["requires_grad"]))
            module.forward = self.originals[name]["forward"]
        self.runtime.install_metadata = self.original_metadata
        self.restored = True
        self.runtime.ours_vqwam_engine = None
        self.runtime.restore_ours_vqwam = None

    def preflight(self, p, out):
        return _preflight(self, p, out)


DECODER_RECEIPT_FORMAT = "cosmos_vqwam_decoder_preflight_v1"


def validate_decoder_receipt(receipt, manifest, modules, expected_device=None):
    """Validate a saved shape-tuning receipt against the current packed target manifest."""
    if receipt.get("format") != DECODER_RECEIPT_FORMAT or receipt.get("status") != "passed":
        raise ValueError("Cosmos VQ decoder preflight receipt did not pass")
    if receipt.get("arm") != ARM or receipt.get("protocol") != manifest.get("protocol"):
        raise ValueError("Decoder receipt arm/protocol differs from the packed artifact")
    if receipt.get("source_job") != manifest.get("source_job"):
        raise ValueError("Decoder receipt belongs to a different training artifact")
    if expected_device is not None and receipt.get("device") != expected_device:
        raise ValueError("Decoder receipt was tuned for a different GPU")
    rows = _target_rows(modules)
    module_rows = receipt.get("modules")
    if not isinstance(module_rows, list) or len(module_rows) != len(rows):
        raise ValueError("Decoder receipt does not cover every target Linear")
    by_shape = {tuple(row.get("shape", ())): row for row in receipt.get("shapes", [])}
    shape_counts = Counter((row["shape"][0], row["shape"][1]) for row in rows)
    if set(by_shape) != set(shape_counts):
        raise ValueError("Decoder receipt shape inventory differs from target modules")
    if len(by_shape) != len(receipt.get("shapes", [])):
        raise ValueError("Decoder receipt contains duplicate shape decisions")
    for shape, count in shape_counts.items():
        row = by_shape[shape]
        if (row.get("module_count") != count or row.get("selected") not in
                {"reference", "block128_w4", "block256_w4", "block512_w4", "block512_w8", "block1024_w8"}):
            raise ValueError(f"Invalid decoder decision for shape {shape}")
    for expected, observed in zip(rows, module_rows):
        if (observed.get("name") != expected["name"] or observed.get("shape") != expected["shape"]
                or observed.get("reference_bitwise") is not True
                or observed.get("selected_bitwise") is not True
                or observed.get("selected") != by_shape[tuple(expected["shape"])]["selected"]):
            raise ValueError(f"Decoder receipt module parity mismatch at {expected['name']}")
    parity = receipt.get("parity", {})
    if (parity.get("reference_checked_modules") != len(rows)
            or parity.get("reference_bitwise_passed") != len(rows)
            or parity.get("selected_checked_modules") != len(rows)
            or parity.get("selected_bitwise_passed") != len(rows)
            or parity.get("failed_modules") != []):
        raise ValueError("Decoder receipt lacks complete 280-module reference and selected-kernel parity")
    return {tuple(shape): row["selected"] for shape, row in by_shape.items()}


def validate_full80_parity_receipt(receipt, manifest):
    """Validate the frozen 40-case by 2-seed endpoint parity receipt."""
    if not isinstance(receipt, dict) or receipt.get("status") != "passed":
        raise ValueError("Full-panel endpoint parity receipt did not pass")
    if (receipt.get("protocol") != manifest.get("protocol")
            or receipt.get("source_job") != manifest.get("source_job")):
        raise ValueError("Full-panel parity receipt belongs to a different artifact")
    if (type(receipt.get("checked_queries")) is not int
            or receipt["checked_queries"] != EXPECTED_PARITY_QUERIES
            or type(receipt.get("expected_queries")) is not int
            or receipt["expected_queries"] != EXPECTED_PARITY_QUERIES):
        raise ValueError("Full-panel parity receipt must cover exactly 80 queries")
    if receipt.get("failed_queries") != []:
        raise ValueError("Full-panel parity receipt contains failed queries")
    return receipt


def _decoder_receipt_path(out):
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    return output / "decoder_tuning.json"


def _write_decoder_receipt(path, receipt):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def _preflight(engine, p, out):
    """Check all 280 decodes, tune a bounded candidate set per actual shape, and install decisions."""
    _compute_guard(engine.runtime.task_root)
    import torch

    runtime, banks = engine.runtime, engine.banks
    if runtime.ours_vqwam_engine is not engine:
        raise RuntimeError("Decoder preflight requires this installed ours_vqwam engine")
    if engine.cache.active:
        raise RuntimeError("Decoder preflight must run outside a query")
    protocol = _protocol_value(p)
    if protocol != engine.manifest.get("protocol"):
        raise ValueError("Preflight protocol differs from the packed artifact")
    if banks.decoder_receipt and banks.decoder_receipt.get("status") == "passed":
        validate_decoder_receipt(banks.decoder_receipt, engine.manifest, banks.modules)
        return banks.decoder_receipt

    device = banks.device
    props = torch.cuda.get_device_properties(device)
    receipt_path = _decoder_receipt_path(out)
    receipt = {
        "format": DECODER_RECEIPT_FORMAT,
        "status": "running",
        "arm": ARM,
        "protocol": protocol,
        "source_job": engine.manifest["source_job"],
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "device": {"name": props.name, "capability": [int(props.major), int(props.minor)]},
        "module_count": len(banks.modules),
        "decode_scope": "VQ decode only; no Linear or Cosmos denoise runs",
        "reference": "decoder_runtime._launch_reference:block256_w4",
        "candidate_configs": [dict(row) for row in banks.decoder_runtime._CANDIDATES],
        "parity": {"reference_checked_modules": 0, "reference_bitwise_passed": 0,
                   "selected_checked_modules": 0, "selected_bitwise_passed": 0,
                   "failed_modules": []},
        "shapes": [],
        "modules": [],
        "receipt_path": str(receipt_path),
    }
    _write_decoder_receipt(receipt_path, receipt)
    micro = _micro_codebook(runtime)
    scratch = banks.default_workspace.scratch
    shape_names = {}
    for name, prepared in banks.prepared.items():
        shape_names.setdefault((int(prepared.rows), int(prepared.width)), []).append(name)

    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            torch.cuda.synchronize(device)
            for name in sorted(banks.prepared):
                prepared = banks.prepared[name]
                banks.decoder_runtime._launch_reference(prepared, scratch, banks.fused_vq._decode_kernel)
                expected = micro.decode_vq(banks.encodings[name])
                if expected.device != device:
                    expected = expected.to(device=device)
                expected = expected.detach().contiguous()
                actual = scratch[:prepared.rows * prepared.width].view(prepared.rows, prepared.width)
                passed = bool(torch.equal(actual.contiguous().view(torch.int16),
                                          expected.view(torch.int16)))
                receipt["modules"].append({"name": name,
                    "shape": [int(prepared.rows), int(prepared.width)],
                    "reference_bitwise": passed, "selected": None, "selected_bitwise": None})
                receipt["parity"]["reference_checked_modules"] += 1
                if passed:
                    receipt["parity"]["reference_bitwise_passed"] += 1
                else:
                    receipt["parity"]["failed_modules"].append(name + ":reference_vs_micro_decode")
                del expected
            if receipt["parity"]["failed_modules"]:
                raise RuntimeError("Reference Triton VQ decoder differs from micro.decode_vq")

            decisions = {}
            for shape, names in sorted(shape_names.items()):
                representative = names[0]
                prepared = banks.prepared[representative]
                expected = micro.decode_vq(banks.encodings[representative]).detach().contiguous()
                if expected.device != device:
                    expected = expected.to(device=device)
                reference_call = lambda: banks.decoder_runtime._launch_reference(
                    prepared, scratch, banks.fused_vq._decode_kernel)
                reference_timing = banks.decoder_runtime._measure_graph(reference_call, device)
                if (not math.isfinite(reference_timing["median_ms"]) or reference_timing["median_ms"] <= 0
                        or not all(math.isfinite(value) and value > 0
                                   for value in reference_timing["samples_ms"])):
                    raise RuntimeError(f"Invalid reference decoder timing for shape {shape}")
                candidates = []
                errors = []
                for config in banks.decoder_runtime._CANDIDATES:
                    name = config["name"]
                    call = lambda config=config: banks.decoder_runtime._launch_candidate(
                        prepared, scratch, config)
                    row = None
                    try:
                        call()
                        torch.cuda.synchronize(device)
                        actual = scratch[:prepared.rows * prepared.width].view(
                            prepared.rows, prepared.width)
                        passed = bool(torch.equal(actual.contiguous().view(torch.int16),
                                                  expected.view(torch.int16)))
                        row = {"name": name, "bitwise": passed, "median_ms": None,
                               "samples_ms": None}
                        if passed:
                            timing = banks.decoder_runtime._measure_graph(call, device)
                            if (not math.isfinite(timing["median_ms"]) or timing["median_ms"] <= 0
                                    or not all(math.isfinite(value) and value > 0
                                               for value in timing["samples_ms"])):
                                raise RuntimeError("decoder candidate timing was non-finite or non-positive")
                            row.update(median_ms=timing["median_ms"], samples_ms=timing["samples_ms"])
                        else:
                            errors.append({"name": name, "error": "raw BF16 output mismatch"})
                        candidates.append(row)
                    except Exception as exc:
                        errors.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
                        candidates.append(row if row is not None else {
                            "name": name, "bitwise": False, "median_ms": None, "samples_ms": None})
                timed_candidates = [row for row in candidates if row["median_ms"] is not None]
                selected = banks.decoder_runtime._choose_config(reference_timing["median_ms"],
                                                                timed_candidates)
                decisions[shape] = selected
                receipt["shapes"].append({"shape": list(shape), "module_count": len(names),
                    "representative": representative,
                    "reference": {"median_ms": reference_timing["median_ms"],
                                  "samples_ms": reference_timing["samples_ms"]},
                    "candidates": candidates, "candidate_failures": errors, "selected": selected})
                del expected

            selected_ok = {}
            for name in sorted(banks.prepared):
                prepared = banks.prepared[name]
                shape = (int(prepared.rows), int(prepared.width))
                selected = decisions[shape]
                if selected == "reference":
                    banks.decoder_runtime._launch_reference(prepared, scratch, banks.fused_vq._decode_kernel)
                else:
                    banks.decoder_runtime._launch_candidate(
                        prepared, scratch, banks.decoder_runtime._CONFIGS[selected])
                expected = micro.decode_vq(banks.encodings[name])
                if expected.device != device:
                    expected = expected.to(device=device)
                expected = expected.detach().contiguous()
                actual = scratch[:prepared.rows * prepared.width].view(prepared.rows, prepared.width)
                passed = bool(torch.equal(actual.contiguous().view(torch.int16),
                                          expected.view(torch.int16)))
                selected_ok[name] = passed
                row = next(item for item in receipt["modules"] if item["name"] == name)
                row.update(selected=selected, selected_bitwise=passed)
                receipt["parity"]["selected_checked_modules"] += 1
                if passed:
                    receipt["parity"]["selected_bitwise_passed"] += 1
                else:
                    receipt["parity"]["failed_modules"].append(name + ":selected_decoder")
                del expected
            if receipt["parity"]["failed_modules"]:
                raise RuntimeError("Selected decoder failed per-Linear BF16 parity")
            receipt["status"] = "passed"
            receipt["selected_configs"] = [
                {"shape": row["shape"], "selected": row["selected"]} for row in receipt["shapes"]]
            receipt["selected_decoder_bitwise_passed"] = all(selected_ok.values())
            validate_decoder_receipt(receipt, engine.manifest, banks.modules)
            banks.decoder_decisions = decisions
            banks.decoder_receipt = receipt
            torch.cuda.synchronize(device)
            _write_decoder_receipt(receipt_path, receipt)
            runtime.install_metadata = engine.metadata()
            return receipt
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        _write_decoder_receipt(receipt_path, receipt)
        banks.decoder_receipt = receipt
        runtime.install_metadata = engine.metadata()
        raise


def install(runtime, p, artifact_dir, root, out):
    """Install our packed Cosmos VQ arm on a loaded legacy ``CosmosRuntime``."""
    _compute_guard(root)
    import torch

    if not isinstance(runtime.target_linears, dict) or len(runtime.target_linears) != EXPECTED_TARGETS:
        raise ValueError(f"runtime.target_linears must contain the frozen {EXPECTED_TARGETS} modules")
    if runtime.controller is None or getattr(runtime.controller, "arm", None) != "bf16":
        raise RuntimeError("Install the ours arm from a restored BF16 CosmosRuntime controller state")
    modules = dict(sorted(runtime.target_linears.items()))
    if list(runtime.target_linears) != list(modules):
        raise ValueError("runtime.target_linears must be sorted by exact module name")
    device = _module_device(modules, torch)
    artifact_dir = Path(artifact_dir).resolve()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = validate_manifest(manifest, modules, expected_protocol=_protocol_value(p))
    tuning_receipt = None
    tuning_decisions = {}
    tuning_path = Path(out) / "decoder_tuning.json"
    if tuning_path.is_file():
        candidate = json.loads(tuning_path.read_text(encoding="utf-8"))
        if candidate.get("status") == "passed":
            props = torch.cuda.get_device_properties(device)
            expected_device = {"name": props.name,
                               "capability": [int(props.major), int(props.minor)]}
            tuning_decisions = validate_decoder_receipt(
                candidate, manifest, modules, expected_device=expected_device)
            candidate["receipt_path"] = str(tuning_path.resolve())
            tuning_receipt = candidate

    controller = runtime.controller
    controller._backup_weights()
    original_metadata = getattr(runtime, "install_metadata", None)
    originals = {name: {"forward": module.forward,
                        "requires_grad": bool(controller._weight_requires_grad[name])}
                 for name, module in modules.items()}

    micro = _micro_codebook(runtime)
    encodings, transforms = {}, {}
    for index, row in enumerate(rows):
        payload = torch.load(artifact_dir / _safe_bank_file(manifest["modules"][index]["file"]),
                             map_location="cpu", weights_only=True)
        if payload.get("name", row["name"]) != row["name"] or payload.get("shape", row["shape"]) != row["shape"]:
            raise ValueError(f"Bank identity differs at {row['name']}")
        encoding, transform = payload["encoding"], payload["transform"]
        micro.check_encoding(encoding)
        if list(encoding.get("shape", ())) != row["shape"]:
            raise ValueError(f"Packed encoding shape differs at {row['name']}")
        width = row["shape"][1]
        transform = dict(transform)
        transform["scale"] = transform["scale"].to(device=device, dtype=torch.float32).reshape(-1).contiguous()
        transform["signs"] = transform["signs"].to(device=device).reshape(-1).contiguous()
        transform.update(input_dim=int(transform["input_dim"]), hadamard=bool(transform["hadamard"]),
                         block=int(transform["block"]))
        if transform["input_dim"] != width or transform["block"] != 128 or not transform["hadamard"]:
            raise ValueError(f"Smooth/H128 metadata differs at {row['name']}")
        encoding = {key: value.to(device=device) if torch.is_tensor(value) else value
                    for key, value in encoding.items()}
        encodings[row["name"]], transforms[row["name"]] = encoding, transform

    rotation_payload = torch.load(artifact_dir / "rotation.pt", map_location="cpu", weights_only=True)
    matrix = rotation_payload.get("matrix")
    if (rotation_payload.get("stream") != "cosmos"
            or rotation_payload.get("parameterization") != "H128_then_U"
            or not torch.is_tensor(matrix) or tuple(matrix.shape) != (128, 128)
            or matrix.dtype != torch.float32):
        raise ValueError("rotation.pt must contain the shared FP32 Cosmos U[128,128]")
    rotation = matrix.to(device=device, dtype=torch.float32).contiguous()
    kernels = _fastwam_helpers(p)

    # Build every wrapper before removing CUDA dense weights, so an install-time
    # validation error leaves the legacy model usable.
    for name, module in modules.items():
        if module.weight.dtype != torch.bfloat16:
            raise ValueError(f"Expected restored BF16 target weight at {name}")
    banks = _Banks(runtime, modules, encodings, transforms, rotation, kernels)
    banks.decoder_decisions = tuning_decisions
    banks.decoder_receipt = tuning_receipt
    try:
        cache = _CrossKVCache(modules)
        graphs = _DenoiseGraphs(runtime, banks, cache)
    except Exception:
        for name, module in modules.items():
            module.forward = banks.original_forwards[name]
        raise
    for module in modules.values():
        module.register_parameter("weight", None)
    engine = Engine(runtime, banks, cache, graphs, originals, controller,
                    original_metadata, manifest, artifact_dir)
    runtime.ours_vqwam_engine = engine
    runtime.restore_ours_vqwam = engine.restore
    runtime.install_metadata = engine.metadata()
    return runtime.install_metadata


def preflight(runtime, p, artifact_dir, root, out):
    """Prove all packed decodes and tune only shape-valid configs before perf runs."""
    _compute_guard(root)
    engine = getattr(runtime, "ours_vqwam_engine", None)
    if engine is None or Path(artifact_dir).resolve() != engine.artifact_dir:
        raise RuntimeError("Install this artifact's ours_vqwam arm before decoder preflight")
    return engine.preflight(p, out)

