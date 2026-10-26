"""Frozen six-arm native Cosmos performance, DiT replay, and fidelity worker.

Run only inside an approved PBS GPU allocation. ``capture`` writes one immutable
BF16 net-forward capture per frozen query; ``benchmark`` records either the full
sampler wall interval or a five-forward generic CUDA-graph replay; ``gate`` checks
native outputs against the same method's decoded reference before performance.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
import gc
from importlib import import_module
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any


FORMAT = "cosmos_native_performance_v1"
PERFORMANCE_BLOCKS = 1
PERFORMANCE_REPETITIONS = 1
PERFORMANCE_SAMPLES = 40
PERFORMANCE_DESIGN = "40ctx_once_v3"
CAPTURE_FORMAT = "cosmos_bf16_net_replay_inputs_v1"
TORCH_DTYPE_MARKER = "__native_perf_torch_dtype__"
TORCH_DEVICE_MARKER = "__native_perf_torch_device__"
COSMOS_ENUM_MARKER = "__native_perf_cosmos_enum__"
COSMOS_ENUM_ALLOWLIST = frozenset({
    ("cosmos_policy._src.predict2.conditioner", "DataType"),
})
MAX_SUMMARY_BYTES = 256 * 1024
ARMS = ("bf16", "spinquant_w4a4", "quarot_w4a4", "svdquant_w4a4",
        "viditq_w4a4", "ours_vqwam")
DIAGNOSTIC_QUERIES = ((3000, 0), (3009, 1), (3020, 0), (3039, 1))
GATE_THRESHOLDS = {"dit_forward_nrmse": 0.01,
                   "normalized_motor_rmse": 0.01,
                   "future_nrmse": 0.01}
OURS_OPTIMIZED_ENDPOINT_THRESHOLD = 1e-6
DIT_GRAPH_POOL_CONTRACT = (
    "five CUDA graphs share one graph_pool_handle; replay is strictly serial in NFE 0..4 order; "
    "each NFE has independent static inputs and an external output buffer")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _bounded_summary(path: Path, value: dict) -> int:
    payload = (json.dumps(value, ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    if len(payload) > MAX_SUMMARY_BYTES:
        raise ValueError(f"summary exceeds {MAX_SUMMARY_BYTES} bytes: {len(payload)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(payload)


def validate_protocol(protocol: dict, arm: str) -> dict:
    if arm not in ARMS or protocol.get("arms") != list(ARMS):
        raise ValueError("Protocol must retain the frozen BF16/Spin/Qua/SVD/ViDiT-Q/Ours six arms")
    performance = protocol.get("performance", {})
    cases, seeds = performance.get("case_ids"), performance.get("seed_indices")
    if (protocol.get("mode") != "libero_direct_policy"
            or protocol.get("inference", {}).get("denoising_steps") != 5
            or cases != list(range(3000, 3040)) or seeds != [0, 1]
            or performance.get("queries") != 80 or performance.get("warmup_queries") != 3):
        raise ValueError("Protocol differs from the frozen 80-query, five-NFE input cohort")
    return {"case_ids": cases, "seed_indices": seeds, "queries": 80,
            "blocks": PERFORMANCE_BLOCKS, "repetitions": PERFORMANCE_REPETITIONS,
            "samples_per_arm": PERFORMANCE_SAMPLES,
            "denoising_steps": 5}


def validate_variant(arm, variant):
    if (variant not in ('current', 'optimized', 'native_fused', 'native_grouped')
            or variant == 'optimized' and arm != 'ours_vqwam'
            or variant == 'native_fused' and arm != 'svdquant_w4a4'
            or variant == 'native_grouped' and arm not in ('spinquant_w4a4', 'quarot_w4a4')):
        raise ValueError('Variant does not belong to the requested method')


def validate_paths(old_root: Path, root: Path, out: Path) -> tuple[Path, Path, Path]:
    old_root, root, out = (p.expanduser().resolve() for p in (old_root, root, out))
    if old_root == root or old_root in root.parents or root in old_root.parents:
        raise ValueError("--root must be isolated from read-only --old-root")
    if out != root and root not in out.parents:
        raise ValueError("--out must be inside --root")
    if old_root == out or old_root in out.parents:
        raise ValueError("--out cannot write under the read-only --old-root")
    return old_root, root, out


def query_keys(protocol: dict) -> list[tuple[int, int]]:
    panel = protocol["performance"]
    return [(int(case), int(seed)) for case in panel["case_ids"]
            for seed in panel["seed_indices"]]


def performance_query_keys(protocol: dict) -> list[tuple[int, int]]:
    return [(int(case), 0) for case in protocol["performance"]["case_ids"]]


def capture_path(root: Path, case_id: int, seed_index: int) -> Path:
    return root / "results" / "replay_inputs" / f"{case_id}_{seed_index}.pt"


def performance_root(root: Path) -> Path:
    return root / "results" / "native_performance_n40"


def sample_path(root: Path, scope: str, variant: str, arm: str, case_id: int,
                seed_index: int, block: int, repetition: int) -> Path:
    return (performance_root(root) / scope / variant / arm / "samples"
            / f"{case_id}_{seed_index}_b{block}_r{repetition}.json")


def sample_identity(*, protocol: str, hardware: dict, binding: dict, scope: str,
                    variant: str, arm: str, case_id: int, seed_index: int,
                    block: int, repetition: int, warmup_count: int) -> dict:
    return {"protocol": protocol, "performance_design": PERFORMANCE_DESIGN,
            "hardware": hardware_key(hardware), "backend_binding": binding,
            "scope": scope, "variant": variant, "arm": arm,
            "case_id": int(case_id), "seed_index": int(seed_index),
            "block": int(block), "repetition": int(repetition),
            "warmup_query_count": int(warmup_count)}


def hardware_key(hardware: dict) -> dict:
    """Persist the actual measured device identity alongside the software SKU."""
    keys = ("gpu_uuid", "gpu_name", "total_memory_bytes", "capability",
            "torch_version", "cuda_version")
    return {key: hardware.get(key) for key in keys}


def compatible_gpu_key(hardware: dict) -> dict:
    """Formal hardware contract: same GPU SKU and software; physical UUID may differ."""
    keys = ("gpu_name", "total_memory_bytes", "capability", "torch_version", "cuda_version")
    return {key: hardware.get(key) for key in keys}


def validate_hardware_compatibility(protocol: dict, hardware: dict) -> None:
    spec = protocol.get("hardware", {})
    expected = {"gpu_name": spec.get("gpu"), "capability": spec.get("capability"),
                "torch_version": spec.get("torch"), "cuda_version": spec.get("cuda_runtime")}
    actual = compatible_gpu_key(hardware)
    missing = [key for key, value in expected.items() if value is None]
    if missing:
        raise ValueError(f"Hardware compatibility specification is incomplete: {missing}")
    differences = {key: {"expected": value, "actual": actual.get(key)}
                   for key, value in expected.items() if actual.get(key) != value}
    if differences:
        raise RuntimeError(f"Active GPU/software differs from the frozen compatibility contract: {differences}")


def reusable_sample(path: Path, expected: dict) -> dict | None:
    if not path.is_file():
        return None
    saved = read_json(path)
    for key, value in expected.items():
        if key == "hardware":
            saved_hardware = saved.get(key, {})
            provenance = saved.get("hardware_receipt") or saved_hardware
            matches = (bool(provenance.get("gpu_uuid"))
                       and bool(saved.get("hostname") or provenance.get("hostname"))
                       and bool(saved.get("pbs_job_id") or provenance.get("pbs_job_id"))
                       and compatible_gpu_key(provenance) == compatible_gpu_key(saved_hardware)
                       and compatible_gpu_key(saved_hardware) == compatible_gpu_key(value))
        else:
            matches = saved.get(key) == value
        if not matches:
            raise RuntimeError(f"Saved native sample binding differs at {key}: {path}")
    return saved


def reusable_performance_sample(path: Path, expected: dict) -> dict | None:
    saved = reusable_sample(path, expected)
    if saved is not None:
        return saved
    # Preserve already timed first-pass contexts from interrupted legacy jobs.
    parts = path.parts
    index = parts.index("native_performance_n40")
    legacy = Path(*parts[:index], "native_performance", *parts[index + 1:])
    legacy_expected = {key: value for key, value in expected.items() if key != "performance_design"}
    saved = reusable_sample(legacy, legacy_expected)
    if saved is None:
        return None
    if saved.get("performance_design") is not None:
        raise RuntimeError(f"Unexpected legacy sample design: {legacy}")
    copied = {**saved, "performance_design": PERFORMANCE_DESIGN,
              "timed_sample_reused_from": str(legacy)}
    _atomic_json(path, copied)
    return copied


def sample_hardware_provenance(records: list[dict], fallback_hardware: dict) -> dict:
    """Summarize actual UUID/host/job receipts without making them resume keys."""
    devices = {}
    for row in records:
        hardware = row.get("hardware", {})
        uuid = hardware.get("gpu_uuid")
        if not uuid:
            continue
        device = devices.setdefault(str(uuid), {"gpu_uuid": str(uuid),
            "hostnames": set(), "pbs_job_ids": set()})
        hostname = row.get("hostname") or hardware.get("hostname")
        pbs_job_id = row.get("pbs_job_id") or hardware.get("pbs_job_id")
        if hostname:
            device["hostnames"].add(str(hostname))
        if pbs_job_id:
            device["pbs_job_ids"].add(str(pbs_job_id))
    if not devices and fallback_hardware.get("gpu_uuid"):
        uuid = str(fallback_hardware["gpu_uuid"])
        devices[uuid] = {"gpu_uuid": uuid,
            "hostnames": {str(fallback_hardware["hostname"])} if fallback_hardware.get("hostname") else set(),
            "pbs_job_ids": {str(fallback_hardware["pbs_job_id"])} if fallback_hardware.get("pbs_job_id") else set()}
    return {"sample_count": len(records), "devices": [
        {"gpu_uuid": uuid, "hostnames": sorted(row["hostnames"]),
         "pbs_job_ids": sorted(row["pbs_job_ids"])}
        for uuid, row in sorted(devices.items())]}


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(float(value) for value in values)
    return {"count": len(ordered), "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered),
            "p95": ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)],
            "max": ordered[-1]}


def _gpu_receipt(torch, device, required_uuid: str | None = None) -> dict:
    props = torch.cuda.get_device_properties(device)
    uuid = getattr(props, "uuid", None)
    if uuid is not None:
        uuid = str(uuid)
    if not uuid:
        result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name",
                                 "--format=csv,noheader"], check=True, capture_output=True,
                                text=True, timeout=15)
        rows = [line.strip().split(",", 1) for line in result.stdout.splitlines() if line.strip()]
        visible = int(torch.cuda.device_count())
        if visible != 1 or len(rows) != 1:
            raise RuntimeError("Cannot bind the active CUDA device to exactly one visible GPU UUID")
        uuid = rows[0][0].strip()
    # Torch returns the bare UUID; nvidia-smi prefixes the same device with GPU-.
    if required_uuid and uuid.removeprefix("GPU-").lower() != required_uuid.removeprefix("GPU-").lower():
        raise RuntimeError(f"GPU UUID differs from --require-gpu-uuid: expected {required_uuid}, got {uuid}")
    if device.type != "cuda":
        raise RuntimeError(f"Expected a CUDA device, found {device}")
    return {"gpu_uuid": uuid, "gpu_name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "capability": [int(props.major), int(props.minor)],
            "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
            "hostname": socket.gethostname(), "pbs_job_id": os.environ.get("PBS_JOBID")}


def _allocation_guard(common) -> None:
    common.guard()
    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("A real PBS compute-node allocation is required")


def _load_context(common, old_root: Path, case_id: int, seed_index: int) -> dict:
    path = old_root / "results" / "teacher" / "contexts" / f"{case_id}_{seed_index}.pt"
    fallback = path.with_name(f"{case_id}.pt")
    if not path.is_file() and not (seed_index == 0 and fallback.is_file()):
        raise FileNotFoundError(f"Frozen old-root context is missing: {path}")
    context = common.load_tensors(path if path.is_file() else fallback)
    if (int(context.get("case_id", -1)), int(context.get("seed_index", -1))) != (case_id, seed_index):
        raise RuntimeError(f"Frozen context identity differs: {path}")
    return context


def _cosmos_enum_type(module: str, qualname: str, path: str) -> type[Enum]:
    identity = (module, qualname)
    if identity not in COSMOS_ENUM_ALLOWLIST:
        raise RuntimeError(f"Unsupported Enum in captured model.net input at {path}: {module}.{qualname}")
    try:
        enum_type = getattr(import_module(module), qualname)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"Allowlisted Cosmos Enum is unavailable at {path}: {module}.{qualname}") from exc
    if (not isinstance(enum_type, type) or not issubclass(enum_type, Enum)
            or enum_type.__module__ != module or enum_type.__qualname__ != qualname):
        raise RuntimeError(f"Allowlisted Cosmos Enum identity changed at {path}: {module}.{qualname}")
    return enum_type


def _cosmos_enum_details(value: Any, path: str) -> dict | None:
    if isinstance(value, Enum):
        enum_type = type(value)
        module, qualname = enum_type.__module__, enum_type.__qualname__
        if _cosmos_enum_type(module, qualname, path) is not enum_type:
            raise RuntimeError(f"Cosmos Enum class identity differs at {path}")
        member = value.name
    elif (isinstance(value, dict) and len(value) == 1
          and COSMOS_ENUM_MARKER in value):
        details = value[COSMOS_ENUM_MARKER]
        if (not isinstance(details, dict) or details.keys() != {"module", "qualname", "member"}
                or any(type(details[key]) is not str for key in details)):
            raise RuntimeError(f"Malformed Cosmos Enum marker at {path}")
        module, qualname, member = details["module"], details["qualname"], details["member"]
        enum_type = _cosmos_enum_type(module, qualname, path)
        if member not in enum_type.__members__ or enum_type.__members__[member].name != member:
            raise RuntimeError(f"Unknown Cosmos Enum member at {path}: {module}.{qualname}.{member}")
    else:
        return None
    return {"module": module, "qualname": qualname, "member": member}


def _cpu_tree(value: Any, torch, path: str = "root") -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True,
                                 memory_format=torch.preserve_format)
    dtype_type = getattr(torch, "dtype", None)
    if dtype_type is not None and isinstance(value, dtype_type):
        return {TORCH_DTYPE_MARKER: str(value).removeprefix("torch.")}
    device_type = getattr(torch, "device", None)
    if device_type is not None and isinstance(value, device_type):
        return {TORCH_DEVICE_MARKER: str(value)}
    enum_details = _cosmos_enum_details(value, path)
    if enum_details is not None:
        return {COSMOS_ENUM_MARKER: enum_details}
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite scalar in captured model.net input at {path}")
        return str(value) if isinstance(value, str) else value
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(_cpu_tree(item, torch, f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, list):
        return [_cpu_tree(item, torch, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, dict):
        if any(isinstance(key, Enum) or not isinstance(key, (str, int, float, bool))
               for key in value):
            raise RuntimeError(f"Opaque mapping key in captured model.net input at {path}")
        return {key: _cpu_tree(item, torch, f"{path}.{key}") for key, item in value.items()}
    raise RuntimeError(f"Opaque/non-tensor model.net value at {path}: {type(value).__name__}")


def _device_tree(value: Any, device, torch, path: str = "root") -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=False,
                        memory_format=torch.preserve_format)
    enum_details = _cosmos_enum_details(value, path)
    if enum_details is not None:
        enum_type = _cosmos_enum_type(enum_details["module"], enum_details["qualname"], path)
        return enum_type.__members__[enum_details["member"]]
    if isinstance(value, dict) and len(value) == 1 and TORCH_DTYPE_MARKER in value:
        dtype = getattr(torch, value[TORCH_DTYPE_MARKER], None)
        dtype_type = getattr(torch, "dtype", None)
        if dtype is None or dtype_type is None or not isinstance(dtype, dtype_type):
            raise RuntimeError(f"Unknown captured torch dtype at {path}")
        return dtype
    if isinstance(value, dict) and len(value) == 1 and TORCH_DEVICE_MARKER in value:
        return torch.device(value[TORCH_DEVICE_MARKER])
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(_device_tree(item, device, torch, f"{path}[{index}]")
                      for index, item in enumerate(value))
    if isinstance(value, list):
        return [_device_tree(item, device, torch, f"{path}[{index}]")
                for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: _device_tree(item, device, torch, f"{path}.{key}")
                for key, item in value.items()}
    raise RuntimeError(f"Opaque/non-tensor replay input at {path}: {type(value).__name__}")


def _tree_signature(value: Any, torch, path: str = "root", *, include_scalar_values=True) -> list[dict]:
    if torch.is_tensor(value):
        return [{"path": path, "kind": "tensor", "shape": list(value.shape),
                 "stride": list(value.stride()), "dtype": str(value.dtype)}]
    enum_details = _cosmos_enum_details(value, path)
    if enum_details is not None:
        return [{"path": path, "kind": "enum", **enum_details}]
    if value is None or isinstance(value, (bool, int, float, str)):
        row = {"path": path, "kind": type(value).__name__}
        if include_scalar_values:
            row["value"] = value
        return [row]
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return sum((_tree_signature(item, torch, f"{path}[{i}]", include_scalar_values=include_scalar_values)
                    for i, item in enumerate(value)), [])
    if isinstance(value, list):
        return sum((_tree_signature(item, torch, f"{path}[{i}]", include_scalar_values=include_scalar_values)
                    for i, item in enumerate(value)), [])
    if isinstance(value, dict):
        return sum((_tree_signature(item, torch, f"{path}.{key}", include_scalar_values=include_scalar_values)
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))), [])
    raise RuntimeError(f"Opaque/non-tensor value prevents safe replay at {path}: {type(value).__name__}")


def _copy_tree_(destination: Any, source: Any, torch, path: str = "root") -> None:
    if torch.is_tensor(destination) and torch.is_tensor(source):
        if destination.shape != source.shape or destination.dtype != source.dtype:
            raise RuntimeError(f"Static CUDA graph input shape/dtype differs at {path}")
        if tuple(destination.stride()) != tuple(source.stride()):
            raise RuntimeError(f"Static CUDA graph input stride differs at {path}")
        destination.copy_(source, non_blocking=False)
        return
    destination_enum = _cosmos_enum_details(destination, path)
    source_enum = _cosmos_enum_details(source, path)
    if destination_enum is not None or source_enum is not None:
        if destination_enum != source_enum:
            raise RuntimeError(f"Static CUDA graph Cosmos Enum differs at {path}")
        return
    if isinstance(source, dict) and len(source) == 1 and TORCH_DTYPE_MARKER in source:
        if not isinstance(destination, torch.dtype) or str(destination).removeprefix("torch.") != source[TORCH_DTYPE_MARKER]:
            raise RuntimeError(f"Static CUDA graph dtype argument differs at {path}")
        return
    if isinstance(source, dict) and len(source) == 1 and TORCH_DEVICE_MARKER in source:
        if not isinstance(destination, torch.device) or str(destination) != source[TORCH_DEVICE_MARKER]:
            raise RuntimeError(f"Static CUDA graph device argument differs at {path}")
        return
    if type(destination) is not type(source):
        raise RuntimeError(f"Static CUDA graph input structure differs at {path}")
    if isinstance(destination, (tuple, list)):
        if len(destination) != len(source):
            raise RuntimeError(f"Static CUDA graph input length differs at {path}")
        for index, (dst, src) in enumerate(zip(destination, source)):
            _copy_tree_(dst, src, torch, f"{path}[{index}]")
        return
    if isinstance(destination, dict):
        if destination.keys() != source.keys():
            raise RuntimeError(f"Static CUDA graph input keys differ at {path}")
        for key in destination:
            _copy_tree_(destination[key], source[key], torch, f"{path}.{key}")
        return
    if destination != source:
        raise RuntimeError(f"Static CUDA graph non-tensor argument differs at {path}")


def _atomic_torch_save(path: Path, payload: dict, torch) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size


def _load_torch(path: Path, torch) -> dict:
    # Existing captures saved torch.__version__ as TorchVersion, a str subclass.
    version_type = import_module("torch.torch_version").TorchVersion
    with torch.serialization.safe_globals([version_type]):
        return torch.load(path, map_location="cpu", weights_only=True)


def _capture_manifest_path(root: Path) -> Path:
    return root / "results" / "replay_inputs" / "manifest.json"


def _capture_one_query(common, runtime, context: dict, *, torch, device) -> dict:
    net = runtime.model.net
    original = net.forward
    calls = []

    def capture_forward(module, *args, **kwargs):
        result = original(*args, **kwargs)
        calls.append({"args": _cpu_tree(args, torch, "args"),
                      "kwargs": _cpu_tree(kwargs, torch, "kwargs"),
                      "output": _cpu_tree(result, torch, "output")})
        return result

    import types
    net.forward = types.MethodType(capture_forward, net)
    try:
        result = common.endpoint(runtime, context)
        torch.cuda.synchronize(device)
    finally:
        net.forward = original
    expected_nfe = int(runtime.protocol["inference"]["denoising_steps"])
    if (len(calls) != expected_nfe or int(result.sampler_evaluations) != expected_nfe
            or int(result.denoiser_calls) != expected_nfe):
        raise RuntimeError(f"Expected exactly {expected_nfe} captured model.net forwards, got {len(calls)}")
    if not math.isfinite(float(result.action.float().abs().max())):
        raise RuntimeError("BF16 capture produced a non-finite endpoint action")
    return {"calls": calls, "sampler_evaluations": int(result.sampler_evaluations),
            "denoiser_calls": int(result.denoiser_calls),
            "endpoint_action_shape": list(result.action.shape),
            "endpoint_future_shapes": {key: list(value.shape) for key, value in result.future_image_slots.items()}}


def _claim_capture_lock(root: Path) -> Path:
    path = root / "results" / "replay_inputs" / ".capture.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"A capture writer already owns {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()} host={socket.gethostname()} pbs={os.environ.get('PBS_JOBID')}\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _capture_panel(common, runtime, protocol: dict, old_root: Path, root: Path, out: Path,
                   torch, device, hardware: dict, deadline: float) -> dict:
    manifest_path = _capture_manifest_path(root)
    previous = read_json(manifest_path) if manifest_path.is_file() else None
    if previous is not None:
        for key, expected in (("format", CAPTURE_FORMAT), ("protocol", protocol["protocol"])):
            if previous.get(key) != expected:
                raise RuntimeError(f"Existing BF16 replay capture differs at {key}; refusing mixed capture")
        if compatible_gpu_key(previous.get("hardware", {})) != compatible_gpu_key(hardware):
            raise RuntimeError("Existing BF16 replay capture has a different GPU SKU/software compatibility binding")
    lock = _claim_capture_lock(root)
    try:
        records = {} if previous is None else dict(previous.get("queries", {}))
        for case_id, seed_index in query_keys(protocol):
            key = f"{case_id}_{seed_index}"
            path = capture_path(root, case_id, seed_index)
            if path.is_file():
                payload = _load_torch(path, torch)
                if (payload.get("format") != CAPTURE_FORMAT or payload.get("protocol") != protocol["protocol"]
                        or compatible_gpu_key(payload.get("hardware", {})) != compatible_gpu_key(hardware)
                        or payload.get("query") != [case_id, seed_index]):
                    raise RuntimeError(f"Existing replay capture binding differs: {path}")
                records[key] = {"file": str(path.relative_to(root)),
                                "bytes": path.stat().st_size,
                                "hardware_uuid": payload["hardware"].get("gpu_uuid"),
                                "hostname": payload["hardware"].get("hostname"),
                                "pbs_job_id": payload["hardware"].get("pbs_job_id")}
                continue
            if time.monotonic() >= deadline:
                break
            context = _load_context(common, old_root, case_id, seed_index)
            captured = _capture_one_query(common, runtime, context, torch=torch, device=device)
            signatures = []
            for index, call in enumerate(captured["calls"]):
                signatures.append({"args": _tree_signature(call["args"], torch, "args"),
                                   "kwargs": _tree_signature(call["kwargs"], torch, "kwargs"),
                                   "output": _tree_signature(call["output"], torch, "output",
                                                               include_scalar_values=False)})
            payload = {"format": CAPTURE_FORMAT, "protocol": protocol["protocol"],
                       "hardware": hardware, "query": [case_id, seed_index], **captured,
                       "forward_signatures": signatures}
            nbytes = _atomic_torch_save(path, payload, torch)
            records[key] = {"file": str(path.relative_to(root)), "bytes": nbytes,
                            "hardware_uuid": hardware.get("gpu_uuid"),
                            "hostname": hardware.get("hostname"),
                            "pbs_job_id": hardware.get("pbs_job_id")}
            _bounded_summary(manifest_path, {"format": CAPTURE_FORMAT, "protocol": protocol["protocol"],
                "status": "checkpointed", "hardware": hardware, "capture_arm": "bf16",
                "capture_scope": "actual common.endpoint five-NFE model.net.forward arguments and outputs",
                "query_count": len(records), "expected_query_count": 80,
                "queries": records, "created_utc": datetime.now(timezone.utc).isoformat()})
        complete = len(records) == 80
        summary = {"format": CAPTURE_FORMAT, "protocol": protocol["protocol"],
                   "status": "complete" if complete else "checkpointed", "complete": complete,
                   "hardware": hardware, "capture_arm": "bf16",
                   "capture_scope": "actual common.endpoint five-NFE model.net.forward arguments and outputs",
                   "query_count": len(records), "expected_query_count": 80,
                   "queries": records, "created_utc": datetime.now(timezone.utc).isoformat()}
        size = _bounded_summary(manifest_path, summary)
        return {"status": summary["status"], "complete": complete,
                "query_count": len(records), "expected_query_count": 80,
                "manifest": str(manifest_path), "summary_bytes": size}
    finally:
        lock.unlink(missing_ok=True)


def _read_capture_manifest(root: Path, protocol: dict, hardware: dict) -> dict:
    path = _capture_manifest_path(root)
    manifest = read_json(path)
    if (manifest.get("format") != CAPTURE_FORMAT or manifest.get("protocol") != protocol["protocol"]
            or manifest.get("status") != "complete" or manifest.get("query_count") != 80
            or compatible_gpu_key(manifest.get("hardware", {})) != compatible_gpu_key(hardware)):
        raise RuntimeError("A complete BF16 capture on this protocol and compatible GPU/software is required")
    if set(manifest.get("queries", {})) != {f"{case}_{seed}" for case, seed in query_keys(protocol)}:
        raise RuntimeError("BF16 replay capture does not cover all frozen 80 queries")
    return manifest


def _load_capture(root: Path, case_id: int, seed_index: int, torch) -> dict:
    payload = _load_torch(capture_path(root, case_id, seed_index), torch)
    if payload.get("query") != [case_id, seed_index] or len(payload.get("calls", [])) != 5:
        raise RuntimeError(f"Malformed five-NFE replay capture for {(case_id, seed_index)}")
    return payload


def _signatures_by_nfe(root: Path, protocol: dict, torch) -> list[dict]:
    signatures = None
    for case_id, seed_index in performance_query_keys(protocol):
        capture = _load_capture(root, case_id, seed_index, torch)
        rows = capture["forward_signatures"]
        if len(rows) != 5:
            raise RuntimeError("Replay capture lacks five static-forward signatures")
        if signatures is None:
            signatures = rows
        elif rows != signatures:
            raise RuntimeError(f"model.net input/output signature differs across the frozen panel at {case_id}/{seed_index}")
    return signatures or []


def _tensor_storage_bytes(tensors, torch, device=None) -> int:
    seen, total = set(), 0
    for tensor in tensors:
        if not torch.is_tensor(tensor) or (device is not None and tensor.device != device):
            continue
        try:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
            if key not in seen:
                seen.add(key)
                total += int(storage.nbytes())
        except RuntimeError:
            total += int(tensor.numel() * tensor.element_size())
    return total


def _non_target_dit_bytes(runtime, torch, device, *, bf16_arm: bool) -> int:
    resident_target_weights = [module for module in runtime.target_linears.values()
                               if module.weight is not None]
    if not bf16_arm and resident_target_weights:
        raise RuntimeError("Native low-bit arm still has dense target weights attached")
    target_adapter_buffers = set()
    for module in runtime.target_linears.values():
        for tensor in module.buffers(recurse=False):
            try:
                storage = tensor.untyped_storage()
                target_adapter_buffers.add((str(tensor.device), int(storage.data_ptr()),
                                            int(storage.nbytes())))
            except RuntimeError:
                pass
    tensors = []
    for parameter in runtime.model.net.parameters():
        tensors.append(parameter)
    seen, total = set(), 0
    for tensor in tensors + list(runtime.model.net.buffers()):
        if tensor.device != device:
            continue
        try:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
            if key in target_adapter_buffers or key in seen:
                continue
            seen.add(key)
            total += int(storage.nbytes())
        except RuntimeError:
            total += int(tensor.numel() * tensor.element_size())
    return total


def _packed_storage_receipt(runtime, metadata: dict, arm: str, torch, device) -> dict:
    if arm == "bf16":
        total = _non_target_dit_bytes(runtime, torch, device, bf16_arm=True)
        return {"adapter_resident_tensor_bytes": 0,
                "unquantized_dit_bytes": total,
                "packed_storage_bytes": total,
                "packed_storage_gib_binary": total / (2 ** 30),
                "storage_scope": "BF16 model.net unique CUDA parameter and buffer tensor storage"}
    if arm == "ours_vqwam":
        total = metadata.get("packed_storage_bytes")
        if (not isinstance(total, int) or isinstance(total, bool) or total <= 0
                or metadata.get("packed_gpu_resident") is not True):
            raise RuntimeError("Ours installer must report actual packed_gpu_resident packed_storage_bytes")
        unquantized = int(metadata.get("unquantized_dit_tensor_bytes", 0))
        adapter_bytes = total - unquantized
        if unquantized < 0 or adapter_bytes <= 0:
            raise RuntimeError("Ours installer must separate packed/transform and unquantized DiT storage")
        return {"adapter_resident_tensor_bytes": adapter_bytes,
                "unquantized_dit_bytes": unquantized,
                "packed_target_and_transform_bytes": adapter_bytes,
                "packed_storage_bytes": total,
                "packed_storage_gib_binary": total / (2 ** 30),
                "storage_scope": metadata.get("packed_storage_scope", "adapter-reported resident DiT tensor payload")}
    payload = metadata.get("packed_target_payload_bytes")
    if isinstance(payload, bool) or not isinstance(payload, int) or payload <= 0:
        payload = metadata.get("native_resident_tensor_bytes")
    if isinstance(payload, bool) or not isinstance(payload, int) or payload <= 0:
        raise RuntimeError("Native adapter must report actual GPU-resident packed_target_payload_bytes; checkpoint bytes are not accepted")
    unquantized = _non_target_dit_bytes(runtime, torch, device, bf16_arm=False)
    total = int(payload) + unquantized
    return {"adapter_resident_tensor_bytes": int(payload),
            "unquantized_dit_bytes": unquantized,
            "packed_storage_bytes": total,
            "packed_storage_gib_binary": total / (2 ** 30),
            "storage_scope": "adapter-reported resident packed target tensors plus unique unquantized model.net CUDA parameters and buffers"}


def _binding(metadata: dict, arm: str, variant: str, root: Path | None = None) -> dict:
    fields = ("implementation", "execution", "execution_scope", "native_backend",
              "native_backend_module", "native_int4_kernel_used", "native_vq_kernel",
              "native_kernel", "native_weight_format", "native_extension",
              "native_backend_source_revision", "native_backend_cutlass_revision",
              "adapter_version", "adapter_epilogue", "adapter_epilogue_source",
              "native_hadamard_backend", "native_hadamard_module", "native_hadamard_source_repository",
              "native_hadamard_source_commit", "native_hadamard_quarot_source_revision",
              "native_hadamard_submodule_path", "native_hadamard_build_receipt",
              "native_hadamard_build_producer", "native_hadamard_build_pbs_job_id",
              "native_hadamard_build_host", "native_hadamard_build_torch",
              "native_hadamard_build_cuda_runtime",
              'lora_down_kernel', 'lora_up_kernel', 'lora_factor_packing', 'lora_factor_runtime_dtype',
              'activation_recipe', 'sidecar_extra_groupwise_a4_quantization',
              "engine_variant", "artifact_dir", "artifact_manifest",
              "original_weight_bank", "rotation_artifact_source", "source_revision",
              "kernel_binding", "weight_group_size", "packed_target_payload_scope",
              "variant")
    result = {key: metadata[key] for key in fields if key in metadata}
    result.update({"arm": arm, "requested_variant": variant})
    if root is not None and metadata.get("native_int4_kernel_used") is True:
        result["extension_build"] = _extension_build_binding(root, arm, metadata)
    return result


def _extension_build_binding(root: Path, arm: str, metadata: dict) -> dict:
    """Bind native gate/performance reuse to the exact extension build producer."""
    if (arm in ('spinquant_w4a4', 'quarot_w4a4')
            and metadata.get('native_backend_module') == 'nunchaku_svdq_native'):
        candidates = (root / 'results/preparation/native_diffusion_build.json',
                      root / 'retrieved/preparation.native_diffusion_build.json')
        arm_build = 'arms.svdquant_w4a4'
        module_name = metadata.get('native_backend_module_path') or 'nunchaku_svdq_native'
        arm_label = arm + ' native_grouped'
    elif arm in ("spinquant_w4a4", "quarot_w4a4"):
        module_name = metadata.get("native_backend_module")
        if module_name == "quarot_official_gemm_fused":
            candidates = (root / "results" / "preparation" / "quarot_fused_build.json",
                          root / "retrieved" / "preparation.quarot_fused_build.json")
        elif module_name == "quarot_official_gemm":
            candidates = (root / "results" / "preparation" / "quarot_build.json",
                          root / "retrieved" / "preparation.quarot_build.json")
        else:
            raise RuntimeError(f"Unexpected native rotation module in build binding: {module_name!r}")
        arm_build = None
        arm_label = module_name
    elif arm in ("svdquant_w4a4", "viditq_w4a4"):
        module_name = metadata.get("native_extension") or metadata.get("native_backend_module")
        fused_svd = bool(module_name and Path(str(module_name)).name.startswith('nunchaku_svdq_fused_native'))
        general_atom = bool(module_name and Path(str(module_name)).name.startswith('viditq_atom_per_channel_native'))
        receipt_name = ('native_viditq_general_build.json' if general_atom else
                        'native_svd_fused_build.json' if fused_svd else 'native_diffusion_build.json')
        candidates = (root / "results" / "preparation" / receipt_name,
                      root / "retrieved" / ('preparation.' + receipt_name))
        arm_build = (None if general_atom else
                     'arms.svdquant_w4a4_fused_candidate' if fused_svd else "arms." + arm)
        arm_label = arm
    else:
        return {}

    receipt_path = next((path for path in candidates if path.is_file()), None)
    if receipt_path is None:
        raise FileNotFoundError(
            f"Native extension build receipt required for {arm}: "
            + ", ".join(str(path) for path in candidates))
    receipt = read_json(receipt_path)
    if receipt.get("status") != "complete" or not receipt.get("pbs_job_id"):
        raise RuntimeError(f"Native extension build receipt is incomplete: {receipt_path}")
    if arm_build is None:
        build = receipt
    else:
        build = receipt
        for key in arm_build.split("."):
            build = build.get(key, {}) if isinstance(build, dict) else {}
    if not isinstance(build, dict) or not build.get("module") or not build.get("source_commit"):
        raise RuntimeError(f"Build receipt lacks module/source_commit for {arm_label}: {receipt_path}")
    receipt_module = str(build["module"])
    def module_token(value):
        return Path(str(value)).name.split(".", 1)[0]

    if module_name and module_token(receipt_module) != module_token(module_name):
        raise RuntimeError(
            f"Build receipt module differs from active native extension for {arm}: "
            f"{receipt_module!r} vs {module_name!r}")
    source_binding = {"path": str(receipt_path.relative_to(root)) if root in receipt_path.parents
                       else receipt_path.name,
                      "pbs_job_id": receipt["pbs_job_id"],
                      "module": receipt_module,
                      "source_commit": build["source_commit"]}
    submodules = build.get("submodules", receipt.get("submodules", {}))
    if isinstance(submodules, dict):
        source_binding["submodule_commits"] = {
            name: row["commit"] for name, row in submodules.items()
            if isinstance(row, dict) and row.get("commit")}
    return source_binding


def _prepare_rotation_backend_import(root: Path) -> Path:
    """Prefer the fused QuaRot extension; use the older GEMM-only build only if absent."""
    extension_root = root / "native_extensions"
    fused = extension_root / "quarot_official_gemm_fused"
    legacy = extension_root / "quarot_official_gemm"
    selected = fused if fused.is_dir() else legacy if legacy.is_dir() else None
    if selected is None:
        raise FileNotFoundError(
            f"Neither fused nor legacy QuaRot GEMM extension directory exists under {extension_root}")
    path = str(selected.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    return selected


def _load_method_adapter(name: str):
    if name == "native_rotations":
        import native_rotations
        return native_rotations
    if name == 'native_svd_fused':
        import native_svd_fused
        return native_svd_fused
    if name == 'native_rotations_grouped':
        import native_rotations_grouped
        return native_rotations_grouped
    try:
        import native_diffusion
    except ImportError as exc:
        raise RuntimeError("native_diffusion.py must be available before SVDQuant/ViDiT-Q native runs") from exc
    return native_diffusion


def _install_reference(runtime, protocol: dict, old_root: Path, arm: str) -> dict:
    import runtime as legacy_runtime
    return legacy_runtime.install_arm(runtime, protocol, old_root, arm, optimized=False)


def _install_native(runtime, protocol: dict, old_root: Path, out: Path, arm: str,
                    variant: str, receipt_binding: dict, *, native_root: Path | None = None) -> tuple[dict, Any]:
    if arm == "bf16":
        import runtime as legacy_runtime
        metadata = legacy_runtime.install_arm(runtime, protocol, old_root, "bf16", optimized=False)
        return metadata, None
    if arm == "ours_vqwam":
        if variant == "current":
            import engine_profile
            metadata = engine_profile._install_arm(runtime, protocol, old_root, out,
                                                   "ours_vqwam", receipt_binding)
            return metadata, getattr(runtime, "restore_ours_vqwam", None)
        import engine_optimized
        result = engine_optimized.install(runtime, protocol, old_root, out)
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        if not isinstance(result, dict):
            raise TypeError("engine_optimized.install must return metadata or (metadata, restore_hook)")
        return result, getattr(runtime, "restore_optimized_engine", None)

    _install_reference(runtime, protocol, old_root, arm)
    adapter_name = ('native_rotations_grouped' if variant == 'native_grouped' else
                    "native_rotations" if arm in ("spinquant_w4a4", "quarot_w4a4") else
                    'native_svd_fused' if variant == 'native_fused' else "native_diffusion")
    adapter = _load_method_adapter(adapter_name)
    installer = getattr(adapter, "install_native", None)
    if not callable(installer):
        raise RuntimeError(f"{adapter_name}.install_native(runtime, protocol, old_root, arm, ...) is required")
    if adapter_name == 'native_rotations':
        result = installer(runtime, protocol, old_root, arm)
    elif adapter_name == 'native_rotations_grouped':
        if native_root is None:
            raise RuntimeError('An explicit native build root is required for grouped rotation adapters')
        result = installer(runtime, protocol, old_root, arm, native_root=native_root)
    else:
        if native_root is None:
            raise RuntimeError('An explicit native build root is required for diffusion adapters')
        result = installer(runtime, protocol, old_root, arm, artifact_dir=native_root)
    if isinstance(result, tuple) and len(result) == 2:
        metadata, restore = result
    else:
        metadata, restore = result, getattr(runtime, "restore_native_diffusion", None)
    if not isinstance(metadata, dict):
        raise TypeError(f"{adapter_name}.install_native must return metadata mapping")
    return metadata, restore


def _restore_to_bf16(runtime, protocol: dict, old_root: Path, restore_hook=None) -> None:
    if callable(restore_hook):
        restore_hook()
    try:
        import runtime as legacy_runtime
        legacy_runtime.install_arm(runtime, protocol, old_root, "bf16", optimized=False)
    except Exception:
        controller = getattr(runtime, "controller", None)
        if controller is not None:
            controller._restore_current()
            controller.arm = "bf16"
            controller.metadata = controller._bf16_metadata()
        raise


KNOWN_NON_DIT_WRAPPERS = (
    "tokenizer", "vae", "video_vae", "text_encoder", "textencoder",
    "t5_encoder", "vision_encoder", "image_encoder")


def _is_cuda_tensor(tensor) -> bool:
    return getattr(getattr(tensor, "device", None), "type", None) == "cuda"


def _tensor_storage_key(tensor):
    try:
        storage = tensor.untyped_storage()
        return (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
    except (AttributeError, RuntimeError, TypeError):
        return None


def _known_wrapper_module_roots(model, torch):
    module_type = getattr(getattr(torch, "nn", None), "Module", None)
    if module_type is None:
        raise RuntimeError("Torch nn.Module is required to inspect known endpoint wrappers")
    for name in KNOWN_NON_DIT_WRAPPERS:
        wrapper = getattr(model, name, None)
        if isinstance(wrapper, module_type):
            yield name, wrapper
        nested = getattr(wrapper, "model", None)
        if isinstance(nested, module_type):
            yield f"{name}.model", nested


def _offload_non_dit(runtime, torch, device) -> dict:
    model, net = runtime.model, runtime.model.net
    net_tensors = list(net.parameters()) + list(net.buffers())
    net_ids = {id(tensor) for tensor in net_tensors}
    net_storage = {key for tensor in net_tensors
                   if (key := _tensor_storage_key(tensor)) is not None}
    rows, checked, visited, retained_shared_tensors = [], [], set(), []

    def module_tensors(module):
        return list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))

    def direct_module_tensors(module):
        return list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))

    def shared_with_net(tensor):
        key = _tensor_storage_key(tensor)
        return id(tensor) in net_ids or (key is not None and key in net_storage)

    def contains_net(module):
        return module is net or any(child is net for child in module.modules())

    def mark_tree_visited(module):
        for child in module.modules():
            visited.add(id(child))

    def walk(module, path):
        identity = id(module)
        if identity in visited:
            return
        visited.add(identity)
        if module is net:
            return
        if contains_net(module):
            for name, child in module.named_children():
                child_path = f"{path}.{name}" if path else name
                walk(child, child_path)
            return

        tensors = module_tensors(module)
        cuda_tensors = [tensor for tensor in tensors if _is_cuda_tensor(tensor)]
        if cuda_tensors:
            shared = [tensor for tensor in cuda_tensors if shared_with_net(tensor)]
            if shared:
                # Never call .to(cpu) on a wrapper that aliases a DiT tensor.
                # Its independent children can still be offloaded separately.
                for name, child in module.named_children():
                    child_path = f"{path}.{name}" if path else name
                    walk(child, child_path)
                remaining = [tensor for tensor in module_tensors(module)
                             if _is_cuda_tensor(tensor)]
                unexpected = [tensor for tensor in remaining if not shared_with_net(tensor)]
                if unexpected:
                    raise RuntimeError(
                        f"Shared-with-DiT wrapper retains unrelated CUDA tensors after child offload: {path}")
                direct_cuda = [tensor for tensor in direct_module_tensors(module)
                               if _is_cuda_tensor(tensor)]
                shared_direct = [tensor for tensor in direct_cuda if shared_with_net(tensor)]
                if shared_direct:
                    shared_bytes = _tensor_storage_bytes(shared_direct, torch)
                    rows.append({"module": path, "status": "kept_shared_with_model_net",
                                 "cuda_tensor_bytes": shared_bytes,
                                 "remaining_cuda_tensor_bytes": shared_bytes})
                    checked.append((module, path, "shared"))
                    retained_shared_tensors.extend(shared_direct)
                return

            before = _tensor_storage_bytes(cuda_tensors, torch)
            module.to(device="cpu")
            remaining = [tensor for tensor in module_tensors(module) if _is_cuda_tensor(tensor)]
            if remaining:
                raise RuntimeError(
                    f"Recognized non-DiT module still has CUDA parameters/buffers after offload: {path}")
            rows.append({"module": path, "status": "offloaded_to_cpu",
                         "cuda_tensor_bytes": before, "remaining_cuda_tensor_bytes": 0})
            checked.append((module, path, "offloaded"))
            mark_tree_visited(module)
            return

        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            walk(child, child_path)

    walk(model, "")
    wrapper_roots = []
    for path, module in _known_wrapper_module_roots(model, torch):
        wrapper_roots.append(path)
        walk(module, path)

    # Check every identified CUDA-bearing wrapper again after all overlapping
    # roots have been visited. The only permitted remainder aliases model.net.
    for module, path, status in checked:
        remaining = [tensor for tensor in module_tensors(module) if _is_cuda_tensor(tensor)]
        if status == "offloaded" and remaining:
            raise RuntimeError(f"Post-offload CUDA tensor remains in {path}")
        if status == "shared" and any(not shared_with_net(tensor) for tensor in remaining):
            raise RuntimeError(f"Post-offload non-DiT CUDA tensor remains in shared wrapper {path}")

    resident = [tensor for tensor in net_tensors if tensor.device == device]
    if not resident:
        raise RuntimeError("model.net lost all CUDA tensors during non-DiT offload")
    stranded = [str(tensor.device) for tensor in net_tensors
                if _is_cuda_tensor(tensor) and tensor.device != device]
    if stranded:
        raise RuntimeError("model.net spans unexpected CUDA devices after offload")
    return {"scope": "model.net only; registered endpoint children plus explicit tokenizer/vae/text-encoder wrappers checked and offloaded before warmup/peak reset",
            "known_wrapper_roots": wrapper_roots,
            "modules": rows,
            "checked_cuda_module_count": len(checked),
            "post_offload_verification": "every identified offloaded module has zero CUDA parameters/buffers; retained tensors must alias model.net",
            "offloaded_cuda_tensor_bytes": sum(row["cuda_tensor_bytes"] for row in rows
                                                if row["status"] == "offloaded_to_cpu"),
            "retained_shared_with_net_cuda_tensor_bytes": _tensor_storage_bytes(
                retained_shared_tensors, torch)}


def _make_graph_bundle(net, captures: list[dict], device, torch) -> list[dict]:
    if len(captures) != 5:
        raise RuntimeError("DiT replay requires exactly five captured forward calls")
    bundles = []
    shared_pool = torch.cuda.graph_pool_handle()
    for nfe, call in enumerate(captures):
        args = _device_tree(call["args"], device, torch, f"nfe{nfe}.args")
        kwargs = _device_tree(call["kwargs"], device, torch, f"nfe{nfe}.kwargs")
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        warm_output = None
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                warm_output = net(*args, **kwargs)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        output_buffer = _empty_output_tree(warm_output, torch, f"nfe{nfe}.output")
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=shared_pool):
            output = net(*args, **kwargs)
            _copy_output_tree_(output_buffer, output, torch, f"nfe{nfe}.output")
        torch.cuda.synchronize(device)
        bundles.append({"graph": graph, "args": args, "kwargs": kwargs,
                        "output": output_buffer, "nfe": nfe,
                        "graph_backend": "generic_net_forward_cuda_graph"})
    return bundles


def _empty_output_tree(value: Any, torch, path: str) -> Any:
    if torch.is_tensor(value):
        return torch.empty_like(value, memory_format=torch.preserve_format)
    if isinstance(value, dict):
        return {key: _empty_output_tree(item, torch, f"{path}.{key}")
                for key, item in value.items()}
    if isinstance(value, list):
        return [_empty_output_tree(item, torch, f"{path}[{index}]")
                for index, item in enumerate(value)]
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return tuple(_empty_output_tree(item, torch, f"{path}[{index}]")
                     for index, item in enumerate(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise RuntimeError(f"Unsupported net output tree for graph replay at {path}: {type(value).__name__}")


def _copy_output_tree_(destination: Any, source: Any, torch, path: str) -> None:
    if torch.is_tensor(destination) and torch.is_tensor(source):
        if (tuple(destination.shape) != tuple(source.shape)
                or destination.dtype != source.dtype or destination.device != source.device):
            raise RuntimeError(f"CUDA graph output shape/dtype/device differs at {path}")
        destination.copy_(source, non_blocking=False)
        return
    if type(destination) is not type(source):
        raise RuntimeError(f"CUDA graph output structure differs at {path}")
    if isinstance(destination, (tuple, list)):
        if len(destination) != len(source):
            raise RuntimeError(f"CUDA graph output length differs at {path}")
        for index, (dst, src) in enumerate(zip(destination, source)):
            _copy_output_tree_(dst, src, torch, f"{path}[{index}]")
        return
    if isinstance(destination, dict):
        if destination.keys() != source.keys():
            raise RuntimeError(f"CUDA graph output mapping keys differ at {path}")
        for key in destination:
            _copy_output_tree_(destination[key], source[key], torch, f"{path}.{key}")
        return
    if destination is None or isinstance(destination, (bool, int, float, str)):
        if type(destination) is not type(source):
            raise RuntimeError(f"CUDA graph output scalar type differs at {path}")
        return
    raise RuntimeError(f"Unsupported net output tree for graph replay at {path}: {type(source).__name__}")


@contextmanager
def _dit_replay_dispatch(runtime):
    """Expose raw net.forward while building generic replay graphs for Ours optimized."""
    optimized_graphs = getattr(runtime, "optimized_vq_graphs", None)
    if optimized_graphs is None:
        yield "model.net.__call__ generic CUDA graphs"
        return
    net = runtime.model.net
    wrapped_forward = net.forward
    raw_forward = optimized_graphs.original
    if not callable(raw_forward):
        raise RuntimeError("Ours optimized graph manager has no raw net.forward for generic DiT replay")
    optimized_graphs.records.clear()
    net.forward = raw_forward
    try:
        yield "raw model.net.forward CUDA graphs; Ours all-NFE/Cross-KV scheduling bypassed in DiT scope"
    finally:
        net.forward = wrapped_forward
        optimized_graphs.records.clear()


def _replay_five(bundles: list[dict], calls: list[dict], device, torch) -> tuple[float, int]:
    if len(bundles) != 5 or len(calls) != 5:
        raise RuntimeError("A timed DiT sample must replay the frozen five NFE inputs")
    if [int(bundle["nfe"]) for bundle in bundles] != list(range(5)):
        raise RuntimeError("Five DiT graphs must replay serially in frozen NFE 0..4 order")
    for bundle, call in zip(bundles, calls):
        _copy_tree_(bundle["args"], call["args"], torch, f"nfe{bundle['nfe']}.args")
        _copy_tree_(bundle["kwargs"], call["kwargs"], torch, f"nfe{bundle['nfe']}.kwargs")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for bundle in bundles:
        bundle["graph"].replay()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)) / 1000.0, int(torch.cuda.max_memory_allocated(device))


def _flat_tensors(value: Any, torch, path: str = "root") -> dict[str, Any]:
    if torch.is_tensor(value):
        return {path: value}
    if isinstance(value, dict):
        return {key: tensor for name, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                for key, tensor in _flat_tensors(item, torch, f"{path}.{name}").items()}
    if isinstance(value, (list, tuple)):
        return {key: tensor for index, item in enumerate(value)
                for key, tensor in _flat_tensors(item, torch, f"{path}[{index}]").items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {}
    raise RuntimeError(f"Unsupported output type in fidelity comparison: {type(value).__name__}")


def compare_tensor_trees(reference, candidate, torch) -> dict:
    left, right = _flat_tensors(reference, torch), _flat_tensors(candidate, torch)
    if left.keys() != right.keys():
        raise RuntimeError("Reference/native output tree paths differ")
    rows = {}
    sum_diff2 = sum_ref2 = 0.0
    count = 0
    max_abs = 0.0
    for path in left:
        a, b = left[path], right[path]
        if tuple(a.shape) != tuple(b.shape):
            raise RuntimeError(f"Reference/native output shape differs at {path}: {a.shape} vs {b.shape}")
        af, bf = a.float(), b.float()
        if not bool(torch.isfinite(af).all()) or not bool(torch.isfinite(bf).all()):
            raise RuntimeError(f"Non-finite reference/native output at {path}")
        delta = af - bf
        diff2 = float(delta.double().square().sum().item())
        ref2 = float(af.double().square().sum().item())
        n = int(af.numel())
        local_max = float(delta.abs().max().item()) if n else 0.0
        rows[path] = {"elements": n, "max_abs": local_max,
                      "rmse": math.sqrt(diff2 / max(n, 1)),
                      "nrmse": math.sqrt(diff2 / max(ref2, 1e-30))}
        sum_diff2 += diff2
        sum_ref2 += ref2
        count += n
        max_abs = max(max_abs, local_max)
    return {"elements": count, "max_abs": max_abs,
            "rmse": math.sqrt(sum_diff2 / max(count, 1)),
            "nrmse": math.sqrt(sum_diff2 / max(sum_ref2, 1e-30)),
            "by_tensor": rows}


def _result_trees(result) -> dict:
    return {"action": result.action, "future": result.future_image_slots}


def _sample_operator_tensor(tensor, width: int, torch):
    if not torch.is_tensor(tensor) or tensor.ndim < 1 or int(tensor.shape[-1]) != width:
        raise RuntimeError("Target Linear operator gate requires a Tensor input/output with matching feature width")
    flat = tensor.reshape(-1, width)
    total_rows = int(flat.shape[0])
    if total_rows <= 0:
        raise RuntimeError("Target Linear operator gate encountered an empty activation")
    sampled_rows = min(32, total_rows)
    if sampled_rows == 1:
        indices_values = [0]
    else:
        indices_values = [round(index * (total_rows - 1) / (sampled_rows - 1))
                          for index in range(sampled_rows)]
    indices = torch.tensor(indices_values, device=tensor.device, dtype=torch.long)
    return flat.index_select(0, indices).contiguous(), indices, total_rows


def _run_net_captures(runtime, capture: dict, torch, device, *, capture_operators: bool = False):
    outputs = []
    operator_inputs = []
    with torch.inference_mode():
        for nfe, call in enumerate(capture["calls"]):
            args = _device_tree(call["args"], device, torch, "args")
            kwargs = _device_tree(call["kwargs"], device, torch, "kwargs")
            handles, records, active = [], {}, {}
            if capture_operators:
                for name, module in sorted(runtime.target_linears.items()):
                    records[name] = []

                    def before(current, current_args, current_kwargs, *, _name=name):
                        if len(current_args) != 1 or current_kwargs:
                            raise RuntimeError(f"Target Linear {_name} has nonstandard args; exact-input gate cannot sample safely")
                        sampled, indices, total_rows = _sample_operator_tensor(
                            current_args[0], int(current.in_features), torch)
                        row = {"args": (sampled.detach().to(device="cpu", copy=True).contiguous(),),
                               "kwargs": {}, "indices": indices.detach().cpu().tolist(),
                               "sampled_rows": int(sampled.shape[0]), "total_rows": total_rows}
                        records[_name].append(row)
                        active.setdefault(_name, []).append(row)

                    def after(current, current_args, current_kwargs, output, *, _name=name):
                        if not active.get(_name):
                            raise RuntimeError(f"Target Linear output hook ran without input: {_name}")
                        row = active[_name].pop()
                        sampled, _indices, total_rows = _sample_operator_tensor(
                            output, int(current.out_features), torch)
                        if total_rows != row["total_rows"]:
                            raise RuntimeError(f"Target Linear {_name} input/output row counts differ")
                        row["output"] = sampled.detach().to(device="cpu", copy=True).contiguous()

                    handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                    handles.append(module.register_forward_hook(after, with_kwargs=True))
            try:
                value = runtime.model.net(*args, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            outputs.append(_cpu_tree(value, torch, "output"))
            if capture_operators:
                if any(active.values()) or any(not rows or any("output" not in row for row in rows)
                                               for rows in records.values()):
                    raise RuntimeError("Representative DiT pass did not capture every target Linear input/output")
                operator_inputs.append({"nfe": nfe, "modules": records})
    torch.cuda.synchronize(device)
    return (outputs, operator_inputs) if capture_operators else outputs


def _replay_native_target_operators(runtime, reference_operators, torch, device) -> dict:
    rows = []
    for nfe_row in reference_operators:
        nfe = int(nfe_row["nfe"])
        modules = nfe_row["modules"]
        if set(modules) != set(runtime.target_linears):
            raise RuntimeError("Saved operator gate module set differs from frozen target Linears")
        with torch.inference_mode():
            for name, module in sorted(runtime.target_linears.items()):
                for call_index, reference_call in enumerate(modules[name]):
                    args = _device_tree(reference_call["args"], device, torch, f"{name}.args")
                    kwargs = _device_tree(reference_call["kwargs"], device, torch, f"{name}.kwargs")
                    native_output = module(*args, **kwargs)
                    candidate = _cpu_tree(native_output, torch, f"{name}.native_output")
                    metrics = compare_tensor_trees(reference_call["output"], candidate, torch)
                    rows.append({"nfe": nfe, "module": name, "call_index": call_index,
                                 "sampled_rows": int(reference_call["sampled_rows"]),
                                 "total_rows": int(reference_call["total_rows"]),
                                 "nrmse": metrics["nrmse"], "rmse": metrics["rmse"],
                                 "max_abs": metrics["max_abs"],
                                 "elements": metrics["elements"]})
    if not rows:
        raise RuntimeError("No native target-Linear operator comparisons were completed")
    covered = {(row["nfe"], row["module"]) for row in rows}
    expected = {(nfe, name) for nfe in range(5) for name in runtime.target_linears}
    if covered != expected:
        missing = sorted(expected - covered)
        raise RuntimeError(f"Operator gate does not cover all 280 targets x five NFE; missing={missing[:8]}")
    worst = max(rows, key=lambda row: row["nrmse"])
    return {"comparison_scope": "native target Linear against decoded reference on identical evenly sampled input activation rows; not a full-token estimate",
            "requested_sampled_rows_per_operator": 32,
            "sampled_rows_policy": "uniformly spaced including first/last; min(32, total input rows)",
            "nfe_indices": sorted({row["nfe"] for row in rows}),
            "target_linear_count": len(runtime.target_linears),
            "operator_calls": len(rows), "nrmse_max": worst["nrmse"],
            "max_abs_max": max(row["max_abs"] for row in rows),
            "worst_operator": {"nfe": worst["nfe"], "module": worst["module"],
                               "call_index": worst["call_index"], "nrmse": worst["nrmse"],
                               "max_abs": worst["max_abs"]}, "per_operator": rows}


def _gate_reference_outputs(common, runtime, protocol, old_root, arm, scope,
                            variant, queries, root, torch, device, hardware,
                            deadline: float, *, capture_operators: bool = False) -> dict:
    metadata = _install_reference(runtime, protocol, old_root, arm)
    reference_binding = _binding(metadata, arm, variant, root)
    saved = {}
    try:
        for case_id, seed_index in queries:
            if time.monotonic() >= deadline:
                break
            key = f"{case_id}_{seed_index}"
            path = _gate_reference_path(root, scope, variant, arm, key)
            if path.is_file():
                prior = _load_torch(path, torch)
                if (prior.get("protocol") != protocol["protocol"] or prior.get("arm") != arm
                        or prior.get("variant") != variant or prior.get("scope") != scope
                        or prior.get("hardware") != compatible_gpu_key(hardware)
                        or prior.get("reference_binding") != reference_binding
                        or prior.get("query") != [case_id, seed_index]):
                    raise RuntimeError(f"Saved gate reference binding differs: {path}")
                saved[key] = str(path.relative_to(root))
                continue
            if scope == "sampler":
                result = common.endpoint(runtime, _load_context(common, old_root, case_id, seed_index))
                values = _cpu_tree(_result_trees(result), torch)
                operator_inputs = None
                if (int(result.sampler_evaluations) != 5 or int(result.denoiser_calls) != 5):
                    raise RuntimeError("Method-specific reference endpoint left the five-NFE frozen path")
            else:
                outputs = _run_net_captures(runtime, _load_capture(root, case_id, seed_index, torch),
                    torch, device, capture_operators=capture_operators)
                if capture_operators:
                    values, operator_inputs = outputs
                else:
                    values, operator_inputs = outputs, None
            _atomic_torch_save(path, {"protocol": protocol["protocol"], "arm": arm,
                                      "variant": variant, "scope": scope,
                                      "hardware": compatible_gpu_key(hardware),
                                      "hardware_uuid_provenance": hardware,
                                      "reference_binding": reference_binding,
                                      "query": [case_id, seed_index],
                                      "output": values,
                                      "operator_inputs": operator_inputs}, torch)
            saved[key] = str(path.relative_to(root))
    finally:
        _restore_to_bf16(runtime, protocol, old_root)
    return {"metadata": metadata, "outputs": saved, "complete": len(saved) == len(queries)}


def _gate_reference_path(root: Path, scope: str, variant: str, arm: str, key: str) -> Path:
    return root / "results" / "native_gate" / scope / variant / arm / "decoded_reference" / f"{key}.pt"


def _gate_query_metrics(scope: str, reference, candidate, torch) -> dict:
    if scope == "dit_replay":
        if len(reference) != 5 or len(candidate) != 5:
            raise RuntimeError("DiT fidelity gate requires five per-NFE outputs")
        rows = [compare_tensor_trees(reference[i], candidate[i], torch) for i in range(5)]
        return {"dit_forward_nrmse": max(row["nrmse"] for row in rows),
                "dit_forward_max_abs": max(row["max_abs"] for row in rows),
                "per_nfe": rows}
    if set(reference["future"]) != set(candidate["future"]):
        raise RuntimeError("Reference/native endpoint future keys differ")
    action = compare_tensor_trees(reference["action"], candidate["action"], torch)
    motor = compare_tensor_trees(reference['action'][..., :6], candidate['action'][..., :6], torch)
    futures = {key: compare_tensor_trees(reference["future"][key], candidate["future"][key], torch)
               for key in sorted(reference["future"])}
    return {"normalized_motor_rmse": motor["nrmse"], "action_max_abs": action["max_abs"],
            "normalized_full_action_rmse": action['nrmse'],
            "motor_max_abs": motor["max_abs"],
            "future_nrmse": max(row["nrmse"] for row in futures.values()),
            "future_max_abs": max(row["max_abs"] for row in futures.values()),
            "action": action, "future": futures}


def _gate_limits(scope: str, arm: str, variant: str, *, full_panel: bool) -> dict:
    if scope == "dit_replay":
        if arm == "ours_vqwam" and variant == "optimized" and full_panel:
            return {"dit_forward_max_abs": OURS_OPTIMIZED_ENDPOINT_THRESHOLD}
        limits = {"dit_forward_nrmse": GATE_THRESHOLDS["dit_forward_nrmse"]}
        if not full_panel:
            limits["linear_operator_nrmse"] = GATE_THRESHOLDS["dit_forward_nrmse"]
        return limits
    if arm == "ours_vqwam" and variant == "optimized" and full_panel:
        return {"action_max_abs": OURS_OPTIMIZED_ENDPOINT_THRESHOLD,
                "future_max_abs": OURS_OPTIMIZED_ENDPOINT_THRESHOLD}
    return {"normalized_motor_rmse": GATE_THRESHOLDS["normalized_motor_rmse"],
            "future_nrmse": GATE_THRESHOLDS["future_nrmse"]}


def _gate_pass(metrics: dict, limits: dict) -> bool:
    return all(math.isfinite(metrics.get(name, float("nan"))) and metrics[name] <= limit
               for name, limit in limits.items())


def _gate_panel(common, runtime, protocol: dict, old_root: Path, root: Path, out: Path,
                arm: str, variant: str, scope: str, torch, device, hardware: dict,
                receipt_binding: dict, deadline: float) -> dict:
    if arm == "bf16":
        raise ValueError("BF16 is the reference arm and has no native-vs-decoded fidelity gate")
    all_queries = query_keys(protocol)
    diagnostic = list(DIAGNOSTIC_QUERIES)
    representative_passed = False
    full_passed = False
    representative_status = "not_run"
    full_status = "not_run"
    native_binding = {}
    for phase, queries in (("representative4", diagnostic), ("full80", all_queries)):
        if phase == "full80" and not representative_passed:
            break
        if len(queries) > 4 and time.monotonic() >= deadline:
            break
        ref = _gate_reference_outputs(common, runtime, protocol, old_root, arm, scope,
                                      variant, queries, root, torch, device, hardware, deadline,
                                      capture_operators=(scope == "dit_replay" and phase == "representative4"))
        if not ref["complete"]:
            phase_status = "checkpointed"
            _bounded_summary(root / "results" / "native_gate" / scope / variant / arm / f"{phase}.json", {
                "format": FORMAT, "protocol": protocol["protocol"], "arm": arm,
                "variant": variant, "scope": scope, "phase": phase,
                "status": phase_status, "completed_reference_queries": len(ref["outputs"]),
                "completed_queries": 0, "expected_queries": len(queries),
                "thresholds": _gate_limits(scope, arm, variant, full_panel=(phase == "full80")),
                "hardware": hardware, "reference_binding": _binding(ref["metadata"], arm, variant, root),
                "created_utc": datetime.now(timezone.utc).isoformat()})
            if phase == "representative4":
                representative_status = phase_status
            else:
                full_status = phase_status
            break
        native_metadata, restore_hook = _install_native(runtime, protocol, old_root, out,
                                                        arm, variant, receipt_binding, native_root=root)
        native_binding = _binding(native_metadata, arm, variant, root)
        query_metrics = []
        try:
            for case_id, seed_index in queries:
                if time.monotonic() >= deadline:
                    break
                key = f"{case_id}_{seed_index}"
                ref_payload = _load_torch(_gate_reference_path(root, scope, variant, arm, key), torch)
                limits = _gate_limits(scope, arm, variant, full_panel=(phase == "full80"))
                gate_sample_path = (root / "results" / "native_gate" / scope / variant / arm
                                    / "samples" / f"{phase}_{key}.json")
                expected_identity = {"protocol": protocol["protocol"], "arm": arm,
                    "variant": variant, "scope": scope, "phase": phase,
                    "hardware": compatible_gpu_key(hardware), "backend_binding": native_binding,
                    "query": [case_id, seed_index], "limits": limits}
                saved = reusable_sample(gate_sample_path, expected_identity)
                if saved is not None:
                    query_metrics.append({"case_id": case_id, "seed_index": seed_index,
                                          "metrics": saved["metrics"], "limits": limits,
                                          "status": saved["status"],
                                          "hardware_uuid_provenance": saved.get("hardware_receipt")})
                    if saved["status"] == "failed":
                        break
                    continue
                if scope == "sampler":
                    result = common.endpoint(runtime, _load_context(common, old_root, case_id, seed_index))
                    if int(result.sampler_evaluations) != 5 or int(result.denoiser_calls) != 5:
                        raise RuntimeError("Native endpoint left the five-NFE frozen path")
                    candidate = _cpu_tree(_result_trees(result), torch)
                else:
                    candidate = _run_net_captures(runtime, _load_capture(root, case_id, seed_index, torch),
                                                  torch, device)
                metrics = _gate_query_metrics(scope, ref_payload["output"], candidate, torch)
                if scope == "dit_replay" and phase == "representative4":
                    if not isinstance(ref_payload.get("operator_inputs"), list):
                        raise RuntimeError("Representative reference capture lacks exact target Linear activations")
                    operator_metrics = _replay_native_target_operators(
                        runtime, ref_payload["operator_inputs"], torch, device)
                    if operator_metrics["nfe_indices"] != list(range(5)):
                        raise RuntimeError("Target Linear operator gate did not cover all five NFE indices")
                    metrics.update({"linear_operator_nrmse": operator_metrics["nrmse_max"],
                                    "linear_operator_max_abs": operator_metrics["max_abs_max"],
                                    "linear_operator_worst": operator_metrics["worst_operator"],
                                    "linear_operator_sampled_rows": operator_metrics["requested_sampled_rows_per_operator"],
                                    "linear_operator_calls": operator_metrics["operator_calls"]})
                    for nfe in operator_metrics["nfe_indices"]:
                        operator_record = {"format": FORMAT, "protocol": protocol["protocol"],
                            "arm": arm, "variant": variant, "scope": scope, "phase": phase,
                            "query": [case_id, seed_index], "nfe": nfe,
                            "hardware": hardware, "backend_binding": native_binding,
                            "sampled_rows_per_operator": 32,
                            "per_operator": [row for row in operator_metrics["per_operator"]
                                             if row["nfe"] == nfe],
                            "created_utc": datetime.now(timezone.utc).isoformat()}
                        operator_path = (root / "results" / "native_gate" / scope / variant / arm
                                         / "operators" / f"{phase}_{key}_nfe{nfe}.json")
                        _bounded_summary(operator_path, operator_record)
                passed = _gate_pass(metrics, limits)
                query_metrics.append({"case_id": case_id, "seed_index": seed_index,
                                      "metrics": metrics, "limits": limits,
                                      "status": "passed" if passed else "failed",
                                      "hardware_uuid_provenance": hardware})
                _atomic_json(gate_sample_path, {**expected_identity, "metrics": metrics,
                    "status": "passed" if passed else "failed", "hardware_receipt": hardware,
                    "pbs_job_id": os.environ.get("PBS_JOBID"), "hostname": socket.gethostname()})
                if not passed:
                    break
        finally:
            _restore_to_bf16(runtime, protocol, old_root, restore_hook)
        has_failed_query = any(row["status"] == "failed" for row in query_metrics)
        phase_status = ("failed" if has_failed_query else
                        "checkpointed" if len(query_metrics) != len(queries) else "passed")
        phase_passed = phase_status == "passed"
        _bounded_summary(root / "results" / "native_gate" / scope / variant / arm / f"{phase}.json", {
            "format": FORMAT, "protocol": protocol["protocol"], "arm": arm,
            "variant": variant, "scope": scope, "phase": phase,
            "status": phase_status,
            "completed_queries": len(query_metrics), "expected_queries": len(queries),
            "thresholds": _gate_limits(scope, arm, variant, full_panel=(phase == "full80")),
            "hardware": hardware, "backend_binding": _binding(native_metadata, arm, variant, root),
            "hardware_compatibility_contract": compatible_gpu_key(hardware),
            "hardware_uuid_provenance": "per-query records may originate from different GPU UUIDs with identical compatibility contract",
            "decoded_reference": ref["metadata"], "queries": query_metrics,
            "created_utc": datetime.now(timezone.utc).isoformat()})
        if phase == "representative4":
            representative_passed = phase_passed
            representative_status = phase_status
        else:
            full_passed = phase_passed
            full_status = phase_status
    return {"representative4_passed": bool(representative_passed),
            "full80_passed": bool(full_passed), "backend_binding": native_binding,
            "representative4_status": representative_status,
            "full80_status": full_status,
            "status": ("passed" if full_passed else
                       "representative_gate_failed" if representative_status == "failed" else
                       "representative_gate_checkpointed" if representative_status == "checkpointed" else
                       "full80_failed" if full_status == "failed" else
                       "full80_checkpointed")}


def _deadline(args) -> float:
    started = os.environ.get("COSMOS_JOB_START_EPOCH")
    if started:
        start_epoch = float(started)
        return time.monotonic() + max(0.0, args.soft_deadline_seconds - (time.time() - start_epoch))
    return time.monotonic() + args.soft_deadline_seconds


def baseline_fidelity_is_report_only(native_protocol: dict, arm: str) -> bool:
    return (arm in ARMS[1:5]
            and native_protocol.get("baseline_fidelity_policy") == "report_only")


def _baseline_report_only_request(root: Path, arm: str, variant: str) -> bool:
    path = root / "native_protocol.json"
    protocol = read_json(path) if path.is_file() else {}
    if not baseline_fidelity_is_report_only(protocol, arm):
        return False
    validate_variant(arm, variant)
    if protocol.get("selected_variants", {}).get(arm) != variant:
        raise RuntimeError(f"Unselected report-only native backend: {arm}/{variant}")
    return True


def _require_gate(root: Path, arm: str, variant: str, scope: str) -> None:
    if arm == "bf16":
        return
    if _baseline_report_only_request(root, arm, variant):
        return
    for gate_scope in ("dit_replay", "sampler"):
        path = root / "results" / "native_gate" / gate_scope / variant / arm / "full80.json"
        summary = read_json(path) if path.is_file() else {}
        if summary.get("status") != "passed" or summary.get("completed_queries") != 80:
            raise RuntimeError(f"Both DiT and endpoint full80 fidelity gates must pass before benchmark: {path}")
    if arm == "ours_vqwam" and variant == "optimized":
        endpoint_path = root / "results" / "native_gate" / "sampler" / variant / arm / "full80.json"
        endpoint = read_json(endpoint_path)
        endpoint_thresholds = endpoint.get("thresholds", {})
        if (endpoint_thresholds.get("action_max_abs") != OURS_OPTIMIZED_ENDPOINT_THRESHOLD
                or endpoint_thresholds.get("future_max_abs") != OURS_OPTIMIZED_ENDPOINT_THRESHOLD):
            raise RuntimeError("Ours optimized performance requires full80 action/future max-abs gates at 1e-6")
        dit_path = root / "results" / "native_gate" / "dit_replay" / variant / arm / "full80.json"
        dit_summary = read_json(dit_path)
        dit_thresholds = dit_summary.get("thresholds", {})
        if dit_thresholds.get("dit_forward_max_abs") != OURS_OPTIMIZED_ENDPOINT_THRESHOLD:
            raise RuntimeError("Ours optimized performance requires full80 DiT output max-abs gate at 1e-6")

        expected_queries = {(case_id, seed) for case_id in range(3000, 3040) for seed in (0, 1)}

        def verify_query_metric(summary, metric, scope_name):
            rows = summary.get("queries")
            if not isinstance(rows, list) or len(rows) != 80:
                raise RuntimeError(f"Ours optimized {scope_name} full80 gate must contain 80 query metrics")
            observed = set()
            for row in rows:
                query = (row.get("case_id"), row.get("seed_index"))
                if query in observed or query not in expected_queries:
                    raise RuntimeError(f"Ours optimized {scope_name} gate query coverage is not the frozen 80 panel")
                observed.add(query)
                value = row.get("metrics", {}).get(metric, float("nan"))
                if (row.get("status") != "passed" or not math.isfinite(value)
                        or value > OURS_OPTIMIZED_ENDPOINT_THRESHOLD):
                    raise RuntimeError(
                        f"Ours optimized {scope_name} query {query} failed {metric} <= 1e-6")
            if observed != expected_queries:
                raise RuntimeError(f"Ours optimized {scope_name} gate omits frozen queries")

        verify_query_metric(endpoint, "action_max_abs", "endpoint action")
        verify_query_metric(endpoint, "future_max_abs", "endpoint future")
        verify_query_metric(dit_summary, "dit_forward_max_abs", "DiT output")


def assert_native_gate_binding(root: Path, arm: str, variant: str, binding: dict) -> None:
    if arm == 'bf16':
        return
    if _baseline_report_only_request(root, arm, variant):
        build = binding.get('extension_build', {})
        if (binding.get('native_int4_kernel_used') is not True
                or not build.get('pbs_job_id') or not build.get('source_commit')):
            raise RuntimeError('Report-only fidelity still requires a proven native extension binding')
        return
    expected = {key: value for key, value in binding.items() if key != 'dit_replay_dispatch'}
    for scope in ('sampler', 'dit_replay'):
        path = root / 'results/native_gate' / scope / variant / arm / 'full80.json'
        saved = read_json(path).get('backend_binding', {})
        saved = {key: value for key, value in saved.items() if key != 'dit_replay_dispatch'}
        if saved != expected:
            raise RuntimeError(f'Passed gate belongs to a different native backend: {path}')


def _run_sampler_benchmark(common, runtime, protocol, old_root, root, out, arm,
                           variant, torch, device, hardware, metadata, binding,
                           storage, deadline) -> dict:
    import benchmark as legacy_benchmark
    contexts = {(case, seed): _load_context(common, old_root, case, seed)
                for case, seed in performance_query_keys(protocol)}
    warmup_count = int(protocol["performance"]["warmup_queries"])
    warmup_rows = []
    for case_id, seed_index in performance_query_keys(protocol)[:warmup_count]:
        if time.monotonic() >= deadline:
            break
        warmup_rows.append({"query": [case_id, seed_index],
                            **legacy_benchmark.run_one(common, runtime,
                                contexts[(case_id, seed_index)], torch=torch, device=device)})
    expected = {}
    for block in range(PERFORMANCE_BLOCKS):
        for repetition in range(PERFORMANCE_REPETITIONS):
            for case_id, seed_index in performance_query_keys(protocol):
                identity = sample_identity(protocol=protocol["protocol"], hardware=hardware,
                    binding=binding, scope="sampler", variant=variant, arm=arm,
                    case_id=case_id, seed_index=seed_index, block=block,
                    repetition=repetition, warmup_count=warmup_count)
                path = sample_path(root, "sampler", variant, arm, case_id, seed_index, block, repetition)
                expected[path] = identity
    records = []
    for path, identity in expected.items():
        saved = reusable_performance_sample(path, identity)
        if saved is not None:
            records.append(saved)
    reused = len(records)
    if len(warmup_rows) == warmup_count:
        for path, identity in expected.items():
            if reusable_performance_sample(path, identity) is not None:
                continue
            if time.monotonic() >= deadline:
                break
            torch.cuda.reset_peak_memory_stats(device)
            measurement = legacy_benchmark.run_one(common, runtime,
                contexts[(identity["case_id"], identity["seed_index"])], torch=torch, device=device)
            peak = int(torch.cuda.max_memory_allocated(device))
            record = {**identity, **measurement, "peak_allocated_bytes": peak,
                      "peak_allocated_gb_decimal": peak / 1e9,
                      "memory_scope": "full common.endpoint; resident model including VAE/tokenizer plus endpoint activations and temporary tensors",
                      "packed_storage_bytes": storage["packed_storage_bytes"],
                      "packed_storage_gib_binary": storage["packed_storage_gib_binary"],
                      "storage_receipt": storage,
                      "warmup": {"query_count": warmup_count,
                                 "query_keys": [[a, b] for a, b in performance_query_keys(protocol)[:warmup_count]]},
                      "pbs_job_id": os.environ.get("PBS_JOBID"), "hostname": socket.gethostname()}
            if peak <= 0 or not math.isfinite(measurement["denoising_seconds"]):
                raise RuntimeError("Invalid endpoint latency or CUDA peak memory")
            _atomic_json(path, record)
            records.append(record)
    complete = len(records) == PERFORMANCE_SAMPLES
    summary = {"format": FORMAT, "protocol": protocol["protocol"],
        "status": "complete" if complete else "checkpointed", "complete": complete,
        "arm": arm, "variant": variant, "scope": "sampler", "hardware": hardware,
        "backend_binding": binding, "install_metadata": metadata,
        "performance_design": PERFORMANCE_DESIGN,
        "coverage": {"queries": PERFORMANCE_SAMPLES, "samples": len(records), "expected_samples": PERFORMANCE_SAMPLES,
                     "reused_samples": reused, "blocks": PERFORMANCE_BLOCKS,
                     "repetitions": PERFORMANCE_REPETITIONS},
        "warmup_queries": warmup_rows, "timing_scope": "synchronized legacy benchmark.run_one sampler.forward wall interval",
        "denoising_wall_seconds": summarize([row["denoising_seconds"] for row in records]),
        "complete_endpoint_query_seconds": summarize([row["complete_endpoint_query_seconds"] for row in records]),
        "peak_allocated_gb_decimal": summarize([row["peak_allocated_gb_decimal"] for row in records]),
        "sample_hardware_provenance": sample_hardware_provenance(records, hardware),
        "storage": storage, "training_updates": 0, "closed_loop_episodes": 0,
        "created_utc": datetime.now(timezone.utc).isoformat()}
    summary_path = performance_root(root) / "sampler" / variant / arm / "summary.json"
    summary["summary_bytes"] = _bounded_summary(summary_path, summary)
    return summary


def _run_dit_benchmark(runtime, protocol, root, out, arm, variant, torch,
                       device, hardware, metadata, binding, storage, deadline) -> dict:
    _read_capture_manifest(root, protocol, hardware)
    signatures = _signatures_by_nfe(root, protocol, torch)
    capture_template = _load_capture(root, *performance_query_keys(protocol)[0], torch)
    offload = _offload_non_dit(runtime, torch, device)
    gc.collect()
    torch.cuda.empty_cache()
    with _dit_replay_dispatch(runtime) as dit_replay_dispatch:
        bundles = _make_graph_bundle(runtime.model.net, capture_template["calls"], device, torch)
    dit_binding = {**binding, "dit_replay_dispatch": dit_replay_dispatch}
    warmup_count = int(protocol["performance"]["warmup_queries"])
    warmup_keys = performance_query_keys(protocol)[:warmup_count]
    warmup_done = 0
    for case_id, seed_index in warmup_keys:
        payload = _load_capture(root, case_id, seed_index, torch)
        _replay_five(bundles, payload["calls"], device, torch)
        warmup_done += 1
    expected = {}
    for block in range(PERFORMANCE_BLOCKS):
        for repetition in range(PERFORMANCE_REPETITIONS):
            for case_id, seed_index in performance_query_keys(protocol):
                identity = sample_identity(protocol=protocol["protocol"], hardware=hardware,
                    binding=dit_binding, scope="dit_replay", variant=variant, arm=arm,
                    case_id=case_id, seed_index=seed_index, block=block,
                    repetition=repetition, warmup_count=warmup_count)
                expected[sample_path(root, "dit_replay", variant, arm, case_id, seed_index,
                                     block, repetition)] = identity
    records = []
    for path, identity in expected.items():
        saved = reusable_performance_sample(path, identity)
        if saved is not None:
            records.append(saved)
    reused = len(records)
    if warmup_done == warmup_count:
        for path, identity in expected.items():
            if reusable_performance_sample(path, identity) is not None:
                continue
            if time.monotonic() >= deadline:
                break
            capture = _load_capture(root, identity["case_id"], identity["seed_index"], torch)
            seconds, peak = _replay_five(bundles, capture["calls"], device, torch)
            record = {**identity, "dit_replay_seconds": seconds,
                      "cuda_event_5forward_ms": seconds * 1000,
                      "peak_allocated_bytes": peak,
                      "peak_allocated_gb_decimal": peak / 1e9,
                      "memory_scope": "model.net resident parameters + five independent fixed CUDA graph input/output sets + shared-pool activations/temporaries; non-DiT modules offloaded",
                      "graph_pool_contract": DIT_GRAPH_POOL_CONTRACT,
                      "non_dit_offload_cuda_tensor_bytes": offload["offloaded_cuda_tensor_bytes"],
                      "non_dit_offload_module_count": sum(
                          row["status"] == "offloaded_to_cpu" for row in offload["modules"]),
                      "packed_storage_bytes": storage["packed_storage_bytes"],
                      "packed_storage_gib_binary": storage["packed_storage_gib_binary"],
                      "storage_receipt": storage,
                      "warmup": {"query_count": warmup_done,
                                 "query_keys": [[a, b] for a, b in warmup_keys]},
                      "pbs_job_id": os.environ.get("PBS_JOBID"), "hostname": socket.gethostname()}
            if not math.isfinite(seconds) or seconds <= 0 or peak <= 0:
                raise RuntimeError("Invalid DiT graph replay time or CUDA peak allocation")
            _atomic_json(path, record)
            records.append(record)
    complete = len(records) == PERFORMANCE_SAMPLES
    summary = {"format": FORMAT, "protocol": protocol["protocol"],
        "status": "complete" if complete else "checkpointed", "complete": complete,
        "arm": arm, "variant": variant, "scope": "dit_replay", "hardware": hardware,
        "backend_binding": dit_binding, "install_metadata": metadata,
        "performance_design": PERFORMANCE_DESIGN,
        "coverage": {"queries": PERFORMANCE_SAMPLES, "samples": len(records), "expected_samples": PERFORMANCE_SAMPLES,
                     "reused_samples": reused, "blocks": PERFORMANCE_BLOCKS,
                     "repetitions": PERFORMANCE_REPETITIONS},
        "warmup_queries": warmup_keys, "warmup_forwards_per_query": 5,
        "graph_capture_count": 5, "graph_backend": bundles[0]["graph_backend"],
        "graph_pool_contract": DIT_GRAPH_POOL_CONTRACT,
        "dit_replay_dispatch": dit_replay_dispatch,
        "sample_hardware_provenance": sample_hardware_provenance(records, hardware),
        "forward_input_signatures": signatures,
        "timing_scope": "CUDA events around exactly five graph replays of frozen model.net.forward inputs; condition/sampler preparation excluded",
        "dit_replay_seconds": summarize([row["dit_replay_seconds"] for row in records]),
        "peak_allocated_gb_decimal": summarize([row["peak_allocated_gb_decimal"] for row in records]),
        "non_dit_offload": offload, "storage": storage,
        "training_updates": 0, "closed_loop_episodes": 0,
        "created_utc": datetime.now(timezone.utc).isoformat()}
    summary_path = performance_root(root) / "dit_replay" / variant / arm / "summary.json"
    summary["summary_bytes"] = _bounded_summary(summary_path, summary)
    return summary


def _prepare_code_imports(old_root: Path, old_code_root: Path | None) -> Path:
    code_root = (old_code_root or old_root).resolve()
    if not (code_root / "common.py").is_file() or not (code_root / "runtime.py").is_file():
        raise FileNotFoundError(f"Read-only legacy common.py/runtime.py not found: {code_root}")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(code_root))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return code_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("capture", "benchmark", "gate"))
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--variant", choices=("current", "optimized", 'native_fused', 'native_grouped'), required=True)
    parser.add_argument("--scope", choices=("sampler", "dit_replay"), required=True)
    parser.add_argument("--require-gpu-uuid")
    parser.add_argument("--old-code-root", type=Path)
    parser.add_argument("--soft-deadline-seconds", type=int, default=6600)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate_variant(args.arm, args.variant)
    if args.soft_deadline_seconds <= 0:
        raise ValueError("--soft-deadline-seconds must be positive")
    old_root, root, out = validate_paths(args.old_root, args.root, args.out)
    protocol = read_json(old_root / "protocol.json")
    validate_protocol(protocol, args.arm)
    if args.mode == "capture" and (args.arm != "bf16" or args.variant != "current"
                                    or args.scope != "dit_replay"):
        raise ValueError("Capture mode requires --arm bf16 --variant current --scope dit_replay")
    code_root = _prepare_code_imports(old_root, args.old_code_root)
    if args.mode != "capture" and args.arm in ("spinquant_w4a4", "quarot_w4a4", "viditq_w4a4"):
        _prepare_rotation_backend_import(root)
    import common
    _allocation_guard(common)
    root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    from startup_trace import startup_trace
    with startup_trace(out):
        torch = common.torch_module()
        runtime = common.prepare_runtime(protocol, old_root, out)
    device = next(runtime.model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"Expected CUDA model, found {device}")
    hardware = _gpu_receipt(torch, device, args.require_gpu_uuid)
    hardware_protocol_path = root / "native_protocol.json"
    hardware_protocol = (read_json(hardware_protocol_path)
                         if hardware_protocol_path.is_file() else protocol)
    validate_hardware_compatibility(hardware_protocol, hardware)
    deadline = _deadline(args)
    receipt_binding = {}
    restore_hook = None
    try:
        if args.mode == "capture":
            if args.arm != "bf16" or args.scope != "dit_replay":
                raise ValueError("Capture mode is BF16-only and uses --scope dit_replay")
            import runtime as legacy_runtime
            metadata = legacy_runtime.install_arm(runtime, protocol, old_root, "bf16", optimized=False)
            summary = _capture_panel(common, runtime, protocol, old_root, root, out,
                                     torch, device, hardware, deadline)
            print(json.dumps({"mode": "capture", **summary, "hardware": hardware,
                              "capture_arm_metadata": _binding(metadata, "bf16", args.variant)},
                             ensure_ascii=False, allow_nan=False), flush=True)
            return 0 if summary["complete"] else 2

        if args.mode == "gate":
            if args.arm == "bf16":
                raise ValueError("BF16 is the reference arm and has no native fidelity gate")
            if args.scope == "dit_replay":
                _read_capture_manifest(root, protocol, hardware)
            # Gate phases install decoded references first and native paths second.
            summary = _gate_panel(common, runtime, protocol, old_root, root, out,
                args.arm, args.variant, args.scope, torch, device, hardware,
                receipt_binding, deadline)
            print(json.dumps({"mode": "gate", **summary, "hardware": hardware,
                              "native_binding": summary["backend_binding"]},
                             ensure_ascii=False, allow_nan=False), flush=True)
            return 0 if summary["status"] == "passed" else 2

        if args.arm != "bf16":
            _require_gate(root, args.arm, args.variant, args.scope)
        metadata, restore_hook = _install_native(runtime, protocol, old_root, out,
                                                 args.arm, args.variant, receipt_binding, native_root=root)
        if not isinstance(metadata, dict):
            raise TypeError("Native installer must return a metadata mapping")
        if args.arm in ("spinquant_w4a4", "quarot_w4a4", "svdquant_w4a4", "viditq_w4a4"):
            if metadata.get("native_int4_kernel_used") is not True:
                raise RuntimeError("Method baseline lacks a positive native integer-GEMM receipt")
        binding = _binding(metadata, args.arm, args.variant, root)
        assert_native_gate_binding(root, args.arm, args.variant, binding)
        storage = _packed_storage_receipt(runtime, metadata, args.arm, torch, device)
        if args.scope == "dit_replay":
            _read_capture_manifest(root, protocol, hardware)
            summary = _run_dit_benchmark(runtime, protocol, root, out, args.arm,
                args.variant, torch, device, hardware, metadata, binding, storage, deadline)
        else:
            summary = _run_sampler_benchmark(common, runtime, protocol, old_root, root,
                out, args.arm, args.variant, torch, device, hardware, metadata,
                binding, storage, deadline)
        if callable(restore_hook):
            restore_hook()
            restore_hook = None
        print(json.dumps({"mode": "benchmark", "status": summary["status"],
                          "arm": args.arm, "variant": args.variant, "scope": args.scope,
                          "summary_path": str(performance_root(root) /
                              args.scope / args.variant / args.arm / "summary.json"),
                          "samples": summary["coverage"]["samples"],
                          "expected_samples": PERFORMANCE_SAMPLES, "hardware": hardware},
                         ensure_ascii=False, allow_nan=False), flush=True)
        return 0 if summary["complete"] else 2
    finally:
        if callable(restore_hook):
            restore_hook()
        if args.arm == "ours_vqwam":
            restore = getattr(runtime, "restore_ours_vqwam", None)
            if callable(restore):
                restore()
            optimized_restore = getattr(runtime, "restore_optimized_engine", None)
            if callable(optimized_restore):
                optimized_restore()
        if callable(getattr(runtime, "restore_native_rotation", None)):
            runtime.restore_native_rotation()
        if callable(getattr(runtime, "restore_native_diffusion", None)):
            runtime.restore_native_diffusion()
        _ = code_root


if __name__ == "__main__":
    raise SystemExit(main())
