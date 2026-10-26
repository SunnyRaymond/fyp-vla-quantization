"""Fast-WAM AQLM banks on top of the frozen Smooth/R0 deployment.

This adapter deliberately keeps the source model weights in a CPU-only master
copy.  The AQLM ``QuantizedWeight`` objects are the only trainable state; the
code indices are uint8 and never receive Adam updates.  The forward path is
the existing signed-Hadamard/Smooth/R0 A4 path followed by a dense BF16
decoded AQLM weight.  No native low-bit GEMM is claimed here.

The implementation is pinned to the official AQLM repository at
``e79a896ed6656fe4ed06193d42d004e7d0bbdbb2``.  The official AQLM source is
loaded lazily on a compute node; this file does not vendor or install it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


AQLM_COMMIT = "e79a896ed6656fe4ed06193d42d004e7d0bbdbb2"
TARGET_COUNT = 614
IN_GROUP_SIZE = 8
NUM_CODEBOOKS = 4
NBITS_PER_CODEBOOK = 8
CODEBOOK_SIZE = 1 << NBITS_PER_CODEBOOK
BACKWARD_MAX_VECTORS = 1 << 20


def _aqlm_decoder():
    """Return the decoder from the pinned official AQLM source."""
    try:
        from src.utils import _dequantize_weight
    except Exception as exc:  # pragma: no cover - exercised on the compute image
        raise RuntimeError(
            "AQLM source is unavailable; install/use the pinned official "
            f"AQLM checkout at {AQLM_COMMIT} on the compute node"
        ) from exc
    return _dequantize_weight


def _source_rotation_module():
    """Import the repository-local frozen R0/Smooth implementation lazily."""
    try:
        import rotation_bank
    except Exception as exc:  # pragma: no cover - exercised by the PBS runtime
        raise RuntimeError(
            "rotation_bank.py must be on the runtime import path; this adapter "
            "does not reimplement the source Smooth/R0 transform"
        ) from exc
    required = ("RotationBanks", "A4", "stream_for", "transformed")
    missing = [name for name in required if not hasattr(rotation_bank, name)]
    if missing:
        raise RuntimeError("rotation_bank.py is missing required symbols: " + ", ".join(missing))
    return rotation_bank


def _protocol_id(protocol: Any) -> Any:
    if isinstance(protocol, Mapping):
        return protocol.get("protocol", protocol.get("id", protocol))
    return protocol


def _bank_protocol_id(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("protocol", value.get("id", value))
    return value


def _as_shape(value: Any, field: str) -> Tuple[int, ...]:
    try:
        shape = tuple(int(v) for v in value)
    except Exception as exc:
        raise RuntimeError(f"Invalid {field} in AQLM bank metadata") from exc
    if not shape or any(v <= 0 for v in shape):
        raise RuntimeError(f"Invalid {field} in AQLM bank metadata: {shape}")
    return shape


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").clone()


def _bf16_identity_ste(value: torch.Tensor) -> torch.Tensor:
    """Round storage to BF16 while keeping an FP32 identity backward."""
    return value + (value.to(dtype=torch.bfloat16).float() - value).detach()


def _set_parameter(module: nn.Module, name: str, value: torch.Tensor, requires_grad: bool) -> None:
    """Replace a Parameter while preserving the official module attribute name."""
    setattr(module, name, nn.Parameter(value, requires_grad=requires_grad))


def _normalize_quantized_weight(qweight: nn.Module, device: torch.device) -> nn.Module:
    """Move a loaded official QuantizedWeight to the runtime device.

    AQLM's official object uses ``codes`` as a non-gradient integer Parameter.
    Saved banks intentionally cast it to uint8.  ``get_codes`` later casts it
    to int32 before embedding/indexing, exactly as the official implementation
    expects.
    """
    if not hasattr(qweight, "codebooks") or not hasattr(qweight, "in_features"):
        raise RuntimeError("AQLM bank does not contain an official QuantizedWeight")

    qweight.to(device=device)
    if getattr(qweight, "codes", None) is None:
        storage = getattr(qweight, "codes_storage", None)
        if storage is not None and hasattr(qweight, "unwrap_codes_"):
            qweight.unwrap_codes_()
        if getattr(qweight, "codes", None) is None:
            raise RuntimeError("FSDP-wrapped AQLM codes are unsupported in a common bank")

    codes = qweight.codes.detach()
    if codes.numel() == 0 or int(codes.max().item()) >= CODEBOOK_SIZE or int(codes.min().item()) < 0:
        raise RuntimeError("AQLM code indices are outside the configured uint8/K256 range")
    _set_parameter(qweight, "codes", codes.to(device=device, dtype=torch.uint8), requires_grad=False)

    books = qweight.codebooks.detach().to(device=device, dtype=torch.float32)
    _set_parameter(qweight, "codebooks", books, requires_grad=True)

    # The experiment uses nominal M4/K256 with row-wise (scale_nbits=0)
    # scales.  Quantized scale indices would need a separate backward contract.
    if getattr(qweight, "scales", None) is None:
        raise RuntimeError(
            "This adapter requires official AQLM scale_nbits=0 with trainable "
            "qweight.scales; quantized scale clusters are intentionally rejected"
        )
    scales = qweight.scales.detach().to(device=device, dtype=torch.float32)
    _set_parameter(qweight, "scales", scales, requires_grad=True)
    return qweight


def _make_qweight_from_state(
    state: Mapping[str, torch.Tensor],
    reference_weight: torch.Tensor,
    metadata: Mapping[str, Any],
    device: torch.device,
) -> nn.Module:
    """Rebuild a minimal official QuantizedWeight without rerunning K-means.

    Object pickles are preferred.  This state-dict fallback is deliberately
    strict: it only accepts the M4/K256/group8, row-scale representation used by
    this experiment and fills the same attributes consumed by official
    ``QuantizedWeight.forward``/``get_codes``.
    """
    try:
        from src.aq import QuantizedWeight
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Pinned official AQLM src.aq is unavailable") from exc

    books = state.get("codebooks")
    codes = state.get("codes")
    scales = state.get("scales")
    if books is None or codes is None or scales is None:
        raise RuntimeError("AQLM state_dict must contain codebooks, codes, and scales")

    original_shape = tuple(int(v) for v in reference_weight.shape)
    qweight = QuantizedWeight.__new__(QuantizedWeight)
    nn.Module.__init__(qweight)
    qweight.out_features, qweight.in_features = original_shape
    qweight.out_group_size = int(metadata.get("out_group_size", 1))
    qweight.in_group_size = int(metadata.get("in_group_size", IN_GROUP_SIZE))
    qweight.num_codebooks = int(metadata.get("num_codebooks", NUM_CODEBOOKS))
    qweight.nbits_per_codebook = int(metadata.get("nbits_per_codebook", NBITS_PER_CODEBOOK))
    qweight.codebook_size = 1 << qweight.nbits_per_codebook
    qweight.codebook_value_nbits = int(metadata.get("codebook_value_nbits", 16))
    qweight.codebook_value_num_groups = int(metadata.get("codebook_value_num_groups", 1))
    qweight.codebook_value_clusters = None
    qweight.scale_nbits = 0
    qweight.scales_are_lossless = True
    qweight.straight_through_gradient = None
    qweight.scales_clusters = None
    qweight.scales_indices = None
    qweight.codes_storage = None
    _set_parameter(qweight, "codebooks", books.to(device=device, dtype=torch.float32), requires_grad=True)
    _set_parameter(qweight, "codes", codes.to(device=device, dtype=torch.uint8), requires_grad=False)
    _set_parameter(qweight, "scales", scales.to(device=device, dtype=torch.float32), requires_grad=True)
    return _normalize_quantized_weight(qweight, device)


class DecodedWeight(torch.autograd.Function):
    """BF16 cached AQLM decode with a chunked official-decoder backward.

    Forward intentionally returns a detached BF16 dense tensor.  Backward
    reconstructs the same official ``_dequantize_weight`` expression from
    BF16-rounded FP32 masters, in bounded output-row chunks, so no per-query
    decoded-weight graph is retained.
    """

    @staticmethod
    def forward(ctx, books, scales, indices, dense):
        ctx.save_for_backward(books, scales, indices)
        ctx.shape = tuple(dense.shape)
        return dense.detach().to(dtype=torch.bfloat16).view_as(dense)

    @staticmethod
    def backward(ctx, grad):
        books, scales, indices = ctx.saved_tensors
        decoder = _aqlm_decoder()
        out_width, _ = ctx.shape
        indices = indices.to(dtype=torch.long)
        num_in_groups = int(indices.shape[1])
        chunk_rows = max(1, BACKWARD_MAX_VECTORS // max(1, num_in_groups))

        grad_books = torch.zeros_like(books) if books.requires_grad else None
        grad_scales = torch.zeros_like(scales) if scales.requires_grad else None
        grad_float = grad.float()

        # A fresh graph is built for each chunk.  This is the key memory bound:
        # only one output-row chunk, not 614 full decoded matrices, is live.
        for start in range(0, out_width, chunk_rows):
            end = min(start + chunk_rows, out_width)
            with torch.enable_grad():
                stored_books = books.detach().to(dtype=torch.bfloat16).float().requires_grad_(True)
                stored_scales = scales.detach().to(dtype=torch.bfloat16).float().requires_grad_(True)
                code_chunk = indices[start:end]
                scale_chunk = stored_scales[start:end]
                decoded = decoder(code_chunk, stored_books, scale_chunk)
                outputs = []
                inputs = []
                if books.requires_grad:
                    outputs.append(stored_books)
                    inputs.append(stored_books)
                if scales.requires_grad:
                    outputs.append(stored_scales)
                    inputs.append(stored_scales)
                if not outputs:
                    continue
                grads = torch.autograd.grad(
                    decoded,
                    inputs,
                    grad_outputs=grad_float[start:end],
                    retain_graph=False,
                    allow_unused=True,
                )
                pos = 0
                if books.requires_grad:
                    value = grads[pos]
                    if value is not None:
                        grad_books.add_(value.to(dtype=grad_books.dtype))
                    pos += 1
                if scales.requires_grad:
                    value = grads[pos]
                    if value is not None:
                        grad_scales[start:end].add_(value[start:end].to(dtype=grad_scales.dtype))

        return grad_books, grad_scales, None, None


def _decode_stored(qweight: nn.Module) -> torch.Tensor:
    """Decode with the exact BF16 storage rounding used by forward."""
    decoder = _aqlm_decoder()
    codes = qweight.codes.detach().to(dtype=torch.long)
    books = qweight.codebooks.detach().to(dtype=torch.bfloat16).float()
    scales = qweight.scales.detach().to(dtype=torch.bfloat16).float()
    return decoder(codes, books, scales).to(dtype=torch.bfloat16)


class AQLMBanks:
    """Install shared Smooth/R0 transforms and common M4/K256 AQLM banks.

    ``mode='bf16'`` uses the untouched original BF16 module weights.  In
    ``mode='aqlm'`` the existing Smooth/R0 transformed A4 operand is paired
    with a dense BF16 AQLM decode.  The implementation is intentionally
    decoded-BF16; it does not claim a native AQLM GEMM.  ``bank_directory``
    is the new AQLM bank directory; the immutable C8 source directory is read
    from ``protocol['calibration']['bank_directory']`` when present.
    """

    def __init__(self, runtime, protocol, bank_directory, load_quantized: bool = True):
        self.runtime = runtime
        self.protocol = protocol
        self.bank_directory = Path(bank_directory)
        source_calibration = protocol.get("calibration", {}) if isinstance(protocol, Mapping) else {}
        self.source_bank_directory = Path(
            source_calibration.get("bank_directory", bank_directory)
        )
        self.modules = runtime.q.modules
        if len(self.modules) != TARGET_COUNT:
            raise RuntimeError(f"Expected exactly {TARGET_COUNT} target Linears, found {len(self.modules)}")

        self.mode = "aqlm" if load_quantized else "bf16"
        self.seen = set()
        self.quantized_weights: Dict[str, nn.Module] = {}
        self.books: Dict[str, nn.Parameter] = {}
        self.scales: Dict[str, nn.Parameter] = {}
        self.decoded: Dict[str, torch.Tensor] = {}
        self._indices_identity: Dict[str, Tuple[int, int]] = {}
        self._original_gpu: Dict[Tuple[str, str, Optional[int], torch.dtype], torch.Tensor] = {}
        self._reference_cache: Dict[str, torch.Tensor] = {}
        self.cache_original_gpu = True
        self.forwards = {name: mod.forward for name, mod in self.modules.items()}

        # Save the original BF16 weights before RotationBanks touches module
        # forwards.  These masters remain CPU-only and are never overwritten.
        self.masters = {name: _cpu_clone(mod.weight) for name, mod in self.modules.items()}
        self._rotation_module = _source_rotation_module()
        self._source_banks = self._make_source_banks()
        self.rotations = self._source_banks.rotations
        self.transforms = self._source_banks.transforms

        self._restore_frozen_r0()
        if load_quantized:
            self._load_quantized_banks()
        self._install_forwards()
        if load_quantized:
            self.refresh()

    def _make_source_banks(self):
        """Load the pinned source C8 transforms through the historical loader."""
        return self._rotation_module.RotationBanks(self.runtime, self.source_bank_directory)

    def _restore_frozen_r0(self) -> None:
        source = self.protocol.get("source_rotation", {}) if isinstance(self.protocol, Mapping) else {}
        deployment = source.get("deployment") or source.get("source_rotation_deployment")
        if not deployment:
            raise RuntimeError("protocol.source_rotation.deployment is required to restore frozen R0")
        try:
            saved = torch.load(deployment, map_location="cpu", weights_only=True)
        except TypeError:  # older torch on the compute image
            saved = torch.load(deployment, map_location="cpu")
        expected_protocol = source.get("protocol")
        saved_protocol = saved.get("protocol", {}) if isinstance(saved, Mapping) else {}
        if (not isinstance(saved_protocol, Mapping)
                or saved_protocol.get("protocol") != expected_protocol
                or saved.get("lane") != source.get("lane")
                or saved.get("arm") != source.get("arm")
                or saved.get("updates") != 8
                or saved.get("target_linears") != TARGET_COUNT):
            raise RuntimeError("Frozen R0 deployment binding differs")
        rotations = saved.get("rotations") if isinstance(saved, Mapping) else None
        if not isinstance(rotations, Mapping):
            raise RuntimeError("Frozen R0 deployment does not contain rotations")
        self.rotations.restore(rotations)
        self.rotations.freeze()
        orthogonality = self.rotations.orthogonality()
        if not orthogonality or max(orthogonality.values()) > 5e-5:
            raise RuntimeError(f"Frozen R0 orthogonality differs: {orthogonality}")

    def _reference_for_shape(self, name: str, padded_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
        mod = self.modules[name]
        master = self.masters[name].to(device=mod.weight.device)
        transformed = self._rotation_module.transformed(
            master,
            self.transforms[name],
            self.rotations,
            self._rotation_module.stream_for(name),
            weight=True,
        )
        target_shape = padded_shape or tuple(transformed.shape)
        if tuple(transformed.shape) != tuple(self.masters[name].shape):
            raise RuntimeError(f"Unexpected transformed shape for {name}: {tuple(transformed.shape)}")
        if target_shape[0] != transformed.shape[0] or target_shape[1] < transformed.shape[1]:
            raise RuntimeError(f"Invalid padded shape for {name}: {target_shape}")
        if target_shape[1] > transformed.shape[1]:
            transformed = F.pad(transformed, (0, target_shape[1] - transformed.shape[1]))
        return transformed.detach().to(dtype=torch.float32)

    def _original_weight(self, name: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get an original BF16 master on-device without touching mod.weight."""
        key = (name, device.type, device.index, dtype)
        if self.cache_original_gpu and key in self._original_gpu:
            return self._original_gpu[key]
        value = self.masters[name].to(device=device, dtype=dtype, non_blocking=True)
        if self.cache_original_gpu:
            self._original_gpu[key] = value
        return value

    def reference_weight(self, name: str, cache_bf16: bool = False) -> torch.Tensor:
        """Return transformed/padded original W, optionally cached as BF16 on GPU."""
        if name not in self.modules:
            raise KeyError(name)
        qweight = self.quantized_weights.get(name)
        if qweight is not None:
            padded_shape = tuple(qweight.shape)
        else:
            out_features, in_features = self.masters[name].shape
            padded_shape = (int(out_features), (int(in_features) + IN_GROUP_SIZE - 1) // IN_GROUP_SIZE * IN_GROUP_SIZE)
        if not cache_bf16:
            return self._reference_for_shape(name, padded_shape)
        cached = self._reference_cache.get(name)
        if cached is None:
            cached = self._reference_for_shape(name, padded_shape).to(dtype=torch.bfloat16)
            self._reference_cache[name] = cached
        return cached

    def release_reference_cache(self, clear_original_gpu: bool = True) -> None:
        """Release cached reference weights and, optionally, original GPU masters."""
        self._reference_cache.clear()
        if clear_original_gpu:
            self._original_gpu.clear()

    def _load_quantized_banks(self) -> None:
        expected_protocol = _protocol_id(self.protocol)
        for index, (name, mod) in enumerate(self.modules.items()):
            path = self.bank_directory / f"{index:04d}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing AQLM common bank: {path}")
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:  # older torch on the compute image
                payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, Mapping):
                raise RuntimeError(f"Invalid AQLM bank payload at {path}")
            if _bank_protocol_id(payload.get("protocol")) != expected_protocol:
                raise RuntimeError(f"AQLM protocol binding differs at {name}")
            if (payload.get("module") != name
                    or int(payload.get("module_index", -1)) != index
                    or payload.get("official_commit") != AQLM_COMMIT):
                raise RuntimeError(f"AQLM module binding differs at {name}")

            original_shape = _as_shape(payload.get("original_shape"), "original_shape")
            padded_shape = _as_shape(payload.get("padded_shape"), "padded_shape")
            if original_shape != tuple(mod.weight.shape) or padded_shape[0] != original_shape[0]:
                raise RuntimeError(f"AQLM shape metadata differs at {name}")
            if padded_shape[1] < original_shape[1] or padded_shape[1] % IN_GROUP_SIZE:
                raise RuntimeError(f"AQLM padded input shape is invalid at {name}: {padded_shape}")

            raw_qweight = payload.get("quantized_weight")
            metadata = payload.get("aqlm", payload.get("quantization", {}))
            if isinstance(raw_qweight, nn.Module):
                qweight = _normalize_quantized_weight(raw_qweight, mod.weight.device)
            elif isinstance(raw_qweight, Mapping):
                reference = self._reference_for_shape(name, padded_shape)
                qweight = _make_qweight_from_state(raw_qweight, reference, metadata, mod.weight.device)
            else:
                raise RuntimeError(f"AQLM bank does not contain quantized_weight at {path}")

            if tuple(qweight.shape) != padded_shape:
                raise RuntimeError(
                    f"AQLM quantized shape differs at {name}: {tuple(qweight.shape)} != {padded_shape}"
                )
            if (qweight.in_group_size != IN_GROUP_SIZE or qweight.num_codebooks != NUM_CODEBOOKS
                    or qweight.nbits_per_codebook != NBITS_PER_CODEBOOK
                    or qweight.out_group_size != 1 or qweight.codebook_value_nbits != 16
                    or qweight.scale_nbits != 0):
                raise RuntimeError(f"AQLM configuration differs at {name}; expected g8/M4/K256/out_group1")
            self.quantized_weights[name] = qweight
            self.books[name] = qweight.codebooks
            self.scales[name] = qweight.scales
            self._record_indices_identity(name)

    def _record_indices_identity(self, name: str) -> None:
        codes = self.quantized_weights[name].codes
        self._indices_identity[name] = (int(codes.data_ptr()), int(codes._version))

    def _indices_are_frozen(self) -> bool:
        if len(self._indices_identity) != len(self.quantized_weights):
            return False
        for name, qweight in self.quantized_weights.items():
            codes = qweight.codes
            identity = self._indices_identity.get(name)
            if identity is None or codes.requires_grad:
                return False
            if int(codes.data_ptr()) != identity[0] or int(codes._version) != identity[1]:
                return False
        return bool(self.quantized_weights)

    def _assert_indices_frozen(self) -> None:
        if not self._indices_are_frozen():
            raise RuntimeError("AQLM code indices changed or became trainable")

    def _padded_operand(self, x: torch.Tensor, name: str) -> torch.Tensor:
        # The public operand deliberately keeps the original input width.  The
        # official fit uses the padded reference weight; runtime/control hooks
        # slice decoded/reference weights back to this original width.
        original_width = int(self.masters[name].shape[1])
        if x.shape[-1] != original_width:
            raise RuntimeError(f"Input width differs from source Linear at {name}: {x.shape[-1]} != {original_width}")
        return x

    def operand(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if name not in self.modules:
            raise KeyError(name)
        value = self._rotation_module.A4.apply(
            self._rotation_module.transformed(
                x,
                self.transforms[name],
                self.rotations,
                self._rotation_module.stream_for(name),
            )
        )
        return self._padded_operand(value, name)

    def weight(self, name: str) -> torch.Tensor:
        """Return a differentiable decoded BF16 weight at original input width."""
        if name not in self.quantized_weights:
            raise RuntimeError(f"AQLM weight is not loaded for {name}")
        qweight = self.quantized_weights[name]
        dense = self.decoded[name]
        decoded = DecodedWeight.apply(qweight.codebooks, qweight.scales, qweight.codes, dense)
        original_width = int(self.masters[name].shape[1])
        return decoded[..., :original_width]

    def _install_forwards(self) -> None:
        for name, mod in self.modules.items():
            original = self.forwards[name]

            def forward(x, *args, _name=name, _mod=mod, **kwargs):
                del args, kwargs  # target Fast-WAM Linears do not consume extras
                self.seen.add(_name)
                if self.mode == "bf16":
                    if self.quantized_weights:
                        # Never depend on mod.weight after a refresh: masters
                        # are retained on CPU as the immutable source of truth.
                        weight = self._original_weight(_name, x.device, torch.bfloat16)
                    else:
                        # Preparation never refreshes decoded banks, so the
                        # untouched module weight is already the BF16 master.
                        weight = _mod.weight
                    if x.dtype != weight.dtype:
                        weight = weight.to(dtype=x.dtype)
                    return F.linear(x, weight, _mod.bias)
                if self.mode != "aqlm":
                    raise RuntimeError(f"Unknown AQLM bank mode: {self.mode}")
                qweight = self.quantized_weights.get(_name)
                if qweight is None:
                    raise RuntimeError(f"AQLM weight is not loaded for {_name}")
                qx = self.operand(x, _name)
                weight = self.weight(_name)
                return F.linear(qx, weight, _mod.bias)

            # Keep a reference for diagnostics and restore it on close.  The
            # source RotationBanks object also retained the same original.
            forward._aqlm_original = original
            mod.forward = forward

    def set_mode(self, mode: str) -> None:
        if mode not in ("bf16", "aqlm"):
            raise ValueError("AQLM mode must be 'bf16' or 'aqlm'")
        if mode == "aqlm" and len(self.quantized_weights) != TARGET_COUNT:
            raise RuntimeError("Cannot enter AQLM mode without all 614 quantized banks")
        self.mode = mode

    @torch.no_grad()
    def refresh(self) -> None:
        if len(self.quantized_weights) != TARGET_COUNT:
            raise RuntimeError("Cannot refresh before all 614 AQLM banks are loaded")
        self._assert_indices_frozen()
        for name, qweight in self.quantized_weights.items():
            dense = _decode_stored(qweight)
            if tuple(dense.shape) != tuple(qweight.shape) or not bool(torch.isfinite(dense).all()):
                raise RuntimeError(f"Nonfinite or malformed AQLM decode at {name}")
            self.decoded[name] = dense
            original_width = int(self.masters[name].shape[1])
            self.modules[name].weight.data = dense[:, :original_width]

    @torch.no_grad()
    def bind_dense(self) -> None:
        """Bind module weights to the current decoded dense views without decoding."""
        if len(self.decoded) != TARGET_COUNT:
            raise RuntimeError("Cannot bind dense weights before all 614 AQLM banks are decoded")
        for name, dense in self.decoded.items():
            original_width = int(self.masters[name].shape[1])
            if (dense.ndim != 2 or dense.shape[0] != self.modules[name].weight.shape[0]
                    or dense.shape[1] < original_width):
                raise RuntimeError(f"Malformed decoded dense weight at {name}")
            self.modules[name].weight.data = dense[:, :original_width]

    def snapshot(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            name: {
                "books": _cpu_clone(self.books[name]),
                "scales": _cpu_clone(self.scales[name]),
            }
            for name in self.modules
        }

    @torch.no_grad()
    def restore(self, state: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        if set(state) != set(self.modules):
            raise RuntimeError("AQLM snapshot target coverage differs")
        for name in self.modules:
            value = state[name]
            if "books" not in value or "scales" not in value:
                raise RuntimeError(f"AQLM snapshot is missing books/scales at {name}")
            self.books[name].copy_(value["books"].to(device=self.books[name].device, dtype=torch.float32))
            self.scales[name].copy_(value["scales"].to(device=self.scales[name].device, dtype=torch.float32))
        self.refresh()

    def parameters(self) -> Sequence[nn.Parameter]:
        values = []
        for name in self.modules:
            if name in self.books:
                values.extend((self.books[name], self.scales[name]))
        return values

    def gradient_report(self) -> Dict[str, Any]:
        connected, finite, nonzero = [], [], []
        for name in self.modules:
            if name not in self.books:
                continue
            grads = [self.books[name].grad, self.scales[name].grad]
            if all(grad is not None for grad in grads):
                connected.append(name)
                if all(bool(torch.isfinite(grad).all()) for grad in grads):
                    finite.append(name)
                if any(bool(grad.abs().sum() > 0) for grad in grads):
                    nonzero.append(name)
        return {
            "targets": len(self.modules),
            "trainable_pairs": len(self.parameters()) // 2,
            "connected": len(connected),
            "finite": len(finite),
            "nonzero": len(nonzero),
            "disconnected_modules": sorted(set(self.modules) - set(connected)),
            "zero_gradient_modules": sorted(set(connected) - set(nonzero)),
            "indices_frozen": self._indices_are_frozen(),
            "aqlm_commit": AQLM_COMMIT,
        }

    def close(self) -> None:
        """Restore original module forwards/weights and release adapter references."""
        for name, mod in self.modules.items():
            mod.forward = self.forwards[name]
            original = self._original_weight(name, mod.weight.device, self.masters[name].dtype)
            mod.weight.data = original
        # RotationBanks.close restores the exact pre-install forwards too.
        if self._source_banks is not None:
            self._source_banks.close()
        self.decoded.clear()
        self.release_reference_cache(clear_original_gpu=True)
        self.mode = "bf16"


def selfcheck(device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Toy gradient parity check against the pinned official decoder.

    This function is intentionally not invoked during import or local syntax
    checks.  Run it only inside the approved compute allocation with AQLM
    available.  It checks both BF16 stored decode parity and books/scales
    gradients through the same official `_dequantize_weight` operation.
    """
    decoder = _aqlm_decoder()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(17)
    books = torch.randn((NUM_CODEBOOKS, CODEBOOK_SIZE, 1, IN_GROUP_SIZE), device=device,
                        generator=generator, dtype=torch.float32, requires_grad=True)
    scales = torch.rand((3, 2, 1, 1), device=device, generator=generator,
                        dtype=torch.float32).clamp_min(0.1).requires_grad_(True)
    indices = torch.randint(0, CODEBOOK_SIZE, (3, 2, NUM_CODEBOOKS), device=device,
                             generator=generator, dtype=torch.uint8)
    with torch.no_grad():
        reference = decoder(indices.long(), books.to(torch.bfloat16).float(),
                            scales.to(torch.bfloat16).float()).to(torch.bfloat16)
    actual = DecodedWeight.apply(books, scales, indices, reference)
    loss = actual.float().square().sum()
    loss.backward()
    actual_books_grad = books.grad.detach().clone()
    actual_scales_grad = scales.grad.detach().clone()

    expected_books = books.detach().clone().requires_grad_(True)
    expected_scales = scales.detach().clone().requires_grad_(True)
    expected_stored_books = _bf16_identity_ste(expected_books)
    expected_stored_scales = _bf16_identity_ste(expected_scales)
    expected = decoder(indices.long(), expected_stored_books, expected_stored_scales).to(torch.bfloat16)
    expected.float().square().sum().backward()
    if not torch.equal(actual.detach(), reference.detach()):
        raise AssertionError("DecodedWeight BF16 forward differs from official decoder")
    if not torch.allclose(actual_books_grad, expected_books.grad, atol=1e-6, rtol=1e-5):
        raise AssertionError("DecodedWeight books gradient differs from official decoder")
    if not torch.allclose(actual_scales_grad, expected_scales.grad, atol=1e-6, rtol=1e-5):
        raise AssertionError("DecodedWeight scales gradient differs from official decoder")
    return {"passed": True, "aqlm_commit": AQLM_COMMIT, "device": str(device)}

