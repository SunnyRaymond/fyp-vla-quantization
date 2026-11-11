"""Local Linear-input rotations and native Fast-WAM W4A4 bank support."""
from __future__ import annotations

import json
import math
import os
import sys
import types
import zlib
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


_BLOCK = 128
_GROUP_SIZE = 128
_FORMAT = "fastwam-rotation-native-bank-v1"


def _load_smooth_helpers():
    """Reuse smooth_vq.py from the runner's PYTHONPATH or a known local root."""
    import importlib
    from importlib.util import module_from_spec, spec_from_file_location

    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        try:
            module = importlib.import_module("smooth_vq")
        except ImportError:
            module = None
        if module is None:
            candidates = []
            smooth_root = os.environ.get("SMOOTH_ROOT")
            if smooth_root:
                root = Path(smooth_root)
                candidates.append(
                    root / "smooth_vq.py" if root.is_dir() else root
                )
            candidates.append(
                Path(__file__).resolve().parents[1]
                / "fastwam-smooth-vq-phases12"
                / "smooth_vq.py"
            )
            source = next((candidate for candidate in candidates if candidate.is_file()), None)
            if source is None:
                raise RuntimeError(
                    "smooth_vq.py was not importable and no SMOOTH_ROOT/local sibling exists"
                )
            spec = spec_from_file_location("_fastwam_rotation_smooth_vq", source)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"could not load shared transform source: {source}")
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = write_bytecode
    return module._fwht, module._right_r


_fwht, _right_r = _load_smooth_helpers()


def _stream_for_name(name: str) -> str:
    if name == "video_expert" or name.startswith("video_expert."):
        return "video"
    if name == "action_expert" or name.startswith("action_expert."):
        return "action"
    if name == "proprio_encoder" or name.startswith("proprio_encoder."):
        return "proprio"
    raise ValueError(f"target Linear is outside the Fast-WAM rotation scope: {name}")


def _linears(modules: Mapping[str, nn.Module]) -> dict[str, nn.Linear]:
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("modules must be a non-empty name-to-Linear mapping")
    result: dict[str, nn.Linear] = {}
    seen: dict[int, tuple[str, str]] = {}
    for name, module in modules.items():
        if not isinstance(name, str) or not isinstance(module, nn.Linear):
            raise TypeError("modules must map string names to nn.Linear instances")
        stream = _stream_for_name(name)
        previous = seen.get(id(module))
        if previous is not None:
            if previous[1] != stream:
                raise ValueError(
                    f"one shared Linear crosses rotation streams: {previous[0]} and {name}"
                )
            continue
        seen[id(module)] = (name, stream)
        result[name] = module
    return result


def _seed_for_stream(seed: int, stream: str) -> int:
    return (int(seed) + zlib.crc32(stream.encode("utf-8"))) & 0x7FFFFFFFFFFFFFFF


def _random_signs(width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bits = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int64)
    return bits.mul(2).sub(1).to(torch.float32)


def _hadamard_matrix(signs: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(_BLOCK, dtype=torch.float32, device=signs.device)
    return signs[:_BLOCK, None] * _fwht(identity)


class _StreamRotation(nn.Module):
    def __init__(
        self,
        signs: torch.Tensor,
        *,
        learned: bool,
        trainable: bool,
    ):
        super().__init__()
        self.register_buffer("base", _hadamard_matrix(signs).contiguous())
        self.register_buffer("signs", signs.contiguous())
        self.raw = nn.Parameter(
            torch.zeros((_BLOCK, _BLOCK), dtype=torch.float32, device=signs.device),
            requires_grad=bool(learned and trainable),
        )
        self.learned = bool(learned)

    def matrix(self) -> torch.Tensor:
        if not self.learned:
            return self.base
        skew = self.raw - self.raw.transpose(0, 1)
        eye = torch.eye(_BLOCK, dtype=skew.dtype, device=skew.device)
        cayley = torch.linalg.solve(eye - 0.5 * skew, eye + 0.5 * skew)
        return self.base @ cayley


class RotationState(nn.Module):
    """One shared R128 per stream, with static signed-Hadamard tails."""

    def __init__(
        self,
        modules: Mapping[str, nn.Linear],
        *,
        mode: str,
        seed: int,
    ):
        super().__init__()
        learned = mode == "learned"
        self.mode = mode
        self.seed = int(seed)
        self.module_stream = {name: _stream_for_name(name) for name in modules}
        self.module_shapes = {
            name: (module.out_features, module.in_features)
            for name, module in modules.items()
        }
        self.streams = nn.ModuleDict()
        self._matrix_cache: dict[str, torch.Tensor] = {}
        for stream in ("video", "action", "proprio"):
            selected = [
                (name, module)
                for name, module in modules.items()
                if self.module_stream[name] == stream
            ]
            if not selected:
                continue
            device = selected[0][1].weight.device
            if any(module.weight.device != device for _, module in selected):
                raise ValueError(f"all {stream} Linear weights must be on one device")
            width = max(module.in_features for _, module in selected)
            # Streams narrower than R128 use only static power-of-two tails.
            # Keep a full sign vector so their unused R128 base still has a valid shape.
            signs = _random_signs(max(width, _BLOCK), _seed_for_stream(seed, stream)).to(device=device)
            self.streams[stream] = _StreamRotation(
                signs,
                learned=learned,
                trainable=width >= _BLOCK,
            )

    @property
    def trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.parameters())

    def matrix(self, stream: str) -> torch.Tensor:
        if stream not in self.streams:
            raise KeyError(f"rotation stream is unavailable: {stream}")
        if stream not in self._matrix_cache:
            self._matrix_cache[stream] = self.streams[stream].matrix()
        return self._matrix_cache[stream]

    def refresh(self) -> None:
        """Clear cached Cayley graphs after an optimizer update."""
        self._matrix_cache.clear()

    def transform(self, value: torch.Tensor, stream: str) -> torch.Tensor:
        if value.ndim == 0 or value.shape[-1] == 0:
            raise ValueError("rotation input must have a non-empty final dimension")
        transform = self.streams[stream]
        width = value.shape[-1]
        full = (width // _BLOCK) * _BLOCK
        pieces = []
        if full:
            blocks = value[..., :full].float().reshape(
                *value.shape[:-1], full // _BLOCK, _BLOCK
            )
            pieces.append(
                torch.matmul(blocks, self.matrix(stream)).reshape(
                    *value.shape[:-1], full
                )
            )
        if full < width:
            tail = value[..., full:width].float()
            pieces.append(
                _right_r(tail, transform.signs[full:width], True, _BLOCK)
            )
        result = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
        return result.to(dtype=value.dtype)

    def stream_for(self, name: str) -> str:
        try:
            return self.module_stream[name]
        except KeyError as exc:
            raise KeyError(f"module was not included when building rotation state: {name}") from exc


def build_rotation_state(
    modules: Mapping[str, nn.Module],
    *,
    mode: str = "fixed",
    seed: int = 20261008,
) -> RotationState:
    """Build a fixed randomized R128 or a Cayley-parameterized learned R128."""
    aliases = {
        "fixed": "fixed",
        "quarot_adapted_w4a4": "fixed",
        "learned": "learned",
        "spinquant_adapted_w4a4": "learned",
    }
    if mode not in aliases:
        raise ValueError("mode must be fixed/quarot_adapted_w4a4 or learned/spinquant_adapted_w4a4")
    targets = _linears(modules)
    return RotationState(targets, mode=aliases[mode], seed=seed)


def build_rotation_optimizer(
    state: RotationState,
    *,
    lr: float = 1e-3,
    **kwargs,
) -> torch.optim.Optimizer:
    """Create Adam for the learned Cayley parameters; call handle.optimizer_step."""
    parameters = [parameter for parameter in state.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("rotation state has no trainable R128 parameters")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be a positive finite value")
    return torch.optim.Adam(parameters, lr=lr, **kwargs)


def _ste_activation(value: torch.Tensor) -> torch.Tensor:
    source = value.float()
    rows = source.reshape(-1, source.shape[-1])
    with torch.no_grad():
        detached = rows.detach()
        maximum = detached.abs().amax(dim=1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / 7.0, torch.zeros_like(maximum))
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        codes = torch.round(detached / safe).clamp(-7, 7)
        dequantized = torch.where(scale > 0, codes * scale, torch.zeros_like(codes))
    quantized = dequantized + (rows - rows.detach())
    return quantized.reshape_as(source)


def _weight_codes_and_scales(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match Quantization.convert_weights: BF16 scale, round, clamp, zero handling."""
    source = weight.detach().to(torch.bfloat16).float()
    out_features, in_features = source.shape
    padded = math.ceil(in_features / _GROUP_SIZE) * _GROUP_SIZE
    groups = padded // _GROUP_SIZE
    codes = torch.zeros((out_features, padded), dtype=torch.int8, device=source.device)
    scales = torch.zeros((out_features, groups), dtype=torch.bfloat16, device=source.device)
    with torch.no_grad():
        for group, start in enumerate(range(0, in_features, _GROUP_SIZE)):
            end = min(start + _GROUP_SIZE, in_features)
            part = source[:, start:end]
            scale = (part.abs().amax(dim=1, keepdim=True) / 7.0).to(torch.bfloat16)
            safe = torch.where(scale > 0, scale, torch.ones_like(scale)).float()
            quantized = torch.round(part / safe).clamp(-7, 7).to(torch.int8)
            quantized = torch.where(scale > 0, quantized, torch.zeros_like(quantized))
            codes[:, start:end] = quantized
            scales[:, group] = scale[:, 0]
    return codes, scales


def _ste_weight(value: torch.Tensor) -> torch.Tensor:
    source = value.to(torch.bfloat16).float()
    codes, scales = _weight_codes_and_scales(value)
    width = value.shape[1]
    padded = codes.shape[1]
    dequantized = torch.zeros_like(codes, dtype=torch.float32)
    with torch.no_grad():
        for group, start in enumerate(range(0, padded, _GROUP_SIZE)):
            end = min(start + _GROUP_SIZE, width)
            if start >= end:
                continue
            safe = torch.where(
                scales[:, group : group + 1] > 0,
                scales[:, group : group + 1],
                torch.ones_like(scales[:, group : group + 1]),
            ).float()
            dequantized[:, start:end] = codes[:, start:end].float() * safe
    source = source[:, :width]
    result = dequantized[:, :width] + (source - source.detach())
    return result


def _reference_a4_dequant(value: torch.Tensor) -> torch.Tensor:
    rows = value.float().reshape(-1, value.shape[-1])
    maximum = rows.abs().amax(dim=1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 7.0, torch.zeros_like(maximum))
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(rows / safe).clamp(-7, 7)
    return (codes * scale).reshape_as(value)


def _reference_w4_dequant(value: torch.Tensor) -> torch.Tensor:
    source = value.to(torch.bfloat16).float()
    _out_features, in_features = source.shape
    decoded = torch.zeros_like(source)
    for start in range(0, in_features, _GROUP_SIZE):
        end = min(start + _GROUP_SIZE, in_features)
        group = source[:, start:end]
        scale = (group.abs().amax(dim=1, keepdim=True) / 7.0).to(torch.bfloat16)
        safe = torch.where(scale > 0, scale, torch.ones_like(scale)).float()
        codes = torch.round(group / safe).clamp(-7, 7)
        decoded[:, start:end] = codes * scale.float()
    return decoded


def ste_bf16_precision_self_check(device: str | torch.device = "cpu") -> dict[str, Any]:
    """Check FP32 fake-dequant math against an independent BF16 fixture."""
    device = torch.device(device)
    layer = nn.Linear(257, 11, bias=True, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        index = torch.arange(11 * 257, device=device, dtype=torch.float32).reshape(11, 257)
        layer.weight.copy_((torch.sin(index * 0.019) * 0.83).to(torch.bfloat16))
        layer.weight[0].zero_()
        layer.bias.copy_(torch.linspace(-0.4, 0.6, 11, device=device).to(torch.bfloat16))
    sample = torch.linspace(-1.31, 1.79, 16 * 257, device=device).reshape(16, 257)
    sample = sample.to(torch.bfloat16)
    sample[0].zero_()
    modules = {"video_expert.precision_fixture": layer}
    state = build_rotation_state(modules, mode="fixed", seed=31)
    with torch.no_grad():
        rotated_input = state.transform(sample, "video")
        rotated_weight = state.transform(layer.weight, "video")
        expected = F.linear(
            _reference_a4_dequant(rotated_input),
            _reference_w4_dequant(rotated_weight),
            layer.bias.float(),
        ).to(torch.bfloat16)
        legacy_bf16 = F.linear(
            _reference_a4_dequant(rotated_input).to(torch.bfloat16),
            _reference_w4_dequant(rotated_weight).to(torch.bfloat16),
            layer.bias,
        )
        handle = install_training_wrappers(
            modules, state, quantized=True, checkpointed=False
        )
        try:
            with torch.autocast(device_type=device.type, enabled=True):
                actual = layer(sample)
        finally:
            restore_original_linears(handle)
    if not torch.equal(actual, expected):
        raise AssertionError("BF16 W4A4 STE output differs from independent FP32-dequant reference")
    if not torch.equal(actual[0], layer.bias):
        raise AssertionError("zero A4 row must return the BF16 bias exactly")
    if not torch.equal(actual[:, 0], layer.bias[0].expand_as(actual[:, 0])):
        raise AssertionError("zero W4 group row must return the BF16 bias exactly")
    legacy_max_abs = float((legacy_bf16.float() - expected.float()).abs().max())
    if not math.isfinite(legacy_max_abs) or legacy_max_abs == 0.0:
        raise AssertionError("BF16 operand regression fixture did not distinguish the legacy forward")
    return {
        "passed": True,
        "final_dtype": str(actual.dtype),
        "input_width": 257,
        "zero_activation_bias_exact": True,
        "zero_weight_bias_exact": True,
        "fp32_reference_max_abs_error": float((actual.float() - expected.float()).abs().max()),
        "legacy_bf16_operand_max_abs_error": legacy_max_abs,
    }


def _training_linear_forward(module: nn.Linear, value: torch.Tensor) -> torch.Tensor:
    state: RotationState = module._rotation_training_state
    stream: str = module._rotation_training_stream
    quantized: bool = module._rotation_training_quantized
    matrix = state.matrix(stream)
    signs = state.streams[stream].signs

    def run(input_value: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
        if quantized:
            with torch.autocast(
                device_type=input_value.device.type, enabled=False
            ):
                rotated_input = _rotate_with_matrix(
                    input_value, rotation_matrix, signs
                )
                rotated_weight = _rotate_with_matrix(
                    module.weight, rotation_matrix, signs
                )
                rotated_input = _ste_activation(rotated_input)
                rotated_weight = _ste_weight(rotated_weight)
                bias = None if module.bias is None else module.bias.float()
                output = F.linear(rotated_input, rotated_weight, bias)
        else:
            rotated_input = _rotate_with_matrix(
                input_value, rotation_matrix, signs
            )
            rotated_weight = _rotate_with_matrix(
                module.weight, rotation_matrix, signs
            )
            output = F.linear(
                rotated_input, rotated_weight, module.bias
            )
        return output.to(dtype=input_value.dtype)

    if (
        module._rotation_training_checkpoint
        and state.trainable
        and torch.is_grad_enabled()
    ):
        return checkpoint(run, value, matrix, use_reentrant=False)
    return run(value, matrix)


class TrainingHandle:
    def __init__(
        self,
        state: RotationState,
        entries: list[tuple[nn.Linear, Any, bool, bool]],
        *,
        quantized: bool,
        checkpointed: bool,
    ):
        self.state = state
        self._entries = entries
        self.quantized = bool(quantized)
        self.checkpointed = bool(checkpointed)
        self._restored = False

    def refresh(self) -> None:
        self.state.refresh()

    def optimizer_step(self, optimizer: torch.optim.Optimizer, closure=None):
        result = optimizer.step(closure)
        self.refresh()
        return result


def install_training_wrappers(
    modules: Mapping[str, nn.Module],
    state: RotationState,
    *,
    quantized: bool = True,
    checkpointed: bool = True,
) -> TrainingHandle:
    """Install train-time xR/WR Linear forwards with optional W4A4 STE."""
    targets = _linears(modules)
    if set(targets) != set(state.module_stream):
        raise ValueError("training modules must match the modules used to build rotation state")
    for name, module in targets.items():
        if module.weight is None:
            raise ValueError(f"training requires the original FP Linear weight: {name}")
        if hasattr(module, "_rotation_training_state"):
            raise RuntimeError(f"training wrapper is already installed: {name}")
    state.refresh()
    entries = []
    for name, module in targets.items():
        forward = module.forward
        weight_grad = module.weight.requires_grad
        bias_grad = bool(module.bias is not None and module.bias.requires_grad)
        module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)
        object.__setattr__(module, "_rotation_training_state", state)
        module._rotation_training_stream = state.stream_for(name)
        module._rotation_training_quantized = bool(quantized)
        module._rotation_training_checkpoint = bool(checkpointed)
        module.forward = types.MethodType(_training_linear_forward, module)
        entries.append((module, forward, weight_grad, bias_grad))
    return TrainingHandle(
        state, entries, quantized=quantized, checkpointed=checkpointed
    )


def restore_original_linears(handle: TrainingHandle) -> None:
    if handle._restored:
        return
    for module, forward, weight_grad, bias_grad in handle._entries:
        module.forward = forward
        module.weight.requires_grad_(weight_grad)
        if module.bias is not None:
            module.bias.requires_grad_(bias_grad)
        for name in (
            "_rotation_training_state",
            "_rotation_training_stream",
            "_rotation_training_quantized",
            "_rotation_training_checkpoint",
        ):
            module.__dict__.pop(name, None)
        module._modules.pop("_rotation_training_state", None)
    handle.refresh()
    handle._restored = True


def _pack_nibbles(values: torch.Tensor) -> torch.Tensor:
    code = values.to(torch.int16).bitwise_and(15).to(torch.uint8)
    if code.shape[-1] % 2:
        code = F.pad(code, (0, 1))
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()


def export_native_bank(
    modules: Mapping[str, nn.Module],
    state: RotationState,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export real packed W4/scales/BF16 bias and the input rotation matrices."""
    targets = _linears(modules)
    if set(targets) != set(state.module_stream):
        raise ValueError("export modules must match the modules used to build rotation state")
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    state.refresh()
    metadata = dict(metadata or {})
    default_arm = (
        "quarot_adapted_w4a4" if state.mode == "fixed" else "spinquant_adapted_w4a4"
    )
    arm = str(metadata.pop("arm", default_arm))
    entries = []
    files: list[Path] = []
    for index, (name, module) in enumerate(sorted(targets.items())):
        stream = state.stream_for(name)
        with torch.no_grad():
            rotated = state.transform(module.weight.detach(), stream)
        codes, scales = _weight_codes_and_scales(rotated)
        packed = _pack_nibbles(codes)
        padded = codes.shape[1]
        bias = (
            None
            if module.bias is None
            else module.bias.detach().to(torch.bfloat16).cpu().contiguous()
        )
        filename = f"linear_{index:04d}.pt"
        target = root / filename
        torch.save(
            {
                "module": name,
                "stream": stream,
                "shape": [module.out_features, module.in_features],
                "padded_in_features": padded,
                "group_size": _GROUP_SIZE,
                "code_range": [-7, 7],
                "nibble_order": "low nibble first; signed two's complement",
                "packed_weight": packed.detach().cpu().contiguous(),
                "weight_scales_bf16": scales.detach().cpu().contiguous(),
                "bias_bf16": bias,
            },
            target,
        )
        files.append(target)
        entries.append(
            {
                "name": name,
                "stream": stream,
                "file": filename,
                "shape": [module.out_features, module.in_features],
                "padded_in_features": padded,
            }
        )

    rotation_file = root / "rotation.pt"
    rotations = {}
    with torch.no_grad():
        for stream in state.streams:
            rotations[stream] = {
                "matrix": state.matrix(stream).detach().float().cpu().contiguous(),
                "signs": state.streams[stream].signs.detach().float().cpu().contiguous(),
            }
    state.refresh()
    torch.save({"format": _FORMAT, "streams": rotations}, rotation_file)
    files.append(rotation_file)

    manifest = {
        "format": _FORMAT,
        "arm": arm,
        "rotation_mode": state.mode,
        "rotation_method": (
            "fixed randomized signed block-Hadamard R128"
            if state.mode == "fixed"
            else "learned R128 initialized from the same signed block-Hadamard and parameterized with a Cayley transform"
        ),
        "rotation_application": (
            "local Linear-input right rotation after upstream normalization, modulation, and nonlinearities; "
            "the same R128 is shared within each stream and fixed signed-Hadamard tails are not trained"
        ),
        "seed": state.seed,
        "group_size": _GROUP_SIZE,
        "weight_bits": 4,
        "activation_bits": 4,
        "weight_quantizer": "symmetric signed W4 G128; BF16 scales; round-to-nearest; clamp [-7, 7]",
        "activation_quantizer": "symmetric signed A4 per-row absmax/7; FP32 scales; round-to-nearest; clamp [-7, 7]",
        "kv_cache_dtype": "bf16",
        "rotation_file": rotation_file.name,
        "module_count": len(entries),
        "modules": entries,
        "experiment_metadata": metadata,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(root),
        "manifest": str(manifest_path),
        "format": _FORMAT,
        "arm": arm,
        "module_count": len(entries),
        "file_bytes": sum(file.stat().st_size for file in files) + manifest_path.stat().st_size,
    }


def _rotate_with_matrix(
    value: torch.Tensor,
    matrix: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    width = value.shape[-1]
    full = (width // _BLOCK) * _BLOCK
    pieces = []
    if full:
        blocks = value[..., :full].float().reshape(
            *value.shape[:-1], full // _BLOCK, _BLOCK
        )
        pieces.append(
            torch.matmul(
                blocks, matrix.to(device=value.device, dtype=torch.float32)
            ).reshape(*value.shape[:-1], full)
        )
    if full < width:
        pieces.append(
            _right_r(
                value[..., full:width].float(),
                signs[full:width].to(device=value.device, dtype=torch.float32),
                True,
                _BLOCK,
            )
        )
    result = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)
    return result.to(dtype=value.dtype)


def _rotation_input_hook(module: nn.Linear, args):
    if not args or not isinstance(args[0], torch.Tensor):
        raise TypeError("rotated Linear expects its input tensor as the first argument")
    value = args[0]
    rotated = _rotate_with_matrix(
        value,
        module._rotation_matrix,
        module._rotation_signs,
    )
    return (rotated, *args[1:])


def _packed_linear_forward(module: nn.Linear, value: torch.Tensor) -> torch.Tensor:
    owner = module._quantization_owner
    if not owner._active:
        raise RuntimeError("packed Linear called while Quantization is disabled")
    return owner._linear(module, value)


def _remove_rotation_hooks(quantization) -> None:
    controller = getattr(quantization, "_rotation_native_state", None)
    if controller is None:
        return
    for handle in controller["hooks"]:
        handle.remove()
    controller["hooks"].clear()
    for module in quantization.modules.values():
        module._buffers.pop("_rotation_matrix", None)
        module._buffers.pop("_rotation_signs", None)
        module.__dict__.pop("_rotation_stream", None)


def apply_native_bank(quantization, path: str | Path) -> dict[str, Any]:
    """Load/replace one packed bank on an existing Quantization model.

    The first call moves only target BF16 weights to a CPU restore slot. Later
    bank switches replace packed tensors in place and leave T5/VAE untouched.
    """
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != _FORMAT:
        raise ValueError(f"unsupported rotation bank format: {manifest.get('format')!r}")
    if manifest.get("group_size") != _GROUP_SIZE or manifest.get("activation_bits") != 4:
        raise ValueError("rotation bank must use native W4A4 with G128")
    if getattr(quantization, "group_size", None) != _GROUP_SIZE:
        raise ValueError("Quantization must be constructed with group_size=128")
    if manifest.get("kv_cache_dtype") != "bf16":
        raise ValueError("rotation bank must keep the KV cache in BF16")
    if getattr(quantization, "_closed", False):
        raise RuntimeError("Quantization is closed")
    modules = _linears(quantization.modules)
    entries = manifest.get("modules")
    if not isinstance(entries, list) or len(entries) != len(modules):
        raise ValueError("rotation bank module count does not match Quantization.modules")
    entry_by_name = {item.get("name"): item for item in entries}
    if len(entry_by_name) != len(entries) or set(entry_by_name) != set(modules):
        raise ValueError("rotation bank module names do not match Quantization.modules")
    rotation_file = manifest.get("rotation_file")
    if not isinstance(rotation_file, str) or Path(rotation_file).name != rotation_file:
        raise ValueError("rotation bank has an invalid rotation_file path")
    rotation_payload = torch.load(root / manifest["rotation_file"], map_location="cpu")
    if rotation_payload.get("format") != _FORMAT:
        raise ValueError("rotation matrix file format does not match the manifest")
    rotations = rotation_payload.get("streams", {})
    expected_streams = {_stream_for_name(name) for name in modules}
    if set(rotations) != expected_streams:
        raise ValueError("rotation matrix stream set does not match target Linears")
    for stream, rotation in rotations.items():
        if tuple(rotation["matrix"].shape) != (_BLOCK, _BLOCK):
            raise ValueError(f"invalid R128 matrix shape for {stream}")
        expected_width = max(
            module.in_features
            for name, module in modules.items()
            if _stream_for_name(name) == stream
        )
        if rotation["signs"].ndim != 1 or rotation["signs"].numel() < expected_width:
            raise ValueError(f"rotation tail signs are too short for {stream}")

    controller = getattr(quantization, "_rotation_native_state", None)
    if controller is None:
        if getattr(quantization, "_weight_state", None) != "bf16":
            raise RuntimeError(
                "first bank application requires original BF16 Linears; use restore_native_bf16 "
                "or reload the source model"
            )
        originals = {}
        for name, module in modules.items():
            if module.weight is None:
                raise RuntimeError(f"original BF16 weight is absent for {name}")
            originals[name] = {
                "weight": module.weight.detach().cpu().clone(),
                "device": module.weight.device,
                "weight_requires_grad": module.weight.requires_grad,
                "bias": None if module.bias is None else module.bias.detach().cpu().clone(),
                "bias_requires_grad": bool(
                    module.bias is not None and module.bias.requires_grad
                ),
                "forward": module.forward,
            }
        controller = {"originals": originals, "hooks": []}
        quantization._rotation_native_state = controller
    elif set(controller["originals"]) != set(modules):
        raise ValueError("existing native-bank restore slot does not match Quantization.modules")

    quantization.disable()
    _remove_rotation_hooks(quantization)
    for handle in getattr(quantization, "_bf16_hooks", []):
        handle.remove()
    if hasattr(quantization, "_bf16_hooks"):
        quantization._bf16_hooks.clear()

    for name, module in modules.items():
        entry = entry_by_name[name]
        stream = _stream_for_name(name)
        filename = entry.get("file")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f"invalid packed weight path for {name}")
        if entry.get("stream") != stream:
            raise ValueError(f"rotation stream mismatch for {name}")
        if entry.get("shape") != [module.out_features, module.in_features]:
            raise ValueError(f"Linear shape mismatch for {name}")
        bank = torch.load(root / entry["file"], map_location="cpu")
        if bank.get("module") != name or bank.get("group_size") != _GROUP_SIZE:
            raise ValueError(f"packed weight metadata mismatch for {name}")
        expected_padded = math.ceil(module.in_features / _GROUP_SIZE) * _GROUP_SIZE
        if int(bank.get("padded_in_features", -1)) != expected_padded:
            raise ValueError(f"padded input width mismatch for {name}")
        expected_weight_shape = (module.out_features, expected_padded // 2)
        expected_scale_shape = (module.out_features, expected_padded // _GROUP_SIZE)
        if tuple(bank["packed_weight"].shape) != expected_weight_shape:
            raise ValueError(f"packed weight tensor shape mismatch for {name}")
        if bank["packed_weight"].dtype != torch.uint8:
            raise ValueError(f"packed weight tensor must be uint8 for {name}")
        if tuple(bank["weight_scales_bf16"].shape) != expected_scale_shape:
            raise ValueError(f"weight scale tensor shape mismatch for {name}")
        if bank["weight_scales_bf16"].dtype != torch.bfloat16:
            raise ValueError(f"weight scales must be BF16 for {name}")
        if module.weight is not None:
            device = module.weight.device
        elif hasattr(module, "_quant_packed_weight"):
            device = module._quant_packed_weight.device
        else:
            raise RuntimeError(f"target device is unavailable for {name}")
        if module.weight is not None:
            module.register_parameter("weight", None)
        packed = bank["packed_weight"].to(device=device).contiguous()
        scales = bank["weight_scales_bf16"].to(device=device, dtype=torch.bfloat16).contiguous()
        module._buffers["_quant_packed_weight"] = packed
        module._buffers["_quant_weight_scales"] = scales
        if "_quant_weight_shape" not in module._buffers:
            module.register_buffer(
                "_quant_weight_shape",
                torch.tensor(bank["shape"], device=device),
                persistent=False,
            )
        else:
            module._buffers["_quant_weight_shape"] = torch.tensor(
                bank["shape"], device=device
            )
        module._quant_padded_k = int(bank["padded_in_features"])
        module._quantization_owner = quantization

        bank_bias = bank["bias_bf16"]
        if (module.bias is None) != (bank_bias is None):
            raise ValueError(f"Linear bias presence mismatch for {name}")
        if module.bias is not None:
            if tuple(bank_bias.shape) != (module.out_features,):
                raise ValueError(f"BF16 bias tensor shape mismatch for {name}")
            module.bias.data = bank_bias.to(device=device, dtype=torch.bfloat16).contiguous()

        rotation = rotations[stream]
        matrix = rotation["matrix"].to(device=device, dtype=torch.float32).contiguous()
        signs = rotation["signs"].to(device=device, dtype=torch.float32).contiguous()
        module._buffers["_rotation_matrix"] = matrix
        module._buffers["_rotation_signs"] = signs
        module._rotation_stream = stream
        module.forward = types.MethodType(_packed_linear_forward, module)
        controller["hooks"].append(module.register_forward_pre_hook(_rotation_input_hook))

    quantization._weight_state = "w4"
    quantization._integer_gemm_calls = 0
    quantization._native_int4_gemm_calls = 0
    return manifest


def restore_native_bf16(quantization) -> None:
    """Restore target BF16 Linears from the CPU slot and remove rotation hooks."""
    controller = getattr(quantization, "_rotation_native_state", None)
    if controller is None:
        return
    quantization.disable()
    _remove_rotation_hooks(quantization)
    for name, module in quantization.modules.items():
        original = controller["originals"][name]
        for buffer_name in (
            "_quant_packed_weight",
            "_quant_weight_scales",
            "_quant_weight_shape",
            "_rotation_matrix",
            "_rotation_signs",
        ):
            module._buffers.pop(buffer_name, None)
        device = original["device"]
        module.register_parameter(
            "weight",
            nn.Parameter(
                original["weight"].to(device=device),
                requires_grad=original["weight_requires_grad"],
            ),
        )
        if original["bias"] is not None:
            if module.bias is None:
                module.register_parameter(
                    "bias",
                    nn.Parameter(
                        original["bias"].to(device=device),
                        requires_grad=original["bias_requires_grad"],
                    ),
                )
            else:
                module.bias.data = original["bias"].to(device=device)
                module.bias.requires_grad_(original["bias_requires_grad"])
        module.forward = original["forward"]
        module.__dict__.pop("_quant_padded_k", None)
        module.__dict__.pop("_quantization_owner", None)
        module.__dict__.pop("_rotation_stream", None)

    quantization._weight_state = "bf16"
    quantization._arm = None
    quantization._active = False
    quantization._activation_bits = quantization.activation_bits
    quantization._kv_active = False
    quantization._native_int4_tensorcore = False
    delattr(quantization, "_rotation_native_state")


def discard_native_bank(quantization) -> None:
    """Drop packed GPU tensors and CPU restore copies without restoring BF16 W."""
    controller = getattr(quantization, "_rotation_native_state", None)
    if controller is None:
        return
    quantization.disable()
    _remove_rotation_hooks(quantization)
    for name, module in quantization.modules.items():
        original = controller["originals"][name]
        module.forward = original["forward"]
        for buffer_name in (
            "_quant_packed_weight",
            "_quant_weight_scales",
            "_quant_weight_shape",
            "_rotation_matrix",
            "_rotation_signs",
        ):
            module._buffers.pop(buffer_name, None)
        module.__dict__.pop("_quant_padded_k", None)
        module.__dict__.pop("_quantization_owner", None)
        module.__dict__.pop("_rotation_stream", None)
    controller["hooks"].clear()
    controller["originals"].clear()
    delattr(quantization, "_rotation_native_state")
    quantization._weight_state = "discarded"
    quantization._arm = None
    quantization._active = False
    quantization._kv_active = False
    quantization._native_int4_tensorcore = False
    quantization._closed = True


def _require_pbs_compute_node() -> None:
    import socket

    job_id = os.environ.get("PBS_JOBID")
    nodefile = os.environ.get("PBS_NODEFILE")
    host = socket.gethostname().split(".", 1)[0]
    if not job_id or not nodefile:
        raise RuntimeError("native W4A4 self-check requires an active PBS allocation")
    if "login" in host.lower():
        raise RuntimeError("native W4A4 self-check refuses to run on a login node")
    try:
        allocated_hosts = {
            node.split(".", 1)[0]
            for node in Path(nodefile).read_text(encoding="utf-8").split()
        }
    except OSError as exc:
        raise RuntimeError(f"cannot read PBS allocation nodefile: {nodefile}") from exc
    if host not in allocated_hosts:
        raise RuntimeError(f"current host {host!r} is not listed in PBS_NODEFILE")


def native_ste_self_check() -> dict[str, Any]:
    """Compare corrected BF16 STE and real native W4A4 on tiny fixtures in PBS."""
    _require_pbs_compute_node()
    import tempfile
    import torch

    from quantization import Quantization

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("native W4A4 self-check requires one allocated CUDA GPU")
    device = torch.device("cuda")
    major, _minor = torch.cuda.get_device_capability(device)
    if major < 8:
        raise RuntimeError("native W4A4 self-check requires SM80+ S4 Tensor Cores")

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    quantization = None
    installed_handle = None
    try:
        precision = ste_bf16_precision_self_check(device)
        torch.manual_seed(20261008)
        model = nn.Module()
        model.video_expert = nn.Module()
        model.video_expert.proj = nn.Linear(257, 17, device=device, dtype=torch.bfloat16)
        model.action_expert = nn.Module()
        model.action_expert.proj = nn.Linear(256, 19, device=device, dtype=torch.bfloat16)
        model.proprio_encoder = nn.Linear(8, 13, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            model.video_expert.proj.weight[0].zero_()
            model.action_expert.proj.weight[0].zero_()
            model.proprio_encoder.weight[0].zero_()
            model.video_expert.proj.bias.copy_(
                torch.linspace(-0.3, 0.4, 17, device=device).to(torch.bfloat16)
            )
            model.action_expert.proj.bias.copy_(
                torch.linspace(-0.2, 0.5, 19, device=device).to(torch.bfloat16)
            )
            model.proprio_encoder.bias.copy_(
                torch.linspace(-0.1, 0.2, 13, device=device).to(torch.bfloat16)
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        quantization = Quantization(model, activation_bits=8, kv4=False, group_size=128)
        modules = quantization.modules
        fixed = build_rotation_state(modules, mode="fixed", seed=43)
        learned = build_rotation_state(modules, mode="learned", seed=43)
        for stream in fixed.streams:
            if not torch.equal(fixed.matrix(stream), learned.matrix(stream)):
                raise AssertionError(f"learned {stream} rotation must start at fixed R128")
        with torch.no_grad():
            learned.streams["video"].raw[0, 1] = 0.125
        learned.refresh()
        if torch.equal(fixed.matrix("video"), learned.matrix("video")):
            raise AssertionError("learned native fixture must contain a nonzero Cayley update")
        if learned.streams["proprio"].raw.requires_grad:
            raise AssertionError("8-wide proprio fixture must remain static")

        probes = {
            "video_expert.proj": torch.linspace(
                -1.1, 1.4, 16 * 257, device=device
            ).reshape(16, 257).to(torch.bfloat16),
            "action_expert.proj": torch.linspace(
                -0.9, 1.2, 16 * 256, device=device
            ).reshape(16, 256).to(torch.bfloat16),
            "proprio_encoder": torch.linspace(
                -0.7, 0.8, 16 * 8, device=device
            ).reshape(16, 8).to(torch.bfloat16),
        }
        for value in probes.values():
            value[0].zero_()

        states = {
            "quarot_adapted_w4a4": fixed,
            "spinquant_adapted_w4a4": learned,
        }
        ste_outputs = {}
        for arm, state in states.items():
            installed_handle = install_training_wrappers(
                modules, state, quantized=True, checkpointed=False
            )
            try:
                with torch.no_grad():
                    ste_outputs[arm] = {
                        name: modules[name](probe).detach().clone()
                        for name, probe in probes.items()
                    }
            finally:
                restore_original_linears(installed_handle)
                installed_handle = None

        native_results = {}
        with tempfile.TemporaryDirectory(prefix="fastwam-native-ste-") as temporary:
            banks = {}
            for arm, state in states.items():
                bank_path = Path(temporary) / arm
                export_native_bank(
                    modules, state, bank_path, metadata={"arm": arm}
                )
                banks[arm] = bank_path

            for arm in states:
                apply_native_bank(quantization, banks[arm])
                quantization.enable("w4a4")
                before_calls = quantization.summary()["native_int4_gemm_calls"]
                errors = {}
                with torch.no_grad():
                    for name, probe in probes.items():
                        actual = modules[name](probe)
                        expected = ste_outputs[arm][name]
                        difference = actual.float() - expected.float()
                        max_abs = float(difference.abs().max())
                        rmse = float(difference.square().mean().sqrt())
                        errors[name] = {
                            "shape": list(actual.shape),
                            "max_abs_error": max_abs,
                            "rmse": rmse,
                        }
                        if not torch.isfinite(actual.float()).all():
                            raise AssertionError(f"nonfinite native output for {arm}/{name}")
                        if max_abs > 0.02:
                            raise AssertionError(
                                f"native/STE tiny fixture mismatch for {arm}/{name}: "
                                f"max_abs_error={max_abs:.8g}"
                            )
                summary = quantization.summary()
                calls = summary["native_int4_gemm_calls"] - before_calls
                if (
                    calls < len(probes)
                    or not summary["native_int4_tensorcore"]
                    or not summary["native_int4_ptx_verified"]
                ):
                    raise AssertionError(
                        f"native PTX/call check failed for {arm}: "
                        f"calls={calls}, summary={summary}"
                    )
                native_results[arm] = {
                    "errors": errors,
                    "native_int4_gemm_calls": calls,
                    "native_int4_tensorcore": summary["native_int4_tensorcore"],
                    "native_int4_ptx_verified": summary["native_int4_ptx_verified"],
                    "native_int4_ptx_line": summary["native_int4_ptx_line"],
                    "bf16_target_linear_weights_absent": summary[
                        "bf16_target_linear_weights_absent"
                    ],
                }
                if not summary["bf16_target_linear_weights_absent"]:
                    raise AssertionError(f"BF16 target weights remained on GPU for {arm}")

        return {
            "passed": True,
            "pbs_jobid": os.environ["PBS_JOBID"],
            "gpu": torch.cuda.get_device_name(device),
            "ste_bf16_precision": precision,
            "static_proprio_trainable": False,
            "learned_nonzero_cayley_fixture": True,
            "native_checks": native_results,
        }
    finally:
        if installed_handle is not None:
            restore_original_linears(installed_handle)
        if quantization is not None:
            if getattr(quantization, "_rotation_native_state", None) is not None:
                discard_native_bank(quantization)
            quantization.close()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def self_check() -> dict[str, Any]:
    """Small CPU-only mathematical and native-bank state checks."""
    torch.manual_seed(17)
    precision = ste_bf16_precision_self_check()
    layer = nn.Linear(300, 11, bias=True, dtype=torch.float32)
    modules = {"video_expert.test": layer}
    fixed = build_rotation_state(modules, mode="fixed", seed=23)
    learned = build_rotation_state(modules, mode="learned", seed=23)
    if not torch.equal(fixed.matrix("video"), learned.matrix("video")):
        raise AssertionError("learned R128 must start exactly at the fixed transform")

    value = torch.randn(4, 300, dtype=torch.float32)
    signs = fixed.streams["video"].signs
    expected_transform = torch.cat(
        (
            _right_r(value[:, :_BLOCK], signs[:_BLOCK], True, _BLOCK),
            _right_r(value[:, _BLOCK : 2 * _BLOCK], signs[:_BLOCK], True, _BLOCK),
            _right_r(value[:, 2 * _BLOCK :], signs[2 * _BLOCK :], True, _BLOCK),
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        fixed.transform(value, "video"), expected_transform, rtol=1e-6, atol=1e-6
    )
    reference = F.linear(value, layer.weight, layer.bias)
    handle_fp = install_training_wrappers(
        modules, fixed, quantized=False, checkpointed=False
    )
    actual = layer(value)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
    restore_original_linears(handle_fp)

    # LIBERO proprio is only 8-D: it must remain a static Hadamard tail, not
    # fail while constructing the stream's otherwise-unused R128 matrix.
    proprio_layer = nn.Linear(8, 4096, bias=True, dtype=torch.float32)
    proprio_modules = {"proprio_encoder": proprio_layer}
    fixed_proprio = build_rotation_state(proprio_modules, mode="fixed", seed=23)
    learned_proprio = build_rotation_state(proprio_modules, mode="learned", seed=23)
    if (fixed_proprio.trainable or learned_proprio.trainable
            or not torch.equal(fixed_proprio.matrix("proprio"), learned_proprio.matrix("proprio"))):
        raise AssertionError("sub-R128 proprio must stay static and match the fixed initialization")
    proprio_value = torch.randn(2, 8, dtype=torch.float32)
    proprio_reference = F.linear(proprio_value, proprio_layer.weight, proprio_layer.bias)
    proprio_handle = install_training_wrappers(
        proprio_modules, fixed_proprio, quantized=False, checkpointed=False
    )
    proprio_actual = proprio_layer(proprio_value)
    torch.testing.assert_close(proprio_actual, proprio_reference, rtol=2e-5, atol=2e-5)
    restore_original_linears(proprio_handle)

    target = torch.randn(4, layer.out_features, dtype=torch.float32)
    raw = learned.streams["video"].raw

    def gradient_for_paths(rotate_input: bool, rotate_weight: bool):
        learned.refresh()
        matrix = learned.matrix("video")
        signs = learned.streams["video"].signs
        rotated_input = _rotate_with_matrix(
            value, matrix if rotate_input else matrix.detach(), signs
        )
        rotated_weight = _rotate_with_matrix(
            layer.weight, matrix if rotate_weight else matrix.detach(), signs
        )
        prediction = F.linear(
            _ste_activation(rotated_input),
            _ste_weight(rotated_weight),
            layer.bias,
        )
        return torch.autograd.grad((prediction * target).mean(), raw)[0]

    input_gradient = gradient_for_paths(True, False)
    weight_gradient = gradient_for_paths(False, True)
    if (
        float(input_gradient.norm()) == 0.0
        or float(weight_gradient.norm()) == 0.0
    ):
        raise AssertionError("both xR and WR must contribute gradients to R")

    handle = install_training_wrappers(
        modules, learned, quantized=True, checkpointed=True
    )
    optimizer = build_rotation_optimizer(learned, lr=1e-3)
    output = layer(value)
    output.square().mean().backward()
    grad = learned.streams["video"].raw.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.norm()) == 0.0:
        raise AssertionError("STE loss must produce a finite nonzero rotation gradient")
    before = learned.matrix("video").detach().clone()
    handle.optimizer_step(optimizer)
    after = learned.matrix("video").detach()
    identity = torch.eye(_BLOCK, dtype=after.dtype)
    torch.testing.assert_close(after.T @ after, identity, rtol=2e-5, atol=2e-5)
    if torch.equal(before, after):
        raise AssertionError("optimizer update did not change the learned rotation")
    restore_original_linears(handle)

    class FakeQuantization:
        def __init__(self, target):
            self.modules = target
            self.model = nn.Sequential(layer)
            self.group_size = _GROUP_SIZE
            self.activation_bits = 4
            self._weight_state = "bf16"
            self._active = False
            self._kv_active = False
            self._bf16_hooks = []
            self._integer_gemm_calls = 0
            self._native_int4_gemm_calls = 0
            self._native_int4_tensorcore = False
            self._closed = False

        def disable(self):
            self._active = False
            self._kv_active = False

    fake_quant = FakeQuantization(modules)
    original = layer.weight.detach().clone()
    with __import__("tempfile").TemporaryDirectory() as temp_dir:
        from pathlib import Path

        root = Path(temp_dir)
        bank = export_native_bank(
            modules,
            fixed,
            root / "quarot_adapted_w4a4",
            metadata={"arm": "quarot_adapted_w4a4"},
        )
        spin_bank = export_native_bank(
            modules,
            learned,
            root / "spinquant_adapted_w4a4",
            metadata={"arm": "spinquant_adapted_w4a4"},
        )
        manifest = apply_native_bank(fake_quant, bank["path"])
        if (
            fake_quant._weight_state != "w4"
            or layer.weight is not None
            or not hasattr(layer, "_quant_packed_weight")
            or bank["module_count"] != 1
            or manifest["module_count"] != 1
        ):
            raise AssertionError("native bank load did not install packed Linear state")
        switched = apply_native_bank(fake_quant, spin_bank["path"])
        if (
            switched["arm"] != "spinquant_adapted_w4a4"
            or fake_quant._weight_state != "w4"
            or layer.weight is not None
        ):
            raise AssertionError("native bank switch did not preserve packed Linear state")
        restore_native_bf16(fake_quant)
        torch.testing.assert_close(layer.weight, original, rtol=0, atol=0)
        if fake_quant._weight_state != "bf16" or hasattr(layer, "_quant_packed_weight"):
            raise AssertionError("native bank restore did not return to BF16 state")
        apply_native_bank(fake_quant, bank["path"])
        discard_native_bank(fake_quant)
        if (
            not fake_quant._closed
            or fake_quant._weight_state != "discarded"
            or hasattr(fake_quant, "_rotation_native_state")
            or hasattr(layer, "_quant_packed_weight")
            or layer.weight is not None
        ):
            raise AssertionError("native bank discard did not release state without BF16 restore")

    return {
        "passed": True,
        "fp32_rotation_equivalence": True,
        "static_sub_r128_proprio_tail": True,
        "ste_bf16_precision": precision,
        "learned_initial_equals_fixed": True,
        "w4a4_ste_rotation_gradient": True,
        "gradients_through_input_and_weight_rotations": True,
        "cayley_orthogonality_after_update": True,
        "native_bank_state_roundtrip": True,
        "native_packed_bank_switch": True,
        "native_bank_discard": True,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python rotation.py --selfcheck | --native-ste-selfcheck"
        )
    if sys.argv[1] == "--selfcheck":
        result = self_check()
    elif sys.argv[1] == "--native-ste-selfcheck":
        result = native_ste_self_check()
    else:
        raise SystemExit(
            "usage: python rotation.py --selfcheck | --native-ste-selfcheck"
        )
    print(json.dumps(result, indent=2))
