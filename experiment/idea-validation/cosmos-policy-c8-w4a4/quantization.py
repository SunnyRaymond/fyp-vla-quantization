"""Cosmos Linear selection and decoded-BF16 W4A4/VQ arm adapters.

The numerical paths here deliberately reuse the experiment helpers through
lazy imports.  Importing this module therefore does not import Fast-WAM,
Cosmos, or torch before the selected runner has prepared its environment.
"""
from __future__ import annotations

import json
import math
import os
import types
from pathlib import Path
from typing import Any, Mapping


BF16_ARM = "bf16"
SPIN_ARM = "spinquant_w4a4"
VQ_ARM = "action_latent_micro_vq_a4"
SPIN_STAGE = "spinquant"
VQ_STAGE = "vq"
VQ_FORMAT = "cosmos_action_latent_micro_vq_a4_v1"
SPIN_FORMAT = "cosmos_spinquant_w4a4_v1"
SPIN_BANK_FORMAT = "fastwam-rotation-native-bank-v1"


def _torch():
    import torch

    return torch


def _decode_spin_bank_weight(bank: Mapping[str, Any], shape: tuple[int, int]):
    """Decode the exported signed W4/G128 payload once into a BF16 Linear weight."""
    torch = _torch()
    out_features, in_features = map(int, shape)
    padded = _ceil_div(in_features, 128) * 128
    packed = bank.get("packed_weight")
    scales = bank.get("weight_scales_bf16")
    if (not torch.is_tensor(packed) or packed.dtype != torch.uint8
            or tuple(packed.shape) != (out_features, padded // 2)):
        raise ValueError("Spin packed weight must be uint8 with the frozen padded G128 shape")
    if (not torch.is_tensor(scales) or scales.dtype != torch.bfloat16
            or tuple(scales.shape) != (out_features, padded // 128)):
        raise ValueError("Spin per-group scales must be BF16 with the frozen G128 shape")
    codes = packed.to(dtype=torch.int16)
    nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(out_features, padded)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles).float()
    decoded = (signed.reshape(out_features, padded // 128, 128)
               * scales.float().unsqueeze(-1)).reshape(out_features, padded)
    return decoded[:, :in_features].to(torch.bfloat16).contiguous()


def _decoded_spin_linear_forward(module, value):
    """Inference-only BF16-operand path using the immutable decoded W4 bank."""
    torch = _torch()
    state = module.__dict__["_cosmos_spin_state"]
    rotation = state["rotate"]
    value = value.to(dtype=module.weight.dtype)
    rotated = rotation(value, state["matrix"], state["signs"])
    activation = state["a4_dequant"](rotated).to(dtype=module.weight.dtype)
    with torch.autocast(device_type=activation.device.type, enabled=False):
        return torch.nn.functional.linear(activation, module.weight, module.bias)


def _helpers():
    """Import the frozen helper implementations after the runner sets PYTHONPATH."""
    import batched_transform
    import micro_codebook
    import rotation
    import smooth_vq
    import train_banks

    if getattr(batched_transform, "IMPLEMENTATION", None) != "batched_h128_v2":
        raise RuntimeError("Cosmos VQ requires the existing batched H128 implementation")
    return rotation, micro_codebook, batched_transform, smooth_vq, train_banks


def _tokens(path: str) -> tuple[str, ...]:
    return tuple(part.lower() for part in path.replace("/", ".").split(".") if part)


def _contains_any(parts: tuple[str, ...], candidates: set[str]) -> bool:
    return any(part in candidates or any(part.startswith(value) for value in candidates)
               for part in parts)


def _is_transformer_block(parts: tuple[str, ...]) -> bool:
    return any(part in {"blocks", "transformer_blocks", "layers"} for part in parts)


def _is_attention_projection(parts: tuple[str, ...]) -> bool:
    return _contains_any(parts, {
        "attn", "attention", "self_attn", "self_attention",
        "cross_attn", "cross_attention",
    })


def _is_main_projection(parts: tuple[str, ...]) -> bool:
    return _is_attention_projection(parts) or _contains_any(parts, {
        "mlp", "ffn", "feed_forward", "feedforward", "ff",
    })


def _is_excluded_projection(parts: tuple[str, ...]) -> bool:
    return _contains_any(parts, {
        "patch_embed", "patch_embedding", "patchify", "out_head", "output_head",
        "output_layer", "final_output", "final_layer", "final_proj", "time_embed",
        "timestep_embed", "time_embedding",
        "adaln", "modulation", "proprio_encoder", "proprio_embed", "action_encoder",
        "action_embed", "text_embedder", "caption_embedder", "position_embed",
    })


def select_target_linears(model, protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Select main DiT attention and MLP ``nn.Linear`` modules by frozen scope.

    Patch/output/time/proprio/action-conditioning and other small embedding
    projections stay BF16.  The returned insertion order is canonical by full
    module name, and should be recorded by the caller in its build manifest.
    """
    torch = _torch()
    dit = getattr(model, "net", None)
    if not isinstance(dit, torch.nn.Module):
        raise RuntimeError("Cosmos target selection requires the model.net DiT subtree")
    selected: dict[str, Any] = {}
    for local_name, module in sorted(dit.named_modules(), key=lambda item: item[0]):
        if not local_name or not isinstance(module, torch.nn.Linear):
            continue
        name = "net." + local_name
        parts = _tokens(name)
        if (_is_transformer_block(parts) and _is_main_projection(parts)
                and not _is_excluded_projection(parts)):
            selected[name] = module
    if not selected:
        raise RuntimeError("No Cosmos DiT attention/MLP Linear modules matched the frozen target scope")
    if len({id(module) for module in selected.values()}) != len(selected):
        raise RuntimeError("The selected target scope contains aliased Linear modules")
    attention = [name for name in selected if _is_attention_projection(_tokens(name))]
    mlp = [name for name in selected if _contains_any(_tokens(name), {
        "mlp", "ffn", "feed_forward", "feedforward", "ff",
    })]
    if not attention or not mlp:
        raise RuntimeError(f"Target scope must contain both attention and MLP Linears; got {len(attention)}/{len(mlp)}")
    frozen_names = None if protocol is None else protocol.get("quantization", {}).get("target_module_names")
    if frozen_names is not None and list(selected) != list(frozen_names):
        raise RuntimeError("Runtime target Linear names differ from protocol's frozen target_module_names")
    return selected


def target_manifest(modules: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic module identity rows without hashing model weights."""
    return [{"name": name, "shape": [int(module.out_features), int(module.in_features)],
             "weight_numel": int(module.weight.numel()),
             "bias_numel": 0 if module.bias is None else int(module.bias.numel())}
            for name, module in sorted(modules.items())]


def _ceil_div(n: int, d: int) -> int:
    return (int(n) + int(d) - 1) // int(d)


def effective_bpw(modules: Mapping[str, Any], model=None) -> dict[str, Any]:
    """Exact tensor-byte accounting for the matched Linear weights and whole DiT.

    Tensors are counted at deployed dtype/packing; JSON/container framing is
    reported separately as serialized bytes. Target Linear biases are excluded
    from the matched-weight ratio and remain in the whole-DiT parameter bytes.
    """
    if not modules:
        raise ValueError("effective_bpw requires a non-empty fixed module mapping")
    rows = target_manifest(modules)
    weights = sum(row["weight_numel"] for row in rows)
    spin_w = spin_sc = 0
    vq_idx = vq_exp = vq_books = vq_gain = vq_transform = 0
    max_input = 0
    for row in rows:
        out_width, in_width = row["shape"]
        groups128 = _ceil_div(in_width, 128)
        padded128 = 128 * groups128
        spin_w += out_width * padded128 // 2
        spin_sc += out_width * groups128 * 2
        qgroups = _ceil_div(in_width, 4)
        qblocks = _ceil_div(in_width, 32)
        vq_idx += out_width * qgroups * 2
        vq_exp += out_width * qblocks
        vq_books += 2 * 256 * 4 * 2
        vq_gain += out_width * 2
        vq_transform += in_width * 2 * 4  # FP32 smooth scales plus FP32 Rademacher signs.
        max_input = max(max_input, in_width)
    spin_rotation = (128 * 128 + max(max_input, 128)) * 4  # R128 matrix and tail signs.
    # Biases are BF16 in every arm and therefore excluded from the matched
    # target-weight bpw numerator; they remain in the whole-DiT denominator.
    spin_bytes = spin_w + spin_sc + spin_rotation
    vq_bytes = vq_idx + vq_exp + vq_books + vq_gain + vq_transform
    result = {
        "target_modules": len(rows), "target_module_names_and_shapes": rows,
        "target_weight_numel": weights,
        "spinquant": {
            "bytes": spin_bytes, "weight_bytes": spin_w, "scale_bytes": spin_sc,
            "transform_metadata_bytes": spin_rotation,
            "effective_bpw": 8.0 * spin_bytes / weights,
        },
        "micro_vq": {
            "bytes": vq_bytes, "index_bytes": vq_idx, "exponent_bytes": vq_exp,
            "codebook_bytes": vq_books, "row_gain_bytes": vq_gain,
            "transform_metadata_bytes": vq_transform,
            "effective_bpw": 8.0 * vq_bytes / weights,
        },
        "same_module_relative_bpw_delta": abs(vq_bytes - spin_bytes) / spin_bytes,
        "same_module_within_5_percent": abs(vq_bytes - spin_bytes) / spin_bytes <= 0.05,
    }
    if model is not None:
        torch = _torch()
        dit = getattr(model, "net", None)
        if not isinstance(dit, torch.nn.Module):
            raise RuntimeError("Cosmos whole-DiT accounting requires the model.net DiT subtree")
        dit_module_ids = {id(module) for module in dit.modules()}
        if any(id(module) not in dit_module_ids for module in modules.values()):
            raise RuntimeError("Target Linear modules are not all inside model.net")
        parameters = list(dit.parameters())
        total = sum(int(parameter.numel()) for parameter in parameters)
        parameter_bytes = sum(int(parameter.numel()) * parameter.element_size()
                              for parameter in parameters)
        target_weight_ids = {id(module.weight) for module in modules.values()}
        untouched_bytes = sum(int(parameter.numel()) * parameter.element_size()
                              for parameter in parameters if id(parameter) not in target_weight_ids)
        if len(target_weight_ids) != len(modules):
            raise RuntimeError("Target modules do not have one unique weight Parameter each")
        dtype_bytes = {}
        for parameter in parameters:
            key = str(parameter.dtype)
            dtype_bytes[key] = dtype_bytes.get(key, 0) + int(parameter.numel()) * parameter.element_size()
        if total < weights or parameter_bytes < untouched_bytes:
            raise RuntimeError("Target parameter count exceeds the whole DiT parameter count")
        result["whole_dit"] = {
            "parameter_numel": total,
            "parameter_bytes": parameter_bytes,
            "parameter_bytes_by_dtype": dtype_bytes,
            "untouched_parameter_bytes": untouched_bytes,
            "bf16_effective_bpw": 8.0 * parameter_bytes / total,
            "spinquant_effective_bpw": 8.0 * (spin_bytes + untouched_bytes) / total,
            "micro_vq_effective_bpw": 8.0 * (vq_bytes + untouched_bytes) / total,
            "precision_scope": "target Linear weights encoded as reported; all remaining model.net parameters retain loaded dtype",
        }
    return result


def _aliases(modules: Mapping[str, Any]) -> dict[str, Any]:
    return {"video_expert." + name: module for name, module in sorted(modules.items())}


def build_spin_training(modules: Mapping[str, Any], *, seed: int = 20261015):
    """Install learned R128 + W4A4 STE without nesting per-Linear checkpoints.

    The pinned Cosmos DiT already wraps each transformer block in its native
    selective activation checkpoint.  Its count-based ``predict2_2b_720``
    policy replays cached ``aten.mm`` outputs in call order, so stacking the
    Fast-WAM per-Linear checkpoint changes cache consumption during backward.
    Keep the official block checkpointing active and disable only that inner
    Linear checkpoint layer. Precompute each learned rotation matrix outside
    SAC: its lazy Cayley construction uses ``aten.mm`` only on the original
    forward, which otherwise shifts the native replay cache on recomputation.
    """
    rotation, _, _, _, _ = _helpers()
    aliases = _aliases(modules)
    state = rotation.build_rotation_state(aliases, mode="learned", seed=seed)
    handle = rotation.install_training_wrappers(
        aliases, state, quantized=True, checkpointed=False
    )
    prewarm_spin_rotations(state)
    return state, handle, aliases


def prewarm_spin_rotations(state):
    """Build learned matrices before the next checkpointed denoiser forward."""
    for stream in state.streams:
        state.matrix(stream)


def vq_train_state(modules: Mapping[str, Any], encodings: Mapping[str, dict],
                   transforms: Mapping[str, dict], *, trainable: bool = True,
                   predecoded_weights: Mapping[str, Any] | None = None):
    """Install trainable books/row gains via the existing VQ decoder autograd."""
    torch = _torch()
    _, micro, _, _, train_banks = _helpers()
    if set(modules) != set(encodings) or set(modules) != set(transforms):
        raise ValueError("VQ bank/transform coverage must exactly match target Linears")
    state = _VQTrainState(modules, encodings, transforms, micro, train_banks,
                           trainable=trainable, predecoded_weights=predecoded_weights)
    state.install()
    return state


class _VQTrainState:
    def __init__(self, modules, encodings, transforms, micro, train_banks, *, trainable,
                 predecoded_weights=None):
        self.modules = dict(sorted(modules.items()))
        self.encodings, self.transforms = {}, {}
        self.micro, self.train_banks = micro, train_banks
        self.books, self.logs, self.active = {}, {}, {}
        self.forwards = {name: module.forward for name, module in self.modules.items()}
        self.trainable = bool(trainable)
        self.predecoded_weights = predecoded_weights
        self.seen: set[str] = set()
        self._restored = False
        torch = _torch()
        for name, module in self.modules.items():
            encoding = dict(encodings[name])
            self.micro.check_encoding(encoding)
            device = module.weight.device
            encoding = {key: value.to(device) if torch.is_tensor(value) else value
                        for key, value in encoding.items()}
            transform = {key: value.to(device) if torch.is_tensor(value) else value
                         for key, value in transforms[name].items()}
            if transform.get("input_dim") != module.in_features or transform.get("block") != 128:
                raise ValueError(f"Invalid H128 transform dimensions for {name}")
            self.encodings[name], self.transforms[name] = encoding, transform
            self.books[name] = torch.nn.Parameter(
                encoding["books"].float().detach().clone(), requires_grad=self.trainable
            )
            gains = encoding["row_scales"].float()
            self.active[name] = (gains > 0).float()
            self.logs[name] = torch.nn.Parameter(
                gains.clamp_min(torch.finfo(torch.float32).tiny).log(), requires_grad=self.trainable
            )
            module.weight.requires_grad_(False)

    def install(self):
        torch = _torch()
        for name, module in self.modules.items():
            def forward(value, _name=name, _module=module):
                self.seen.add(_name)
                transformed = self.train_banks.A4Input.apply(value, self.transforms[_name])
                if self.trainable:
                    decoded = self.train_banks.DecodedWeight.apply(
                        self.books[_name], self.logs[_name], self.active[_name],
                        self.encodings[_name]["indices"], self.encodings[_name]["scale_exponents"],
                        _module.weight,
                    )
                    return torch.nn.functional.linear(transformed, decoded, _module.bias)
                with torch.autocast(device_type=transformed.device.type, enabled=False):
                    return torch.nn.functional.linear(
                        transformed.to(dtype=_module.weight.dtype), _module.weight, _module.bias
                    )
            module.forward = forward
        self.refresh()

    def parameters(self):
        return [parameter for name in self.modules
                for parameter in (self.books[name], self.logs[name]) if parameter.requires_grad]

    def stored_encoding(self, name):
        torch = _torch()
        value = dict(self.encodings[name])
        value["books"] = self.books[name].detach().to(torch.bfloat16)
        value["row_scales"] = (self.logs[name].detach().exp() * self.active[name]).to(torch.bfloat16)
        return value

    def refresh(self):
        torch = _torch()
        with torch.no_grad():
            for name, module in self.modules.items():
                encoding = self.stored_encoding(name)
                fixed = self.encodings[name]
                if (not torch.equal(encoding["indices"], fixed["indices"])
                        or not torch.equal(encoding["scale_exponents"], fixed["scale_exponents"])):
                    raise RuntimeError(f"VQ indices/exponents changed during book/gain training at {name}")
                if not self.trainable and self.predecoded_weights is not None:
                    decoded = self.predecoded_weights[name].to(
                        device=module.weight.device, dtype=module.weight.dtype
                    )
                else:
                    decoded = self.micro.decode_vq(encoding).to(device=module.weight.device,
                                                                dtype=module.weight.dtype)
                module.weight.data.copy_(decoded)

    def snapshot(self):
        return {name: {"books": self.books[name].detach().cpu().clone(),
                       "log_scales": self.logs[name].detach().cpu().clone()}
                for name in self.modules}

    def restore(self, state):
        torch = _torch()
        if set(state) != set(self.modules):
            raise RuntimeError("VQ checkpoint master coverage differs from the fixed target modules")
        with torch.no_grad():
            for name in self.modules:
                self.books[name].copy_(state[name]["books"].to(self.books[name].device))
                self.logs[name].copy_(state[name]["log_scales"].to(self.logs[name].device))
        self.refresh()

    def deployment(self):
        return {name: {"encoding": self.stored_encoding(name), "transform": {
            key: value.detach().cpu() if _torch().is_tensor(value) else value
            for key, value in self.transforms[name].items()}}
                for name in self.modules}

    def restore_forwards(self):
        if self._restored:
            return
        for name, module in self.modules.items():
            module.forward = self.forwards[name]
        self._restored = True


class QuantizationController:
    """Switch BF16, adapted SpinQuant W4A4, and micro-VQ/A4 on one loaded model."""
    def __init__(self, model, target_linears: Mapping[str, Any], *, root: str | Path | None = None,
                 protocol: Mapping[str, Any] | None = None):
        if not target_linears:
            raise ValueError("QuantizationController requires a non-empty target map")
        torch = _torch()
        for name, module in target_linears.items():
            if (module.weight.dtype != torch.bfloat16
                    or (module.bias is not None and module.bias.dtype != torch.bfloat16)):
                raise RuntimeError(f"Cosmos quantization requires BF16 target weights and biases: {name}")
        self.model = model
        self.modules = dict(sorted(target_linears.items()))
        self.root = None if root is None else Path(root)
        self.protocol = protocol
        self._forwards = {name: module.forward for name, module in self.modules.items()}
        self._base_weights_cpu = None
        self._weight_requires_grad = {name: module.weight.requires_grad for name, module in self.modules.items()}
        self._spin_state = None
        self._spin_handle = None
        self._vq_state = None
        self._artifact_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._device_rotation_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.arm = BF16_ARM
        self.metadata = self._bf16_metadata()

    def _bf16_metadata(self):
        size_report = effective_bpw(self.modules, self.model)
        return {
            "arm": BF16_ARM, "execution": "bf16_reference",
            "execution_scope": "decoded_bf16_numerical_comparison",
            "native_int4_kernel_used": False,
            "weight_format": "bf16", "activation_format": "bf16",
            "inference_weight_operand_dtype": "torch.bfloat16",
            "inference_activation_operand_dtype": "torch.bfloat16",
            "transform": "none", "target_module_count": len(self.modules),
            "target_weight_numel": sum(int(module.weight.numel()) for module in self.modules.values()),
            "same_module_effective_bpw": None,
            # The evaluator reserves BPW comparisons for compressed arms; the
            # actual BF16 baseline byte count remains in calibration size_report.
            "whole_dit_effective_bpw": None,
            "whole_dit_parameter_numel": size_report["whole_dit"]["parameter_numel"],
            "bpw_match_relative_delta": None, "artifact_dir": None,
            "artifact_manifest": None,
        }

    def _backup_weights(self):
        if self._base_weights_cpu is None:
            self._base_weights_cpu = {name: module.weight.detach().cpu().clone()
                                      for name, module in self.modules.items()}

    def _restore_current(self):
        if self._spin_handle is not None:
            rotation, _, _, _, _ = _helpers()
            rotation.restore_original_linears(self._spin_handle)
            self._spin_handle = None
            self._spin_state = None
        if self._vq_state is not None:
            self._vq_state.restore_forwards()
            self._vq_state = None
        for name, module in self.modules.items():
            module.forward = self._forwards[name]
            module.__dict__.pop("_cosmos_spin_state", None)
        if self._base_weights_cpu is not None:
            torch = _torch()
            with torch.no_grad():
                for name, module in self.modules.items():
                    module.weight.data.copy_(self._base_weights_cpu[name].to(
                        device=module.weight.device, dtype=module.weight.dtype))
                    module.weight.requires_grad_(self._weight_requires_grad[name])

    def _artifact_path(self, arm: str, artifact_dir=None) -> Path:
        if artifact_dir is not None:
            return Path(artifact_dir)
        if self.root is not None:
            stage = "spinquant" if arm == SPIN_ARM else "vq"
            return self.root / "results" / stage
        raise ValueError("artifact_dir or controller root is required to install a trained arm")

    def _cache_key(self, arm: str, path: Path) -> tuple[str, str]:
        return arm, str(path.expanduser().resolve())

    def _load_spin_artifact(self, path: Path) -> dict[str, Any]:
        torch = _torch()
        key = self._cache_key(SPIN_ARM, path)
        if key in self._artifact_cache:
            return self._artifact_cache[key]
        bank_root = path / "packed_bank"
        deployment = torch.load(path / "deployment.pt", map_location="cpu", weights_only=True)
        if (deployment.get("format") != SPIN_FORMAT
                or deployment.get("target_modules") != target_manifest(self.modules)):
            raise RuntimeError("SpinQuant deployment format or target-module manifest differs")
        manifest_path = bank_root / "manifest.json"
        bank_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (bank_manifest.get("format") != SPIN_BANK_FORMAT
                or bank_manifest.get("rotation_mode") != "learned"
                or bank_manifest.get("group_size") != 128
                or bank_manifest.get("activation_bits") != 4):
            raise RuntimeError("SpinQuant packed bank is not the frozen learned W4A4/G128 artifact")
        expected_names = {"video_expert." + name for name in self.modules}
        entries = bank_manifest.get("modules", [])
        entry_by_name = {entry.get("name"): entry for entry in entries}
        if len(entry_by_name) != len(entries) or set(entry_by_name) != expected_names:
            raise RuntimeError("SpinQuant packed bank module names differ from fixed target scope")
        rotation_file = bank_manifest.get("rotation_file")
        if not isinstance(rotation_file, str) or Path(rotation_file).name != rotation_file:
            raise RuntimeError("SpinQuant packed bank has an invalid rotation path")
        rotation_payload = torch.load(bank_root / rotation_file, map_location="cpu", weights_only=True)
        if rotation_payload.get("format") != SPIN_BANK_FORMAT:
            raise RuntimeError("SpinQuant R128 artifact has an unexpected format")
        rotations = rotation_payload.get("streams", {})
        if set(rotations) != {"video"}:
            raise RuntimeError("Cosmos SpinQuant requires one shared video R128 stream")
        rotation = rotations["video"]
        if (tuple(rotation["matrix"].shape) != (128, 128)
                or rotation["matrix"].dtype != torch.float32
                or rotation["signs"].dtype != torch.float32
                or rotation["signs"].ndim != 1
                or rotation["signs"].numel() < max(module.in_features for module in self.modules.values())):
            raise RuntimeError("SpinQuant R128 payload must use FP32 matrix and sign tensors")
        decoded_weights = {}
        for name, module in self.modules.items():
            alias = "video_expert." + name
            entry = entry_by_name[alias]
            if (entry.get("stream") != "video"
                    or entry.get("shape") != [module.out_features, module.in_features]):
                raise RuntimeError(f"SpinQuant packed-bank identity mismatch for {name}")
            filename = entry.get("file")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise RuntimeError(f"SpinQuant packed-bank file path is invalid for {name}")
            payload = torch.load(bank_root / filename, map_location="cpu", weights_only=True)
            if (payload.get("module") != alias
                    or payload.get("shape") != [module.out_features, module.in_features]
                    or int(payload.get("group_size", -1)) != 128
                    or payload.get("padded_in_features") != _ceil_div(module.in_features, 128) * 128
                    or payload.get("code_range") != [-7, 7]
                    or payload.get("nibble_order") != "low nibble first; signed two's complement"):
                raise RuntimeError(f"SpinQuant packed W4 metadata mismatch for {name}")
            bank_bias = payload.get("bias_bf16")
            if (module.bias is None) != (bank_bias is None):
                raise RuntimeError(f"SpinQuant packed bank bias presence mismatch for {name}")
            if bank_bias is not None and (bank_bias.dtype != torch.bfloat16
                                          or tuple(bank_bias.shape) != (module.out_features,)):
                raise RuntimeError(f"SpinQuant packed bank bias shape/dtype mismatch for {name}")
            if bank_bias is not None and not torch.equal(bank_bias, module.bias.detach().cpu()):
                bank_flat = bank_bias.detach().float().reshape(-1)
                base_flat = module.bias.detach().cpu().float().reshape(-1)
                delta = (bank_flat - base_flat).abs()
                finite_delta = delta[torch.isfinite(delta)]
                max_abs = float(finite_delta.max()) if finite_delta.numel() else None
                bank_head = [float(value) for value in bank_flat[:3]]
                base_head = [float(value) for value in base_flat[:3]]
                bank_finite = bool(torch.isfinite(bank_flat).all())
                base_finite = bool(torch.isfinite(base_flat).all())
                raise RuntimeError(
                    f"SpinQuant packed bank bias differs from the BF16 base model at {name}: "
                    f"finite_bank={bank_finite}, finite_base={base_finite}, max_abs={max_abs}, "
                    f"bank_first3={bank_head}, base_first3={base_head}"
                )
            decoded_weights[name] = _decode_spin_bank_weight(
                payload, (module.out_features, module.in_features)
            )
        loaded = {
            "metadata": dict(deployment["install_metadata"]),
            "deployment_manifest": path / "deployment.pt",
            "packed_bank_manifest": manifest_path,
            "decoded_weights": decoded_weights,
            "rotation": {"matrix": rotation["matrix"].contiguous(),
                         "signs": rotation["signs"].contiguous()},
        }
        self._artifact_cache[key] = loaded
        return loaded

    def _load_vq_artifact(self, path: Path) -> dict[str, Any]:
        torch = _torch()
        key = self._cache_key(VQ_ARM, path)
        if key in self._artifact_cache:
            return self._artifact_cache[key]
        _, micro, _, _, _ = _helpers()
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("format") != VQ_FORMAT
                or manifest.get("target_modules") != target_manifest(self.modules)):
            raise RuntimeError("VQ artifact format or target-module manifest differs")
        entries = {row.get("name"): row for row in manifest.get("modules", [])}
        if set(entries) != set(self.modules):
            raise RuntimeError("VQ artifact module names differ from the fixed target scope")
        encodings, transforms = {}, {}
        for name, module in self.modules.items():
            row = entries[name]
            filename = row.get("file")
            if not isinstance(filename, str) or Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise RuntimeError(f"VQ artifact module path is invalid for {name}")
            payload = torch.load(path / filename, map_location="cpu", weights_only=True)
            if (payload.get("name") != name
                    or payload.get("shape") != [module.out_features, module.in_features]):
                raise RuntimeError(f"VQ artifact module identity mismatch for {name}")
            encoding, transform = payload["encoding"], payload["transform"]
            micro.check_encoding(encoding)
            if transform.get("input_dim") != module.in_features or transform.get("block") != 128:
                raise RuntimeError(f"VQ H128 transform metadata mismatch for {name}")
            encodings[name], transforms[name] = encoding, transform
        decoded_weights = {
            name: micro.decode_vq(encoding).detach().cpu().contiguous()
            for name, encoding in encodings.items()
        }
        loaded = {"metadata": dict(manifest["install_metadata"]), "manifest": manifest,
                  "manifest_path": manifest_path, "encodings": encodings,
                  "transforms": transforms, "decoded_weights": decoded_weights}
        self._artifact_cache[key] = loaded
        return loaded

    def set_arm(self, arm: str, artifact_dir: str | Path | None = None) -> dict[str, Any]:
        if arm not in (BF16_ARM, SPIN_ARM, VQ_ARM):
            raise ValueError(f"Unknown quantization arm: {arm}")
        self._restore_current()
        if arm == BF16_ARM:
            self.arm = arm
            self.metadata = self._bf16_metadata()
            return self.metadata
        torch = _torch()
        self._backup_weights()
        path = self._artifact_path(arm, artifact_dir)
        if arm == SPIN_ARM:
            rotation, _, _, _, _ = _helpers()
            artifact = self._load_spin_artifact(path)
            decoded_weights = artifact["decoded_weights"]
            device_rotations = {}
            for name, module in self.modules.items():
                device_key = (str(path.expanduser().resolve()), "video", str(module.weight.device))
                if device_key not in self._device_rotation_cache:
                    cpu_rotation = artifact["rotation"]
                    self._device_rotation_cache[device_key] = {
                        "matrix": cpu_rotation["matrix"].to(
                            device=module.weight.device, dtype=torch.float32
                        ).contiguous(),
                        "signs": cpu_rotation["signs"].to(
                            device=module.weight.device, dtype=torch.float32
                        ).contiguous(),
                    }
                device_rotations[str(module.weight.device)] = self._device_rotation_cache[device_key]
                module.weight.data.copy_(decoded_weights[name].to(
                    device=module.weight.device, dtype=torch.bfloat16
                ))
                module.__dict__["_cosmos_spin_state"] = {
                    **device_rotations[str(module.weight.device)],
                    "rotate": rotation._rotate_with_matrix,
                    "a4_dequant": rotation._reference_a4_dequant,
                }
                module.forward = types.MethodType(_decoded_spin_linear_forward, module)
            self._spin_state = artifact
            self.metadata = dict(artifact["metadata"])
            self.metadata.update({
                "execution_scope": "decoded_bf16_numerical_comparison",
                "native_int4_kernel_used": False,
                "weight_format": "packed_int4_group128_bf16_scales",
                "activation_format": "dynamic_symmetric_int4_per_token_qmax7_ste",
                "transform": "learned_shared_cayley_r128",
                "artifact_dir": str(path),
                "artifact_manifest": str(artifact["deployment_manifest"]),
                "packed_bank_manifest": str(artifact["packed_bank_manifest"]),
                "inference_weight_operand_dtype": "torch.bfloat16",
                "inference_activation_operand_dtype": "torch.bfloat16",
                "inference_path": "decode_exported_packed_w4_g128_once_then_bf16_linear",
            })
        else:
            _, micro, _, _, _ = _helpers()
            artifact = self._load_vq_artifact(path)
            self._vq_state = vq_train_state(
                self.modules, artifact["encodings"], artifact["transforms"], trainable=False,
                predecoded_weights=artifact["decoded_weights"],
            )
            self.metadata = dict(artifact["metadata"])
            self.metadata.update({
                "execution_scope": "decoded_bf16_numerical_comparison",
                "native_int4_kernel_used": False,
                "weight_format": "m2_k256_d4_bf16_books_b32_uint8_exponents",
                "activation_format": "dynamic_symmetric_int4_per_token_qmax7_ste",
                "transform": "smooth05_batched_h128_v2",
                "artifact_dir": str(path),
                "artifact_manifest": str(artifact["manifest_path"]),
                "inference_weight_operand_dtype": "torch.bfloat16",
                "inference_activation_operand_dtype": "torch.bfloat16",
                "inference_path": "fixed_decoded_bf16_weights_and_bf16_a4_linear",
            })
        self.arm = arm
        return self.metadata

    def close(self):
        self._restore_current()


def save_vq_deployment(directory: str | Path, modules: Mapping[str, Any], bank_state,
                       size_report: dict[str, Any], *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write one immutable per-Linear VQ bank plus a small target manifest."""
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bank_dir = directory / "bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    rows = target_manifest(modules)
    modules_payload = bank_state.deployment()
    entries = []
    tensor_bytes = 0
    for index, row in enumerate(rows):
        name = row["name"]
        filename = f"linear_{index:04d}.pt"
        payload = {"name": name, "shape": row["shape"], **modules_payload[name]}
        payload = {
            key: ({field: value.detach().cpu().contiguous() if torch.is_tensor(value) else value
                   for field, value in group.items()} if isinstance(group, dict) else group)
            for key, group in payload.items()
        }
        temporary = bank_dir / (filename + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, bank_dir / filename)
        tensor_bytes += sum(value.numel() * value.element_size()
                            for group in (payload["encoding"], payload["transform"])
                            for value in group.values() if torch.is_tensor(value))
        entries.append({**row, "file": f"bank/{filename}"})
    install_metadata = {
        "arm": VQ_ARM, "execution": "decoded_bf16_fake_a4",
        "execution_scope": "decoded_bf16_numerical_comparison", "native_int4_kernel_used": False,
        "weight_format": "m2_k256_d4_bf16_books_b32_uint8_exponents",
        "activation_format": "dynamic_symmetric_int4_per_token_qmax7_ste",
        "weight_bits": 4, "activation_bits": 4, "target_module_count": len(rows),
        "target_weight_numel": sum(row["weight_numel"] for row in rows),
        "bpw_match_relative_delta": size_report["same_module_relative_bpw_delta"],
        "transform": "smooth05_batched_h128_v2",
        "transform_details": {"name": "smooth05_hadamard", "block": 128,
                              "implementation": "batched_h128_v2"},
        "scale_block": 32, "books": "2xK256xd4_BF16", "fixed_indices": True,
        "fixed_scale_exponents": True, "same_module_effective_bpw": size_report["micro_vq"]["effective_bpw"],
        "whole_dit_effective_bpw": size_report.get("whole_dit", {}).get("micro_vq_effective_bpw"),
        "tensor_payload_bytes": tensor_bytes, "artifact_dir": str(directory),
        **dict(metadata or {}),
    }
    manifest = {"format": VQ_FORMAT, "arm": VQ_ARM, "target_modules": rows,
                "modules": entries, "install_metadata": install_metadata,
                "size_report": size_report}
    manifest_path = directory / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, manifest_path)
    install_metadata["serialized_artifact_bytes"] = sum(
        file.stat().st_size for file in bank_dir.rglob("*") if file.is_file()) + manifest_path.stat().st_size
    return install_metadata


def save_spin_deployment(directory: str | Path, modules: Mapping[str, Any], rotation_state,
                         aliases: Mapping[str, Any], size_report: dict[str, Any], *,
                         seed: int, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Write the learned shared R128 plus the existing packed W4/scales reference bank."""
    import torch
    rotation, _, _, _, _ = _helpers()

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    bank_dir = directory / "packed_bank"
    rotation.export_native_bank(aliases, rotation_state, bank_dir,
                                metadata={"arm": "spinquant_adapted_w4a4"})
    probe = _spin_training_vs_deployed_probe(
        modules, aliases, rotation_state, bank_dir, seed=seed
    )
    state = {key: value.detach().cpu().clone() for key, value in rotation_state.state_dict().items()}
    install_metadata = {
        "arm": SPIN_ARM, "execution": "decoded_bf16_fake_a4",
        "execution_scope": "decoded_bf16_numerical_comparison", "native_int4_kernel_used": False,
        "weight_format": "packed_int4_group128_bf16_scales",
        "activation_format": "dynamic_symmetric_int4_per_token_qmax7_ste",
        "weight_bits": 4, "activation_bits": 4, "target_module_count": len(modules),
        "target_weight_numel": sum(module.weight.numel() for module in modules.values()),
        "bpw_match_relative_delta": size_report["same_module_relative_bpw_delta"],
        "transform": "learned_shared_cayley_r128",
        "transform_details": {"name": "spinquant_adapted_w4a4", "block": 128,
                              "implementation": "learned_cayley_r128"},
        "same_module_effective_bpw": size_report["spinquant"]["effective_bpw"],
        "whole_dit_effective_bpw": size_report.get("whole_dit", {}).get("spinquant_effective_bpw"),
        "training_vs_deployed_probe": probe,
        "artifact_dir": str(directory), **dict(metadata or {}),
    }
    deployment = {"format": SPIN_FORMAT, "seed": int(seed), "rotation_state": state,
                  "target_modules": target_manifest(modules), "install_metadata": install_metadata,
                  "size_report": size_report, "training_updates": 200}
    temporary = directory / "deployment.pt.tmp"
    torch.save(deployment, temporary)
    os.replace(temporary, directory / "deployment.pt")
    install_metadata["packed_bank_bytes"] = sum(file.stat().st_size for file in bank_dir.rglob("*") if file.is_file())
    return install_metadata


def _spin_training_vs_deployed_probe(modules, aliases, rotation_state, bank_dir, *, seed):
    """Compare a small layer's FP32 STE training path with exported BF16 inference."""
    torch = _torch()
    rotation, _, _, _, _ = _helpers()
    bank_manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = {entry["name"]: entry for entry in bank_manifest["modules"]}
    name = min(modules, key=lambda item: (modules[item].weight.numel(), item))
    alias = "video_expert." + name
    if alias not in entries:
        raise RuntimeError(f"SpinQuant probe layer is missing from packed bank: {alias}")
    payload = torch.load(bank_dir / entries[alias]["file"], map_location="cpu", weights_only=True)
    module = modules[name]
    deployed_weight = _decode_spin_bank_weight(
        payload, (module.out_features, module.in_features)
    ).to(device=module.weight.device, dtype=torch.bfloat16)
    stream = rotation_state.stream_for(alias)
    matrix, signs = rotation_state.matrix(stream), rotation_state.streams[stream].signs
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 9173)
    probe_input = torch.randn((2, module.in_features), generator=generator, dtype=torch.float32)
    probe_input = probe_input.to(device=module.weight.device, dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast(device_type=probe_input.device.type, enabled=False):
        training_output = module(probe_input)
        rotated = rotation._rotate_with_matrix(probe_input, matrix, signs)
        deployed_activation = rotation._reference_a4_dequant(rotated).to(torch.bfloat16)
        deployed_output = torch.nn.functional.linear(
            deployed_activation, deployed_weight, module.bias
        )
    delta = training_output.float() - deployed_output.float()
    rmse = delta.square().mean().sqrt()
    reference_rms = training_output.float().square().mean().sqrt().clamp_min(
        torch.finfo(torch.float32).eps
    )
    return {
        "module": name, "probe": "deterministic_synthetic_bf16_input",
        "input_shape": list(probe_input.shape),
        "training_activation_operand_dtype": "torch.float32 (STE A4 dequant)",
        "training_weight_operand_dtype": "torch.float32 (STE W4 dequant)",
        "deployed_activation_operand_dtype": str(deployed_activation.dtype),
        "deployed_weight_operand_dtype": str(deployed_weight.dtype),
        "output_dtype": str(deployed_output.dtype),
        "max_abs_error": float(delta.abs().max()), "rmse": float(rmse),
        "relative_rmse": float(rmse / reference_rms),
        "claim": "measured probe only; training FP32 STE and deployed BF16 operands are not asserted bitwise equal",
    }

