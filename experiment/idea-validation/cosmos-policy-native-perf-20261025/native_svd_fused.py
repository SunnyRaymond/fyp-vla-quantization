"""Candidate adapter for the frozen SVDQuant bank with Nunchaku LoRA fusion.

The residual A4 activation path remains the existing whole-K symmetric recipe.
Nunchaku's down helper performs an additional groupwise A4 quantization whose
packed activation and scales are intentionally discarded; only its official
FP32 LoRA activation is consumed by the GEMM up epilogue.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys


NUNCHAKU_COMMIT = "302e0e97024ebd68688fe890e5df83731edf7b54"
DEEPCOMPRESSOR_COMMIT = "69f3473f5e1c1504bae35cc50c7858ef900a9b17"
ADAPTER_VERSION = "cosmos_nunchaku_w4a4_g128_rank16_lora_fused_candidate_v1"
PACKING_SOURCE = (
    "DeepCompressor NunchakuWeightPacker.pack_lowrank_weight at "
    "deepcompressor/backend/nunchaku/utils.py:147-175, commit "
    + DEEPCOMPRESSOR_COMMIT
    + "; Nunchaku kernel commit " + NUNCHAKU_COMMIT
)


def _diffusion():
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import native_diffusion
    return native_diffusion


def _extension(artifact_dir=None):
    name = "nunchaku_svdq_fused_native"
    here = Path(__file__).resolve().parent
    bases = [here]
    if artifact_dir is not None:
        bases.insert(0, Path(artifact_dir).resolve())
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    directories = [base / "native_extensions" / "svdquant_w4a4_fused" for base in bases]
    binary = next((path for directory in directories if directory.is_dir()
                   for path in directory.glob(name + "*")
                   if path.is_file() and path.name.endswith(suffixes)), None)
    if binary is None:
        raise RuntimeError(
            f"Native extension {name} is missing; build it with build_svd_fused.py in CPU PBS"
        )
    cached = sys.modules.get(name)
    if cached is not None:
        if Path(getattr(cached, "__file__", "")).resolve() != binary.resolve():
            raise RuntimeError(f"A different {name} binary is already loaded")
        return cached
    spec = importlib.util.spec_from_file_location(name, binary)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native fusion extension: {binary}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _fusion_shape_contract(batch: int, out_features: int, padded_m: int,
                           padded_n: int, padded_k: int, rank: int = 16) -> dict:
    if batch <= 0 or padded_m != ((batch + 255) // 256) * 256:
        raise ValueError("Nunchaku fusion M must be the true batch padded to 256 rows")
    if out_features <= 0 or padded_n != ((out_features + 127) // 128) * 128:
        raise ValueError("Nunchaku fusion N must be padded to 128 columns")
    if padded_k <= 0 or padded_k % 128:
        raise ValueError("Nunchaku fused down input K must be padded to a multiple of 128")
    if rank != 16:
        raise ValueError("This candidate preserves the frozen SVDQuant rank-16 branch")
    return {"activation": (padded_m, padded_k), "down": (padded_k, rank),
            "up": (padded_n, rank), "output": (batch, out_features)}


def _fp16_factor(weight, torch, label: str):
    if not bool(torch.isfinite(weight).all().item()):
        raise RuntimeError(f"Frozen SVDQuant {label} factor contains nonfinite BF16 values")
    converted = weight.to(dtype=torch.float16).contiguous()
    if not bool(torch.isfinite(converted).all().item()):
        raise RuntimeError(f"Frozen SVDQuant {label} factor overflows during runtime FP16 conversion")
    return converted


def _pack_lowrank_weight(weight, *, down: bool, padded_n: int, padded_k: int,
                         functional, torch):
    """Reproduce the official Nunchaku low-rank lane/pack permutation.

    Logical down is [R,K], logical up is [N,R].  For Nunchaku's 4-bit
    GEMM_W4A4_FP16 geometry, pack_n=16 and pack_k=16.
    """
    pack_n, pack_k = 16, 16
    if weight.dtype != torch.float16 or not weight.is_cuda:
        raise ValueError("Nunchaku fused LoRA factors must be CUDA FP16")
    if not weight.is_contiguous() or weight.dim() != 2:
        raise ValueError("Nunchaku fused LoRA factors must be contiguous matrices")
    if down:
        rank, columns = map(int, weight.shape)
        if rank != 16 or columns > padded_k:
            raise ValueError("Frozen LoRA down factor must be [16,K] within padded K")
        logical = functional.pad(weight, (0, padded_k - columns), value=0)
        r, c = map(int, logical.shape)
        if r % pack_n or c % pack_k:
            raise ValueError("Nunchaku down factor padding does not align to pack tiles")
        r_packs, c_packs = r // pack_n, c // pack_k
        packed = logical.view(r_packs, pack_n, c_packs, pack_k).permute(2, 0, 1, 3)
    else:
        rows, rank = map(int, weight.shape)
        if rank != 16 or rows > padded_n:
            raise ValueError("Frozen LoRA up factor must be [N,16] within padded N")
        logical = functional.pad(weight, (0, 0, 0, padded_n - rows), value=0)
        c, r = map(int, logical.shape)
        if c % pack_n or r % pack_k:
            raise ValueError("Nunchaku up factor padding does not align to pack tiles")
        c_packs, r_packs = c // pack_n, r // pack_k
        packed = logical.view(c_packs, pack_n, r_packs, pack_k).permute(0, 2, 1, 3)
    packed = packed.reshape(c_packs, r_packs, 2, 8, 1, 2, 4, 2)
    packed = packed.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
    return packed.view(c, r)


def install_native(runtime, p, old_root, arm="svdquant_w4a4", artifact_dir=None,
                   backend=None):
    """Install the candidate native adapter and return metadata plus restore hook."""
    import torch
    import torch.nn.functional as functional

    if arm != "svdquant_w4a4":
        raise ValueError("SVD fused candidate supports only the frozen svdquant_w4a4 arm")
    if not torch.cuda.is_available():
        raise RuntimeError("Native SVDQuant fusion requires the approved CUDA runtime")
    base = _diffusion()
    root = Path(old_root).resolve()
    baseline = base._load_baseline(root)
    protocol = base._legacy_protocol(root, p)
    directory = base._bank_directory(root, arm, artifact_dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    modules = dict(sorted(runtime.target_linears.items()))
    rows = baseline._target_rows(modules)
    if (manifest.get("status") != "complete" or manifest.get("arm") != arm
            or manifest.get("protocol") != protocol.get("protocol")
            or manifest.get("target_manifest") != rows
            or len(rows) != baseline.EXPECTED_MODULES):
        raise RuntimeError("Only the complete frozen SVDQuant bank for these 280 Linear modules can be installed")
    _, _, group, _ = baseline._target_config(protocol)
    if group != 128:
        raise RuntimeError("The frozen SVDQuant checkpoint must retain symmetric G128 weight scales")

    extension = _extension(artifact_dir)
    required = ("pack_activation", "pack_weight", "pack_activation_scales", "pack_weight_scales",
                "quantize_down", "matmul_lora_up")
    if any(not callable(getattr(extension, name, None)) for name in required):
        raise RuntimeError("Fused Nunchaku extension is missing a required native entry point")

    controller = getattr(runtime, "controller", None)
    prior_restore = getattr(runtime, "_cosmos_native_restore", None)
    if callable(prior_restore):
        prior_restore()
    if controller is None:
        raise RuntimeError("Native SVDQuant requires the controller's CPU BF16 restore shadow")
    controller._restore_current()
    if controller._base_weights_cpu is None:
        controller._backup_weights()
    if controller._base_weights_cpu is None or set(controller._base_weights_cpu) != set(modules):
        raise RuntimeError("Controller CPU BF16 shadow does not cover all 280 native targets")
    if any(module.weight is None or module.weight.dtype != torch.bfloat16
           or (module.bias is not None and module.bias.dtype != torch.bfloat16)
           for module in modules.values()):
        raise RuntimeError("Fusion installation requires restored BF16 target Linear modules")

    devices = {name: module.weight.device for name, module in modules.items()}
    old_forwards = {name: module.forward for name, module in modules.items()}

    def restore():
        base._restore_bf16_parameters(modules, controller, devices, torch)
        for name, module in modules.items():
            module.forward = old_forwards[name]
            module.__dict__.pop("_cosmos_native_arm", None)
        controller._restore_current()
        controller.arm = "bf16"
        controller.metadata = controller._bf16_metadata()
        runtime.current_arm = "bf16"
        runtime.quantization_evidence = controller.metadata
        runtime.__dict__.pop("_cosmos_native_restore", None)

    prepared = []
    try:
        for index, (name, module) in enumerate(modules.items()):
            shape = (int(module.weight.shape[0]), int(module.weight.shape[1]))
            entry = manifest["entries"][index]
            if entry.get("name") != name or tuple(entry.get("shape", ())) != shape:
                raise RuntimeError(f"SVDQuant manifest target differs at module {index}: {name}")
            bank_path = base._module_bank_path(directory, entry, index)
            bank = baseline._read_bank(bank_path, arm, name, index, shape, group, torch)
            bank = baseline._device_tree(bank, module.weight.device, torch)
            rank = int(bank.get("rank", 0))
            if rank != 16 or tuple(bank["up"].shape) != (shape[0], rank) \
                    or tuple(bank["down"].shape) != (rank, shape[1]):
                raise RuntimeError(f"Frozen SVDQuant factor shapes/rank changed at {name}")

            core = base._nunchaku_bank(bank, module, extension, torch, functional, rank)
            padded_k, padded_n = int(core["padded_k"]), int(core["padded_n"])
            sidecar_k = ((padded_k + 127) // 128) * 128
            packed_up = _pack_lowrank_weight(
                _fp16_factor(bank["up"], torch, "up"), down=False,
                padded_n=padded_n, padded_k=padded_k, functional=functional, torch=torch)
            packed_down = _pack_lowrank_weight(
                _fp16_factor(bank["down"], torch, "down"), down=True,
                padded_n=padded_n, padded_k=sidecar_k, functional=functional, torch=torch)
            _fusion_shape_contract(1, shape[0], 256, padded_n, sidecar_k, rank)
            item = {key: core[key] for key in
                    ("packed_weight", "packed_wscales", "padded_k", "padded_n", "scales", "shape")}
            item.update({"lora_up": packed_up, "lora_down": packed_down,
                         "rank": rank, "sidecar_k": sidecar_k})
            prepared.append((name, module, item))
            del bank, core, packed_up, packed_down
            module.register_parameter("weight", None)

        for name, module, item in prepared:
            out_features = int(item["shape"][0])
            def forward(value, _module=module, _out_features=out_features, _item=item):
                original_shape = value.shape
                columns = int(original_shape[-1])
                if columns != int(_item["shape"][1]):
                    raise RuntimeError(f"SVDQuant fused input width changed at {_module}")
                flat = value.reshape(-1, columns)
                normalized = flat.float() / _item["scales"]
                scale = normalized.abs().amax(dim=-1, keepdim=True) / 7
                safe = torch.where(scale > 0, scale, torch.ones_like(scale))
                activation_codes = normalized.div(safe).round().clamp(-7, 7).to(torch.int8)
                if _item["padded_k"] > columns:
                    activation_codes = functional.pad(
                        activation_codes, (0, _item["padded_k"] - columns), value=0)
                batch = int(flat.shape[0])
                padded_m = ((batch + 255) // 256) * 256
                if padded_m != batch:
                    activation_codes = functional.pad(activation_codes,
                                                      (0, 0, 0, padded_m - batch), value=0)
                groups = _item["padded_k"] // 64
                a_scales = safe.to(torch.float16).reshape(1, batch).expand(groups, batch)
                if padded_m != batch:
                    a_scales = functional.pad(a_scales, (0, padded_m - batch), value=1)
                a_scales = extension.pack_activation_scales(a_scales.contiguous())
                packed_activation = extension.pack_activation(activation_codes.contiguous())

                _fusion_shape_contract(
                    batch, _out_features, padded_m, _item["padded_n"], _item["sidecar_k"], _item["rank"])
                sidecar_input = normalized.to(dtype=torch.float16)
                if _item["sidecar_k"] > columns:
                    sidecar_input = functional.pad(sidecar_input,
                                                    (0, _item["sidecar_k"] - columns), value=0)
                sidecar_input = sidecar_input.contiguous()
                lora_act = extension.quantize_down(sidecar_input, _item["lora_down"])
                if tuple(lora_act.shape) != (padded_m, _item["rank"]):
                    raise RuntimeError("Nunchaku quantize_down returned an unexpected padded LoRA activation shape")

                output_shape = base._nunchaku_output_shape(
                    batch, padded_m, _item["padded_n"], _out_features)
                output = torch.empty(output_shape, device=value.device, dtype=torch.float16)
                extension.matmul_lora_up(
                    packed_activation, _item["packed_weight"], output,
                    a_scales, _item["packed_wscales"], lora_act, _item["lora_up"])
                result = output[:, :_out_features].to(torch.bfloat16)
                if _module.bias is not None:
                    result = result + _module.bias
                return result.reshape(*original_shape[:-1], _out_features)

            module.forward = forward
            module._cosmos_native_arm = arm

        packed_bytes = base._unique_cuda_storage_bytes([item for _, _, item in prepared], torch)
        bias_bytes = base._unique_cuda_storage_bytes(
            [module.bias for module in modules.values() if module.bias is not None], torch)
        metadata = {
            "protocol": protocol.get("protocol"),
            "arm": arm,
            "adapter_version": ADAPTER_VERSION,
            "implementation": "nunchaku_official_w4a4_int4_rank16_lora_fused_candidate",
            "target_linears": len(modules),
            "target_weight_numel": manifest["target_weight_numel"],
            "weight_group_size": 128,
            "weight_scale_dtype": "torch.float16",
            "activation_bits": 4,
            "activation_recipe": "existing_per_token_whole_K_symmetric_A4",
            "native_int4_kernel_used": True,
            "weight_requantized": False,
            "weight_recalibrated": False,
            "new_training_updates": 0,
            "new_closed_loop_episodes": 0,
            "global_svd_rank": 16,
            "native_kernel": "Nunchaku official W4A4 INT4 GEMM plus official SVDQuant LoRA down/up kernels (candidate)",
            "lora_down_kernel": "quantize_w4a4_act_fuse_lora; only its FP32 LoRA activation is retained",
            "lora_up_kernel": "GEMM_W4A4_Launch lora_up epilogue with one scale 1.0",
            "sidecar_extra_groupwise_a4_quantization": True,
            "sidecar_groupwise_a4_outputs_discarded": ["packed_activation", "activation_scales"],
            "sidecar_input_recipe": "normalized activation (x / checkpoint Smooth), K padded to official down helper's 128-column tile, FP16 runtime cast, no extra Smooth",
            "lora_factor_packing": PACKING_SOURCE,
            "lora_factor_runtime_dtype": "FP16 converted from frozen BF16 factors; checkpoint tensors are unchanged",
            "lora_factor_fp16_numeric_difference": "FP16 conversion and fused FP32 LoRA accumulation differ from the prior decoded-BF16 two-GEMM reference; requires the predeclared layer/full80 gate",
            "original_weight_bank": str(manifest_path),
            "artifact_manifest": str(manifest_path),
            "execution": "packed_gpu_resident_native_int4_gemm_with_official_lora_fusion_candidate",
            "native_activation_scale_dtype": "torch.float16",
            "packed_gpu_resident": True,
            "packed_weight_gpu_resident": True,
            "bf16_target_weight_numel_resident": 0,
            "bf16_target_weight_numel_cuda_resident": 0,
            "bf16_restore_shadow_cpu_numel": sum(int(m.out_features * m.in_features)
                                                    for m in modules.values()),
            "packed_target_payload_bytes": packed_bytes,
            "packed_target_payload_scope": "unique CUDA tensor storage for packed INT4 residual, G64-expanded scales, checkpoint Smooth, and packed FP16 rank-16 factors; excludes transient activation/output",
            "bf16_target_bias_storage_bytes": bias_bytes,
            "native_resident_tensor_bytes": packed_bytes + bias_bytes,
            "reference_module_weights_retained": False,
            "native_extension": extension.__file__,
            "native_extension_source_commit": NUNCHAKU_COMMIT,
            "gpu_execution_tested": False,
            "adaptation_boundary": manifest.get("adaptation_boundary"),
            "candidate_only": True,
        }
        controller.arm = arm
        controller.metadata = dict(metadata)
        runtime.quantization_evidence = dict(metadata)
        runtime.current_arm = arm
        runtime._cosmos_native_restore = restore
        return metadata, restore
    except Exception:
        restore()
        raise
