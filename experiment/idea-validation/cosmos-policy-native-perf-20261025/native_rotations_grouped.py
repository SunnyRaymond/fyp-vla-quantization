"""Grouped INT4 rotation adapters using the public Nunchaku W4A4 core.

This adapter keeps the frozen SpinQuant/QuaRot checkpoint codes, scales, and
rotation transforms. It repacks each full-K signed INT4 matrix for Nunchaku's
public no-LoRA W4A4 kernel and repeats each checkpoint G128 scale over its two
adjacent G64 kernel groups. The G128 conversion and the rotation deployment
glue are task-local; this is not an official full Cosmos SpinQuant/QuaRot
inference engine.
"""
from __future__ import annotations

import json
from pathlib import Path
import types

import native_diffusion
import native_rotations


ADAPTER_VERSION = "cosmos_rotation_nunchaku_w4a4_g128_to_g64_v1"
NUNCHAKU_ARM = "svdquant_w4a4"
NUNCHAKU_SOURCE_COMMIT = "302e0e97024ebd68688fe890e5df83731edf7b54"
CHECKPOINT_GROUP_SIZE = 128
KERNEL_GROUP_SIZE = 64


def _group_repeat_count(checkpoint_group_size=CHECKPOINT_GROUP_SIZE,
                        kernel_group_size=KERNEL_GROUP_SIZE):
    if (checkpoint_group_size <= 0 or kernel_group_size <= 0
            or checkpoint_group_size % kernel_group_size):
        raise ValueError("checkpoint G128 must be an integer multiple of the kernel G64")
    return checkpoint_group_size // kernel_group_size


def _repeat_group_scales(scales, torch):
    """Map [N,G128] checkpoint scales to [N,G64] by exact adjacent repeats."""
    del torch  # Kept in the signature so the packing path is easy to fake-test.
    return scales.repeat_interleave(_group_repeat_count(), dim=1)


def _receipt_candidates(native_root: Path):
    return (
        native_root / "results" / "preparation" / "native_diffusion_build.json",
        native_root / "retrieved" / "preparation.native_diffusion_build.json",
    )


def _read_nunchaku_build(native_root: Path, extension):
    receipt_path = next((path for path in _receipt_candidates(native_root)
                         if path.is_file()), None)
    if receipt_path is None:
        raise FileNotFoundError(
            "Nunchaku grouped adapter requires results/preparation/native_diffusion_build.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    arm = receipt.get("arms", {}).get(NUNCHAKU_ARM, {})
    module_path = getattr(extension, "__file__", None)
    if receipt.get("status") != "complete" or not receipt.get("pbs_job_id"):
        raise RuntimeError("Nunchaku extension build receipt is incomplete or has no PBS producer")
    source_commit = arm.get("source_commit")
    if source_commit != NUNCHAKU_SOURCE_COMMIT:
        raise RuntimeError("Nunchaku build receipt is not bound to the pinned public-core source commit")
    if not module_path or not arm.get("module"):
        raise RuntimeError("Nunchaku extension and build receipt must identify the loaded module")
    if Path(module_path).resolve() != Path(arm["module"]).resolve():
        raise RuntimeError("Loaded Nunchaku extension differs from the recorded build module")
    if not callable(getattr(extension, "matmul", None)):
        raise RuntimeError("Nunchaku W4A4 extension has no callable matmul")
    packers = ("pack_activation", "pack_weight", "pack_activation_scales",
               "pack_weight_scales")
    if any(not callable(getattr(extension, name, None)) for name in packers):
        raise RuntimeError("Nunchaku W4A4 extension is missing packed INT4 conversion entry points")
    return receipt_path, receipt, arm


def _weight_scales_fp16(scales, arm, torch):
    """Preserve QuaRot FP16 scales and require exact Spin BF16->FP16 representation."""
    if not bool(torch.isfinite(scales).all().item()):
        raise RuntimeError("Frozen G128 weight scales contain non-finite values")
    if arm == native_rotations.QUAROT_ARM:
        if scales.dtype != torch.float16:
            raise RuntimeError("QuaRot checkpoint scales must remain their original FP16 values")
        return scales
    if arm != native_rotations.SPIN_ARM or scales.dtype != torch.bfloat16:
        raise RuntimeError("SpinQuant checkpoint scales must be frozen BF16 values")
    converted = scales.to(dtype=torch.float16)
    if not bool(torch.isfinite(converted).all().item()):
        raise RuntimeError("SpinQuant BF16 scales are not finite after FP16 conversion")
    if not torch.equal(converted.to(dtype=torch.bfloat16), scales):
        raise RuntimeError("SpinQuant BF16 scales are not exactly representable as FP16")
    return converted


def _unpack_signed_grouped_weights(weight_groups, torch):
    """Join frozen G128 nibble groups and decode signed two's-complement codes."""
    group_count = _weight_group_count(weight_groups)
    if group_count <= 0:
        raise RuntimeError("Frozen rotation checkpoint has no packed G128 weight groups")
    packed = torch.cat(tuple(weight_groups), dim=1).contiguous()
    unsigned = native_diffusion._unpack_int4(packed, torch).to(torch.int16)
    signed = torch.where(unsigned >= 8, unsigned - 16, unsigned).to(torch.int8)
    return signed.contiguous(), packed


def _weight_group_count(weight_groups):
    """Get a tuple/list or Tensor[G,N,K/2] group count without Tensor truthiness."""
    try:
        count = len(weight_groups)
    except TypeError as exc:
        raise RuntimeError("Frozen rotation checkpoint must expose packed G128 groups") from exc
    shape = getattr(weight_groups, "shape", None)
    if shape is not None and (len(shape) != 3 or int(shape[0]) != count):
        raise RuntimeError("Frozen rotation packed weights must have [G,N,K/2] shape")
    return count


def _registered_buffer(module, name):
    """Read a registered nn.Module buffer, with plain-attribute compatibility."""
    buffers = getattr(module, "_buffers", None)
    value = buffers.get(name) if hasattr(buffers, "get") else None
    if value is not None:
        return value
    return module.__dict__.get(name)


def _rotation_payload(module, arm, value, torch, hadamard_backend):
    if arm == native_rotations.SPIN_ARM:
        state = module.__dict__.get("_cosmos_spin_state")
        if not isinstance(state, dict) or not callable(state.get("rotate")):
            raise RuntimeError("SpinQuant learned R128 transform state is unavailable")
        return state["rotate"](
            value.to(dtype=module._native_activation_dtype), state["matrix"], state["signs"])
    if arm == native_rotations.QUAROT_ARM:
        signs = _registered_buffer(module, "_native_rotation_signs")
        if signs is None:
            raise RuntimeError("QuaRot frozen H128 signs are unavailable")
        return native_rotations._rotate_quarot(value, signs, torch, hadamard_backend)
    raise ValueError(f"unsupported rotation arm: {arm}")


def _grouped_forward(module, value, torch, functional, extension, arm,
                     hadamard_backend, evidence):
    context = module.__dict__.get("_native_grouped_context")
    if not isinstance(context, dict):
        raise RuntimeError("Grouped Nunchaku rotation context was removed")
    original_shape = value.shape
    if int(original_shape[-1]) != int(module.in_features):
        raise RuntimeError("Grouped native Linear received a mismatched input width")
    transformed = _rotation_payload(module, arm, value, torch, hadamard_backend)
    if arm == native_rotations.QUAROT_ARM:
        evidence["native_hadamard_kernel_used"] = True
        evidence["native_hadamard_calls"] += 1
        evidence["native_hadamard_input_rows"] += int(value.numel() // 128)
    rows = transformed.float().reshape(-1, module.in_features).contiguous()
    if rows.shape[0] == 0 or rows.shape[1] > context["padded_k"]:
        raise ValueError("Nunchaku A4 GEMM received an empty or over-wide Linear input")
    maximum = rows.abs().amax(dim=1, keepdim=True)
    activation_scales_fp32 = maximum / 7.0
    safe = torch.where(activation_scales_fp32 > 0, activation_scales_fp32,
                       torch.ones_like(activation_scales_fp32))
    activation_codes = torch.round(rows / safe).clamp(-7, 7).to(torch.int8)
    activation_codes = torch.where(activation_scales_fp32 > 0, activation_codes,
                                   torch.zeros_like(activation_codes))
    if context["padded_k"] != rows.shape[1]:
        activation_codes = functional.pad(
            activation_codes, (0, context["padded_k"] - int(rows.shape[1])), value=0)
    batch = int(activation_codes.shape[0])
    padded_m = ((batch + 255) // 256) * 256
    if padded_m != batch:
        activation_codes = functional.pad(
            activation_codes, (0, 0, 0, padded_m - batch), value=0)

    # Nunchaku requires one scale for each G64 group. Repeating the unchanged
    # whole-K per-token scale preserves the existing qmax7 activation codes.
    group_count = context["padded_k"] // KERNEL_GROUP_SIZE
    # All-zero rows have all-zero codes, so the kernel's finite identity scale
    # is an exactly equivalent representation without synchronizing the GPU.
    row_scales = torch.where(activation_scales_fp32 == 0,
                             torch.ones_like(activation_scales_fp32),
                             activation_scales_fp32).to(torch.float16)
    a_scales = row_scales.reshape(1, batch).expand(group_count, batch)
    if padded_m != batch:
        a_scales = functional.pad(a_scales, (0, padded_m - batch), value=1)
    a_scales = extension.pack_activation_scales(a_scales.contiguous())
    packed_activation = extension.pack_activation(activation_codes.contiguous())

    output_shape = native_diffusion._nunchaku_output_shape(
        batch, padded_m, context["padded_n"], module.out_features)
    output = torch.empty(output_shape, device=value.device, dtype=torch.float16)
    extension.matmul(packed_activation, context["packed_weight"], output,
                     a_scales, context["packed_weight_scales"])
    evidence["native_integer_gemm_calls"] += 1
    evidence["native_target_linear_calls"] += 1
    result = output[:, :module.out_features].to(torch.bfloat16)
    if module.bias is not None:
        result = result + module.bias
    return result.reshape(*original_shape[:-1], module.out_features)


def _unique_cuda_storage_bytes(values, torch):
    """Count unique CUDA storages, including aliases only once."""
    seen = set()

    def visit(value):
        if torch.is_tensor(value):
            if value.is_cuda:
                storage = value.untyped_storage()
                seen.add((str(value.device), int(storage.data_ptr()), int(storage.nbytes())))
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(values)
    return sum(size for _, _, size in seen)


def install_native(runtime, p, old_root, arm, artifact_dir=None, native_root=None,
                   backend=None):
    """Install a grouped Nunchaku W4A4 candidate and return (metadata, restore).

    `artifact_dir` is the existing SpinQuant or QuaRot artifact directory, as
    accepted by :func:`native_rotations.install_native`. `native_root` points
    to the root containing the CPU-built Nunchaku extension and build receipt;
    it defaults to this adapter's directory. The frozen artifact remains under
    `old_root`/`p` when `artifact_dir` is omitted.
    """
    if arm not in (native_rotations.SPIN_ARM, native_rotations.QUAROT_ARM):
        raise ValueError(f"unsupported grouped rotation arm: {arm}")
    import torch
    import torch.nn.functional as functional

    native_root = Path(native_root or Path(__file__).resolve().parent).resolve()
    extension = native_diffusion._extension(NUNCHAKU_ARM, native_root)
    receipt_path, build_receipt, build_arm = _read_nunchaku_build(native_root, extension)

    # Reuse the existing frozen artifact/rotation preparation before replacing
    # its split-G128 GEMMs with the public full-K Nunchaku kernel.
    rotation_metadata, restore_rotation = native_rotations.install_native(
        runtime, p, old_root, arm, artifact_dir=artifact_dir, backend=backend)
    modules = dict(sorted(runtime.target_linears.items()))
    hadamard_backend = None
    try:
        if arm == native_rotations.QUAROT_ARM:
            gemm_backend = native_rotations._load_backend(backend)
            hadamard_backend, _, _ = native_rotations._load_hadamard_backend(gemm_backend)
    except Exception:
        restore_rotation()
        raise

    prepared = []
    evidence = {
        "native_integer_gemm_calls": 0,
        "native_target_linear_calls": 0,
        "native_hadamard_kernel_used": False,
        "native_hadamard_calls": 0,
        "native_hadamard_input_rows": 0,
    }
    try:
        for name, module in modules.items():
            original = module.__dict__.get("_native_rotation_context")
            if not isinstance(original, dict) or original.get("arm") != arm:
                raise RuntimeError(f"frozen rotation context is missing for {name}")
            weight_groups = original.get("weight_groups")
            scales = original.get("scales")
            padded_k = int(original.get("padded_width", 0))
            if padded_k <= 0 or padded_k % CHECKPOINT_GROUP_SIZE:
                raise RuntimeError(f"frozen G128 padded K is invalid for {name}")
            if _weight_group_count(weight_groups) != padded_k // CHECKPOINT_GROUP_SIZE:
                raise RuntimeError(f"frozen G128 group count differs from padded K for {name}")
            weight_codes, joined_codes = _unpack_signed_grouped_weights(weight_groups, torch)
            del joined_codes
            out_features = int(module.out_features)
            padded_n = ((out_features + 127) // 128) * 128
            if padded_n != out_features:
                weight_codes = functional.pad(weight_codes,
                                              (0, 0, 0, padded_n - out_features), value=0)

            scale_fp16 = _weight_scales_fp16(scales, arm, torch)
            # [N,G128] -> [G64,N], duplicating each original scale exactly.
            scale_g64 = _repeat_group_scales(
                scale_fp16, torch).transpose(0, 1).contiguous()
            if padded_n != out_features:
                scale_g64 = functional.pad(scale_g64, (0, padded_n - out_features), value=1)
            packed_weight = extension.pack_weight(weight_codes.contiguous())
            packed_weight_scales = extension.pack_weight_scales(scale_g64.contiguous())

            # No BF16 weight or split-G128 code/scale copy may remain on GPU.
            original.pop("weight_groups", None)
            original.pop("scales", None)
            module._buffers.pop("_native_rotation_scales", None)
            del weight_codes, scales, scale_fp16, scale_g64

            prepared.append((name, module, {
                "packed_weight": packed_weight,
                "packed_weight_scales": packed_weight_scales,
                "padded_k": padded_k,
                "padded_n": padded_n,
                "out_features": out_features,
            }))
    except Exception:
        restore_rotation()
        raise

    # Delete the old per-G128 GPU codes/scales before installing the new
    # resident payload; only QuaRot signs or Spin's learned R128 state remain.
    try:
        for name, module, context in prepared:
            old_context = module.__dict__["_native_rotation_context"]
            old_context.pop("weight_groups", None)
            old_context.pop("scales", None)
            module._buffers.pop("_native_rotation_scales", None)
            module._native_rotation_context = {
                "arm": arm,
                "padded_width": context["padded_k"],
            }
            module._native_grouped_context = context

            def forward(current, value, _torch=torch, _functional=functional,
                        _extension=extension, _arm=arm, _hadamard=hadamard_backend):
                return _grouped_forward(current, value, _torch, _functional,
                                        _extension, _arm, _hadamard, evidence)

            module.forward = types.MethodType(forward, module)
    except Exception:
        for _, module, _ in prepared:
            module.__dict__.pop("_native_grouped_context", None)
        restore_rotation()
        raise

    packed_weight_bytes = _unique_cuda_storage_bytes(
        [item[2]["packed_weight"] for item in prepared], torch)
    packed_scale_bytes = _unique_cuda_storage_bytes(
        [item[2]["packed_weight_scales"] for item in prepared], torch)
    rotation_values = []
    biases = []
    for _, module, _ in prepared:
        if arm == native_rotations.SPIN_ARM:
            state = module.__dict__.get("_cosmos_spin_state", {})
            rotation_values.extend((state.get("matrix"), state.get("signs")))
        else:
            signs = _registered_buffer(module, "_native_rotation_signs")
            if signs is None:
                raise RuntimeError("QuaRot frozen H128 signs are unavailable for resident accounting")
            rotation_values.append(signs)
        if module.bias is not None:
            biases.append(module.bias)
    rotation_bytes = _unique_cuda_storage_bytes(rotation_values, torch)
    bias_bytes = _unique_cuda_storage_bytes(biases, torch)
    resident_payload_bytes = _unique_cuda_storage_bytes(
        [{"packed": [item[2]["packed_weight"] for item in prepared],
          "scales": [item[2]["packed_weight_scales"] for item in prepared],
          "rotation": rotation_values}], torch)
    resident_bytes = _unique_cuda_storage_bytes(
        [{"payload": [{"packed_weight": item[2]["packed_weight"],
                        "packed_scales": item[2]["packed_weight_scales"]}
                       for item in prepared],
          "rotation": rotation_values, "bf16_biases": biases}], torch)

    metadata = {
        "arm": arm,
        "adapter_version": ADAPTER_VERSION,
        "artifact_dir": rotation_metadata.get('artifact_dir'),
        "original_weight_bank": rotation_metadata.get('artifact_dir'),
        "execution": "public_nunchaku_w4a4_no_lora_full_k_grouped_g64_adapter",
        "native_backend": ("public Nunchaku W4A4 FP16-scale no-LoRA GEMM core with task-local "
                           "Cosmos G128-to-G64 repacking; not an official full SpinQuant/QuaRot engine"),
        "native_backend_module": getattr(extension, "__name__", "nunchaku_svdq_native"),
        "native_backend_module_path": str(Path(extension.__file__).resolve()),
        "native_backend_source_commit": NUNCHAKU_SOURCE_COMMIT,
        "native_backend_build_pbs_job_id": build_receipt["pbs_job_id"],
        "native_backend_build_host": build_receipt.get("host"),
        "native_backend_build_receipt": str(receipt_path.resolve()),
        "native_backend_upstream_sources": build_arm.get("upstream_sources", []),
        "native_backend_task_local_sources": build_arm.get("task_local_sources", []),
        "native_int4_kernel_used": True,
        "native_integer_gemm_calls_per_target_linear": 1,
        "native_integer_gemm_calls": evidence["native_integer_gemm_calls"],
        "native_target_linear_calls": evidence["native_target_linear_calls"],
        "packed_gpu_resident": True,
        "packed_weight_gpu_resident": True,
        "packed_activation_gpu_resident_during_gemm": True,
        "packed_gemm_input_dtype": "torch.uint8",
        "native_integer_gemm_output_dtype": "torch.float16",
        "checkpoint_weight_group_size": CHECKPOINT_GROUP_SIZE,
        "native_kernel_weight_group_size": KERNEL_GROUP_SIZE,
        "checkpoint_group_to_kernel_group_mapping": "each frozen G128 scale is repeated unchanged over its two adjacent G64 groups",
        "weight_scale_dtype_checkpoint": rotation_metadata.get("weight_scale_dtype"),
        "weight_scale_dtype_kernel": "torch.float16",
        "weight_group_size": KERNEL_GROUP_SIZE,
        "original_checkpoint_group_size": CHECKPOINT_GROUP_SIZE,
        "weight_scale_conversion": ("unchanged original QuaRot FP16 values" if arm == native_rotations.QUAROT_ARM
                                    else "SpinQuant BF16 converted only after exact finite FP16 round-trip check"),
        "activation_quantizer": "existing dynamic per-token symmetric signed INT4 qmax7 over the full K",
        "activation_scale_dtype_checkpoint": "torch.float32",
        "activation_scale_dtype_kernel": "torch.float16",
        "activation_scale_conversion": "FP32 per-token scale cast to FP16 for public Nunchaku core; GPU parity gate required",
        "zero_activation_row_scale": "uses scale 1 with all-zero codes, preserving the zero output",
        "bias_dtype": "torch.bfloat16",
        "output_dtype": "torch.bfloat16",
        "output_shape_contract": "M padded to 256 for packed A; true M x Npad output; N padded to 128 and cropped to true N",
        "rotation": rotation_metadata.get("rotation"),
        "rotation_artifact_source": rotation_metadata.get("rotation_artifact_source"),
        "native_hadamard_backend": rotation_metadata.get("native_hadamard_backend"),
        "native_hadamard_module": rotation_metadata.get("native_hadamard_module"),
        "native_hadamard_source_commit": rotation_metadata.get("native_hadamard_source_commit"),
        "native_hadamard_source_repository": rotation_metadata.get("native_hadamard_source_repository"),
        "native_hadamard_quarot_source_revision": rotation_metadata.get(
            "native_hadamard_quarot_source_revision"),
        "native_hadamard_submodule_path": rotation_metadata.get("native_hadamard_submodule_path"),
        "native_hadamard_build_receipt": rotation_metadata.get("native_hadamard_build_receipt"),
        "native_hadamard_build_producer": rotation_metadata.get("native_hadamard_build_producer"),
        "native_hadamard_build_pbs_job_id": rotation_metadata.get("native_hadamard_build_pbs_job_id"),
        "native_hadamard_build_host": rotation_metadata.get("native_hadamard_build_host"),
        "learned_spin_rotation_backend": ("existing learned R128 transform, unchanged" if
                                          arm == native_rotations.SPIN_ARM else None),
        "method_artifact_is_full_official_engine": False,
        "g128_to_g64_is_task_local_adapter": True,
        "packed_weight_tensor_bytes": packed_weight_bytes,
        "packed_weight_scale_tensor_bytes": packed_scale_bytes,
        "rotation_tensor_bytes": rotation_bytes,
        "packed_target_payload_bytes": resident_payload_bytes,
        "bf16_target_bias_tensor_bytes": bias_bytes,
        "native_resident_tensor_bytes": resident_bytes,
        "bf16_target_weight_numel_cuda_resident": 0,
        "gpu_operator_gate_status": "not_run_by_installer; requires approved GPU parity gate",
        "target_module_count": len(prepared),
        "new_training_updates": 0,
        "new_closed_loop_episodes": 0,
    }
    evidence.update(metadata)
    metadata = evidence
    runtime.current_arm = arm
    runtime.controller.arm = arm
    runtime.controller.metadata = metadata
    runtime.quantization_evidence = metadata
    runtime.native_kernel_stats = metadata

    def restore():
        for module in modules.values():
            module.__dict__.pop("_native_grouped_context", None)
        restore_rotation()

    runtime.restore_native_rotation = restore
    return metadata, restore
