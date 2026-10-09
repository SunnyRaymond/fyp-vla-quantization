"""Run the frozen Smooth/Hadamard Scalar bank with the existing native S4 GEMM."""
import json
import math
from pathlib import Path
import types

import torch
from torch import nn
import torch.nn.functional as F

import smooth_vq as core
import quantization as native

ARM = "smooth05_hadamard_scalar_w4a4"
SOURCE_RECIPE = "smooth05_hadamard_scalar_a4"


def transform_input(value, scale, signs):
    """Batch independent H128 blocks; preserve the source FP32 butterfly order."""
    width = value.shape[-1]
    scaled = value.float() / scale
    full = width // 128 * 128
    pieces = []
    if full:
        blocks = (scaled[..., :full] * signs[:full]).reshape(*value.shape[:-1], full // 128, 128)
        pieces.append(core._fwht(blocks).reshape(*value.shape[:-1], full))
    if full < width:
        pieces.append(core._right_r(scaled[..., full:], signs[full:], True, 128))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-1)


def native_packing(encoded, device):
    """Re-layout original nibbles with zero-padded rows; never re-quantize W."""
    n, k = map(int, encoded["shape"])
    if encoded.get("kind") != "scalar" or encoded.get("group") != 128:
        raise ValueError("Expected the original signed Scalar W4 G128 encoding")
    packed = encoded["packed"]
    scales = encoded["scales"]
    padded_k = math.ceil(k / 128) * 128
    if packed.dtype != torch.uint8 or packed.numel() != (n * k + 1) // 2:
        raise ValueError("Invalid original packed weight shape or dtype")
    if scales.dtype != torch.bfloat16 or tuple(scales.shape) != (n, padded_k // 128):
        raise ValueError("Invalid original BF16 group scales")
    if k == padded_k:
        result = packed.reshape(n, k // 2)
    else:
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten()[:n * k].reshape(n, k)
        codes = F.pad(codes, (0, padded_k - k), value=0)
        result = native._pack_nibbles(codes)
    return result.to(device=device).contiguous(), scales.to(device=device).contiguous(), padded_k


def _forward(module, value):
    owner = module._quantization_owner
    if not owner._active:
        raise RuntimeError("Scalar native Linear was called while quantization is disabled")
    transformed = transform_input(value, module._smooth_scale, module._smooth_signs)
    # Preserve FP32 transformed values through A4 encoding, as in the source recipe.
    return owner._linear(module, transformed)


def install(quantization, bank_path):
    bank_path = Path(bank_path)
    receipt = json.loads((bank_path / "receipt.json").read_text())
    full = bank_path.parent.parent
    protocol = json.loads((full / "protocol.json").read_text())
    winners = json.loads((full / "frozen_winners.json").read_text())
    config = next(c for c in protocol["phase1_configs"] if c["name"] == "smooth05_hadamard")
    if (receipt.get("config") != "smooth05_hadamard" or receipt.get("kind") != "scalar"
            or winners["winners"]["scalar"] != "smooth05_hadamard"
            or config["alpha"] != 0.5 or config["hadamard"] is not True
            or protocol["hadamard_block"] != 128 or protocol["transform_seed"] != 20261006):
        raise ValueError("The frozen Scalar Smooth alpha=0.5 + Hadamard bank is mismatched")
    if len(quantization.modules) != 614 or receipt["target_module_count"] != 614:
        raise ValueError("Expected the original 614 target Linears")
    quantization.disable()
    if quantization.weights_converted:
        raise RuntimeError("Install the original Scalar bank on the original BF16 pipeline")
    for handle in quantization._bf16_hooks:
        handle.remove()
    quantization._bf16_hooks.clear()
    original_forwards = {}
    with torch.inference_mode():
        for index, (name, module) in enumerate(quantization.modules.items()):
            bank = torch.load(bank_path / ("%04d.pt" % index), map_location="cpu", weights_only=True)
            if bank["module"] != name or tuple(bank["encoded"]["shape"]) != tuple(module.weight.shape):
                raise ValueError("Original Scalar module identity/shape mismatch: " + name)
            transform = bank["transform"]
            if (transform["hadamard"] is not True or transform["block"] != 128
                    or transform["input_dim"] != module.in_features
                    or transform["scale"].shape != (module.in_features,)
                    or transform["signs"].shape != (module.in_features,)
                    or not torch.isfinite(transform["scale"]).all()
                    or not ((transform["scale"] >= 1 / 16) & (transform["scale"] <= 16)).all()):
                raise ValueError("Original Smooth/Hadamard transform is invalid: " + name)
            device = module.weight.device
            packed, scales, padded_k = native_packing(bank["encoded"], device)
            module.register_parameter("weight", None)
            module.register_buffer("_quant_packed_weight", packed, persistent=False)
            module.register_buffer("_quant_weight_scales", scales, persistent=False)
            module.register_buffer("_quant_weight_shape", torch.tensor(bank["encoded"]["shape"], device=device), persistent=False)
            module.register_buffer("_smooth_scale", transform["scale"].to(device=device, dtype=torch.float32), persistent=False)
            module.register_buffer("_smooth_signs", transform["signs"].to(device=device, dtype=torch.float32), persistent=False)
            module._quant_padded_k = padded_k
            module._quantization_owner = quantization
            original_forwards[name] = module.forward
            module.forward = types.MethodType(_forward, module)
            if index % 100 == 0:
                print("SCALAR_BANK_LOAD modules=%d/614" % (index + 1), flush=True)
    quantization._scalar_original_forwards = original_forwards
    quantization._weight_state = "w4"
    quantization._integer_gemm_calls = quantization._native_int4_gemm_calls = 0
    metadata = {"arm": ARM, "source_recipe": SOURCE_RECIPE, "module_count": 614,
                "source_bank": str(bank_path), "smooth_alpha": 0.5, "transform_seed": 20261006,
                "transforms_reused": True, "weight_codes_reused": True,
                "new_calibration_cases": 0, "new_training_steps": 0,
                "native_arithmetic": "original W4 codes/BF16 group scales; FP32 transformed-input A4 encoding; native S4 MMA with group-scaled FP32 accumulation then BF16 output"}
    print("SCALAR_BANK_READY " + json.dumps(metadata), flush=True)
    return metadata


def self_check(owner):
    generator = torch.Generator(device="cuda").manual_seed(20261009)
    errors = []
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for width in (7, 128, 257):
            value = torch.randn((2, 16, width), generator=generator, device="cuda", dtype=torch.bfloat16)
            transform = {"scale": torch.exp(torch.randn(width, generator=generator, device="cuda").clamp(-2, 2)),
                         "signs": (torch.randint(0, 2, (width,), generator=generator, device="cuda") * 2 - 1).float(),
                         "hadamard": True, "block": 128, "input_dim": width}
            actual = transform_input(value, transform["scale"], transform["signs"])
            expected = core.transform_input(value, transform)
            assert torch.equal(actual, expected), "Batched transform changed source FP32 arithmetic"
            module = nn.Linear(width, 17, device="cuda", dtype=torch.bfloat16)
            encoded = core.scalar_quantize(core.transform_weight(module.weight, transform))
            packed, scales, padded_k = native_packing(encoded, "cuda")
            decoded_codes = native._unpack_nibbles(packed, width).float()
            wscale = scales.float().repeat_interleave(128, dim=1)[:, :width]
            weight = decoded_codes * wscale
            assert torch.equal(weight.to(torch.bfloat16), core.decode_scalar_quantized(encoded["packed"], encoded["scales"], encoded["shape"], 128))
            module.register_buffer("_quant_packed_weight", packed)
            module.register_buffer("_quant_weight_scales", scales)
            module._quant_padded_k = padded_k
            codes, a_scale = native._quantize_activation(actual, 4)
            reference = F.linear(codes.float() * a_scale[:, None], weight, module.bias.float()).reshape(2, 16, 17).to(torch.bfloat16)
            output = owner._linear(module, actual)
            error = float((output.float() - reference.float()).abs().max())
            assert torch.allclose(output.float(), reference.float(), rtol=0.02, atol=0.02), "Native fixture mismatch"
            errors.append(error)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    summary = owner.summary()
    assert summary["native_int4_tensorcore"] is True and summary["native_int4_ptx_verified"] is True
    return {"source_transform_bitwise_equal": True, "source_weight_decode_bitwise_equal": True,
            "native_fixture_widths": [7, 128, 257], "native_fixture_max_abs_errors": errors,
            "native_int4_ptx_verified": True}


def validate_runtime(runtime, metadata):
    import run_diagnostic as source
    source_root = Path("/scratch/users/ntu/yguo017/fastwam-a4-phases12-20261006")
    full = Path(metadata["source_bank"]).parent.parent
    check = self_check(runtime.q)
    plan = json.loads((source_root / "plan.json").read_text())
    row = source.validate_plan(plan, 1)
    observation, observation_meta = source.load_case(source_root, row)
    runtime.description = observation_meta["description"]
    seed = row["sampler_seeds"][0]
    reference = None
    with (full / "actions.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            if record["case_id"] == 1 and record["sampler_seed"] == seed and record["arm"] == SOURCE_RECIPE:
                reference = torch.tensor(record["action"], dtype=torch.float32)
                break
    if reference is None:
        raise RuntimeError("Original frozen Scalar reference action is unavailable")
    action, _ = runtime.infer(runtime.datum(observation), seed, measure=False)
    delta = action.detach().float().cpu() - reference
    check["fixed_input_diagnostic"] = {"case_id": 1, "sampler_seed": seed,
        "source_arm": SOURCE_RECIPE, "episodes": 0,
        "action_rmse_32_native_vs_source_decoded_bf16": float(delta.square().mean().sqrt()),
        "motor_rmse_first10_native_vs_source_decoded_bf16": float(delta[:10, :6].square().mean().sqrt()),
        "scope": "Reported numerical drift; original decoded-BF16 execution differs from native FP32 group arithmetic. No gate waiver inherited."}
    runtime.arm_calls.clear()
    runtime.calls = 0
    print("SCALAR_NATIVE_SETUP_CHECK " + json.dumps(check), flush=True)
    return check


def close(quantization):
    quantization.disable()
    for name, forward in getattr(quantization, "_scalar_original_forwards", {}).items():
        module = quantization.modules[name]
        module.forward = forward
        for key in ("_quant_packed_weight", "_quant_weight_scales", "_quant_weight_shape", "_smooth_scale", "_smooth_signs"):
            module._buffers.pop(key, None)
        for key in ("_quantization_owner", "_quant_padded_k"):
            if hasattr(module, key):
                delattr(module, key)
    if hasattr(quantization, "_scalar_original_forwards"):
        delattr(quantization, "_scalar_original_forwards")
