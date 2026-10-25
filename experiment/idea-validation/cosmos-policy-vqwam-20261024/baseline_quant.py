"""Cosmos-adapted W4A4 quantization baselines with explicit decoded-BF16 scope."""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

ARMS = ("quarot_w4a4", "svdquant_w4a4", "viditq_w4a4")
FORMAT = "cosmos-adapted-linear-bank-v1"
IMPLEMENTATIONS = {
    "quarot_w4a4": "cosmos_adapted_random_local_h128_scalar_w4a4_v1",
    "svdquant_w4a4": "cosmos_adapted_smoothed_low_rank_residual_w4a4_v1",
    "viditq_w4a4": "cosmos_adapted_timestep_calibrated_mixed_w4_w8_v1",
}
EXPECTED_MODULES = 280
EXPECTED_WEIGHT_NUMEL = 1_761_607_680


def _torch():
    import torch
    return torch


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value, torch):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(_cpu_tree(value, torch), temporary)
    os.replace(temporary, path)


def _cpu_tree(value, torch):
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").contiguous()
    if isinstance(value, dict):
        return {key: _cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item, torch) for item in value)
    return value


def _target_config(protocol):
    quant = protocol.get("quantization", {})
    baselines = quant.get("baselines", {})
    target = float(quant.get("target_effective_bpw",
                             quant.get("target_bpw", protocol.get("target_bpw", 4.29))))
    tolerance = float(quant.get("target_bpw_tolerance",
                                quant.get("effective_bpw_max_relative_difference",
                                          quant.get("max_relative_target_bpw_delta", 0.05))))
    if not math.isfinite(target) or target <= 4.0 or not 0 < tolerance <= 0.10:
        raise ValueError("The frozen target BPW and tolerance are invalid")
    group = int(quant.get("group_size", baselines.get("group_size", 128)))
    if group != 128:
        raise ValueError("These Cosmos-adapted W4 baselines require G128")
    seed = int(quant.get("seed", protocol.get("seed", 20261024)))
    return target, tolerance, group, seed


def _method_config(protocol, arm):
    return protocol.get("methods", {}).get(arm, {})


def _baseline_config(protocol, arm):
    key = {"svdquant_w4a4": "svdquant", "viditq_w4a4": "viditq",
           "quarot_w4a4": "quarot"}[arm]
    return protocol.get("quantization", {}).get("baselines", {}).get(key, {})


def _alpha_candidates(protocol, arm):
    method = _method_config(protocol, arm)
    baseline_cfg = _baseline_config(protocol, arm)
    key_candidates = ("smoothing_alpha_candidates", "alpha_candidates", "smoothing_alphas")
    for key in key_candidates:
        if key in method:
            values = [float(value) for value in method[key]]
            if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
                raise ValueError(f"Invalid {arm} calibration alpha grid")
            return list(dict.fromkeys(values))
    key = "smoothing_candidates" if arm == "svdquant_w4a4" else "balance_candidates"
    if key in baseline_cfg:
        values = [float(value) for value in baseline_cfg[key]]
        if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError(f"Invalid {arm} calibration alpha grid")
        return list(dict.fromkeys(values))
    if arm == "svdquant_w4a4":
        return [0.0, 0.25, 0.5, 0.75, 1.0]
    return [0.35, 0.45, 0.5665, 0.65, 0.75]


def _rank_candidates(protocol):
    method = _method_config(protocol, "svdquant_w4a4")
    values = method.get("rank_candidates_descending", protocol.get(
        "svd_rank_candidates_descending",
        _baseline_config(protocol, "svdquant_w4a4").get("rank_candidates", [16, 8, 4, 2, 1])))
    ranks = sorted({int(value) for value in values if int(value) > 0}, reverse=True)
    if not ranks:
        raise ValueError("SVDQuant needs at least one positive global rank candidate")
    return ranks


def _shapes(modules):
    return [(name, (int(module.weight.shape[0]), int(module.weight.shape[1])))
            for name, module in sorted(modules.items())]


def _target_rows(modules):
    return [{"name": name, "shape": list(shape),
             "weight_numel": int(shape[0] * shape[1])}
            for name, shape in _shapes(modules)]


def _parameters(shapes):
    return sum(rows * columns for _, (rows, columns) in shapes)


def _padded_columns(columns, group):
    return ((columns + group - 1) // group) * group


def _base_bytes(shapes, arm, group):
    parts = {"weight_codes": 0, "scales": 0, "zero_points": 0,
             "smoothing": 0, "rotation_signs": 0, "low_rank_factors": 0}
    for _, (rows, columns) in shapes:
        padded = _padded_columns(columns, group)
        groups = rows * (padded // group)
        parts["weight_codes"] += rows * padded // 2
        parts["scales"] += groups * 2
        if arm == "viditq_w4a4":
            parts["zero_points"] += groups
            parts["smoothing"] += columns * 4
            parts["rotation_signs"] += columns
        elif arm == "svdquant_w4a4":
            parts["smoothing"] += columns * 4
    return parts


def _rank_bytes(shapes, rank):
    return sum(2 * min(rank, rows, columns) * (rows + columns)
               for _, (rows, columns) in shapes)


def _choose_svd_rank(shapes, params, group, target_bpw, tolerance, candidates):
    max_bytes = math.floor(target_bpw * (1 + tolerance) * params / 8)
    for rank in candidates:
        parts = _base_bytes(shapes, "svdquant_w4a4", group)
        estimate = sum(parts.values()) + _rank_bytes(shapes, rank)
        if estimate <= max_bytes:
            return rank, estimate
    raise RuntimeError("No positive SVD rank fits the frozen effective-BPW ceiling")


def _stats_rows(entry, width, torch, device=None):
    def decode(row):
        # Statistics are loaded from CPU artifacts; place every global and
        # per-timestep tensor beside the weight before fitting on the GPU.
        absmax = torch.as_tensor(row["absmax"], dtype=torch.float32, device=device).reshape(-1)
        sum_sq = torch.as_tensor(row["sum_sq"], dtype=torch.float32, device=device).reshape(-1)
        count = torch.as_tensor(row["count"], dtype=torch.float32, device=device)
        if absmax.numel() != width or sum_sq.numel() != width:
            raise ValueError(f"Calibration statistic width differs: expected {width}")
        if count.numel() not in (1, width):
            raise ValueError("Calibration count must be scalar or per-input-channel")
        variance = sum_sq / count.reshape(-1).clamp_min(1) if count.numel() == width else (
            sum_sq / count.clamp_min(1))
        if (not bool(torch.isfinite(absmax).all()) or not bool(torch.isfinite(variance).all())
                or bool((absmax < 0).any()) or bool((variance < 0).any())):
            raise ValueError("Calibration statistics contain nonfinite or negative values")
        return absmax.clamp_min(1e-8), variance.clamp_min(0)

    by_timestep = entry.get("by_timestep")
    if not by_timestep:
        maximum, variance = decode(entry)
        return [(maximum, variance)], maximum
    values = list(by_timestep.values()) if isinstance(by_timestep, dict) else list(by_timestep)
    if len(values) != 5:
        raise ValueError("ViDiT-Q timestep calibration must contain all five sampler evaluations")
    steps = [decode(row) for row in values]
    maximum = torch.stack([row[0] for row in steps]).amax(0)
    return steps, maximum


def _fwht(value, torch):
    width = value.shape[-1]
    if width != 128:
        raise ValueError("Local H128 rotation received a non-128 block")
    shape = value.shape
    result = value.float().contiguous()
    half = 1
    while half < width:
        shaped = result.reshape(*shape[:-1], width // (2 * half), 2, half)
        left, right = shaped[..., 0, :], shaped[..., 1, :]
        result = torch.stack((left + right, left - right), dim=-2).reshape(shape)
        half *= 2
    return result.float() * (1.0 / math.sqrt(width))


def rotate(value, signs, inverse=False):
    torch = _torch()
    width = value.shape[-1]
    if width % 128:
        raise ValueError(f"Cosmos-adapted local H128 requires input width divisible by 128, got {width}")
    sign = signs.to(device=value.device, dtype=torch.float32)
    x = value.float().reshape(*value.shape[:-1], -1, 128)
    sign = sign.reshape(-1, 128)
    if inverse:
        result = _fwht(x, torch) * sign
    else:
        result = _fwht(x * sign, torch)
    return result.reshape_as(value)


def _activation_quantize(value, bits=4, affine=False):
    torch = _torch()
    x = value.float()
    if affine:
        qmax = (1 << bits) - 1
        lower = x.amin(-1, keepdim=True).clamp_max(0)
        upper = x.amax(-1, keepdim=True).clamp_min(0)
        scale = (upper - lower) / qmax
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = (-lower / safe).round().clamp(0, qmax)
        return ((x / safe).round().add(zero).clamp(0, qmax) - zero).mul(scale).to(torch.bfloat16)
    qmax = (1 << (bits - 1)) - 1
    scale = x.abs().amax(-1, keepdim=True) / qmax
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    return (x.div(safe).round().clamp(-qmax, qmax).mul(scale)).to(torch.bfloat16)


def quantize_weight(weight, group=128, bits=4, affine=False):
    torch = _torch()
    import torch.nn.functional as functional
    rows, columns = weight.shape
    padded = _padded_columns(columns, group)
    blocks = functional.pad(weight.float(), (0, padded - columns)).reshape(rows, -1, group)
    if affine:
        qmax = (1 << bits) - 1
        lower = blocks.amin(-1).clamp_max(0)
        upper = blocks.amax(-1).clamp_min(0)
        scales = (upper - lower) / qmax
        safe = torch.where(scales > 0, scales, torch.ones_like(scales))
        zeros = (-lower / safe).round().clamp(0, qmax).to(torch.uint8)
        codes = (blocks / safe[..., None]).round().add(zeros[..., None]).clamp(
            0, qmax).to(torch.uint8)
    else:
        if bits != 4:
            raise ValueError("Only signed symmetric W4 is used by this adapted implementation")
        scales = blocks.abs().amax(-1) / 7
        safe = torch.where(scales > 0, scales, torch.ones_like(scales))
        codes = (blocks / safe[..., None]).round().clamp(-7, 7).add(8).to(torch.uint8)
        zeros = None
    scales = scales.to(torch.float16)
    codes = codes.reshape(rows, padded)
    if bits == 4:
        codes = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return {"codes": codes.contiguous(), "scales": scales.contiguous(),
            "zero_points": zeros, "shape": (rows, columns), "group_size": group,
            "bits": bits, "affine": bool(affine)}


def decode_weight(encoded, dtype=None):
    torch = _torch()
    codes = encoded["codes"]
    rows, columns = encoded["shape"]
    group, bits = int(encoded["group_size"]), int(encoded["bits"])
    padded = _padded_columns(columns, group)
    if bits == 4:
        nibbles = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(rows, padded)
    elif bits == 8:
        nibbles = codes.to(torch.uint8)
    else:
        raise ValueError("Encoded weight precision must be four or eight bits")
    blocks = nibbles.float().reshape(rows, -1, group)
    if encoded["affine"]:
        offset = encoded["zero_points"].float()[..., None]
    else:
        offset = 8
    decoded = ((blocks - offset) * encoded["scales"].float()[..., None]).reshape(rows, padded)
    return decoded[:, :columns].to(dtype or torch.bfloat16).contiguous()


def _random_signs(width, seed, device, torch):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int8)
    return (values * 2 - 1).to(device=device, dtype=torch.int8)


def _weighted_error(error, scale, steps, torch):
    # Equal-weight each of the five fixed denoising evaluations.
    energies = []
    for _, variance in steps:
        energies.append((error.float().square() * (variance / scale.square())[None, :]).sum())
    return float(torch.stack(energies).mean().detach())


def _deadline_check(deadline, reserve=30):
    if deadline is not None and time.monotonic() + reserve >= deadline:
        raise TimeoutError("soft deadline reached before the next calibration candidate")


def _fit_quarot(module, name, index, group, seed, torch):
    weight = module.weight.detach().float()
    width = weight.shape[1]
    signs = _random_signs(width, seed + index * 7919, weight.device, torch)
    encoded = quantize_weight(rotate(weight, signs), group=group, bits=4, affine=False)
    return {"format": FORMAT, "arm": "quarot_w4a4", "module": name, "module_index": index,
            "shape": tuple(weight.shape), "encoded": encoded, "signs": signs,
            "weight_bits": 4, "activation_bits": 4, "rank": 0,
            "calibration": {"calibration_fit": False,
                            "rotation": "random_local_h128_with_int8_sign_vector",
                            "seed": seed + index * 7919}}


def _svd_factors(matrix, rank, seed, oversampling, iterations, torch):
    rows, columns = matrix.shape
    retained = min(rank, rows, columns)
    q = min(retained + max(0, oversampling), min(rows, columns))
    devices = [matrix.device.index or 0] if matrix.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        u, singular, v = torch.svd_lowrank(matrix, q=q, niter=iterations)
    root = singular[:retained].clamp_min(0).sqrt()
    up = (u[:, :retained] * root).to(torch.bfloat16).contiguous()
    down = (root[:, None] * v[:, :retained].t()).to(torch.bfloat16).contiguous()
    return up, down, retained


def _fit_svd(module, name, index, stats, group, rank, seed, alphas, method, deadline, torch):
    weight = module.weight.detach().float()
    _, columns = weight.shape
    steps, act_max = _stats_rows(stats, columns, torch, weight.device)
    weight_max = weight.abs().amax(0).clamp_min(1e-8)
    mean_variance = torch.stack([variance for _, variance in steps]).mean(0)
    oversampling = int(method.get("svd_oversampling", 8))
    iterations = int(method.get("svd_power_iterations", 2))
    if oversampling < 0 or iterations < 0:
        raise ValueError("SVD randomized approximation settings must be nonnegative")
    best = None
    scores = []
    for alpha in alphas:
        _deadline_check(deadline)
        smoothing = act_max.pow(alpha) / weight_max.pow(1 - alpha)
        smoothing = smoothing.clamp_min(1e-8)
        transformed = weight * smoothing
        importance = (mean_variance.clamp_min(1e-12).sqrt() / smoothing).clamp_min(1e-8)
        importance = importance / importance.median().clamp_min(1e-8)
        factors_up, factors_down, retained = _svd_factors(
            transformed * importance[None, :], rank, seed + index * 1009,
            oversampling, iterations, torch)
        down = (factors_down.float() / importance[None, :]).to(torch.bfloat16).contiguous()
        low_rank = factors_up.float() @ down.float()
        residual = transformed - low_rank
        encoded = quantize_weight(residual, group=group, bits=4, affine=False)
        decoded_residual = decode_weight(encoded, dtype=torch.float32)
        score = _weighted_error(low_rank + decoded_residual - transformed,
                                smoothing, steps, torch)
        scores.append({"alpha": alpha, "weighted_reconstruction_error": score})
        if best is None or score < best[0]:
            best = (score, alpha, smoothing, factors_up, down, retained, encoded)
    score, alpha, smoothing, up, down, retained, encoded = best
    return {"format": FORMAT, "arm": "svdquant_w4a4", "module": name, "module_index": index,
            "shape": tuple(weight.shape), "encoded": encoded, "smoothing": smoothing,
            "up": up, "down": down, "weight_bits": 4, "activation_bits": 4, "rank": retained,
            "calibration": {"selected_alpha": alpha, "candidate_scores": scores,
                            "selected_weighted_reconstruction_error": score,
                            "svd": "randomized_truncated", "oversampling": oversampling,
                            "power_iterations": iterations, "seed": seed + index * 1009,
                            "factor_shapes": {"up": list(up.shape), "down": list(down.shape)},
                            "factor_dtype": "torch.bfloat16",
                            "residual_fit_dtype": "torch.float32",
                            "residual_storage": "signed_int4_group128_codes_with_float16_scales",
                            "low_rank_activation": "decoded_bf16"}}


def _vidit_balance(weight, act_max, alpha):
    torch = _torch()
    weight_max = weight.abs().amax(0).clamp_min(1e-8)
    return (weight_max.pow(alpha) / act_max.clamp_min(1e-8).pow(1 - alpha)).clamp_min(1e-8)


def _vidit_quantized(weight, balance, signs, group, bits, torch):
    # The first quantizer and the post-rotation quantizer are both retained as
    # the method's two-stage PTQ calibration step; only the latter is deployed.
    first = quantize_weight(weight / balance, group, bits=bits, affine=True)
    prequantized = decode_weight(first, dtype=torch.float32)
    encoded = quantize_weight(rotate(prequantized, signs), group, bits=bits, affine=True)
    return encoded


def _vidit_error(weight, balance, signs, encoded, steps, torch):
    decoded = decode_weight(encoded, dtype=torch.float32)
    estimate = rotate(decoded, signs, inverse=True)
    error = estimate - weight / balance
    energies = []
    for _, variance in steps:
        energies.append((error.square() * (variance * balance.square())[None, :]).sum())
    return float(torch.stack(energies).mean().detach())


def _fit_vidit(module, name, index, stats, group, seed, alphas, method, deadline, torch):
    weight = module.weight.detach().float()
    _, columns = weight.shape
    steps, act_max = _stats_rows(stats, columns, torch, weight.device)
    signs = _random_signs(columns, seed + index * 7919, weight.device, torch)
    scores = []
    best = None
    for alpha in alphas:
        _deadline_check(deadline)
        balance = _vidit_balance(weight, act_max, alpha)
        w4 = _vidit_quantized(weight, balance, signs, group, 4, torch)
        w8 = _vidit_quantized(weight, balance, signs, group, 8, torch)
        error4 = _vidit_error(weight, balance, signs, w4, steps, torch)
        error8 = _vidit_error(weight, balance, signs, w8, steps, torch)
        scores.append({"alpha": alpha, "w4_weighted_output_error": error4,
                       "w8_weighted_output_error": error8})
        if best is None or error4 < best[0]:
            best = (error4, alpha, balance, w4, error8, w8)
    error4, alpha, balance, encoded4, error8, encoded8 = best
    padded = _padded_columns(columns, group)
    extra_bytes = weight.shape[0] * padded // 2
    gain = max(0.0, error4 - error8)
    return {"format": FORMAT, "arm": "viditq_w4a4", "module": name, "module_index": index,
            "shape": tuple(weight.shape), "encoded": encoded4, "smoothing": balance, "signs": signs,
            "weight_bits": 4, "activation_bits": 4, "rank": 0,
            "calibration": {"selected_alpha": alpha, "candidate_scores": scores,
                            "selected_w4_weighted_output_error": error4,
                            "selected_w8_weighted_output_error": error8,
                            "w8_protection_gain": gain, "w8_extra_bytes": extra_bytes,
                            "protection": "whole_linear_w8a4_only",
                            "rotation_seed": seed + index * 7919,
                            "timestep_calibration": "five_sampler_evaluations_equal_weight"}}


def _bank_bytes(bank):
    torch = _torch()
    encoded = bank["encoded"]
    parts = {
        "weight_codes": encoded["codes"],
        "scales": encoded["scales"],
        "zero_points": encoded.get("zero_points"),
        "smoothing": bank.get("smoothing"),
        "rotation_signs": bank.get("signs"),
        "low_rank_factors": None,
    }
    parts["low_rank_factors"] = (bank["up"], bank["down"]) if "up" in bank else None
    result = {}
    for key, value in parts.items():
        if isinstance(value, tuple):
            result[key] = sum(tensor.numel() * tensor.element_size() for tensor in value)
        else:
            result[key] = (value.numel() * value.element_size()
                           if torch.is_tensor(value) else 0)
    return result


def _validate_bank(bank, arm, name, index, shape, group, torch):
    encoded = bank.get("encoded", {})
    rows, columns = shape
    padded = _padded_columns(columns, group)
    expected_code_shape = (rows, padded // 2) if int(encoded.get("bits", 4)) == 4 else (rows, padded)
    if (bank.get("format") != FORMAT or bank.get("arm") != arm or bank.get("module") != name
            or bank.get("module_index") != index or tuple(bank.get("shape", ())) != shape
            or tuple(encoded.get("shape", ())) != shape or encoded.get("group_size") != group
            or tuple(encoded.get("codes").shape) != expected_code_shape
            or encoded.get("codes").dtype != torch.uint8
            or tuple(encoded.get("scales").shape) != (rows, padded // group)
            or encoded.get("scales").dtype != torch.float16):
        raise RuntimeError(f"Baseline bank identity or tensor layout differs for {arm}/{index}")
    bits = int(encoded["bits"])
    if bits not in (4, 8) or bank.get("weight_bits") != bits:
        raise RuntimeError("Baseline bank records an unsupported or inconsistent weight precision")
    if bank.get("activation_bits") != 4 or (arm != "viditq_w4a4" and bits != 4):
        raise RuntimeError("Baseline bank precision differs from its declared adapted method")
    if bool(encoded.get("affine")) != (arm == "viditq_w4a4"):
        raise RuntimeError("Baseline bank quantizer differs from the requested method")
    zeros = encoded.get("zero_points")
    if arm == "viditq_w4a4":
        if (not torch.is_tensor(zeros) or zeros.dtype != torch.uint8
                or tuple(zeros.shape) != (rows, padded // group)):
            raise RuntimeError("ViDiT-Q adapted banks require affine uint8 zero points")
    elif zeros is not None:
        raise RuntimeError("Symmetric baseline bank unexpectedly stores zero points")
    if arm in ("quarot_w4a4", "viditq_w4a4"):
        if tuple(bank.get("signs", torch.empty(0)).shape) != (columns,):
            raise RuntimeError("Baseline H128 sign vector has an invalid width")
        if not bool(((bank["signs"] == -1) | (bank["signs"] == 1)).all()):
            raise RuntimeError("Baseline H128 sign vector contains values outside {-1,1}")
    if arm in ("svdquant_w4a4", "viditq_w4a4"):
        if tuple(bank.get("smoothing", torch.empty(0)).shape) != (columns,):
            raise RuntimeError("Baseline channel-balance vector has an invalid width")
        if (not bool(torch.isfinite(bank["smoothing"]).all())
                or not bool((bank["smoothing"] > 0).all())):
            raise RuntimeError("Baseline channel-balance vector must be finite and positive")
    if (not bool(torch.isfinite(encoded["scales"]).all())
            or not bool((encoded["scales"] > 0).all())):
        raise RuntimeError("Baseline group scales must be finite and positive")
    if arm == "svdquant_w4a4":
        rank = int(bank.get("rank", 0))
        if (rank <= 0 or tuple(bank["up"].shape) != (rows, rank)
                or tuple(bank["down"].shape) != (rank, columns)
                or bank["up"].dtype != torch.bfloat16 or bank["down"].dtype != torch.bfloat16):
            raise RuntimeError("SVDQuant adapted low-rank branch has an invalid stored rank")
        if not bool(torch.isfinite(bank["up"]).all()) or not bool(torch.isfinite(bank["down"]).all()):
            raise RuntimeError("SVDQuant low-rank factors contain nonfinite values")


def _load_stats(root, modules, torch):
    path = Path(root) / "results" / "teacher" / "statistics.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Shared BF16 Linear calibration statistics are missing: {path}")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if set(data) != set(modules):
        raise RuntimeError("Teacher statistics do not cover the exact 280 target Linear modules")
    result = {}
    for name, module in modules.items():
        width = int(module.weight.shape[1])
        entry = data[name]
        _stats_rows(entry, width, torch)
        result[name] = entry
    return result, path


def _manifest_base(protocol, arm, rows, recipe, rank):
    boundaries = {
        "quarot_w4a4": "Local H128 rotation for the selected Linear inputs and weights only; no whole-model attention or KV transform.",
        "svdquant_w4a4": "Finite calibration-only smoothing grid and one randomized truncated SVD per Linear; no iterative residual refinement.",
        "viditq_w4a4": "Five-step diagonal-statistic channel balance with greedy complete Linear W8A4 protection; not the full paper training recipe.",
    }
    return {"format": FORMAT, "protocol": protocol.get("protocol"), "arm": arm,
            "pbs_job_id": os.environ["PBS_JOBID"],
            "implementation": IMPLEMENTATIONS[arm], "status": "checkpointed", "phase": "bank_fitting",
            "target_linears": len(rows), "target_weight_numel": sum(row["weight_numel"] for row in rows),
            "calibration_contexts": len(protocol.get("calibration", {}).get("case_ids", [])),
            "calibration_timestep_evaluations": int(
                protocol.get("inference", {}).get("denoising_steps", 5)),
            "target_manifest": rows, "recipe": recipe, "global_svd_rank": rank,
            "adaptation_boundary": {
                "scope": boundaries[arm],
                "execution": "decoded_bf16_fake_a4_eager",
                "official_full_method_reproduction": False,
            },
            "entries": [{"name": row["name"], "shape": row["shape"],
                         "file": f"banks/linear_{index:04d}.pt",
                         "weight_numel": row["weight_numel"], "status": "pending"}
                        for index, row in enumerate(rows)]}


def _verify_manifest(saved, protocol, arm, rows, recipe, rank):
    if (saved.get("format") != FORMAT or saved.get("protocol") != protocol.get("protocol")
            or saved.get("arm") != arm or saved.get("implementation") != IMPLEMENTATIONS[arm]
            or saved.get("target_manifest") != rows or saved.get("recipe") != recipe
            or saved.get("global_svd_rank") != rank):
        raise RuntimeError("Existing baseline manifest belongs to a different frozen recipe or target scope")
    entries = saved.get("entries")
    if not isinstance(entries, list) or len(entries) != len(rows):
        raise RuntimeError("Existing baseline manifest has an incomplete module index")
    for index, (entry, row) in enumerate(zip(entries, rows)):
        if (entry.get("name") != row["name"] or entry.get("shape") != row["shape"]
                or entry.get("file") != f"banks/linear_{index:04d}.pt"):
            raise RuntimeError("Existing baseline manifest module identity differs")


def _read_bank(path, arm, name, index, shape, group, torch):
    bank = torch.load(path, map_location="cpu", weights_only=True)
    _validate_bank(bank, arm, name, index, shape, group, torch)
    return bank


def _entry_from_bank(bank, path, file_name=None):
    parts = _bank_bytes(bank)
    calibration = bank.get("calibration", {})
    return {"name": bank["module"], "shape": list(bank["shape"]),
            "file": file_name or path.as_posix(), "status": "complete",
            "weight_bits": int(bank["weight_bits"]), "activation_bits": int(bank["activation_bits"]),
            "rank": int(bank.get("rank", 0)), "bytes_by_kind": parts,
            "encoded_bytes": sum(parts.values()),
            "serialized_bytes": path.stat().st_size if path.is_file() else 0,
            "calibration": calibration}


def _summary(manifest_path, manifest):
    entries = manifest["entries"]
    complete_entries = [entry for entry in entries if entry.get("status") == "complete"]
    parts = {}
    for entry in complete_entries:
        for key, value in entry["bytes_by_kind"].items():
            parts[key] = parts.get(key, 0) + int(value)
    count = sum(int(entry["weight_numel"]) for entry in complete_entries)
    bytes_total = sum(parts.values())
    complete = (len(complete_entries) == manifest["target_linears"]
                and count == manifest["target_weight_numel"])
    return {"protocol": manifest.get("protocol"), "pbs_job_id": manifest.get("pbs_job_id"),
            "arm": manifest["arm"],
            "status": manifest["status"], "phase": manifest.get("phase"),
            "implementation": manifest["implementation"],
            "adaptation_boundary": manifest.get("adaptation_boundary"),
            "completed_modules": len(complete_entries), "expected_modules": manifest["target_linears"],
            "target_weight_numel": manifest["target_weight_numel"],
            "completed_weight_numel": count,
            "global_svd_rank": manifest.get("global_svd_rank"),
            "weight_group_size": manifest.get("recipe", {}).get("group_size"),
            "weight_scale_dtype": "torch.float16",
            "encoded_bytes_by_kind": parts, "encoded_target_bytes": bytes_total,
            "encoded_target_bpw": (8 * bytes_total / manifest["target_weight_numel"]
                                   if complete else None),
            "partial_payload_bpw_over_full_target": (
                8 * bytes_total / manifest["target_weight_numel"] if not complete else None),
            "calibration_contexts": manifest.get("calibration_contexts"),
            "calibration_timestep_evaluations": manifest.get("calibration_timestep_evaluations"),
            "serialized_bank_bytes": sum(int(entry.get("serialized_bytes", 0))
                                         for entry in complete_entries),
            "storage_scope": "tensor payload BPW; PyTorch bank-container bytes reported separately",
            "bpw_within_target_tolerance": manifest.get("bpw_within_target_tolerance"),
            "protection": manifest.get("protection"),
            "bank_manifest": str(manifest_path)}


def calibrate_baseline(runtime, protocol, arm, root, out, soft_deadline_seconds):
    """Prepare one independent, resumable baseline bank on the PBS compute node."""
    from common import guard
    guard()
    torch = _torch()
    if arm not in ARMS:
        raise ValueError(f"Unknown baseline arm {arm!r}; expected one of {ARMS}")
    modules = dict(sorted(runtime.target_linears.items()))
    rows, shapes = _target_rows(modules), _shapes(modules)
    if len(modules) != EXPECTED_MODULES or _parameters(shapes) != EXPECTED_WEIGHT_NUMEL:
        raise RuntimeError("Runtime target_linears differ from the frozen 280-module Cosmos contract")
    devices = {module.weight.device.type for module in modules.values()}
    if devices != {"cuda"}:
        raise RuntimeError("Baseline tensor preparation must run on the allocated PBS compute-node GPU")
    if any(module.weight.dtype != torch.bfloat16
           or (module.bias is not None and module.bias.dtype != torch.bfloat16)
           for module in modules.values()):
        raise RuntimeError("Cosmos-adapted baselines require BF16 target weights and biases")
    target_bpw, tolerance, group, seed = _target_config(protocol)
    if arm in ("quarot_w4a4", "viditq_w4a4") and any(
            shape[1] % 128 for _, shape in shapes):
        raise RuntimeError("Local H128 baselines require every target input width to be divisible by 128")
    params = _parameters(shapes)
    rank = None
    if arm == "svdquant_w4a4":
        rank, _ = _choose_svd_rank(shapes, params, group, target_bpw, tolerance,
                                   _rank_candidates(protocol))
    alphas = _alpha_candidates(protocol, arm) if arm in ("svdquant_w4a4", "viditq_w4a4") else []
    recipe = {"group_size": group, "target_bpw": target_bpw, "target_bpw_tolerance": tolerance,
              "seed": seed, "alpha_candidates": alphas,
              "rank_candidates": _rank_candidates(protocol) if arm == "svdquant_w4a4" else [],
              "method_settings": {**_baseline_config(protocol, arm), **_method_config(protocol, arm)},
              "svd_oversampling": int(_method_config(protocol, arm).get("svd_oversampling", 8))
              if arm == "svdquant_w4a4" else None,
              "svd_power_iterations": int(_method_config(protocol, arm).get("svd_power_iterations", 2))
              if arm == "svdquant_w4a4" else None,
              "selected_global_rank": rank}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _verify_manifest(manifest, protocol, arm, rows, recipe, rank)
    else:
        manifest = _manifest_base(protocol, arm, rows, recipe, rank)
    manifest["pbs_job_id"] = os.environ["PBS_JOBID"]
    _atomic_json(manifest_path, manifest)
    stats, stats_path = _load_stats(root, modules, torch)
    start = time.monotonic()
    deadline = None if soft_deadline_seconds is None or float(soft_deadline_seconds) <= 0 else (
        start + float(soft_deadline_seconds))
    for index, (name, module) in enumerate(modules.items()):
        entry = manifest["entries"][index]
        bank_path = out / entry["file"]
        if bank_path.is_file():
            bank = _read_bank(bank_path, arm, name, index,
                              (int(module.weight.shape[0]), int(module.weight.shape[1])), group, torch)
            rebuilt = _entry_from_bank(bank, bank_path, entry["file"])
            if arm == "viditq_w4a4" and entry.get("status") == "complete":
                rebuilt["calibration"] = entry["calibration"]
            manifest["entries"][index] = {**entry, **rebuilt}
            continue
        try:
            _deadline_check(deadline)
            if arm == "quarot_w4a4":
                bank = _fit_quarot(module, name, index, group, seed, torch)
            elif arm == "svdquant_w4a4":
                bank = _fit_svd(module, name, index, stats[name], group, rank,
                                seed, alphas, _method_config(protocol, arm), deadline, torch)
            else:
                bank = _fit_vidit(module, name, index, stats[name], group,
                                  seed, alphas, _method_config(protocol, arm), deadline, torch)
            _validate_bank(bank, arm, name, index,
                           (int(module.weight.shape[0]), int(module.weight.shape[1])), group, torch)
            _atomic_torch(bank_path, bank, torch)
            manifest["entries"][index] = _entry_from_bank(bank, bank_path, entry["file"])
            manifest["entries"][index]["weight_numel"] = int(module.weight.numel())
            manifest["status"] = "checkpointed"
            _atomic_json(manifest_path, manifest)
        except TimeoutError:
            manifest["status"] = "checkpointed"
            manifest["phase"] = "bank_fitting"
            _atomic_json(manifest_path, manifest)
            summary = _summary(manifest_path, manifest)
            summary["statistics_path"] = str(stats_path)
            _atomic_json(out / "summary.json", summary)
            return summary
    if arm == "viditq_w4a4":
        plan = manifest.get("protection", {}).get("selected_modules")
        if plan is None:
            completed = [entry for entry in manifest["entries"] if entry.get("status") == "complete"]
            base_bytes = sum(int(entry["encoded_bytes"]) for entry in completed)
            target_bytes = math.floor(target_bpw * params / 8)
            remaining = max(0, target_bytes - base_bytes)
            candidates = []
            for entry in completed:
                calibration = entry.get("calibration", {})
                extra = int(calibration.get("w8_extra_bytes", 0))
                gain = float(calibration.get("w8_protection_gain", 0))
                if extra > 0 and gain > 0:
                    candidates.append((gain / extra, entry["name"], extra))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            selected = []
            for _, name, extra in candidates:
                if extra <= remaining:
                    selected.append(name)
                    remaining -= extra
            manifest["protection"] = {
                "scheme": "calibration_weighted_whole_linear_w8a4",
                "selected_modules": selected,
                "configuration": "mixed_w4_w8a4" if selected else "uniform_w4",
                "w4_base_bytes": base_bytes,
                "target_bytes": target_bytes,
                "incremental_bytes": target_bytes - base_bytes - remaining,
                "unspent_bytes": remaining,
                "selection": "greedy descending weighted_output_error_reduction_per_incremental_byte",
            }
            plan = selected
            manifest["phase"] = "whole_layer_protection"
            _atomic_json(manifest_path, manifest)
        selected = set(plan)
        for index, (name, module) in enumerate(modules.items()):
            if name not in selected:
                continue
            entry = manifest["entries"][index]
            bank_path = out / entry["file"]
            bank = _read_bank(bank_path, arm, name, index,
                              (int(module.weight.shape[0]), int(module.weight.shape[1])), group, torch)
            if int(bank["weight_bits"]) == 8:
                continue
            try:
                _deadline_check(deadline)
                # Recreate the chosen alpha using the same calibration-only grid.
                bank8 = _fit_vidit(module, name, index, stats[name], group,
                                   seed, [float(bank["calibration"]["selected_alpha"])],
                                   _method_config(protocol, arm), deadline, torch)
                bank8["encoded"] = _vidit_quantized(
                    module.weight.detach().float(), bank8["smoothing"], bank8["signs"],
                    group, 8, torch)
                bank8["weight_bits"] = 8
                _atomic_torch(bank_path, bank8, torch)
                manifest["entries"][index] = _entry_from_bank(bank8, bank_path, entry["file"])
                manifest["entries"][index]["weight_numel"] = int(module.weight.numel())
                _atomic_json(manifest_path, manifest)
            except TimeoutError:
                manifest["status"] = "checkpointed"
                manifest["phase"] = "whole_layer_protection"
                _atomic_json(manifest_path, manifest)
                summary = _summary(manifest_path, manifest)
                summary["statistics_path"] = str(stats_path)
                _atomic_json(out / "summary.json", summary)
                return summary
    actual_parts = {}
    total_bytes = 0
    for index, (name, module) in enumerate(modules.items()):
        entry = manifest["entries"][index]
        bank = _read_bank(out / entry["file"], arm, name, index,
                          (int(module.weight.shape[0]), int(module.weight.shape[1])), group, torch)
        parts = _bank_bytes(bank)
        if sum(parts.values()) != int(entry["encoded_bytes"]):
            raise RuntimeError(f"Bank accounting differs after reload for {arm}/{index}")
        for key, value in parts.items():
            actual_parts[key] = actual_parts.get(key, 0) + int(value)
        total_bytes += sum(parts.values())
    bpw = 8 * total_bytes / params
    within = target_bpw * (1 - tolerance) <= bpw <= target_bpw * (1 + tolerance)
    if not within:
        raise RuntimeError(f"Actual {arm} representation is outside the frozen BPW tolerance: {bpw:.6f}")
    manifest.update({"status": "complete", "phase": "complete", "encoded_bytes_by_kind": actual_parts,
                     "encoded_target_bytes": total_bytes, "encoded_target_bpw": bpw,
                     "bpw_within_target_tolerance": True,
                     "statistics_path": str(stats_path),
                     "calibration_contexts": len(protocol.get("calibration", {}).get("case_ids", [])),
                     "calibration_timestep_evaluations": int(
                         protocol.get("inference", {}).get("denoising_steps", 5))})
    _atomic_json(manifest_path, manifest)
    summary = _summary(manifest_path, manifest)
    summary["statistics_path"] = str(stats_path)
    _atomic_json(out / "summary.json", summary)
    return summary


def install_baseline(runtime, protocol, arm, artifact_dir):
    """Install one completed bank as an eager decoded-BF16/fake-A4 reference."""
    from common import guard
    guard()
    torch = _torch()
    import torch.nn.functional as functional
    if arm not in ARMS:
        raise ValueError(f"Unknown baseline arm {arm!r}")
    directory = Path(artifact_dir).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("arm") != arm
            or manifest.get("protocol") != protocol.get("protocol")):
        raise RuntimeError("Only a complete matching baseline bank can be installed")
    modules = dict(sorted(runtime.target_linears.items()))
    rows = _target_rows(modules)
    if manifest.get("target_manifest") != rows or len(rows) != EXPECTED_MODULES:
        raise RuntimeError("Install runtime target_linears differ from the completed bank manifest")
    if any(module.weight.dtype != torch.bfloat16
           or (module.bias is not None and module.bias.dtype != torch.bfloat16)
           for module in modules.values()):
        raise RuntimeError("Baseline installation requires BF16 target weights and biases")
    if arm in ("quarot_w4a4", "viditq_w4a4") and any(
            int(module.weight.shape[1]) % 128 for module in modules.values()):
        raise RuntimeError("Local H128 baseline installation requires input widths divisible by 128")
    controller = getattr(runtime, "controller", None)
    if controller is not None:
        controller._restore_current()
        for module in modules.values():
            module.__dict__.pop("_cosmos_adapted_baseline_arm", None)
        controller._backup_weights()
    elif any(hasattr(module, "_cosmos_adapted_baseline_arm") for module in modules.values()):
        raise RuntimeError("A runtime without a controller must be freshly loaded from BF16")
    _, _, group, _ = _target_config(protocol)
    weight_bits, protected = [], []
    actual_parts, actual_bytes = {}, 0
    for index, (name, module) in enumerate(modules.items()):
        entry = manifest["entries"][index]
        bank_path = (directory / entry["file"]).resolve()
        try:
            bank_path.relative_to(directory)
        except ValueError as exc:
            raise RuntimeError("Baseline manifest bank path escapes its artifact directory") from exc
        bank = _read_bank(bank_path, arm, name, index,
                          (int(module.weight.shape[0]), int(module.weight.shape[1])), group, torch)
        parts = _bank_bytes(bank)
        entry_bytes = sum(parts.values())
        if entry_bytes != int(entry.get("encoded_bytes", -1)):
            raise RuntimeError(f"Installed bank storage differs from its manifest: {arm}/{index}")
        for key, value in parts.items():
            actual_parts[key] = actual_parts.get(key, 0) + int(value)
        actual_bytes += entry_bytes
        device = module.weight.device
        bank = _device_tree(bank, device, torch)
        decoded = decode_weight(bank["encoded"], dtype=torch.bfloat16)
        with torch.no_grad():
            module.weight.data.copy_(decoded)
        if arm == "viditq_w4a4":
            balance, signs = bank["smoothing"], bank["signs"]
            def forward(value, _module=module, _balance=balance, _signs=signs):
                transformed = rotate(value.float() * _balance, _signs)
                activation = _activation_quantize(transformed, bits=4, affine=True)
                return functional.linear(activation, _module.weight, _module.bias)
            if int(bank["weight_bits"]) == 8:
                protected.append(name)
        elif arm == "quarot_w4a4":
            signs = bank["signs"]
            def forward(value, _module=module, _signs=signs):
                activation = _activation_quantize(rotate(value, _signs), bits=4, affine=False)
                return functional.linear(activation, _module.weight, _module.bias)
        else:
            smoothing, up, down = bank["smoothing"], bank["up"], bank["down"]
            def forward(value, _module=module, _smoothing=smoothing, _up=up, _down=down):
                normalized = value.float() / _smoothing
                activation = _activation_quantize(normalized, bits=4, affine=False)
                main = functional.linear(activation, _module.weight, _module.bias)
                low = functional.linear(normalized.to(torch.bfloat16), _down)
                return main + functional.linear(low, _up)
        module.forward = forward
        module._cosmos_adapted_baseline_arm = arm
        weight_bits.append(int(bank["weight_bits"]))
    if (actual_bytes != int(manifest.get("encoded_target_bytes", -1))
            or actual_parts != manifest.get("encoded_bytes_by_kind")
            or manifest.get("target_weight_numel") != EXPECTED_WEIGHT_NUMEL):
        raise RuntimeError("Installed baseline payload totals differ from the completed manifest")
    metadata = {"protocol": protocol.get("protocol"), "arm": arm,
            "implementation": IMPLEMENTATIONS[arm], "target_linears": len(modules),
            "target_weight_numel": manifest["target_weight_numel"],
            "effective_bpw": manifest["encoded_target_bpw"],
            "encoded_target_bytes": manifest["encoded_target_bytes"],
            "encoded_target_bpw": manifest["encoded_target_bpw"],
            "encoded_bytes_by_kind": manifest["encoded_bytes_by_kind"],
            "weight_group_size": group, "weight_scale_dtype": "torch.float16",
            "serialized_bank_bytes": sum(int(entry.get("serialized_bytes", 0))
                                         for entry in manifest["entries"]),
            "weight_bits_histogram": {str(bits): weight_bits.count(bits) for bits in sorted(set(weight_bits))},
            "protected_modules": protected,
            "weight_bits": 4, "protected_weight_bits": 8, "activation_bits": 4,
            "protected_activation_bits": 4,
            "weight_quantizer": ("affine_group128_int4_with_selected_group128_int8"
                                 if arm == "viditq_w4a4" else "symmetric_signed_group128_int4"),
            "activation_quantizer": ("dynamic_per_token_affine_int4_zero_included"
                                     if arm == "viditq_w4a4"
                                     else "dynamic_per_token_symmetric_signed_int4_qmax7"),
            "execution": "decoded_bf16_fake_a4_eager",
            "native_int4_kernel_used": False,
            "weight_operand_dtype": "torch.bfloat16",
            "activation_operand_dtype": "torch.bfloat16",
            "inference_path": "per_linear_eager_fake_quant_with_decoded_bf16_weights",
            "calibration_contexts": len(protocol.get("calibration", {}).get("case_ids", [])),
            "calibration_timestep_evaluations": int(
                protocol.get("inference", {}).get("denoising_steps", 5)),
            "global_svd_rank": manifest.get("global_svd_rank"),
            "adaptation_boundary": manifest.get("adaptation_boundary"),
            "artifact_manifest": str(manifest_path)}
    if controller is not None:
        controller.arm = arm
        controller.metadata = dict(metadata)
    runtime.quantization_evidence = dict(metadata)
    return metadata


def _device_tree(value, device, torch):
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _device_tree(item, device, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_device_tree(item, device, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_device_tree(item, device, torch) for item in value)
    return value
