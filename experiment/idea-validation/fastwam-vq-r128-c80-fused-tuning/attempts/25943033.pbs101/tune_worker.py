"""Choose measured Triton tilings inside the approved PBS GPU allocation."""
from __future__ import annotations

import json
import math
import os
import random
import statistics
from pathlib import Path

from kernel_worker import cuda_time, progress
from reference_profile import freeze_numerics, guard, save

SEED = 20261023
SHAPES = sorted({
    (294, 3072, 3072), (32, 3072, 1024), (129, 3072, 3072),
    (32, 1024, 3072), (129, 3072, 1024), (294, 14336, 3072),
    (294, 3072, 14336), (32, 1024, 4096), (32, 4096, 1024),
    (1, 4096, 8), (32, 7, 1024), (32, 1024, 7),
})


def _fatal_cuda_error(exc):
    message = str(exc).lower()
    return any(text in message for text in (
        "illegal memory access", "illegal access", "illegal address",
        "cuda_error_illegal_address", "device-side assert",
        "unspecified launch failure",
    ))


def _nrmse(actual, reference):
    import torch

    difference = actual.float() - reference.float()
    numerator = difference.square().mean().sqrt()
    denominator = reference.float().square().mean().sqrt().clamp_min(1e-12)
    value = float((numerator / denominator).item())
    del difference, numerator, denominator
    return value


def _to_cuda_encoding(encoding, torch):
    return {key: value.to(device="cuda") if isinstance(value, torch.Tensor) else value
            for key, value in encoding.items()}


def _measure(call):
    blocks = [cuda_time(call, repeat=10) for _ in range(3)]
    return blocks, statistics.median(blocks)


def _evaluate_candidate(name, x, prepared, bias, reference, limit, fused_vq):
    import torch

    try:
        actual = fused_vq.prepared_linear(x, prepared, bias, config=name)
        finite = bool(torch.isfinite(actual).all().item())
        if not finite:
            return {"config": name, "status": "excluded", "nrmse": None,
                    "exclusion_reason": {"type": "NonFiniteOutput",
                                         "message": "candidate output contains NaN or Inf"}}
        nrmse = _nrmse(actual, reference)
        del actual
        if not math.isfinite(nrmse) or nrmse > limit:
            return {"config": name, "status": "excluded", "nrmse": nrmse,
                    "exclusion_reason": {"type": "ValidationNRMSE",
                                         "message": f"NRMSE {nrmse} exceeds limit {limit}"}}
        blocks, median = _measure(
            lambda: fused_vq.prepared_linear(x, prepared, bias, config=name))
        return {"config": name, "status": "valid", "nrmse": nrmse,
                "gpu_ms_blocks": blocks, "median_gpu_ms": median}
    except Exception as exc:
        if _fatal_cuda_error(exc):
            raise
        return {"config": name, "status": "excluded", "nrmse": None,
                "exclusion_reason": {"type": type(exc).__name__, "message": str(exc)}}


def main():
    guard()
    artifacts = Path(os.environ["ARTIFACTS"])
    p = json.loads((artifacts / "protocol.json").read_text())
    root = Path(p["remote_root"]) / "results"

    import torch
    import export_runtime as export
    import fused_vq

    freeze_numerics(torch)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if len(SHAPES) != 12 or p["tuning"]["shape_count"] != len(SHAPES):
        raise ValueError("Tuner shape contract must contain exactly 12 cases")
    limit = float(p["tuning"]["validation_nrmse_max"])
    if len(fused_vq.CONFIGS) != p["tuning"]["candidate_count"]:
        raise ValueError("CONFIGS count differs from the frozen tuning protocol")

    progress(root, "tuner", "loading_cpu_bundle", expected_shapes=len(SHAPES))
    bundle = export.load_bundle(p["export_reuse"]["bundle_path"])
    requested_nk = {(n, k) for _, n, k in SHAPES}
    first_by_nk = {}
    for module, encoding in bundle["quantized"].items():
        nk = tuple(map(int, encoding["shape"]))
        if nk in requested_nk:
            first_by_nk.setdefault(nk, (module, encoding))
    missing = sorted(requested_nk - first_by_nk.keys())
    if missing:
        raise ValueError(f"No actual quantized module found for N,K shapes: {missing}")

    # Copy each first-matching encoding once and reuse it for every requested M.
    prepared_by_nk = {}
    for nk, (module, cpu_encoding) in first_by_nk.items():
        encoding = _to_cuda_encoding(cpu_encoding, torch)
        decoded = fused_vq.decode_weight(encoding)
        prepared = fused_vq.prepare(encoding)
        prepared_by_nk[nk] = (module, prepared, decoded)
        del encoding
    del bundle

    record = {
        "protocol": p["protocol"], "pbs_jobid": os.environ["PBS_JOBID"],
        "complete": False, "seed": SEED, "candidate_count": len(fused_vq.CONFIGS),
        "validation_nrmse_max": limit,
        "allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "shapes": [],
    }
    save(root / "tuning.json", record)
    order_rng = random.Random(SEED)
    selected = {}
    receipts = root / "tuning_receipts"

    for shape_index, (m, n, k) in enumerate(SHAPES, start=1):
        module, prepared, decoded = prepared_by_nk[(n, k)]
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        bias = torch.zeros((n,), device="cuda", dtype=torch.bfloat16)
        reference = torch.nn.functional.linear(x, decoded, bias)
        if not bool(torch.isfinite(reference).all().item()):
            raise RuntimeError(f"Non-finite F.linear reference for {(m, n, k)}")
        baseline_blocks, baseline_median = _measure(
            lambda: torch.nn.functional.linear(x, decoded, bias))

        names = list(fused_vq.CONFIGS)
        order_rng.shuffle(names)
        candidates = [_evaluate_candidate(name, x, prepared, bias, reference, limit, fused_vq)
                      for name in names]
        valid = [row for row in candidates if row["status"] == "valid"]
        shape_key = f"{m},{n},{k}"
        shape_row = {
            "shape": {"M": m, "N": n, "K": k}, "module": module,
            "candidate_order": names, "baseline_f_linear": {
                "gpu_ms_blocks": baseline_blocks, "median_gpu_ms": baseline_median,
            },
            "candidates": candidates,
        }
        if valid:
            winner = min(valid, key=lambda row: row["median_gpu_ms"])
            selected[shape_key] = {
                "config": winner["config"], "median_gpu_ms": winner["median_gpu_ms"],
                "module": module, "nrmse": winner["nrmse"],
            }
            shape_row["selected"] = selected[shape_key]
            shape_row["valid_candidate_count"] = len(valid)
            receipt_status = "complete"
        else:
            shape_row["selected"] = None
            shape_row["valid_candidate_count"] = 0
            receipt_status = "no_valid_candidate"

        record["shapes"].append(shape_row)
        record["completed_shapes"] = shape_index
        save(root / "tuning.json", record)
        save(receipts / f"shape_{m}_{n}_{k}.json", {
            "protocol": p["protocol"], "pbs_jobid": os.environ["PBS_JOBID"],
            "status": receipt_status, "shape": shape_row["shape"], "module": module,
            "valid_candidate_count": len(valid), "selected": shape_row["selected"],
            "baseline_f_linear_median_gpu_ms": baseline_median,
        })
        progress(root, "tuner", "shape_complete", index=shape_index,
                 expected=len(SHAPES), shape=shape_key, module=module,
                 valid_candidates=len(valid), selected_config=(
                     shape_row["selected"]["config"] if valid else None))
        del x, bias, reference
        if not valid:
            raise RuntimeError(f"No valid fused VQ candidate for shape {shape_key}")

    if len(selected) != 12:
        raise RuntimeError(f"Tuner completed {len(selected)}/12 required shapes")
    record["complete"] = True
    record["completed_shapes"] = len(SHAPES)
    save(root / "tuning.json", record)
    save(root / "selected_configs.json", {
        "complete": True, "protocol": p["protocol"],
        "pbs_jobid": os.environ["PBS_JOBID"], "selected": selected,
    })
    progress(root, "tuner", "complete", shapes=len(selected), expected=len(SHAPES))


if __name__ == "__main__":
    main()
