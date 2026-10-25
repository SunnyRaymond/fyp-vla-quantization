#!/usr/bin/env python3
"""CPU-only numeric diagnosis of the saved R128 Cayley transform."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

SOURCE_OURS_PBS_JOB_ID = "25965916.pbs101"
ORTHOGONALITY_THRESHOLD = 5e-5


def _max_abs(value, torch):
    return float(value.abs().max().item())


def diagnose(root, out, expected_source_job):
    from common import guard, load_tensors, read_json, save_json, torch_module

    guard()
    torch = torch_module()
    torch.set_num_threads(int(os.environ.get("COSMOS_CPU_THREADS", "4")))
    root, out = Path(root), Path(out)
    checkpoint_path = root / "results" / "ours" / "checkpoint.pt"
    source_summary_path = root / "results" / "ours" / "summary.json"
    summary_path = out / "summary.json"
    diag_job_id = os.environ["PBS_JOBID"]
    summary = {
        "protocol": "cosmos-policy-vqwam-v1",
        "status": "failed",
        "pbs_job_id": diag_job_id,
        "diagnostic_pbs_job_id": diag_job_id,
        "source_checkpoint": str(checkpoint_path),
        "source_summary": str(source_summary_path),
        "expected_source_ours_pbs_job_id": expected_source_job,
        "orthogonality_threshold": ORTHOGONALITY_THRESHOLD,
        "execution_scope": "cpu_fp32_fp64_diagnostic_not_cuda_tf32_evidence",
        "cuda_tf32_measured_here": False,
        "pinned_model_loader_allow_tf32": True,
        "model_forwards": 0,
        "checkpoint_modified": False,
        "worker_writes_pipeline_marker": False,
        "summary_path": str(summary_path),
    }

    try:
        source_summary = read_json(source_summary_path)
        source_job_id = source_summary.get("pbs_job_id")
        if source_job_id != expected_source_job:
            raise RuntimeError(
                f"Ours summary PBS identity differs: expected {expected_source_job}, got {source_job_id}"
            )
        checkpoint = load_tensors(checkpoint_path, device="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError("Ours checkpoint payload must be a mapping")
        raw = checkpoint.get("rotation_raw")
        if not torch.is_tensor(raw):
            raise TypeError("Ours checkpoint has no tensor rotation_raw")
        raw = raw.detach().to(device="cpu")
        raw_shape = list(raw.shape)
        raw_finite = bool(torch.isfinite(raw).all().item())
        rotation_completed = int(checkpoint.get("rotation_completed", -1))
        vq_completed = int(checkpoint.get("vq_completed", -1))
        summary.update({
            "source_ours_pbs_job_id": source_job_id,
            "source_summary_status": source_summary.get("status"),
            "source_summary_rotation_completed": source_summary.get("rotation_completed"),
            "source_summary_vq_completed": source_summary.get("completed_updates"),
            "rotation_completed": rotation_completed,
            "rotation_expected": int(source_summary.get("rotation_expected", 8)),
            "vq_completed": vq_completed,
            "vq_expected": int(source_summary.get("expected_updates", 8)),
            "rotation_raw_shape": raw_shape,
            "rotation_raw_dtype": str(raw.dtype),
            "rotation_raw_finite": raw_finite,
            "rotation_raw_max_abs": float(raw.abs().max().item()) if raw.numel() else None,
            "torch_version": torch.__version__,
            "compute_device": "cpu",
        })
        if (rotation_completed != int(source_summary.get("rotation_completed", -1))
                or vq_completed != int(source_summary.get("completed_updates", -1))):
            raise RuntimeError("Checkpoint counters differ from the source ours summary")
        if (rotation_completed != 8 or vq_completed != 0
                or int(source_summary.get("rotation_completed", -1)) != 8
                or int(source_summary.get("completed_updates", -1)) != 0):
            raise RuntimeError("Source is not the expected R8/8, VQ0/8 checkpoint")
        if raw_shape != [128, 128]:
            raise ValueError(f"Expected rotation_raw shape [128, 128], got {raw_shape}")
        if not raw_finite:
            raise ValueError("rotation_raw contains nonfinite values")

        with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
            raw32 = raw.to(dtype=torch.float32)
            identity32 = torch.eye(128, dtype=torch.float32, device="cpu")
            skew32 = raw32 - raw32.t()
            matrix32 = torch.linalg.solve(identity32 - 0.5 * skew32,
                                          identity32 + 0.5 * skew32)
            matrix32_finite = bool(torch.isfinite(matrix32).all().item())
            fp32_gram_error = None
            fp32_cayley_gram_fp64_error = None
            if matrix32_finite:
                gram32 = matrix32.t() @ matrix32
                fp32_gram_error = _max_abs(gram32 - identity32, torch)
                matrix32_as64 = matrix32.to(dtype=torch.float64)
                identity64 = torch.eye(128, dtype=torch.float64, device="cpu")
                gram32_as64 = matrix32_as64.t() @ matrix32_as64
                fp32_cayley_gram_fp64_error = _max_abs(gram32_as64 - identity64, torch)

            raw64 = raw.to(dtype=torch.float64)
            identity64 = torch.eye(128, dtype=torch.float64, device="cpu")
            skew64 = raw64 - raw64.t()
            matrix64 = torch.linalg.solve(identity64 - 0.5 * skew64,
                                          identity64 + 0.5 * skew64)
            matrix64_finite = bool(torch.isfinite(matrix64).all().item())
            fp64_gram_error = None
            if matrix64_finite:
                gram64 = matrix64.t() @ matrix64
                fp64_gram_error = _max_abs(gram64 - identity64, torch)

        from training_state import rotation_orthogonality
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
            helper_error = rotation_orthogonality(matrix32)
        helper_matches = (fp32_cayley_gram_fp64_error is not None
                          and abs(helper_error - fp32_cayley_gram_fp64_error) <= 1e-12)
        summary["training_gate_under_cpu_bf16_autocast"] = {
            "orthogonality_max_abs": helper_error,
            "matches_explicit_fp64_gram": helper_matches,
        }

        summary.update({
            "fp32_cayley": {
                "matrix_shape": list(matrix32.shape),
                "matrix_finite": matrix32_finite,
                "gram_error_float32_max_abs": fp32_gram_error,
                "gram_error_float64_accumulation_max_abs": fp32_cayley_gram_fp64_error,
            },
            "fp64_cayley": {
                "matrix_shape": list(matrix64.shape),
                "matrix_finite": matrix64_finite,
                "gram_error_float64_max_abs": fp64_gram_error,
            },
        })
        summary["orthogonality_gate_error_fp32_matrix_fp64_gram"] = fp32_cayley_gram_fp64_error
        summary["legacy_fp32_gram_error"] = fp32_gram_error
        summary["legacy_fp32_gram_passed"] = (
            fp32_gram_error is not None and fp32_gram_error <= ORTHOGONALITY_THRESHOLD
        )
        summary["fp32_cayley_fp64_gram_passed"] = (
            matrix32_finite and fp32_cayley_gram_fp64_error is not None
            and fp32_cayley_gram_fp64_error <= ORTHOGONALITY_THRESHOLD
        )
        summary["fp64_cayley_passed"] = (
            matrix64_finite and fp64_gram_error is not None
            and fp64_gram_error <= ORTHOGONALITY_THRESHOLD
        )
        summary["orthogonality_gate_passed"] = (
            summary["fp32_cayley_fp64_gram_passed"] and summary["fp64_cayley_passed"]
            and helper_matches
        )
        if summary["orthogonality_gate_passed"]:
            summary["status"] = "complete"
        else:
            summary["status"] = "failed"
            summary["failure_reason"] = "one or more orthogonality checks exceeded the unchanged threshold"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)

    save_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-ours-pbs-job-id", default=SOURCE_OURS_PBS_JOB_ID)
    args = parser.parse_args()
    summary = diagnose(args.root, args.out, args.source_ours_pbs_job_id)
    print(summary)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
