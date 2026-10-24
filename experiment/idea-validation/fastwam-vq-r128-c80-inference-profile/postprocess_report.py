"""Review pulled inference-profile receipts without loading models or raw traces."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
DEFAULT_PULL = HERE / "pulled" / "25938083.pbs101"
PROTOCOL = "fastwam-vq-r128-c80-inference-profile-v1"
PRIMARY_MODES = ("bf16", "legacy", "inference_dense")
DIAGNOSTIC_MODE = "stream_decode"
GiB = 1024 ** 3


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read JSON receipt: {path}") from exc


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_summary(summary: dict) -> dict:
    if not isinstance(summary, dict) or summary.get("protocol") != PROTOCOL:
        raise RuntimeError("summary.json is missing or belongs to another protocol")
    for key in ("complete", "export_complete", "parity_passed", "profile_complete"):
        if summary.get(key) is not True:
            raise RuntimeError(f"summary.json does not pass {key}")
    if type(summary.get("new_training_updates")) is not int or summary["new_training_updates"] != 0:
        raise RuntimeError("summary.json does not confirm zero new training updates")
    modes = summary.get("modes")
    if not isinstance(modes, dict):
        raise RuntimeError("summary.json has no measured mode receipts")
    for mode in PRIMARY_MODES:
        row = modes.get(mode) or {}
        if row.get("queries") != 80 or row.get("primary_measurements") is not True:
            raise RuntimeError(f"{mode} must contain 80 primary measurements")
    diagnostic = modes.get(DIAGNOSTIC_MODE) or {}
    if diagnostic.get("queries") != 2 or diagnostic.get("primary_measurements") is not False:
        raise RuntimeError("stream_decode must remain a two-query diagnostic")
    if not isinstance(summary.get("attribution", {}).get("inference_dense", {}).get("top_gemm_shapes"), list):
        raise RuntimeError("summary.json has no inference_dense GEMM shape receipt")
    return summary


def classify_gemm_shapes(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Only exact video/action tags become VQ kernel candidates."""
    candidates, bottlenecks = [], []
    for row in rows:
        if not isinstance(row, dict):
            continue
        stream = row.get("stream")
        shape_ok = all(type(row.get(key)) is int and row[key] > 0 for key in ("M", "N", "K"))
        measures_ok = type(row.get("calls")) is int and row["calls"] >= 0 and _finite(row.get("gpu_ms"))
        if not shape_ok or not measures_ok:
            continue
        item = {key: row[key] for key in ("stream", "M", "N", "K", "calls", "gpu_ms") if key in row}
        if stream in ("video", "action"):
            candidates.append(item)
        else:
            item["classification"] = (
                "whole_model_other" if stream == "other" else "unknown_proprio_or_other"
            )
            bottlenecks.append(item)
    candidates.sort(key=lambda row: row["gpu_ms"], reverse=True)
    bottlenecks.sort(key=lambda row: row["gpu_ms"], reverse=True)
    return candidates, bottlenecks


def load_target_linear_shards(directory: Path) -> list[dict]:
    rows = []
    shard_dir = directory / "per_layer"
    if not shard_dir.is_dir():
        return rows
    for path in sorted(shard_dir.glob("inference_dense_*.json")):
        value = read_json(path)
        if not isinstance(value, list):
            raise RuntimeError(f"Expected a target-linear row list: {path}")
        for row in value:
            if isinstance(row, dict) and row.get("category") == "target_linear":
                if _finite(row.get("gpu_ms")):
                    rows.append({key: row.get(key) for key in ("label", "calls", "gpu_ms")})
    return sorted(rows, key=lambda row: row["gpu_ms"] or 0, reverse=True)


def storage_metrics(storage: dict) -> dict:
    def gib(path, field):
        value = (storage.get(path) or {}).get(field)
        return value / GiB if type(value) is int and value >= 0 else None

    target_bf16 = storage.get("target_bf16_weight_bytes")
    return {
        "original_dense_model_state_GiB": gib("original_dense_model_state", "unique_storage_bytes"),
        "target_bf16_reference_GiB": target_bf16 / GiB if type(target_bf16) is int and target_bf16 >= 0 else None,
        "encoded_target_unique_GiB": gib("encoding", "unique_storage_bytes"),
        "target_plus_transforms_unique_GiB": gib("target_with_transforms", "unique_storage_bytes"),
        "unquantized_model_parameters_GiB": gib("unquantized_model_parameters", "unique_storage_bytes"),
        "whole_bundle_unique_tensor_storage_GiB": gib("whole_bundle", "unique_storage_bytes"),
        "torch_archive_file_GiB": storage.get("torch_file_bytes") / GiB
        if type(storage.get("torch_file_bytes")) is int and storage["torch_file_bytes"] >= 0 else None,
        "source_job": storage.get("source_job"),
        "arm": storage.get("arm"),
        "target_modules": storage.get("target_modules"),
        "claim_scope": storage.get("claim_scope"),
    }


def _fmt(value, digits=3) -> str:
    return f"{value:.{digits}f}" if _finite(value) else "n/a"


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = Path(stream.name)
    temporary.replace(path)


def _storage_rows(metrics: dict) -> list[tuple[str, str]]:
    labels = (
        ("Original dense model state, unique tensor storage", "original_dense_model_state_GiB"),
        ("Target weights at BF16 reference size", "target_bf16_reference_GiB"),
        ("Encoded target weight payload", "encoded_target_unique_GiB"),
        ("Target encoding plus transforms and rotations", "target_plus_transforms_unique_GiB"),
        ("Unquantized model parameters", "unquantized_model_parameters_GiB"),
        ("Whole bundle, unique tensor storage", "whole_bundle_unique_tensor_storage_GiB"),
        ("Serialized inference_bundle.pt archive", "torch_archive_file_GiB"),
    )
    return [(label, _fmt(metrics[key], 5)) for label, key in labels if metrics.get(key) is not None]


def _mode_rows(summary: dict) -> list[dict]:
    rows = []
    for name in (*PRIMARY_MODES, DIAGNOSTIC_MODE):
        mode = summary["modes"][name]
        metrics = mode.get("metrics") or {}
        latency = metrics.get("full_query_wall_ms") or {}
        rows.append({
            "mode": name, "queries": mode["queries"],
            "scope": "diagnostic only" if name == DIAGNOSTIC_MODE else "primary",
            "p50_ms": latency.get("p50"), "p95_ms": latency.get("p95"),
            "gpu": mode.get("gpu"),
        })
    return rows


def build_outputs(directory: Path) -> tuple[dict, str, str]:
    summary = validate_summary(read_json(directory / "summary.json"))
    storage_path = directory / "storage.json"
    storage = read_json(storage_path) if storage_path.is_file() else summary.get("storage", {})
    if storage.get("source_job") != summary.get("source_job") or storage.get("arm") != summary.get("source_arm"):
        raise RuntimeError("storage receipt does not match the summary source job/arm")

    attribution = summary["attribution"]["inference_dense"]
    candidates, bottlenecks = classify_gemm_shapes(attribution["top_gemm_shapes"])
    target_linears = load_target_linear_shards(directory)
    storage = storage_metrics(storage)
    modes = _mode_rows(summary)
    plan = {
        "protocol": summary["protocol"], "pbs_jobid": summary.get("pbs_jobid"),
        "source_job": summary.get("source_job"), "source_arm": summary.get("source_arm"),
        "status": "reviewed_measured_profile_plan", "scope": summary.get("scope"),
        "primary_modes": [row for row in modes if row["scope"] == "primary"],
        "diagnostic_modes": [row for row in modes if row["scope"] == "diagnostic only"],
        "vq_kernel_candidates_video_action_only": candidates,
        "whole_model_or_unknown_gemm_bottlenecks": bottlenecks,
        "target_linear_module_attribution": target_linears,
        "target_linear_shard_count": len(target_linears),
        "storage_receipt": storage,
        "nested_profiler_warning": "CUDA event categories are nested; do not sum target-linear, GEMM, stage, or component times.",
        "evidence_limits": {
            "new_training_updates": 0, "native_low_bit_kernel_implemented": False,
            "native_kernel_speedup_claim": False, "closed_loop_claim": False,
            "stream_decode_is_diagnostic_only": True,
            "unknown_proprio_or_other_is_not_a_vq_kernel_target": True,
        },
        "kernel_plan": {
            "candidate_shapes_source": "summary.attribution.inference_dense.top_gemm_shapes",
            "candidate_filter": "stream exactly video or action; preserve observed M/N/K/calls/gpu_ms only",
            "whole_model_handling": "other remains a whole-model bottleneck; proprio_or_other remains unknown and is excluded from VQ kernel candidates",
            "priority_shapes": candidates[:12],
            "preserve_checkpoint": True,
            "new_training_updates": 0,
            "native_kernel_implemented": False,
            "speedup_claim": False,
            "implementation_note": "Use only the measured video/action shapes to guide a future packed VQ consumer; validate numerics and full-policy latency on the same GPU before making any speed claim.",
        },
    }

    plan_json = json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    k_lines = [
        "# C80 R128 VQ kernel plan (reviewed)", "",
        f"Profile job: `{summary.get('pbs_jobid')}`; source checkpoint job: `{summary.get('source_job')}` / `{summary.get('source_arm')}`.",
        "",
        f"Primary scope: {summary.get('scope')}",
        "The three primary modes each use 80 queries. `stream_decode` uses 2 queries for a residency diagnostic only.",
        "",
        "Kernel candidates are restricted to rows explicitly tagged `video` or `action`. Other/proprio rows are excluded from this helper's video/action candidates; use REVIEWED_KERNEL_PLAN.json for the final exact-name proprio correction.",
        "",
        "| Stream | M | N | K | Calls | Intrusive GPU ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in candidates[:12]:
        k_lines.append(
            f"| {row['stream']} | {row['M']} | {row['N']} | {row['K']} | {row['calls']} | {_fmt(row['gpu_ms'])} |"
        )
    if not candidates:
        k_lines.append("| No explicitly tagged video/action rows | — | — | — | — | — |")
    k_lines += [
        "",
        "## Non-candidate GEMM bottlenecks", "",
        "These measurements remain at the full model boundary or unresolved. The pulled summary no longer carries the original module label for this merged bucket, so it cannot be split after aggregation.",
        "",
        "| Stream bucket | M | N | K | Calls | Intrusive GPU ms | Classification |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in bottlenecks:
        k_lines.append(
            f"| {row.get('stream', 'unknown')} | {row['M']} | {row['N']} | {row['K']} | {row['calls']} | {_fmt(row['gpu_ms'])} | {row['classification']} |"
        )
    if not bottlenecks:
        k_lines.append("| No preserved unknown/other GEMM rows | — | — | — | — | — | — |")
    k_lines += [
        "",
        "Target-linear module attributions are available separately, but their nested times must not be summed with GEMM or stage events.",
        "",
        "The export uses BF16 GEMM with decoded VQ weights and dynamic fake-A4 operands. No native low-bit kernel was implemented and no kernel speedup is claimed. Preserve the checkpoint and make no training updates.",
        "",
    ]
    k_plan_md = "\n".join(k_lines)

    r_lines = [
        "# C80 R128 inference profile (reviewed receipts)", "",
        f"Profile job: `{summary.get('pbs_jobid')}`; checkpoint source: `{summary.get('source_job')}` / `{summary.get('source_arm')}`.",
        "",
        f"Measured boundary: {summary.get('scope')}",
        "Observation-to-datum preprocessing was measured separately. Environment and robot motion are outside this profile.",
        "",
        "| Mode | Queries | Scope | p50 wall ms | p95 wall ms | GPU |",
        "|---|---:|---|---:|---:|---|",
    ]
    for row in modes:
        r_lines.append(
            f"| {row['mode']} | {row['queries']} | {row['scope']} | {_fmt(row['p50_ms'])} | {_fmt(row['p95_ms'])} | {row['gpu'] or 'n/a'} |"
        )
    r_lines += ["", "## Storage receipt", "", "| Measure | GiB |", "|---|---:|"]
    r_lines.extend(f"| {label} | {value} |" for label, value in _storage_rows(storage))
    r_lines += [
        "",
        "## Measured GEMM grouping", "",
        "Only explicitly tagged video/action shapes are VQ-kernel candidates. Other/proprio_or_other shapes remain whole-model or unknown bottlenecks.",
        "",
        "| Stream | M | N | K | Calls | Intrusive GPU ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in candidates:
        r_lines.append(
            f"| {row['stream']} | {row['M']} | {row['N']} | {row['K']} | {row['calls']} | {_fmt(row['gpu_ms'])} |"
        )
    if not candidates:
        r_lines.append("| No explicitly tagged video/action rows | — | — | — | — | — |")
    r_lines += [
        "",
        "Intrusive single-query profiler events are nested and must not be added together. The 2-query stream_decode mode is diagnostic only and is excluded from primary latency interpretation.",
        "No retraining occurred. No native low-bit kernel or kernel speedup claim is supported by this report. Fixed-input profiling does not establish closed-loop performance.",
        "",
    ]
    return plan, k_plan_md, "\n".join(r_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pulled_dir", nargs="?", type=Path, default=DEFAULT_PULL)
    args = parser.parse_args()
    plan, kernel_md, results_md = build_outputs(args.pulled_dir)
    out = args.pulled_dir
    _atomic_write(out / "corrected_kernel_plan.json", json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    _atomic_write(out / "KERNEL_PLAN.zh.md", kernel_md)
    _atomic_write(out / "RESULTS.reviewed.zh.md", results_md)
    print(json.dumps({
        "status": "written", "job_id": plan["pbs_jobid"],
        "kernel_candidate_shapes": len(plan["vq_kernel_candidates_video_action_only"]),
        "whole_model_or_unknown_shapes": len(plan["whole_model_or_unknown_gemm_bottlenecks"]),
        "target_linear_shards": plan["target_linear_shard_count"],
        "outputs": ["corrected_kernel_plan.json", "KERNEL_PLAN.zh.md", "RESULTS.reviewed.zh.md"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
