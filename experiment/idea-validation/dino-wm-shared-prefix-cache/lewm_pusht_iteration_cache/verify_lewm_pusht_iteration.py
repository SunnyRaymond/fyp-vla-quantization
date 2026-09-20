#!/usr/bin/env python3
"""Read-only verifier for the frozen LeWM PushT iteration-cache run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def max_abs(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return math.inf
        return max((max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max((max_abs(a, b) for a, b in zip(left, right)), default=0.0)
    if finite(left) and finite(right):
        return abs(float(left) - float(right))
    return math.inf


def json_safe(value: Any) -> Any:
    """Keep failed non-finite diagnostics JSON-compliant at the output boundary."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    freeze = load(args.freeze)
    summary = load(args.summary)
    errors: list[str] = []
    rows = summary.get("records", [])
    expected_obs = list(freeze["official_pusht"]["eval"]["observation_ids"])
    timing = freeze["timing"]
    threshold = float(freeze["decision_gate"]["max_abs_threshold"])
    grouped: dict[tuple[str, bool, int], dict[str, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"records[{index}] is not an object")
            continue
        key = (row.get("observation_id"), bool(row.get("warmup")), row.get("technical_repeat"))
        if row.get("path") not in ("baseline", "iteration_cache"):
            errors.append(f"records[{index}] has invalid path")
            continue
        grouped.setdefault(key, {})[row["path"]] = row
        for field in (
            "full_planner_latency_ms",
            "plan_section_latency_ms",
            "cem_loop_latency_ms",
            "cache_setup_latency_ms",
            "peak_memory_mib",
        ):
            value = row.get(field)
            lower_bound_failed = (
                finite(value) and float(value) < 0
                if field == "cache_setup_latency_ms"
                else finite(value) and float(value) <= 0
            )
            if not finite(value) or lower_bound_failed:
                bound = "non-negative" if field == "cache_setup_latency_ms" else "positive"
                errors.append(f"records[{index}].{field} is not {bound} and finite")

    expected_units = len(expected_obs) * (int(timing["warmup_repeats"]) + int(timing["technical_repeats"]))
    if len(grouped) != expected_units:
        errors.append(f"expected {expected_units} paired units, got {len(grouped)}")

    comparisons: list[dict[str, Any]] = []
    exact = True
    within = True
    full_reductions: list[float] = []
    plan_reductions: list[float] = []
    cem_reductions: list[float] = []
    memory_ratios: list[float] = []
    for key, unit in sorted(grouped.items()):
        if set(unit) != {"baseline", "iteration_cache"}:
            errors.append(f"{key} is not paired")
            continue
        baseline = unit["baseline"]
        cached = unit["iteration_cache"]
        if baseline.get("seed") != cached.get("seed"):
            errors.append(f"{key} has different seeds")
        fields = {
            "final_actions": (baseline.get("final_actions"), cached.get("final_actions")),
            "final_first_actions": (baseline.get("final_first_actions"), cached.get("final_first_actions")),
            "costs": (baseline.get("costs"), cached.get("costs")),
            "decision_trace": (baseline.get("decision_trace"), cached.get("decision_trace")),
        }
        diffs = {field: max_abs(left, right) for field, (left, right) in fields.items()}
        row_exact = all(value == 0.0 for value in diffs.values())
        row_within = all(value <= threshold for value in diffs.values())
        exact = exact and row_exact
        within = within and row_within
        if not row_within:
            errors.append(f"{key} decision difference exceeds {threshold:g}: {diffs}")
        comparison = {"observation_id": key[0], "warmup": key[1], "technical_repeat": key[2], **diffs}
        comparisons.append(comparison)
        if not key[1]:
            full_reductions.append(1.0 - float(cached["full_planner_latency_ms"]) / float(baseline["full_planner_latency_ms"]))
            plan_reductions.append(1.0 - float(cached["plan_section_latency_ms"]) / float(baseline["plan_section_latency_ms"]))
            cem_reductions.append(1.0 - float(cached["cem_loop_latency_ms"]) / float(baseline["cem_loop_latency_ms"]))
            memory_ratios.append(float(cached["peak_memory_mib"]) / float(baseline["peak_memory_mib"]))

    decision_pass = bool(comparisons) and within and not errors
    full_median = statistics.median(full_reductions) if full_reductions else None
    plan_median = statistics.median(plan_reductions) if plan_reductions else None
    cem_median = statistics.median(cem_reductions) if cem_reductions else None
    max_memory = max(memory_ratios) if memory_ratios else None
    full_pass = full_median is not None and full_median >= float(freeze["system_gate"]["minimum_paired_median_full_solve_latency_reduction"])
    memory_pass = max_memory is not None and max_memory <= float(freeze["system_gate"]["maximum_peak_memory_ratio"])
    status = "PASS" if decision_pass and full_pass and memory_pass and not errors else "FAIL"
    result = {
        "status": status,
        "decision_gate": {
            "status": "PASS" if decision_pass else "FAIL",
            "mode": "bitwise_exact" if exact else ("max_abs_threshold" if within else "FAIL"),
            "threshold": threshold,
        },
        "system_gates": {
            "full_planner": {"status": "PASS" if full_pass else "FAIL", "paired_median_reduction": full_median, "minimum": freeze["system_gate"]["minimum_paired_median_full_solve_latency_reduction"]},
            "peak_memory": {"status": "PASS" if memory_pass else "FAIL", "max_ratio": max_memory, "maximum": freeze["system_gate"]["maximum_peak_memory_ratio"]},
        },
        "latency": {
            "paired_median_full_planner_reduction": full_median,
            "paired_median_plan_section_reduction": plan_median,
            "paired_median_cem_loop_reduction": cem_median,
        },
        "comparisons": comparisons,
        "errors": errors,
    }
    print(json.dumps(json_safe(result), indent=2, allow_nan=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
