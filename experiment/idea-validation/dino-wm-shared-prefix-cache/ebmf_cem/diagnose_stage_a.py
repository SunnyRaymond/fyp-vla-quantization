"""Small, model-free diagnostics for an existing EBMF-CEM Stage A JSONL."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = float(rank)
    return result


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((a - left_mean) ** 2 for a in left)
    right_ss = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else float("nan")


def describe(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def main() -> None:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]

    calls: list[dict[str, float | int | str]] = []
    round_pooled: list[dict[str, float | int]] = []
    for round_index in range(10):
        pooled_errors: list[float] = []
        for record in records:
            row = record["rounds"][round_index]
            pooled_errors.extend(
                float(cheap) - float(full)
                for cheap, full in zip(row["J_tilde"], row["J_full"])
            )
        bias = statistics.median(pooled_errors)
        residuals = [error - bias for error in pooled_errors]
        round_pooled.append(
            {
                "round_index": round_index,
                "epsilon_raw_max_abs": max(abs(value) for value in pooled_errors),
                "median_signed_bias": bias,
                "centered_max_abs": max(abs(value) for value in residuals),
                "centered_p90_abs": percentile([abs(value) for value in residuals], 0.90),
                "mae": statistics.fmean(abs(value) for value in pooled_errors),
            }
        )

    for record in records:
        for row in record["rounds"]:
            cheap = [float(value) for value in row["J_tilde"]]
            full = [float(value) for value in row["J_full"]]
            cheap_order = sorted(range(len(cheap)), key=lambda index: (cheap[index], index))
            full_order = sorted(range(len(full)), key=lambda index: (full[index], index))
            cheap_top = set(cheap_order[:30])
            full_top = set(full_order[:30])
            full_sorted = [full[index] for index in full_order]
            cheap_sorted = [cheap[index] for index in cheap_order]
            epsilon = float(row["epsilon_round"])
            cutoff = full_sorted[29]
            calls.append(
                {
                    "observation_id": record["observation_id"],
                    "round_index": int(row["round_index"]),
                    "spearman": pearson(ranks(cheap), ranks(full)),
                    "top30_overlap": len(cheap_top & full_top) / 30.0,
                    "full_range": full_sorted[-1] - full_sorted[0],
                    "full_p90_p10": percentile(full, 0.90) - percentile(full, 0.10),
                    "full_boundary_gap_31_minus_30": full_sorted[30] - full_sorted[29],
                    "full_count_within_raw_epsilon_of_cutoff": sum(
                        abs(value - cutoff) <= epsilon for value in full
                    ),
                    "cheap_tail_above_top30_cutoff": cheap_sorted[-1] - cheap_sorted[29],
                    "two_epsilon": 2.0 * epsilon,
                    "raw_interval_covers_entire_cheap_tail": int(
                        cheap_sorted[-1] - cheap_sorted[29] <= 2.0 * epsilon
                    ),
                }
            )

    metric_names = [
        "spearman",
        "top30_overlap",
        "full_range",
        "full_p90_p10",
        "full_boundary_gap_31_minus_30",
        "full_count_within_raw_epsilon_of_cutoff",
        "cheap_tail_above_top30_cutoff",
        "two_epsilon",
        "raw_interval_covers_entire_cheap_tail",
    ]
    output = {
        "source": str(source),
        "n_observations": len(records),
        "n_planner_calls": len(calls),
        "unit_note": "The six observations are independent cases; 60 planner calls are nested technical measurements.",
        "planner_call_metrics": {name: describe([float(row[name]) for row in calls]) for name in metric_names},
        "round_pooled_error_metrics": round_pooled,
        "calls": calls,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
