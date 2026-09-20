#!/usr/bin/env python3
"""Read-only verifier for the frozen paired PushT CEM transfer."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def numeric_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(numeric_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(numeric_tree(item) for item in value)
    return finite(value)


def max_abs_diff(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return math.inf
        return max((max_abs_diff(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        return max((max_abs_diff(a, b) for a, b in zip(left, right)), default=0.0)
    if finite(left) and finite(right):
        return abs(float(left) - float(right))
    return math.inf


def add(errors: list[str], where: str, message: str) -> None:
    errors.append(f"{where}: {message}")


def validate_trace(
    errors: list[str], value: Any, where: str, *, opt_steps: int, n_evals: int, topk: int
) -> None:
    if not isinstance(value, list) or len(value) != opt_steps:
        add(errors, where, f"must contain {opt_steps} CEM iterations")
        return
    for iteration, item in enumerate(value):
        item_where = f"{where}[{iteration}]"
        if not isinstance(item, dict) or set(item) != {"trajectories"}:
            add(errors, item_where, "must contain exactly trajectories")
            continue
        trajectories = item["trajectories"]
        if not isinstance(trajectories, list) or len(trajectories) != n_evals:
            add(errors, item_where, f"must contain {n_evals} trajectory traces")
            continue
        for traj, trace in enumerate(trajectories):
            trace_where = f"{item_where}.trajectories[{traj}]"
            required = {"elite_indices", "elite_losses", "mu", "sigma", "first_action"}
            if not isinstance(trace, dict) or set(trace) != required:
                add(errors, trace_where, "has an invalid decision-trace schema")
                continue
            if not isinstance(trace["elite_indices"], list) or len(trace["elite_indices"]) != topk:
                add(errors, trace_where, "elite_indices length does not equal topk")
            if not isinstance(trace["elite_losses"], list) or len(trace["elite_losses"]) != topk:
                add(errors, trace_where, "elite_losses length does not equal topk")
            for field in ("elite_indices", "elite_losses", "mu", "sigma", "first_action"):
                if not numeric_tree(trace[field]):
                    add(errors, trace_where, f"{field} must be finite numeric data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze", type=Path, default=Path(__file__).with_name("PUSHT_TRANSFER_FREEZE.json")
    )
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    try:
        freeze = load_json(args.freeze)
        summary = load_json(args.summary)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "INPUT_ERROR", "error": str(exc)}, indent=2))
        return 2

    try:
        if freeze.get("schema") != "dino-wm-shared-prefix-cache.pusht-iteration-cache-freeze":
            add(errors, "freeze", "unexpected schema")
        expected_schema = "dino-wm-shared-prefix-cache.pusht-iteration-cache-results"
        if summary.get("schema") != expected_schema:
            add(errors, "summary", f"schema must equal {expected_schema!r}")

        official = freeze["official_pushT"]
        timing = freeze["timing"]
        protocol = freeze["protocol"]
        settings = official["cem_settings"]
        observations = summary.get("observations")
        if observations != protocol["observation_ids"]:
            add(errors, "summary.observations", "does not match the two frozen observation IDs")
        if summary.get("warmup_repeats") != timing["warmup_repeats"]:
            add(errors, "summary.warmup_repeats", "does not match freeze")
        if summary.get("technical_repeats") != timing["technical_repeats"]:
            add(errors, "summary.technical_repeats", "does not match freeze")
        if summary.get("cem_settings") != settings:
            add(errors, "summary.cem_settings", "does not match freeze")
        for field in (
            "measurement_unit",
            "timing_boundary",
            "plan_section_boundary",
            "includes_preprocessing",
            "includes_environment_interaction",
            "device_synchronization",
            "path_order_policy",
        ):
            if summary.get(field) != timing[field]:
                add(errors, f"summary.{field}", "does not match freeze")

        rows = summary.get("records")
        if not isinstance(rows, list):
            add(errors, "summary.records", "must be an array")
            rows = []
        expected_observations = observations if isinstance(observations, list) else []
        expected_warmups = set(range(int(timing["warmup_repeats"])))
        expected_technical = set(range(int(timing["technical_repeats"])))
        grouped: dict[tuple[str, bool, int], dict[str, dict[str, Any]]] = {}
        observation_index = {value: index for index, value in enumerate(expected_observations)}
        for index, row in enumerate(rows, 1):
            where = f"records[{index}]"
            if not isinstance(row, dict):
                add(errors, where, "must be an object")
                continue
            observation_id = row.get("observation_id")
            path = row.get("path")
            warmup = row.get("warmup")
            repeat = row.get("technical_repeat")
            if observation_id not in observation_index:
                add(errors, where, "unknown observation_id")
                continue
            if path not in timing["paths"]:
                add(errors, where, "path is outside the frozen path set")
                continue
            if not isinstance(warmup, bool) or not isinstance(repeat, int) or isinstance(repeat, bool):
                add(errors, where, "warmup/repeat fields are invalid")
                continue
            expected_repeats = expected_warmups if warmup else expected_technical
            if repeat not in expected_repeats:
                add(errors, where, "repeat index is outside the frozen range")
            if not isinstance(row.get("planner_call_id"), str) or not row["planner_call_id"]:
                add(errors, where, "planner_call_id must be non-empty")
            if not isinstance(row.get("seed"), int) or isinstance(row["seed"], bool):
                add(errors, where, "seed must be an integer")
            for timing_field in (
                "latency_ms",
                "full_planner_latency_ms",
                "plan_section_latency_ms",
                "cem_loop_latency_ms",
            ):
                if not finite(row.get(timing_field)) or row[timing_field] <= 0:
                    add(errors, where, f"{timing_field} must be finite and positive")
            if not finite(row.get("cache_setup_latency_ms")) or row["cache_setup_latency_ms"] < 0:
                add(errors, where, "cache_setup_latency_ms must be finite and non-negative")
            if not isinstance(row.get("cache_build_count"), int) or row["cache_build_count"] < 0:
                add(errors, where, "cache_build_count must be a non-negative integer")
            if not finite(row.get("peak_memory_mib")) or row["peak_memory_mib"] < 0:
                add(errors, where, "peak_memory_mib must be finite and non-negative")
            for field in ("final_actions", "final_first_actions", "losses", "decision_trace"):
                if not numeric_tree(row.get(field)):
                    add(errors, where, f"{field} must be finite numeric data")
            if isinstance(row.get("losses"), list) and len(row["losses"]) != settings["opt_steps"]:
                add(errors, where, "loss trace length does not equal frozen opt_steps")
            actual_n_evals = (
                len(row["final_actions"])
                if isinstance(row.get("final_actions"), list)
                else 0
            )
            if actual_n_evals < 1:
                add(errors, where, "final_actions must contain at least one evaluation")
            validate_trace(
                errors,
                row.get("decision_trace"),
                f"{where}.decision_trace",
                opt_steps=int(settings["opt_steps"]),
                n_evals=actual_n_evals,
                topk=int(settings["topk"]),
            )
            order = row.get("path_order")
            if not isinstance(order, list) or set(order) != set(timing["paths"]) or len(order) != 2:
                add(errors, where, "path_order must list both paths exactly once")
            else:
                expected_first = (
                    timing["paths"][0]
                    if (observation_index[observation_id] + repeat) % 2 == 0
                    else timing["paths"][1]
                )
                if order[0] != expected_first:
                    add(errors, where, "path_order violates frozen interleaving")
            key = (observation_id, warmup, repeat)
            unit = grouped.setdefault(key, {})
            if path in unit:
                add(errors, where, "duplicate observation/path/repeat row")
            unit[path] = row

        expected_unit_count = len(expected_observations) * (
            len(expected_warmups) + len(expected_technical)
        )
        if len(grouped) != expected_unit_count:
            add(errors, "records", f"expected {expected_unit_count} observation/repeat units")

        comparisons: list[dict[str, Any]] = []
        exact_all = True
        threshold = float(freeze["decision_gate"]["max_abs_threshold"])
        within_threshold_all = True
        technical_latency_reductions: list[float] = []
        technical_plan_section_reductions: list[float] = []
        memory_ratios: list[float] = []
        for key, unit in sorted(grouped.items()):
            where = f"unit {key[0]}/{key[1]}/{key[2]}"
            if set(unit) != set(timing["paths"]):
                add(errors, where, "must contain exactly both paths")
                continue
            baseline = unit["baseline"]
            cached = unit["iteration_cache"]
            if baseline.get("seed") != cached.get("seed"):
                add(errors, where, "paired paths use different seeds")
            if baseline.get("planner_call_id") != cached.get("planner_call_id"):
                add(errors, where, "paired paths use different planner_call_id")
            diff = max(
                max_abs_diff(baseline.get("final_actions"), cached.get("final_actions")),
                max_abs_diff(baseline.get("final_first_actions"), cached.get("final_first_actions")),
                max_abs_diff(baseline.get("losses"), cached.get("losses")),
                max_abs_diff(baseline.get("decision_trace"), cached.get("decision_trace")),
            )
            exact = diff == 0.0
            within = diff <= threshold
            exact_all = exact_all and exact
            within_threshold_all = within_threshold_all and within
            if not within:
                add(errors, where, f"decision trace exceeds frozen threshold {threshold:g}")
            comparisons.append(
                {
                    "observation_id": key[0],
                    "warmup": key[1],
                    "technical_repeat": key[2],
                    "seed": baseline.get("seed"),
                    "decision_max_abs_diff": diff,
                    "bitwise_exact": exact,
                    "within_threshold": within,
                }
            )
            if not key[1]:
                base_latency = float(baseline["full_planner_latency_ms"])
                cached_latency = float(cached["full_planner_latency_ms"])
                technical_latency_reductions.append(1.0 - cached_latency / base_latency)
                base_section = float(baseline["plan_section_latency_ms"])
                cached_section = float(cached["plan_section_latency_ms"])
                technical_plan_section_reductions.append(
                    1.0 - cached_section / base_section
                )
                base_memory = float(baseline["peak_memory_mib"])
                cached_memory = float(cached["peak_memory_mib"])
                memory_ratios.append(cached_memory / base_memory if base_memory else math.inf)

        for observation_id in expected_observations:
            for warmup, expected in ((True, expected_warmups), (False, expected_technical)):
                observed = {
                    repeat
                    for (obs, flag, repeat) in grouped
                    if obs == observation_id and flag == warmup
                }
                if observed != expected:
                    add(errors, f"records.{observation_id}", f"repeat set mismatch for warmup={warmup}")
                for repeat in expected:
                    unit = grouped.get((observation_id, warmup, repeat), {})
                    if set(unit) != set(timing["paths"]):
                        add(errors, f"records.{observation_id}.{warmup}.{repeat}", "missing one paired path")

        integrity_errors = list(errors)
        decision_pass = (
            len(comparisons) == expected_unit_count
            and within_threshold_all
            and not integrity_errors
        )
        progression = freeze["progression_gate"]
        paired_reduction = (
            statistics.median(technical_latency_reductions)
            if technical_latency_reductions
            else None
        )
        paired_plan_section_reduction = (
            statistics.median(technical_plan_section_reductions)
            if technical_plan_section_reductions
            else None
        )
        memory_ratio = max(memory_ratios) if memory_ratios else None
        latency_pass = (
            paired_reduction is not None
            and paired_reduction >= progression["minimum_paired_median_full_plan_latency_reduction"]
        )
        memory_pass = (
            memory_ratio is not None
            and memory_ratio <= progression["maximum_peak_memory_ratio"]
        )
        if not latency_pass:
            add(errors, "progression_gate.latency", "paired median full-plan reduction is below the frozen 0.10 gate")
        if not memory_pass:
            add(errors, "progression_gate.memory", "peak-memory ratio exceeds the frozen 1.10 gate")
        progression_pass = decision_pass and latency_pass and memory_pass
        result = {
            "status": "PASS" if progression_pass and not errors else "FAIL",
            "decision_gate": {
                "status": "PASS" if decision_pass else "FAIL",
                "mode": "bitwise_exact" if exact_all else ("max_abs_threshold" if within_threshold_all else "FAIL"),
                "threshold": threshold,
            },
            "system_gates": {
                "latency": {
                    "status": "PASS" if latency_pass else "FAIL",
                    "paired_median_full_plan_latency_reduction": paired_reduction,
                    "minimum_required": progression["minimum_paired_median_full_plan_latency_reduction"],
                },
                "plan_section": {
                    "status": "MEASURED",
                    "paired_median_plan_section_latency_reduction": paired_plan_section_reduction,
                },
                "peak_memory": {
                    "status": "PASS" if memory_pass else "FAIL",
                    "max_peak_memory_ratio": memory_ratio,
                    "maximum_allowed": progression["maximum_peak_memory_ratio"],
                },
            },
            "progression_gate": {
                "status": "PASS" if progression_pass and not errors else "FAIL",
                "next_step_authorized": "LeWorldModel PushT only if PASS",
            },
            "comparison_count": len(comparisons),
            "comparisons": comparisons,
            "paired_median_latency_reduction": paired_reduction,
            "paired_median_full_planner_latency_reduction": paired_reduction,
            "paired_median_plan_section_latency_reduction": paired_plan_section_reduction,
            "max_peak_memory_ratio": memory_ratio,
            "errors": errors,
        }
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        result = {"status": "INPUT_ERROR", "error": str(exc), "errors": errors}

    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
