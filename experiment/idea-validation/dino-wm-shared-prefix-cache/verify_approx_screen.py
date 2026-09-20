#!/usr/bin/env python3
"""Verify the frozen decision-safe approximate-cache engineering screen.

This verifier reads recorded arrays and timing rows only.  It does not load a
model, run CUDA work, use hashes, or turn the screen into a statistical test.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


PATHS = ("baseline", "factorized_cache")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    return records


def add_error(errors: list[str], where: str, message: str) -> None:
    errors.append(f"{where}: {message}")


def require(mapping: dict[str, Any], key: str, where: str, errors: list[str]) -> Any:
    if key not in mapping:
        add_error(errors, where, f"missing required field {key!r}")
        return None
    return mapping[key]


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def scalar_identifier(value: Any) -> bool:
    return (isinstance(value, str) and bool(value)) or (
        isinstance(value, int) and not isinstance(value, bool)
    )


def flatten_numeric(value: Any, where: str, errors: list[str]) -> tuple[Any, list[float]]:
    """Return a nested-list shape signature and finite numeric leaves."""
    if isinstance(value, list):
        if not value:
            add_error(errors, where, "numeric array must not be empty")
            return ("list", 0), []
        children = [flatten_numeric(item, where, errors) for item in value]
        return ("list", tuple(child[0] for child in children)), [
            number for child in children for number in child[1]
        ]
    if not finite_number(value):
        add_error(errors, where, "value must be finite and numeric")
        return ("scalar",), []
    return ("scalar",), [float(value)]


def exact_difference(left: Any, right: Any, where: str, errors: list[str]) -> tuple[float | None, float | None]:
    left_shape, left_values = flatten_numeric(left, f"{where}.baseline", errors)
    right_shape, right_values = flatten_numeric(right, f"{where}.factorized_cache", errors)
    if left_shape != right_shape or len(left_values) != len(right_values):
        add_error(errors, where, "baseline and factorized_cache arrays have different shapes")
        return None, None
    differences = [abs(left_value - right_value) for left_value, right_value in zip(left_values, right_values)]
    max_abs = max(differences) if differences else 0.0
    l2 = math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left_values, right_values)))
    return max_abs, l2


def verify_path_record(path_record: Any, path_name: str, where: str, errors: list[str]) -> bool:
    if not isinstance(path_record, dict):
        add_error(errors, where, "must be an object")
        return False
    path_value = require(path_record, "path", where, errors)
    if path_value != path_name:
        add_error(errors, where, f"path must equal {path_name!r}")
    seed = require(path_record, "noise_schedule_seed", where, errors)
    if not scalar_identifier(seed):
        add_error(errors, where, "noise_schedule_seed must be a scalar identifier")
    valid = True
    for field in ("input_mu", "input_sigma", "mu", "sigma", "first_action"):
        value = require(path_record, field, where, errors)
        if not isinstance(value, list):
            add_error(errors, f"{where}.{field}", "must be a non-empty numeric array")
            valid = False
        else:
            before = len(errors)
            flatten_numeric(value, f"{where}.{field}", errors)
            valid = valid and len(errors) == before
    topk = require(path_record, "topk_indices", where, errors)
    if not isinstance(topk, list) or len(topk) != 30:
        add_error(errors, f"{where}.topk_indices", "must contain exactly 30 indices")
        valid = False
    elif (
        len(set(topk)) != 30
        or not all(isinstance(index, int) and not isinstance(index, bool) for index in topk)
        or not all(0 <= index < 300 for index in topk)
    ):
        add_error(errors, f"{where}.topk_indices", "must be 30 unique integer indices in [0,299]")
        valid = False
    return valid


def verify_decision(records: list[Any], contract: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    settings = contract["scope"]["fixed_settings"]
    gate = contract["decision_gate"]
    thresholds = gate["screen_thresholds"]
    observation_ids: list[str] = []
    passed_observations = 0

    if len(records) != 10:
        add_error(errors, "decision", "must contain exactly 10 observation records")

    for record_number, record in enumerate(records, 1):
        where = f"decision line {record_number}"
        record_errors_before = len(errors)
        if not isinstance(record, dict):
            add_error(errors, where, "record must be an object")
            continue
        observation_id = require(record, "observation_id", where, errors)
        if not isinstance(observation_id, str) or not observation_id:
            add_error(errors, where, "observation_id must be a non-empty string")
        elif observation_id in observation_ids:
            add_error(errors, where, "observation_id is duplicated")
        else:
            observation_ids.append(observation_id)
        observation_seed = require(record, "observation_seed", where, errors)
        if not scalar_identifier(observation_seed):
            add_error(errors, where, "observation_seed must be a scalar identifier")
        for field, expected in (
            ("candidate_count", settings["candidate_count"]),
            ("horizon", settings["horizon"]),
            ("topk", settings["topk"]),
            ("cem_opt_steps", settings["cem_opt_steps"]),
        ):
            value = require(record, field, where, errors)
            if value != expected:
                add_error(errors, where, f"{field} must equal frozen value {expected!r}")

        rounds = require(record, "rounds", where, errors)
        if not isinstance(rounds, list) or len(rounds) != gate["round_count"]:
            add_error(errors, f"{where}.rounds", f"must contain exactly {gate['round_count']} rounds")
            continue

        round_map: dict[int, dict[str, Any]] = {}
        for round_number, round_record in enumerate(rounds):
            round_where = f"{where}.rounds[{round_number}]"
            if not isinstance(round_record, dict):
                add_error(errors, round_where, "round must be an object")
                continue
            round_index = require(round_record, "round_index", round_where, errors)
            if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index in round_map:
                add_error(errors, round_where, "round_index must be unique integer")
                continue
            round_map[round_index] = round_record
        if set(round_map) != set(range(gate["round_count"])):
            add_error(errors, f"{where}.rounds", "round_index must be exactly 0..9")
            continue

        previous: dict[str, dict[str, Any] | None] = {path: None for path in PATHS}
        observation_failed = False
        for round_index in range(gate["round_count"]):
            round_record = round_map[round_index]
            round_where = f"{where}.rounds[{round_index}]"
            round_seed = require(round_record, "noise_schedule_seed", round_where, errors)
            if not scalar_identifier(round_seed):
                add_error(errors, f"{round_where}.noise_schedule_seed", "must be a scalar identifier")
            path_records: dict[str, dict[str, Any]] = {}
            for path in PATHS:
                path_record = round_record.get(path)
                if not verify_path_record(path_record, path, f"{round_where}.{path}", errors):
                    observation_failed = True
                if isinstance(path_record, dict):
                    path_records[path] = path_record
                    if path_record.get("noise_schedule_seed") != round_seed:
                        add_error(errors, f"{round_where}.{path}", "does not reuse the paired planner noise schedule seed")
                    prior = previous[path]
                    if prior is not None:
                        if path_record.get("input_mu") != prior.get("mu"):
                            add_error(errors, f"{round_where}.{path}.input_mu", "does not equal previous-round true mu")
                        if path_record.get("input_sigma") != prior.get("sigma"):
                            add_error(errors, f"{round_where}.{path}.input_sigma", "does not equal previous-round true sigma")
            if round_index == 0 and set(path_records) == set(PATHS):
                if path_records["baseline"].get("input_mu") != path_records["factorized_cache"].get("input_mu"):
                    add_error(errors, f"{round_where}.input_mu", "round-0 path inputs are not paired")
                if path_records["baseline"].get("input_sigma") != path_records["factorized_cache"].get("input_sigma"):
                    add_error(errors, f"{round_where}.input_sigma", "round-0 path inputs are not paired")

            if set(path_records) != set(PATHS):
                observation_failed = True
                continue
            baseline = path_records["baseline"]
            factorized = path_records["factorized_cache"]
            baseline_topk = baseline.get("topk_indices")
            factorized_topk = factorized.get("topk_indices")
            if isinstance(baseline_topk, list) and isinstance(factorized_topk, list):
                overlap = len(set(baseline_topk).intersection(factorized_topk)) / 30.0
                recorded_overlap = require(round_record, "topk_set_overlap", round_where, errors)
                if not finite_number(recorded_overlap) or recorded_overlap != overlap:
                    add_error(errors, f"{round_where}.topk_set_overlap", f"must equal recomputed value {overlap}")
                if overlap != thresholds["per_round_topk_set_overlap"]:
                    observation_failed = True
                    add_error(errors, f"{round_where}.topk_set_overlap", "fails the per-round overlap gate")
            else:
                observation_failed = True

            mu_diff, _ = exact_difference(baseline.get("mu"), factorized.get("mu"), f"{round_where}.mu", errors)
            sigma_diff, _ = exact_difference(baseline.get("sigma"), factorized.get("sigma"), f"{round_where}.sigma", errors)
            action_diff, action_l2 = exact_difference(
                baseline.get("first_action"), factorized.get("first_action"), f"{round_where}.first_action", errors
            )
            for field, computed in (
                ("mu_max_abs_diff", mu_diff),
                ("sigma_max_abs_diff", sigma_diff),
                ("first_action_max_abs_diff", action_diff),
                ("first_action_l2_diff", action_l2),
            ):
                recorded = require(round_record, field, round_where, errors)
                if computed is None or not finite_number(recorded) or recorded != computed:
                    add_error(errors, f"{round_where}.{field}", "does not equal the recomputed finite value")
            if mu_diff is None or mu_diff > thresholds["per_round_mu_max_abs"]:
                observation_failed = True
                add_error(errors, f"{round_where}.mu_max_abs_diff", "fails the per-round mu gate")
            if sigma_diff is None or sigma_diff > thresholds["per_round_sigma_max_abs"]:
                observation_failed = True
                add_error(errors, f"{round_where}.sigma_max_abs_diff", "fails the per-round sigma gate")
            previous = {path: path_records[path] for path in PATHS}

        final_max = require(record, "final_first_action_max_abs_diff", where, errors)
        final_l2 = require(record, "final_first_action_l2_diff", where, errors)
        last = round_map.get(gate["round_count"] - 1)
        if isinstance(last, dict):
            if final_max != last.get("first_action_max_abs_diff"):
                add_error(errors, f"{where}.final_first_action_max_abs_diff", "must equal the final round value")
            if final_l2 != last.get("first_action_l2_diff"):
                add_error(errors, f"{where}.final_first_action_l2_diff", "must equal the final round value")
        if not finite_number(final_max) or final_max > thresholds["final_first_action_max_abs"]:
            observation_failed = True
            add_error(errors, f"{where}.final_first_action_max_abs_diff", "fails the final first-action gate")
        if not finite_number(final_l2):
            observation_failed = True
        if len(errors) == record_errors_before and not observation_failed:
            passed_observations += 1

    if len(observation_ids) != 10:
        add_error(errors, "decision", "must have 10 unique observation_id values")
    return {
        "status": "PASS" if not errors and passed_observations == 10 else "FAIL",
        "observation_count": len(observation_ids),
        "passed_observations": passed_observations,
        "observation_ids": observation_ids,
        "errors": errors,
    }


def verify_system(summary: Any, contract: dict[str, Any], decision_ids: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    config = contract["system_gate"]
    timing = config["timing"]
    if not isinstance(summary, dict):
        return {"status": "FAIL", "errors": ["system summary must be an object"]}
    subset = require(summary, "latency_subset_observation_ids", "system", errors)
    if not isinstance(subset, list) or len(subset) != config["latency_subset"]["count"] or len(set(subset)) != len(subset):
        add_error(errors, "system.latency_subset_observation_ids", "must contain exactly two unique IDs")
        subset = []
    if not all(isinstance(item, str) and item in decision_ids for item in subset):
        add_error(errors, "system.latency_subset_observation_ids", "IDs must come from decision.jsonl")
    expected_metadata = {
        "warmup_repeats": timing["warmup_repeats"],
        "technical_repeats": timing["technical_repeats"],
        "measurement_unit": timing["measurement_unit"],
        "timing_boundary": timing["timing_boundary"],
        "includes_preprocessing": timing["includes_preprocessing"],
        "includes_environment_interaction": timing["includes_environment_interaction"],
        "device_synchronization": timing["device_synchronization"],
        "path_order_policy": "interleaved_seeded",
    }
    for field, expected in expected_metadata.items():
        value = require(summary, field, "system", errors)
        if value != expected:
            add_error(errors, "system", f"{field} must equal frozen value {expected!r}")
    path_order_seed = require(summary, "path_order_seed", "system", errors)
    if not scalar_identifier(path_order_seed):
        add_error(errors, "system.path_order_seed", "must be a scalar identifier")
    records = require(summary, "records", "system", errors)
    if not isinstance(records, list):
        add_error(errors, "system.records", "must be an array")
        return {"status": "FAIL", "errors": errors}

    units: dict[tuple[str, str], dict[str, dict[str, dict[int, tuple[Any, Any]]]]] = {}
    seen: set[tuple[str, str, str, bool, int]] = set()
    call_observations: dict[str, str] = {}
    for index, record in enumerate(records, 1):
        where = f"system record {index}"
        if not isinstance(record, dict):
            add_error(errors, where, "record must be an object")
            continue
        observation = require(record, "observation_id", where, errors)
        call = require(record, "planner_call_id", where, errors)
        path = require(record, "path", where, errors)
        warmup = require(record, "warmup", where, errors)
        repeat = require(record, "technical_repeat", where, errors)
        latency = require(record, "latency_ms", where, errors)
        memory = require(record, "peak_memory_mib", where, errors)
        if not isinstance(observation, str) or observation not in subset:
            add_error(errors, where, "observation_id must be one latency-subset ID")
            continue
        if not isinstance(call, str) or not call:
            add_error(errors, where, "planner_call_id must be a non-empty string")
            continue
        if call in call_observations and call_observations[call] != observation:
            add_error(errors, where, "planner_call_id is paired with multiple observations")
        call_observations[call] = observation
        if path not in PATHS:
            add_error(errors, where, f"path must be one of {list(PATHS)}")
            continue
        if not isinstance(warmup, bool) or not isinstance(repeat, int) or isinstance(repeat, bool):
            add_error(errors, where, "warmup must be boolean and technical_repeat integer")
            continue
        if not finite_number(latency) or latency <= 0:
            add_error(errors, where, "latency_ms must be finite and positive")
        if not finite_number(memory) or memory < 0:
            add_error(errors, where, "peak_memory_mib must be finite and non-negative")
        row_key = (observation, call, path, warmup, repeat)
        if row_key in seen:
            add_error(errors, where, "duplicate timing row")
        seen.add(row_key)
        unit = units.setdefault((observation, call), {path_name: {"warmup": {}, "technical": {}} for path_name in PATHS})
        bucket = "warmup" if warmup else "technical"
        unit[path][bucket][repeat] = (latency, memory)

    reductions: list[float] = []
    memory_ratios: list[float] = []
    expected_warmups = set(range(timing["warmup_repeats"]))
    expected_technical = set(range(timing["technical_repeats"]))
    for unit_id, unit in units.items():
        where = f"system unit {unit_id[0]}/{unit_id[1]}"
        for path in PATHS:
            if set(unit[path]["warmup"]) != expected_warmups:
                add_error(errors, where, f"{path} must contain warmup repeats 0..{timing['warmup_repeats'] - 1}")
            if set(unit[path]["technical"]) != expected_technical:
                add_error(errors, where, f"{path} must contain technical repeats 0..{timing['technical_repeats'] - 1}")
        if any(
            set(unit[path][bucket]) != expected
            for path in PATHS
            for bucket, expected in (("warmup", expected_warmups), ("technical", expected_technical))
        ):
            continue
        baseline = unit["baseline"]["technical"]
        factorized = unit["factorized_cache"]["technical"]
        if any(
            not (finite_number(values[0]) and values[0] > 0 and finite_number(values[1]) and values[1] >= 0)
            for values in list(baseline.values()) + list(factorized.values())
        ):
            continue
        base_latency = statistics.median(baseline[index][0] for index in expected_technical)
        factorized_latency = statistics.median(factorized[index][0] for index in expected_technical)
        reductions.append(1.0 - factorized_latency / base_latency)
        base_memory = max(baseline[index][1] for index in expected_technical)
        factorized_memory = max(factorized[index][1] for index in expected_technical)
        memory_ratios.append(factorized_memory / base_memory if base_memory else (0.0 if factorized_memory == 0 else math.inf))

    if len(units) != config["latency_subset"]["count"]:
        add_error(errors, "system", "must contain exactly two observation/planner-call units")
    if {observation for observation, _ in units} != set(subset):
        add_error(errors, "system", "records do not cover exactly the declared two observations")
    reduction = statistics.median(reductions) if reductions else None
    memory_ratio = max(memory_ratios) if memory_ratios else None
    if reduction is None or reduction < config["latency_gate"]["minimum_paired_median_reduction"]:
        add_error(errors, "system.latency_gate", "paired median reduction is below 0.20")
    if memory_ratio is None or memory_ratio > config["memory_gate"]["maximum_peak_memory_ratio"]:
        add_error(errors, "system.memory_gate", "peak memory ratio exceeds 1.10")
    return {
        "status": "PASS" if not errors else "FAIL",
        "unit_count": len(units),
        "paired_median_reduction": reduction,
        "max_peak_memory_ratio": memory_ratio,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name("APPROX_FREEZE.json"))
    parser.add_argument("--decision", type=Path, required=True, help="decision.jsonl")
    parser.add_argument("--system", type=Path, required=True, help="system_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract = load_json(args.contract)
        if contract.get("schema") != "dino-wm-shared-prefix-cache.approximate-freeze":
            raise ValueError("unexpected approximate experiment contract schema")
        decision = verify_decision(load_jsonl(args.decision), contract)
        system = verify_system(load_json(args.system), contract, decision["observation_ids"])
    except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "INPUT_ERROR", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    authorized = decision["status"] == "PASS" and system["status"] == "PASS"
    result = {
        "experiment_id": contract["experiment_id"],
        "claim_boundary": contract["claim_boundary"],
        "decision_gate": decision,
        "system_gate": system,
        "shadow_closed_loop_authorized": authorized,
        "status": "PASS" if authorized else "FAIL",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if authorized else 1


if __name__ == "__main__":
    raise SystemExit(main())
