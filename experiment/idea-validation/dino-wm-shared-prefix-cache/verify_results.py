#!/usr/bin/env python3
"""Verify the frozen RankSafe-EPC experiment artifacts.

The verifier intentionally uses recorded values and booleans only.  It does
not load a model, run inference, calculate hashes, or apply numeric tolerances.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


PATHS = ("baseline", "iteration_cache", "factorized_cache")
CACHED_PATHS = PATHS[1:]
COMPARISON_FIELDS = (
    "prefix",
    "rollout",
    "objective",
    "full_order",
    "topk30",
    "mu",
    "sigma",
    "first_action",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[Any]:
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


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_tree(value: Any) -> bool:
    if isinstance(value, list):
        return all(_numeric_tree(item) for item in value)
    return _finite_number(value)


def _add(errors: list[str], where: str, message: str) -> None:
    errors.append(f"{where}: {message}")


def _require(mapping: dict[str, Any], key: str, where: str, errors: list[str]) -> Any:
    if key not in mapping:
        _add(errors, where, f"missing required field {key!r}")
        return None
    return mapping[key]


def _verify_tensor_record(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        _add(errors, where, "must be an object")
        return
    equal = _require(value, "equal_to_baseline", where, errors)
    if not isinstance(equal, bool):
        _add(errors, where, "equal_to_baseline must be boolean")
    shape = _require(value, "shape", where, errors)
    if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape):
        _add(errors, where, "shape must be a non-negative integer array")
    dtype = _require(value, "dtype", where, errors)
    if not isinstance(dtype, str) or not dtype:
        _add(errors, where, "dtype must be a non-empty string")
    device = _require(value, "device", where, errors)
    if not isinstance(device, str) or not device:
        _add(errors, where, "device must be a non-empty string")


def _verify_path(
    path_record: Any,
    path_name: str,
    candidate_count: int,
    expected_cache_hit: bool,
    errors: list[str],
) -> None:
    where = f"paths.{path_name}"
    if not isinstance(path_record, dict):
        _add(errors, where, "must be an object")
        return
    cache_hit = _require(path_record, "cache_hit", where, errors)
    if cache_hit is not expected_cache_hit:
        _add(errors, where, f"cache_hit must be {expected_cache_hit}")
    for field in ("prefix", "rollout"):
        value = _require(path_record, field, where, errors)
        _verify_tensor_record(value, f"{where}.{field}", errors)
        if isinstance(value, dict) and value.get("equal_to_baseline") is not True:
            _add(errors, f"{where}.{field}", "must report exact equality to baseline")

    objective = _require(path_record, "objective", where, errors)
    if not isinstance(objective, list) or len(objective) != candidate_count:
        _add(errors, f"{where}.objective", f"must contain exactly {candidate_count} values")
    elif not all(_finite_number(value) for value in objective):
        _add(errors, f"{where}.objective", "must contain only finite numeric values")

    full_order = _require(path_record, "full_order", where, errors)
    if not isinstance(full_order, list) or len(full_order) != candidate_count:
        _add(errors, f"{where}.full_order", f"must contain exactly {candidate_count} indices")
    elif not all(isinstance(value, int) and not isinstance(value, bool) for value in full_order):
        _add(errors, f"{where}.full_order", "must contain integer candidate indices")
    elif set(full_order) != set(range(candidate_count)):
        _add(errors, f"{where}.full_order", "must be a permutation of candidate indices")

    topk = _require(path_record, "topk30", where, errors)
    if not isinstance(topk, list) or len(topk) != 30:
        _add(errors, f"{where}.topk30", "must contain exactly 30 indices")
    elif isinstance(full_order, list) and topk != full_order[:30]:
        _add(errors, f"{where}.topk30", "must equal full_order[:30]")
    if isinstance(topk, list) and not all(isinstance(value, int) and not isinstance(value, bool) for value in topk):
        _add(errors, f"{where}.topk30", "must contain integer candidate indices")

    for field in ("mu", "sigma", "first_action"):
        value = _require(path_record, field, where, errors)
        if not _numeric_tree(value):
            _add(errors, f"{where}.{field}", "must be a non-empty finite numeric array")
        elif value == []:
            _add(errors, f"{where}.{field}", "must not be empty")


def _verify_mechanism(records: list[Any], contract: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    settings = contract["scope"]["fixed_settings"]
    candidate_count = settings["candidate_count"]
    seen_ids: set[str] = set()
    valid_records = 0

    for index, record in enumerate(records, 1):
        where = f"mechanism line {index}"
        if not isinstance(record, dict):
            _add(errors, where, "record must be an object")
            continue
        population_id = _require(record, "population_id", where, errors)
        if not isinstance(population_id, str) or not population_id:
            _add(errors, where, "population_id must be a non-empty string")
        elif population_id in seen_ids:
            _add(errors, where, "population_id is duplicated")
        else:
            seen_ids.add(population_id)
        observation_id = _require(record, "observation_id", where, errors)
        if not isinstance(observation_id, str) or not observation_id:
            _add(errors, where, "observation_id must be a non-empty string")
        population_seed = _require(record, "candidate_population_seed", where, errors)
        if population_seed is None or isinstance(population_seed, (dict, list, bool)):
            _add(errors, where, "candidate_population_seed must be a scalar identifier")
        cem_iteration = _require(record, "cem_iteration", where, errors)
        if not isinstance(cem_iteration, int) or isinstance(cem_iteration, bool) or cem_iteration < 0:
            _add(errors, where, "cem_iteration must be a non-negative integer")
        for field, expected in (
            ("candidate_count", candidate_count),
            ("horizon", settings["horizon"]),
            ("topk", settings["topk"]),
        ):
            value = _require(record, field, where, errors)
            if value != expected:
                _add(errors, where, f"{field} must equal frozen value {expected!r}")

        paths = _require(record, "paths", where, errors)
        if not isinstance(paths, dict) or set(paths) != set(PATHS):
            _add(errors, f"{where}.paths", f"must contain exactly {list(PATHS)}")
            continue
        for path_name in PATHS:
            _verify_path(
                paths[path_name],
                path_name,
                candidate_count,
                expected_cache_hit=(path_name != "baseline"),
                errors=errors,
            )

        baseline = paths["baseline"]
        if not isinstance(baseline, dict):
            continue
        baseline_objective = baseline.get("objective")
        baseline_order = baseline.get("full_order")
        if isinstance(baseline_objective, list) and len(baseline_objective) == candidate_count:
            derived_order = sorted(range(candidate_count), key=lambda item: (baseline_objective[item], item))
            if baseline_order != derived_order:
                _add(errors, f"{where}.paths.baseline.full_order", "does not match stable objective ordering")

        for path_name in CACHED_PATHS:
            current = paths[path_name]
            current_where = f"{where}.paths.{path_name}"
            if not isinstance(current, dict):
                continue
            canary = current.get("canary")
            if not isinstance(canary, dict) or canary.get("executed") is not True:
                _add(errors, f"{current_where}.canary", "must report executed=true")
            if not isinstance(canary, dict) or canary.get("fallback_triggered") is not False:
                _add(errors, f"{current_where}.canary", "must report fallback_triggered=false")
            for field in COMPARISON_FIELDS:
                if field in ("prefix", "rollout"):
                    current_value = current.get(field)
                    if not isinstance(current_value, dict) or current_value.get("equal_to_baseline") is not True:
                        _add(errors, f"{current_where}.{field}", "exact equality comparison failed")
                elif current.get(field) != baseline.get(field):
                    _add(errors, f"{current_where}.{field}", "differs from baseline")
            if current.get("first_action") != baseline.get("first_action"):
                _add(errors, current_where, "first-action mismatch (immediate mechanism failure)")

        controls = _require(record, "negative_controls", where, errors)
        if not isinstance(controls, dict) or set(controls) != {"wrong_boundary", "stale_observation"}:
            _add(errors, f"{where}.negative_controls", "must contain only wrong_boundary and stale_observation")
        elif any(token in controls for token in ("randomized_mask", "randomized-mask", "all_zero_mask")):
            _add(errors, f"{where}.negative_controls", "legacy mask permutation control is forbidden")
        else:
            for control_name in ("wrong_boundary", "stale_observation"):
                control = controls[control_name]
                control_where = f"{where}.negative_controls.{control_name}"
                if not isinstance(control, dict):
                    _add(errors, control_where, "must be an object")
                    continue
                for field in ("fallback_disabled", "mismatch_detected", "downstream_mismatch"):
                    if control.get(field) is not True:
                        _add(errors, control_where, f"{field} must be true")
        valid_records += 1

    if not records:
        _add(errors, "mechanism", "JSONL contains no records")
    return {
        "status": "PASS" if not errors else "FAIL",
        "record_count": valid_records,
        "errors": errors,
    }


def _verify_system(summary: Any, contract: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    config = contract["system_gate"]
    minimum_reduction = config["latency_gate"]["minimum_paired_median_reduction"]
    maximum_memory_ratio = config["memory_gate"]["maximum_peak_memory_ratio"]
    if not isinstance(summary, dict):
        return {"status": "FAIL", "errors": ["system summary must be an object"]}
    for field, expected in (
        ("warmup_repeats", config["warmup_repeats"]),
        ("measurement_unit", config["unit"]),
        ("timing_boundary", config["timing_boundary"]),
        ("includes_preprocessing", config["includes_preprocessing"]),
        ("includes_environment_interaction", config["includes_environment_interaction"]),
        ("device_synchronization", config["device_synchronization"]),
    ):
        value = _require(summary, field, "system", errors)
        if value != expected:
            _add(errors, "system", f"{field} must equal frozen value {expected!r}")
    technical_repeats = _require(summary, "technical_repeats", "system", errors)
    if not isinstance(technical_repeats, int) or technical_repeats < config["minimum_technical_repeats"]:
        _add(errors, "system.technical_repeats", f"must be an integer >= {config['minimum_technical_repeats']}")

    records = _require(summary, "records", "system", errors)
    if not isinstance(records, list):
        _add(errors, "system.records", "must be an array")
        return {"status": "FAIL", "errors": errors}

    units: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    seen_rows: set[tuple[str, str, str, bool, int]] = set()
    call_to_observation: dict[str, str] = {}
    for index, record in enumerate(records, 1):
        where = f"system record {index}"
        if not isinstance(record, dict):
            _add(errors, where, "record must be an object")
            continue
        observation = _require(record, "observation_id", where, errors)
        call = _require(record, "planner_call_id", where, errors)
        path = _require(record, "path", where, errors)
        warmup = _require(record, "warmup", where, errors)
        repeat = _require(record, "technical_repeat", where, errors)
        latency = _require(record, "latency_ms", where, errors)
        memory = _require(record, "peak_memory_mib", where, errors)
        if not isinstance(observation, str) or not isinstance(call, str) or not observation or not call:
            _add(errors, where, "observation_id and planner_call_id must be non-empty strings")
            continue
        if call in call_to_observation and call_to_observation[call] != observation:
            _add(errors, where, "planner_call_id is paired with multiple observations")
        call_to_observation[call] = observation
        if path not in PATHS:
            _add(errors, where, f"path must be one of {list(PATHS)}")
            continue
        if not isinstance(warmup, bool) or not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 0:
            _add(errors, where, "warmup must be boolean and technical_repeat a non-negative integer")
            continue
        if not _finite_number(latency) or latency <= 0:
            _add(errors, where, "latency_ms must be finite and positive")
        if not _finite_number(memory) or memory < 0:
            _add(errors, where, "peak_memory_mib must be finite and non-negative")
        row_key = (observation, call, path, warmup, repeat)
        if row_key in seen_rows:
            _add(errors, where, "duplicate timing row")
        seen_rows.add(row_key)
        unit = units.setdefault((observation, call), {})
        path_rows = unit.setdefault(path, {"warmup": {}, "technical": {}})
        bucket = "warmup" if warmup else "technical"
        path_rows[bucket][repeat] = (latency, memory)

    per_call_reductions: dict[str, list[float]] = {path: [] for path in CACHED_PATHS}
    memory_ratios: dict[str, list[float]] = {path: [] for path in CACHED_PATHS}
    for unit_id, unit in units.items():
        where = f"system unit {unit_id[0]}/{unit_id[1]}"
        if set(unit) != set(PATHS):
            _add(errors, where, f"must contain all three paths {list(PATHS)}")
            continue
        baseline_rows = unit["baseline"]
        warmup_labels = set(baseline_rows["warmup"])
        if len(warmup_labels) != config["warmup_repeats"]:
            _add(errors, where, f"baseline must contain exactly {config['warmup_repeats']} warmup repeats")
        baseline_labels = set(baseline_rows["technical"])
        if len(baseline_labels) < config["minimum_technical_repeats"]:
            _add(errors, where, "baseline has too few technical repeats")
        for path in PATHS:
            rows = unit[path]
            if set(rows["warmup"]) != warmup_labels:
                _add(errors, where, f"{path} warmup labels are not paired")
            if set(rows["technical"]) != baseline_labels:
                _add(errors, where, f"{path} technical labels are not paired")
        if not baseline_labels:
            continue
        if any(set(unit[path]["technical"]) != baseline_labels for path in PATHS):
            continue
        all_timing_values = [
            value
            for path in PATHS
            for bucket in ("warmup", "technical")
            for value in unit[path][bucket].values()
        ]
        if any(
            not (_finite_number(latency) and latency > 0 and _finite_number(memory) and memory >= 0)
            for latency, memory in all_timing_values
        ):
            continue
        base_latency = statistics.median(baseline_rows["technical"][n][0] for n in baseline_labels)
        base_memory = max(baseline_rows["technical"][n][1] for n in baseline_labels)
        if base_latency <= 0:
            continue
        for path in CACHED_PATHS:
            cache_latency = statistics.median(unit[path]["technical"][n][0] for n in baseline_labels)
            cache_memory = max(unit[path]["technical"][n][1] for n in baseline_labels)
            per_call_reductions[path].append(1.0 - cache_latency / base_latency)
            memory_ratios[path].append(cache_memory / base_memory if base_memory else (0.0 if cache_memory == 0 else math.inf))

    report: dict[str, Any] = {"status": "FAIL" if errors else "PASS", "unit_count": len(units), "errors": errors}
    report["paired_median_reduction"] = {}
    report["max_peak_memory_ratio"] = {}
    for path in CACHED_PATHS:
        reductions = per_call_reductions[path]
        ratios = memory_ratios[path]
        reduction = statistics.median(reductions) if reductions else None
        ratio = max(ratios) if ratios else None
        report["paired_median_reduction"][path] = reduction
        report["max_peak_memory_ratio"][path] = ratio
        if reduction is None or reduction < minimum_reduction:
            _add(errors, f"system.{path}", f"paired median latency reduction is below {minimum_reduction:.2f}")
        if ratio is None or ratio > maximum_memory_ratio:
            _add(errors, f"system.{path}", f"peak memory ratio exceeds {maximum_memory_ratio:.2f}")
    report["status"] = "PASS" if not errors else "FAIL"
    return report


def _verify_closed_loop(records: list[Any]) -> dict[str, Any]:
    errors: list[str] = []
    seen: set[tuple[str, int]] = set()
    for index, record in enumerate(records, 1):
        where = f"closed-loop line {index}"
        if not isinstance(record, dict):
            _add(errors, where, "record must be an object")
            continue
        case_id = _require(record, "case_id", where, errors)
        round_id = _require(record, "mpc_round", where, errors)
        paths = _require(record, "paths", where, errors)
        if not isinstance(case_id, str) or not isinstance(round_id, int) or isinstance(round_id, bool):
            _add(errors, where, "case_id must be string and mpc_round integer")
            continue
        key = (case_id, round_id)
        if key in seen:
            _add(errors, where, "duplicate case/MPC-round record")
        seen.add(key)
        if not isinstance(paths, dict) or set(paths) != set(PATHS):
            _add(errors, f"{where}.paths", f"must contain exactly {list(PATHS)}")
            continue
        for path in PATHS:
            result = paths[path]
            if not isinstance(result, dict):
                _add(errors, f"{where}.paths.{path}", "must be an object")
                continue
            action = _require(result, "first_action", f"{where}.paths.{path}", errors)
            success = _require(result, "success", f"{where}.paths.{path}", errors)
            distance = _require(result, "final_goal_distance", f"{where}.paths.{path}", errors)
            if not _numeric_tree(action):
                _add(errors, f"{where}.paths.{path}.first_action", "must be a finite numeric array")
            if not isinstance(success, bool):
                _add(errors, f"{where}.paths.{path}.success", "must be boolean")
            if not _finite_number(distance) or distance < 0:
                _add(errors, f"{where}.paths.{path}.final_goal_distance", "must be finite and non-negative")
        baseline = paths["baseline"]
        if isinstance(baseline, dict):
            for path in CACHED_PATHS:
                current = paths[path]
                if isinstance(current, dict) and current.get("first_action") != baseline.get("first_action"):
                    _add(errors, f"{where}.paths.{path}", "first-action mismatch")
                if isinstance(current, dict) and current.get("success") != baseline.get("success"):
                    _add(errors, f"{where}.paths.{path}", "success outcome mismatch")
    if not records:
        _add(errors, "closed-loop", "JSONL contains no records")
    return {"status": "PASS" if not errors else "FAIL", "record_count": len(records), "errors": errors}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name("EXPERIMENT_FREEZE.json"))
    parser.add_argument("--mechanism", type=Path, help="mechanism JSONL")
    parser.add_argument("--system", type=Path, help="system summary JSON")
    parser.add_argument("--closed-loop", type=Path, help="closed-loop JSONL; only checked after both pre-gates pass")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.mechanism and not args.system:
        print("at least one of --mechanism or --system is required", file=sys.stderr)
        return 2
    try:
        contract = _load_json(args.contract)
        if contract.get("schema") != "dino-wm-shared-prefix-cache.experiment-freeze":
            raise ValueError("unexpected experiment contract schema")
    except (OSError, json.JSONDecodeError, ValueError, AttributeError) as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"experiment_id": contract["experiment_id"]}
    mechanism_pass = False
    system_pass = False
    supplied_pre_gates = 0
    try:
        if args.mechanism:
            supplied_pre_gates += 1
            result["mechanism"] = _verify_mechanism(_load_jsonl(args.mechanism), contract)
            mechanism_pass = result["mechanism"]["status"] == "PASS"
        else:
            result["mechanism"] = {"status": "NOT_RUN", "errors": ["--mechanism was not supplied"]}
        if args.system:
            supplied_pre_gates += 1
            result["system"] = _verify_system(_load_json(args.system), contract)
            system_pass = result["system"]["status"] == "PASS"
        else:
            result["system"] = {"status": "NOT_RUN", "errors": ["--system was not supplied"]}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result["input_error"] = str(exc)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    pre_gates_authorized = bool(args.mechanism and args.system and mechanism_pass and system_pass)
    result["pre_gates_authorized"] = pre_gates_authorized
    if args.closed_loop:
        if not pre_gates_authorized:
            result["closed_loop"] = {
                "status": "NOT_AUTHORIZED",
                "errors": ["closed-loop is authorized only after mechanism and system gates pass"],
            }
        else:
            try:
                result["closed_loop"] = _verify_closed_loop(_load_jsonl(args.closed_loop))
            except (OSError, ValueError) as exc:
                result["closed_loop"] = {"status": "FAIL", "errors": [str(exc)]}
    else:
        result["closed_loop"] = {"status": "NOT_RUN", "errors": []}

    closed_loop_pass = result["closed_loop"]["status"] in ("PASS", "NOT_RUN")
    supplied_pass = (not args.mechanism or mechanism_pass) and (not args.system or system_pass)
    result["status"] = "PASS" if supplied_pre_gates > 0 and supplied_pass and closed_loop_pass else "FAIL"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
