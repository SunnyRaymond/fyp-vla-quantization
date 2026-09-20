#!/usr/bin/env python3
"""Read-only verifier for the frozen Rank Student Stage A.

The verifier only reads JSON/JSONL artifacts.  It recomputes stable rankings
and recall from the recorded full-teacher and student score arrays; it does not
load a checkpoint, run inference, access CUDA, or use hashes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONTRACT_NAME = "RANK_STUDENT_STAGE_A_FREEZE.json"
SCHEMA = "dino-wm-shared-prefix-cache.rank-student-stage-a-freeze"
TEACHER_SCHEMA = "dino-wm-shared-prefix-cache.rank-student-teacher-record"
EVAL_SCHEMA = "dino-wm-shared-prefix-cache.rank-student-evaluation-record"
TRAINING_SCHEMA = "dino-wm-shared-prefix-cache.rank-student-training-summary"
SELECTION_SCHEMA = "dino-wm-shared-prefix-cache.rank-student-selection"

TRAIN_IDS = tuple(f"wall_case_{index:02d}" for index in range(12))
CALIBRATION_IDS = tuple(f"wall_case_{index:02d}" for index in range(12, 16))
HELDOUT_IDS = tuple(f"wall_case_{index:02d}" for index in range(16, 20))
ROUNDS = tuple(range(10))
K = 300
HORIZON = 5
TOPK = 30
OPT_STEPS = 10
M_GRID = (60, 90, 120, 150, 180)
VALUE_TOL = 1e-12
CHAIN_TOL = 1e-12


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


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def scalar_identifier(value: Any) -> bool:
    return (isinstance(value, str) and bool(value)) or (
        isinstance(value, int) and not isinstance(value, bool)
    )


def close(left: float, right: float, tolerance: float = VALUE_TOL) -> bool:
    return math.isclose(left, right, rel_tol=VALUE_TOL, abs_tol=tolerance)


def add_error(errors: dict[str, list[str]], gate: str, where: str, message: str) -> None:
    errors.setdefault(gate, []).append(f"{where}: {message}")


def require(mapping: dict[str, Any], key: str, where: str, errors: dict[str, list[str]], gate: str) -> Any:
    if key not in mapping:
        add_error(errors, gate, where, f"missing required field {key!r}")
        return None
    return mapping[key]


def flatten_numeric(value: Any) -> tuple[Any, list[float]]:
    if isinstance(value, list):
        children = [flatten_numeric(item) for item in value]
        return ("list", tuple(child[0] for child in children)), [
            number for child in children for number in child[1]
        ]
    if finite_number(value):
        return ("scalar",), [float(value)]
    return ("invalid",), []


def numeric_array(value: Any) -> tuple[Any, list[float]]:
    shape, leaves = flatten_numeric(value)
    return shape, leaves


def valid_vector(value: Any) -> bool:
    shape, leaves = numeric_array(value)
    return bool(leaves) and shape != ("invalid",)


def valid_candidate_actions(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != K:
        return False
    return all(isinstance(item, list) and valid_vector(item) for item in value)


def same_numeric_array(left: Any, right: Any) -> bool:
    left_shape, left_values = numeric_array(left)
    right_shape, right_values = numeric_array(right)
    return left_shape == right_shape and len(left_values) == len(right_values) and all(
        close(a, b, CHAIN_TOL) for a, b in zip(left_values, right_values)
    )


def stable_rank(values: list[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (values[index], index))


def stable_topk(values: list[float], k: int = TOPK) -> list[int]:
    return stable_rank(values)[:k]


def exact_ids(value: Any, expected: tuple[str, ...]) -> bool:
    return isinstance(value, list) and value == list(expected)


def validate_contract(contract: Any, errors: dict[str, list[str]]) -> None:
    if not isinstance(contract, dict):
        add_error(errors, "contract", "contract", "must be a JSON object")
        return
    for key, expected in (("schema", SCHEMA), ("schema_version", 1), ("status", "frozen")):
        if contract.get(key) != expected:
            add_error(errors, "contract", f"contract.{key}", f"must equal {expected!r}")

    splits = contract.get("splits")
    if not isinstance(splits, dict):
        add_error(errors, "contract", "contract.splits", "must be an object")
    else:
        for name, expected in (
            ("train", TRAIN_IDS),
            ("calibration", CALIBRATION_IDS),
            ("heldout", HELDOUT_IDS),
        ):
            item = splits.get(name, {})
            if item.get("count") != len(expected) or not exact_ids(item.get("required_ids"), expected):
                add_error(errors, "contract", f"contract.splits.{name}", f"must freeze {list(expected)!r}")
        if splits.get("disjoint_required") is not True:
            add_error(errors, "contract", "contract.splits.disjoint_required", "must be true")

    settings = contract.get("scope", {}).get("fixed_settings", {})
    for key, expected in (
        ("candidate_count", K),
        ("horizon", HORIZON),
        ("topk", TOPK),
        ("cem_opt_steps", OPT_STEPS),
    ):
        if settings.get(key) != expected:
            add_error(errors, "contract", f"contract.scope.fixed_settings.{key}", f"must equal {expected}")
    if contract.get("scope", {}).get("round_count") != len(ROUNDS):
        add_error(errors, "contract", "contract.scope.round_count", "must equal 10")

    selection = contract.get("selection_gate", {})
    if selection.get("candidate_M_grid") != list(M_GRID):
        add_error(errors, "contract", "contract.selection_gate.candidate_M_grid", f"must equal {list(M_GRID)!r}")
    if selection.get("selection_source") != "calibration_only":
        add_error(errors, "contract", "contract.selection_gate.selection_source", "must equal calibration_only")
    if selection.get("no_heldout_tuning") is not True:
        add_error(errors, "contract", "contract.selection_gate.no_heldout_tuning", "must be true")
    if contract.get("heldout_gate", {}).get("selected_M_max") != 180:
        add_error(errors, "contract", "contract.heldout_gate.selected_M_max", "must equal 180")


def validate_common_record(
    record: Any,
    expected_schema: str,
    expected_split: str,
    expected_ids: tuple[str, ...],
    where: str,
    errors: dict[str, list[str]],
    teacher_only: bool,
) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        add_error(errors, "schema_completeness", where, "record must be an object")
        return None
    for key, expected in (("schema", expected_schema), ("schema_version", 1), ("split", expected_split)):
        if record.get(key) != expected:
            add_error(errors, "schema_completeness", f"{where}.{key}", f"must equal {expected!r}")
    observation_id = require(record, "observation_id", where, errors, "schema_completeness")
    if observation_id not in expected_ids:
        add_error(errors, "split_identity", f"{where}.observation_id", "is not in the frozen split")
    if not scalar_identifier(require(record, "observation_seed", where, errors, "schema_completeness")):
        add_error(errors, "schema_completeness", f"{where}.observation_seed", "must be a scalar identifier")
    for key, expected in (
        ("candidate_count", K),
        ("horizon", HORIZON),
        ("topk", TOPK),
        ("cem_opt_steps", OPT_STEPS),
    ):
        if require(record, key, where, errors, "schema_completeness") != expected:
            add_error(errors, "schema_completeness", f"{where}.{key}", f"must equal {expected}")

    rounds = require(record, "rounds", where, errors, "schema_completeness")
    if not isinstance(rounds, list) or len(rounds) != len(ROUNDS):
        add_error(errors, "schema_completeness", f"{where}.rounds", "must contain exactly ten rounds")
        return record
    round_map: dict[int, dict[str, Any]] = {}
    for position, item in enumerate(rounds):
        round_where = f"{where}.rounds[{position}]"
        if not isinstance(item, dict):
            add_error(errors, "schema_completeness", round_where, "round must be an object")
            continue
        round_index = item.get("round_index")
        if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index in round_map:
            add_error(errors, "schema_completeness", round_where, "round_index must be a unique integer")
            continue
        round_map[round_index] = item
    if set(round_map) != set(ROUNDS):
        add_error(errors, "schema_completeness", f"{where}.rounds", "round_index must be exactly 0..9")
        return record

    previous_mu: Any = None
    previous_sigma: Any = None
    for round_index in ROUNDS:
        item = round_map[round_index]
        round_where = f"{where}.rounds[{round_index}]"
        for seed_field in ("candidate_population_seed", "cem_noise_seed"):
            if not scalar_identifier(require(item, seed_field, round_where, errors, "schema_completeness")):
                add_error(errors, "schema_completeness", f"{round_where}.{seed_field}", "must be a scalar identifier")

        candidate_actions = require(item, "candidate_actions", round_where, errors, "schema_completeness")
        if not valid_candidate_actions(candidate_actions):
            add_error(errors, "schema_completeness", f"{round_where}.candidate_actions", "must be 300 non-empty finite numeric action arrays")
        for field in ("input_mu", "input_sigma", "output_mu", "output_sigma", "first_action"):
            value = require(item, field, round_where, errors, "schema_completeness")
            if not valid_vector(value):
                add_error(errors, "schema_completeness", f"{round_where}.{field}", "must be a non-empty finite numeric array")
        input_mu = item.get("input_mu")
        input_sigma = item.get("input_sigma")
        if round_index > 0:
            if not same_numeric_array(input_mu, previous_mu):
                add_error(errors, "schema_completeness", round_where, "input_mu does not continue preceding output_mu")
            if not same_numeric_array(input_sigma, previous_sigma):
                add_error(errors, "schema_completeness", round_where, "input_sigma does not continue preceding output_sigma")
        previous_mu = item.get("output_mu")
        previous_sigma = item.get("output_sigma")

        full = require(item, "J_full", round_where, errors, "schema_completeness")
        if not isinstance(full, list) or len(full) != K or not all(finite_number(value) for value in full):
            add_error(errors, "schema_completeness", f"{round_where}.J_full", f"must contain exactly {K} finite numeric values")

        if item.get("cem_update_source") != "full_teacher":
            add_error(errors, "teacher_only_cem_update", f"{round_where}.cem_update_source", "must equal full_teacher")
        if item.get("student_used_for_cem_update") is not False:
            add_error(errors, "teacher_only_cem_update", f"{round_where}.student_used_for_cem_update", "must be false")
        if teacher_only:
            continue
        for field in ("student_context_scores", "student_action_only_scores"):
            scores = require(item, field, round_where, errors, "schema_completeness")
            if not isinstance(scores, list) or len(scores) != K or not all(finite_number(value) for value in scores):
                add_error(errors, "schema_completeness", f"{round_where}.{field}", f"must contain exactly {K} finite numeric values")
        forbidden = {"selected_M", "candidate_M", "M"}.intersection(item)
        if forbidden:
            add_error(errors, "selection_provenance", round_where, f"per-call M fields are prohibited: {sorted(forbidden)!r}")
    return record


def validate_records(
    records: list[Any],
    expected_schema: str,
    expected_split: str,
    expected_ids: tuple[str, ...],
    errors: dict[str, list[str]],
    teacher_only: bool,
) -> None:
    gate = "split_identity"
    if len(records) != len(expected_ids):
        add_error(errors, gate, expected_split, f"must contain exactly {len(expected_ids)} records")
    seen: list[str] = []
    for number, record in enumerate(records, 1):
        validated = validate_common_record(
            record, expected_schema, expected_split, expected_ids,
            f"{expected_split} line {number}", errors, teacher_only,
        )
        if isinstance(validated, dict):
            observation_id = validated.get("observation_id")
            if observation_id in seen:
                add_error(errors, gate, f"{expected_split} line {number}", "observation_id is duplicated")
            elif observation_id in expected_ids:
                seen.append(observation_id)
    if tuple(seen) != expected_ids:
        add_error(errors, gate, expected_split, f"observation IDs must be exactly {list(expected_ids)!r}")


def call_rows(records: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("rounds"), list):
            continue
        for item in record["rounds"]:
            if isinstance(item, dict):
                rows.append(item)
    return rows


def recall_metrics(records: list[Any], candidate_m: int) -> dict[str, Any]:
    context_recalls: list[float] = []
    action_recalls: list[float] = []
    for item in call_rows(records):
        full = item.get("J_full")
        context = item.get("student_context_scores")
        action = item.get("student_action_only_scores")
        if not (
            isinstance(full, list) and len(full) == K and all(finite_number(value) for value in full)
            and isinstance(context, list) and len(context) == K and all(finite_number(value) for value in context)
            and isinstance(action, list) and len(action) == K and all(finite_number(value) for value in action)
        ):
            continue
        full_top = set(stable_topk([float(value) for value in full]))
        context_top = set(stable_rank([float(value) for value in context])[:candidate_m])
        action_top = set(stable_rank([float(value) for value in action])[:candidate_m])
        context_recalls.append(len(full_top.intersection(context_top)) / TOPK)
        action_recalls.append(len(full_top.intersection(action_top)) / TOPK)

    def summarize(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"call_count": 0, "median_recall": None, "min_recall": None, "recalls": []}
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            median = ordered[middle]
        else:
            median = (ordered[middle - 1] + ordered[middle]) / 2.0
        return {
            "call_count": len(values),
            "median_recall": median,
            "min_recall": min(values),
            "recalls": values,
        }

    return {"context_aware": summarize(context_recalls), "action_only": summarize(action_recalls)}


def metrics_match(recorded: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(recorded, dict):
        return False
    if recorded.get("call_count") != expected["call_count"]:
        return False
    for key in ("median_recall", "min_recall"):
        left = recorded.get(key)
        right = expected[key]
        if right is None:
            if left is not None:
                return False
        elif not finite_number(left) or not close(float(left), float(right)):
            return False
    if "recalls" in recorded:
        values = recorded["recalls"]
        if not isinstance(values, list) or len(values) != len(expected["recalls"]):
            return False
        if not all(finite_number(a) and close(float(a), float(b)) for a, b in zip(values, expected["recalls"])):
            return False
    return True


def validate_training_summary(summary: Any, errors: dict[str, list[str]]) -> None:
    if not isinstance(summary, dict):
        add_error(errors, "training_provenance", "student_training_summary", "must be a JSON object")
        return
    for key, expected in (("schema", TRAINING_SCHEMA), ("schema_version", 1), ("training_status", "completed")):
        if summary.get(key) != expected:
            add_error(errors, "training_provenance", f"student_training_summary.{key}", f"must equal {expected!r}")
    if summary.get("teacher_data_file") != "teacher_data.jsonl":
        add_error(errors, "training_provenance", "student_training_summary.teacher_data_file", "must equal teacher_data.jsonl")
    if summary.get("train_observation_ids") != list(TRAIN_IDS):
        add_error(errors, "training_provenance", "student_training_summary.train_observation_ids", "must equal frozen train IDs")
    if summary.get("teacher_data_observation_count") != len(TRAIN_IDS):
        add_error(errors, "training_provenance", "student_training_summary.teacher_data_observation_count", "must equal 12")
    if summary.get("teacher_data_round_count") != len(TRAIN_IDS) * len(ROUNDS):
        add_error(errors, "training_provenance", "student_training_summary.teacher_data_round_count", "must equal 120")
    if summary.get("uses_train_only") is not True:
        add_error(errors, "no_heldout_tuning", "student_training_summary.uses_train_only", "must be true")
    if summary.get("calibration_used_for_training") is not False:
        add_error(errors, "no_heldout_tuning", "student_training_summary.calibration_used_for_training", "must be false")
    if summary.get("heldout_used_for_training") is not False:
        add_error(errors, "no_heldout_tuning", "student_training_summary.heldout_used_for_training", "must be false")
    context = summary.get("context_aware_student")
    if not isinstance(context, dict) or context.get("status") != "trained":
        add_error(errors, "training_provenance", "student_training_summary.context_aware_student", "must report status trained")
    control = summary.get("action_only_control")
    if not isinstance(control, dict) or control.get("reported") is not True:
        add_error(errors, "context_vs_action_only", "student_training_summary.action_only_control", "must report action-only control")


def validate_selection(
    selection: Any,
    calibration_records: list[Any],
    errors: dict[str, list[str]],
) -> int | None:
    if not isinstance(selection, dict):
        add_error(errors, "selection_provenance", "selection", "must be a JSON object")
        return None
    for key, expected in (("schema", SELECTION_SCHEMA), ("schema_version", 1)):
        if selection.get(key) != expected:
            add_error(errors, "selection_provenance", f"selection.{key}", f"must equal {expected!r}")
    if selection.get("candidate_M_grid") != list(M_GRID):
        add_error(errors, "selection_provenance", "selection.candidate_M_grid", f"must equal {list(M_GRID)!r}")
    if selection.get("selection_rule") != "smallest_feasible_M_calibration_only":
        add_error(errors, "selection_provenance", "selection.selection_rule", "must identify calibration-only smallest feasible M")
    if selection.get("selected_by") != "calibration_only":
        add_error(errors, "selection_provenance", "selection.selected_by", "must equal calibration_only")
    if selection.get("calibration_observation_ids") != list(CALIBRATION_IDS):
        add_error(errors, "selection_provenance", "selection.calibration_observation_ids", "must equal frozen calibration IDs")
    if selection.get("heldout_observation_ids") != list(HELDOUT_IDS):
        add_error(errors, "selection_provenance", "selection.heldout_observation_ids", "must equal frozen heldout IDs")
    if selection.get("heldout_used_for_selection") is not False:
        add_error(errors, "no_heldout_tuning", "selection.heldout_used_for_selection", "must be false")

    expected_metrics = {str(candidate_m): recall_metrics(calibration_records, candidate_m) for candidate_m in M_GRID}
    recorded_metrics = selection.get("calibration_metrics")
    if not isinstance(recorded_metrics, dict):
        add_error(errors, "selection_provenance", "selection.calibration_metrics", "must record every frozen M")
    else:
        for candidate_m in M_GRID:
            recorded = recorded_metrics.get(str(candidate_m))
            expected = expected_metrics[str(candidate_m)]
            if not isinstance(recorded, dict):
                add_error(errors, "selection_provenance", f"selection.calibration_metrics.{candidate_m}", "is missing")
                continue
            for metric_name in ("context_aware", "action_only"):
                if not metrics_match(recorded.get(metric_name), expected[metric_name]):
                    add_error(errors, "selection_provenance", f"selection.calibration_metrics.{candidate_m}.{metric_name}", "does not match recomputed metrics")

    feasible = []
    for candidate_m in M_GRID:
        context_metrics = expected_metrics[str(candidate_m)]["context_aware"]
        if (
            context_metrics["call_count"] == len(CALIBRATION_IDS) * len(ROUNDS)
            and context_metrics["median_recall"] == 1.0
            and context_metrics["min_recall"] >= 29 / 30
        ):
            feasible.append(candidate_m)
    expected_selected = min(feasible) if feasible else None
    selected_m = selection.get("selected_M")
    if selected_m is not None and (not isinstance(selected_m, int) or isinstance(selected_m, bool) or selected_m not in M_GRID):
        add_error(errors, "selection_provenance", "selection.selected_M", "must be null or one frozen candidate M")
    elif selected_m != expected_selected:
        add_error(errors, "selection_provenance", "selection.selected_M", f"must equal recomputed calibration choice {expected_selected!r}")
    return selected_m if isinstance(selected_m, int) and not isinstance(selected_m, bool) else None


def verify(
    contract: Any,
    teacher_records: list[Any],
    training_summary: Any,
    calibration_records: list[Any],
    heldout_records: list[Any],
    selection: Any,
) -> dict[str, Any]:
    errors: dict[str, list[str]] = {}
    validate_contract(contract, errors)
    validate_records(teacher_records, TEACHER_SCHEMA, "train", TRAIN_IDS, errors, teacher_only=True)
    validate_training_summary(training_summary, errors)
    validate_records(calibration_records, EVAL_SCHEMA, "calibration", CALIBRATION_IDS, errors, teacher_only=False)
    validate_records(heldout_records, EVAL_SCHEMA, "heldout", HELDOUT_IDS, errors, teacher_only=False)
    selected_m = validate_selection(selection, calibration_records, errors)

    heldout_metrics: dict[str, Any] | None = None
    heldout_pass = False
    if selected_m is not None:
        heldout_metrics = recall_metrics(heldout_records, selected_m)
        context = heldout_metrics["context_aware"]
        heldout_pass = (
            selected_m <= 180
            and context["call_count"] == len(HELDOUT_IDS) * len(ROUNDS)
            and context["median_recall"] == 1.0
            and context["min_recall"] >= 29 / 30
        )
        if not heldout_pass:
            add_error(errors, "heldout_gate", "heldout", "selected fixed M does not meet median/min context-aware recall gate")
    else:
        add_error(errors, "heldout_gate", "heldout", "cannot evaluate heldout gate without a calibration-selected M")

    contract_pass = not errors.get("contract")
    split_pass = not errors.get("split_identity")
    schema_pass = not errors.get("schema_completeness")
    training_pass = not errors.get("training_provenance")
    provenance_pass = not errors.get("selection_provenance")
    no_tuning_pass = not errors.get("no_heldout_tuning")
    teacher_update_pass = not errors.get("teacher_only_cem_update")
    metric_pass = not errors.get("context_vs_action_only")
    heldout_gate_pass = heldout_pass and not errors.get("heldout_gate")
    verdict = "PASS" if all((contract_pass, split_pass, schema_pass, training_pass, provenance_pass, no_tuning_pass, teacher_update_pass, metric_pass, heldout_gate_pass)) else "FAIL"
    flat_errors = [message for messages in errors.values() for message in messages]
    return {
        "verdict": verdict,
        "stage": "stage_a_teacher_student_calibration_heldout_offline",
        "decision": {
            "selected_M": selected_m,
            "stage_a_offline_screen": verdict,
            "later_chained_stage_b_authorized": verdict == "PASS",
            "latency_claim_authorized": False,
            "closed_loop_claim_authorized": False,
        },
        "gates": {
            "contract": contract_pass,
            "split_identity": split_pass,
            "schema_completeness": schema_pass,
            "train_only_student_training": training_pass,
            "selection_provenance": provenance_pass,
            "no_heldout_tuning": no_tuning_pass,
            "teacher_only_cem_update": teacher_update_pass,
            "context_vs_action_only_recorded": metric_pass,
            "heldout_context_aware_gate": heldout_gate_pass,
        },
        "metrics": {
            "train_observations": len(teacher_records),
            "calibration_calls": len(call_rows(calibration_records)),
            "heldout_calls": len(call_rows(heldout_records)),
            "calibration": {str(candidate_m): recall_metrics(calibration_records, candidate_m) for candidate_m in M_GRID},
            "heldout_at_selected_M": heldout_metrics,
        },
        "errors": flat_errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="artifact directory")
    parser.add_argument("--contract", type=Path, help="frozen contract JSON")
    parser.add_argument("--teacher-data", type=Path, help="train teacher data JSONL")
    parser.add_argument("--training-summary", type=Path, help="student training summary JSON")
    parser.add_argument("--calibration", type=Path, help="calibration results JSONL")
    parser.add_argument("--heldout", type=Path, help="heldout results JSONL")
    parser.add_argument("--selection", type=Path, help="calibration selection JSON")
    parser.add_argument("--output", type=Path, help="verifier result JSON; defaults to verifier.json under root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    paths = {
        "contract": args.contract or root / CONTRACT_NAME,
        "teacher_data": args.teacher_data or root / "teacher_data.jsonl",
        "training_summary": args.training_summary or root / "student_training_summary.json",
        "calibration": args.calibration or root / "calibration_results.jsonl",
        "heldout": args.heldout or root / "heldout_results.jsonl",
        "selection": args.selection or root / "selection.json",
    }
    output = args.output or root / "verifier.json"
    try:
        contract = load_json(paths["contract"])
        teacher_records = load_jsonl(paths["teacher_data"])
        training_summary = load_json(paths["training_summary"])
        calibration_records = load_jsonl(paths["calibration"])
        heldout_records = load_jsonl(paths["heldout"])
        selection = load_json(paths["selection"])
        result = verify(contract, teacher_records, training_summary, calibration_records, heldout_records, selection)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"verdict": "ERROR", "error": str(exc)}
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except OSError:
            pass
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
