#!/usr/bin/env python3
"""Read-only verifier for the frozen EBMF-CEM Stage A calibration.

The verifier reads JSON/JSONL records only.  It does not load a model, run
inference, access CUDA, write result files, or compute hashes.  Its purpose is
to make the calibration provenance and the pre-frozen ambiguity gate explicit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


OBSERVATION_IDS = tuple(f"wall_case_{index:02d}" for index in range(6))
REQUIRED_ROUNDS = tuple(range(10))
K = 300
TOPK = 30
CHAIN_TOL = 1e-12
VALUE_TOL = 1e-12


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


def add_error(gate_errors: dict[str, list[str]], gate: str, where: str, message: str) -> None:
    gate_errors.setdefault(gate, []).append(f"{where}: {message}")


def numeric_array(value: Any, where: str, gate_errors: dict[str, list[str]], gate: str) -> tuple[Any, list[float]]:
    """Return a shape signature and finite leaves for a non-empty numeric array."""

    if not isinstance(value, list) or not value:
        add_error(gate_errors, gate, where, "must be a non-empty numeric array")
        return ("invalid",), []
    child_results = [numeric_array(item, where, gate_errors, gate) if isinstance(item, list) else _number_leaf(item, where, gate_errors, gate) for item in value]
    shape = ("list", tuple(result[0] for result in child_results))
    leaves = [number for result in child_results for number in result[1]]
    return shape, leaves


def _number_leaf(value: Any, where: str, gate_errors: dict[str, list[str]], gate: str) -> tuple[Any, list[float]]:
    if not finite_number(value):
        add_error(gate_errors, gate, where, "all values must be finite numbers")
        return ("scalar",), []
    return ("scalar",), [float(value)]


def require(mapping: dict[str, Any], key: str, where: str, gate_errors: dict[str, list[str]], gate: str) -> Any:
    if key not in mapping:
        add_error(gate_errors, gate, where, f"missing required field {key!r}")
        return None
    return mapping[key]


def stable_topk(values: list[float], k: int = TOPK) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (values[index], index))[:k]


def same_numeric_array(left: Any, right: Any) -> bool:
    left_shape, left_values = flatten_numeric(left)
    right_shape, right_values = flatten_numeric(right)
    return left_shape == right_shape and len(left_values) == len(right_values) and all(
        close(a, b, CHAIN_TOL) for a, b in zip(left_values, right_values)
    )


def flatten_numeric(value: Any) -> tuple[Any, list[float]]:
    if isinstance(value, list):
        children = [flatten_numeric(item) for item in value]
        return ("list", tuple(child[0] for child in children)), [
            number for child in children for number in child[1]
        ]
    if finite_number(value):
        return ("scalar",), [float(value)]
    return ("invalid",), []


def validate_contract(contract: Any, gate_errors: dict[str, list[str]]) -> None:
    if not isinstance(contract, dict):
        add_error(gate_errors, "contract", "contract", "must be a JSON object")
        return
    expected = {
        "schema": "dino-wm-shared-prefix-cache.ebmf-cem-stage-a-freeze",
        "schema_version": 1,
        "status": "frozen",
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            add_error(gate_errors, "contract", f"contract.{key}", f"must equal {value!r}")
    scope = contract.get("scope")
    if not isinstance(scope, dict):
        add_error(gate_errors, "contract", "contract.scope", "must be an object")
        return
    observations = scope.get("observations", {})
    if observations.get("count") != 6 or observations.get("required_ids") != list(OBSERVATION_IDS):
        add_error(gate_errors, "contract", "contract.scope.observations", "must freeze six wall_case_00..05 observations")
    if scope.get("round_count") != 10:
        add_error(gate_errors, "contract", "contract.scope.round_count", "must equal 10")
    settings = scope.get("fixed_settings", {})
    for key, value in (("candidate_count", K), ("horizon", 5), ("topk", TOPK), ("cem_opt_steps", 10)):
        if settings.get(key) != value:
            add_error(gate_errors, "contract", f"contract.scope.fixed_settings.{key}", f"must equal {value}")
    evaluator = scope.get("cheap_evaluator", {})
    for key, value in (("name", "cheap_L4"), ("layer_count", 4), ("total_predictor_layers", 6)):
        if evaluator.get(key) != value:
            add_error(gate_errors, "contract", f"contract.scope.cheap_evaluator.{key}", f"must equal {value!r}")
    ambiguity = contract.get("ambiguity_gate", {})
    if ambiguity.get("median_max") != 0.25 or ambiguity.get("p90_max") != 0.4:
        add_error(gate_errors, "contract", "contract.ambiguity_gate", "must freeze median<=0.25 and p90<=0.40")


def verify_records(records: list[Any], contract: dict[str, Any]) -> dict[str, Any]:
    gate_errors: dict[str, list[str]] = {}
    validate_contract(contract, gate_errors)
    calibration_rows: list[dict[str, Any]] = []
    seen_ids: list[str] = []

    if len(records) != len(OBSERVATION_IDS):
        add_error(gate_errors, "completeness_finite", "calibration", "must contain exactly six non-empty JSONL records")

    settings = contract.get("scope", {}).get("fixed_settings", {})
    expected_evaluator = {"name": "cheap_L4", "layer_count": 4, "total_predictor_layers": 6}

    for record_number, record in enumerate(records, 1):
        where = f"calibration line {record_number}"
        if not isinstance(record, dict):
            add_error(gate_errors, "completeness_finite", where, "record must be an object")
            continue
        observation_id = require(record, "observation_id", where, gate_errors, "completeness_finite")
        if not isinstance(observation_id, str) or not observation_id:
            add_error(gate_errors, "completeness_finite", where, "observation_id must be a non-empty string")
        elif observation_id in seen_ids:
            add_error(gate_errors, "completeness_finite", where, "observation_id is duplicated")
        else:
            seen_ids.append(observation_id)
        observation_seed = require(record, "observation_seed", where, gate_errors, "completeness_finite")
        if not scalar_identifier(observation_seed):
            add_error(gate_errors, "completeness_finite", where, "observation_seed must be a scalar identifier")
        for field, expected in (("candidate_count", K), ("horizon", 5), ("topk", TOPK), ("cem_opt_steps", 10)):
            if require(record, field, where, gate_errors, "completeness_finite") != expected:
                add_error(gate_errors, "completeness_finite", where, f"{field} must equal frozen value {expected}")

        evaluator = require(record, "cheap_evaluator", where, gate_errors, "cheap_identity")
        if not isinstance(evaluator, dict) or any(evaluator.get(key) != value for key, value in expected_evaluator.items()):
            add_error(gate_errors, "cheap_identity", where, "cheap_evaluator must identify cheap_L4 as exactly 4/6 layers")

        rounds = require(record, "rounds", where, gate_errors, "completeness_finite")
        if not isinstance(rounds, list) or len(rounds) != 10:
            add_error(gate_errors, "completeness_finite", f"{where}.rounds", "must contain exactly ten rounds")
            continue

        round_map: dict[int, dict[str, Any]] = {}
        for round_number, round_record in enumerate(rounds):
            round_where = f"{where}.rounds[{round_number}]"
            if not isinstance(round_record, dict):
                add_error(gate_errors, "completeness_finite", round_where, "round must be an object")
                continue
            round_index = require(round_record, "round_index", round_where, gate_errors, "completeness_finite")
            if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index in round_map:
                add_error(gate_errors, "completeness_finite", round_where, "round_index must be a unique integer")
                continue
            round_map[round_index] = round_record
        if set(round_map) != set(REQUIRED_ROUNDS):
            add_error(gate_errors, "completeness_finite", f"{where}.rounds", "round_index must be exactly 0..9")
            continue

        previous_output_mu: Any = None
        previous_output_sigma: Any = None
        for round_index in REQUIRED_ROUNDS:
            round_record = round_map[round_index]
            round_where = f"{where}.rounds[{round_index}]"
            for seed_field in ("candidate_population_seed", "cem_noise_seed"):
                seed = require(round_record, seed_field, round_where, gate_errors, "completeness_finite")
                if not scalar_identifier(seed):
                    add_error(gate_errors, "completeness_finite", f"{round_where}.{seed_field}", "must be a scalar identifier")

            arrays: dict[str, Any] = {}
            for field in ("input_mu", "input_sigma", "output_mu", "output_sigma", "first_action"):
                value = require(round_record, field, round_where, gate_errors, "completeness_finite")
                shape, leaves = numeric_array(value, f"{round_where}.{field}", gate_errors, "completeness_finite")
                arrays[field] = value
                if shape == ("invalid",) or not leaves:
                    continue

            if round_index > 0:
                if not same_numeric_array(arrays["input_mu"], previous_output_mu):
                    add_error(gate_errors, "completeness_finite", round_where, "input_mu does not continue the previous output_mu")
                if not same_numeric_array(arrays["input_sigma"], previous_output_sigma):
                    add_error(gate_errors, "completeness_finite", round_where, "input_sigma does not continue the previous output_sigma")
            previous_output_mu = arrays["output_mu"]
            previous_output_sigma = arrays["output_sigma"]

            j_tilde = require(round_record, "J_tilde", round_where, gate_errors, "completeness_finite")
            j_full = require(round_record, "J_full", round_where, gate_errors, "completeness_finite")
            abs_error = require(round_record, "abs_error", round_where, gate_errors, "completeness_finite")
            value_arrays: dict[str, list[float]] = {}
            for field, value in (("J_tilde", j_tilde), ("J_full", j_full), ("abs_error", abs_error)):
                if not isinstance(value, list) or len(value) != K:
                    add_error(gate_errors, "completeness_finite", f"{round_where}.{field}", f"must contain exactly {K} values")
                    value_arrays[field] = []
                    continue
                _, leaves = numeric_array(value, f"{round_where}.{field}", gate_errors, "completeness_finite")
                value_arrays[field] = leaves if len(leaves) == K else []
            if not value_arrays["J_tilde"] or not value_arrays["J_full"] or not value_arrays["abs_error"]:
                continue

            recomputed_errors = [abs(a - b) for a, b in zip(value_arrays["J_tilde"], value_arrays["J_full"])]
            for index, (recorded, expected) in enumerate(zip(value_arrays["abs_error"], recomputed_errors)):
                if not close(recorded, expected):
                    add_error(gate_errors, "completeness_finite", f"{round_where}.abs_error[{index}]", "does not equal abs(J_tilde-J_full)")
                    break

            cheap_top30 = require(round_record, "cheap_top30", round_where, gate_errors, "completeness_finite")
            full_top30 = require(round_record, "full_top30", round_where, gate_errors, "completeness_finite")
            for field, indices in (("cheap_top30", cheap_top30), ("full_top30", full_top30)):
                if not isinstance(indices, list) or len(indices) != TOPK or len(set(indices)) != TOPK or not all(
                    isinstance(index, int) and not isinstance(index, bool) and 0 <= index < K for index in indices
                ):
                    add_error(gate_errors, "completeness_finite", f"{round_where}.{field}", f"must contain {TOPK} unique integer indices in [0,299]")
                else:
                    expected_indices = stable_topk(value_arrays["J_tilde" if field == "cheap_top30" else "J_full"])
                    if indices != expected_indices:
                        add_error(gate_errors, "completeness_finite", f"{round_where}.{field}", "does not match stable objective ranking")

            full_cutoff = require(round_record, "full_cutoff", round_where, gate_errors, "completeness_finite")
            expected_full_top30 = stable_topk(value_arrays["J_full"])
            expected_cutoff = value_arrays["J_full"][expected_full_top30[-1]]
            if not finite_number(full_cutoff) or not close(float(full_cutoff), expected_cutoff):
                add_error(gate_errors, "completeness_finite", f"{round_where}.full_cutoff", "does not equal the 30th stable-ranked full objective")

            if "epsilon" in round_record or "epsilon_global" in round_record:
                add_error(gate_errors, "epsilon_provenance", round_where, "alternate epsilon field is prohibited")
            if "epsilon_source" in round_record and round_record["epsilon_source"] != "calibration_max_abs_error":
                add_error(gate_errors, "epsilon_provenance", round_where, "epsilon_source must be calibration_max_abs_error")
            epsilon_round = require(round_record, "epsilon_round", round_where, gate_errors, "epsilon_provenance")
            if not finite_number(epsilon_round) or float(epsilon_round) < 0:
                add_error(gate_errors, "epsilon_provenance", f"{round_where}.epsilon_round", "must be a finite non-negative number")

            calibration_rows.append({
                "observation_id": observation_id,
                "round_index": round_index,
                "round": round_record,
                "j_tilde": value_arrays["J_tilde"],
                "j_full": value_arrays["J_full"],
                "abs_error": recomputed_errors,
            })

    if set(seen_ids) != set(OBSERVATION_IDS):
        add_error(gate_errors, "completeness_finite", "calibration.observation_id", "must be exactly wall_case_00..wall_case_05")

    epsilon_by_round: dict[int, float] = {}
    for round_index in REQUIRED_ROUNDS:
        rows = [row for row in calibration_rows if row["round_index"] == round_index]
        if len(rows) != len(OBSERVATION_IDS):
            add_error(gate_errors, "epsilon_provenance", f"round {round_index}", "must have six calibration observations")
            continue
        epsilon_by_round[round_index] = max(max(row["abs_error"]) for row in rows)

    ambiguity_ratios: list[float] = []
    for row in calibration_rows:
        round_index = row["round_index"]
        if round_index not in epsilon_by_round:
            continue
        round_record = row["round"]
        epsilon = epsilon_by_round[round_index]
        recorded_epsilon = round_record.get("epsilon_round")
        if not finite_number(recorded_epsilon) or not close(float(recorded_epsilon), epsilon):
            add_error(gate_errors, "epsilon_provenance", f"{row['observation_id']}.rounds[{round_index}].epsilon_round", "does not equal calibration maximum absolute error")

        upper = [value + epsilon for value in row["j_tilde"]]
        tau = upper[stable_topk(upper)[-1]]
        recorded_tau = round_record.get("tau")
        if not finite_number(recorded_tau) or not close(float(recorded_tau), tau):
            add_error(
                gate_errors,
                "completeness_finite",
                f"{row['observation_id']}.rounds[{round_index}].tau",
                "does not equal the frozen 30th stable-ranked upper bound",
            )
        expected_a = {index for index, value in enumerate(row["j_tilde"]) if value - epsilon <= tau}
        recorded_a = round_record.get("A")
        if not isinstance(recorded_a, list) or len(set(recorded_a)) != len(recorded_a) or not all(
            isinstance(index, int) and not isinstance(index, bool) and 0 <= index < K for index in recorded_a
        ) or set(recorded_a) != expected_a:
            add_error(gate_errors, "completeness_finite", f"{row['observation_id']}.rounds[{round_index}].A", "does not match the frozen interval rule")
            actual_a = expected_a
        else:
            actual_a = set(recorded_a)
        expected_coverage = set(stable_topk(row["j_full"])).issubset(expected_a)
        if round_record.get("full_top30_in_A") is not expected_coverage:
            add_error(gate_errors, "completeness_finite", f"{row['observation_id']}.rounds[{round_index}].full_top30_in_A", "does not match recomputed coverage")
        ambiguity_ratios.append(len(actual_a) / K)

    median_ratio: float | None = None
    p90_ratio: float | None = None
    ambiguity_pass = False
    if len(ambiguity_ratios) == 60:
        ordered = sorted(ambiguity_ratios)
        median_ratio = (ordered[29] + ordered[30]) / 2.0
        p90_ratio = ordered[53]  # nearest-rank ceil(0.90*60)=54, one-based
        ambiguity_pass = median_ratio <= 0.25 and p90_ratio <= 0.40
        if not ambiguity_pass:
            add_error(gate_errors, "ambiguity_gate", "ambiguity_gate", "median or nearest-rank p90 exceeds the frozen threshold")
    else:
        add_error(gate_errors, "ambiguity_gate", "ambiguity_gate", "requires exactly 60 valid observation-round records")

    completeness_pass = not gate_errors.get("completeness_finite")
    cheap_identity_pass = not gate_errors.get("cheap_identity")
    epsilon_pass = not gate_errors.get("epsilon_provenance")
    contract_pass = not gate_errors.get("contract")
    readiness_pass = contract_pass and completeness_pass and cheap_identity_pass and epsilon_pass
    verdict = "PASS" if readiness_pass and ambiguity_pass else "FAIL"
    errors = [error for messages in gate_errors.values() for error in messages]
    return {
        "verdict": verdict,
        "stage": "stage_a_calibration",
        "gate": {
            "contract": contract_pass,
            "completeness_finite": completeness_pass,
            "cheap_identity_4_of_6": cheap_identity_pass,
            "epsilon_calibration_only": epsilon_pass,
            "ambiguity": ambiguity_pass,
            "stage_b_held_out_authorized": verdict == "PASS"
        },
        "metrics": {
            "valid_observation_records": len(seen_ids),
            "valid_observation_round_records": len(calibration_rows),
            "epsilon_round": [epsilon_by_round[index] for index in REQUIRED_ROUNDS if index in epsilon_by_round],
            "ambiguity_ratio_count": len(ambiguity_ratios),
            "median_ambiguity_ratio": median_ratio,
            "p90_ambiguity_ratio_nearest_rank": p90_ratio,
            "decision_claim_authorized": False,
            "latency_claim_authorized": False
        },
        "errors": errors
    }


def synthetic_records(failing: bool = False) -> list[dict[str, Any]]:
    """Make a tiny deterministic schema exercise without loading any model."""

    records: list[dict[str, Any]] = []
    for observation_index, observation_id in enumerate(OBSERVATION_IDS):
        rounds: list[dict[str, Any]] = []
        previous_mu = [0.0, 0.0]
        previous_sigma = [1.0, 1.0]
        for round_index in REQUIRED_ROUNDS:
            j_tilde = [float(index) + observation_index * 0.0001 + round_index * 0.00001 for index in range(K)]
            j_full = [value + (0.0002 if index == 0 else 0.0) for index, value in enumerate(j_tilde)]
            abs_error = [abs(a - b) for a, b in zip(j_tilde, j_full)]
            cheap_top30 = stable_topk(j_tilde)
            full_top30 = stable_topk(j_full)
            synthetic_epsilon = 0.0002
            upper = [value + synthetic_epsilon for value in j_tilde]
            tau = upper[stable_topk(upper)[-1]]
            rounds.append({
                "round_index": round_index,
                "candidate_population_seed": 110000 + observation_index * 100 + round_index,
                "cem_noise_seed": 120000 + observation_index * 100 + round_index,
                "input_mu": previous_mu,
                "input_sigma": previous_sigma,
                "output_mu": [round_index + 0.1, round_index + 0.2],
                "output_sigma": [1.0, 1.0],
                "first_action": [round_index + 0.1],
                "J_tilde": j_tilde,
                "J_full": j_full,
                "abs_error": abs_error,
                "cheap_top30": cheap_top30,
                "full_top30": full_top30,
                "full_cutoff": j_full[full_top30[-1]],
                "epsilon_round": synthetic_epsilon,
                "epsilon_source": "calibration_max_abs_error",
                "tau": tau,
                "A": list(range(30)),
                "full_top30_in_A": True
            })
            previous_mu = rounds[-1]["output_mu"]
            previous_sigma = rounds[-1]["output_sigma"]
        records.append({
            "observation_id": observation_id,
            "observation_seed": 1000 + observation_index,
            "candidate_count": K,
            "horizon": 5,
            "topk": TOPK,
            "cem_opt_steps": 10,
            "cheap_evaluator": {"name": "cheap_L4", "layer_count": 4, "total_predictor_layers": 6},
            "rounds": rounds
        })
    if failing:
        records[0]["cheap_evaluator"]["layer_count"] = 5
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name("EBMF_CEM_STAGE_A_FREEZE.json"))
    parser.add_argument("--calibration", type=Path, help="Stage A calibration JSONL")
    parser.add_argument("--synthetic", choices=("pass", "fail"), help="run an in-memory synthetic verifier exercise")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        contract = load_json(args.contract)
        if args.synthetic:
            records = synthetic_records(failing=args.synthetic == "fail")
        elif args.calibration:
            records = load_jsonl(args.calibration)
        else:
            print("provide --calibration PATH or --synthetic pass|fail", file=sys.stderr)
            return 2
        result = verify_records(records, contract)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"verdict": "ERROR", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
