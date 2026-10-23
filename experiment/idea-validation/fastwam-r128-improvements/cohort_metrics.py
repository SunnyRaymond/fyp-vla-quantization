"""Aggregate the frozen fixed-input Fast-WAM R128 Joint8 comparison."""
from __future__ import annotations

import json
import math
from pathlib import Path


PROTOCOL = "fastwam-vq-r128-joint-c8-v1"
CASE_IDS = tuple(range(3000, 3040))
SEED_INDICES = (0, 1)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
C8_CASE_IDS = (2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070)
C8_SUITE_BY_CASE = {
    2000: "libero_spatial", 2019: "libero_spatial",
    2021: "libero_object", 2025: "libero_object",
    2051: "libero_goal", 2057: "libero_goal",
    2066: "libero_10", 2070: "libero_10",
}
LATENT_WEIGHT = 0.04245452683169643
HELDOUT_ARMS = ("H_W16A4", "R8_W16A4", "H_VQ8", "R8_VQ8", "Joint8")
CALIBRATION_ARMS = HELDOUT_ARMS + ("H_VQ0", "R8_VQ0")
METRICS = (
    "first10motor_rmse",
    "future_latent_nmse",
    "mixed_loss",
    "full32motor_rmse",
    "late22motor_rmse",
    "first10gripper_rmse",
    "first10motor_mse",
)
LABELS = {
    "first10motor_rmse": "First10 motor RMSE",
    "future_latent_nmse": "Future latent NMSE",
    "mixed_loss": "Mixed loss",
    "full32motor_rmse": "Full32 motor RMSE",
    "late22motor_rmse": "Late22 motor RMSE",
    "first10gripper_rmse": "First10 gripper RMSE",
    "first10motor_mse": "First10 motor MSE",
}


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _normalized_record(record, latent_weight):
    row = dict(record)
    rmse = _finite_nonnegative(row.get("first10motor_rmse"), "first10motor_rmse")
    row["first10motor_mse"] = _finite_nonnegative(
        row.get("first10motor_mse", rmse * rmse), "first10motor_mse"
    )
    if "mixed_loss" not in row:
        weight = _finite_nonnegative(latent_weight, "latent_weight")
        future = _finite_nonnegative(row.get("future_latent_nmse"), "future_latent_nmse")
        row["mixed_loss"] = row["first10motor_mse"] + weight * future
    for metric in METRICS:
        row[metric] = _finite_nonnegative(row.get(metric), metric)
    return row


def aggregate_records(
    records,
    latent_weight=LATENT_WEIGHT,
    *,
    case_ids=CASE_IDS,
    seed_indices=SEED_INDICES,
    suite_by_case=None,
    cases_per_suite=10,
):
    """Mean seed -> case -> suite, then weight the four suites equally."""
    case_ids, seed_indices = tuple(case_ids), tuple(seed_indices)
    normalized = [_normalized_record(row, latent_weight) for row in records]
    expected = {(case, seed) for case in case_ids for seed in seed_indices}
    found, case_suites = {}, {}
    for row in normalized:
        case, seed = row.get("case_id"), row.get("seed_index")
        if (isinstance(case, bool) or not isinstance(case, int)
                or isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("case_id and seed_index must be integers")
        key = (case, seed)
        if key in found:
            raise ValueError(f"duplicate query for case {case}, seed {seed}")
        if key not in expected:
            raise ValueError(f"unexpected query identity: {key}")
        suite = row.get("suite")
        if suite not in SUITES:
            raise ValueError(f"unexpected suite for case {case}: {suite!r}")
        if suite_by_case is not None and suite != suite_by_case.get(case):
            raise ValueError(f"unexpected suite binding for case {case}: {suite!r}")
        if case in case_suites and case_suites[case] != suite:
            raise ValueError(f"case {case} has inconsistent suite bindings")
        case_suites[case] = suite
        found[key] = row
    if found.keys() != expected:
        missing = sorted(expected - found.keys())
        raise ValueError(f"expected exactly {len(expected)} case-seed queries; missing {missing[:8]}")

    by_case, suite_cases = {}, {suite: [] for suite in SUITES}
    for case in case_ids:
        rows = [found[(case, seed)] for seed in seed_indices]
        means = {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
        suite = case_suites[case]
        by_case[str(case)] = {"suite": suite, **means}
        suite_cases[suite].append(means)
    if any(len(suite_cases[suite]) != cases_per_suite for suite in SUITES):
        raise ValueError(f"each suite must contain exactly {cases_per_suite} cases")
    by_suite = {
        suite: {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
        for suite, rows in suite_cases.items()
    }
    overall = {
        metric: sum(by_suite[suite][metric] for suite in SUITES) / len(SUITES)
        for metric in METRICS
    }
    return {
        "queries": len(normalized),
        "cases": len(by_case),
        "seed_indices": list(seed_indices),
        "aggregation": "mean seeds within case, mean cases within suite, mean four suites equally",
        "overall": overall,
        "by_suite": by_suite,
        "by_case": by_case,
    }


def _relative_change(candidate, baseline):
    return None if baseline == 0 else 100.0 * (candidate / baseline - 1.0)


def _paired_case_counts(baseline, candidate):
    if set(baseline["by_case"]) != set(candidate["by_case"]):
        raise ValueError("baseline and candidate case sets differ")
    result = {}
    for metric in METRICS:
        improved = worsened = tied = 0
        for case, before in baseline["by_case"].items():
            after = candidate["by_case"][case]
            if before["suite"] != after["suite"]:
                raise ValueError(f"baseline and candidate suite differ for case {case}")
            improved += after[metric] < before[metric]
            worsened += after[metric] > before[metric]
            tied += after[metric] == before[metric]
        result[metric] = {"cases": len(baseline["by_case"]), "improved": improved,
                          "worsened": worsened, "tied": tied}
    return result


def _comparison(baseline_name, candidate_name, arms):
    baseline, candidate = arms[baseline_name], arms[candidate_name]
    return {
        "baseline_arm": baseline_name,
        "candidate_arm": candidate_name,
        "candidate_relative_change_percent": {
            metric: _relative_change(candidate["overall"][metric], baseline["overall"][metric])
            for metric in METRICS
        },
        "suite_relative_change_percent": {
            suite: {
                metric: _relative_change(candidate["by_suite"][suite][metric],
                                         baseline["by_suite"][suite][metric])
                for metric in METRICS
            }
            for suite in SUITES
        },
        "paired_case_win_loss": _paired_case_counts(baseline, candidate),
    }

