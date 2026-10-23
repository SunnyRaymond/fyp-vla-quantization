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


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _protocol_check(p):
    if p.get("protocol") != PROTOCOL:
        raise ValueError(f"unexpected protocol: {p.get('protocol')!r}")
    evaluation = p.get("evaluation", {})
    if evaluation.get("case_ids") != list(CASE_IDS) or evaluation.get("seed_indices") != list(SEED_INDICES):
        raise ValueError("protocol must use the frozen 3000..3039 x seeds [0,1] cohort")
    calibration = p.get("calibration", {})
    if calibration.get("case_ids") != list(C8_CASE_IDS) or calibration.get("seed_indices") != [0]:
        raise ValueError("protocol must use the frozen C8 calibration cohort and seed 0")
    if tuple(p.get("suites", ())) != SUITES or p.get("target_linears") != 614:
        raise ValueError("protocol suite order or 614 target Linears differ from the frozen setup")
    weight = _finite_nonnegative(p.get("training", {}).get("latent_weight"), "training.latent_weight")
    if not math.isclose(weight, LATENT_WEIGHT, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("protocol latent_weight differs from the frozen value")


def _aggregate_arms(records_by_arm, names, **kwargs):
    if set(records_by_arm) != set(names):
        raise ValueError(f"expected arms {list(names)}; found {sorted(records_by_arm)}")
    return {name: aggregate_records(records_by_arm[name], LATENT_WEIGHT, **kwargs) for name in names}


def _markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _provenance_rows(provenance):
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a dictionary")
    preferred = [key for key in ("reuse", "reused_arms", "reuse_sources", "training_cost")
                 if key in provenance]
    keys = preferred + [key for key in provenance if key not in preferred]
    return [
        f"| {_markdown_cell(key)} | {_markdown_cell(json.dumps(provenance[key], ensure_ascii=False, sort_keys=True))} |"
        for key in keys
    ]


def _build_markdown(report):
    lines = [
        "# Fast-WAM R128 Joint8 固定输入聚合结果",
        "",
        f"Protocol：`{report['protocol']}`；target Linears：614；heldout 每臂40 cases × 2 seeds = 80 queries。",
        "heldout 按 seed → case → suite 聚合，四个 suite 等权；相对变化为 candidate / baseline - 1，负值表示误差下降。",
        "",
        "## Heldout comparisons",
        "",
    ]
    for key, comparison in report["comparisons"].items():
        baseline_name, candidate_name = comparison["baseline_arm"], comparison["candidate_arm"]
        baseline = report["heldout"][baseline_name]["overall"]
        candidate = report["heldout"][candidate_name]["overall"]
        lines += [
            f"### {candidate_name} vs {baseline_name}",
            "",
            "| Metric | Baseline | Candidate | Relative change |",
            "|---|---:|---:|---:|",
        ]
        for metric in METRICS:
            change = comparison["candidate_relative_change_percent"][metric]
            delta = "n/a" if change is None else f"{change:+.3f}%"
            lines.append(
                f"| {LABELS[metric]} | {baseline[metric]:.9f} | {candidate[metric]:.9f} | {delta} |"
            )
        wins = comparison["paired_case_win_loss"]["first10motor_rmse"]
        lines += [
            "",
            f"First10 motor RMSE 的 case 配对计数：improved {wins['improved']}，worsened {wins['worsened']}，tied {wins['tied']}（共 {wins['cases']} cases）。",
            "",
        ]

    lines += [
        "## C8 calibration 阶段表",
        "",
        "C8 含8 cases（四 suite 各2）与 seed 0；每格为先按 case 聚合后四 suite 等权的 mean。",
        "",
        "| Arm | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in CALIBRATION_ARMS:
        overall = report["calibration"][arm]["overall"]
        lines.append("| " + arm + " | " + " | ".join(f"{overall[m]:.9f}" for m in METRICS) + " |")
    lines += [
        "",
        "## 复用与训练成本来源",
        "",
        "H_VQ8 与 R8_VQ8 的旧 VQ endpoint records 可复用；复用来源与训练成本逐项照录 worker 传入的 provenance。",
        "",
        "| Provenance 项 | worker 传入值 |",
        "|---|---|",
        *_provenance_rows(report["provenance"]),
        "",
        "Joint8 与 H_VQ8 从同一 H_VQ0 初始化，均按相同 VQ learning rate、顺序、loss 完成8次更新；Joint8 额外包含 R 参数及计算。Joint8 直接对 compressed student 做 coadaptation；训练中间的 R 不保证每一步都等价于原 FP 函数。",
        "",
        "该结果只描述固定输入的 decoded-BF16/A4 配对指标。paired case 计数不是显著性检验；不构成 closed-loop、native-kernel、延迟或 equal-compute/walltime 结论。",
        "",
    ]
    return "\n".join(lines)


def write_report(output_dir, p, heldout, calibration, provenance):
    """Write comparison.json and concise Chinese results for the frozen cohorts."""
    _protocol_check(p)
    heldout_agg = _aggregate_arms(heldout, HELDOUT_ARMS)
    calibration_agg = _aggregate_arms(
        calibration, CALIBRATION_ARMS, case_ids=C8_CASE_IDS, seed_indices=(0,),
        suite_by_case=C8_SUITE_BY_CASE, cases_per_suite=2,
    )
    comparisons = {
        "noVQ_R8_vs_H": _comparison("H_W16A4", "R8_W16A4", heldout_agg),
        "Joint8_vs_H_VQ8": _comparison("H_VQ8", "Joint8", heldout_agg),
        "Joint8_vs_R8_VQ8": _comparison("R8_VQ8", "Joint8", heldout_agg),
    }
    report = {
        "protocol": PROTOCOL,
        "complete": True,
        "parameters": p,
        "heldout_queries_per_arm": 80,
        "heldout_cases": 40,
        "heldout_seed_indices": list(SEED_INDICES),
        "calibration_queries_per_arm": 8,
        "calibration_case_ids": list(C8_CASE_IDS),
        "calibration_seed_indices": [0],
        "heldout": heldout_agg,
        "calibration": calibration_agg,
        "comparisons": comparisons,
        "provenance": provenance,
        "claim_scope": "Descriptive paired fixed-input decoded-BF16/A4 metrics on the frozen heldout and C8 cohorts; no significance, closed-loop, native-kernel, latency, or equal-compute/walltime claim.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "comparison.json", report)
    (out / "RESULTS.zh.md").write_text(_build_markdown(report), encoding="utf-8")
    return report
