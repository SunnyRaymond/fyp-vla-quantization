"""Aggregate the frozen 40-context x 2-seed fixed-input comparison."""
from __future__ import annotations

import json
import math
from pathlib import Path


PROTOCOL = "fastwam-vq-learned-r128-c8-v1"
CASE_IDS = tuple(range(3000, 3040))
SEED_INDICES = (0, 1)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
METRICS = (
    "first10motor_rmse",
    "future_latent_nmse",
    "mixed_loss",
    "full32motor_rmse",
    "late22motor_rmse",
    "first10gripper_rmse",
    "first10motor_mse",
)
def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _normalized_record(record, latent_weight):
    row = dict(record)
    motor_rmse = _finite_nonnegative(row.get("first10motor_rmse"), "first10motor_rmse")
    motor_mse = row.get("first10motor_mse", motor_rmse * motor_rmse)
    row["first10motor_mse"] = _finite_nonnegative(motor_mse, "first10motor_mse")
    if "mixed_loss" not in row:
        if latent_weight is None:
            raise ValueError("latent_weight is required when mixed_loss is absent")
        weight = _finite_nonnegative(latent_weight, "latent_weight")
        future = _finite_nonnegative(row.get("future_latent_nmse"), "future_latent_nmse")
        row["mixed_loss"] = row["first10motor_mse"] + weight * future
    for metric in METRICS:
        row[metric] = _finite_nonnegative(row.get(metric), metric)
    return row


def aggregate_records(records, latent_weight=None):
    """Return case-seed, case, suite, and equal-suite-weighted metric means.

    ``latent_weight`` is optional when records already include ``mixed_loss``.
    It lets the same function accept the source baseline's native five-metric schema.
    """
    normalized = [_normalized_record(record, latent_weight) for record in records]
    expected = {(case, seed) for case in CASE_IDS for seed in SEED_INDICES}
    found = {}
    case_suites = {}
    for row in normalized:
        case, seed = row.get("case_id"), row.get("seed_index")
        if isinstance(case, bool) or not isinstance(case, int) or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("case_id and seed_index must be integers")
        key = (case, seed)
        if key in found:
            raise ValueError(f"duplicate query for case {case}, seed {seed}")
        if key not in expected:
            raise ValueError(f"unexpected query identity: {key}")
        suite = row.get("suite")
        if suite not in SUITES:
            raise ValueError(f"unexpected suite for case {case}: {suite!r}")
        if case in case_suites and case_suites[case] != suite:
            raise ValueError(f"case {case} has inconsistent suite bindings")
        case_suites[case] = suite
        found[key] = row
    if found.keys() != expected:
        missing = sorted(expected - found.keys())
        raise ValueError(f"expected exactly 80 case-seed queries; missing {missing[:8]}")

    by_case = {}
    suite_cases = {suite: [] for suite in SUITES}
    for case in CASE_IDS:
        rows = [found[(case, seed)] for seed in SEED_INDICES]
        suite = case_suites[case]
        means = {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
        by_case[str(case)] = {"suite": suite, **means}
        suite_cases[suite].append(means)
    if any(len(suite_cases[suite]) != 10 for suite in SUITES):
        raise ValueError("each of the four suites must contain exactly 10 cases")

    by_suite = {
        suite: {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
        for suite, rows in suite_cases.items()
    }
    overall = {metric: sum(by_suite[suite][metric] for suite in SUITES) / len(SUITES)
               for metric in METRICS}
    return {
        "queries": len(normalized),
        "cases": len(by_case),
        "seed_indices": list(SEED_INDICES),
        "aggregation": "mean seeds within context, mean 10 contexts within suite, mean four suites equally",
        "overall": overall,
        "by_suite": by_suite,
        "by_case": by_case,
    }


def _relative_change(candidate, baseline):
    if baseline == 0:
        return None
    return 100.0 * (candidate / baseline - 1.0)


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
            if after[metric] < before[metric]:
                improved += 1
            elif after[metric] > before[metric]:
                worsened += 1
            else:
                tied += 1
        result[metric] = {"cases": 40, "improved": improved, "worsened": worsened, "tied": tied}
    return result


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_report(output_dir, p, baseline_records, candidate_records, provenance):
    """Write comparison.json and RESULTS.zh.md for the paired fixed-input run."""
    if p.get("protocol") != PROTOCOL:
        raise ValueError(f"unexpected protocol: {p.get('protocol')!r}")
    evaluation = p.get("evaluation", {})
    if evaluation.get("case_ids") != list(CASE_IDS) or evaluation.get("seed_indices") != list(SEED_INDICES):
        raise ValueError("protocol must use the frozen 3000..3039 x seeds [0,1] cohort")
    if tuple(p.get("suites", ())) != SUITES:
        raise ValueError("protocol suite order or names differ from the frozen four-suite cohort")
    training = p.get("training", {})
    latent_weight = training.get("latent_weight")
    if latent_weight is None:
        latent_weight = training.get("rho", 1.5) * training.get("normalization_c", 0.028303017887797623)
    latent_weight = _finite_nonnegative(latent_weight, "training.latent_weight")
    baseline = aggregate_records(baseline_records, latent_weight)
    candidate = aggregate_records(candidate_records, latent_weight)
    changes = {
        metric: _relative_change(candidate["overall"][metric], baseline["overall"][metric])
        for metric in METRICS
    }
    suite_changes = {
        suite: {
            metric: _relative_change(candidate["by_suite"][suite][metric], baseline["by_suite"][suite][metric])
            for metric in METRICS
        }
        for suite in SUITES
    }
    report = {
        "protocol": p.get("protocol", PROTOCOL),
        "complete": True,
        "parameters": p,
        "paired_queries_per_arm": 80,
        "contexts": 40,
        "seeds_per_context": 2,
        "arms": {"baseline_H_plus_VQ8": baseline, "candidate_R8_plus_VQ8": candidate},
        "candidate_relative_change_percent": changes,
        "suite_relative_change_percent": suite_changes,
        "paired_case_win_loss": _paired_case_counts(baseline, candidate),
        "provenance": provenance,
        "claim_scope": "Descriptive paired fixed-input decoded-BF16/A4 results on the frozen original-heldout cohort; no statistical-significance, closed-loop, or native-kernel claim.",
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "comparison.json", report)

    labels = {
        "first10motor_rmse": "First10 motor RMSE",
        "future_latent_nmse": "Future latent NMSE",
        "mixed_loss": "Total mixed loss",
        "full32motor_rmse": "Full32 motor RMSE",
        "late22motor_rmse": "Late22 motor RMSE",
        "first10gripper_rmse": "First10 gripper RMSE",
        "first10motor_mse": "First10 motor MSE",
    }
    lines = [
        "# H+VQ8 与 R8+VQ8 固定输入配对结果",
        "",
        f"Protocol：`{report['protocol']}`；每臂40 contexts × 2 seeds = 80 queries。",
        "先对每个context的两seed取均值，再对每个suite的10个context取均值，最后四suite等权。",
        "相对变化为 candidate / baseline - 1；负值表示误差下降。",
        "",
        "## Overall",
        "",
        "| Metric | H+VQ8 baseline | R8+VQ8 candidate | Candidate change |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        delta = changes[metric]
        delta_text = "n/a" if delta is None else f"{delta:+.3f}%"
        lines.append(f"| {labels[metric]} | {baseline['overall'][metric]:.9f} | {candidate['overall'][metric]:.9f} | {delta_text} |")
    lines += ["", "## By suite", "", "| Suite | Arm | First10 motor RMSE | Future latent NMSE | Total mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for suite in SUITES:
        for arm_name, data in (("H+VQ8", baseline), ("R8+VQ8", candidate)):
            row = data["by_suite"][suite]
            lines.append(
                f"| {suite} | {arm_name} | {row['first10motor_rmse']:.9f} | {row['future_latent_nmse']:.9f} | {row['mixed_loss']:.9f} | {row['full32motor_rmse']:.9f} | {row['late22motor_rmse']:.9f} | {row['first10gripper_rmse']:.9f} |"
            )
    lines += ["", "## Paired case win/loss", "", "每个context先对两个seed取均值；误差下降记为 candidate improved。", "", "| Metric | Improved | Worsened | Tied | Cases |", "|---|---:|---:|---:|---:|"]
    paired = report["paired_case_win_loss"]
    for metric in METRICS:
        row = paired[metric]
        lines.append(f"| {labels[metric]} | {row['improved']} | {row['worsened']} | {row['tied']} | {row['cases']} |")
    lines += [
        "",
        "该结果是固定输入、decoded-BF16/A4 的描述性配对比较。paired case counts 不是显著性检验；不构成 closed-loop 或 native-kernel 结论。",
        "",
    ]
    (out / "RESULTS.zh.md").write_text("\n".join(lines), encoding="utf-8")
    return report
