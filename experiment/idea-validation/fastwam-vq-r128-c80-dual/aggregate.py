"""Aggregate the frozen C80 dual-arm fixed-input comparison."""
from __future__ import annotations

import json
import math


PROTOCOL = "fastwam-vq-r128-c80-dual-v1"
CALIBRATION_CASE_IDS = tuple(range(2000, 2080))
HELDOUT_CASE_IDS = tuple(range(3000, 3040))
SEED_INDICES = (0, 1)
CALIBRATION_SEEDS = (0,)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CALIBRATION_ARMS = ("R0_VQ0", "Seq8", "Joint8")
HELDOUT_ARMS = ("R0_VQ0", "Seq8", "Joint8", "H_VQ8", "R0_W16A4")
NEW_ARMS = ("R0_VQ0", "Seq8", "Joint8")
REFERENCE_ARMS = ("H_VQ8", "R0_W16A4")
REFERENCE_BINDING_IDS = {
    "H_VQ8": {"protocol": "fastwam-vq-r128-joint-c8-v1", "arm": "H_VQ8", "split": "heldout"},
    "R0_W16A4": {
        "protocol": "fastwam-r128-improvements-v1", "arm": "C80_once_lr1e4",
        "lane": "coverage_c80", "split": "heldout",
    },
}
LATENT_WEIGHT = 0.04245452683169643
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


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _check_protocol(protocol):
    protocol_id = protocol.get("protocol") if isinstance(protocol, dict) else None
    if protocol_id != PROTOCOL:
        raise ValueError(f"unexpected protocol: {protocol_id!r}")
    if tuple(protocol.get("suites", ())) != SUITES or protocol.get("target_linears") != 614:
        raise ValueError("protocol suites or target Linears differ from the frozen setup")

    calibration = protocol.get("calibration", {})
    if (calibration.get("case_ids") != list(CALIBRATION_CASE_IDS)
            or calibration.get("c80_case_ids") != list(CALIBRATION_CASE_IDS)
            or calibration.get("seed_indices") != [0]):
        raise ValueError("protocol must use C80 cases 2000..2079 with seed_index 0")

    heldout = protocol.get("heldout", protocol.get("evaluation", {}))
    evaluation = protocol.get("evaluation")
    if (heldout.get("case_ids") != list(HELDOUT_CASE_IDS)
            or heldout.get("seed_indices") != list(SEED_INDICES)
            or heldout.get("queries") != 80):
        raise ValueError("protocol must use heldout cases 3000..3039, seeds [0,1], and 80 queries")
    if evaluation is not None and (
            evaluation.get("case_ids") != list(HELDOUT_CASE_IDS)
            or evaluation.get("seed_indices") != list(SEED_INDICES)):
        raise ValueError("heldout and evaluation cohorts differ")

    training = protocol.get("training", {})
    expected_training = {
        "latent_weight": LATENT_WEIGHT,
        "vq_updates": 8,
        "rotation_updates": 8,
        "contexts_per_update": 10,
        "vq_learning_rate": 5e-4,
        "rotation_learning_rate": 1e-5,
    }
    for key, expected in expected_training.items():
        value = training.get(key)
        if isinstance(expected, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(
                    float(value), expected, rel_tol=0.0, abs_tol=1e-15):
                raise ValueError(f"protocol training.{key} differs from the frozen value")
        elif value != expected:
            raise ValueError(f"protocol training.{key} must be {expected}")
    if protocol.get("endpoints") != list(NEW_ARMS):
        raise ValueError("protocol endpoints must be R0_VQ0, Seq8, and Joint8")
    reference_bindings = protocol.get("reference_arms", {})
    if set(reference_bindings) != set(REFERENCE_ARMS):
        raise ValueError("protocol references must be H_VQ8 and R0_W16A4")
    for arm, expected in REFERENCE_BINDING_IDS.items():
        if any(reference_bindings[arm].get(key) != value for key, value in expected.items()):
            raise ValueError(f"protocol source binding differs for {arm}")
    return heldout, training


def _check_source_binding(row, arm, protocol):
    declared = protocol["reference_arms"][arm]
    source = row.get("source_binding")
    if not isinstance(source, dict):
        raise ValueError(f"reused arm {arm} requires source_binding")
    for key in ("protocol", "arm", "split"):
        if source.get(key) != declared.get(key):
            raise ValueError(f"source_binding {key} differs for {arm}")
    for key in ("root", "lane", "job"):
        if key in source and key in declared and source[key] != declared[key]:
            raise ValueError(f"source_binding {key} differs for {arm}")
    if "pbs_jobid" in source and source["pbs_jobid"] != declared.get("job"):
        raise ValueError(f"source_binding pbs_jobid differs for {arm}")
    if source.get("split") != "heldout":
        raise ValueError(f"reused arm {arm} must bind to heldout source records")
    for key in ("case_id", "seed_index", "suite"):
        if key in source and source[key] != row.get(key):
            raise ValueError(f"source_binding {key} differs for {arm}")


def _aggregate(records, arm, split, protocol, case_ids, seeds, contexts_per_suite):
    protocol_id = protocol["protocol"]
    expected = {(case, seed) for case in case_ids for seed in seeds}
    found = {}
    case_suites = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{arm}/{split} records must be dictionaries")
        case, seed = record.get("case_id"), record.get("seed_index")
        if (isinstance(case, bool) or not isinstance(case, int)
                or isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("case_id and seed_index must be integers")
        key = (case, seed)
        if key in found:
            raise ValueError(f"duplicate query for {arm}/{split}: {key}")
        if key not in expected:
            raise ValueError(f"unexpected query identity for {arm}/{split}: {key}")
        if record.get("arm") != arm or record.get("split") != split:
            raise ValueError(f"arm/split binding differs for {arm}/{split} {key}")
        record_protocol = record.get("protocol")
        if record_protocol != protocol_id:
            if arm not in REFERENCE_ARMS:
                raise ValueError(f"protocol binding differs for {arm}/{split} {key}")
            if not record.get("source_binding"):
                raise ValueError(f"legacy protocol binding for {arm}/{split} {key} requires source_binding")
            if record_protocol != protocol["reference_arms"][arm].get("protocol"):
                raise ValueError(f"legacy protocol binding differs for {arm}/{split} {key}")
        if arm in REFERENCE_ARMS:
            _check_source_binding(record, arm, protocol)
        suite = record.get("suite")
        if suite not in SUITES:
            raise ValueError(f"unexpected suite for case {case}: {suite!r}")
        if case in case_suites and case_suites[case] != suite:
            raise ValueError(f"case {case} has inconsistent suite bindings")
        case_suites[case] = suite

        values = {metric: _number(record.get(metric), metric) for metric in METRICS}
        formula = values["first10motor_mse"] + LATENT_WEIGHT * values["future_latent_nmse"]
        if not math.isclose(values["mixed_loss"], formula, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(f"mixed_loss does not match the frozen loss for {arm}/{split} {key}")
        found[key] = {"suite": suite, **values}

    if found.keys() != expected:
        missing = sorted(expected - found.keys())
        raise ValueError(f"{arm}/{split} needs exactly {len(expected)} records; missing {missing[:8]}")
    suite_counts = {suite: 0 for suite in SUITES}
    for suite in case_suites.values():
        suite_counts[suite] += 1
    if any(count != contexts_per_suite for count in suite_counts.values()):
        raise ValueError(f"{arm}/{split} must have {contexts_per_suite} contexts per suite")

    by_case = {}
    suite_cases = {suite: [] for suite in SUITES}
    for case in case_ids:
        rows = [found[(case, seed)] for seed in seeds]
        means = {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
        suite = case_suites[case]
        by_case[str(case)] = {"suite": suite, **means}
        suite_cases[suite].append(means)
    by_suite = {
        suite: {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in METRICS
        }
        for suite, rows in suite_cases.items()
    }
    overall = {
        metric: sum(by_suite[suite][metric] for suite in SUITES) / len(SUITES)
        for metric in METRICS
    }
    return {
        "queries": len(records),
        "contexts": len(by_case),
        "seed_indices": list(seeds),
        "aggregation": "mean seeds within context, mean contexts within suite, mean four suites equally",
        "overall": overall,
        "by_suite": by_suite,
        "by_case": by_case,
    }


def _relative_change(candidate, baseline):
    return None if baseline == 0 else 100.0 * (candidate / baseline - 1.0)


def _paired_counts(baseline, candidate):
    if set(baseline["by_case"]) != set(candidate["by_case"]):
        raise ValueError("paired arms have different context sets")
    result = {}
    for metric in METRICS:
        improved = worsened = tied = 0
        for case, before in baseline["by_case"].items():
            after = candidate["by_case"][case]
            if before["suite"] != after["suite"]:
                raise ValueError(f"paired arms have different suite bindings for case {case}")
            improved += after[metric] < before[metric]
            worsened += after[metric] > before[metric]
            tied += after[metric] == before[metric]
        result[metric] = {
            "contexts": len(baseline["by_case"]),
            "improved": improved,
            "worsened": worsened,
            "tied": tied,
        }
    return result


def _compare(baseline_name, candidate_name, arms):
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
        "paired_context_win_loss": _paired_counts(baseline, candidate),
    }


def build_comparison(protocol, calibration, heldout, provenance):
    """Validate exact cohorts and return compact, descriptive aggregates."""
    _, training = _check_protocol(protocol)
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a dictionary")
    if set(calibration) != set(CALIBRATION_ARMS):
        raise ValueError(f"calibration arms must be {list(CALIBRATION_ARMS)}")
    if set(heldout) != set(HELDOUT_ARMS):
        raise ValueError(f"heldout arms must be {list(HELDOUT_ARMS)}")

    calibration_full = {
        arm: _aggregate(calibration[arm], arm, "calibration", protocol,
                        CALIBRATION_CASE_IDS, CALIBRATION_SEEDS, 20)
        for arm in CALIBRATION_ARMS
    }
    heldout_full = {
        arm: _aggregate(heldout[arm], arm, "heldout", protocol,
                        HELDOUT_CASE_IDS, SEED_INDICES, 10)
        for arm in HELDOUT_ARMS
    }
    # Enforce the same case-to-suite map for paired contexts across all arms.
    reference_case_suites = {
        case: heldout_full["R0_VQ0"]["by_case"][str(case)]["suite"]
        for case in HELDOUT_CASE_IDS
    }
    for arm, result in heldout_full.items():
        if any(result["by_case"][str(case)]["suite"] != suite
               for case, suite in reference_case_suites.items()):
            raise ValueError(f"{arm} suite assignments differ from R0_VQ0")
    calibration_case_suites = {
        case: calibration_full["R0_VQ0"]["by_case"][str(case)]["suite"]
        for case in CALIBRATION_CASE_IDS
    }
    for arm, result in calibration_full.items():
        if any(result["by_case"][str(case)]["suite"] != suite
               for case, suite in calibration_case_suites.items()):
            raise ValueError(f"{arm} suite assignments differ from R0_VQ0 calibration")

    comparisons = {
        "Joint8_vs_Seq8": _compare("Seq8", "Joint8", heldout_full),
        "Seq8_vs_R0_VQ0": _compare("R0_VQ0", "Seq8", heldout_full),
        "Joint8_vs_R0_VQ0": _compare("R0_VQ0", "Joint8", heldout_full),
    }
    for arm in ("Seq8", "Joint8"):
        for reference in REFERENCE_ARMS:
            comparisons[f"{arm}_vs_{reference}"] = _compare(reference, arm, heldout_full)

    def compact(arms):
        return {
            arm: {key: value for key, value in result.items() if key != "by_case"}
            for arm, result in arms.items()
        }

    source_bindings = {
        arm: {key: value for key, value in protocol["reference_arms"][arm].items()
              if key in ("root", "protocol", "arm", "lane", "split", "job")}
        for arm in REFERENCE_ARMS
    }
    return {
        "protocol": PROTOCOL,
        "complete": True,
        "target_linears": 614,
        "cohorts": {
            "calibration": {"case_ids": [2000, 2079], "contexts": 80,
                            "seed_indices": [0], "queries_per_arm": 80},
            "heldout": {"case_ids": [3000, 3039], "contexts": 40,
                        "seed_indices": [0, 1], "queries_per_arm": 80},
            "suites": list(SUITES),
            "aggregation": "mean seeds within context, mean contexts within suite, mean four suites equally",
        },
        "training": {
            "vq_updates_each": training["vq_updates"],
            "rotation_updates_joint": training["rotation_updates"],
            "rotation_frozen_seq8": True,
            "contexts_per_update": training["contexts_per_update"],
            "vq_learning_rate": training["vq_learning_rate"],
            "rotation_learning_rate_joint": training["rotation_learning_rate"],
            "latent_weight": training["latent_weight"],
            "checkpoint_selection": "fixed final update 8; no heldout-based selection",
        },
        "calibration": compact(calibration_full),
        "heldout": compact(heldout_full),
        "comparisons": comparisons,
        "reference_source_bindings": source_bindings,
        "provenance": provenance,
        "claim_scope": (
            "Descriptive paired fixed-input decoded-BF16/A4 metrics. Paired context counts are not significance tests; "
            "no closed-loop, native-kernel, equal-walltime, or equal-compute claim. Joint8 includes 8 additional R updates."
        ),
    }


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric_table(rows):
    lines = [
        "| Arm / suite | " + " | ".join(LABELS[metric] for metric in METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for name, values in rows:
        lines.append("| " + _cell(name) + " | " + " | ".join(f"{values[m]:.9f}" for m in METRICS) + " |")
    return lines


def render_markdown(comparison):
    """Render the validated aggregates and evidence limits in Chinese."""
    lines = [
        "# Fast-WAM R128 C80 dual-arm 固定输入结果",
        "",
        f"Protocol：`{comparison['protocol']}`；目标 Linears：{comparison['target_linears']}。",
        "校准集为 cases 2000–2079、seed 0；heldout 为 cases 3000–3039、seeds 0/1。每个 heldout arm 含 40 contexts × 2 seeds = 80 queries。",
        "聚合顺序为 seed → context → suite，四个 suite 等权；表格报告七项固定输入指标。",
        "",
        "Seq8 与 Joint8 均从共用 R0/VQ0 起点训练 8 次 VQ 更新、每次 10 contexts。Seq8 冻结 R；Joint8 另以 LR=1e-5 做 8 次 R 更新，因此增加 R 参数更新与计算。VQ LR=5e-4。",
        "所有新臂使用固定 final update 8；不按 heldout 结果挑选 checkpoint 或超参数。",
        "",
        "## C80 calibration：按 suite",
        "",
    ]
    rows = []
    for arm in CALIBRATION_ARMS:
        for suite in SUITES:
            rows.append((f"{arm} / {suite}", comparison["calibration"][arm]["by_suite"][suite]))
    lines += _metric_table(rows)
    lines += ["", "## Heldout：按 suite", ""]
    rows = []
    for arm in HELDOUT_ARMS:
        for suite in SUITES:
            rows.append((f"{arm} / {suite}", comparison["heldout"][arm]["by_suite"][suite]))
    lines += _metric_table(rows)
    lines += ["", "## Heldout overall：四 suite 等权", ""]
    rows = [(arm, comparison["heldout"][arm]["overall"]) for arm in HELDOUT_ARMS]
    lines += _metric_table(rows)

    lines += ["", "## 配对比较", "", "每项胜负按同一 40 个 context 配对；先对 seeds 0/1 求均值。误差越低记为 improved。"]
    for name, result in comparison["comparisons"].items():
        baseline_name, candidate_name = result["baseline_arm"], result["candidate_arm"]
        baseline = comparison["heldout"][baseline_name]["overall"]
        candidate = comparison["heldout"][candidate_name]["overall"]
        lines += ["", f"### {candidate_name} vs {baseline_name}", "",
                  "| Metric | Baseline | Candidate | Relative change | Context improved / worsened / tied |",
                  "|---|---:|---:|---:|---:|"]
        for metric in METRICS:
            change = result["candidate_relative_change_percent"][metric]
            change_text = "n/a" if change is None else f"{change:+.3f}%"
            counts = result["paired_context_win_loss"][metric]
            lines.append(
                f"| {LABELS[metric]} | {baseline[metric]:.9f} | {candidate[metric]:.9f} | {change_text} | "
                f"{counts['improved']} / {counts['worsened']} / {counts['tied']} of {counts['contexts']} |"
            )
        lines += ["", "按 suite 的相对变化（candidate / baseline − 1）：", ""]
        lines += ["| Suite | " + " | ".join(LABELS[m] for m in METRICS) + " |",
                  "|---|" + "---:|" * len(METRICS)]
        for suite in SUITES:
            changes = result["suite_relative_change_percent"][suite]
            lines.append("| " + suite + " | " + " | ".join(
                "n/a" if changes[m] is None else f"{changes[m]:+.3f}%" for m in METRICS
            ) + " |")

    lines += ["", "## 复用来源", "", "| Arm | Source binding |", "|---|---|"]
    for arm, binding in comparison["reference_source_bindings"].items():
        lines.append(f"| {arm} | `{_cell(json.dumps(binding, ensure_ascii=False, sort_keys=True))}` |")
    if comparison["provenance"]:
        lines += ["", "其他 provenance：", "", "```json",
                  json.dumps(comparison["provenance"], ensure_ascii=False, indent=2, sort_keys=True), "```"]
    lines += [
        "",
        "结果是描述性的 paired fixed-input decoded-BF16/A4 evidence；context 胜负计数不是显著性检验。它不构成 closed-loop、native-kernel、equal-compute 或 equal-walltime 结论。",
        "",
    ]
    return "\n".join(lines)
