"""Validate and summarize the frozen Fast-WAM R128 C80 transport cohorts."""
from __future__ import annotations

import json
import math


PROTOCOL = "fastwam-vq-r128-c80-transport-v1"
CALIBRATION_CASES = tuple(range(2000, 2080))
HELDOUT_CASES = tuple(range(3000, 3040))
SEEDS = (0, 1)
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
    "first10motor_rmse": "前 10 步 motor RMSE",
    "future_latent_nmse": "Future latent NMSE",
    "mixed_loss": "Mixed loss",
    "full32motor_rmse": "全 32 步 motor RMSE",
    "late22motor_rmse": "后 22 步 motor RMSE",
    "first10gripper_rmse": "前 10 步 gripper RMSE",
    "first10motor_mse": "前 10 步 motor MSE",
}
ANCHORS = ("Seq", "Joint")
LATENT_WEIGHT = 0.04245452683169643
AGGREGATION = "seed 均值 → context 均值 → 四个 suite 等权均值"
COMPARISON_PAIRS = (
    ("Comp8_vs_Uncomp8", "Uncomp8", "Comp8"),
    ("Comp8_vs_Seq_A4", "Seq_A4", "Comp8"),
    ("Uncomp8_vs_Joint_A4", "Joint_A4", "Uncomp8"),
    ("Seq_to_Joint_A4_vs_Seq_A4", "Seq_A4", "Seq_to_Joint_A4"),
    ("Joint_to_Seq_A4_vs_Joint_A4", "Joint_A4", "Joint_to_Seq_A4"),
    ("Seq_to_Joint_noA4_vs_Seq_noA4", "Seq_noA4", "Seq_to_Joint_noA4"),
    ("Joint_to_Seq_noA4_vs_Joint_noA4", "Joint_noA4", "Joint_to_Seq_noA4"),
)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _check_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("protocol") != PROTOCOL:
        raise ValueError(f"unexpected protocol: {getattr(protocol, 'get', lambda *_: None)('protocol')!r}")
    cohorts = protocol.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ValueError("protocol cohorts are missing")
    suites = cohorts.get("suites")
    expected_suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    if not isinstance(suites, list) or tuple(suites) != expected_suites:
        raise ValueError("protocol suites differ from the frozen four-suite setup")
    if protocol.get("probe_arms") != [
            "Seq_A4", "Seq_to_Joint_A4", "Joint_A4", "Joint_to_Seq_A4",
            "Seq_noA4", "Seq_to_Joint_noA4", "Joint_noA4", "Joint_to_Seq_noA4"]:
        raise ValueError("protocol probe_arms differ from the frozen setup")
    if protocol.get("training_arms") != ["Comp8", "Uncomp8"]:
        raise ValueError("protocol training_arms must be Comp8 and Uncomp8")
    calibration = cohorts.get("calibration")
    heldout = cohorts.get("heldout")
    if (not isinstance(calibration, dict)
            or calibration.get("case_ids") != list(CALIBRATION_CASES)
            or calibration.get("seed_indices") != [0]
            or calibration.get("contexts") != 80 or calibration.get("queries") != 80):
        raise ValueError("protocol calibration cohort must be cases 2000..2079 with seed 0")
    if (not isinstance(heldout, dict)
            or heldout.get("case_ids") != list(HELDOUT_CASES)
            or heldout.get("seed_indices") != [0, 1]
            or heldout.get("contexts") != 40 or heldout.get("queries") != 80):
        raise ValueError("protocol heldout cohort must be cases 3000..3039 with seeds [0,1]")
    decision = protocol.get("decision")
    gates = protocol.get("gates")
    training = protocol.get("training")
    if not isinstance(decision, dict) or not isinstance(gates, dict) or not isinstance(training, dict):
        raise ValueError("protocol decision, gates, or training settings are missing")
    if decision.get("major_support_fraction") != 0.5 or decision.get("required_suites_improved") != 3:
        raise ValueError("protocol decision thresholds differ from the frozen values")
    latent_weight = _number(training.get("latent_weight"), "training.latent_weight")
    if not math.isclose(latent_weight, LATENT_WEIGHT, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("protocol training.latent_weight differs from the frozen value")
    atol = _number(gates.get("replay_metric_atol"), "gates.replay_metric_atol")
    residual_fraction = _number(
        gates.get("bf16_noA4_residual_gap_fraction"),
        "gates.bf16_noA4_residual_gap_fraction",
    )
    if not math.isclose(residual_fraction, 0.1, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("protocol noA4 residual fraction must be 0.1")
    return tuple(suites), decision, atol, residual_fraction, latent_weight


def _expected(split):
    if split == "calibration":
        return CALIBRATION_CASES, (0,), 20
    if split == "heldout":
        return HELDOUT_CASES, SEEDS, 10
    raise ValueError(f"unexpected split: {split!r}")


def _validate_source_binding(binding, row):
    if binding is None:
        return
    if not isinstance(binding, dict):
        raise ValueError("source_binding must be a dictionary when present")
    for key in ("case_id", "seed_index", "suite", "split"):
        if key in binding and binding[key] != row.get(key):
            raise ValueError(f"source_binding.{key} differs from its record")


def _validate_records(protocol_id, records, arm, split, cases, seeds, suites, latent_weight):
    expected = {(case, seed) for case in cases for seed in seeds}
    contexts_per_suite = 20 if split == "calibration" else 10
    found = {}
    case_suites = {}
    sampler_seeds = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError(f"{arm}/{split} records must be dictionaries")
        case, seed = row.get("case_id"), row.get("seed_index")
        if (isinstance(case, bool) or not isinstance(case, int)
                or isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("case_id and seed_index must be integers")
        key = (case, seed)
        if key in found:
            raise ValueError(f"duplicate record for {arm}/{split}: {key}")
        if key not in expected:
            raise ValueError(f"unexpected record identity for {arm}/{split}: {key}")
        if row.get("protocol") != protocol_id or row.get("arm") != arm or row.get("split") != split:
            raise ValueError(f"protocol/arm/split binding differs for {arm}/{split} {key}")
        sampler_seed = row.get("sampler_seed")
        if isinstance(sampler_seed, bool) or not isinstance(sampler_seed, int) or sampler_seed < 0:
            raise ValueError(f"sampler_seed must be a non-negative integer for {arm}/{split} {key}")
        sampler_seeds[key] = sampler_seed
        suite = row.get("suite")
        if suite not in suites:
            raise ValueError(f"unexpected suite for {arm}/{split} {key}: {suite!r}")
        if case in case_suites and case_suites[case] != suite:
            raise ValueError(f"case {case} has inconsistent suite bindings across seeds")
        case_suites[case] = suite
        _validate_source_binding(row.get("source_binding"), row)
        values = {metric: _number(row.get(metric), metric) for metric in METRICS}
        formula = values["first10motor_mse"] + latent_weight * values["future_latent_nmse"]
        if not math.isclose(values["mixed_loss"], formula, rel_tol=1e-5, abs_tol=1e-8):
            raise ValueError(f"mixed_loss does not match the frozen formula for {arm}/{split} {key}")
        found[key] = {"suite": suite, **values}

    if found.keys() != expected:
        missing = sorted(expected - found.keys())
        raise ValueError(f"{arm}/{split} requires exactly {len(expected)} records; missing {missing[:8]}")
    suite_counts = {suite: 0 for suite in suites}
    for suite in case_suites.values():
        suite_counts[suite] += 1
    if any(count != contexts_per_suite for count in suite_counts.values()):
        raise ValueError(f"{arm}/{split} requires {contexts_per_suite} contexts per suite")
    return found, case_suites, sampler_seeds


def _aggregate(found, cases, seeds, suites):
    by_case = {}
    suite_cases = {suite: [] for suite in suites}
    for case in cases:
        rows = [found[(case, seed)] for seed in seeds]
        values = {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
        suite = rows[0]["suite"]
        by_case[case] = {"suite": suite, **values}
        suite_cases[suite].append(values)
    by_suite = {
        suite: {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
        for suite, rows in suite_cases.items()
    }
    overall = {
        metric: sum(by_suite[suite][metric] for suite in suites) / len(suites)
        for metric in METRICS
    }
    return {
        "queries": len(found),
        "contexts": len(by_case),
        "seed_indices": list(seeds),
        "overall": overall,
        "by_suite": by_suite,
        "_by_case": by_case,
    }


def _change(baseline, candidate):
    delta = {metric: candidate[metric] - baseline[metric] for metric in METRICS}
    return {
        "baseline": baseline,
        "candidate": candidate,
        "absolute_delta": delta,
        "relative_change_percent": {
            metric: None if baseline[metric] == 0 else 100.0 * delta[metric] / baseline[metric]
            for metric in METRICS
        },
    }


def _paired_counts(baseline, candidate, suites):
    if set(baseline) != set(candidate):
        raise ValueError("paired arms have different context sets")
    counts = {suite: {metric: [0, 0, 0] for metric in METRICS} for suite in suites}
    for case, before in baseline.items():
        after = candidate[case]
        if before["suite"] != after["suite"]:
            raise ValueError(f"paired arms have different suite bindings for case {case}")
        for metric in METRICS:
            idx = 0 if after[metric] < before[metric] else (1 if after[metric] > before[metric] else 2)
            counts[before["suite"]][metric][idx] += 1
    def shape(values):
        return {"improved": values[0], "worsened": values[1], "tied": values[2], "contexts": sum(values)}
    by_suite = {suite: {metric: shape(values) for metric, values in rows.items()} for suite, rows in counts.items()}
    overall = {
        metric: shape([sum(counts[suite][metric][i] for suite in suites) for i in range(3)])
        for metric in METRICS
    }
    return {"overall": overall, "by_suite": by_suite}


def _compare(name, baseline_arm, candidate_arm, aggregates, suites):
    baseline, candidate = aggregates[baseline_arm], aggregates[candidate_arm]
    return {
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "overall": _change(baseline["overall"], candidate["overall"]),
        "by_suite": {
            suite: _change(baseline["by_suite"][suite], candidate["by_suite"][suite])
            for suite in suites
        },
        "paired_context_win_loss": _paired_counts(baseline["_by_case"], candidate["_by_case"], suites),
    }


def _replay_gate(records, atol):
    result = {}
    overall_max = 0.0
    for split in ("calibration", "heldout"):
        base = records["Joint_A4"][split]
        replay = records["Uncomp8"][split]
        by_metric = {}
        for metric in METRICS:
            maximum = max(abs(replay[key][metric] - base[key][metric]) for key in base)
            by_metric[metric] = maximum
            overall_max = max(overall_max, maximum)
        result[split] = {
            "queries": len(base),
            "baseline_arm": "Joint_A4",
            "replay_arm": "Uncomp8",
            "by_metric_max_abs": by_metric,
            "max_abs": max(by_metric.values()),
            "pass": max(by_metric.values()) <= atol,
        }
    return {
        "atol": atol,
        "max_abs": overall_max,
        "pass": overall_max <= atol,
        "by_split": result,
    }


def _validate_pair_records(pair_records, records, suites):
    expected_by_split = {}
    for split in ("calibration", "heldout"):
        cases, seeds, _ = _expected(split)
        expected_by_split[split] = {(case, seed) for case in cases for seed in seeds}
    found = {}
    fields = ("action_max_abs", "future_max_abs", "action_nmse", "future_nmse")
    for row in pair_records:
        if not isinstance(row, dict):
            raise ValueError("pair_records entries must be dictionaries")
        anchor, split = row.get("anchor"), row.get("split")
        if anchor not in ANCHORS or split not in expected_by_split:
            raise ValueError(f"unexpected pair anchor/split: {anchor!r}/{split!r}")
        case, seed = row.get("case_id"), row.get("seed_index")
        if (isinstance(case, bool) or not isinstance(case, int)
                or isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("pair case_id and seed_index must be integers")
        key = (anchor, split, case, seed)
        if key in found:
            raise ValueError(f"duplicate pair residual: {key}")
        if (case, seed) not in expected_by_split[split]:
            raise ValueError(f"unexpected pair residual identity: {key}")
        native_arm = "Seq_noA4" if anchor == "Seq" else "Joint_noA4"
        expected_row = records[native_arm][split][(case, seed)]
        if row.get("suite") != expected_row["suite"]:
            raise ValueError(f"pair suite binding differs for {key}")
        found[key] = {field: _number(row.get(field), field) for field in fields}
    for anchor in ANCHORS:
        for split, identities in expected_by_split.items():
            actual = {(case, seed) for a, s, case, seed in found if a == anchor and s == split}
            if actual != identities:
                missing = sorted(identities - actual)
                raise ValueError(f"{anchor}/{split} pair residual requires {len(identities)} rows; missing {missing[:8]}")
    if len(found) != 2 * sum(len(identities) for identities in expected_by_split.values()):
        raise ValueError("pair_records contains unexpected extra entries")
    return found


def _residual_summary(pair_records):
    fields = ("action_max_abs", "future_max_abs", "action_nmse", "future_nmse")
    summary = {}
    for anchor in ANCHORS:
        summary[anchor] = {}
        for split in ("calibration", "heldout"):
            rows = [values for (a, s, _, _), values in pair_records.items() if a == anchor and s == split]
            summary[anchor][split] = {
                "queries": len(rows),
                "metrics": {
                    field: {"max": max(row[field] for row in rows),
                            "mean": sum(row[field] for row in rows) / len(rows)}
                    for field in fields
                },
            }
    return summary


def _compact_provenance(provenance):
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be a dictionary")
    result = {}
    for key, value in provenance.items():
        name = str(key)
        lowered = name.lower()
        if any(token in lowered for token in ("history", "records", "episodes", "training_log")):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            if not isinstance(value, str) or len(value) <= 512:
                result[name] = value
        elif isinstance(value, dict):
            small = {
                str(k): v for k, v in value.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
            if len(small) <= 20 and all(not isinstance(v, str) or len(v) <= 512 for v in small.values()):
                result[name] = small
    return result


def build_comparison(protocol, records, pair_records, provenance):
    """Validate exact cohorts and return compact aggregates and predeclared gates."""
    suites, decision, replay_atol, residual_fraction, latent_weight = _check_protocol(protocol)
    if not isinstance(records, dict):
        raise ValueError("records must map each arm to calibration and heldout records")
    arms = tuple(protocol["probe_arms"] + protocol["training_arms"])
    if set(records) != set(arms):
        raise ValueError(f"records arms must be exactly {list(arms)}")
    if not isinstance(pair_records, list):
        raise ValueError("pair_records must be a list")

    validated = {arm: {} for arm in arms}
    aggregates = {arm: {} for arm in arms}
    suite_maps = {split: {} for split in ("calibration", "heldout")}
    sampler_maps = {split: {} for split in ("calibration", "heldout")}
    for arm in arms:
        if not isinstance(records[arm], dict) or set(records[arm]) != {"calibration", "heldout"}:
            raise ValueError(f"{arm} must contain exactly calibration and heldout")
        for split in ("calibration", "heldout"):
            cases, seeds, _ = _expected(split)
            found, case_suites, sampler_seeds = _validate_records(
                PROTOCOL, records[arm][split], arm, split, cases, seeds, suites, latent_weight
            )
            if not suite_maps[split]:
                suite_maps[split] = case_suites
            elif case_suites != suite_maps[split]:
                raise ValueError(f"{arm}/{split} suite assignments differ across arms")
            if not sampler_maps[split]:
                sampler_maps[split] = sampler_seeds
            elif sampler_seeds != sampler_maps[split]:
                raise ValueError(f"{arm}/{split} sampler_seed assignments differ across arms")
            validated[arm][split] = found
            aggregates[arm][split] = _aggregate(found, cases, seeds, suites)

    pairs = _validate_pair_records(pair_records, validated, suites)
    comparisons = {
        name: _compare(name, baseline, candidate,
                       {arm: aggregates[arm]["heldout"] for arm in arms}, suites)
        for name, baseline, candidate in COMPARISON_PAIRS
    }
    replay = _replay_gate(validated, replay_atol)
    historical = provenance.get('uncomp8_reference') == 'historical_Joint8'
    if historical:
        for split in ('calibration', 'heldout'):
            for row in records['Uncomp8'][split]:
                source = row.get('source_binding') or {}
                if (source.get('kind') != 'historical_reuse' or source.get('arm') != 'Joint8'
                        or source.get('job') != '25912619.pbs101'):
                    raise ValueError('Historical Uncomp8 must identify the original Joint8 source')
        if replay['max_abs'] != 0:
            raise ValueError('Historical Joint8 reference must preserve the original metrics exactly')
        replay.update(status='not_run_historical_reference', available=False,
                      identity_check_pass=True)
        replay['pass'] = None
    residuals = _residual_summary(pairs)

    heldout = {arm: aggregates[arm]["heldout"] for arm in arms}
    seq_mixed = heldout["Seq_A4"]["overall"]["mixed_loss"]
    uncomp_mixed = heldout["Uncomp8"]["overall"]["mixed_loss"]
    comp_mixed = heldout["Comp8"]["overall"]["mixed_loss"]
    gap = uncomp_mixed - seq_mixed
    recovery = (uncomp_mixed - comp_mixed) / gap if gap > 0 else None
    motor_ok = heldout["Comp8"]["overall"]["first10motor_rmse"] <= heldout["Uncomp8"]["overall"]["first10motor_rmse"]
    latent_ok = heldout["Comp8"]["overall"]["future_latent_nmse"] <= heldout["Uncomp8"]["overall"]["future_latent_nmse"]
    improved_suites = sum(
        heldout["Comp8"]["by_suite"][suite]["mixed_loss"]
        < heldout["Uncomp8"]["by_suite"][suite]["mixed_loss"]
        for suite in suites
    )
    primary_pass = (
        gap > 0 and recovery is not None and recovery >= decision["major_support_fraction"]
        and motor_ok and latent_ok and improved_suites >= decision["required_suites_improved"]
    )

    joint_seq_gap = (
        heldout["Joint_A4"]["overall"]["mixed_loss"]
        - heldout["Seq_A4"]["overall"]["mixed_loss"]
    )
    no_a4_deltas = {
        name: comparisons[name]["overall"]["absolute_delta"]["mixed_loss"]
        for name in ("Seq_to_Joint_noA4_vs_Seq_noA4", "Joint_to_Seq_noA4_vs_Joint_noA4")
    }
    max_no_a4_delta = max(abs(delta) for delta in no_a4_deltas.values())
    residual_limit = residual_fraction * joint_seq_gap if joint_seq_gap > 0 else None
    geometry_pass = residual_limit is not None and max_no_a4_delta <= residual_limit
    if joint_seq_gap <= 0:
        geometry_status = "原始 Joint_A4−Seq_A4 gap 非正，几何残差比例未定义"
    elif geometry_pass:
        geometry_status = "noA4 数值残差低于阈值，可作强几何归因解释"
    else:
        geometry_status = "numeric confound：noA4 BF16 mixed 残差超过阈值，不作强几何归因"
    if historical:
        conclusion = ('满足主要贡献的效应门槛' if primary_pass else '未达到主要贡献的效应门槛') + '；对照复用原Joint8，未做同期Uncomp8重训，归因强度受限'
    elif not replay["pass"]:
        conclusion = "inconclusive：Uncomp8 对 Joint_A4 的 replay gate 未通过"
    elif primary_pass:
        conclusion = "支持“本固定配方中补偿设计是主要贡献因素”"
    else:
        conclusion = "未达到预设的主要贡献支持条件"

    training = protocol["training"]
    result = {
        "protocol": PROTOCOL,
        "complete": True,
        "cohorts": {
            "calibration": {"case_ids": [2000, 2079], "contexts": 80,
                            "seed_indices": [0], "queries_per_arm": 80},
            "heldout": {"case_ids": [3000, 3039], "contexts": 40,
                        "seed_indices": [0, 1], "queries_per_arm": 80},
            "suites": list(suites),
            "aggregation": AGGREGATION,
        },
        "training": {
            "updates": training.get("updates"),
            "contexts_per_update": training.get("contexts_per_update"),
            "latent_weight": latent_weight,
            "checkpoint": training.get("checkpoint"),
            "anchor": training.get("anchor"),
            "compensation_execution": protocol["compensation"].get("execution"),
            "codebooks_and_exponents_remain_in_anchor_coordinates": True,
        },
        "calibration": {
            arm: {key: value for key, value in aggregates[arm]["calibration"].items() if key != "_by_case"}
            for arm in arms
        },
        "heldout": {
            arm: {key: value for key, value in aggregates[arm]["heldout"].items() if key != "_by_case"}
            for arm in arms
        },
        "comparisons": comparisons,
        "replay_gate": replay,
        "pair_residuals": residuals,
        "decision": {
            "gap_definition": "Uncomp8 mixed_loss − Seq_A4 mixed_loss",
            "gap_absolute": gap,
            "gap_relative_to_Seq_A4_percent": None if seq_mixed == 0 else 100.0 * gap / seq_mixed,
            "recovered_absolute": uncomp_mixed - comp_mixed,
            "recovery_fraction": recovery,
            "recovery_percent": None if recovery is None else 100.0 * recovery,
            "recovery_defined": gap > 0,
            "comp8_motor_not_worse_than_uncomp8": motor_ok,
            "comp8_latent_not_worse_than_uncomp8": latent_ok,
            "suites_with_mixed_improvement": improved_suites,
            "required_suites_improved": decision["required_suites_improved"],
            "primary_criteria_pass": primary_pass,
            "replay_gate_pass": replay["pass"],
            "conclusion": conclusion,
            "geometry_attribution": {
                "original_positive_A4_gap": joint_seq_gap,
                "noA4_mixed_delta_by_route": no_a4_deltas,
                "max_absolute_noA4_mixed_delta": max_no_a4_delta,
                "allowed_absolute_residual": residual_limit,
                "residual_fraction_gate": residual_fraction,
                "pass": geometry_pass,
                "status": geometry_status,
            },
        },
        "provenance": _compact_provenance(provenance),
        "claim_scope": (
            "单一固定训练 seed 与冻结配方下的 fixed-input decoded-BF16/A4 证据；"
            "支持的范围仅是本补偿干预，不证明唯一或普遍因果。transport 探针是同一浮点函数的量化几何比较，"
            "单独不能证明训练回退原因。Comp8 使用显式 post-decode BF16 weight transport；"
            "不代表 native VQ 执行、bpw、kernel 或 closed-loop 结果。配对 context 计数不是显著性检验。"
        ),
    }
    if historical:
        result['training']['uncomp8_reference'] = 'historical_Joint8'
        result['claim_scope'] += ' 用户要求停止Uncomp8重跑：Uncomp8表格项仅是原Joint8的历史结果别名，非本次新训练；没有同期完整重放证据。'
    if len(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")) >= 256 * 1024:
        raise ValueError("comparison exceeds the 256 KiB compact output limit")
    return result


def _fmt(value):
    return "未定义" if value is None else f"{value:.9g}"


def render_markdown(comparison):
    """Render compact Chinese tables and the predeclared interpretation."""
    lines = [
        "# Fast-WAM R128 C80 transport 固定输入结果",
        "",
        f"Protocol：`{comparison['protocol']}`。校准集为 cases 2000–2079、seed 0；heldout 为 cases 3000–3039、seeds 0/1。",
        "Heldout 每臂为 40 contexts × 2 seeds；校准集每臂为 80 contexts。聚合顺序是 seed → context → suite，四个 suite 等权。",
        "",
        comparison["training"].get("compensation_execution", ""),
        "VQ indices/codebooks 与 B32 exponents 留在 anchor 坐标；Comp8 多做显式解码权重 transport。此处是 decoded-BF16/A4 的 fixed-input 度量。",
        "",
        "## Calibration：overall / per suite",
        "",
    ]
    for split in ("calibration", "heldout"):
        if split == "heldout":
            lines += ["", "## Heldout：overall / per suite", ""]
        cohorts = comparison[split]
        lines += ["| Arm / suite | " + " | ".join(LABELS[m] for m in METRICS) + " |", "|---|" + "---:|" * len(METRICS)]
        for arm, values in cohorts.items():
            for suite in comparison["cohorts"]["suites"]:
                row = values["by_suite"][suite]
                lines.append("| " + arm + " / " + suite + " | " + " | ".join(_fmt(row[m]) for m in METRICS) + " |")
            lines.append("| " + arm + " / overall | " + " | ".join(_fmt(values["overall"][m]) for m in METRICS) + " |")

    lines += ["", "## Heldout 配对比较", "", "绝对变化为 candidate − baseline；相对变化以 baseline 为分母。context 胜负先对每个 context 的两个 seed 求均值，n=40。"]
    for name, result in comparison["comparisons"].items():
        overall = result["overall"]
        lines += ["", f"### {name}：{result['candidate_arm']} vs {result['baseline_arm']}", "",
                  "| Metric | Baseline | Candidate | Absolute Δ | Relative Δ | Improved / worsened / tied (n=40) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for metric in METRICS:
            count = result["paired_context_win_loss"]["overall"][metric]
            relative = overall["relative_change_percent"][metric]
            lines.append(
                f"| {LABELS[metric]} | {_fmt(overall['baseline'][metric])} | {_fmt(overall['candidate'][metric])} | "
                f"{_fmt(overall['absolute_delta'][metric])} | {_fmt(relative)}% | "
                f"{count['improved']} / {count['worsened']} / {count['tied']} |"
            )
        lines += ["", "逐 suite 绝对 / 相对变化：", "", "| Suite | Metric | Absolute Δ | Relative Δ |", "|---|---|---:|---:|"]
        for suite in comparison["cohorts"]["suites"]:
            for metric in METRICS:
                change = result["by_suite"][suite]
                relative = change["relative_change_percent"][metric]
                lines.append(f"| {suite} | {LABELS[metric]} | {_fmt(change['absolute_delta'][metric])} | {_fmt(relative)}% |")

    replay = comparison["replay_gate"]
    lines += ["", "## Replay gate", ""]
    if replay.get('status') == 'not_run_historical_reference':
        lines += ['Uncomp8未重跑，表格中的Uncomp8直接复用原Joint8完整结果。以下零差值只检查复用记录一致性，**不是同期训练replay通过**。']
    else:
        lines += [f"逐 query 比较 Uncomp8 与原 Joint_A4 的七指标；阈值 {replay['atol']:.3g}，全体 max abs={replay['max_abs']:.9g}，"
                  f"{'通过' if replay['pass'] else '未通过'}。"]
    for split, values in replay["by_split"].items():
        lines.append(f"- {split}：{values['queries']} queries，max abs={values['max_abs']:.9g}；"
                     + ", ".join(f"{metric}={maximum:.9g}" for metric, maximum in values["by_metric_max_abs"].items()))

    lines += ["", "## 预设判读", "",
              f"gap = Uncomp8 − Seq_A4 = {_fmt(comparison['decision']['gap_absolute'])} "
              f"({_fmt(comparison['decision']['gap_relative_to_Seq_A4_percent'])}% of Seq_A4); "
              f"recovered = {_fmt(comparison['decision']['recovered_absolute'])}; "
              f"recovery = {_fmt(comparison['decision']['recovery_percent'])}% "
              f"({'defined' if comparison['decision']['recovery_defined'] else 'undefined：gap≤0'}).",
              f"Comp8 motor / latent 不退：{comparison['decision']['comp8_motor_not_worse_than_uncomp8']} / "
              f"{comparison['decision']['comp8_latent_not_worse_than_uncomp8']}；mixed 改善 suite："
              f"{comparison['decision']['suites_with_mixed_improvement']}/4；主要判据："
              f"{'通过' if comparison['decision']['primary_criteria_pass'] else '未通过'}。",
              f"判读：**{comparison['decision']['conclusion']}**。",
              "",
              "### noA4 BF16 几何残差",
              "",
              f"{comparison['decision']['geometry_attribution']['status']}。原始 positive A4 gap="
              f"{_fmt(comparison['decision']['geometry_attribution']['original_positive_A4_gap'])}；"
              f"最大绝对 noA4 mixed delta={_fmt(comparison['decision']['geometry_attribution']['max_absolute_noA4_mixed_delta'])}；"
              f"允许值={_fmt(comparison['decision']['geometry_attribution']['allowed_absolute_residual'])}。",
              "", "## Pair tensor residual summaries", "",
              "下表每项列出 80 queries 上的最大值与均值；这是完整 pair cohort 的汇总。",
              "", "| Anchor | Split | Residual | Max | Mean |", "|---|---|---|---:|---:|"]
    for anchor in ANCHORS:
        for split in ("calibration", "heldout"):
            values = comparison["pair_residuals"][anchor][split]
            for metric, summary in values["metrics"].items():
                lines.append(f"| {anchor} | {split} | {metric} | {summary['max']:.9g} | {summary['mean']:.9g} |")
    if comparison["provenance"]:
        lines += ["", "## Provenance", "", "```json",
                  json.dumps(comparison["provenance"], ensure_ascii=False, indent=2, sort_keys=True), "```"]
    lines += ["", comparison["claim_scope"], ""]
    return "\n".join(lines)
