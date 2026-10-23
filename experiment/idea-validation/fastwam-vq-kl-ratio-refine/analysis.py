"""Aggregate cached ratio-only refinements and matched dense controls on PBS CPU."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import statistics


ARMS = ("mse_nmse", "dual_kl")
METRICS = ("first10motor_rmse", "future_latent_nmse", "action_kl", "latent_kl")
MAX_JSON = 200 * 1024


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def guard():
    job, nodefile = os.environ.get("PBS_JOBID", "").strip(), os.environ.get("PBS_NODEFILE", "").strip()
    host = socket.getfqdn().lower().rstrip(".")
    require(job and nodefile and Path(nodefile).is_file(), "A real PBS allocation is required")
    require("login" not in host, f"Refusing analysis on login host {host}")
    nodes = {line.strip().split("/", 1)[0].lower().rstrip(".")
             for line in Path(nodefile).read_text().splitlines() if line.strip()}
    names = {host, host.split(".", 1)[0]}
    require(any(node in names or node.split(".", 1)[0] in names for node in nodes),
            "Current host is absent from PBS_NODEFILE")
    return job


def load_helper():
    path = Path(__file__).with_name("refinement_analysis.py")
    spec = importlib.util.spec_from_file_location("refinement_analysis", path)
    require(spec is not None and spec.loader is not None, "refinement_analysis.py is required")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path, helper):
    return helper.read_json(path)


def validate_protocol(p):
    require(p.get("protocol") == "fastwam-vq-kl-ratio-refine-v1"
            and p.get("arms") == list(ARMS)
            and p.get("directions") == ["saved_adam", "fresh_adam"]
            and p.get("fractions") == [1.0, 0.5, 0.25]
            and p.get("calibration_case_ids") == list(range(2000, 2080))
            and p.get("heldout_case_ids") == list(range(3000, 3040))
            and p.get("heldout_seed_indices") == [0, 1]
            and p.get("gradient_query_budget_per_arm") == 80
            and p.get("checkpoint_arm") == "mse_nmse"
            and p.get("maximum_refinement_updates") == 1,
            "Frozen ratio-only cohorts, arms, or gradient/update budget differ")
    require(p.get("source_mse_job") == "25862598.pbs101"
            and p.get("matched_dense_mse_job") == "25881910.pbs101"
            and p.get("matched_dense_kl_job") == "25882284.pbs101",
            "Frozen checkpoint or dense-control job bindings differ")


def validate_sources(p, helper):
    source = Path(p["source_root"])
    old = read_json(source / "protocol.json", helper)
    require(old.get("remote_root") == str(source)
            and old.get("source_per_channel_root") == p["source_mse_root"]
            and old.get("source_per_channel_job") == p["source_mse_job"],
            "Original protocol does not bind the MSE80 checkpoint")
    scaling = read_json(source / "results" / "prepare" / "loss_scaling.json", helper)
    require(scaling.get("complete") is True and scaling.get("protocol") == old["protocol"]
            and scaling.get("heldout_used") is False
            and scaling.get("case_ids") == p["calibration_case_ids"]
            and scaling.get("seed_indices") == [0]
            and len(scaling.get("initial_rows", [])) == 80,
            "Frozen loss scaling is not C80-only")
    training = old.get("training", {})
    require(helper.close(scaling["original_latent_weight"],
                         training.get("rho", float("nan")) * training.get("normalization_c", float("nan")))
            and helper.close(scaling["action_kl_coefficient"],
                             scaling["initial_means"]["action_mse"] / scaling["initial_means"]["action_kl"])
            and helper.close(scaling["latent_kl_coefficient"],
                             scaling["initial_means"]["latent_nmse"] / scaling["initial_means"]["latent_kl"]),
            "Original loss coefficients changed")

    mse_root = Path(p["source_mse_root"])
    mse_protocol = read_json(mse_root / "protocol.json", helper)
    training_summary = read_json(mse_root / "results" / "train_eval" / "summary.json", helper)
    receipt = read_json(mse_root / "results" / "train_eval" / "training_receipt.json", helper)
    require(mse_protocol.get("remote_root") == str(mse_root)
            and training_summary.get("complete") is True and training_summary.get("unit") == "train_eval"
            and training_summary.get("protocol") == mse_protocol["protocol"]
            and training_summary.get("pbs_jobid") == p["source_mse_job"]
            and training_summary.get("updates") == 80
            and receipt.get("complete") is True and receipt.get("protocol") == mse_protocol["protocol"]
            and receipt.get("pbs_jobid") == p["source_mse_job"] and receipt.get("updates") == 80,
            "Original MSE80 training provenance differs")
    prepare = read_json(source / "results" / "prepare" / "summary.json", helper)
    require(prepare.get("complete") is True and prepare.get("protocol") == old["protocol"]
            and prepare.get("initial_calibration_queries") == 80
            and prepare.get("control_evaluation_queries") == 160
            and prepare.get("new_teacher_queries") == 0 and prepare.get("pbs_jobid"),
            "Original MSE80 reference preparation is incomplete")
    return old, scaling, prepare["pbs_jobid"]


def parse_endpoint(raw, p, ref, arm, split, case, seed, plan, protocol, updates, scaling):
    require(raw.get("protocol") == protocol and raw.get("arm") == arm and raw.get("split") == split
            and raw.get("case_id") == case and raw.get("seed_index") == seed
            and raw.get("suite") == plan["suite"] and raw.get("plan_row") == plan
            and raw.get("sampler_seed") == plan["sampler_seeds"][seed]
            and raw.get("updates") == updates and raw.get("pbs_jobid"),
            f"{split}/{arm}/{case}/{seed} frozen endpoint binding differs")
    values = raw.get("metrics", raw)
    action_mse = values.get("first10motor_mse", values.get("action_mse"))
    latent_nmse = values.get("future_latent_nmse", values.get("latent_nmse"))
    motor_rmse = values.get("first10motor_rmse")
    if motor_rmse is None and action_mse is not None:
        motor_rmse = math.sqrt(float(action_mse))
    metrics = {"first10motor_mse": action_mse, "first10motor_rmse": motor_rmse,
               "future_latent_nmse": latent_nmse, "action_kl": values.get("action_kl"),
               "latent_kl": values.get("latent_kl")}
    for name, value in metrics.items():
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and float(value) >= 0,
                f"{split}/{arm}/{case}/{seed} metric {name} is missing or invalid")
        metrics[name] = float(value)
    require(math.isclose(metrics["first10motor_rmse"] ** 2, metrics["first10motor_mse"],
                         rel_tol=1e-5, abs_tol=1e-8),
            f"{split}/{arm}/{case}/{seed} motor MSE/RMSE disagree")
    metrics["objective_mse_nmse"] = (metrics["first10motor_mse"]
        + scaling["original_latent_weight"] * metrics["future_latent_nmse"])
    metrics["objective_dual_kl"] = (scaling["action_kl_coefficient"] * metrics["action_kl"]
        + scaling["original_latent_weight"] * scaling["latent_kl_coefficient"] * metrics["latent_kl"])
    expected = metrics["objective_dual_kl" if arm == "dual_kl" else "objective_mse_nmse"]
    objective = values.get("objective_loss", values.get("objective"))
    require(isinstance(objective, (int, float)) and math.isfinite(float(objective))
            and math.isclose(float(objective), expected, rel_tol=1e-5, abs_tol=1e-8),
            f"{split}/{arm}/{case}/{seed} own objective differs from the frozen formula")
    metrics["objective_loss"] = float(objective)
    matrices = {name: ref.matrix(raw.get(name), name)
                for name in ("action", "commands", "teacher_action", "teacher_commands")}
    return {"case_id": case, "seed_index": seed, "suite": plan["suite"],
            "pbs_jobid": raw["pbs_jobid"], "metrics": metrics, "raw": matrices,
            "source_record_reused": raw.get("source_record_reused") is True, "source": raw}


def aggregate(records, suites, split):
    cases = {}
    for row in records:
        case = row["case_id"]
        target = cases.setdefault(case, {"suite": row["suite"], "seeds": {}})
        require(target["suite"] == row["suite"] and row["seed_index"] not in target["seeds"],
                f"Duplicate or changed case binding in {split}/{case}")
        target["seeds"][row["seed_index"]] = row["metrics"]
    expected_seeds = {0} if split == "calibration" else {0, 1}
    case_values = {}
    for case, row in cases.items():
        require(set(row["seeds"]) == expected_seeds, f"{split}/{case} seed coverage differs")
        names = row["seeds"][0].keys()
        case_values[case] = {name: sum(seed[name] for seed in row["seeds"].values()) / len(expected_seeds)
                             for name in names}
        case_values[case]["_suite"] = row["suite"]
    per_suite_expected = 20 if split == "calibration" else 10
    by_suite = {}
    for suite in suites:
        selected = [row for row in case_values.values() if row["_suite"] == suite]
        require(len(selected) == per_suite_expected, f"{split}/{suite} case balance differs")
        by_suite[suite] = {name: sum(row[name] for row in selected) / len(selected)
                           for name in next(iter(case_values.values())) if name != "_suite"}
        by_suite[suite]["cases"] = len(selected)
    overall = {name: sum(by_suite[suite][name] for suite in suites) / len(suites)
               for name in next(iter(case_values.values())) if name != "_suite"}
    queries = len(records)
    require(len(cases) == (80 if split == "calibration" else 40)
            and queries == (80 if split == "calibration" else 80),
            f"{split} aggregate coverage differs")
    return {"cases": len(cases), "queries": queries, "case_values": case_values,
            "by_suite": by_suite, "overall": overall}


def read_panel(p, helper, ref, rows, suites, root, arm, split, protocol,
               updates, scaling, source_records=None, teachers=None,
               allowed_jobs=None, source_jobs=None, ratio_tags=False, source_protocol=None):
    cases = p["calibration_case_ids"] if split == "calibration" else p["heldout_case_ids"]
    seeds = [0] if split == "calibration" else p["heldout_seed_indices"]
    records = []
    for case in cases:
        for seed in seeds:
            raw = helper.read_json(helper.endpoint_path(root, split, arm, case, seed))
            require(raw.get("unit", arm) == arm, f"{split}/{arm}/{case}/{seed} unit differs")
            if ratio_tags:
                require(raw.get("source_updates") == 80
                        and raw.get("refinement_updates") == updates - 80
                        and raw.get("total_updates") == updates and raw.get("updates") == updates,
                        f"{split}/{arm}/{case}/{seed} ratio update tags differ")
            reused = raw.get("source_record_reused") is True
            if allowed_jobs is not None and raw.get("pbs_jobid") not in allowed_jobs:
                require(reused and raw.get("pbs_jobid") == raw.get("source_record_pbs_jobid")
                        and source_jobs and raw.get("pbs_jobid") in source_jobs,
                        f"{split}/{arm}/{case}/{seed} PBS id is not an attempt or explicit source reuse")
            if reused:
                require(split == "calibration" and updates == 80 and source_records is not None
                        and raw.get("source_record_protocol") == source_protocol
                        and raw.get("source_record_arm") == "mse_nmse"
                        and raw.get("source_record_pbs_jobid") == source_records[(case, seed)]["pbs_jobid"]
                        and raw.get("curated_by_pbs_jobid") in (allowed_jobs or set()),
                        f"Invalid source no-step reuse provenance for {split}/{arm}/{case}/{seed}")
            parsed = parse_endpoint(raw, p, ref, arm, split, case, seed, rows[case], protocol,
                                    updates, scaling)
            if teachers is not None:
                base = source_records[(case, seed)] if source_records is not None else None
                expected_teacher = teachers[(case, seed)]
                require((parsed["raw"]["teacher_action"], parsed["raw"]["teacher_commands"])
                        == expected_teacher, f"Shared teacher changed for {split}/{arm}/{case}/{seed}")
                if reused and base is not None:
                    require(parsed["raw"]["action"] == base["raw"]["action"]
                            and parsed["raw"]["commands"] == base["raw"]["commands"],
                            f"Reused source outputs changed for {split}/{arm}/{case}/{seed}")
                    for key in ("first10motor_mse", "first10motor_rmse", "future_latent_nmse", "action_kl", "latent_kl"):
                        require(helper.close(parsed["metrics"][key], base["metrics"][key]),
                                f"Reused source metric changed for {split}/{arm}/{case}/{seed}/{key}")
            records.append(parsed)
    return aggregate(records, suites, split), records


def validate_selection(selection, arm, helper):
    require(isinstance(selection, dict) and selection.get("complete") is True
            and selection.get("heldout_used") is False and selection.get("sampling_scope") == "C80"
            and selection.get("method") == "ratio_only_full_C80_gradient_actual_loss_selection",
            f"{arm} selection is incomplete or not C80-only")
    baseline, candidates, conditions = (selection.get("baseline"), selection.get("candidates"),
                                         selection.get("conditions"))
    require(isinstance(baseline, dict) and isinstance(candidates, list) and len(candidates) == 6
            and isinstance(conditions, list) and len(conditions) == 7,
            f"{arm} selection must contain no-step plus six candidates")
    for row in conditions:
        value = float(row.get("objective_mean", float("nan")))
        require(row.get("queries") == 80 and math.isfinite(value) and value >= 0,
                f"{arm} selection objective or query budget is invalid")
    chosen = selection.get("chosen")
    require(isinstance(chosen, dict) and any(
        all(chosen.get(key) == row.get(key) for key in ("condition", "direction", "fraction"))
        and helper.close(chosen.get("objective_mean", float("nan")), row.get("objective_mean", float("nan")))
        for row in conditions), f"{arm} chosen condition is absent from its C80 grid")
    minimum = min(float(row["objective_mean"]) for row in conditions)
    require(float(chosen["objective_mean"]) <= minimum + max(1e-12, abs(minimum) * 1e-8),
            f"{arm} chosen condition is not the C80 minimum")
    require(all(row.get("queries") == 80 for row in candidates), f"{arm} candidate budget differs")
    return chosen


def validate_new_arm(p, helper, ref, rows, suites, scaling, root, arm,
                     known_jobs, source_jobs, source_records, source_teachers, source_protocol):
    unit_root = root / "results" / arm
    summary = helper.read_json(unit_root / "summary.json")
    receipt = helper.read_json(unit_root / "training_receipt.json")
    selection = summary.get("selection", {})
    chosen = validate_selection(selection, arm, helper)
    job, refine = summary.get("pbs_jobid"), summary.get("refinement_updates")
    require(summary.get("complete") is True and summary.get("protocol") == p["protocol"]
            and summary.get("unit") == arm and summary.get("checkpoint_arm") == "mse_nmse"
            and job in known_jobs and summary.get("source_updates") == 80
            and refine in (0, 1) and summary.get("total_updates") == 80 + refine
            and summary.get("gradient_queries_new") == 0 and summary.get("gradient_queries_reused") == 80
            and summary.get("new_teacher_queries") == 0 and summary.get("new_backward_queries") == 0
            and summary.get("evaluation_queries") == 160 and summary.get("calibration_queries") == 80
            and summary.get("heldout_queries") == 80 and summary.get("ratio_only") is True
            and summary.get("D_unchanged") is True and summary.get("trainable_ratio_coordinates") == 614
            and summary.get("frozen_channel_coordinates") == 1806863,
            f"{arm} ratio-only summary/provenance gate failed")
    for key in ("source_parity_passed", "deployment_parity_passed", "frozen_verified"):
        helper.required_pass(summary, key, f"{arm} {key}")
    require((refine == 0) == (chosen.get("condition") == "no_step")
            and helper.read_json(unit_root / "selection.json") == selection,
            f"{arm} refinement count or saved selection differs")
    require(receipt.get("complete") is True and receipt.get("protocol") == p["protocol"]
            and receipt.get("pbs_jobid") == job and receipt.get("source_updates") == 80
            and receipt.get("refinement_updates") == refine and receipt.get("total_updates") == 80 + refine,
            f"{arm} training receipt differs from summary")
    for key in ("ratio_only", "D_unchanged", "gradient_queries_new", "gradient_queries_reused",
                "new_teacher_queries", "new_backward_queries"):
        require(receipt.get(key) == summary.get(key), f"{arm} receipt {key} differs")
    for key in ("source_parity_passed", "deployment_parity_passed", "frozen_verified"):
        helper.required_pass(receipt, key, f"{arm} receipt {key}")

    panels = {}
    records = {}
    for split in ("calibration", "heldout"):
        panels[split], records[split] = read_panel(
            p, helper, ref, rows, suites, root, arm, split, p["protocol"], 80 + refine,
            scaling, source_records[split], source_teachers[split], known_jobs, source_jobs,
            True, source_protocol)
    selected_mean = panels["calibration"]["overall"]["objective_" + arm]
    require(helper.close(selected_mean, chosen["objective_mean"]),
            f"{arm} selected C80 objective differs from final calibration endpoints")
    return summary, selection, panels, records


def validate_dense(p, helper, ref, rows, suites, scaling, root, arm, expected_job,
                   source_jobs, source_records, source_teachers):
    root = Path(root)
    protocol = helper.read_json(root / "protocol.json")
    expected_protocol = ("fastwam-vq-kl-fullbatch-refine-v1" if arm == "mse_nmse"
                         else "fastwam-vq-kl-mse-warmstart-v1")
    require(protocol.get("protocol") == expected_protocol and protocol.get("remote_root") == str(root),
            f"Dense {arm} protocol differs")
    summary = helper.read_json(root / "results" / arm / "summary.json")
    require(summary.get("complete") is True and summary.get("protocol") == expected_protocol
            and summary.get("pbs_jobid") == expected_job and summary.get("source_updates") == 80
            and summary.get("refinement_updates") == 1 and summary.get("total_updates") == 81
            and summary.get("heldout_used") is False,
            f"Dense {arm} completed summary differs")
    if arm == "dual_kl":
        require(summary.get("unit") == arm and summary.get("checkpoint_arm") == "mse_nmse",
                "Dense dual-KL checkpoint binding differs")
    selection = summary.get("selection", {})
    require(selection.get("complete") is True and selection.get("heldout_used") is False
            and selection.get("sampling_scope") == "C80",
            f"Dense {arm} selection was not C80-only")
    known = helper.pbs_jobs_in_handles(root) | {expected_job}
    stats, _ = read_panel(p, helper, ref, rows, suites, root, arm, "heldout", expected_protocol,
                          81, scaling, source_records["heldout"], source_teachers["heldout"],
                          known, source_jobs)
    return summary, stats


def paired(left, right, suites, metrics):
    cases = sorted(left["case_values"])
    require(cases == sorted(right["case_values"]), "Paired panel case sets differ")
    suite_for = {case: left["case_values"][case]["_suite"] for case in cases}
    result = {}
    for metric in metrics:
        delta = {case: right["case_values"][case][metric] - left["case_values"][case][metric]
                 for case in cases}
        by_suite = {suite: sum(delta[c] for c in cases if suite_for[c] == suite)
                    / sum(suite_for[c] == suite for c in cases) for suite in suites}
        vals = list(delta.values())
        result[metric] = {"cases": len(vals), "mean_delta_equal_suite": sum(by_suite.values()) / len(suites),
                          "median_delta": statistics.median(vals), "improved": sum(v < 0 for v in vals),
                          "worsened": sum(v > 0 for v in vals), "tied": sum(v == 0 for v in vals),
                          "by_suite_mean_delta": by_suite, "case_deltas": delta}
    return result


def interaction(ratio, dense, suites, suite_for, metrics):
    result = {}
    for metric in metrics:
        values = {case: ratio[metric]["case_deltas"][case] - dense[metric]["case_deltas"][case]
                  for case in ratio[metric]["case_deltas"]}
        by_suite = {suite: sum(v for case, v in values.items() if suite_for[case] == suite)
                    / sum(suite_for[case] == suite for case in values) for suite in suites}
        result[metric] = {"cases": len(values), "mean_delta_equal_suite": sum(by_suite.values()) / len(suites),
                          "median_delta": statistics.median(values.values()), "by_suite_mean_delta": by_suite,
                          "case_deltas": values}
    return result


def report(comparison, paired_rows, interactions, decision, suites):
    def f(value):
        return f"{value:.8g}"
    names = {"original_mse": "原 MSE80", "dense_mse": "dense MSE-refined",
             "dense_kl": "dense KL-refined", "ratio_mse": "ratio-only MSE-refined",
             "ratio_kl": "ratio-only KL-refined"}
    lines = ["# Ratio-only KL refinement 的固定输入对照", "",
             "两组新 arm 均从同一 MSE80 checkpoint 出发，固定 D、只更新 614 个 clipping ratios，且复用已有 C80 gradient。每组最多物化一次缓存梯度更新；没有新增 backward 或 teacher query。Held-out case 先在 case 内平均两个 seed，再按四个 suite 等权。", "",
             "## Held-out means", "", "| Condition | First10 motor RMSE | Future latent NMSE | Action KL | Latent KL |",
             "|---|---:|---:|---:|---:|"]
    for panel, name in names.items():
        vals = comparison["heldout"][panel]["overall"]
        lines.append(f"| {name} | " + " | ".join(f(vals[m]) for m in METRICS) + " |")
    lines += ["", "## Paired deltas (right minus left)", "",
              "负值代表该误差或 KL 下降。Improved 统计单位为 40 个 seed-averaged case；仅为固定 cohort 描述统计，不作显著性或训练复现推断。", "",
              "| Contrast | Metric | Equal-suite mean Δ | Median case Δ | Improved / 40 | Suite mean Δ |",
              "|---|---|---:|---:|---:|---|"]
    for contrast, outcomes in paired_rows.items():
        for metric in METRICS:
            item = outcomes[metric]
            suite_text = ", ".join(f"{s}: {f(item['by_suite_mean_delta'][s])}" for s in suites)
            lines.append(f"| {contrast} | {metric} | {f(item['mean_delta_equal_suite'])} | "
                         f"{f(item['median_delta'])} | {item['improved']}/40 | {suite_text} |")
    lines += ["", "## Loss × trainable-subspace interaction", "",
              "定义为 `(ratio KL − ratio MSE) − (dense KL − dense MSE)`，逐 case 计算后汇总；它描述固定终点的差异，不识别机制。", "",
              "| Metric | Equal-suite interaction | Median case | Suite means |", "|---|---:|---:|---|"]
    for metric, item in interactions.items():
        suite_text = ", ".join(f"{s}: {f(item['by_suite_mean_delta'][s])}" for s in suites)
        lines.append(f"| {metric} | {f(item['mean_delta_equal_suite'])} | {f(item['median_delta'])} | {suite_text} |")
    screen = decision["ratio_kl_vs_ratio_mse_screen"]
    lines += ["", "## Engineering screen", "",
              f"Ratio KL vs ratio MSE screen: **{screen['passes']}** (two primary means lower={screen['both_primary_means_lower']}; "
              f"motor median Δ<0={screen['motor_median_below_zero']}; motor improved={screen['motor_improved_cases']}/40, threshold 24).",
              f"Ratio KL lowers both primary means vs original MSE80: **{decision['ratio_kl_vs_original_mse']['both_primary_means_lower']}**. "
              f"Original-MSE follow-up screen: **{decision['ratio_kl_vs_original_mse']['passes']}**; "
              f"both raw KL terms lower vs ratio MSE: **{decision['raw_kl_both_lower_vs_ratio_mse']}**; "
              f"both raw KL terms lower vs original MSE: **{decision['raw_kl_both_lower_vs_original_mse']}**.", "",
              "证据范围是 decoded-BF16 compensated VQ、dynamic fakeA4/BF16 KV 的 fixed-input endpoint；没有 native-kernel speed、rollout 或 closed-loop 证据。Gripper 不是本分析重点。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    job = guard()
    p = json.loads(Path(args.protocol).read_text(encoding="utf-8-sig"))
    validate_protocol(p)
    helper = load_helper()
    ref = helper.load_reference()
    old, scaling, prepare_job = validate_sources(p, helper)
    rows, suites = helper.collect_plan(p, ref)
    require(len(rows) == 120 and len(suites) == 4, "Frozen plan coverage differs")

    source_root = Path(p["source_root"])
    source_stats, source_records, source_teachers = {}, {}, {}
    for split in ("calibration", "heldout"):
        source_stats[split], baseline_rows = read_panel(
            p, helper, ref, rows, suites, source_root, "mse_nmse", split, old["protocol"], 80,
            scaling, allowed_jobs={prepare_job})
        source_records[split] = {(r["case_id"], r["seed_index"]): r for r in baseline_rows}
        source_teachers[split] = {key: (r["raw"]["teacher_action"], r["raw"]["teacher_commands"])
                                  for key, r in source_records[split].items()}
    source_jobs = {prepare_job}
    new_root = Path(p["remote_root"])
    new_jobs = helper.pbs_jobs_in_handles(new_root)
    new_summaries, new_selections, new_stats, new_records = {}, {}, {"calibration": {}, "heldout": {}}, {"calibration": {}, "heldout": {}}
    for arm in ARMS:
        summary, selection, panels, recs = validate_new_arm(
            p, helper, ref, rows, suites, scaling, new_root, arm, new_jobs, source_jobs,
            source_records, source_teachers, old["protocol"])
        new_summaries[arm], new_selections[arm] = summary, selection
        for split in ("calibration", "heldout"):
            new_stats[split][arm], new_records[split][arm] = panels[split], recs[split]

    dense_mse_root, dense_kl_root = Path(p["matched_dense_root"]), Path(p["matched_dense_kl_root"])
    dense_mse_summary, dense_mse = validate_dense(
        p, helper, ref, rows, suites, scaling, dense_mse_root, "mse_nmse",
        p["matched_dense_mse_job"], source_jobs, source_records, source_teachers)
    dense_kl_summary, dense_kl = validate_dense(
        p, helper, ref, rows, suites, scaling, dense_kl_root, "dual_kl",
        p["matched_dense_kl_job"], source_jobs, source_records, source_teachers)
    held = {"original_mse": source_stats["heldout"], "dense_mse": dense_mse, "dense_kl": dense_kl,
            "ratio_mse": new_stats["heldout"]["mse_nmse"], "ratio_kl": new_stats["heldout"]["dual_kl"]}
    contrasts = {"ratio_kl_minus_ratio_mse": ("ratio_mse", "ratio_kl"),
                 "ratio_mse_minus_original_mse": ("original_mse", "ratio_mse"),
                 "ratio_kl_minus_original_mse": ("original_mse", "ratio_kl"),
                 "ratio_mse_minus_dense_mse": ("dense_mse", "ratio_mse"),
                 "ratio_kl_minus_dense_kl": ("dense_kl", "ratio_kl"),
                 "dense_kl_minus_dense_mse": ("dense_mse", "dense_kl")}
    paired_rows = {name: paired(held[left], held[right], suites, METRICS)
                   for name, (left, right) in contrasts.items()}
    suite_for = {case: source_stats["heldout"]["case_values"][case]["_suite"]
                 for case in source_stats["heldout"]["case_values"]}
    interactions = interaction(paired_rows["ratio_kl_minus_ratio_mse"],
                               paired_rows["dense_kl_minus_dense_mse"], suites, suite_for, METRICS)

    main_pair, original_pair = paired_rows["ratio_kl_minus_ratio_mse"], paired_rows["ratio_kl_minus_original_mse"]
    primary_means = lambda left, right: all(
        held[left]["overall"][metric] > held[right]["overall"][metric]
        for metric in ("first10motor_rmse", "future_latent_nmse"))
    screen = {"both_primary_means_lower": primary_means("ratio_mse", "ratio_kl"),
              "motor_median_below_zero": main_pair["first10motor_rmse"]["median_delta"] < 0,
              "motor_improved_cases": main_pair["first10motor_rmse"]["improved"]}
    screen["passes"] = (screen["both_primary_means_lower"] and screen["motor_median_below_zero"]
                        and screen["motor_improved_cases"] >= 24)
    original_motor = original_pair["first10motor_rmse"]
    original_screen = {"both_primary_means_lower": primary_means("original_mse", "ratio_kl"),
                       "motor_median_delta": original_motor["median_delta"],
                       "motor_improved_cases": original_motor["improved"]}
    original_screen["passes"] = (original_screen["both_primary_means_lower"]
                                  and original_screen["motor_median_delta"] < 0
                                  and original_screen["motor_improved_cases"] >= 24)
    decision = {"protocol": p["protocol"], "complete": True, "pbs_jobid": job,
                "ratio_kl_vs_ratio_mse_screen": screen,
                "ratio_kl_vs_original_mse": original_screen,
                "ratio_mse_vs_original_mse_both_primary_means_lower": primary_means("original_mse", "ratio_mse"),
                "ratio_mse_vs_dense_mse": {m: paired_rows["ratio_mse_minus_dense_mse"][m]["mean_delta_equal_suite"] for m in METRICS},
                "ratio_kl_vs_dense_kl": {m: paired_rows["ratio_kl_minus_dense_kl"][m]["mean_delta_equal_suite"] for m in METRICS},
                "raw_kl_both_lower_vs_ratio_mse": all(held["ratio_kl"]["overall"][m] < held["ratio_mse"]["overall"][m]
                                                       for m in ("action_kl", "latent_kl")),
                "raw_kl_both_lower_vs_original_mse": all(held["ratio_kl"]["overall"][m] < held["original_mse"]["overall"][m]
                                                         for m in ("action_kl", "latent_kl")),
                "raw_kl_mean_deltas_vs_ratio_mse": {m: main_pair[m]["mean_delta_equal_suite"] for m in ("action_kl", "latent_kl")},
                "raw_kl_mean_deltas_vs_original_mse": {m: original_pair[m]["mean_delta_equal_suite"] for m in ("action_kl", "latent_kl")},
                "loss_subspace_interaction": interactions,
                "research_screen_passes": screen["passes"] and original_screen["passes"],
                "unit_of_analysis": "40 held-out cases after seed mean within case; four suites equally weighted",
                "statistical_significance": False, "independent_training_replicates": False,
                "rollouts": 0, "closed_loop_evaluated": False, "native_kernel_speed_evaluated": False}
    comparison = {"protocol": p["protocol"], "complete": True, "pbs_jobid": job,
                  "loss_scaling": {key: scaling[key] for key in (
                      "original_latent_weight", "action_kl_coefficient", "latent_kl_coefficient")},
                  "heldout": {name: {"overall": stats["overall"], "by_suite": stats["by_suite"]}
                              for name, stats in held.items()},
                  "calibration_new_arms": {arm: {"objective_mean": new_stats["calibration"][arm]["overall"]["objective_" + arm],
                                                  "chosen": new_selections[arm]["chosen"],
                                                  "metrics": new_stats["calibration"][arm]["overall"]}
                                            for arm in ARMS},
                  "paired_contrasts": {name: {m: {k: v for k, v in item.items() if k != "case_deltas"}
                                               for m, item in values.items()}
                                       for name, values in paired_rows.items()},
                  "loss_subspace_interaction": {m: {k: v for k, v in item.items() if k != "case_deltas"}
                                                for m, item in interactions.items()},
                  "evidence_scope": p["evidence_scope"]}
    paired_details = {"protocol": p["protocol"], "complete": True,
                      "unit_of_analysis": decision["unit_of_analysis"],
                      "contrasts": paired_rows, "loss_subspace_interaction": interactions}
    summary = {"protocol": p["protocol"], "complete": True, "pbs_jobid": job, "arms": list(ARMS),
               "calibration_records": 160, "heldout_records": 160,
               "new_teacher_queries": 0, "new_backward_queries": 0,
               "gradient_queries_new_per_arm": 0, "gradient_queries_reused_per_arm": 80,
               "D_unchanged": True, "ratio_only": True, "source_checkpoint_arm": "mse_nmse",
               "source_mse_job": p["source_mse_job"], "source_prepare_job": prepare_job,
               "gradient_source_jobs": p["gradient_source_jobs"],
               "refinement_updates": {arm: new_summaries[arm]["refinement_updates"] for arm in ARMS},
               "total_updates": {arm: new_summaries[arm]["total_updates"] for arm in ARMS},
               "selection_conditions": {arm: new_selections[arm]["conditions"] for arm in ARMS},
               "source_parity_passed": True, "deployment_parity_passed": True, "frozen_verified": True}
    out = Path(p["remote_root"]) / "results"
    helper.save_json(out / "comparison.json", comparison, MAX_JSON)
    helper.save_json(out / "paired_details.json", paired_details, MAX_JSON)
    helper.save_json(out / "decision.json", decision, MAX_JSON)
    helper.save_json(out / "summary.json", summary, MAX_JSON)
    (out / "ANALYSIS.zh.md").write_text(report(comparison, paired_rows, interactions, decision, suites), encoding="utf-8")
    print("RATIO_REFINE_ANALYSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
