"""Aggregate the frozen MSE-checkpoint to dual-KL warm-start screen on PBS CPU."""
import argparse
import importlib.util
import math
from pathlib import Path


ARM = "dual_kl"
PANELS = ("original_mse", "matched_mse_refined", "previous_dual_refined", "warmstart_dual")
METRICS = ("first10motor_rmse", "full32motor_rmse", "late22motor_rmse",
           "future_latent_nmse", "action_kl", "latent_kl")
MAX_JSON_BYTES = 200 * 1024


def load_sibling(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Required CPU aggregate helper is missing: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-8)


def pass_check(container, key, label, helper):
    helper.require(key in container and helper.passed(container[key]), f"{label} did not explicitly pass")


def validate_protocol(p, helper):
    helper.require(p.get("protocol") == "fastwam-vq-kl-mse-warmstart-v1"
                   and p.get("arm") == ARM and p.get("checkpoint_arm") == "mse_nmse"
                   and p.get("diagnostic_unit") == "mse_dual",
                   "Warm-start protocol or checkpoint/loss binding differs")
    helper.require(p.get("gradient_queries_new") == 0 and p.get("gradient_queries_reused") == 80
                   and p.get("maximum_refinement_updates") == 1,
                   "Warm-start gradient or update budget differs")
    helper.require(p.get("directions") == ["saved_adam", "fresh_adam"]
                   and p.get("fractions") == [1.0, 0.5, 0.25]
                   and p.get("calibration_case_ids") == list(range(2000, 2080))
                   and p.get("heldout_case_ids") == list(range(3000, 3040))
                   and p.get("heldout_seed_indices") == [0, 1],
                   "Frozen candidate grid or query cohorts differ")
    helper.require(p.get("source_mse_job") == "25862598.pbs101"
                   and p.get("diagnostic_job") == "25881167.pbs101"
                   and p.get("matched_mse_job") == "25881910.pbs101"
                   and p.get("matched_dual_job") == "25881829.pbs101",
                   "Frozen source, diagnostic, or matched-run job binding differs")


def validate_loss_scaling(p, helper):
    source_root = Path(p["source_root"])
    source_protocol = helper.read_json(source_root / "protocol.json")
    helper.require(source_protocol.get("remote_root") == str(source_root)
                   and source_protocol.get("source_per_channel_root") == p["source_mse_root"],
                   "Original dual-KL source protocol binding differs")
    scaling = helper.read_json(source_root / "results" / "prepare" / "loss_scaling.json")
    helper.require(scaling.get("complete") is True and scaling.get("protocol") == source_protocol["protocol"]
                   and scaling.get("heldout_used") is False
                   and scaling.get("case_ids") == p["calibration_case_ids"]
                   and scaling.get("seed_indices") == [0] and len(scaling.get("initial_rows", [])) == 80,
                   "Frozen loss scaling is not complete C80-only calibration")
    training = source_protocol.get("training", {})
    helper.require(close(scaling["original_latent_weight"],
                         training.get("rho", float("nan")) * training.get("normalization_c", float("nan"))),
                   "Original latent loss weight differs from the source training protocol")
    initial = scaling.get("initial_means", {})
    helper.require(close(scaling["action_kl_coefficient"], initial["action_mse"] / initial["action_kl"])
                   and close(scaling["latent_kl_coefficient"], initial["latent_nmse"] / initial["latent_kl"]),
                   "Frozen KL coefficients differ from the original calibration rule")
    return source_protocol, scaling


def validate_source_training(p, helper):
    root = Path(p["source_mse_root"])
    source_protocol = helper.read_json(root / "protocol.json")
    summary = helper.read_json(root / "results" / "train_eval" / "summary.json")
    receipt = helper.read_json(root / "results" / "train_eval" / "training_receipt.json")
    helper.require(source_protocol.get("remote_root") == str(root)
                   and summary.get("complete") is True and summary.get("unit") == "train_eval"
                   and summary.get("protocol") == source_protocol.get("protocol")
                   and summary.get("pbs_jobid") == p["source_mse_job"]
                   and summary.get("updates") == 80 and summary.get("new_teacher_queries") == 0,
                   "Original MSE/NMSE training summary is incomplete or mismatched")
    # The source receipt intentionally has no unit field; bind it by protocol, job and update count.
    helper.require(receipt.get("complete") is True and receipt.get("protocol") == source_protocol.get("protocol")
                   and receipt.get("pbs_jobid") == p["source_mse_job"] and receipt.get("updates") == 80,
                   "Original MSE/NMSE training receipt is incomplete or mismatched")
    return source_protocol


def collect_plan(p, helper, ref):
    rows, suites = helper.collect_plan(p, ref)
    return rows, suites


def parse_panel(p, helper, ref, scaling, plan_rows, suites, root, arm, split, expected_updates,
                source_protocol=None, expected_job=None, allowed_jobs=None, require_update_tags=False):
    cases = p["calibration_case_ids"] if split == "calibration" else p["heldout_case_ids"]
    seeds = [0] if split == "calibration" else p["heldout_seed_indices"]
    records = []
    for case in cases:
        for seed in seeds:
            raw = ref.read_json(helper.endpoint_path(root, split, arm, case, seed))
            helper.require(raw.get("unit", arm) == arm, f"{arm}/{split}/{case}/{seed} unit differs")
            if require_update_tags:
                helper.require(raw.get("source_updates") == 80
                               and raw.get("refinement_updates") == expected_updates - 80
                               and raw.get("total_updates") == expected_updates
                               and raw.get("updates") == expected_updates,
                               f"{arm}/{split}/{case}/{seed} update tags differ")
            if expected_job is not None:
                helper.require(raw.get("pbs_jobid") == expected_job,
                               f"{arm}/{split}/{case}/{seed} PBS job differs")
            if allowed_jobs is not None:
                helper.require(raw.get("pbs_jobid") in allowed_jobs,
                               f"{arm}/{split}/{case}/{seed} PBS job is absent from the matched handles")
            records.append(helper.parse_record(raw, p, arm, split, case, seed, plan_rows[case], ref,
                                                scaling, expected_updates=expected_updates,
                                                source_protocol=source_protocol))
    stats = helper.aggregate_full(records, suites, ref)
    expected_cases = len(cases)
    expected_queries = 80
    expected_per_suite = 20 if split == "calibration" else 10
    helper.require(stats["cases"] == expected_cases and stats["queries"] == expected_queries
                   and all(stats["by_suite"][suite]["cases"] == expected_per_suite for suite in suites),
                   f"{split}/{arm} endpoint coverage or suite balance differs")
    return {"records": records, "stats": stats}


def assert_shared_teacher(panel, original, helper, label):
    baseline = {(row["case_id"], row["seed_index"]): row["raw"]
                for row in original["records"]}
    for record in panel["records"]:
        key = (record["case_id"], record["seed_index"])
        for name in ("teacher_action", "teacher_commands"):
            helper.require(record["raw"][name] == baseline[key][name],
                           f"{label}/{key} changed the frozen shared {name}")


def validate_matched(p, helper, ref, scaling, plan_rows, suites):
    root = Path(p["matched_root"])
    protocol = ref.read_json(root / "protocol.json")
    helper.require(protocol.get("protocol") == "fastwam-vq-kl-fullbatch-refine-v1"
                   and protocol.get("remote_root") == str(root),
                   "Matched refinement protocol differs")
    result = {}
    expected_jobs = {"mse_nmse": p["matched_mse_job"], "dual_kl": p["matched_dual_job"]}
    matched_jobs = helper.pbs_jobs_in_handles(root) | set(expected_jobs.values())
    for arm in ("mse_nmse", "dual_kl"):
        summary = ref.read_json(root / "results" / arm / "summary.json")
        receipt = ref.read_json(root / "results" / arm / "training_receipt.json")
        selection = helper.validate_selection(summary, arm, protocol)
        helper.require(summary.get("complete") is True and summary.get("protocol") == protocol["protocol"]
                       and summary.get("unit") == arm and summary.get("pbs_jobid") == expected_jobs[arm]
                       and summary.get("source_updates") == 80 and summary.get("refinement_updates") == 1
                       and summary.get("total_updates") == 81 and summary.get("heldout_used") is False
                       and summary.get("new_teacher_queries") == 0,
                       f"Matched {arm} summary is incomplete or mismatched")
        helper.require(receipt.get("complete") is True and receipt.get("protocol") == protocol["protocol"]
                       and receipt.get("unit") == arm and receipt.get("pbs_jobid") == expected_jobs[arm]
                       and receipt.get("source_updates") == 80 and receipt.get("refinement_updates") == 1
                       and receipt.get("total_updates") == 81,
                       f"Matched {arm} receipt is incomplete or mismatched")
        for key in ("source_parity_passed", "deployment_parity_passed", "frozen_verified"):
            pass_check(summary, key, f"Matched {arm} {key}", helper)
        helper.require(ref.read_json(root / "results" / arm / "selection.json") == selection,
                       f"Matched {arm} saved selection differs from its summary")
        result[arm] = {}
        for split in ("calibration", "heldout"):
            result[arm][split] = parse_panel(p, helper, ref, scaling, plan_rows, suites, root, arm, split, 81,
                                             source_protocol=protocol["protocol"],
                                             allowed_jobs=matched_jobs,
                                             require_update_tags=True)
    return result, protocol


def validate_diagnostic(p, helper):
    root = Path(p["diagnostic_root"])
    protocol = helper.read_json(root / "protocol.json")
    summary = helper.read_json(root / "results" / "mse_dual" / "summary.json")
    conditions = helper.read_json(root / "results" / "mse_dual" / "direction_summary.json")
    helper.require(protocol.get("remote_root") == str(root)
                   and protocol.get("protocol") == "fastwam-vq-kl-direction-v1"
                   and summary.get("pbs_jobid") == p["diagnostic_job"]
                   and summary.get("checkpoint_arm") == "mse_nmse" and summary.get("loss") == "dual_kl"
                   and summary.get("complete") is True and summary.get("heldout_used") is False
                   and summary.get("gradient_query_count") == 80 and summary.get("training_updates") == 0,
                   "Reusable MSE-start dual-KL diagnostic is incomplete or mismatched")
    helper.require(isinstance(conditions, list) and len(conditions) == 6
                   and summary.get("direction_conditions") == conditions,
                   "Diagnostic does not contain the six complete frozen C80 candidates")
    seen = set()
    for row in conditions:
        key = (row.get("direction"), float(row.get("fraction", -1)))
        helper.require(key not in seen and key[0] in p["directions"] and key[1] in p["fractions"]
                       and row.get("queries") == 80 and math.isfinite(float(row.get("objective", float("nan")))),
                       "Diagnostic C80 candidate grid or objective is malformed")
        seen.add(key)
    expected = {(d, f) for d in p["directions"] for f in p["fractions"]}
    helper.require(seen == expected, "Diagnostic candidate set differs from the frozen six conditions")
    return summary, {(row["direction"], float(row["fraction"])): row for row in conditions}


def validate_warmstart(p, root, helper, ref, scaling, plan_rows, suites,
                       original_mse, source_protocol, source_mse_protocol, diagnostic, new_jobs):
    summary = ref.read_json(root / "results" / ARM / "summary.json")
    receipt = ref.read_json(root / "results" / ARM / "training_receipt.json")
    selection = summary.get("selection")
    helper.require(isinstance(selection, dict), "Warm-start selection is missing")
    job = summary.get("pbs_jobid")
    helper.require(summary.get("complete") is True and summary.get("protocol") == p["protocol"]
                   and summary.get("unit") == ARM and summary.get("arm") == ARM
                   and summary.get("checkpoint_arm") == "mse_nmse" and summary.get("loss") == "dual_kl"
                   and job in new_jobs and summary.get("heldout_used") is False,
                   "Warm-start summary is incomplete or mismatched")
    for key, value in {"source_updates": 80, "gradient_query_budget": 80,
                       "gradient_queries_new": 0, "gradient_queries_reused": 80,
                       "calibration_queries": 80, "heldout_queries": 80,
                       "evaluation_queries": 160, "new_teacher_queries": 0}.items():
        helper.require(summary.get(key) == value, f"Warm-start summary {key} differs from the frozen budget")
    refinement_updates = summary.get("refinement_updates")
    total_updates = summary.get("total_updates")
    helper.require(refinement_updates in (0, 1) and total_updates == 80 + refinement_updates,
                   "Warm-start update count exceeds the one-step budget")
    for key in ("source_parity_passed", "deployment_parity_passed", "frozen_verified",
                "source_original_endpoint_replay"):
        pass_check(summary, key, f"Warm-start {key}", helper)
    helper.require(receipt.get("complete") is True and receipt.get("protocol") == p["protocol"]
                   and receipt.get("unit") == ARM and receipt.get("arm") == ARM
                   and receipt.get("pbs_jobid") == job and receipt.get("checkpoint_arm") == "mse_nmse"
                   and receipt.get("loss") == "dual_kl",
                   "Warm-start training receipt is incomplete or mismatched")
    for key in ("source_updates", "refinement_updates", "total_updates", "gradient_query_budget",
                "gradient_queries_new", "gradient_queries_reused", "calibration_queries",
                "heldout_queries", "evaluation_queries", "new_teacher_queries"):
        helper.require(receipt.get(key) == summary.get(key), f"Warm-start receipt {key} differs from summary")
    for key in ("source_parity_passed", "deployment_parity_passed", "frozen_verified", "frozen_checks"):
        pass_check(receipt, key, f"Warm-start receipt {key}", helper)
    helper.require(ref.read_json(root / "results" / ARM / "selection.json") == selection,
                   "Warm-start saved selection differs from its summary")

    matched_ref = {"root": p["matched_root"], "mse_job_id": p["matched_mse_job"],
                   "dual_job_id": p["matched_dual_job"]}
    helper.require(summary.get("matched_comparison_reference") == matched_ref
                   and receipt.get("matched_comparison_reference") == matched_ref,
                   "Warm-start matched-control provenance differs")
    expected_gradient = {"kind": "completed_diagnostic_full_C80",
                         "path": str(Path(p["diagnostic_root"]) / "results" / "mse_dual" / "gradient_state.pt"),
                         "diagnostic_unit": "mse_dual", "checkpoint_arm": "mse_nmse",
                         "loss": "dual_kl", "diagnostic_pbs_jobid": p["diagnostic_job"]}
    helper.require(summary.get("gradient_source") == expected_gradient
                   and receipt.get("gradient_source") == expected_gradient,
                   "Warm-start gradient source is not the frozen MSE-start diagnostic")

    source_objective = original_mse["calibration"]["stats"]["objective_overall"]["objective_dual_kl"]
    helper.require(close(selection["baseline_objective_mean"], source_objective)
                   and close(diagnostic["baseline_metrics"]["objective"], source_objective),
                   "MSE-checkpoint dual-KL baseline differs from original C80 endpoints/diagnostic")
    coefficients = selection.get("objective_coefficients", {})
    helper.require(selection.get("diagnostic_pbs_jobid") == p["diagnostic_job"]
                   and close(coefficients.get("action_kl", float("nan")), scaling["action_kl_coefficient"])
                   and close(coefficients.get("latent_kl", float("nan")), scaling["latent_kl_coefficient"])
                   and close(coefficients.get("latent_weight", float("nan")), scaling["original_latent_weight"]),
                   "Warm-start selection loss coefficients or diagnostic binding differ")
    diagnostic_rows = validate_selection_candidates(selection, diagnostic, p, helper)
    expected_updates = total_updates
    parsed = {}
    for split in ("calibration", "heldout"):
        data = parse_panel(p, helper, ref, scaling, plan_rows, suites, root, ARM, split, expected_updates,
                           require_update_tags=True)
        source_records = {(r["case_id"], r["seed_index"]): r
                          for r in original_mse[split]["records"]}
        for item in data["records"]:
            case, seed = item["case_id"], item["seed_index"]
            raw = ref.read_json(helper.endpoint_path(root, split, ARM, case, seed))
            original = source_records[(case, seed)]
            helper.require(raw.get("pbs_jobid") in new_jobs and raw.get("source_record_reused") is False
                           and raw.get("checkpoint_arm") == "mse_nmse" and raw.get("loss") == "dual_kl"
                           and raw.get("source_record_protocol") == source_protocol["protocol"]
                           and raw.get("source_record_arm") == "mse_nmse"
                           and raw.get("source_record_pbs_jobid") == original["pbs_jobid"],
                           f"Warm-start {split}/{case}/{seed} checkpoint/PBS provenance differs")
            helper.require(raw.get("source_checkpoint_pbs_jobid") == p["source_mse_job"]
                           and raw.get("source_checkpoint_protocol") == source_mse_protocol["protocol"]
                           and raw.get("diagnostic_protocol") == "fastwam-vq-kl-direction-v1"
                           and raw.get("diagnostic_unit") == "mse_dual"
                           and raw.get("diagnostic_pbs_jobid") == p["diagnostic_job"]
                           and raw.get("diagnostic_condition") == selection["chosen"].get("condition"),
                           f"Warm-start {split}/{case}/{seed} source/diagnostic provenance differs")
            if split == "calibration":
                helper.require(raw.get("diagnostic_replay_verified") is True,
                               f"Warm-start selected C80 endpoint lacks diagnostic replay check")
                if selection["chosen"].get("condition") == "no_step":
                    helper.require(close(item["metrics"]["first10motor_mse"], original["metrics"]["first10motor_mse"])
                                   and close(item["metrics"]["future_latent_nmse"], original["metrics"]["future_latent_nmse"])
                                   and close(item["metrics"]["action_kl"], original["metrics"]["action_kl"])
                                   and close(item["metrics"]["latent_kl"], original["metrics"]["latent_kl"])
                                   and close(item["metrics"]["objective_loss"], original["objectives"]["dual_kl"]),
                                   f"Warm-start no-step C80 losses differ from source/diagnostic baseline for {case}")
                else:
                    chosen = selection["chosen"]
                    direction = chosen["direction"]
                    fraction = float(chosen["fraction"])
                    label = direction + "_f" + str(fraction).replace(".", "p")
                    diag_case = ref.read_json(Path(p["diagnostic_root"]) / "results" / "mse_dual"
                                              / "directions" / label / f"{case}_0.json")
                    helper.require(diag_case.get("protocol") == "fastwam-vq-kl-direction-v1"
                                   and diag_case.get("unit") == "mse_dual"
                                   and diag_case.get("direction") == direction
                                   and float(diag_case.get("fraction")) == fraction
                                   and diag_case.get("case_id") == case,
                                   f"Warm-start C80 selected diagnostic binding differs for {case}")
                    for metric, diag_metric in (("first10motor_mse", "action_mse"),
                                                ("future_latent_nmse", "latent_nmse"),
                                                ("action_kl", "action_kl"), ("latent_kl", "latent_kl")):
                        helper.require(close(item["metrics"][metric], diag_case[diag_metric]),
                                       f"Warm-start selected C80 {metric} differs from diagnostic for {case}")
                    helper.require(close(item["metrics"]["objective_loss"], diag_case["objective"]),
                                   f"Warm-start selected C80 objective differs from diagnostic for {case}")
        assert_shared_teacher(data, original_mse[split], helper, f"warmstart/{split}")
        parsed[split] = data
    selected_mean = parsed["calibration"]["stats"]["objective_overall"]["objective_dual_kl"]
    helper.require(close(selected_mean, selection["chosen"]["objective_mean"]),
                   "Warm-start final C80 mean differs from its selected diagnostic condition")
    helper.require((refinement_updates == 0) == (selection["chosen"].get("condition") == "no_step"),
                   "Warm-start update count differs from the selected no-step/candidate condition")
    return summary, receipt, selection, parsed


def validate_selection_candidates(selection, diagnostic, p, helper):
    rows = {(row["direction"], float(row["fraction"])): row for row in diagnostic["direction_conditions"]}
    require = helper.require
    require(selection.get("complete") is True and selection.get("heldout_used") is False
            and selection.get("sampling_scope") == "C80"
            and selection.get("method") == "reused_mse_dual_full_C80_actual_loss_selection"
            and selection.get("selected_from") == "baseline_plus_6_diagnostic_candidates"
            and selection.get("checkpoint_arm") == "mse_nmse" and selection.get("loss") == "dual_kl"
            and selection.get("diagnostic_unit") == "mse_dual"
            and selection.get("diagnostic_pbs_jobid") == p["diagnostic_job"],
            "Warm-start selection is not the frozen C80 diagnostic-only selection")
    conditions = selection.get("conditions")
    require(isinstance(conditions, list) and len(conditions) == 7
            and selection.get("candidate_count") == 6 and selection.get("condition_count") == 7,
            "Warm-start selection must cover no-step plus all six diagnostic candidates")
    require(conditions[0].get("condition") == "no_step"
            and conditions[0].get("source") == "reused_source_C80_baseline"
            and conditions[0].get("queries") == 80
            and conditions[0].get("diagnostic_pbs_jobid") == p["diagnostic_job"]
            and close(conditions[0].get("objective_mean"), selection["baseline_objective_mean"]),
            "Warm-start selection baseline row differs")
    condition_rows = {(row.get("direction"), float(row.get("fraction", -1))): row
                      for row in conditions[1:]}
    candidate_rows = {(row["direction"], float(row["fraction"])): row
                      for row in selection["candidates"]}
    require(len(condition_rows) == 6 and set(condition_rows) == set(candidate_rows),
            "Warm-start seven-condition table differs from its six candidates")
    for row in selection["candidates"]:
        key = (row["direction"], float(row["fraction"]))
        require(key in rows and row.get("optimizer") == key[0] and row.get("queries") == 80
                and row.get("source") == "reused_complete_diagnostic_C80"
                and row.get("diagnostic_pbs_jobid") == p["diagnostic_job"]
                and row.get("refinement_updates") == 1
                and close(row["objective_mean"], rows[key]["objective"])
                and close(row.get("diagnostic_mean", row["objective_mean"]), rows[key]["objective"]),
                "Warm-start C80 candidate objective differs from its original diagnostic")
        require(close(condition_rows[key]["objective_mean"], row["objective_mean"]),
                "Warm-start condition table objective differs from the candidate list")
    best = min(conditions, key=lambda row: (float(row["objective_mean"]), row["condition"]))
    require(selection["chosen"].get("condition") == best["condition"]
            and close(selection["chosen"]["objective_mean"], best["objective_mean"]),
            "Warm-start choice is not the minimum C80 dual-KL mean")
    return rows


def markdown_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in rows)


def fmt(value):
    return f"{float(value):.8g}"


def make_report(p, suites, stats, paired, selection, decision):
    held = stats["heldout"]
    panel_rows = [[panel, *[fmt(held[panel]["stats"]["overall"][metric]) for metric in METRICS]]
                  for panel in PANELS]
    contrast_order = ("warmstart_dual_minus_matched_mse", "warmstart_dual_minus_original_mse",
                      "warmstart_dual_minus_previous_dual", "previous_dual_minus_matched_mse",
                      "matched_mse_minus_original_mse")
    contrast_rows = []
    for contrast in contrast_order:
        for metric in ("first10motor_rmse", "future_latent_nmse", "action_kl", "latent_kl"):
            item = paired[contrast][metric]
            contrast_rows.append([contrast, metric, fmt(item["mean_delta_equal_suite"]),
                                  fmt(item["median_delta"]),
                                  f"{item['improved']}/{item['worsened']}/{item['tied']}",
                                  ", ".join(f"{suite}: {fmt(item['by_suite_mean_delta'][suite])}" for suite in suites),
                                  "[" + ", ".join(fmt(v) for v in item["equal_suite_leave_one_case_out_mean_range"]) + "]"])
    objective_rows = [[name,
                       fmt(paired["warmstart_dual_minus_matched_mse"]["objective_" + name]["mean_delta_equal_suite"]),
                       fmt(paired["warmstart_dual_minus_matched_mse"]["objective_" + name]["median_delta"])]
                      for name in ("mse_nmse", "action_kl", "latent_kl", "dual_kl")]
    select_rows = [[row["condition"], row["direction"], row["fraction"], row["objective_mean"]]
                   for row in selection["conditions"]]
    return "\n".join([
        "# Fast-WAM MSE-start dual-KL warm-start aggregate", "",
        "Held-out 描述以 40 个 case 为单位：先在每个 case 内平均两个 seed，再对四个 suite 等权。四个条件共用冻结 cohort 与 teacher；每个 arm 只有一条训练轨迹，因此 bootstrap 仅描述固定输入 case 波动，不代表训练重复，不作显著性结论。", "",
        "证据范围是 decoded BF16 compensated VQ、dynamic fakeA4 与 BF16 KV 的 fixed-input 结果；没有 rollout、closed-loop 或 native-kernel speed 证据。Gripper 不作主判断。", "",
        "## 四个 checkpoint 的 held-out 描述统计", "",
        markdown_table(["条件", *METRICS], panel_rows), "",
        "## 配对对比", "",
        "Δ 定义为对比名称左侧条件减右侧条件；负值表示左侧指标较低。改善/变差/相同按 40 个 case 计数，LOCO 仍对四个 suite 等权；bootstrap 为固定 seed 的描述性 case 重抽样。主对照是 warm-start dual-KL 对 matched MSE-refined；同时单独报告相对 original MSE。", "",
        markdown_table(["对比", "指标", "等 suite mean Δ", "case median Δ", "改善/变差/相同", "suite mean Δ", "等 suite LOCO range"], contrast_rows), "",
        "四种固定 loss 公式下的 objective 配对变化（每列均使用同一公式计算，不能横向比较不同公式的绝对值）：", "",
        markdown_table(["公式", "warm-start dual − matched MSE mean Δ", "case median Δ"], objective_rows), "",
        "## C80 选优与预设 screen", "",
        f"选择只使用 C80 的 dual-KL objective；基线加六个原 diagnostic 候选共七个条件。选中 `{selection['chosen']['condition']}`，refinement updates={decision['refinement_updates']}。", "",
        markdown_table(["条件", "方向", "fraction", "C80 dual-KL objective mean"], select_rows), "",
        f"主 screen（warm-start dual-KL vs matched MSE-refined）：{decision['primary_screen']['passes']}；是否超过 original MSE（motor RMSE 与 latent NMSE 的等-suite mean 均较低）：{decision['vs_original_mse']['beats_both_primary_means']}。原筛选阈值仍是 motor RMSE case median Δ<0 且至少 24/40 case 改善。", "",
        f"Raw action KL Δ vs matched MSE={fmt(paired['warmstart_dual_minus_matched_mse']['action_kl']['mean_delta_equal_suite'])}; raw latent KL Δ={fmt(paired['warmstart_dual_minus_matched_mse']['latent_kl']['mean_delta_equal_suite'])}. 仅在两项都下降时标记为 both-information-loss improvement：{decision['both_raw_kl_terms_lower_vs_matched_mse']}。", "",
        "该结果是固定 checkpoint 的 fixed-input engineering screen；不支持成功率、closed-loop、native latency 或独立训练复现结论。", ""
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    helper = load_sibling("refinement_analysis.py")
    ref = helper.load_reference()
    job_id = ref.guard()
    p = ref.read_json(args.protocol)
    validate_protocol(p, helper)
    root = Path(p["remote_root"])
    helper.require(p.get("remote_root") == str(root), "Warm-start output root differs")
    source_protocol, scaling = validate_loss_scaling(p, helper)
    source_mse_protocol = validate_source_training(p, helper)
    plan_rows, suites = collect_plan(p, helper, ref)
    diag_summary, _ = validate_diagnostic(p, helper)

    # Original MSE endpoints are from the completed dual-KL source panel; training provenance is checked above.
    source_records = {}
    for split in ("calibration", "heldout"):
        source_records[split] = parse_panel(p, helper, ref, scaling, plan_rows, suites, Path(p["source_root"]),
                                            "mse_nmse", split, 80,
                                            source_protocol=source_protocol["protocol"])
    original_mse = source_records
    original_mse["protocol"] = source_protocol["protocol"]

    matched, matched_protocol = validate_matched(p, helper, ref, scaling, plan_rows, suites)
    for split in ("calibration", "heldout"):
        for arm in ("mse_nmse", "dual_kl"):
            assert_shared_teacher(matched[arm][split], original_mse[split], helper,
                                  f"matched/{arm}/{split}")
    new_jobs = helper.pbs_jobs_in_handles(root)
    summary, receipt, selection, warmstart = validate_warmstart(
        p, root, helper, ref, scaling, plan_rows, suites, original_mse,
        source_protocol, source_mse_protocol, diag_summary, new_jobs)

    # Keep one equal-suite summary per checkpoint and split.
    stats = {split: {
        "original_mse": original_mse[split],
        "matched_mse_refined": matched["mse_nmse"][split],
        "previous_dual_refined": matched["dual_kl"][split],
        "warmstart_dual": warmstart[split],
    } for split in ("calibration", "heldout")}
    contrasts = {
        "warmstart_dual_minus_matched_mse": ("matched_mse_refined", "warmstart_dual"),
        "warmstart_dual_minus_original_mse": ("original_mse", "warmstart_dual"),
        "warmstart_dual_minus_previous_dual": ("previous_dual_refined", "warmstart_dual"),
        "previous_dual_minus_matched_mse": ("matched_mse_refined", "previous_dual_refined"),
        "matched_mse_minus_original_mse": ("original_mse", "matched_mse_refined"),
    }
    paired = {name: helper.pair_summary(stats["heldout"][left]["stats"],
                                        stats["heldout"][right]["stats"], suites, ref, 20261020 + i * 1000)
              for i, (name, (left, right)) in enumerate(contrasts.items())}
    primary = paired["warmstart_dual_minus_matched_mse"]
    original_pair = paired["warmstart_dual_minus_original_mse"]
    core_checks = {
        "motor_rmse_lower_equal_suite_mean": stats["heldout"]["warmstart_dual"]["stats"]["overall"]["first10motor_rmse"]
                                                < stats["heldout"]["matched_mse_refined"]["stats"]["overall"]["first10motor_rmse"],
        "future_latent_nmse_lower_equal_suite_mean": stats["heldout"]["warmstart_dual"]["stats"]["overall"]["future_latent_nmse"]
                                                        < stats["heldout"]["matched_mse_refined"]["stats"]["overall"]["future_latent_nmse"],
        "motor_case_median_delta_negative": primary["first10motor_rmse"]["median_delta"] < 0,
        "motor_improved_cases_at_least_24": primary["first10motor_rmse"]["improved"] >= 24,
        "refinement_performed": summary["refinement_updates"] == 1,
    }
    beats_original = {
        "motor_rmse_lower_equal_suite_mean": stats["heldout"]["warmstart_dual"]["stats"]["overall"]["first10motor_rmse"]
                                                < stats["heldout"]["original_mse"]["stats"]["overall"]["first10motor_rmse"],
        "future_latent_nmse_lower_equal_suite_mean": stats["heldout"]["warmstart_dual"]["stats"]["overall"]["future_latent_nmse"]
                                                        < stats["heldout"]["original_mse"]["stats"]["overall"]["future_latent_nmse"],
        "motor_case_median_delta_negative": original_pair["first10motor_rmse"]["median_delta"] < 0,
        "motor_improved_cases_at_least_24": original_pair["first10motor_rmse"]["improved"] >= 24,
    }
    raw_dual = stats["heldout"]["warmstart_dual"]["stats"]["overall"]
    raw_mse = stats["heldout"]["matched_mse_refined"]["stats"]["overall"]
    both_kl_lower = raw_dual["action_kl"] < raw_mse["action_kl"] and raw_dual["latent_kl"] < raw_mse["latent_kl"]
    decision = {
        "protocol": p["protocol"], "complete": True, "pbs_jobid": job_id,
        "primary_comparison": "warmstart dual_kl from original mse_nmse checkpoint minus matched mse_nmse refined",
        "selection_acceptance": "passed_C80_only_minimum_of_original_mse_checkpoint_dual_kl_objective",
        "refinement_updates": summary["refinement_updates"], "total_updates": summary["total_updates"],
        "primary_screen": {"checks": core_checks, "passes": all(core_checks.values()),
                           "median_delta": primary["first10motor_rmse"]["median_delta"],
                           "improved_cases": primary["first10motor_rmse"]["improved"],
                           "required_improved_cases": 24},
        "vs_original_mse": {"checks": beats_original,
                            "beats_both_primary_means": beats_original["motor_rmse_lower_equal_suite_mean"]
                                                      and beats_original["future_latent_nmse_lower_equal_suite_mean"],
                            "motor_median_delta": original_pair["first10motor_rmse"]["median_delta"],
                            "motor_improved_cases": original_pair["first10motor_rmse"]["improved"]},
        "both_raw_kl_terms_lower_vs_matched_mse": both_kl_lower,
        "raw_kl_mean_deltas_vs_matched_mse": {
            "action_kl": primary["action_kl"]["mean_delta_equal_suite"],
            "latent_kl": primary["latent_kl"]["mean_delta_equal_suite"]},
        "unit_of_analysis": "40 held-out cases after two-seed mean within case; four suites equally weighted",
        "statistical_significance": False, "independent_training_replicates": False,
        "rollouts": 0, "closed_loop_evaluated": False, "native_kernel_speed_evaluated": False,
    }

    out_stats = {}
    for split in ("calibration", "heldout"):
        out_stats[split] = {panel: {"queries": item["stats"]["queries"], "cases": item["stats"]["cases"],
                                   "overall": item["stats"]["overall"],
                                   "by_suite": item["stats"]["by_suite"],
                                   "objective_overall": item["stats"].get("objective_overall", {}),
                                   "objective_by_suite": item["stats"].get("objective_by_suite", {})}
                           for panel, item in stats[split].items()}
    paired_brief = {contrast: {metric: {k: v for k, v in item.items() if k != "case_deltas"}
                               for metric, item in values.items()} for contrast, values in paired.items()}
    comparison = {
        "protocol": p["protocol"], "complete": True, "pbs_jobid": job_id,
        "calibration_records": 80, "heldout_records": 80, "new_teacher_queries": 0,
        "arm": ARM, "arms": [ARM], "checkpoint_arm": "mse_nmse",
        "source_updates": 80, "refinement_updates": summary["refinement_updates"],
        "total_updates": summary["total_updates"], "gradient_queries_new": 0,
        "gradient_queries_reused": 80, "new_model_queries": summary.get("new_model_queries"),
        "forward_accounting": summary.get("forward_accounting"),
        "suite_weighting": "seed mean within case, case mean within suite, four suites equal",
        "loss_scaling": {k: scaling[k] for k in ("action_kl_coefficient", "latent_kl_coefficient",
                                                   "original_latent_weight", "heldout_used")},
        "objective_formulae": {
            "mse_nmse": "first10motor_mse + original_latent_weight * future_latent_nmse",
            "action_kl": "action_kl_coefficient * action_kl + original_latent_weight * future_latent_nmse",
            "latent_kl": "first10motor_mse + original_latent_weight * latent_kl_coefficient * latent_kl",
            "dual_kl": "action_kl_coefficient * action_kl + original_latent_weight * latent_kl_coefficient * latent_kl"},
        "diagnostic": {"pbs_jobid": p["diagnostic_job"], "unit": "mse_dual", "gradient_queries_reused": 80,
                       "selected_condition": selection["chosen"]["condition"]},
        "matched_jobs": {"mse_nmse": p["matched_mse_job"], "dual_kl": p["matched_dual_job"]},
        "splits": out_stats, "paired_contrasts": paired_brief, "decision": decision,
        "evidence_scope": p["evidence_scope"],
    }
    helper.save_json(root / "results" / "comparison.json", comparison, MAX_JSON_BYTES)
    paired_details = {
        "protocol": p["protocol"], "complete": True, "pbs_jobid": job_id,
        "case_unit": "40 held-out cases after seed mean; 80 query endpoints are not 80 cases",
        "paired_case_deltas": {contrast: {metric: item["case_deltas"] for metric, item in values.items()}
                                for contrast, values in paired.items()},
        "case_metrics": {case: {panel: {metric: stats["heldout"][panel]["stats"]["case_values"][case][metric]
                                         for metric in helper.PAIRED_METRICS}
                                    for panel in PANELS}
                         for case in p["heldout_case_ids"]},
        "equal_suite_loco_ranges": {contrast: {metric: item["equal_suite_leave_one_case_out_mean_range"]
                                               for metric, item in values.items()}
                                    for contrast, values in paired.items()},
    }
    helper.save_json(root / "results" / "paired_details.json", paired_details, MAX_JSON_BYTES)
    helper.save_json(root / "results" / "decision.json", decision)
    aggregate = {
        "protocol": p["protocol"], "complete": True, "pbs_jobid": job_id,
        "calibration_records": 80, "heldout_records": 80, "new_teacher_queries": 0,
        "arm": ARM, "arms": [ARM], "checkpoint_arm": "mse_nmse",
        "source_updates": 80, "refinement_updates": summary["refinement_updates"],
        "total_updates": summary["total_updates"], "gradient_queries_new": 0,
        "gradient_queries_reused": 80, "new_gradient_queries": 0, "reused_gradient_queries": 80,
        "gradient_query_budget": 80, "evaluation_queries": 160,
        "new_model_queries": summary.get("new_model_queries"),
        "forward_accounting": summary.get("forward_accounting"),
        "selection_condition": selection["chosen"]["condition"], "suites": suites,
    }
    helper.save_json(root / "results" / "aggregate" / "summary.json", aggregate)
    (root / "results" / "ANALYSIS.zh.md").write_text(
        make_report(p, suites, stats, paired, selection, decision), encoding="utf-8")
    print("KL_MSE_WARMSTART_AGGREGATE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
