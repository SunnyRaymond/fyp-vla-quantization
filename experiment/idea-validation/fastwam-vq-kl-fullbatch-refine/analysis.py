"""Aggregate the frozen full-C80 KL refinement endpoints on a PBS CPU node."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics


ARMS = ("mse_nmse", "dual_kl")
CONDITIONS = ("first10motor_mse", "first10motor_rmse", "full32motor_rmse",
              "late22motor_rmse", "future_latent_nmse", "action_kl", "latent_kl")
OBJECTIVES = ("mse_nmse", "action_kl", "latent_kl", "dual_kl")
PAIRED_METRICS = (*CONDITIONS, *("objective_" + name for name in OBJECTIVES))
MAX_JSON_BYTES = 200 * 1024
TOL = {"rel_tol": 1e-5, "abs_tol": 1e-8}


def load_reference():
    path = Path(__file__).with_name("comparison_reference.py")
    spec = importlib.util.spec_from_file_location("comparison_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("comparison_reference.py is required beside analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("Required artifact is missing: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value, limit=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    if limit is not None and len(data.encode("utf-8")) >= limit:
        raise RuntimeError(f"{path.name} exceeds the {limit}-byte report limit")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(a, b):
    return math.isclose(float(a), float(b), **TOL)


def passed(value):
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() in ("pass", "passed", "complete", "ok")
    if isinstance(value, dict):
        for key in ("passed", "pass", "complete"):
            if key in value:
                return passed(value[key])
        for key in ("status", "result"):
            if key in value:
                return passed(value[key])
    return False


def required_pass(container, key, label):
    require(key in container and passed(container[key]), f"{label} did not explicitly pass")


def metric_map(record, ref, weight, alpha_action, alpha_latent):
    values = record.get("metrics", record)
    require(isinstance(values, dict), "Endpoint metrics object is missing")
    base = {name: ref.finite_metric(values, name) for name in ref.METRICS
            if name != "mixed_loss"}
    # Keep this generic MSE+NMSE reference term available even when workers omit it.
    base["mixed_loss"] = base["first10motor_mse"] + weight * base["future_latent_nmse"]
    formulas = {
        "mse_nmse": base["first10motor_mse"] + weight * base["future_latent_nmse"],
        "action_kl": alpha_action * base["action_kl"] + weight * base["future_latent_nmse"],
        "latent_kl": base["first10motor_mse"] + weight * alpha_latent * base["latent_kl"],
        "dual_kl": alpha_action * base["action_kl"] + weight * alpha_latent * base["latent_kl"],
    }
    base["objective_loss"] = ref.finite_metric(values, "objective_loss")
    return base, formulas


def query_key(case, seed):
    return f"{case}_{seed}.json"


def endpoint_path(root, split, arm, case, seed):
    return Path(root) / "results" / split / arm / "queries" / query_key(case, seed)


def validate_binding(record, protocol, arm, split, case, seed, plan_row, ref,
                     source_protocol=None, updates=None):
    expected_protocol = source_protocol or protocol["protocol"]
    require(record.get("protocol") == expected_protocol and record.get("arm") == arm
            and record.get("split") == split and record.get("case_id") == case
            and record.get("seed_index") == seed and record.get("plan_row") == plan_row,
            f"{arm}/{split}/{case}/{seed} binding differs from the frozen source plan")
    require(record.get("suite") == plan_row["suite"]
            and record.get("sampler_seed") == plan_row["sampler_seeds"][seed],
            f"{arm}/{split}/{case}/{seed} suite or sampler seed differs")
    if updates is not None:
        require(record.get("updates") == updates, f"Source endpoint {arm}/{case}/{seed} update count differs")
    require(bool(record.get("pbs_jobid")), f"{arm}/{case}/{seed} endpoint lacks pbs_jobid")
    return record


def parse_record(record, protocol, arm, split, case, seed, plan_row, ref, scaling,
                 expected_updates=None, source_protocol=None):
    validate_binding(record, protocol, arm, split, case, seed, plan_row, ref,
                     source_protocol=source_protocol, updates=expected_updates)
    values, formulas = metric_map(record, ref, scaling["original_latent_weight"],
                                  scaling["action_kl_coefficient"],
                                  scaling["latent_kl_coefficient"])
    own = formulas[arm]
    require(close(values["objective_loss"], own),
            f"{arm}/{split}/{case}/{seed} objective_loss disagrees with its fixed formula")
    action = ref.matrix(record.get("action"), "action")
    teacher_action = ref.matrix(record.get("teacher_action"), "teacher_action")
    commands = ref.matrix(record.get("commands"), "commands")
    teacher_commands = ref.matrix(record.get("teacher_commands"), "teacher_commands")
    action_bias, action_dims = ref.motor_abs_chunk_bias(action, teacher_action)
    command_bias, command_dims = ref.motor_abs_chunk_bias(commands, teacher_commands)
    return {
        "case_id": case, "seed_index": seed, "suite": plan_row["suite"],
        "pbs_jobid": record["pbs_jobid"], "metrics": values, "objectives": formulas,
        "action_motor_abs_chunk_bias": action_bias,
        "action_motor_abs_chunk_bias_by_dimension": action_dims,
        "command_motor_abs_chunk_bias": command_bias,
        "command_motor_abs_chunk_bias_by_dimension": command_dims,
        "source_record_reused": record.get("source_record_reused") is True,
        "raw": {"action": action, "teacher_action": teacher_action,
                "commands": commands, "teacher_commands": teacher_commands,
                "protocol": record["protocol"]},
    }


def source_context(protocol, ref, plan_rows, suites, scaling):
    source_root = Path(protocol["source_root"])
    source_mse_root = Path(protocol["source_mse_root"])
    diagnostic_root = Path(protocol["diagnostic_root"])
    old_protocol = read_json(source_root / "protocol.json")
    require(old_protocol.get("remote_root") == str(source_root), "Source protocol root differs")
    require(old_protocol.get("source_per_channel_root") == str(source_mse_root),
            "Source MSE checkpoint root differs from the frozen source protocol")
    source_scaling = read_json(source_root / "results" / "prepare" / "loss_scaling.json")
    require(source_scaling.get("complete") is True
            and source_scaling.get("protocol") == old_protocol["protocol"]
            and source_scaling.get("heldout_used") is False
            and source_scaling.get("case_ids") == protocol["calibration_case_ids"]
            and source_scaling.get("seed_indices") == [0]
            and len(source_scaling.get("initial_rows", [])) == 80,
            "The original loss scaling is not complete C80-only calibration")
    training = old_protocol.get("training", {})
    require(close(source_scaling["original_latent_weight"],
                  training.get("rho", float("nan")) * training.get("normalization_c", float("nan"))),
            "Frozen original latent weight differs from the source training protocol")
    initial = source_scaling.get("initial_means", {})
    require(close(source_scaling["action_kl_coefficient"], initial["action_mse"] / initial["action_kl"])
            and close(source_scaling["latent_kl_coefficient"], initial["latent_nmse"] / initial["latent_kl"]),
            "Frozen KL coefficients differ from the original C80 loss-scaling rule")
    for key in ("action_kl_coefficient", "latent_kl_coefficient", "original_latent_weight"):
        require(close(source_scaling[key], scaling[key]), f"Frozen loss-scaling field changed: {key}")

    mse_protocol = read_json(source_mse_root / "protocol.json")
    require(mse_protocol.get("remote_root") == str(source_mse_root),
            "Original MSE/NMSE source protocol root differs")
    mse_receipt = read_json(source_mse_root / "results" / "train_eval" / "training_receipt.json")
    mse_summary = read_json(source_mse_root / "results" / "train_eval" / "summary.json")
    require(mse_receipt.get("complete") is True and mse_receipt.get("updates") == 80
            and mse_receipt.get("protocol") == mse_protocol["protocol"]
            and mse_receipt.get("pbs_jobid") == protocol["source_mse_job"],
            "Original MSE/NMSE source is not a complete final-80 training receipt")
    require(mse_summary.get("complete") is True and mse_summary.get("updates") == 80
            and mse_summary.get("unit") == "train_eval"
            and mse_summary.get("protocol") == mse_protocol["protocol"]
            and mse_summary.get("pbs_jobid") == protocol["source_mse_job"]
            and mse_summary.get("new_teacher_queries") == 0,
            "Original MSE/NMSE source summary is incomplete")
    dual_summary = read_json(source_root / "results" / "dual_kl" / "summary.json")
    dual_receipt = dual_summary.get("training_receipt", {})
    require(dual_summary.get("complete") is True and dual_summary.get("updates") == 80
            and dual_summary.get("new_teacher_queries") == 0
            and dual_receipt.get("updates") == 80 and dual_receipt.get("complete") is True,
            "Original dual-KL source is not a complete final-80 run")
    prepare_summary = read_json(source_root / "results" / "prepare" / "summary.json")
    require(prepare_summary.get("complete") is True
            and prepare_summary.get("protocol") == old_protocol["protocol"]
            and prepare_summary.get("initial_calibration_queries") == 80
            and prepare_summary.get("control_evaluation_queries") == 160
            and prepare_summary.get("new_teacher_queries") == 0,
            "Original source control replay is incomplete")
    diag_protocol = read_json(diagnostic_root / "protocol.json")
    diag_summary = read_json(diagnostic_root / "results" / "dual_dual" / "summary.json")
    diag_directions = read_json(diagnostic_root / "results" / "dual_dual" / "direction_summary.json")
    diag_aggregate = read_json(diagnostic_root / "results" / "aggregate" / "summary.json")
    require(diag_protocol.get("remote_root") == str(diagnostic_root)
            and diag_protocol.get("protocol") == "fastwam-vq-kl-direction-v1",
            "Dual-KL direction diagnostic protocol differs")
    require(diag_summary.get("complete") is True and diag_summary.get("unit") == "dual_dual"
            and diag_summary.get("heldout_used") is False
            and diag_summary.get("gradient_query_count") == 80
            and diag_summary.get("training_updates") == 0
            and diag_summary.get("pbs_jobid"),
            "Reusable dual-KL C80 diagnostic summary is incomplete")
    require(isinstance(diag_directions, list) and len(diag_directions) == 6
            and diag_summary.get("direction_conditions") == diag_directions,
            "Reusable dual-KL diagnostic lacks its six frozen C80 candidates")
    require(diag_aggregate.get("complete") is True
            and diag_aggregate.get("parity_cases") == 320
            and diag_aggregate.get("gradient_queries") == 320
            and diag_aggregate.get("direction_conditions") == 24
            and diag_aggregate.get("training_updates") == 0,
            "Full direction diagnostic aggregate is incomplete")

    original, source_jobs = {}, set()
    for split in ("calibration", "heldout"):
        original[split] = {}
        cases = protocol["calibration_case_ids"] if split == "calibration" else protocol["heldout_case_ids"]
        seeds = [0] if split == "calibration" else protocol["heldout_seed_indices"]
        for arm in ARMS:
            records = []
            for case in cases:
                for seed in seeds:
                    record = read_json(endpoint_path(source_root, split, arm, case, seed))
                    row = plan_rows[case]
                    parsed = parse_record(record, protocol, arm, split, case, seed, row, ref,
                                          scaling, expected_updates=80,
                                          source_protocol=old_protocol["protocol"])
                    source_jobs.add(parsed["pbs_jobid"])
                    records.append(parsed)
            stats = aggregate_full(records, suites, ref)
            original[split][arm] = {"records": records, "stats": stats}
    return original, source_jobs, diag_summary


def collect_plan(protocol, ref):
    source_root = Path(protocol["source_root"])
    cases = protocol["calibration_case_ids"] + protocol["heldout_case_ids"]
    require(len(protocol["calibration_case_ids"]) == 80
            and protocol["calibration_case_ids"] == list(range(2000, 2080)),
            "Calibration cohort must be the frozen C80 set")
    require(len(protocol["heldout_case_ids"]) == 40
            and protocol["heldout_case_ids"] == list(range(3000, 3040))
            and protocol["heldout_seed_indices"] == [0, 1],
            "Held-out cohort or seed panel differs from the frozen plan")
    plan_rows, suites = {}, []
    for split, split_cases, seeds in (("calibration", protocol["calibration_case_ids"], [0]),
                                      ("heldout", protocol["heldout_case_ids"], [0, 1])):
        for case in split_cases:
            rows = []
            for seed in seeds:
                record = read_json(endpoint_path(source_root, split, "mse_nmse", case, seed))
                row = record.get("plan_row")
                require(isinstance(row, dict) and row.get("case_id") == case
                        and isinstance(row.get("sampler_seeds"), list)
                        and len(row["sampler_seeds"]) > max(seeds),
                        f"Source control query has no complete frozen plan row for {case}")
                require(record.get("arm") == "mse_nmse" and record.get("split") == split
                        and record.get("case_id") == case and record.get("seed_index") == seed
                        and record.get("plan_row") == row
                        and record.get("sampler_seed") == row["sampler_seeds"][seed],
                        f"Source control binding differs for {split}/{case}/{seed}")
                rows.append(row)
            require(all(row == rows[0] for row in rows), f"Frozen plan row changed across seeds for {case}")
            plan_rows[case] = rows[0]
            if rows[0]["suite"] not in suites:
                suites.append(rows[0]["suite"])
    require(len(cases) == len(set(cases)) and len(plan_rows) == 120 and len(suites) == 4,
            "Frozen panel must contain 120 unique cases across four suites")
    for suite in suites:
        require(sum(plan_rows[c]["suite"] == suite for c in protocol["calibration_case_ids"]) == 20
                and sum(plan_rows[c]["suite"] == suite for c in protocol["heldout_case_ids"]) == 10,
                f"Frozen suite balance differs for {suite}")
    return plan_rows, suites


def validate_selection(summary, arm, protocol):
    selection = summary.get("selection")
    require(isinstance(selection, dict) and selection.get("complete") is True
            and selection.get("heldout_used") is False
            and selection.get("sampling_scope") == "C80"
            and selection.get("method") == "full_C80_gradient_actual_loss_selection",
            f"{arm} selection is incomplete, not C80-only, or has another method")
    baseline = selection.get("baseline_objective_mean")
    require(isinstance(baseline, (int, float)) and math.isfinite(float(baseline)) and baseline >= 0,
            f"{arm} selection lacks the C80 baseline objective")
    candidates = selection.get("candidates")
    require(isinstance(candidates, list) and len(candidates) == 6,
            f"{arm} selection must contain exactly six Adam candidates")
    seen = set()
    for row in candidates:
        require(isinstance(row, dict), f"{arm} candidate row is malformed")
        direction, fraction = row.get("direction"), row.get("fraction")
        require(direction in protocol["directions"] and fraction in protocol["fractions"],
                f"{arm} candidate direction or fraction is outside the frozen grid")
        key = (direction, float(fraction))
        require(key not in seen, f"{arm} selection repeats a candidate")
        seen.add(key)
        value = row.get("objective_mean")
        require(isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0,
                f"{arm} candidate lacks a finite C80 objective")
        require(row.get("optimizer") == direction,
                f"{arm} candidate optimizer family differs from its direction")
        expected_source = "new_actual_loss_C80" if arm == "mse_nmse" else "reused_complete_diagnostic_C80"
        require(row.get("source") == expected_source and row.get("queries") == 80,
                f"{arm} candidate C80 query source or budget differs")
    expected = {(d, float(f)) for d in protocol["directions"] for f in protocol["fractions"]}
    require(seen == expected, f"{arm} candidate grid differs from the frozen six conditions")
    chosen = selection.get("chosen")
    require(isinstance(chosen, dict), f"{arm} chosen condition is missing")
    chosen_value = chosen.get("objective_mean")
    require(isinstance(chosen_value, (int, float)) and math.isfinite(float(chosen_value)),
            f"{arm} chosen condition lacks a C80 objective")
    all_values = [float(baseline)] + [float(row["objective_mean"]) for row in candidates]
    require(float(chosen_value) <= min(all_values) + max(1e-12, abs(min(all_values)) * 1e-8),
            f"{arm} chosen condition is not a C80 own-objective minimum")
    condition = chosen.get("condition")
    if condition == "no_step":
        require(close(chosen_value, baseline), f"{arm} no_step objective differs from baseline")
    else:
        direction, fraction = chosen.get("direction"), chosen.get("fraction")
        match = [r for r in candidates if r["direction"] == direction
                 and close(r["fraction"], fraction) and close(r["objective_mean"], chosen_value)]
        require(len(match) == 1, f"{arm} chosen candidate does not match the frozen C80 grid")
    return selection


def pbs_jobs_in_handles(root):
    path = Path(root) / "control" / "handles.json"
    payload = read_json(path)
    jobs = set()
    pattern = re.compile(r"\b\d+\.[A-Za-z0-9_.-]+\b")
    def visit(value):
        if isinstance(value, str):
            jobs.update(pattern.findall(value))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(payload)
    return jobs


def validate_arm_runs(protocol, root, ref, plan_rows, known_new_jobs):
    summaries, receipts, parsed = {}, {}, {s: {} for s in ("calibration", "heldout")}
    expected_counts = {"source_updates": 80, "gradient_query_budget": 80,
                       "calibration_queries": 80, "heldout_queries": 80,
                       "evaluation_queries": 160, "new_teacher_queries": 0}
    scaling = read_json(Path(protocol["source_root"]) / "results" / "prepare" / "loss_scaling.json")
    source_ref, source_jobs, diag_summary = source_context(protocol, ref, plan_rows,
                                                            suites_from_plan(protocol, plan_rows), scaling)
    known_jobs = set(known_new_jobs) | source_jobs
    for arm in ARMS:
        summary = read_json(root / "results" / arm / "summary.json")
        require(summary.get("complete") is True and summary.get("protocol") == protocol["protocol"]
                and summary.get("unit") == arm,
                f"{arm} summary is incomplete or mismatched")
        selection = validate_selection(summary, arm, protocol)
        source_baseline = source_ref["calibration"][arm]["stats"]["objective_overall"]["objective_" + arm]
        require(close(selection["baseline_objective_mean"], source_baseline),
                f"{arm} C80 selection baseline differs from original final-80 endpoints")
        for key, value in expected_counts.items():
            require(summary.get(key) == value, f"{arm} summary {key} differs from the frozen budget")
        expected_new, expected_reused = (80, 0) if arm == "mse_nmse" else (0, 80)
        require(summary.get("gradient_queries_new") == expected_new
                and summary.get("gradient_queries_reused") == expected_reused,
                f"{arm} full-C80 gradient query provenance differs")
        gradient_source = summary.get("gradient_source")
        require(isinstance(gradient_source, dict) and gradient_source.get("case_count") == 80
                and gradient_source.get("gradient_queries_new") == expected_new
                and gradient_source.get("gradient_queries_reused") == expected_reused,
                f"{arm} gradient source record differs from the frozen budget plan")
        if arm == "dual_kl":
            require(gradient_source.get("diagnostic_pbs_jobid") == diag_summary.get("pbs_jobid")
                    and gradient_source.get("gradient_state") == str(Path(protocol["diagnostic_root"])
                                                                       / "results" / "dual_dual" / "gradient_state.pt"),
                    "dual_kl gradient was not reused from the completed diagnostic")
        refine = summary.get("refinement_updates")
        total = summary.get("total_updates")
        require(refine in (0, 1) and total == 80 + refine
                and summary.get("source_updates") == 80,
                f"{arm} update count must be 80 source updates plus at most one refinement")
        require(summary.get("heldout_used") is False,
                f"{arm} training or selection reports held-out use")
        required_pass(summary, "source_parity_passed", f"{arm} source parity")
        required_pass(summary, "deployment_parity_passed", f"{arm} deployment parity")
        required_pass(summary, "frozen_verified", f"{arm} frozen checks")
        required_pass(summary, "source_original_endpoint_replay", f"{arm} source endpoint replay")
        job = summary.get("pbs_jobid")
        require(job in known_new_jobs, f"{arm} summary job id is absent from current handles")
        receipt = read_json(root / "results" / arm / "training_receipt.json")
        require(receipt.get("complete") is True and receipt.get("protocol") == protocol["protocol"]
                and receipt.get("unit") == arm and receipt.get("pbs_jobid") == job,
                f"{arm} training receipt is incomplete or mismatched")
        for key in ("source_updates", "refinement_updates", "total_updates",
                    "gradient_query_budget", "gradient_queries_new", "gradient_queries_reused"):
            require(receipt.get(key) == summary.get(key), f"{arm} receipt {key} differs from summary")
        required_pass(receipt, "source_parity_passed", f"{arm} receipt source parity")
        required_pass(receipt, "deployment_parity_passed", f"{arm} receipt deployment parity")
        required_pass(receipt, "frozen_verified", f"{arm} receipt frozen checks")
        required_pass(receipt, "frozen_checks", f"{arm} receipt frozen checks")
        selection_file = read_json(root / "results" / arm / "selection.json")
        require(selection_file == selection, f"{arm} saved selection differs from its complete summary")
        expected_refinement = 0 if selection["chosen"].get("condition") == "no_step" else 1
        require(refine == expected_refinement,
                f"{arm} update count disagrees with its C80 no_step/candidate selection")
        summaries[arm], receipts[arm] = summary, receipt
        parsed["calibration"][arm], parsed["heldout"][arm] = [], []
        for split in ("calibration", "heldout"):
            cases = protocol["calibration_case_ids"] if split == "calibration" else protocol["heldout_case_ids"]
            seeds = [0] if split == "calibration" else protocol["heldout_seed_indices"]
            source_arm = source_ref[split][arm]
            source_records = {(r["case_id"], r["seed_index"]): r for r in source_arm["records"]}
            seen = set()
            for case in cases:
                for seed in seeds:
                    record = read_json(endpoint_path(root, split, arm, case, seed))
                    require(record.get("protocol") == protocol["protocol"]
                            and record.get("unit") == arm and record.get("source_updates") == 80
                            and record.get("refinement_updates") == refine
                            and record.get("total_updates") == total
                            and record.get("updates") == total,
                            f"{arm}/{split}/{case}/{seed} update or protocol tags differ")
                    require(record.get("pbs_jobid") in known_jobs,
                            f"{arm}/{split}/{case}/{seed} job id is neither a known run nor a source attempt")
                    row = plan_rows[case]
                    item = parse_record(record, protocol, arm, split, case, seed, row, ref,
                                        scaling, expected_updates=total)
                    reused = record.get("source_record_reused") is True
                    if record["pbs_jobid"] not in known_new_jobs:
                        require(reused and refine == 0,
                                f"{arm}/{split}/{case}/{seed} old job id lacks explicit source reuse")
                    if reused:
                        require(refine == 0 and total == 80,
                                f"{arm}/{split}/{case}/{seed} source endpoint reused after a refinement")
                        original = source_records[(case, seed)]
                        require(record.get("pbs_jobid") == original["pbs_jobid"],
                                f"{arm}/{split}/{case}/{seed} reused source job id differs")
                        for key in ("action", "teacher_action", "commands", "teacher_commands"):
                            require(item["raw"][key] == original["raw"][key],
                                    f"{arm}/{split}/{case}/{seed} reused raw {key} differs")
                        for name in ref.METRICS:
                            require(close(item["metrics"][name], original["metrics"][name]),
                                    f"{arm}/{split}/{case}/{seed} reused metric {name} differs")
                        require(record.get("source_record_protocol") == original["raw"]["protocol"]
                                and record.get("source_record_arm") == arm,
                                f"{arm}/{split}/{case}/{seed} source reuse provenance differs")
                        require(record.get("source_pbs_jobid") == original["pbs_jobid"]
                                and record.get("curated_by_pbs_jobid") in known_new_jobs,
                                f"{arm}/{split}/{case}/{seed} source reuse attempt provenance differs")
                    key = (case, seed)
                    require(key not in seen, f"Duplicate {split}/{arm} endpoint {key}")
                    seen.add(key)
                    parsed[split][arm].append(item)
            expected = {(c, s) for c in cases for s in seeds}
            require(seen == expected and len(parsed[split][arm]) == len(expected),
                    f"{split}/{arm} endpoint coverage differs")
            if split == "calibration":
                selected_mean = (sum(row["metrics"]["objective_loss"] for row in parsed[split][arm])
                                 / len(expected))
                require(close(selected_mean, selection["chosen"]["objective_mean"]),
                        f"{arm} final C80 endpoint mean differs from its selected condition")
        summaries[arm]["_selection"] = selection
        summaries[arm]["_source_stats"] = source_ref
    return summaries, receipts, parsed, source_ref


def suites_from_plan(protocol, plan_rows):
    suites = []
    for case in protocol["calibration_case_ids"]:
        suite = plan_rows[case]["suite"]
        if suite not in suites:
            suites.append(suite)
    return suites


def aggregate_full(records, suites, ref):
    # The shared helper handles all established metrics; this second pass adds the four cross-objectives.
    stats = ref.aggregate_arm(records, suites)
    by_case = {}
    for record in records:
        case = record["case_id"]
        row = by_case.setdefault(case, {"suite": record["suite"], "objectives": {}})
        require(row["suite"] == record["suite"], "Suite changed within a case")
        for name, value in record["objectives"].items():
            row["objectives"].setdefault(name, []).append(value)
    case_values = {case: {name: sum(values) / len(values) for name, values in row["objectives"].items()}
                   for case, row in by_case.items()}
    for case, row in case_values.items():
        row["_suite"] = by_case[case]["suite"]
        for name in OBJECTIVES:
            stats["case_values"][case]["objective_" + name] = row[name]
    by_suite = {}
    for suite in suites:
        selected = [row for row in case_values.values() if row["_suite"] == suite]
        by_suite[suite] = {"objective_" + name: sum(row[name] for row in selected) / len(selected)
                           for name in OBJECTIVES}
        by_suite[suite]["cases"] = len(selected)
        stats["by_suite"][suite].update(by_suite[suite])
    overall = {"objective_" + name: sum(by_suite[suite]["objective_" + name] for suite in suites) / len(suites)
               for name in OBJECTIVES}
    stats["overall"].update(overall)
    stats["objective_overall"] = overall
    stats["objective_by_suite"] = by_suite
    stats["objective_case_values"] = case_values
    return stats


def pair_summary(left, right, suites, ref, seed):
    cases = sorted(left["case_values"])
    require(cases == sorted(right["case_values"]), "Paired held-out case sets differ")
    out = {}
    for metric in PAIRED_METRICS:
        deltas = {case: right["case_values"][case][metric] - left["case_values"][case][metric]
                  for case in cases}
        suite_means = {suite: sum(deltas[c] for c in cases
                                  if left["case_values"][c]["_suite"] == suite)
                       / sum(left["case_values"][c]["_suite"] == suite for c in cases)
                       for suite in suites}
        loco = []
        for excluded in cases:
            suite = left["case_values"][excluded]["_suite"]
            means = [sum(deltas[c] for c in cases
                         if left["case_values"][c]["_suite"] == candidate
                         and (candidate != suite or c != excluded))
                     / sum(left["case_values"][c]["_suite"] == candidate
                           and (candidate != suite or c != excluded) for c in cases)
                     for candidate in suites]
            loco.append(sum(means) / len(means))
        values = list(deltas.values())
        out[metric] = {
            "cases": len(cases), "mean_delta_equal_suite": sum(suite_means.values()) / len(suites),
            "median_delta": statistics.median(values),
            "improved": sum(v < 0 for v in values), "worsened": sum(v > 0 for v in values),
            "tied": sum(v == 0 for v in values), "by_suite_mean_delta": suite_means,
            "equal_suite_leave_one_case_out_mean_range": [min(loco), max(loco)],
            "fixed_seed_case_bootstrap_95pct_ci": ref.bootstrap_interval(values, seed + len(out)),
            "case_deltas": deltas,
        }
    return out


def display_value(value):
    return f"{value:.8g}"


def markdown_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in rows)


def report(protocol, suites, summaries, stats, paired, decision):
    split = "heldout"
    rows = []
    for arm in ARMS:
        rows.append([arm + " original", *[display_value(stats["original"][split][arm]["overall"][m])
                                             for m in CONDITIONS]])
        rows.append([arm + " refined", *[display_value(stats["refined"][split][arm]["overall"][m])
                                             for m in CONDITIONS]])
    delta_rows = [[metric, display_value(item["mean_delta_equal_suite"]),
                   display_value(item["median_delta"]),
                   f"{item['improved']}/{item['worsened']}/{item['tied']}",
                   ", ".join(f"{s}: {display_value(item['by_suite_mean_delta'][s])}" for s in suites),
                   "[" + ", ".join(display_value(v) for v in item["equal_suite_leave_one_case_out_mean_range"]) + "]"]
                  for metric, item in paired["dual_refined_minus_mse_refined"].items()
                  if metric in CONDITIONS]
    objective_rows = [[name,
                       display_value(paired["dual_refined_minus_mse_refined"]["objective_" + name]["mean_delta_equal_suite"]),
                       display_value(paired["dual_refined_minus_mse_refined"]["objective_" + name]["median_delta"])]
                      for name in OBJECTIVES]
    selection_rows = [[arm, summaries[arm]["_selection"]["chosen"].get("condition"),
                       summaries[arm]["refinement_updates"],
                       display_value(summaries[arm]["_selection"]["baseline_objective_mean"]),
                       arm + " formula",
                       display_value(summaries[arm]["_selection"]["chosen"]["objective_mean"])]
                      for arm in ARMS]
    return "\n".join([
        "# Fast-WAM KL full-C80 refinement aggregate", "",
        "分析单位是固定 checkpoint 下的输入 cohort：calibration 为 C80 的 80 个 case；held-out 为 40 个 case、每个 2 个 seed。先在 case 内对 seed 求均值，再在四个 suite 内对 case 求均值，最后四个 suite 等权。两条训练轨迹各只有一个 checkpoint；case bootstrap 仅描述固定 seed 的 case 波动，不代表训练重复，也不作显著性结论。", "",
        "证据范围为 fixed-input decoded-BF16 student evaluation、fake-A4 execution 与 BF16 KV 缓存配置。没有 rollout、closed-loop 或 native-kernel speed 证据。Gripper 不进入主门槛。", "",
        "## Held-out 描述统计", "",
        markdown_table(["条件", *CONDITIONS], rows), "",
        "## 主配对：dual_kl refined 减 mse_nmse refined", "",
        "负值表示 dual_kl refined 指标较低。改善/变差/相同按 40 个 case 计数；seed 已先在 case 内平均。LOCO 每次从所属 suite 移除一个 case，仍对四个 suite 等权。Bootstrap 区间使用固定 seed 对 40 个 case 重抽样，仅作描述。", "",
        markdown_table(["指标", "等 suite mean Δ", "case median Δ", "改善/变差/相同", "suite mean Δ", "等 suite LOCO range"], delta_rows), "",
        "同一公式下的 objective loss 配对变化如下；不同 loss 公式的绝对值不可横向比较。", "",
        markdown_table(["固定 loss 公式", "等 suite mean Δ", "case median Δ"], objective_rows), "",
        "## 相对原始 final-80 checkpoint 的 held-out 配对上下文", "",
        markdown_table(["对比", "first10 motor RMSE Δ", "future latent NMSE Δ", "action KL Δ", "latent KL Δ"],
                       [[label] + [display_value(paired[label][metric]["mean_delta_equal_suite"])
                                   for metric in ("first10motor_rmse", "future_latent_nmse", "action_kl", "latent_kl")]
                        for label in ("mse_refined_minus_mse_original", "dual_refined_minus_dual_original",
                                      "dual_refined_minus_original_mse")]), "",
        "## C80 校准选优", "",
        "两组都在 C80 自身目标上，从 baseline/no_step 与六个 Adam direction/fraction 候选中选均值最低者；selection 未使用 held-out。`no_step` 表示没有 refinement improvement。", "",
        "各 arm 的 C80 objective 使用各自损失公式；这里按 arm 标注，不横向比较公式不同的 objective 数值。", "",
        markdown_table(["Arm", "选择", "refinement updates", "C80 baseline objective", "公式", "C80 selected objective"], selection_rows), "",
        "## 预设 joint screen", "",
        f"joint gate: {decision['candidate_screen']['joint_gate']['passes']}（dual refined held-out first-10 motor RMSE 和 future latent NMSE 都低于 matched mse_nmse refined；motor case median Δ < 0；至少 24/40 case motor RMSE 改善）。实际 refinement 条件：{decision['candidate_screen']['joint_gate']['dual_refinement_performed']}。若选中 `no_step`，按无 refinement improvement 解释。", "",
        f"Action KL 相对 matched MSE refined 变化：{display_value(paired['dual_refined_minus_mse_refined']['action_kl']['mean_delta_equal_suite'])}；latent KL 变化：{display_value(paired['dual_refined_minus_mse_refined']['latent_kl']['mean_delta_equal_suite'])}。只有两个原始 KL 项均下降时，才称两项 information loss 同时改善。", "",
        "单次训练与固定输入结果不能支持 closed-loop、成功率、native latency 或独立训练复现结论。", ""
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    ref = load_reference()
    job_id = ref.guard()
    protocol = read_json(args.protocol)
    require(protocol.get("protocol") == "fastwam-vq-kl-fullbatch-refine-v1",
            "Unexpected full-C80 refinement protocol")
    require(protocol.get("arms") == list(ARMS)
            and protocol.get("fractions") == [1.0, 0.5, 0.25]
            and protocol.get("directions") == ["saved_adam", "fresh_adam"]
            and protocol.get("gradient_query_budget_per_arm") == 80
            and protocol.get("maximum_refinement_updates") == 1,
            "Frozen arm, candidate, or update budget changed")
    root = Path(protocol["remote_root"])
    plan_rows, suites = collect_plan(protocol, ref)
    known_new_jobs = pbs_jobs_in_handles(root)
    summaries, receipts, parsed, source = validate_arm_runs(
        protocol, root, ref, plan_rows, known_new_jobs)
    stats = {"original": {}, "refined": {}}
    split_stats = {"calibration": {}, "heldout": {}}
    for split in ("calibration", "heldout"):
        for arm in ARMS:
            stats["original"].setdefault(split, {})
            stats["refined"].setdefault(split, {})
            stats["original"][split][arm] = source[split][arm]["stats"]
            stats["refined"][split][arm] = aggregate_full(parsed[split][arm], suites, ref)
            expected_cases = 80 if split == "calibration" else 40
            per_suite = 20 if split == "calibration" else 10
            result = stats["refined"][split][arm]
            require(result["cases"] == expected_cases and result["queries"] == (80 if split == "calibration" else 80)
                    and all(result["by_suite"][s]["cases"] == per_suite for s in suites),
                    f"{split}/{arm} coverage or suite balance differs")
            split_stats[split][arm] = result
    paired = {
        "dual_refined_minus_mse_refined": pair_summary(
            stats["refined"]["heldout"]["mse_nmse"], stats["refined"]["heldout"]["dual_kl"],
            suites, ref, 20261020),
        "mse_refined_minus_mse_original": pair_summary(
            stats["original"]["heldout"]["mse_nmse"], stats["refined"]["heldout"]["mse_nmse"],
            suites, ref, 20262020),
        "dual_refined_minus_dual_original": pair_summary(
            stats["original"]["heldout"]["dual_kl"], stats["refined"]["heldout"]["dual_kl"],
            suites, ref, 20263020),
        "dual_refined_minus_original_mse": pair_summary(
            stats["original"]["heldout"]["mse_nmse"], stats["refined"]["heldout"]["dual_kl"],
            suites, ref, 20264020),
    }
    main_pair = paired["dual_refined_minus_mse_refined"]
    motor = main_pair["first10motor_rmse"]
    held = stats["refined"]["heldout"]
    dual = held["dual_kl"]["overall"]
    mse = held["mse_nmse"]["overall"]
    raw_kl_both_lower = dual["action_kl"] < mse["action_kl"] and dual["latent_kl"] < mse["latent_kl"]
    gate_checks = {
        "dual_refinement_performed": summaries["dual_kl"]["refinement_updates"] == 1,
        "motor_rmse_lower": dual["first10motor_rmse"] < mse["first10motor_rmse"],
        "future_latent_nmse_lower": dual["future_latent_nmse"] < mse["future_latent_nmse"],
        "motor_case_median_delta_negative": motor["median_delta"] < 0,
        "motor_improved_cases_at_least_24": motor["improved"] >= 24,
        "both_raw_kl_terms_lower": raw_kl_both_lower,
    }
    gate_core = all(gate_checks[k] for k in ("motor_rmse_lower", "future_latent_nmse_lower",
                                             "motor_case_median_delta_negative",
                                             "motor_improved_cases_at_least_24"))
    decision = {
        "protocol": protocol["protocol"], "complete": True, "pbs_jobid": job_id,
        "unit_of_analysis": "40 held-out cases after seed mean within case; four suites equally weighted",
        "selection_acceptance": {arm: "passed_C80_only_own_objective_minimum"
                                  for arm in ARMS},
        "candidate_screen": {"joint_gate": {"checks": gate_checks,
                                                  "core_gate_passes": gate_core,
                                                  "passes": gate_core and gate_checks["dual_refinement_performed"],
                                                  "dual_refinement_performed": gate_checks["dual_refinement_performed"],
                                                  "raw_kl_both_lower": raw_kl_both_lower,
                                                  "motor_median_delta": motor["median_delta"],
                                                  "motor_improved_cases": motor["improved"],
                                                  "required_motor_improved_cases": 24}},
        "interpretation": "fixed_input engineering screen only; no significance, independent training replication, rollout, closed-loop, or native-kernel speed claim",
    }
    stats_out = {}
    for split in ("calibration", "heldout"):
        stats_out[split] = {}
        for mode in ("original", "refined"):
            stats_out[split][mode] = {}
            for arm in ARMS:
                item = stats[mode][split][arm]
                stats_out[split][mode][arm] = {
                    "queries": item["queries"], "cases": item["cases"],
                    "overall": item["overall"], "by_suite": item["by_suite"],
                    "objective_overall": item.get("objective_overall", {}),
                    "objective_by_suite": item.get("objective_by_suite", {}),
                }
    paired_brief = {contrast: {metric: {k: v for k, v in item.items() if k != "case_deltas"}
                               for metric, item in outcomes.items()}
                    for contrast, outcomes in paired.items()}
    comparison = {
        "protocol": protocol["protocol"], "complete": True, "pbs_jobid": job_id,
        "calibration_records": 160, "heldout_records": 160,
        "new_teacher_queries": 0, "arms": list(ARMS),
        "source_updates_per_arm": 80,
        "refinement_updates": {arm: summaries[arm]["refinement_updates"] for arm in ARMS},
        "total_updates": {arm: summaries[arm]["total_updates"] for arm in ARMS},
        "gradient_queries": {arm: {"budget": 80,
                                    "new": summaries[arm]["gradient_queries_new"],
                                    "reused": summaries[arm]["gradient_queries_reused"]}
                             for arm in ARMS},
        "actual_model_queries": {arm: summaries[arm].get("new_model_queries") for arm in ARMS},
        "forward_accounting": {arm: summaries[arm].get("forward_accounting") for arm in ARMS},
        "source_record_reused_queries": {arm: {split: sum(r["source_record_reused"]
                                                               for r in parsed[split][arm])
                                                  for split in ("calibration", "heldout")}
                                          for arm in ARMS},
        "suite_weighting": "seed mean within case, case mean within suite, four suites equal",
        "loss_scaling": {key: read_json(Path(protocol["source_root"]) / "results" / "prepare" / "loss_scaling.json")[key]
                         for key in ("action_kl_coefficient", "latent_kl_coefficient", "original_latent_weight", "heldout_used")},
        "objective_formulae": {
            "mse_nmse": "first10motor_mse + original_latent_weight * future_latent_nmse",
            "action_kl": "action_kl_coefficient * action_kl + original_latent_weight * future_latent_nmse",
            "latent_kl": "first10motor_mse + original_latent_weight * latent_kl_coefficient * latent_kl",
            "dual_kl": "action_kl_coefficient * action_kl + original_latent_weight * latent_kl_coefficient * latent_kl",
        },
        "splits": stats_out, "paired_contrasts": paired_brief,
        "decision": decision,
        "claims": {"statistical_significance": False, "independent_training_replicates": False,
                   "rollouts": 0, "closed_loop_evaluated": False, "native_kernel_speed_evaluated": False,
                   "gripper_is_primary": False},
    }
    save_json(root / "results" / "comparison.json", comparison, MAX_JSON_BYTES)
    paired_details = {
        "protocol": protocol["protocol"], "complete": True, "pbs_jobid": job_id,
        "case_unit": "held-out seed mean within each of 40 cases; never treat 80 queries as 80 cases",
        "paired_contrast": "dual_kl_refined - mse_nmse_refined",
        "paired_case_deltas": {contrast: {metric: item["case_deltas"] for metric, item in outcomes.items()}
                                for contrast, outcomes in paired.items()},
        "case_metrics": {case: {
            mode + "." + arm: {metric: stats[mode]["heldout"][arm]["case_values"][case][metric]
                                for metric in PAIRED_METRICS}
            for mode in ("original", "refined") for arm in ARMS}
            for case in protocol["heldout_case_ids"]},
        "loco": {contrast: {metric: item["equal_suite_leave_one_case_out_mean_range"]
                            for metric, item in outcomes.items()}
                 for contrast, outcomes in paired.items()},
    }
    save_json(root / "results" / "paired_details.json", paired_details, MAX_JSON_BYTES)
    save_json(root / "results" / "decision.json", decision)
    aggregate_summary = {
        "protocol": protocol["protocol"], "complete": True, "pbs_jobid": job_id,
        "calibration_records": 160, "heldout_records": 160,
        "new_teacher_queries": 0, "arms": list(ARMS),
        "source_updates_per_arm": {arm: 80 for arm in ARMS},
        "refinement_updates": {arm: summaries[arm]["refinement_updates"] for arm in ARMS},
        "total_updates": {arm: summaries[arm]["total_updates"] for arm in ARMS},
        "gradient_query_budget_per_arm": 80,
        "gradient_queries_new": {arm: summaries[arm]["gradient_queries_new"] for arm in ARMS},
        "gradient_queries_reused": {arm: summaries[arm]["gradient_queries_reused"] for arm in ARMS},
        "evaluation_queries_per_arm": 160,
        "actual_model_queries": {arm: summaries[arm].get("new_model_queries") for arm in ARMS},
        "forward_accounting": {arm: summaries[arm].get("forward_accounting") for arm in ARMS},
        "source_record_reused_queries": {arm: {split: sum(r["source_record_reused"]
                                                               for r in parsed[split][arm])
                                                  for split in ("calibration", "heldout")}
                                          for arm in ARMS},
        "one_training_trajectory_per_arm": True, "suites": suites,
    }
    save_json(root / "results" / "aggregate" / "summary.json", aggregate_summary)
    report_text = report(protocol, suites, summaries, stats, paired, decision)
    (root / "results" / "ANALYSIS.zh.md").write_text(report_text, encoding="utf-8")
    print("KL_FULL_C80_REFINE_AGGREGATE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
