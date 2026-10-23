"""Summarize the frozen action+future-latent run on its PBS CPU allocation."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics


ACTION_METRICS = (
    "first10motor_rmse", "full32motor_rmse", "late22motor_rmse", "first10gripper_rmse",
)
FUTURE_METRICS = (
    "future_latent_rmse", "future_latent_nmse",
    "temporal_delta_global_rmse", "temporal_delta_global_nmse",
)
FUTURE_PROFILES = ("future_rmse", "future_nmse", "temporal_delta_rmse", "temporal_delta_nmse")
DIAGNOSTIC_ARMS = (
    "bf16", "parent_self", "parent_teacher_future", "distilled_self", "distilled_teacher_future",
)
OLD_ACTION_ROOT = Path("/scratch/users/ntu/yguo017/fastwam-vq-action-distill-20261011")
OLD_ACTION_JOB = "25764781.pbs101"
MAX_ARTIFACT_BYTES = 256 * 1024
EPSILON = 1e-6


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def guard():
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    if not os.environ.get("PBS_JOBID", "").strip() or not nodefile or "login" in host.lower():
        raise RuntimeError("Analysis requires a genuine PBS compute allocation")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if host not in nodes:
        raise RuntimeError("This host is outside the PBS allocation")


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value, name):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)), f"{name} must be a finite number")
    return float(value)


def close(actual, expected, name, tolerance=EPSILON):
    actual = finite(actual, name)
    expected = finite(expected, name + " expected")
    require(abs(actual - expected) <= tolerance, f"{name} mismatch: {actual} vs {expected}")


def frozen_contexts(protocol):
    contexts = []
    ordinal = 0
    for split, cases, seeds in (
        ("calibration", protocol["calibration_cases"], protocol["calibration_seed_indices"]),
        ("selection", protocol["selection_cases"], protocol["evaluation_seed_indices"]),
        ("test", protocol["test_cases"], protocol["evaluation_seed_indices"]),
    ):
        for case_id in cases:
            shard = ordinal % 2
            for seed_index in seeds:
                contexts.append({"split": split, "case_id": case_id,
                                 "seed_index": seed_index, "shard": shard})
            ordinal += 1
    require(len(contexts) == 60, "Protocol must define exactly 60 frozen contexts")
    return contexts


def exact_context_index(rows, expected_keys, label):
    indexed = {}
    for row in rows:
        key = (row["case_id"], row["seed_index"])
        require(key not in indexed, f"Duplicate {label} context: {key}")
        indexed[key] = row
    expected, actual = set(expected_keys), set(indexed)
    require(actual == expected,
            f"{label} coverage mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return indexed


def select_split_epoch(rows, split, epoch):
    return [row for row in rows if row.get("split") == split and row.get("epoch") == epoch]


def case_means(rows, metric, cases, seeds, label):
    indexed = exact_context_index(rows, [(case, seed) for case in cases for seed in seeds], label)
    return {
        case: statistics.mean(finite(indexed[(case, seed)][metric], f"{label} {metric}")
                             for seed in seeds)
        for case in cases
    }


def vector_case_means(rows, metric, cases, seeds, label):
    indexed = exact_context_index(rows, [(case, seed) for case in cases for seed in seeds], label)
    result = {}
    for case in cases:
        vectors = [indexed[(case, seed)][metric] for seed in seeds]
        require(all(isinstance(vector, list) and vector for vector in vectors),
                f"{label} {metric} must be a nonempty list")
        width = len(vectors[0])
        require(all(len(vector) == width for vector in vectors), f"{label} {metric} list lengths differ")
        result[case] = [statistics.mean(finite(vector[i], f"{label} {metric}[{i}]")
                                        for vector in vectors) for i in range(width)]
    return result


def action_array(row, label):
    action = next((row[name] for name in ("action", "action_32x7", "raw_normalized_action_32x7")
                   if name in row), None)
    require(isinstance(action, list) and len(action) == 32, f"{label} action must be a 32x7 list")
    for t, values in enumerate(action):
        require(isinstance(values, list) and len(values) == 7, f"{label} action row {t} must have seven values")
        for value in values:
            finite(value, f"{label} action coordinate")
    return action


def action_metrics(action, reference, label):
    require(len(action) == len(reference) == 32, f"{label} action horizons differ")
    errors = [[finite(action[t][d], label) - finite(reference[t][d], label)
               for d in range(7)] for t in range(32)]

    def rmse(values):
        return math.sqrt(statistics.mean(value * value for value in values))

    return {
        "first10motor_rmse": rmse([errors[t][d] for t in range(10) for d in range(6)]),
        "full32motor_rmse": rmse([errors[t][d] for t in range(32) for d in range(6)]),
        "late22motor_rmse": rmse([errors[t][d] for t in range(10, 32) for d in range(6)]),
        "first10gripper_rmse": rmse([errors[t][6] for t in range(10)]),
    }


def existing_value(row, aliases, label, required=True):
    for name in aliases:
        if name in row:
            return finite(row[name], label)
    if required:
        raise RuntimeError(f"{label} is missing; expected one of {aliases}")
    return None


def check_stored_action_metrics(row, calculated, label, source=False):
    aliases = {
        "first10motor_rmse": ("motor_rmse_first10_vs_bf16", "first10motor_rmse", "motor_rmse_first10"),
        "first10gripper_rmse": ("gripper_rmse_first10_vs_bf16", "first10gripper_rmse"),
        "full32motor_rmse": ("motor_rmse_32", "motor_rmse_32_vs_bf16", "full32motor_rmse"),
    }
    for metric, names in aliases.items():
        stored = existing_value(row, names, f"{label} stored {metric}", required=source)
        if stored is not None:
            close(stored, calculated[metric], f"{label} stored {metric}")


def test_arms_from_summary(summary):
    test = summary.get("test")
    require(isinstance(test, (dict, list)), "Source full summary must declare test arms under 'test'")
    candidates = test if isinstance(test, list) else None
    if isinstance(test, dict):
        for key in ("arms", "per_arm", "by_arm", "metrics_by_arm", "results_by_arm"):
            value = test.get(key)
            if isinstance(value, (dict, list)):
                candidates = list(value) if isinstance(value, dict) else value
                break
        if candidates is None:
            candidates = [key for key, value in test.items()
                          if isinstance(value, dict) and key not in ("case_count", "context_count", "metrics")]
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        candidates = [row.get("arm") for row in candidates]
    require(isinstance(candidates, list) and candidates and all(isinstance(arm, str) for arm in candidates),
            "Could not read the declared test-arm names from source summary['test']")
    require(len(candidates) == len(set(candidates)), "Source summary declares duplicate test arms")
    require(len(candidates) == 7, f"Expected seven complete source test arms, got {len(candidates)}")
    require({"bf16", "row_output_vq_a4", "row_action_vq_a4"}.issubset(candidates),
            "Source test arms are missing the required BF16, parent, or selected action-VQ reference")
    if isinstance(test, dict):
        if "case_count" in test:
            require(test["case_count"] == 22, "Source summary test case count must be 22")
        if "context_count" in test:
            require(test["context_count"] == 44, "Source summary test context count must be 44")
    return candidates


def read_source_actions(protocol, cases, seeds, cached_teacher):
    source = Path(protocol["source_root"])
    summary = read_json(source / "results" / "full" / "summary.json")
    require(summary.get("status") == "complete", "Action-aware source summary is incomplete")
    source_job = summary.get("pbs_jobid", summary.get("job_id", summary.get("full_job_id")))
    require(source_job == protocol["source_full_job"], "Action-aware source job id mismatch")
    arms = test_arms_from_summary(summary)
    plan = read_json(source / "plan.json")
    inputs = plan.get("inputs")
    require(isinstance(inputs, list), "Action-aware plan has no inputs list")
    metadata = {row["case_id"]: row for row in inputs}
    require(set(cases).issubset(metadata), "Action-aware plan is missing one or more test cases")

    test_rows = []
    calibration_rows = []
    selection_rows = []
    actions_path = source / "results" / "full" / "actions.jsonl"
    require(actions_path.is_file(), f"Action source is missing: {actions_path}")
    with actions_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            row["split"] = row.get("split", row.get("group"))
            row["case_id"] = int(row["case_id"])
            row["seed_index"] = int(row.get("seed_index", row.get("evaluation_seed_index")))
            row["arm"] = row.get("arm", row.get("label"))
            split = row.get("split")
            if split == "selection":
                selection_rows.append(row)
            elif split == "calibration":
                calibration_rows.append(row)
            elif split == "test":
                test_rows.append(row)
    require(test_rows, "Action source contains no test rows")
    undeclared = sorted({row.get("arm") for row in test_rows} - set(arms))
    require(not undeclared, f"Source test rows contain undeclared arms: {undeclared}")
    expected = [(case, seed) for case in cases for seed in seeds]
    raw_by_arm, values_by_arm = {}, {}
    exact_context_index([row for row in test_rows if row.get("arm") == "bf16"],
                        expected, "source BF16 test")
    for arm in arms:
        raw = [row for row in test_rows if row.get("arm") == arm]
        indexed = exact_context_index(raw, expected, f"source {arm} test")
        computed = {}
        for key, row in indexed.items():
            target_row = cached_teacher[key]
            if row.get("sampler_seed") is not None and target_row.get("sampler_seed") is not None:
                require(int(row["sampler_seed"]) == int(target_row["sampler_seed"]),
                        f"Source {arm} sampler seed differs from BF16 at {key}")
            action = action_array(row, f"source {arm} {key}")
            reference = action_array(target_row, f"source BF16 {key}")
            metrics = action_metrics(action, reference, f"source {arm} {key}")
            check_stored_action_metrics(row, metrics, f"source {arm} {key}", source=True)
            if arm == "bf16":
                require(max(metrics.values()) <= EPSILON, f"Source BF16 row is not its own reference at {key}")
            computed[key] = metrics
        raw_by_arm[arm], values_by_arm[arm] = indexed, computed

    parent_arm = protocol["initial_arm"]
    selection = {}
    selection_values = {}
    selection_expected = [(case, seed) for case in protocol["selection_cases"]
                          for seed in protocol["evaluation_seed_indices"]]
    for arm in ("bf16", parent_arm):
        rows = [row for row in selection_rows if row.get("arm") == arm]
        indexed = exact_context_index(rows, selection_expected, f"source {arm} selection")
        selection[arm] = indexed
    selection_teacher = {key: cached_teacher[key] for key in selection_expected}
    for key, row in selection["bf16"].items():
        source_teacher = action_array(row, f"source BF16 selection {key}")
        cached_action = action_array(selection_teacher[key], f"cached BF16 selection {key}")
        require(max(action_metrics(source_teacher, cached_action, f"BF16 selection parity {key}").values()) <= EPSILON,
                f"Source BF16 selection differs from the verified cache at {key}")
    for key, row in selection[parent_arm].items():
        selection_values[key] = action_metrics(action_array(row, f"source parent selection {key}"),
                                               action_array(selection_teacher[key], f"source BF16 selection {key}"),
                                               f"source parent selection {key}")
        check_stored_action_metrics(row, selection_values[key], f"source parent selection {key}", source=True)

    calibration_expected = [(case, seed) for case in protocol["calibration_cases"]
                            for seed in protocol["calibration_seed_indices"]]
    cal_by_arm = {}
    for arm in (parent_arm,):
        rows = [row for row in calibration_rows if row.get("arm") == arm]
        cal_by_arm[arm] = exact_context_index(rows, calibration_expected, f"source {arm} calibration")
    calibration_values = {parent_arm: {}}
    for key, row in cal_by_arm[parent_arm].items():
        target_row = cached_teacher[key]
        if row.get("sampler_seed") is not None and target_row.get("sampler_seed") is not None:
            require(int(row["sampler_seed"]) == int(target_row["sampler_seed"]),
                    f"Source parent calibration sampler seed differs from cache at {key}")
        metrics = action_metrics(action_array(row, f"source {parent_arm} calibration {key}"),
                                 action_array(target_row, f"cached BF16 calibration {key}"),
                                 f"source {parent_arm} calibration {key}")
        check_stored_action_metrics(row, metrics, f"source {parent_arm} calibration {key}")
        calibration_values[parent_arm][key] = metrics
    return (summary, arms, metadata, raw_by_arm, values_by_arm, selection,
            selection_teacher, selection_values, cal_by_arm, calibration_values)


def rows_for_eval(rows, split, epoch, cases, seeds, label):
    selected = select_split_epoch(rows, split, epoch)
    return exact_context_index(selected, [(case, seed) for case in cases for seed in seeds], label)


def values_from_eval(indexed, reference, label):
    values = {}
    for key, row in indexed.items():
        action = action_array(row, f"{label} {key}")
        metrics = action_metrics(action, action_array(reference[key], f"BF16 teacher {key}"), label)
        check_stored_action_metrics(row, metrics, f"{label} {key}")
        values[key] = metrics
    return values


def load_new_checkpoint(protocol, job, cases, seeds, torch):
    root = Path(protocol["remote_root"])
    result = root / "results" / f"full-{job}"
    summary = read_json(result / "summary.json")
    require(summary.get("status") == "complete" and summary.get("protocol") == protocol["protocol"],
            "New experiment summary is incomplete or has a protocol mismatch")
    require(summary.get("job_id") == job, "New experiment summary job id mismatch")
    require(summary.get("updates") == 16 and summary.get("completed_epochs") == 2
            and summary.get("target_linears") == 614, "New experiment did not complete the frozen training schedule")
    require(summary.get("test_queries_before_freeze") == 0 and summary.get("teacher_inference_queries") == 0,
            "New experiment summary reports forbidden pre-freeze or teacher queries")
    require(summary.get("reference_model_forward_count") == 0, "Reference preparation used model forwards")
    require(summary.get("optimizer_reset_after_update1") is True
            and summary.get("optimizer_schedule") == "matched_baseline_optimizer_reset_after_update1",
            "New experiment did not use the matched one-time Adam reset schedule")

    payload = torch.load(result / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    require(payload.get("schema_version") == 2, "New checkpoint schema must be version 2")
    state = payload.get("run_state")
    require(isinstance(state, dict), "New checkpoint has no run_state")
    rows = state.get("evaluation_rows")
    training = state.get("training_history")
    history = state.get("selection_history")
    require(isinstance(rows, list) and isinstance(training, list) and isinstance(history, list),
            "New checkpoint is missing cross-allocation evaluation or training rows")
    # Copy only the plain-data rows needed below before releasing checkpoint tensors.
    rows, training, history = list(rows), list(training), list(history)
    gradient_parity = state.get("gradient_parity")
    selected_epoch = summary.get("selected_epoch")
    require(selected_epoch in (0, 1, 2) and state.get("selected_epoch") == selected_epoch,
            "Selected epoch differs between summary and checkpoint")
    del payload
    require(len(rows) == summary.get("evaluation_contexts"),
            "Checkpoint evaluation_rows count differs from the complete-run summary")
    require(summary.get("selection_history") == history,
            "Checkpoint and final summary selection histories differ")
    require(all(row.get("epoch") == selected_epoch for row in rows if row.get("split") in ("test", "calibration")),
            "Final calibration/test rows must belong to the chosen checkpoint epoch")
    return result, summary, rows, training, history, gradient_parity, selected_epoch


def validate_new_training(protocol, summary, rows, training, history, gradient_parity, selected_epoch,
                          source_selection, source_selection_values, source_selection_teacher):
    require(len(training) == 16, f"Expected all 16 update records, got {len(training)}")
    require([row.get("update") for row in training] == list(range(1, 17)),
            "Training history must contain updates 1 through 16 exactly once")
    for row in training:
        require(row.get("connected") == 614 and row.get("finite") == 614 and row.get("nonzero", 0) > 0,
                f"Gradient coverage failed at update {row.get('update')}")
        require(row.get("video_steps") == 10 and row.get("action_steps") == 10,
                f"Expected 10 video and 10 action denoising steps at update {row.get('update')}")
        require(finite(row.get("scaled_latent_loss"), "scaled latent loss") > 0,
                f"Future-latent loss did not contribute at update {row.get('update')}")
        close(row.get("latent_weight"), summary["latent_weight"],
              f"update {row.get('update')} latent weight", 1e-15)
    require(training[0].get("optimizer_reset_performed_this_update") is True,
            "Adam reset evidence is missing from update 1")
    require(all(not row.get("optimizer_reset_performed_this_update") for row in training[1:]),
            "Adam reset occurred more than once")
    require(isinstance(gradient_parity, dict) and gradient_parity.get("passed") is True,
            "First-update checkpointed action/future parity evidence is missing")
    for target in ("action", "future"):
        parity = gradient_parity.get(target, {})
        require(parity.get("max_abs") == 0 and parity.get("rmse") == 0,
                f"First-update {target} parity with the cached parent must be exact")
    require(gradient_parity.get("video_steps") == 10 and gradient_parity.get("action_steps") == 10,
            "First-update parity evidence must cover 10 video and 10 action steps")

    epochs = [row.get("epoch") for row in history]
    require(epochs == [0, 1, 2], f"Selection history must record initialization and two epochs: {epochs}")
    scores = [(finite(row["mean_motor_rmse_first10"], "selection score"), row["epoch"]) for row in history]
    expected_epoch = min(scores)[1]
    require(selected_epoch == expected_epoch, "Chosen epoch does not match the frozen selection rule")

    selection_cases = protocol["selection_cases"]
    selection_seeds = protocol["evaluation_seed_indices"]
    for epoch in (1, 2):
        indexed = rows_for_eval(rows, "selection", epoch, selection_cases, selection_seeds,
                                f"new selection epoch {epoch}")
        action_values = values_from_eval(indexed, source_selection_teacher,
                                         f"new selection epoch {epoch}")
        case_scores = {case: statistics.mean(action_values[(case, seed)]["first10motor_rmse"]
                                             for seed in selection_seeds)
                       for case in selection_cases}
        score = statistics.mean(case_scores.values())
        history_row = next(row for row in history if row["epoch"] == epoch)
        close(history_row["mean_motor_rmse_first10"], score, f"new epoch {epoch} selection score")

    initial = next(row for row in history if row["epoch"] == 0)
    initial_cases = {case: statistics.mean(source_selection_values[(case, seed)]["first10motor_rmse"]
                                           for seed in selection_seeds)
                     for case in selection_cases}
    close(initial["mean_motor_rmse_first10"], statistics.mean(initial_cases.values()),
          "initial cached selection score")
    require(len(source_selection["row_output_vq_a4"]) == 8, "Initial parent selection rows are incomplete")


def read_reference_cache(protocol, torch):
    summary = read_json(protocol["reference_summary"])
    require(summary.get("status") == "complete" and summary.get("protocol") == protocol["protocol"],
            "Reference preflight summary is incomplete or mismatched")
    require(summary.get("latent_diagnostic_jobs") == protocol["latent_diagnostic_jobs"],
            "Reference summary diagnostic jobs differ from protocol")
    require(summary.get("context_count") == 60 and summary.get("parent_context_count") == 60
            and summary.get("calibration_context_count") == 8 and summary.get("model_forward_count") == 0
            and summary.get("target_linears") == 614, "Reference summary coverage or forward-count gate failed")
    require(summary.get("calibration_ids") == protocol["calibration_cases"],
            "Reference calibration IDs differ from protocol")
    close(summary.get("latent_loss_fraction"), 0.1, "latent loss fraction", tolerance=1e-12)
    close(summary.get("latent_weight"), 0.0029918822946277566, "frozen latent weight", tolerance=1e-15)
    parity = summary.get("action_source_parity", {})
    for arm in ("bf16", "parent"):
        require(parity.get(arm, {}).get("contexts") == 60
                and parity.get(arm, {}).get("max_abs") == 0,
                f"Reference preparation {arm} action parity must be exact on all 60 contexts")
    cache = torch.load(Path(protocol["reference_cache"]), map_location="cpu", weights_only=True, mmap=True)
    require(cache.get("schema_version") == 1 and cache.get("protocol") == protocol["protocol"],
            "Reference cache schema or protocol mismatch")
    require(cache.get("source_root") == protocol["source_root"]
            and cache.get("source_full_job") == protocol["source_full_job"]
            and cache.get("initial_arm") == protocol["initial_arm"], "Reference cache source identity mismatch")
    expected = {f"{row['case_id']}:{row['seed_index']}" for row in frozen_contexts(protocol)}
    require(set(cache.get("teacher", {})) == expected and set(cache.get("parent", {})) == expected,
            "Reference cache teacher/parent maps do not cover all 60 contexts")
    require(cache.get("calibration_ids") == protocol["calibration_cases"],
            "Reference cache calibration IDs differ from protocol")
    close(cache["latent_loss"].get("latent_weight"), summary["latent_weight"], "cache latent weight", 1e-15)
    calibration_teacher = {}
    teacher_actions = {}
    for context in frozen_contexts(protocol):
        case_id, seed_index = context["case_id"], context["seed_index"]
        key = f"{case_id}:{seed_index}"
        teacher = cache["teacher"][key]
        action = teacher["action"]
        require(tuple(action.shape) == (32, 7), f"Cached BF16 action has wrong shape at {key}")
        teacher_actions[(case_id, seed_index)] = {"case_id": case_id, "seed_index": seed_index,
                                                   "sampler_seed": teacher.get("sampler_seed"),
                                                   "action": action.tolist()}
    del cache
    return summary, teacher_actions


def read_diagnostic_future(protocol):
    root = Path(protocol["latent_diagnostic_root"])
    contexts = frozen_contexts(protocol)
    by_shard = {shard: [row for row in contexts if row["shard"] == shard] for shard in (0, 1)}
    result = {arm: {} for arm in ("parent_self", "distilled_self")}
    for shard in (0, 1):
        directory = root / "results" / f"shard-{shard}"
        summary = read_json(directory / "summary.json")
        require(summary.get("status") == "complete" and summary.get("protocol") == "fastwam-vq-latent-diagnostic-v1"
                and summary.get("job_id") == protocol["latent_diagnostic_jobs"][str(shard)]
                and summary.get("shard") == shard and summary.get("context_count") == 30
                and summary.get("expected_context_count") == 30 and summary.get("selected_epoch") == 2,
                f"Latent diagnostic shard {shard} summary failed its completion gate")
        for context in by_shard[shard]:
            path = directory / f"case_{context['case_id']}_seed_{context['seed_index']}.json"
            receipt = read_json(path)
            require(receipt.get("protocol") == "fastwam-vq-latent-diagnostic-v1"
                    and receipt.get("context") == context, f"Diagnostic receipt context mismatch: {path.name}")
            arms, rows = receipt.get("arms"), receipt.get("rows")
            require(isinstance(arms, list) and set(arms) == set(DIAGNOSTIC_ARMS) and len(arms) == 5
                    and isinstance(rows, list) and len(rows) == 5,
                    f"Diagnostic receipt arm coverage mismatch: {path.name}")
            row_index = {}
            for row in rows:
                arm = row.get("arm")
                require(arm in DIAGNOSTIC_ARMS and arm not in row_index,
                        f"Duplicate or unknown diagnostic arm in {path.name}")
                require((row.get("case_id"), row.get("seed_index"), row.get("split"))
                        == (context["case_id"], context["seed_index"], context["split"]),
                        f"Diagnostic row context mismatch in {path.name}")
                if arm in result:
                    for metric in FUTURE_METRICS:
                        finite(row[metric], f"{path.name} {arm} {metric}")
                    for metric in FUTURE_PROFILES:
                        require(isinstance(row.get(metric), list) and len(row[metric]) == 2,
                                f"{path.name} {arm} {metric} must contain two future frames")
                        for value in row[metric]:
                            finite(value, f"{path.name} {arm} {metric}")
                row_index[arm] = row
            require(set(row_index) == set(DIAGNOSTIC_ARMS), f"Incomplete diagnostic receipt: {path.name}")
            key = (context["case_id"], context["seed_index"])
            for arm in result:
                result[arm][key] = row_index[arm]
    expected = {(row["case_id"], row["seed_index"]) for row in contexts}
    for arm, rows in result.items():
        require(set(rows) == expected and len(rows) == 60, f"Diagnostic {arm} does not cover all 60 contexts")
    return result


def mean_summary(values):
    series = list(values.values())
    require(series, "Cannot summarize an empty set of cases")
    return {"case_count": len(series), "mean": statistics.mean(series), "median": statistics.median(series)}


def paired_summary(baseline, target, baseline_name, target_name):
    require(set(baseline) == set(target), "Paired comparison case sets differ")
    cases = sorted(baseline)
    deltas = {case: target[case] - baseline[case] for case in cases}
    worst = sorted(cases, key=lambda case: (deltas[case], case), reverse=True)[:5]
    return {
        "baseline_arm": baseline_name, "target_arm": target_name, "sampling_unit": "test case",
        "baseline": mean_summary(baseline), "target": mean_summary(target),
        "mean_paired_delta": statistics.mean(deltas.values()),
        "median_paired_delta": statistics.median(deltas.values()),
        "improved_cases": sum(value < 0 for value in deltas.values()),
        "worsened_cases": sum(value > 0 for value in deltas.values()),
        "unchanged_cases": sum(value == 0 for value in deltas.values()),
        "worst_five_cases": [{"case_id": case, "baseline": baseline[case], "target": target[case],
                              "delta": deltas[case]} for case in worst],
        "case_deltas": {str(case): deltas[case] for case in cases},
    }


def aggregate_action_rows(indexed_values, cases, seeds):
    return {metric: case_means([{"case_id": case, "seed_index": seed,
                                  **indexed_values[(case, seed)]}
                                 for case in cases for seed in seeds], metric, cases, seeds,
                               f"action {metric}") for metric in ACTION_METRICS}


def summarize_action_rows(indexed_values, cases, seeds):
    return {metric: mean_summary(values)
            for metric, values in aggregate_action_rows(indexed_values, cases, seeds).items()}


def aggregate_future_rows(indexed_rows, cases, seeds, label):
    scalar = {metric: case_means([{"case_id": case, "seed_index": seed,
                                   metric: indexed_rows[(case, seed)][metric]}
                                  for case in cases for seed in seeds], metric, cases, seeds,
                                f"{label} {metric}") for metric in FUTURE_METRICS}
    profiles = {metric: vector_case_means([{"case_id": case, "seed_index": seed,
                                           metric: indexed_rows[(case, seed)][metric]}
                                          for case in cases for seed in seeds], metric, cases, seeds,
                                          f"{label} {metric}") for metric in FUTURE_PROFILES}
    return {**scalar, **profiles}


def write_bounded(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    require(len(data) <= MAX_ARTIFACT_BYTES, f"{path.name} exceeds 256 KiB")
    Path(path).write_bytes(data)


def markdown_report(report):
    lines = [
        "# Fast-WAM action + future latent：冻结队列结果汇总", "",
        f"- 状态：`complete`；analysis PBS job `{report['analysis_job']}`；实验 job `{report['experiment_job']}`。",
        f"- Sampling unit：{report['cohort']['test_cases']} 个 test case；每个 case 先平均两个 seed，再对 case 等权平均。",
        f"- Cohort：calibration {report['cohort']['calibration_cases']} cases / {report['cohort']['calibration_contexts']} contexts；"
        f"test {report['cohort']['test_cases']} cases / {report['cohort']['test_contexts']} contexts。",
        "- 范围：此前已见 cohort 上的 paired fixed-input recovery；0 new model forwards、0 episodes、非 native low-bit kernel；"
        "结果不代表 closed-loop success、独立泛化或 native W4A4 latency。",
        "- 正 delta 表示误差变大；负 delta 表示误差改善。只报告描述性均值、中位 paired delta、case 计数和尾部 case，不做 p-value。", "",
        "## 训练与选择", "",
        f"- 训练完成 {report['training']['updates']} updates / {report['training']['completed_epochs']} epochs；"
        f"chosen epoch={report['training']['selected_epoch']}，last-trained epoch={report['training']['last_trained_epoch']}。"
        "chosen checkpoint 按四个 selection case 的 first-10 motor RMSE 选择；epoch 0 是未更新的初始化点。",
        f"- 冻结 latent weight `w={report['calibration_fit']['latent_weight']:.16g}`；"
        f"calibration 初始 action MSE={report['calibration_fit']['mean_cal_initial_action_mse']:.8g}，"
        f"future NMSE={report['calibration_fit']['mean_cal_initial_future_nmse']:.8g}，"
        f"fraction={report['calibration_fit']['latent_loss_fraction']:.3g}。",
        "- fraction=0.1 表示初始化 calibration 均值上 latent 项的 loss 数值为 action 项的 10%，不表示梯度贡献的 10%。",
        f"- Gradient coverage：16/16 updates 的 connected 与 finite 均为 614/614；nonzero 最低为 "
        f"{report['training']['nonzero_gradient_min']}；Adam 在 update 1 后清空一次，之后 resume 保留 optimizer。", "",
        f"- 首次 checkpointed parity：action max abs={report['training']['first_update_checkpointed_parity']['action']['max_abs']:.8g}，"
        f"future max abs={report['training']['first_update_checkpointed_parity']['future']['max_abs']:.8g}；"
        "loss recipe 同时包含 action 与 future latent supervision。",
        "- 每个 update 的 connected/finite/nonzero、10+10 denoising steps、scaled latent loss 与 reset 记录保存在 analysis.json。", "",
        "### Selection history", "",
        "| Epoch | First-10 motor RMSE | Future latent RMSE | Future latent NMSE | 备注 |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in report["selection_history"]:
        future = row.get("future_metrics", {}).get("metrics", {})
        lines.append(f"| {row['epoch']} | {row['mean_motor_rmse_first10']:.8g} | "
                     f"{future.get('future_latent_rmse', float('nan')):.8g} | "
                     f"{future.get('future_latent_nmse', float('nan')):.8g} | "
                     f"{'chosen' if row['epoch'] == report['training']['selected_epoch'] else row.get('source', '')} |")
    lines += ["", "## Action fidelity", "",
              "下表覆盖 source full summary 声明的全部 7 个完整 test arms、旧 final-action distilled 和本次 chosen action+latent checkpoint。"
              "selection-only lambda candidates 不属于完整 test arms。所有指标按同一 seed 的 BF16 action 复算；"
              "source 已有的 first-10 motor/gripper 与 full-32 指标均逐行复核。", "",
              "| Method | First-10 motor RMSE | Full-32 motor RMSE | Late-22 motor RMSE | First-10 gripper RMSE |",
              "|---|---:|---:|---:|---:|"]
    for arm, values in report["action_test_arms"].items():
        if arm == "action_latent_chosen":
            continue
        label = "final-action distilled (epoch 2)" if arm == "old_action_only" else f"source `{arm}`"
        lines.append(f"| {label} | {values['first10motor_rmse']['mean']:.8g} | "
                     f"{values['full32motor_rmse']['mean']:.8g} | {values['late22motor_rmse']['mean']:.8g} | "
                     f"{values['first10gripper_rmse']['mean']:.8g} |")
    lines.append(f"| chosen action+latent (epoch {report['training']['selected_epoch']}) | "
                 + " | ".join(f"{report['action_test_new'][metric]['mean']:.8g}" for metric in ACTION_METRICS)
                 + " |")
    lines += ["", "### Calibration action fidelity", "",
              "使用冻结的 8 个同源 calibration case、seed 0；此表展示最终选定 checkpoint 的 action 指标。", "",
              "| Arm | First-10 motor RMSE | Full-32 motor RMSE | Late-22 motor RMSE | First-10 gripper RMSE |",
              "|---|---:|---:|---:|---:|"]
    for arm, values in report["action_calibration_arms"].items():
        lines.append(f"| {arm} | " + " | ".join(f"{values[metric]['mean']:.8g}" for metric in ACTION_METRICS) + " |")
    lines += ["", "### Chosen action+latent vs parent and old action-only", "",
              "每行的 delta 先按 case 配对，再在 22 个 case 上汇总；`improved` 指 target RMSE 低于 baseline。", "",
              "| Metric | Baseline | Baseline mean | Chosen mean | Mean paired delta | Median paired delta | Improved / 22 | Worst 5 case IDs |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for comparison_name, comparison in report["core_action_comparisons"].items():
        for metric, result in comparison.items():
            worst = ", ".join(str(row["case_id"]) for row in result["worst_five_cases"])
            lines.append(f"| {metric} | {result['baseline_arm']} | {result['baseline']['mean']:.8g} | "
                         f"{result['target']['mean']:.8g} | {result['mean_paired_delta']:.8g} | "
                         f"{result['median_paired_delta']:.8g} | {result['improved_cases']} | {worst} |")
    lines += ["", "## Future latent fidelity", "",
              "Parent 与 old action-only 的 future 指标复用既有 diagnostic JSON receipts 的 `parent_self` / `distilled_self` rows；"
              "没有读取诊断 `.pt`，也没有新 forward。表中均值对每个 case 的两个 seed 先平均。"
              "Future 指标比较的是 BF16 模型的最终 clean latent，排除被 observation clamp 的 frame 0，包含两个 future latent frames；"
              "它衡量量化后对 BF16 future 的保真度，不是对真实未来视频的准确度。", "",
              "| Arm | Future latent RMSE | Future latent NMSE | Temporal delta RMSE | Temporal delta NMSE |",
              "|---|---:|---:|---:|---:|"]
    for arm, values in report["future_test_arms"].items():
        lines.append(f"| {arm} | " + " | ".join(f"{values[metric]['mean']:.8g}" for metric in FUTURE_METRICS) + " |")
    lines += ["", "### Calibration future fidelity", "",
              "使用同一组 8 个 calibration contexts、seed 0，观察 future loss 的拟合与 test 表现是否一致。", "",
              "| Arm | Future latent RMSE | Future latent NMSE | Temporal delta RMSE | Temporal delta NMSE |",
              "|---|---:|---:|---:|---:|"]
    for arm, values in report["future_calibration_arms"].items():
        lines.append(f"| {arm} | " + " | ".join(f"{values[metric]['mean']:.8g}" for metric in FUTURE_METRICS) + " |")
    lines += ["", "### Per-frame latent profile means", "",
              "以下列出两帧的 RMSE/NMSE profiles；case 与 seed 按同一 paired protocol 聚合。", "",
              "| Arm | Profile | Frame 1 | Frame 2 |",
              "|---|---|---:|---:|"]
    for arm, profiles in report["future_profile_means"].items():
        for metric, values in profiles.items():
            lines.append(f"| {arm} | {metric} | " + " | ".join(f"{value:.8g}" for value in values) + " |")
    lines += ["", "### Future paired comparisons", "",
              "| Metric | Baseline | Baseline mean | Chosen mean | Mean paired delta | Median paired delta | Improved / 22 | Worst 5 case IDs |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for comparison in report["core_future_comparisons"].values():
        for metric, result in comparison.items():
            worst = ", ".join(str(row["case_id"]) for row in result["worst_five_cases"])
            lines.append(f"| {metric} | {result['baseline_arm']} | {result['baseline']['mean']:.8g} | "
                         f"{result['target']['mean']:.8g} | {result['mean_paired_delta']:.8g} | "
                         f"{result['median_paired_delta']:.8g} | {result['improved_cases']} | {worst} |")
    lines += ["", "## Original / Plus 探索性分组", "",
              "按 10 个 Original 与 12 个 Plus test cases 分组，仅作描述性查看，不作独立验证结论。", "",
              "| Contrast | Domain | Metric | Cases | Mean paired delta | Improved | Worsened |",
              "|---|---|---|---:|---:|---:|---:|"]
    for contrast, groups in report["exploratory_domain_comparisons"].items():
        for domain, metrics in groups.items():
            for metric, result in metrics.items():
                lines.append(f"| {contrast} | {domain} | {metric} | {result['case_count']} | "
                             f"{result['mean_paired_delta']:.8g} | {result['improved_cases']} | {result['worsened_cases']} |")
    highlights = [row for row in report["case_rows"] if row["case_id"] in (1016, 1021)]
    if highlights:
        lines += ["", "## 重点 case", "", "以下列出 1016 与 1021（如在冻结 cohort 中）；这里只展示实际计算结果，不预设方向。", "",
                  "| Case | Domain / dimension | New minus parent first10 motor | New minus old action-only first10 motor | New minus parent future latent NMSE |",
                  "|---:|---|---:|---:|---:|"]
        for row in highlights:
            lines.append(f"| {row['case_id']} | {row['domain']} / {row['dimension']} | "
                         f"{row['new_minus_parent_first10motor_rmse']:.8g} | "
                         f"{row['new_minus_old_action_only_first10motor_rmse']:.8g} | "
                         f"{row['new_minus_parent_future_latent_nmse']:.8g} |")
    lines += ["", "## Evidence boundary", "",
              "这是一组此前 cohort 上的 paired recovery measurement，action/future 对照都使用固定输入和缓存参考。"
              "它不能证明 closed-loop success、独立泛化、native kernel 性能或 native W4A4 latency。"
              "新模型的 test 是 freeze 后 chosen checkpoint 的评估；`chosen epoch` 可能早于 `last-trained epoch=2`，"
              "因此应按实际 selection 结果解释，不能把 epoch 0 直接当成训练无效的证据。", ""]
    return "\n".join(lines)


def make_plot(path, case_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.6), constrained_layout=True)
    cases = [row["case_id"] for row in case_rows]
    x = list(range(len(cases)))
    width = 0.38
    for offset, key, color, label in (
        (-width / 2, "new_minus_parent_first10motor_rmse", "#276fbf", "chosen − parent"),
        (width / 2, "new_minus_old_action_only_first10motor_rmse", "#d95f59", "chosen − old action-only"),
    ):
        axes[0].bar([value + offset for value in x], [row[key] for row in case_rows],
                    width=width, color=color, label=label)
    axes[0].axhline(0, color="#777777", linewidth=0.7)
    axes[0].set_ylabel("First-10 motor RMSE delta")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[0].set_title("Paired fixed-input changes by test case; negative is better")
    for offset, key, color, label in (
        (-width / 2, "new_minus_parent_future_latent_nmse", "#279d85", "chosen − parent"),
        (width / 2, "new_minus_old_action_only_future_latent_nmse", "#8d6cab", "chosen − old action-only"),
    ):
        axes[1].bar([value + offset for value in x], [row[key] for row in case_rows],
                    width=width, color=color, label=label)
    axes[1].axhline(0, color="#777777", linewidth=0.7)
    axes[1].set_ylabel("Future latent NMSE delta")
    axes[1].set_xticks(x, [str(case) for case in cases], fontsize=7)
    axes[1].set_xlabel("Test case ID")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    for dpi in (110, 90, 72):
        fig.savefig(path, dpi=dpi)
        if Path(path).stat().st_size <= MAX_ARTIFACT_BYTES:
            plt.close(fig)
            return True
    plt.close(fig)
    raise RuntimeError("paired_recovery.png remains above 256 KiB after lowering resolution")


def run(args):
    guard()
    from test_action_latent_analysis import self_check
    self_check()
    import torch

    protocol = read_json(args.protocol)
    require(protocol.get("protocol") == "fastwam-vq-action-latent-v1", "Unexpected action-latent protocol")
    root = Path(protocol["remote_root"])
    require(re.fullmatch(r"[0-9]+\.pbs[0-9]+", args.job) is not None, "Invalid experiment PBS job id")
    require(protocol.get("baseline_action_distill_job") == OLD_ACTION_JOB, "Old action-only baseline job mismatch")
    cases, seeds = protocol["test_cases"], protocol["evaluation_seed_indices"]
    require(len(cases) == 22 and len(seeds) == 2, "The frozen test cohort must be 22 cases by two seeds")
    args.out.mkdir(parents=True, exist_ok=False)

    reference_summary, teacher_actions = read_reference_cache(protocol, torch)
    (source_summary, source_arms, metadata, source_rows, source_values, source_selection,
     source_selection_teacher, source_selection_values, source_cal_rows, source_cal_values) = read_source_actions(
        protocol, cases, seeds, teacher_actions)
    reference = teacher_actions

    result_root, new_summary, new_rows, training, history, gradient_parity, selected_epoch = load_new_checkpoint(
        protocol, args.job, cases, seeds, torch)
    selected_test = rows_for_eval(new_rows, "test", selected_epoch, cases, seeds, "new selected test")
    selected_cal = rows_for_eval(new_rows, "calibration", selected_epoch,
                                 protocol["calibration_cases"], protocol["calibration_seed_indices"],
                                 "new selected calibration")
    new_test_values = values_from_eval(selected_test, reference, "new selected test")
    new_cal_values = values_from_eval(selected_cal, reference, "new selected calibration")
    validate_new_training(protocol, new_summary, new_rows, training, history, gradient_parity, selected_epoch,
                          source_selection, source_selection_values, source_selection_teacher)

    old_root = OLD_ACTION_ROOT / "results" / f"full-{OLD_ACTION_JOB}"
    old_summary = read_json(old_root / "summary.json")
    require(old_summary.get("status") == "complete" and old_summary.get("job_id") == OLD_ACTION_JOB,
            "Old action-only baseline summary is incomplete or mismatched")
    require(old_summary.get("completed_epochs") == 2 and old_summary.get("updates") == 16,
            "Old action-only baseline did not complete its frozen two-epoch schedule")
    old_payload = torch.load(old_root / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    old_state = old_payload.get("run_state")
    require(isinstance(old_state, dict) and isinstance(old_state.get("evaluation_rows"), list),
            "Old action-only checkpoint has no cross-allocation evaluation_rows")
    old_selected_epoch = old_summary.get("selected_epoch")
    require(old_selected_epoch == 2, "Frozen old action-only actions and diagnostic futures must both use epoch 2")
    old_rows = list(old_state["evaluation_rows"])
    del old_payload
    old_test = rows_for_eval(old_rows, "test", old_selected_epoch, cases, seeds, "old action-only test")
    old_cal = rows_for_eval(old_rows, "calibration", old_selected_epoch,
                            protocol["calibration_cases"], protocol["calibration_seed_indices"],
                            "old action-only calibration")
    old_test_values = values_from_eval(old_test, reference, "old action-only test")
    old_cal_values = values_from_eval(old_cal, reference, "old action-only calibration")

    diagnostic = read_diagnostic_future(protocol)
    diagnostic_test = {arm: {key: row for key, row in rows.items() if key[0] in cases}
                       for arm, rows in diagnostic.items()}
    new_future_test = {key: row for key, row in selected_test.items()}
    future_rows = {
        "parent_self": diagnostic_test["parent_self"],
        "old_action_only": diagnostic_test["distilled_self"],
        "action_latent_chosen": new_future_test,
    }
    future_values = {arm: aggregate_future_rows(rows, cases, seeds, arm)
                     for arm, rows in future_rows.items()}
    calibration_cases, calibration_seeds = protocol["calibration_cases"], protocol["calibration_seed_indices"]
    future_calibration_rows = {
        "parent_self": {key: row for key, row in diagnostic["parent_self"].items() if key[0] in calibration_cases},
        "old_action_only": {key: row for key, row in diagnostic["distilled_self"].items() if key[0] in calibration_cases},
        "action_latent_chosen": selected_cal,
    }
    future_calibration_values = {
        arm: aggregate_future_rows(rows, calibration_cases, calibration_seeds, arm)
        for arm, rows in future_calibration_rows.items()
    }
    action_context_values = {
        **{arm: source_values[arm] for arm in source_arms},
        "old_action_only": old_test_values,
        "action_latent_chosen": new_test_values,
    }
    action_test_arms = {arm: summarize_action_rows(source_values[arm], cases, seeds)
                        for arm in source_arms}
    action_test_arms["old_action_only"] = summarize_action_rows(old_test_values, cases, seeds)
    action_test_arms["action_latent_chosen"] = summarize_action_rows(new_test_values, cases, seeds)
    action_cal_arms = {arm: summarize_action_rows(source_cal_values[arm], protocol["calibration_cases"],
                                                  protocol["calibration_seed_indices"])
                       for arm in source_cal_values}
    action_cal_arms["old_action_only"] = summarize_action_rows(
        old_cal_values, protocol["calibration_cases"], protocol["calibration_seed_indices"])
    action_cal_arms["action_latent_chosen"] = summarize_action_rows(
        new_cal_values, protocol["calibration_cases"], protocol["calibration_seed_indices"])

    core_action_comparisons = {}
    for baseline in (protocol["initial_arm"], "old_action_only"):
        core_action_comparisons[baseline] = {
            metric: paired_summary(
                aggregate_action_rows(action_context_values[baseline], cases, seeds)[metric],
                aggregate_action_rows(action_context_values["action_latent_chosen"], cases, seeds)[metric],
                baseline, "action_latent_chosen") for metric in ACTION_METRICS
        }
    core_future_comparisons = {}
    for baseline in ("parent_self", "old_action_only"):
        core_future_comparisons[baseline] = {
            metric: paired_summary(future_values[baseline][metric],
                                   future_values["action_latent_chosen"][metric],
                                   baseline, "action_latent_chosen") for metric in FUTURE_METRICS
        }

    case_rows = []
    source_by_arm_case = {arm: aggregate_action_rows(values, cases, seeds) for arm, values in source_values.items()}
    action_by_arm_case = {arm: aggregate_action_rows(values, cases, seeds) for arm, values in action_context_values.items()}
    for case in cases:
        meta = metadata[case]
        row = {key: meta.get(key) for key in
               ("domain", "suite", "dimension", "variant", "original_task", "state_id")}
        row["case_id"] = case
        for arm in source_arms:
            for metric in ACTION_METRICS:
                row[f"source_{arm}_{metric}"] = source_by_arm_case[arm][metric][case]
        for arm in ("old_action_only", "action_latent_chosen"):
            for metric in ACTION_METRICS:
                row[f"{arm}_{metric}"] = action_by_arm_case[arm][metric][case]
        for arm, values in future_values.items():
            for metric in FUTURE_METRICS:
                row[f"{arm}_{metric}"] = values[metric][case]
            for metric in FUTURE_PROFILES:
                row[f"{arm}_{metric}"] = json.dumps(values[metric][case], separators=(",", ":"))
        for metric in ACTION_METRICS:
            row[f"new_minus_parent_{metric}"] = (action_by_arm_case["action_latent_chosen"][metric][case]
                                                   - source_by_arm_case[protocol["initial_arm"]][metric][case])
            row[f"new_minus_old_action_only_{metric}"] = (action_by_arm_case["action_latent_chosen"][metric][case]
                                                          - action_by_arm_case["old_action_only"][metric][case])
        for metric in FUTURE_METRICS:
            row[f"new_minus_parent_{metric}"] = (future_values["action_latent_chosen"][metric][case]
                                                  - future_values["parent_self"][metric][case])
            row[f"new_minus_old_action_only_{metric}"] = (future_values["action_latent_chosen"][metric][case]
                                                           - future_values["old_action_only"][metric][case])
        case_rows.append(row)

    exploratory = {}
    for baseline in ("parent_self", "old_action_only"):
        exploratory[baseline] = {}
        for domain in ("original", "plus"):
            domain_cases = [case for case in cases
                            if str(metadata[case].get("domain", "")).lower() == domain]
            require(len(domain_cases) in (10, 12), f"Unexpected {domain} test case count: {len(domain_cases)}")
            exploratory[baseline][domain] = {}
            for metric, base_map, target_map in (
                ("first10motor_rmse", action_by_arm_case[protocol["initial_arm"]]["first10motor_rmse"],
                 action_by_arm_case["action_latent_chosen"]["first10motor_rmse"])
                if baseline == "parent_self" else
                ("first10motor_rmse", action_by_arm_case["old_action_only"]["first10motor_rmse"],
                 action_by_arm_case["action_latent_chosen"]["first10motor_rmse"]),
                ("future_latent_nmse", future_values[baseline]["future_latent_nmse"],
                 future_values["action_latent_chosen"]["future_latent_nmse"]),
            ):
                baseline_values = {case: base_map[case] for case in domain_cases}
                target_values = {case: target_map[case] for case in domain_cases}
                item = paired_summary(baseline_values, target_values, baseline, "action_latent_chosen")
                exploratory[baseline][domain][metric] = {
                    "case_count": len(domain_cases), "mean_paired_delta": item["mean_paired_delta"],
                    "improved_cases": item["improved_cases"], "worsened_cases": item["worsened_cases"],
                    "unchanged_cases": item["unchanged_cases"],
                }

    source_test_summary = source_summary["test"]
    source_test_context_count = (source_test_summary.get("context_count", 44)
                                 if isinstance(source_test_summary, dict) else 44)
    reference_jobs = {"source_vq": protocol["source_full_job"], "old_action_only": OLD_ACTION_JOB,
                      "new_action_latent": args.job,
                      "latent_diagnostic_0": protocol["latent_diagnostic_jobs"]["0"],
                      "latent_diagnostic_1": protocol["latent_diagnostic_jobs"]["1"],
                      "reference_preflight": reference_summary["job_id"]}
    reference_roots = {
        "source_vq": str(Path(protocol["source_root"]) / "results" / "full"),
        "old_action_only": str(OLD_ACTION_ROOT / "results" / f"full-{OLD_ACTION_JOB}"),
        "new_action_latent": str(result_root),
        "latent_diagnostic": str(Path(protocol["latent_diagnostic_root"]) / "results"),
    }
    training_report = {
        "updates": len(training), "completed_epochs": new_summary["completed_epochs"],
        "selected_epoch": selected_epoch, "last_trained_epoch": 2,
        "connected_min": min(row["connected"] for row in training),
        "finite_min": min(row["finite"] for row in training),
        "nonzero_gradient_min": min(row["nonzero"] for row in training),
        "optimizer_reset_after_update1": True,
        "optimizer_schedule": new_summary["optimizer_schedule"],
        "first_update_checkpointed_parity": gradient_parity,
        "loss_recipe": protocol["loss"],
        "per_update_gradient_coverage": [
            {key: row.get(key) for key in
             ("update", "epoch", "connected", "finite", "nonzero", "video_steps", "action_steps",
              "scaled_latent_loss", "latent_weight", "optimizer_reset_performed_this_update")}
            for row in training],
    }
    calibration_fit = {key: reference_summary[key] for key in
                       ("mean_cal_initial_action_mse", "mean_cal_initial_future_nmse",
                        "latent_loss_fraction", "latent_weight", "latent_weight_policy", "calibration_ids")}
    report = {
        "status": "complete", "protocol": protocol["protocol"], "analysis_job": os.environ["PBS_JOBID"],
        "experiment_job": args.job, "result_root": str(result_root),
        "cohort": {"calibration_cases": 8, "calibration_contexts": 8,
                   "test_cases": 22, "test_contexts": 44, "seeds_per_test_case": 2,
                   "test_sampling_unit": "case after averaging the two paired seed-level RMSEs"},
        "test_reference_arms": source_arms, "test_reference_arm_count": len(source_arms),
        "source_selection_only_note": "Other lambda candidates recorded only on selection are not complete test arms and are excluded from the 44-context test table.",
        "new_model_forwards": 0, "episodes": 0, "native_low_bit_kernel": False,
        "claim_scope": "Previously seen cohort paired fixed-input recovery; decoded-BF16 reference compute only.",
        "reference_jobs": reference_jobs, "reference_roots": reference_roots,
        "reference": {"source_summary_status": source_summary["status"],
                      "source_summary_test_context_count": source_test_context_count,
                      "reference_contexts": 60, "parent_cache_contexts": 60,
                      "reference_model_forward_count": reference_summary["model_forward_count"],
                      "parent_bf16_action_parity_max_abs": {
                          arm: reference_summary["action_source_parity"][arm]["max_abs"]
                          for arm in ("bf16", "parent")},
                      "latent_diagnostic_selected_epoch": 2},
        "calibration_fit": calibration_fit,
        "training": training_report, "selection_history": history,
        "action_test_arms": action_test_arms, "action_calibration_arms": action_cal_arms,
        "action_test_new": action_test_arms["action_latent_chosen"],
        "future_test_arms": {arm: {metric: mean_summary(values[metric]) for metric in FUTURE_METRICS}
                             for arm, values in future_values.items()},
        "future_calibration_arms": {arm: {metric: mean_summary(values[metric]) for metric in FUTURE_METRICS}
                                    for arm, values in future_calibration_values.items()},
        "future_profile_means": {arm: {metric: [statistics.mean(values[metric][case][frame] for case in cases)
                                                for frame in range(len(next(iter(values[metric].values()))))]
                                        for metric in FUTURE_PROFILES}
                                 for arm, values in future_values.items()},
        "core_action_comparisons": core_action_comparisons,
        "core_future_comparisons": core_future_comparisons,
        "exploratory_domain_comparisons": exploratory,
        "case_rows": case_rows,
        "gates": {"new_summary_complete": True, "new_checkpoint_schema": 2,
                  "new_evaluation_rows_from_checkpoint": True, "new_selected_rows_complete": True,
                  "source_test_arms_complete": True, "source_first10_metrics_recomputed": True,
                  "reference_cache_teacher_parent_60": True, "latent_receipt_rows_reused": True,
                  "training_updates_with_full_gradient_coverage": 16,
                  "selected_epoch_is_frozen_selection_winner": True,
                  "test_queries_before_freeze": 0, "model_forwards_during_analysis": 0},
    }
    for arm in ("bf16", "parent"):
        parity = finite(reference_summary["action_source_parity"][arm]["max_abs"], f"{arm} parity")
        require(reference_summary["action_source_parity"][arm]["contexts"] == 60 and parity == 0,
                f"Reference preparation {arm} action parity must be exact on all 60 contexts")

    summary = {
        "status": "complete", "protocol": protocol["protocol"], "analysis_job": os.environ["PBS_JOBID"],
        "experiment_job": args.job, "test_case_count": 22, "test_context_count": 44,
        "calibration_context_count": 8, "test_reference_arms": len(source_arms),
        "new_model_forwards": 0, "episodes": 0, "native_low_bit_kernel": False,
        "selected_epoch": selected_epoch, "last_trained_epoch": 2,
        "reference_jobs": reference_jobs, "exact_baseline_root": str(OLD_ACTION_ROOT / "results" / f"full-{OLD_ACTION_JOB}"),
        "exact_baseline_roots": {"source_vq": reference_roots["source_vq"],
                                 "old_action_only": reference_roots["old_action_only"]},
        "exact_new_result_root": str(result_root), "latent_weight": calibration_fit["latent_weight"],
        "all_60_teacher_parent_cache_contexts": True, "all_16_updates_gradient_covered": True,
        "plot_generated": False,
    }
    write_bounded(args.out / "analysis.json", report)
    write_bounded(args.out / "summary.json", summary)
    fields = list(case_rows[0])
    with (args.out / "paired_cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(case_rows)
    require((args.out / "paired_cases.csv").stat().st_size <= MAX_ARTIFACT_BYTES,
            "paired_cases.csv exceeds 256 KiB")
    markdown = markdown_report(report)
    require(len(markdown.encode("utf-8")) <= MAX_ARTIFACT_BYTES, "REPORT.zh.md exceeds 256 KiB")
    (args.out / "REPORT.zh.md").write_text(markdown, encoding="utf-8")
    plot_generated = make_plot(args.out / "paired_recovery.png", case_rows)
    summary["plot_generated"] = plot_generated
    write_bounded(args.out / "summary.json", summary)
    (args.out / "ANALYSIS_COMPLETE").touch()
    print("ANALYSIS_COMPLETE " + json.dumps({"test_case_count": 22, "test_context_count": 44,
                                             "selected_epoch": selected_epoch}, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
