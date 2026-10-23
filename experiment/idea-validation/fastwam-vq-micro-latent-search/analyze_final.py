"""Allocation-only aggregation and report for the frozen two-round final evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import statistics


PROTOCOL = "fastwam-vq-micro-latent-search-v1"
MICRO_PROTOCOL = "fastwam-vq-microscale-v1"
ACTION_KEYS = (
    "motor_rmse_first10_vs_bf16",
    "gripper_rmse_first10_vs_bf16",
    "motor_rmse_32_vs_bf16",
    "motor_rmse_late22_vs_bf16",
)
MICRO_STORED_ACTION_KEYS = ACTION_KEYS[:3]
FUTURE_SCALARS = (
    "future_latent_rmse", "future_latent_nmse", "temporal_delta_global_rmse",
    "temporal_delta_global_nmse", "clamp_firstframe_max_abs",
)
FUTURE_VECTORS = ("future_rmse", "future_nmse", "temporal_delta_rmse", "temporal_delta_nmse")
MAX_OUTPUT_BYTES = 256 * 1024
MAX_INPUT_JSON_BYTES = 4 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def guard():
    host = socket.gethostname().split(".")[0]
    nodefile = os.environ.get("PBS_NODEFILE", "")
    require(bool(os.environ.get("PBS_JOBID")) and bool(nodefile) and "login" not in host.lower(),
            "Final analysis requires a genuine PBS compute allocation")
    nodes = {line.split(".")[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    require(host in nodes, "Analysis host is not listed in PBS_NODEFILE")
    require(not os.environ.get("PBS_GPUFILE"), "Final analysis must use a CPU-only allocation")


def read_json(path, *, limit=MAX_INPUT_JSON_BYTES):
    path = Path(path)
    require(path.is_file(), f"Required file is missing: {path}")
    require(path.stat().st_size <= limit, f"Control JSON is unexpectedly large: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    require(len(content.encode("utf-8")) <= MAX_OUTPUT_BYTES,
            f"Analysis JSON exceeds 256 KiB: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_text(path, content):
    content = content.rstrip() + "\n"
    require(len(content.encode("utf-8")) <= MAX_OUTPUT_BYTES,
            f"Analysis report exceeds 256 KiB: {Path(path).name}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def finite_nonnegative(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"Invalid {label}")
    number = float(value)
    require(math.isfinite(number) and number >= 0, f"Nonfinite or negative {label}")
    return number


def finite_positive(value, label):
    number = finite_nonnegative(value, label)
    require(number > 0, f"{label} must be positive")
    return number


def close(actual, expected, label, *, rel_tol=2e-6, abs_tol=2e-8):
    require(math.isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=abs_tol),
            f"Stored and recomputed values differ for {label}: {actual} vs {expected}")


def candidate_counts(protocol):
    additional = protocol.get("additional_round1_candidates", [])
    require(isinstance(additional, list), "Additional first-round candidates must be a list")
    round1_count = len(protocol["search"]["round1_rhos"]) + len(additional)
    round2_count = protocol["search"]["round2_count"]
    require(type(round2_count) is int and round2_count == 3,
            "Final analysis requires the frozen three-point second round")
    return round1_count, round2_count, round1_count + round2_count


def first_round_specs(protocol):
    base = [{"run_id": f"r1-r{index:03d}", "rho": finite_nonnegative(rho, "first-round rho")}
            for index, rho in enumerate(protocol["search"]["round1_rhos"])]
    additional = protocol.get("additional_round1_candidates", [])
    require(isinstance(additional, list) and all(isinstance(item, dict) for item in additional),
            "Additional first-round candidate identities are malformed")
    base.extend({"run_id": item.get("run_id"), "rho": finite_nonnegative(item.get("rho"), "additional rho")}
                for item in additional)
    require(all(isinstance(item["run_id"], str) and item["run_id"].startswith("r1-r") for item in base)
            and len({item["run_id"] for item in base}) == len(base)
            and len({item["rho"] for item in base}) == len(base),
            "First-round candidate identities or rhos are duplicate or malformed")
    return base


def selection_metric(row, name):
    if name in row:
        return row[name]
    future = row.get("future_metrics", {})
    return future.get("metrics", {}).get(name)


def context_key(case, seed):
    return f"{case}:{seed}"


def validate_action_aggregate(summary, cases, contexts, label):
    require(isinstance(summary, dict), f"{label} action aggregate is missing")
    require(summary.get("case_count") == len(cases) and summary.get("context_count") == contexts,
            f"{label} action coverage differs")
    per_case = summary.get("per_case")
    expected = {str(case) for case in cases}
    require(isinstance(per_case, dict) and set(per_case) == expected,
            f"{label} action per-case keys differ")
    for case in cases:
        values = per_case[str(case)]
        for key in ACTION_KEYS:
            finite_nonnegative(values.get(key), f"{label}/{case}/{key}")
    aggregates = {
        "mean_motor_rmse_first10": ACTION_KEYS[0],
        "mean_gripper_rmse_first10": ACTION_KEYS[1],
        "mean_motor_rmse_32": ACTION_KEYS[2],
        "mean_motor_rmse_late22": ACTION_KEYS[3],
    }
    means = {}
    for aggregate, metric in aggregates.items():
        measured = statistics.mean(per_case[str(case)][metric] for case in cases)
        close(summary.get(aggregate), measured, f"{label}/{aggregate}")
        means[aggregate] = measured
    return {"case_count": len(cases), "context_count": contexts,
            "per_case": per_case, **means}


def aggregate_action_rows(rows, cases, seeds, metrics_module, torch):
    expected = {(case, seed) for case in cases for seed in seeds}
    seen, by_case = set(), {str(case): [] for case in cases}
    for row in rows:
        identity = (row["case_id"], row["seed_index"])
        require(identity in expected and identity not in seen, f"Unexpected or duplicate action row {identity}")
        seen.add(identity)
        action = torch.tensor(row["action"], dtype=torch.float32, device="cpu")
        reference = row["teacher_action"]
        measured = metrics_module.action_metrics(torch, action, reference)
        values = {
            "motor_rmse_first10_vs_bf16": measured["first10motor_rmse"],
            "gripper_rmse_first10_vs_bf16": measured["first10gripper_rmse"],
            "motor_rmse_32_vs_bf16": measured["full32motor_rmse"],
            "motor_rmse_late22_vs_bf16": measured["late22motor_rmse"],
        }
        by_case[str(row["case_id"])].append(values)
    require(seen == expected, f"Action rows do not cover exactly {len(expected)} frozen contexts")
    per_case = {
        str(case): {metric: statistics.mean(row[metric] for row in by_case[str(case)])
                    for metric in ACTION_KEYS}
        for case in cases
    }
    means = {
        "mean_motor_rmse_first10": statistics.mean(per_case[str(case)][ACTION_KEYS[0]] for case in cases),
        "mean_gripper_rmse_first10": statistics.mean(per_case[str(case)][ACTION_KEYS[1]] for case in cases),
        "mean_motor_rmse_32": statistics.mean(per_case[str(case)][ACTION_KEYS[2]] for case in cases),
        "mean_motor_rmse_late22": statistics.mean(per_case[str(case)][ACTION_KEYS[3]] for case in cases),
    }
    return {"case_count": len(cases), "context_count": len(expected),
            "per_case": per_case, **means}


def compare_actions(baseline, selected, cases, label):
    metrics = {
        "mean_motor_rmse_first10": ACTION_KEYS[0],
        "mean_gripper_rmse_first10": ACTION_KEYS[1],
        "mean_motor_rmse_32": ACTION_KEYS[2],
        "mean_motor_rmse_late22": ACTION_KEYS[3],
    }
    per_case = {}
    for case in cases:
        case_key = str(case)
        base_row, selected_row = baseline["per_case"][case_key], selected["per_case"][case_key]
        per_case[case_key] = {}
        for aggregate, metric in metrics.items():
            base_value = finite_nonnegative(base_row[metric], f"{label}/{case}/baseline/{metric}")
            selected_value = finite_nonnegative(selected_row[metric], f"{label}/{case}/selected/{metric}")
            per_case[case_key][aggregate] = {
                "baseline": base_value, "selected": selected_value,
                "selected_minus_baseline": selected_value - base_value,
            }
    metric_summary = {}
    for aggregate in metrics:
        deltas = [per_case[str(case)][aggregate]["selected_minus_baseline"] for case in cases]
        metric_summary[aggregate] = {
            "baseline": baseline[aggregate], "selected": selected[aggregate],
            "selected_minus_baseline": selected[aggregate] - baseline[aggregate],
            "improved_cases": sum(delta < 0 for delta in deltas),
            "worsened_cases": sum(delta > 0 for delta in deltas),
            "tied_cases": sum(delta == 0 for delta in deltas),
            "case_count": len(cases),
        }
    return {"label": label, "case_count": len(cases),
            "context_count": baseline["context_count"], "metrics": metric_summary,
            "per_case": per_case}


def validate_future_aggregate(summary, cases, contexts, label):
    require(isinstance(summary, dict) and summary.get("case_count") == len(cases)
            and summary.get("context_count") == contexts, f"{label} future coverage differs")
    per_case = summary.get("per_case")
    require(isinstance(per_case, dict) and set(per_case) == {str(case) for case in cases},
            f"{label} future per-case keys differ")
    metrics = summary.get("metrics")
    require(isinstance(metrics, dict), f"{label} future metrics are missing")
    for key in FUTURE_SCALARS:
        finite_nonnegative(metrics.get(key), f"{label}/{key}")
    for key in FUTURE_VECTORS:
        values = metrics.get(key)
        require(isinstance(values, list) and len(values) == 3, f"{label}/{key} must have three channels")
        for index, value in enumerate(values):
            finite_nonnegative(value, f"{label}/{key}/{index}")
    for case in cases:
        row = per_case[str(case)]
        for key in FUTURE_SCALARS:
            finite_nonnegative(row.get(key), f"{label}/{case}/{key}")
        for key in FUTURE_VECTORS:
            values = row.get(key)
            require(isinstance(values, list) and len(values) == 3,
                    f"{label}/{case}/{key} must have three channels")
            for index, value in enumerate(values):
                finite_nonnegative(value, f"{label}/{case}/{key}/{index}")
    return summary


def validate_teacher_cache(torch, protocol, reference_summary, cache_path):
    require(reference_summary.get("status") == "complete"
            and reference_summary.get("protocol") == PROTOCOL
            and reference_summary.get("reference_cache") == str(cache_path)
            and reference_summary.get("teacher_context_count") == 60
            and reference_summary.get("teacher_forward_count") == 0,
            "Shared teacher reference summary does not match the frozen protocol")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    require(cache.get("schema_version") == 2 and cache.get("status") == "complete"
            and cache.get("protocol") == PROTOCOL
            and cache.get("source_root") == protocol["source_root"]
            and cache.get("source_full_job") == protocol["source_full_job"]
            and cache.get("initial_arm") == protocol["initial_arm"]
            and cache.get("bank_root") == protocol["bank_root"]
            and cache.get("bank_full_job") == protocol["bank_full_job"]
            and cache.get("teacher_context_count") == 60
            and cache.get("parent_context_count") == 16,
            "Shared reference cache identity differs")
    teacher = cache.get("teacher")
    require(isinstance(teacher, dict) and len(teacher) == 60,
            "Shared BF16 teacher cache must contain exactly 60 contexts")
    expected = set()
    for split, cases, seeds in (
        ("calibration", protocol["calibration_cases"], protocol["calibration_seed_indices"]),
        ("selection", protocol["selection_cases"], protocol["evaluation_seed_indices"]),
        ("test", protocol["test_cases"], protocol["evaluation_seed_indices"]),
    ):
        expected.update(context_key(case, seed) for case in cases for seed in seeds)
    require(set(teacher) == expected, "Teacher cache context keys differ from frozen 8/8/44 cohort")
    for key, row in teacher.items():
        case, seed = (int(value) for value in key.split(":"))
        require(row.get("case_id") == case and row.get("seed_index") == seed
                and type(row.get("sampler_seed")) is int,
                f"Teacher cache identity metadata is invalid at {key}")
        action, future = row.get("action"), row.get("future")
        require(torch.is_tensor(action) and tuple(action.shape) == (32, 7)
                and action.is_floating_point() and bool(torch.isfinite(action).all()),
                f"Teacher action is invalid at {key}")
        require(torch.is_tensor(future) and tuple(future.shape) == (1, 48, 3, 14, 28)
                and future.is_floating_point() and bool(torch.isfinite(future).all()),
                f"Teacher future is invalid at {key}")
    return teacher


def load_micro_test_actions(torch, metrics_module, protocol, teacher):
    root = Path(protocol["bank_root"])
    result_dir = root / "results" / "full"
    summary = read_json(result_dir / "summary.json")
    require(summary.get("status") == "complete" and summary.get("protocol") == MICRO_PROTOCOL
            and summary.get("job_id") == protocol["bank_full_job"]
            and summary.get("source_full_job") == protocol["source_full_job"]
            and summary.get("fitted_linears") == 614 and summary.get("new_queries") == 52,
            "Declared micro bank run is not the completed 614-linears source")
    test_summary = summary.get("test")
    require(isinstance(test_summary, dict) and test_summary.get("case_count") == 22
            and test_summary.get("context_count") == 44,
            "Original micro test summary does not cover 22 cases / 44 contexts")
    rows = []
    seen = set()
    with (result_dir / "actions.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("arm") != protocol["initial_arm"] or row.get("split") != "test":
                continue
            case, seed = row.get("case_id"), row.get("seed_index")
            identity = (case, seed)
            require(case in protocol["test_cases"] and seed in protocol["evaluation_seed_indices"]
                    and identity not in seen, f"Unexpected or duplicate micro test action at line {line_number}")
            key = context_key(case, seed)
            teacher_row = teacher[key]
            require(row.get("sampler_seed") == teacher_row.get("sampler_seed"),
                    f"Micro action sampler seed differs from teacher cache at {key}")
            rows.append({"case_id": case, "seed_index": seed, "action": row.get("action"),
                         "teacher_action": teacher_row.get("action")})
            seen.add(identity)
    expected = {(case, seed) for case in protocol["test_cases"]
                for seed in protocol["evaluation_seed_indices"]}
    require(seen == expected, "Original micro actions do not cover exactly the frozen test44 contexts")
    aggregate = aggregate_action_rows(rows, protocol["test_cases"],
                                      protocol["evaluation_seed_indices"], metrics_module, torch)
    original_metrics = {
        "mean_motor_rmse_first10": "motor_rmse_first10_vs_bf16",
        "mean_gripper_rmse_first10": "gripper_rmse_first10_vs_bf16",
        "mean_motor_rmse_32": "motor_rmse_32_vs_bf16",
    }
    for aggregate_key, stored_key in original_metrics.items():
        close(test_summary.get(stored_key), aggregate[aggregate_key],
              f"original micro summary {stored_key}")
    original_per_case = test_summary.get("per_case")
    require(isinstance(original_per_case, dict)
            and set(original_per_case) == {str(case) for case in protocol["test_cases"]},
            "Original micro per-case action summary differs from frozen test cases")
    for case in protocol["test_cases"]:
        for aggregate_key, stored_key in original_metrics.items():
            close(original_per_case[str(case)].get(stored_key),
                  aggregate["per_case"][str(case)][stored_key],
                  f"original micro per-case {case}/{stored_key}")
    return aggregate, summary


def validate_choice_binding(protocol, choice, final_summary, choice_path, final_summary_path):
    root = Path(protocol["remote_root"]).resolve()
    _, _, expected_candidates = candidate_counts(protocol)
    require(Path(choice_path).resolve() == (root / "optimal_selection.json").resolve(),
            "Choice input must be ROOT/optimal_selection.json")
    require(choice.get("status") == "complete" and choice.get("protocol") == PROTOCOL
            and choice.get("rounds_complete") == 2 and choice.get("candidate_count") == expected_candidates,
            "Frozen choice is not a complete two-round selection over all frozen candidates")
    require(len(choice.get("selection_source_jobs", [])) == expected_candidates
            and len(set(choice.get("selection_source_jobs", []))) == expected_candidates
            and choice.get("test_used_for_selection") is False
            and choice.get("model_forward_count") == 0,
            "Frozen choice source jobs or test-isolation flag differ")
    selected_epoch = choice.get("selected_epoch")
    require(type(selected_epoch) is int and selected_epoch in (0, 1, 2),
            "Frozen choice selected epoch must be 0, 1, or 2")
    run_id, selected_job = choice.get("selected_run_id"), choice.get("selected_job_id")
    require(isinstance(run_id, str) and run_id.startswith(("r1-", "r2-"))
            and isinstance(selected_job, str) and selected_job,
            "Frozen choice selected run/job binding is missing")
    expected_result = root / "results" / f"{run_id}-{selected_job}"
    require(Path(choice.get("selected_result_dir", "")).resolve() == expected_result.resolve(),
            "Frozen choice selected_result_dir does not match its run/job")
    expected_final = root / "results" / f"final-{final_summary.get('job_id')}" / "summary.json"
    require(Path(final_summary_path).resolve() == expected_final.resolve(),
            "Final summary path does not match ROOT/results/final-JOB/summary.json")
    require(final_summary.get("status") == "complete" and final_summary.get("mode") == "final"
            and final_summary.get("protocol") == PROTOCOL and final_summary.get("round") == 2
            and final_summary.get("rounds_complete") == 2,
            "Final evaluation summary is incomplete or not in final mode")
    binding = {
        "selected_run_id": run_id, "selected_job_id": selected_job,
        "selected_epoch": selected_epoch, "selected_round": choice.get("selected_round"),
        "rho": choice.get("rho"), "latent_weight": choice.get("latent_weight"),
        "normalization_c": choice.get("normalization_c"),
    }
    require(binding["selected_round"] in (1, 2), "Frozen choice selected_round must be 1 or 2")
    require(isinstance(choice.get("selection_job_id"), str) and choice["selection_job_id"]
            and choice.get("selection_source_jobs") == final_summary.get("selection_source_jobs"),
            "Final summary does not preserve the verified chooser job/source bindings")
    for key, expected in binding.items():
        actual = final_summary.get(key)
        if key in ("rho", "latent_weight", "normalization_c"):
            close(actual, expected, f"final choice binding {key}", rel_tol=1e-12, abs_tol=1e-12)
        else:
            require(actual == expected, f"Final summary differs from frozen choice at {key}")
    require(final_summary.get("target_linears") == 614
            and final_summary.get("completed_epochs") == 2 and final_summary.get("updates") == 0
            and final_summary.get("training_performed") is False
            and final_summary.get("calibration_context_count") == 8
            and final_summary.get("test_context_count") == 44
            and final_summary.get("test_forward_count") == 44
            and final_summary.get("teacher_forward_count") == 0
            and final_summary.get("parent_test_future_forward_count") == 0
            and final_summary.get("test_independence_boundary") == protocol["reuse_boundary"],
            "Final evaluation does not meet the frozen no-training 8/44/0-forward gates")
    require(Path(final_summary.get("selected_result_dir", "")).resolve() == expected_result.resolve()
            and final_summary.get("selection_job_id") == choice["selection_job_id"],
            "Final summary selected-result or chooser-job binding differs")
    require(final_summary.get("selection_source_jobs") == choice["selection_source_jobs"],
            "Final summary selection source jobs differ from optimal selection")
    close(choice.get("latent_weight"), float(choice.get("rho")) * float(choice.get("normalization_c")),
          "frozen lambda=rho*c", rel_tol=1e-12, abs_tol=1e-12)
    return root, binding


def validate_selected_search(protocol, choice, final_summary, selection_cases):
    result_dir = Path(choice["selected_result_dir"])
    candidate = read_json(result_dir / "summary.json")
    require(candidate.get("status") == "search_complete"
            and candidate.get("protocol") == PROTOCOL
            and candidate.get("job_id") == choice["selected_job_id"]
            and candidate.get("run_id") == choice["selected_run_id"]
            and candidate.get("round") == choice["selected_round"],
            "Selected search summary identity differs from optimal selection")
    for key in ("rho", "latent_weight", "normalization_c"):
        close(candidate.get(key), choice.get(key), f"selected search {key}", rel_tol=1e-12, abs_tol=1e-12)
    history = candidate.get("selection_history")
    require(isinstance(history, list) and [row.get("epoch") for row in history] == [0, 1, 2],
            "Selected search summary lacks complete epoch 0/1/2 history")
    selected_row = history[choice["selected_epoch"]]
    require(selected_row.get("epoch") == choice["selected_epoch"], "Selected epoch row is misplaced")
    selected_metrics = choice.get("selected_metrics")
    require(isinstance(selected_metrics, dict), "Optimal selection lacks selected_metrics")
    for key, value in selected_metrics.items():
        close(selection_metric(selected_row, key), value, f"chosen epoch metric {key}")
    final_selection = final_summary.get("selection")
    require(isinstance(final_selection, dict), "Final summary lacks selected selection metrics")
    for key in ("case_count", "context_count", "epoch", "p90_motor_rmse_first10"):
        if key in selected_row:
            require(final_selection.get(key) == selected_row[key],
                    f"Final selected selection row differs at {key}")
    for key in ("future_latent_nmse", "temporal_delta_global_nmse"):
        close(selection_metric(final_selection, key), selected_metrics[key],
              f"final selection future metric {key}")
    for key in ACTION_KEYS:
        aggregate_key = {
            ACTION_KEYS[0]: "mean_motor_rmse_first10",
            ACTION_KEYS[1]: "mean_gripper_rmse_first10",
            ACTION_KEYS[2]: "mean_motor_rmse_32",
            ACTION_KEYS[3]: "mean_motor_rmse_late22",
        }[key]
        for case in selection_cases:
            close(final_selection.get("per_case", {}).get(str(case), {}).get(key),
                  selected_row.get("per_case", {}).get(str(case), {}).get(key),
                  f"final selection per-case {case}/{key}")
        close(final_selection.get(aggregate_key), selected_row.get(aggregate_key),
              f"final selection aggregate {aggregate_key}")
    return candidate, selected_row


def selection_table(choice, protocol):
    points = choice.get("points")
    round1_count, round2_count, candidate_count = candidate_counts(protocol)
    expected_points = candidate_count * 3
    require(isinstance(points, list) and len(points) == expected_points,
            f"Frozen choice must contain {expected_points} points across {candidate_count} candidates and three epochs")
    candidate_keys = {(point.get("round"), point.get("run_id")) for point in points}
    require(len(candidate_keys) == candidate_count and {point.get("epoch") for point in points} == {0, 1, 2},
            f"Selection points do not contain {candidate_count} runs x epochs 0/1/2")
    expected_first = {(item["run_id"], item["rho"]) for item in first_round_specs(protocol)}
    actual_first = {(point.get("run_id"), float(point["rho"])) for point in points if point.get("round") == 1}
    require(len(actual_first) == round1_count and actual_first == expected_first,
            "First-round selection points differ from the frozen candidate identities/rhos")
    require(sum(1 for round_number, _ in candidate_keys if round_number == 2) == round2_count,
            "Selection points do not contain the frozen second-round candidate count")
    for round_number, run_id in candidate_keys:
        run_points = [point for point in points
                      if (point.get("round"), point.get("run_id")) == (round_number, run_id)]
        require({point.get("epoch") for point in run_points} == {0, 1, 2}
                and len(run_points) == 3
                and len({float(point["rho"]) for point in run_points}) == 1,
                f"Candidate {run_id} does not have one bound point at each epoch")
    require(len({float(point["rho"]) for point in points}) == candidate_count,
            "Frozen search candidates do not have distinct rho values")
    guard_names = protocol["search"]["guardrails"]
    require(choice.get("guardrail_fraction") == protocol["search"]["guardrail_fraction"]
            and choice.get("near_tie_fraction") == protocol["search"]["near_tie_fraction"],
            "Choice guard/near-tie policy differs from the frozen protocol")
    for name in guard_names:
        finite_nonnegative(choice["baseline"].get(name), f"selection baseline/{name}")
    for point in points:
        failures = point.get("guard_failures")
        require(type(point.get("eligible")) is bool and isinstance(failures, list)
                and all(name in guard_names for name in failures), "Malformed guardrail flags in choice point")
        metrics = point.get("metrics", {})
        finite_nonnegative(point.get("rho"), f"selection/{point.get('run_id')}/rho")
        finite_nonnegative(point.get("latent_weight"), f"selection/{point.get('run_id')}/lambda")
        for key in ("mean_motor_rmse_first10", "mean_motor_rmse_late22",
                    "mean_gripper_rmse_first10", "p90_motor_rmse_first10"):
            finite_nonnegative(metrics.get(key), f"selection/{point.get('run_id')}/{point.get('epoch')}/{key}")
        expected_failures = [name for name in guard_names
                             if metrics[name] > choice["baseline"][name]
                             * (1 + choice["guardrail_fraction"]) + 1e-12]
        require(failures == expected_failures and point["eligible"] is (not expected_failures),
                f"Guard flags do not recompute for {point.get('run_id')} epoch {point.get('epoch')}")
    selected_key = (choice["selected_run_id"], choice["selected_epoch"])
    chosen_point = next((point for point in points
                         if (point.get("run_id"), point.get("epoch")) == selected_key), None)
    require(chosen_point is not None and chosen_point.get("eligible") is True,
            "Frozen near-tie winner is absent or guard-ineligible")
    strict = choice.get("strict_eligible_minimum")
    require(isinstance(strict, dict) and strict.get("eligible") is True,
            "Frozen strict eligible minimum is missing")
    eligible = [point for point in points if point["eligible"]]
    recomputed_strict = min(eligible, key=lambda point: (
        point["metrics"]["mean_motor_rmse_first10"], point["rho"], point["epoch"], point["run_id"]))
    minimum = recomputed_strict["metrics"]["mean_motor_rmse_first10"]
    near = [point for point in eligible if point["metrics"]["mean_motor_rmse_first10"]
            <= minimum * (1 + choice["near_tie_fraction"]) + 1e-12]
    recomputed_winner = min(near, key=lambda point: (point["rho"], point["epoch"], point["run_id"]))
    require((strict.get("run_id"), strict.get("epoch"))
            == (recomputed_strict["run_id"], recomputed_strict["epoch"]),
            "Stored strict eligible minimum does not recompute")
    require((choice["selected_run_id"], choice["selected_epoch"])
            == (recomputed_winner["run_id"], recomputed_winner["epoch"]),
            "Stored near-tie winner does not recompute")
    return sorted(points, key=lambda item: (item["round"], item["rho"], item["epoch"], item["run_id"])), chosen_point, strict


def self_check():
    keys = ("m", "g")
    rows = [
        {"case_id": 4, "seed_index": 0, "m": 1.0, "g": 2.0},
        {"case_id": 4, "seed_index": 1, "m": 3.0, "g": 4.0},
        {"case_id": 8, "seed_index": 0, "m": 5.0, "g": 6.0},
        {"case_id": 8, "seed_index": 1, "m": 7.0, "g": 8.0},
    ]
    per_case = {}
    for case in (4, 8):
        per_case[str(case)] = {key: statistics.mean(row[key] for row in rows if row["case_id"] == case)
                               for key in keys}
    require(per_case["4"] == {"m": 2.0, "g": 3.0}
            and per_case["8"] == {"m": 6.0, "g": 7.0},
            "Self-check seed aggregation failed")
    require(statistics.mean(per_case[str(case)]["m"] for case in (4, 8)) == 4.0,
            "Self-check case-balanced aggregation failed")
    paired = [1.0 - 2.0, 4.0 - 3.0]
    require(sum(delta < 0 for delta in paired) == 1 and sum(delta > 0 for delta in paired) == 1,
            "Self-check paired-sign counts failed")


def _selection_markdown(points):
    lines = [
        "| Round | Candidate | rho | Epoch | lambda | First10 motor | Late22 motor | First10 gripper | P90 first10 | Guard result |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for point in points:
        metrics = point["metrics"]
        failures = point["guard_failures"]
        flags = "PASS" if not failures else "FAIL: " + ", ".join(failures)
        lines.append(
            f"| {point['round']} | `{point['run_id']}` | {point['rho']:.8g} | {point['epoch']} | "
            f"{point['latent_weight']:.8g} | {metrics['mean_motor_rmse_first10']:.7f} | "
            f"{metrics['mean_motor_rmse_late22']:.7f} | {metrics['mean_gripper_rmse_first10']:.7f} | "
            f"{metrics['p90_motor_rmse_first10']:.7f} | {flags} |"
        )
    return lines


def _action_markdown(label, comparison):
    lines = [f"### {label}", "", "| Metric | Baseline | Selected | Selected − baseline | Improved cases | Worsened cases |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, row in comparison["metrics"].items():
        lines.append(f"| {key} | {row['baseline']:.8f} | {row['selected']:.8f} | "
                     f"{row['selected_minus_baseline']:+.8f} | {row['improved_cases']}/{row['case_count']} | "
                     f"{row['worsened_cases']}/{row['case_count']} |")
    return lines


def make_report(protocol, choice, final_summary, binding, points, strict, action_comparisons,
                micro_summary, future, selected_search):
    selected = choice["selected_metrics"]
    strict_metrics = strict["metrics"]
    test = action_comparisons["micro_parent_vs_selected_test"]
    chosen_selected_round = binding["selected_round"]
    report = [
        "# Fast-WAM micro latent search：最终配对数值报告", "",
        f"最终参数：rho={binding['rho']:.8g}，lambda={binding['latent_weight']:.8g}，"
        f"normalization_c={binding['normalization_c']:.8g}；选择 `{binding['selected_run_id']}` "
        f"(search round {chosen_selected_round}) 的 epoch {binding['selected_epoch']}。",
        f"Final PBS job `{final_summary['job_id']}`；校准覆盖 {final_summary['calibration_context_count']} 个 context，"
        f"test 覆盖 {final_summary['test_context_count']} 个 context，student test forward {final_summary['test_forward_count']} 次；"
        "analysis 自身新增 model/teacher/parent-test forward 均为 0。", "",
        "## 两轮 selection 与 guard", "",
        f"冻结策略：三个 guardrail 相对未训练 micro parent 的允许上限为 +{choice['guardrail_fraction']:.0%}；"
        f"near-tie 窗口为 {choice['near_tie_fraction']:.0%}。表中 rho/candidate 各列出 epoch 0/1/2；"
        "guard flags 覆盖 late-22 motor、first-10 gripper 和 first-10 motor P90。", "",
        *_selection_markdown(points), "",
        f"严格 eligible 最小值：`{strict['run_id']}` epoch {strict['epoch']}，"
        f"rho={strict['rho']:.8g}，first-10 motor RMSE={strict_metrics['mean_motor_rmse_first10']:.8f}。",
        f"Near-tie winner：`{choice['selected_run_id']}` epoch {choice['selected_epoch']}，"
        f"rho={binding['rho']:.8g}，first-10 motor RMSE={selected['mean_motor_rmse_first10']:.8f}，"
        f"eligible={next(p['eligible'] for p in points if p['run_id']==choice['selected_run_id'] and p['epoch']==choice['selected_epoch'])}。",
    ]
    if choice.get("epoch0_fallback"):
        report += ["", "该 winner 是 epoch 0 fallback；不能将它解释为正 lambda 训练带来的改善。"]
    report += ["", "## Initial 与 selected action metrics", ""]
    report += _action_markdown("Selection（4 cases / 8 contexts）", action_comparisons["initial_to_selected_selection"])
    report += [""]
    report += _action_markdown("Calibration（8 cases / 8 contexts）", action_comparisons["initial_to_selected_calibration"])
    report += [""]
    report += _action_markdown("Test：已缓存 micro parent 对 selected final（22 cases / 44 contexts）",
                               action_comparisons["micro_parent_vs_selected_test"])
    report += ["", "Test 的四个 action RMSE 按 seed 先在 case 内平均，再对 22 个 case 等权平均。"
               "负的 selected − baseline 表示 selected RMSE 更低。原 micro summary 的 first-10、gripper、full-32 aggregate "
               f"已由 44 个缓存 action 复算并核对（summary job `{micro_summary['job_id']}`）。", "",
               "## Selected future 对 BF16 teacher", "",
               "下列为 selected final 在 test44 上相对缓存 BF16 teacher 的比较；协议没有 micro parent test future，"
               "因此没有生成或推断该对照。", "",
               "| Metric | Case-balanced value |", "|---|---:|"]
    future_metrics = future["metrics"]
    for key in FUTURE_SCALARS:
        report.append(f"| {key} | {future_metrics[key]:.8f} |")
    for key in FUTURE_VECTORS:
        report.append(f"| {key} (3 latent channels) | {', '.join(f'{value:.8f}' for value in future_metrics[key])} |")
    report += ["", "## Evidence boundary", "",
               "Selection 只有历史 4 cases / 8 contexts；test 为原有 22 cases / 44 contexts，结果已见，"
               "本次属于配对 recovery diagnostic，不是新的独立泛化评估。没有 complementary/interactions sweep，"
               "没有额外 teacher forward、parent test-future forward、闭环成功率或 native low-bit kernel 证据。",
               "所有 action 指标基于已缓存动作；future 指标来自 final selected forward 对 BF16 teacher。"]
    return "\n".join(report)


def run(args):
    guard()
    self_check()
    print("FINAL_ANALYSIS_SELF_CHECK_PASSED", flush=True)
    import torch
    import metrics

    require(not torch.cuda.is_available(), "Final analysis was expected to run on CPU only")
    protocol = read_json(args.protocol)
    choice = read_json(args.choice)
    final_summary = read_json(args.final_summary)
    require(protocol.get("protocol") == PROTOCOL, "Unexpected experiment protocol")
    root, binding = validate_choice_binding(protocol, choice, final_summary,
                                             args.choice, args.final_summary)
    cases_cal = protocol["calibration_cases"]
    cases_selection = protocol["selection_cases"]
    cases_test = protocol["test_cases"]
    seeds_eval = protocol["evaluation_seed_indices"]
    require((len(cases_cal), len(cases_selection), len(cases_test), len(seeds_eval)) == (8, 4, 22, 2),
            "Protocol cohort counts differ from the approved 8/4/22 by two-seed design")
    require(protocol.get("reference_cache") == str(root / "reference_cache.pt")
            and protocol.get("reference_summary") == str(root / "reference_summary.json"),
            "Protocol reference files do not resolve under ROOT")
    ref_summary = read_json(protocol["reference_summary"])
    teacher = validate_teacher_cache(torch, protocol, ref_summary, protocol["reference_cache"])

    micro_parent, micro_summary = load_micro_test_actions(torch, metrics, protocol, teacher)
    final_test = validate_action_aggregate(final_summary.get("test"), cases_test, 44, "selected test")
    initial_test = validate_action_aggregate(final_summary.get("initial_test"), cases_test, 44,
                                             "initial micro-parent test")
    for case in cases_test:
        for metric in ACTION_KEYS:
            close(initial_test["per_case"][str(case)][metric],
                  micro_parent["per_case"][str(case)][metric],
                  f"initial test/micro parent per-case {case}/{metric}")
    for aggregate_key in ("mean_motor_rmse_first10", "mean_gripper_rmse_first10",
                          "mean_motor_rmse_32", "mean_motor_rmse_late22"):
        close(initial_test[aggregate_key], micro_parent[aggregate_key],
              f"initial test/micro parent {aggregate_key}")

    selected_search, selected_row = validate_selected_search(protocol, choice, final_summary, cases_selection)
    selection_cases = cases_selection
    initial_selection = validate_action_aggregate(final_summary.get("initial_selection"),
                                                  selection_cases, 8, "initial selection")
    selected_selection = validate_action_aggregate(final_summary.get("selection"),
                                                   selection_cases, 8, "selected selection")
    initial_calibration = validate_action_aggregate(
        final_summary.get("initial_calibration", {}).get("action"), cases_cal, 8, "initial calibration")
    selected_calibration = validate_action_aggregate(final_summary.get("calibration"),
                                                     cases_cal, 8, "selected calibration")
    selected_future = validate_future_aggregate(final_summary.get("test_future_metrics"),
                                               cases_test, 44, "selected test-vs-BF16")
    initial_selection_future = validate_future_aggregate(
        final_summary.get("initial_selection_future_metrics"), selection_cases, 8,
        "initial selection-vs-BF16")
    initial_calibration_future = validate_future_aggregate(
        final_summary.get("initial_calibration", {}).get("future_metrics"), cases_cal, 8,
        "initial calibration-vs-BF16")
    selected_calibration_future = validate_future_aggregate(
        final_summary.get("calibration_future_metrics"), cases_cal, 8,
        "selected calibration-vs-BF16")
    selected_selection_future = validate_future_aggregate(
        selected_row.get("future_metrics"), selection_cases, 8,
        "selected selection-vs-BF16")

    action_comparisons = {
        "initial_to_selected_selection": compare_actions(initial_selection, selected_selection,
                                                         selection_cases, "selection"),
        "initial_to_selected_calibration": compare_actions(initial_calibration, selected_calibration,
                                                           cases_cal, "calibration"),
        "micro_parent_vs_selected_test": compare_actions(micro_parent, final_test, cases_test,
                                                         "test micro-parent vs selected"),
    }
    sorted_points, chosen_point, strict_point = selection_table(choice, protocol)
    require(chosen_point["run_id"] == binding["selected_run_id"]
            and chosen_point["epoch"] == binding["selected_epoch"],
            "Winner metadata differs between choice and selection table")
    require(strict_point["metrics"]["mean_motor_rmse_first10"]
            == choice["strict_eligible_minimum"]["metrics"]["mean_motor_rmse_first10"],
            "Strict eligible minimum metadata is inconsistent")

    job_id = os.environ["PBS_JOBID"]
    out = args.out.resolve()
    require(out == (root / "results" / f"analysis-{job_id}").resolve(),
            "Analysis output must be ROOT/results/analysis-JOB")
    require(not out.exists(), f"Analysis output already exists: {out}")
    out.mkdir(parents=True)
    comparison = {
        "protocol": PROTOCOL, "analysis_job_id": job_id,
        "final_job_id": final_summary["job_id"], "choice_job_id": choice["selection_job_id"],
        "selected_binding": binding,
        "selection_points": sorted_points,
        "strict_eligible_minimum": strict_point,
        "near_tie_winner": chosen_point,
        "guardrail_baseline": choice["baseline"],
        "action_comparisons": action_comparisons,
        "selected_future_vs_bf16_test44": selected_future,
        "initial_selection_future_vs_bf16": initial_selection_future,
        "selected_selection_future_vs_bf16": selected_selection_future,
        "initial_calibration_future_vs_bf16": initial_calibration_future,
        "selected_calibration_future_vs_bf16": selected_calibration_future,
        "micro_parent_test_recomputed": micro_parent,
        "micro_parent_original_summary_metrics": {
            "motor_rmse_first10_vs_bf16": micro_summary["test"]["motor_rmse_first10_vs_bf16"],
            "gripper_rmse_first10_vs_bf16": micro_summary["test"]["gripper_rmse_first10_vs_bf16"],
            "motor_rmse_32_vs_bf16": micro_summary["test"]["motor_rmse_32_vs_bf16"],
        },
        "selection_data_boundary": {
            "case_count": len(cases_selection), "context_count": 8,
            "test_case_count": len(cases_test), "test_context_count": 44,
            "test_previously_seen": True,
            "reuse_boundary": protocol["reuse_boundary"],
        },
        "new_analysis_forwards": {"model": 0, "teacher": 0, "micro_parent_test_future": 0},
    }
    report = make_report(protocol, choice, final_summary, binding, sorted_points, strict_point,
                         action_comparisons, micro_summary, selected_future, selected_search)
    write_json(out / "comparison.json", comparison)
    write_text(out / "report.zh.md", report)
    summary = {
        "status": "complete", "protocol": PROTOCOL, "job_id": job_id,
        "final_job_id": final_summary["job_id"], "choice_job_id": choice["selection_job_id"],
        "selected_run_id": binding["selected_run_id"], "selected_job_id": binding["selected_job_id"],
        "selected_epoch": binding["selected_epoch"], "selected_round": binding["selected_round"],
        "rho": binding["rho"], "latent_weight": binding["latent_weight"],
        "normalization_c": binding["normalization_c"], "selection_point_count": len(sorted_points),
        "calibration_context_count": 8, "selection_context_count": 8,
        "test_context_count": 44, "test_previously_seen": True,
        "model_forward_count": 0, "teacher_forward_count": 0,
        "parent_test_future_forward_count": 0,
        "artifacts": {"summary": str(out / "summary.json"),
                      "report": str(out / "report.zh.md"),
                      "comparison": str(out / "comparison.json")},
    }
    write_json(out / "summary.json", summary)
    (out / "SEARCH_REPORT_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("SEARCH_REPORT_COMPLETE " + json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--final-summary", type=Path, required=True)
    parser.add_argument("--choice", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
