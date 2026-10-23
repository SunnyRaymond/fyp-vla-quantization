"""Verify and summarize the paired C8/C80 fixed-input calibration comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics


PROTOCOL = "fastwam-vq-calibration-suites-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
METRICS = ("motor_rmse_first10_vs_bf16", "gripper_rmse_first10_vs_bf16",
           "motor_rmse_32_vs_bf16")
METRIC_LABELS = {
    "motor_rmse_first10_vs_bf16": "前10步 6维 motor RMSE",
    "gripper_rmse_first10_vs_bf16": "前10步 gripper RMSE",
    "motor_rmse_32_vs_bf16": "32步 6维 motor RMSE",
}
MAX_SUMMARY_BYTES = 256 * 1024
MAX_INPUT_BYTES = 1024 * 1024
TIE_TOLERANCE = 1e-12
JOB_ID = re.compile(r"^\d+(?:\[\d*\])?\.[A-Za-z0-9_.-]+$")


def require_compute_allocation() -> None:
    job_id, nodefile = os.environ.get("PBS_JOBID"), os.environ.get("PBS_NODEFILE")
    if not job_id or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("Comparison requires a genuine PBS compute allocation")
    host = socket.gethostname().split(".", 1)[0]
    nodes = {line.split(".", 1)[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if "login" in host.lower() or host not in nodes:
        raise RuntimeError(f"Comparison requires an allocated compute node; host={host}")


def read_json(path: Path, limit: int = MAX_INPUT_BYTES) -> dict:
    size = path.stat().st_size
    if size > limit:
        raise RuntimeError(f"Refusing oversized JSON ({size} bytes): {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if temporary.stat().st_size >= MAX_SUMMARY_BYTES:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Comparison JSON exceeds the 256 KiB control-file limit: {path}")
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    if temporary.stat().st_size >= MAX_SUMMARY_BYTES:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Comparison report exceeds the 256 KiB control-file limit: {path}")
    os.replace(temporary, path)


def validate_protocol(protocol: dict, root: Path) -> None:
    if (protocol.get("protocol") != PROTOCOL or protocol.get("remote_root") != str(root)
            or tuple(protocol.get("suites", ())) != SUITES
            or protocol.get("target_linears") != 614):
        raise RuntimeError("Frozen protocol identity, suites, root, or target layer count differs")
    transform = protocol.get("transform", {})
    if (transform.get("name"), transform.get("alpha"), transform.get("hadamard_block")) != (
            "smooth05_hadamard", 0.5, 128):
        raise RuntimeError("The comparison requires the frozen Smooth0.5/H128 transforms")
    vq = protocol.get("vq", {})
    if (vq.get("dimension"), vq.get("codebooks"), vq.get("entries"), vq.get("books_dtype"),
            vq.get("row_scale_dtype"), vq.get("samples"), vq.get("iterations"),
            vq.get("seed"), vq.get("local_tuning_steps"), vq.get("action_lambda")) != (
            4, 2, 256, "bfloat16", "bfloat16", 8192, 8, 20261006, 0, 0):
        raise RuntimeError("Unexpected VQ recipe in the frozen comparison protocol")
    if protocol.get("microscaling", {}).get("scale_block") != 32:
        raise RuntimeError("The comparison requires the frozen B32 microscaling format")
    evaluation = protocol.get("evaluation", {})
    if (evaluation.get("case_count"), evaluation.get("cases_per_suite"),
            tuple(evaluation.get("seed_indices", ())), evaluation.get("query_count_per_cohort"),
            evaluation.get("domain"), evaluation.get("episodes"),
            evaluation.get("predicted_actions_executed")) != (40, 10, (0, 1), 80, "original", 0, 0):
        raise RuntimeError("The frozen 40-case paired evaluation differs")
    if (evaluation.get("primary_metric") !=
            "first10 x6 motor RMSE versus same-input same-seed BF16; average seeds within case, cases within suite, then four suites equally"
            or tuple(evaluation.get("secondary_metrics", ())) != (
                "gripper_rmse_first10_vs_bf16", "motor_rmse_32_vs_bf16")):
        raise RuntimeError("The frozen primary/secondary RMSE metrics differ")
    if ((protocol.get("calibration_cohorts", {}).get("c8", {}).get("contexts"),
         protocol.get("calibration_cohorts", {}).get("c80", {}).get("contexts")) != (8, 80)):
        raise RuntimeError("The frozen C8/C80 context counts differ")


def validate_plan(root: Path, protocol: dict) -> tuple[dict, dict[int, dict], dict[str, list[int]]]:
    plan = read_json(root / protocol["data"]["plan_file"])
    rows = plan.get("inputs")
    expected_ids = list(range(2000, 2080)) + list(range(3000, 3040))
    if (plan.get("status") != "complete" or plan.get("protocol") != PROTOCOL
            or not isinstance(rows, list) or len(rows) != 120
            or [int(row.get("case_id", -1)) for row in rows] != expected_ids
            or (plan.get("input_count"), plan.get("calibration_count"), plan.get("test_count"))
            != (120, 80, 40)):
        raise RuntimeError("The frozen 80-calibration/40-test input plan is incomplete")
    if plan.get("suite_order") != list(SUITES):
        raise RuntimeError("The frozen plan suite order differs")
    by_id = {int(row["case_id"]): row for row in rows}
    cohorts = plan.get("calibration_cohorts", {})
    c8, c80 = cohorts.get("c8"), cohorts.get("c80")
    if (not isinstance(c8, list) or not isinstance(c80, list)
            or any(type(value) is not int for value in c8 + c80)
            or len(c8) != 8 or len(set(c8)) != 8 or c8 != sorted(c8)
            or c80 != list(range(2000, 2080)) or not set(c8) <= set(c80)):
        raise RuntimeError("The frozen nested C8/C80 membership is malformed")
    cohort_ids = {"c8": c8, "c80": c80}
    for row in rows:
        if (row.get("domain") != "original" or row.get("suite") not in SUITES
                or not isinstance(row.get("sampler_seeds"), list)
                or len(row["sampler_seeds"]) != 2
                or any(type(seed) is not int for seed in row["sampler_seeds"])
                or row["sampler_seeds"][0] == row["sampler_seeds"][1]):
            raise RuntimeError(f"Malformed original-domain input identity: {row.get('case_id')}")
    calibration = [by_id[case] for case in range(2000, 2080)]
    test = [by_id[case] for case in range(3000, 3040)]
    for suite in SUITES:
        suite_cal = [row for row in calibration if row["suite"] == suite]
        suite_test = [row for row in test if row["suite"] == suite]
        selected = [row for row in suite_cal if int(row["case_id"]) in set(c8)]
        if (len(suite_cal), len(suite_test), len(selected),
                len({int(row["task_id"]) for row in selected})) != (20, 10, 2, 2):
            raise RuntimeError(f"Frozen per-suite context/task coverage differs for {suite}")
        if {int(row["task_id"]) for row in suite_test} != set(range(10)):
            raise RuntimeError(f"Frozen held-out test task coverage differs for {suite}")
        for task in range(10):
            task_cal = [row for row in suite_cal if int(row["task_id"]) == task]
            task_test = [row for row in suite_test if int(row["task_id"]) == task]
            if (len(task_cal), len({int(row["demo_id"]) for row in task_cal}), len(task_test)) != (2, 2, 1):
                raise RuntimeError(f"C80/test demo coverage differs for {suite}/task {task}")
            if int(task_test[0]["demo_id"]) in {int(row["demo_id"]) for row in task_cal}:
                raise RuntimeError(f"Held-out demonstration overlaps calibration for {suite}/task {task}")
        if [sum(row["phase"] == phase for row in suite_cal)
                for phase in ("early", "middle", "late")] != [7, 7, 6]:
            raise RuntimeError(f"Frozen C80 phase coverage differs for {suite}")
    return plan, by_id, cohort_ids


def verify_job_artifact(root: Path, summary: dict, cohort: str) -> None:
    job_id = summary.get("pbs_jobid")
    if not isinstance(job_id, str) or not JOB_ID.fullmatch(job_id):
        raise RuntimeError(f"{cohort} summary has no valid PBS job identity")
    artifact = root / "artifacts" / job_id
    if ((artifact / "PIPELINE_COMPLETE").is_file() is False
            or (artifact / "PIPELINE_CHECKPOINTED").exists()
            or not (artifact / "exit_code.txt").is_file()
            or (artifact / "exit_code.txt").read_text(encoding="utf-8").strip() != "0"):
        raise RuntimeError(f"{cohort} does not have a successful complete artifact receipt")
    receipt = read_json(artifact / "summary.json", MAX_SUMMARY_BYTES)
    if receipt != summary:
        raise RuntimeError(f"{cohort} results/artifact summaries disagree")


def verify_common_summary(summary: dict, protocol: dict, cohort: str,
                          expected_case_ids: list[int]) -> None:
    if (summary.get("status") != "complete" or summary.get("protocol") != PROTOCOL
            or summary.get("cohort") != cohort or summary.get("case_ids") != expected_case_ids
            or (summary.get("case_count"), summary.get("query_count")) != (40, 80)
            or summary.get("target_linears") != 614
            or summary.get("queries_completed") != 80 or summary.get("expected_queries") != 80
            or summary.get("episodes") != 0 or summary.get("predicted_actions_executed") != 0
            or summary.get("real_quant") is not False
            or summary.get("native_low_bit_kernel") is not False
            or summary.get("source_full_job") != protocol["source_full_job"]
            or summary.get("source_micro_full_job") != protocol["source_micro_full_job"]):
        raise RuntimeError(f"{cohort} summary identity, boundary, or evaluation coverage differs")
    if "A100" not in str(summary.get("device", "")):
        raise RuntimeError(f"{cohort} summary does not record the expected A100 device")
    identity = summary.get("model_identity")
    if not isinstance(identity, dict) or identity.get("mode") != "idm":
        raise RuntimeError(f"{cohort} summary lacks the shared IDM model identity")


def load_summary(root: Path, protocol: dict, cohort: str) -> dict:
    path = root / "results" / cohort / "summary.json"
    summary = read_json(path, MAX_SUMMARY_BYTES)
    verify_job_artifact(root, summary, cohort)
    verify_common_summary(summary, protocol, cohort, list(range(3000, 3040)))
    return summary


def finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"Expected a finite number for {label}")
    return float(value)


def checked_action(value, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 32:
        raise RuntimeError(f"Malformed [32,7] action in {label}")
    result = []
    for step in value:
        if not isinstance(step, list) or len(step) != 7:
            raise RuntimeError(f"Malformed [32,7] action in {label}")
        result.append([finite_number(component, label) for component in step])
    return result


def action_metrics(action: list[list[float]], reference: list[list[float]]) -> dict[str, float]:
    delta = [[action[t][d] - reference[t][d] for d in range(7)] for t in range(32)]
    first10_motor = [delta[t][d] ** 2 for t in range(10) for d in range(6)]
    first10_gripper = [delta[t][6] ** 2 for t in range(10)]
    motor32 = [delta[t][d] ** 2 for t in range(32) for d in range(6)]
    return {
        "motor_rmse_first10_vs_bf16": math.sqrt(statistics.mean(first10_motor)),
        "gripper_rmse_first10_vs_bf16": math.sqrt(statistics.mean(first10_gripper)),
        "motor_rmse_32_vs_bf16": math.sqrt(statistics.mean(motor32)),
    }


def check_saved_metrics(record: dict, calculated: dict[str, float], label: str) -> None:
    for metric in METRICS:
        saved = finite_number(record.get(metric), f"{label}/{metric}")
        if not math.isclose(saved, calculated[metric], rel_tol=2e-5, abs_tol=5e-8):
            raise RuntimeError(f"Saved {metric} disagrees with paired actions in {label}")


def check_arm_summary_metrics(summary: dict, per_case: dict[int, dict],
                              per_suite: dict[str, dict], suite_macro: dict[str, float]) -> None:
    test = summary.get("test", {})
    stored_cases = test.get("per_case", {})
    if set(stored_cases) != {str(case) for case in per_case}:
        raise RuntimeError(f"{summary.get('cohort')} summary case aggregate keys differ")
    if set(test.get("per_suite", {})) != set(SUITES):
        raise RuntimeError(f"{summary.get('cohort')} summary suite aggregate keys differ")
    if set(test.get("suite_macro", {})) < set(METRICS):
        raise RuntimeError(f"{summary.get('cohort')} summary lacks suite-macro metrics")
    for case, metrics in per_case.items():
        for metric in METRICS:
            value = finite_number(stored_cases[str(case)].get(metric), f"summary/{case}/{metric}")
            if not math.isclose(value, metrics[metric], rel_tol=2e-5, abs_tol=5e-8):
                raise RuntimeError(f"{summary.get('cohort')} per-case summary differs at {case}/{metric}")
    for suite in SUITES:
        for metric in METRICS:
            value = finite_number(test["per_suite"][suite].get(metric), f"summary/{suite}/{metric}")
            if not math.isclose(value, per_suite[suite][metric], rel_tol=2e-5, abs_tol=5e-8):
                raise RuntimeError(f"{summary.get('cohort')} per-suite summary differs at {suite}/{metric}")
    for metric in METRICS:
        value = finite_number(test["suite_macro"].get(metric), f"summary/suite_macro/{metric}")
        if not math.isclose(value, suite_macro[metric], rel_tol=2e-5, abs_tol=5e-8):
            raise RuntimeError(f"{summary.get('cohort')} suite-macro summary differs for {metric}")


def load_queries(root: Path, protocol: dict, plan_rows: dict[int, dict],
                 teacher_summary: dict, arm_summary: dict, cohort: str):
    teacher_records: dict[tuple[int, int], dict] = {}
    arm_records: dict[tuple[int, int], dict] = {}
    arm_metrics: dict[tuple[int, int], dict] = {}
    teacher_root = root / "results" / "teacher" / "query_records"
    arm_root = root / "results" / cohort / "query_records"
    identity = teacher_summary["model_identity"]
    if arm_summary["model_identity"] != identity:
        raise RuntimeError(f"{cohort} model identity differs from the common teacher")
    for case in range(3000, 3040):
        row = plan_rows[case]
        for seed_index, sampler_seed in enumerate(row["sampler_seeds"]):
            key = case, seed_index
            teacher = read_json(teacher_root / f"test_{case}_{seed_index}.json", MAX_SUMMARY_BYTES)
            arm = read_json(arm_root / f"test_{case}_{seed_index}.json", MAX_SUMMARY_BYTES)
            common = (case, seed_index, int(sampler_seed), row["suite"], int(row["task_id"]), "test")
            observed_teacher = (teacher.get("case_id"), teacher.get("seed_index"),
                                teacher.get("sampler_seed"), teacher.get("suite"),
                                teacher.get("task_id"), teacher.get("split"))
            if (teacher.get("protocol") != PROTOCOL or observed_teacher != common
                    or teacher.get("arm") != "bf16" or teacher.get("observed_module_count") != 614
                    or teacher.get("model_identity") != identity):
                raise RuntimeError(f"Teacher query identity/coverage differs for {case}/{seed_index}")
            observed_arm = (arm.get("case_id"), arm.get("seed_index"), arm.get("sampler_seed"),
                            arm.get("suite"), arm.get("task_id"), arm.get("split"))
            if (arm.get("protocol") != PROTOCOL or arm.get("cohort") != cohort
                    or observed_arm != common or arm.get("arm") != "micro_scale_vq_a4"
                    or arm.get("observed_module_count") != 614
                    or arm.get("teacher_source_full_job") != protocol["source_full_job"]
                    or arm.get("teacher_model_identity") != identity):
                raise RuntimeError(f"{cohort} query identity/teacher pairing differs for {case}/{seed_index}")
            teacher_action = checked_action(teacher.get("action"), f"teacher {case}/{seed_index}")
            arm_action = checked_action(arm.get("action"), f"{cohort} {case}/{seed_index}")
            recomputed = action_metrics(arm_action, teacher_action)
            check_saved_metrics(arm, recomputed, f"{cohort}/{case}/{seed_index}")
            saved_metrics = {metric: finite_number(arm.get(metric), f"{cohort}/{case}/{seed_index}/{metric}")
                             for metric in METRICS}
            teacher_records[key], arm_records[key], arm_metrics[key] = teacher, arm, saved_metrics
    if len(teacher_records) != 80 or len(arm_records) != 80:
        raise RuntimeError(f"{cohort} paired denominator is not 40 cases × 2 seeds")
    return teacher_records, arm_records, arm_metrics


def summarize_arm(plan_rows: dict[int, dict], query_metrics: dict[tuple[int, int], dict[str, float]],
                  summary: dict) -> dict:
    per_case = {}
    per_suite = {}
    for case in range(3000, 3040):
        seeds = sorted(seed for found_case, seed in query_metrics if found_case == case)
        if seeds != [0, 1]:
            raise RuntimeError(f"Expected seed indices [0,1] for test case {case}")
        per_case[case] = {metric: statistics.mean(query_metrics[case, seed][metric] for seed in seeds)
                          for metric in METRICS}
    for suite in SUITES:
        cases = [case for case in range(3000, 3040) if plan_rows[case]["suite"] == suite]
        if len(cases) != 10 or len(cases) * 2 != 20:
            raise RuntimeError(f"Expected 10 cases × 2 seeds for {suite}")
        per_suite[suite] = {metric: statistics.mean(per_case[case][metric] for case in cases)
                            for metric in METRICS}
    suite_macro = {metric: statistics.mean(per_suite[suite][metric] for suite in SUITES)
                   for metric in METRICS}
    check_arm_summary_metrics(summary, per_case, per_suite, suite_macro)
    return {"per_case": {str(case): per_case[case] for case in sorted(per_case)},
            "per_suite": per_suite, "suite_macro": suite_macro}


def delta_record(c8: float, c80: float) -> dict:
    delta = c80 - c8
    return {"c8": c8, "c80": c80, "delta_c80_minus_c8": delta,
            "relative_change_fraction": (delta / c8 if c8 != 0 else None)}


def compare_arms(plan_rows: dict[int, dict], c8: dict, c80: dict) -> dict:
    outcomes = {}
    per_case = {}
    for metric in METRICS:
        wins = ties = losses = 0
        for case in range(3000, 3040):
            a = c8["per_case"][str(case)][metric]
            b = c80["per_case"][str(case)][metric]
            d = b - a
            if d < -TIE_TOLERANCE:
                wins += 1
            elif d > TIE_TOLERANCE:
                losses += 1
            else:
                ties += 1
            per_case.setdefault(str(case), {"suite": plan_rows[case]["suite"], "query_count": 2,
                                            "metrics": {}})
            per_case[str(case)]["metrics"][metric] = delta_record(a, b)
        outcomes[metric] = {"c80_wins": wins, "ties": ties, "c80_losses": losses,
                            "cases": 40, "tie_tolerance_absolute": TIE_TOLERANCE}

    per_suite = {}
    for suite in SUITES:
        per_suite[suite] = {"case_count": 10, "query_count": 20,
                            "metrics": {metric: delta_record(c8["per_suite"][suite][metric],
                                                              c80["per_suite"][suite][metric])
                                        for metric in METRICS}}
    suite_macro = {metric: delta_record(c8["suite_macro"][metric], c80["suite_macro"][metric])
                   for metric in METRICS}
    return {"delta_definition": "C80 minus C8; negative RMSE change favors C80",
            "per_case": per_case, "wins_ties_losses": outcomes,
            "per_suite": per_suite, "suite_macro": suite_macro}


def query_timing(arms: dict[str, dict[tuple[int, int], dict]]) -> dict | None:
    values = {}
    for arm, records in arms.items():
        times = []
        for record in records.values():
            value = record.get("query_wall_seconds")
            if value is None:
                return None
            seconds = finite_number(value, f"{arm}/query_wall_seconds")
            if seconds <= 0:
                return None
            times.append(seconds)
        if len(times) != 80:
            return None
        values[arm] = {"query_count": len(times), "median_seconds": statistics.median(times),
                       "mean_seconds": statistics.mean(times)}
    return {"scope": "per-query input load/datum preparation plus decoded-BF16 model inference",
            "interpretation": "descriptive only; not a benchmark or latency claim; no native low-bit kernel",
            "cohorts": values}


def focus_gripper_diagnostic(teacher_records: dict[tuple[int, int], dict],
                             arm_records: dict[str, dict[tuple[int, int], dict]],
                             paired: dict) -> dict:
    cases = (3027, 3029, 3031)
    per_query = []
    for case in cases:
        for seed_index in (0, 1):
            key = case, seed_index
            teacher_action = teacher_records[key]["action"]
            row = {"case_id": case, "seed_index": seed_index,
                   "sampler_seed": teacher_records[key]["sampler_seed"]}
            for cohort in ("c8", "c80"):
                record = arm_records[cohort][key]
                differences = [abs(record["action"][step][6] - teacher_action[step][6])
                               for step in range(10)]
                maximum = max(range(10), key=differences.__getitem__)
                row[cohort] = {
                    "gripper_rmse_first10_vs_bf16": finite_number(
                        record["gripper_rmse_first10_vs_bf16"],
                        f"{cohort}/{case}/{seed_index}/gripper_rmse_first10_vs_bf16"),
                    "max_abs_gripper_difference_first10": differences[maximum],
                    "max_difference_step_index": maximum,
                }
            per_query.append(row)

    two_seed_case_means = {}
    for case in cases:
        c8 = paired["per_case"][str(case)]["metrics"]["gripper_rmse_first10_vs_bf16"]["c8"]
        c80 = paired["per_case"][str(case)]["metrics"]["gripper_rmse_first10_vs_bf16"]["c80"]
        two_seed_case_means[str(case)] = {
            "query_count_per_cohort": 2, "c8": c8, "c80": c80,
            "delta_c80_minus_c8": c80 - c8,
        }
    c8_mean = statistics.mean(item["c8"] for item in two_seed_case_means.values())
    c80_mean = statistics.mean(item["c80"] for item in two_seed_case_means.values())
    return {
        "cases": list(cases),
        "selection_basis": "Cases were fixed from preliminary C8 findings before this comparison; diagnostic only, not method selection.",
        "metric": "Saved first10 gripper RMSE versus the same-input, same-seed BF16 teacher.",
        "maximum_difference_definition": "Maximum absolute gripper output difference over the first 10 normalized output steps; step index is zero-based.",
        "step_interpretation": "A normalized output difference only; it does not establish an executed action or closed-loop failure.",
        "per_query": per_query,
        "two_seed_case_means": two_seed_case_means,
        "three_case_mean": {
            "case_count": 3, "query_count_per_cohort": 6,
            "c8": c8_mean, "c80": c80_mean,
            "delta_c80_minus_c8": c80_mean - c8_mean,
            "aggregation": "Equal mean of the three per-case means; each case mean averages its two seeds.",
        },
    }


def format_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.8g}"


def build_report(result: dict) -> str:
    paired = result["paired_comparison"]
    lines = [
        "# C8/C80 calibration-cohort 配对比较",
        "",
        "共同 BF16 teacher 下，在同一 original-demo 测试输入与相同 sampler seed 上比较 decoded-BF16 Microscaling VQ。每个 cohort 覆盖 4 suites × 10 cases × 2 seeds，即每 cohort 80 个 query、614 个 target Linear。RMSE 越低越好；下表变化定义为 C80 − C8。",
        "",
        "## Calibration 覆盖",
        "",
        "| Cohort | 每 suite contexts | 总 contexts | 覆盖方式 |",
        "| --- | ---: | ---: | --- |",
        "| C8 | 2 | 8 | 每 suite 来自 2 个不同 tasks；嵌套于 C80 |",
        "| C80 | 20 | 80 | 覆盖 40 tasks；每 task 2 个不同 demonstrations |",
        "",
        "C80 每 suite 固定为 7 early、7 middle、6 late snapshots。",
        "",
        "## 四套件 macro 指标",
        "",
        "| 指标（主指标/次指标） | C8 | C80 | 绝对变化 | 相对变化 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        row = paired["suite_macro"][metric]
        relative = row["relative_change_fraction"]
        role = "主指标" if metric == METRICS[0] else "次指标"
        lines.append(f"| {role}：{METRIC_LABELS[metric]} | {format_value(row['c8'])} | {format_value(row['c80'])} | {format_value(row['delta_c80_minus_c8'])} | {format_value(None if relative is None else relative * 100)}% |")
    lines.extend(["", "相对变化栏以百分比表示；负的 RMSE 变化代表 C80 误差较低。", "",
                  "## 每 suite 主指标", "",
                  "| Suite | Cases | Queries | C8 motor RMSE | C80 motor RMSE | 绝对变化 | 相对变化 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for suite in SUITES:
        item = paired["per_suite"][suite]
        metric = item["metrics"]["motor_rmse_first10_vs_bf16"]
        relative = metric["relative_change_fraction"]
        lines.append(f"| {suite} | 10 | 20 | {format_value(metric['c8'])} | {format_value(metric['c80'])} | {format_value(metric['delta_c80_minus_c8'])} | {format_value(None if relative is None else relative * 100)}% |")
    lines.extend(["", "## Paired per-case outcomes", "",
                  "胜负以每 case 的两个 seed 平均 RMSE 判定，tie tolerance 为绝对差 1e-12。", "",
                  "| 指标 | C80 误差较低 | Tie | C80 误差较高 |",
                  "| --- | ---: | ---: | ---: |"])
    for metric in METRICS:
        item = paired["wins_ties_losses"][metric]
        lines.append(f"| {METRIC_LABELS[metric]} | {item['c80_wins']} | {item['ties']} | {item['c80_losses']} |")
    lines.extend(["", "每个 case 的逐项 C8/C80 值与 delta 保存在 `comparison.json`。", "",
                  "## 预先固定 cases 的 gripper 配对诊断", "",
                  "以下 3027、3029、3031 三个 case 根据 C8 初步结果预先固定，只作诊断，不用于选择 quantization 方法。最大差异的 step 是归一化输出中的 0-based 索引，不代表已执行动作或 closed-loop 失败。", "",
                  "| Case | Seed | Sampler seed | C8 first10 RMSE | C8 最大绝对差 @ step | C80 first10 RMSE | C80 最大绝对差 @ step |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    focus = result["focus_gripper"]
    for item in focus["per_query"]:
        c8, c80 = item["c8"], item["c80"]
        lines.append(
            f"| {item['case_id']} | {item['seed_index']} | {item['sampler_seed']} | "
            f"{format_value(c8['gripper_rmse_first10_vs_bf16'])} | "
            f"{format_value(c8['max_abs_gripper_difference_first10'])} @ {c8['max_difference_step_index']} | "
            f"{format_value(c80['gripper_rmse_first10_vs_bf16'])} | "
            f"{format_value(c80['max_abs_gripper_difference_first10'])} @ {c80['max_difference_step_index']} |")
    lines.extend(["", "| Case（两 seed 均值） | C8 | C80 | C80 − C8 |",
                  "| ---: | ---: | ---: | ---: |"])
    for case, item in focus["two_seed_case_means"].items():
        lines.append(f"| {case} | {format_value(item['c8'])} | {format_value(item['c80'])} | {format_value(item['delta_c80_minus_c8'])} |")
    overall = focus["three_case_mean"]
    lines.append(f"| 三 case 等权均值 | {format_value(overall['c8'])} | {format_value(overall['c80'])} | {format_value(overall['delta_c80_minus_c8'])} |")
    lines.extend(["",
                  "## 解读范围", "",
                  "C8 与 C80 同时改变 calibration context 数量、task 覆盖与 phase 覆盖，因此结果比较的是两种预先固定的 calibration recipe，不能分离为单一数量因果效应。Smooth0.5/H128 transforms、VQ 设置与测试 cohort 均固定。这是 fixed-input decoded-BF16/A4 数值比较；不主张统计显著性、closed-loop 成功率、native low-bit kernel 或推理速度。"])
    timing = result.get("query_timing_diagnostic")
    if timing:
        lines.extend(["", "## 描述性 per-query wall time", "",
                      "下列时间包含输入加载/数据准备与 decoded-BF16 forward，仅作记录，不是 benchmark 或 latency claim。", "",
                      "| Cohort | Queries | Median seconds | Mean seconds |",
                      "| --- | ---: | ---: | ---: |"])
        for cohort in ("teacher", "c8", "c80"):
            item = timing["cohorts"][cohort]
            lines.append(f"| {cohort} | {item['query_count']} | {format_value(item['median_seconds'])} | {format_value(item['mean_seconds'])} |")
    return "\n".join(lines) + "\n"


def compare(root: Path, protocol_path: Path, out: Path) -> dict:
    protocol = read_json(protocol_path, MAX_SUMMARY_BYTES)
    validate_protocol(protocol, root)
    plan, plan_rows, cohort_ids = validate_plan(root, protocol)
    expected_cases = list(range(3000, 3040))
    summaries = {cohort: load_summary(root, protocol, cohort)
                 for cohort in ("teacher", "c8", "c80")}
    teacher = summaries["teacher"]
    for cohort in ("c8", "c80"):
        summary = summaries[cohort]
        expected_contexts = protocol["calibration_cohorts"][cohort]["contexts"]
        if (summary.get("calibration_contexts"), summary.get("calibration_completed"),
                summary.get("calibrated_contexts"), summary.get("fitted_linears"),
                summary.get("calibration_case_ids")) != (
                expected_contexts, expected_contexts, expected_contexts, 614, cohort_ids[cohort]):
            raise RuntimeError(f"{cohort} calibration context/614-layer coverage differs")
        if (summary.get("suites") != list(SUITES)
                or summary.get("transform") != protocol["transform"]
                or summary.get("vq") != protocol["vq"]
                or summary.get("microscaling") != protocol["microscaling"]
                or summary.get("method_selection") is not False
                or summary.get("test_used_for_method_selection") is not False):
            raise RuntimeError(f"{cohort} frozen method identity or no-selection boundary differs")
    if len({summaries[name]["pbs_jobid"] for name in summaries}) != 3:
        raise RuntimeError("Teacher/C8/C80 must be distinct completed PBS jobs")
    identity = teacher["model_identity"]
    if any(summaries[cohort]["model_identity"] != identity for cohort in ("c8", "c80")):
        raise RuntimeError("Teacher, C8 and C80 model identities are not identical")
    teacher_query_map: dict[tuple[int, int], dict] = {}
    cohort_data = {}
    arm_query_records = {}
    for cohort in ("c8", "c80"):
        teacher_records, arm_records, arm_metrics = load_queries(
            root, protocol, plan_rows, teacher, summaries[cohort], cohort)
        if teacher_query_map and any(teacher_query_map[key] != teacher_records[key]
                                     for key in teacher_query_map):
            raise RuntimeError("C8 and C80 did not use the same saved BF16 teacher queries")
        teacher_query_map = teacher_records
        arm_query_records[cohort] = arm_records
        aggregate = summarize_arm(plan_rows, arm_metrics, summaries[cohort])
        cohort_data[cohort] = aggregate
    if (teacher.get("suites") != {suite: 10 for suite in SUITES}
            or teacher.get("calibration_contexts") != 0):
        raise RuntimeError("Common BF16 teacher suite coverage differs")
    paired = compare_arms(plan_rows, cohort_data["c8"], cohort_data["c80"])
    focus_gripper = focus_gripper_diagnostic(teacher_query_map, arm_query_records, paired)
    result = {
        "status": "complete", "protocol": PROTOCOL,
        "comparison_pbs_jobid": os.environ["PBS_JOBID"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_jobs": {cohort: summaries[cohort]["pbs_jobid"] for cohort in summaries},
        "shared_test": {"case_ids": expected_cases, "case_count": 40, "cases_per_suite": 10,
                        "query_count_per_cohort": 80, "sampler_seed_indices": [0, 1],
                        "paired_sampler_seeds_verified_against_plan": True,
                        "common_teacher_model_identity": identity,
                        "suite_order": list(SUITES)},
        "calibration": {cohort: {"context_count": len(cohort_ids[cohort]),
                                 "contexts_per_suite": protocol["calibration_cohorts"][cohort]["per_suite"],
                                 "case_ids": cohort_ids[cohort]}
                        for cohort in ("c8", "c80")},
        "arms": {cohort: {"pbs_jobid": summaries[cohort]["pbs_jobid"],
                          "device": summaries[cohort]["device"],
                          "target_linears": summaries[cohort]["target_linears"],
                          "fitted_linears": summaries[cohort]["fitted_linears"],
                          "calibration_contexts": summaries[cohort]["calibration_contexts"],
                          **cohort_data[cohort]}
                 for cohort in ("c8", "c80")},
        "paired_comparison": paired,
        "focus_gripper": focus_gripper,
        "method": {"name": "Microscaling additive VQ decoded BF16 with dynamic A4 inputs",
                   "transform": protocol["transform"], "vq": protocol["vq"],
                   "microscaling": protocol["microscaling"],
                   "quantity_task_phase_coverage_change_together": True},
        "evidence_boundary": {"reference": "shared BF16 teacher, same input and sampler seed",
                              "input_domain": "original LIBERO held-out demonstrations",
                              "closed_loop_episodes": 0, "native_low_bit_kernel": False,
                              "statistical_significance_test": False,
                              "latency_or_speedup_claim": False,
                              "claim_scope": protocol["claim_scope"]},
    }
    timing = query_timing({"teacher": teacher_query_map, **arm_query_records})
    if timing is not None:
        result["query_timing_diagnostic"] = timing
    out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "comparison.json", result)
    write_text_atomic(out / "report.md", build_report(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="Remote campaign root containing data/ and results/")
    parser.add_argument("--out", type=Path, required=True,
                        help="Directory receiving comparison.json and report.md")
    args = parser.parse_args()
    require_compute_allocation()
    compare(args.root, args.protocol, args.out)
    print(f"CALIBRATION_COMPARISON_COMPLETE out={args.out}", flush=True)


if __name__ == "__main__":
    main()
