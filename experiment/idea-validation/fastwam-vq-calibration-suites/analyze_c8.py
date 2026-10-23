"""Summarize saved C8 and BF16 teacher queries without model inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import statistics

from compare_calibration import (
    MAX_SUMMARY_BYTES, METRICS, PROTOCOL, SUITES, checked_action, finite_number,
    load_queries, load_summary, read_json, summarize_arm, validate_plan,
    validate_protocol,
)


def require_compute_allocation() -> None:
    job, nodefile = os.environ.get("PBS_JOBID"), os.environ.get("PBS_NODEFILE")
    if not job or not nodefile or not Path(nodefile).is_file():
        raise RuntimeError("C8 analysis requires a genuine PBS compute allocation")
    host = socket.gethostname().split(".", 1)[0]
    nodes = {line.split(".", 1)[0] for line in Path(nodefile).read_text(encoding="utf-8").split()}
    if "login" in host.lower() or host not in nodes:
        raise RuntimeError(f"C8 analysis requires an allocated compute node; host={host}")


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if tmp.stat().st_size >= MAX_SUMMARY_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Control output exceeds 256 KiB: {path}")
    os.replace(tmp, path)


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    if tmp.stat().st_size >= MAX_SUMMARY_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Report exceeds 256 KiB: {path}")
    os.replace(tmp, path)


def largest_cases(summary: dict, plan_rows: dict[int, dict], metric: str,
                  count: int = 5) -> list[dict]:
    values = summary["test"]["per_case"]
    ranked = sorted(values, key=lambda case: (-finite_number(values[case].get(metric), metric), int(case)))
    return [{"case_id": int(case), "suite": plan_rows[int(case)]["suite"],
             "case_mean_over_2_seeds": finite_number(values[case][metric], metric)}
            for case in ranked[:count]]


def step_diagnostics(arm: list[list[float]], teacher: list[list[float]]) -> dict:
    return {
        "step_indexing": "zero-based",
        "gripper_abs_action_difference_by_step": [abs(arm[t][6] - teacher[t][6]) for t in range(32)],
        "tail_motor_max_abs_action_difference_by_step": [
            max(abs(arm[t][d] - teacher[t][d]) for d in range(6)) for t in range(10, 32)],
        "tail_motor_steps": [10, 31],
    }


def describe_queries(top_cases: dict[str, list[dict]], plan_rows: dict[int, dict],
                     teacher_records: dict, arm_records: dict) -> list[dict]:
    selected = sorted({item["case_id"] for items in top_cases.values() for item in items})
    details = []
    for case in selected:
        for seed_index in (0, 1):
            teacher = teacher_records[case, seed_index]
            arm = arm_records[case, seed_index]
            teacher_action = checked_action(teacher["action"], f"teacher/{case}/{seed_index}")
            arm_action = checked_action(arm["action"], f"c8/{case}/{seed_index}")
            details.append({
                "case_id": case, "suite": plan_rows[case]["suite"],
                "task_id": int(plan_rows[case]["task_id"]), "seed_index": seed_index,
                "sampler_seed": int(plan_rows[case]["sampler_seeds"][seed_index]),
                "saved_rmse": {metric: finite_number(arm.get(metric), metric) for metric in METRICS},
                **step_diagnostics(arm_action, teacher_action),
            })
    return details


def build_report(result: dict) -> str:
    macro = result["metrics"]["suite_macro"]
    lines = [
        "# C8 preliminary 数值分析",
        "",
        "本报告只汇总已保存的 C8 与共同 BF16 teacher query records，没有新增 full-model forward。分母为 4 suites × 10 cases × 2 sampler seeds = 80 queries；每 case 先平均两个 seeds，每 suite 平均 10 cases，最后对四个 suites 等权 macro-average。RMSE 越低表示与同输入、同 seed BF16 teacher 的数值差异越小。",
        "",
        "## C8 suite-macro 指标",
        "",
        "| 指标 | RMSE |",
        "| --- | ---: |",
        f"| 主指标：前10步 6维 motor RMSE | {macro['motor_rmse_first10_vs_bf16']:.8g} |",
        f"| 次指标：前10步 gripper RMSE | {macro['gripper_rmse_first10_vs_bf16']:.8g} |",
        f"| 次指标：32步 6维 motor RMSE | {macro['motor_rmse_32_vs_bf16']:.8g} |",
        "",
        "## Per-suite 指标",
        "",
        "| Suite | Cases | Queries | 前10步 motor RMSE | 前10步 gripper RMSE | 32步 motor RMSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for suite in SUITES:
        values = result["metrics"]["per_suite"][suite]
        lines.append(f"| {suite} | 10 | 20 | {values['motor_rmse_first10_vs_bf16']:.8g} | {values['gripper_rmse_first10_vs_bf16']:.8g} | {values['motor_rmse_32_vs_bf16']:.8g} |")

    for title, metric in (("32步 motor RMSE 最大 cases", "motor_rmse_32_vs_bf16"),
                          ("前10步 gripper RMSE 最大 cases", "gripper_rmse_first10_vs_bf16")):
        lines.extend(["", f"## {title}（原 C8 summary 排序，case 内先平均两个 seeds）", "",
                      "| Rank | Case | Suite | RMSE |", "| ---: | ---: | --- | ---: |"])
        for index, item in enumerate(result["largest_error_cases"][metric], 1):
            lines.append(f"| {index} | {item['case_id']} | {item['suite']} | {item['case_mean_over_2_seeds']:.8g} |")

    lines.extend(["", "## Top-case saved-query step diagnostics", "",
                  "以下为上述 top-case union 中每条已保存 query 相对同输入、同 seed teacher 的 normalized action 绝对差。gripper 列按 32 个 step 展开；tail motor 列是 step 10–31 每步 6 个 motor 维度中的最大绝对差。它们是固定输入数值诊断，不表示物理夹爪动作或 closed-loop 失败。", "",
                  "| Case | Seed index | Sampler seed | gripper abs diff by step 0–31 | tail motor max abs diff by step 10–31 |",
                  "| ---: | ---: | ---: | --- | --- |"])
    for item in result["top_case_query_diagnostics"]:
        grip = ",".join(f"{value:.5g}" for value in item["gripper_abs_action_difference_by_step"])
        motor = ",".join(f"{value:.5g}" for value in item["tail_motor_max_abs_action_difference_by_step"])
        lines.append(f"| {item['case_id']} | {item['seed_index']} | {item['sampler_seed']} | `{grip}` | `{motor}` |")
    lines.extend(["", "## 范围与限制", "",
                  "C8 使用 8 个 original-domain calibration contexts（每 suite 2 个、来自不同 tasks），拟合并评估全部 614 个 Linears。Smooth0.5/H128 transforms 沿用冻结设置；action λ=0、local tuning steps=0；结果是 decoded-BF16 Microscaling VQ 与动态 A4 输入的 fixed-input 比较，没有 native low-bit kernel 或 closed-loop episode。C80 尚未完成，本报告不判断 C8 与 C80 谁更好；本报告也不比较 original-only cohort 与旧 mixed cohort 的优劣。"])
    return "\n".join(lines) + "\n"


def analyze(root: Path, protocol_path: Path, out: Path) -> dict:
    protocol = read_json(protocol_path, MAX_SUMMARY_BYTES)
    validate_protocol(protocol, root)
    _, plan_rows, cohort_ids = validate_plan(root, protocol)
    teacher = load_summary(root, protocol, "teacher")
    c8 = load_summary(root, protocol, "c8")
    ids = cohort_ids["c8"]
    if (c8.get("calibration_contexts"), c8.get("calibration_completed"),
            c8.get("calibrated_contexts"), c8.get("calibration_case_ids"),
            c8.get("fitted_linears")) != (8, 8, 8, ids, 614):
        raise RuntimeError("C8 must have all 8 calibration contexts and 614 fitted Linears")
    if (c8.get("suites") != list(SUITES) or c8.get("transform") != protocol["transform"]
            or c8.get("vq") != protocol["vq"] or c8.get("microscaling") != protocol["microscaling"]
            or c8.get("method_selection") is not False
            or c8.get("test_used_for_method_selection") is not False):
        raise RuntimeError("C8 recipe or no-selection boundary differs from the frozen protocol")
    teacher_records, arm_records, query_metrics = load_queries(
        root, protocol, plan_rows, teacher, c8, "c8")
    metrics = summarize_arm(plan_rows, query_metrics, c8)
    top_cases = {metric: largest_cases(c8, plan_rows, metric)
                 for metric in ("motor_rmse_32_vs_bf16", "gripper_rmse_first10_vs_bf16")}
    result = {
        "status": "complete", "phase": "c8_preliminary_analysis", "protocol": PROTOCOL,
        "analysis_pbs_jobid": os.environ["PBS_JOBID"], "source_jobs": {
            "teacher": teacher["pbs_jobid"], "c8": c8["pbs_jobid"]},
        "cohort": "c8", "calibration_context_count": 8,
        "calibration_case_ids": ids, "fitted_linears": 614,
        "shared_test": {"case_count": 40, "cases_per_suite": 10, "query_count": 80,
                        "seed_indices": [0, 1], "suite_order": list(SUITES),
                        "teacher_model_identity": teacher["model_identity"]},
        "metrics": metrics,
        "largest_error_cases": top_cases,
        "top_case_query_diagnostics": describe_queries(top_cases, plan_rows,
                                                        teacher_records, arm_records),
        "method": {"transform": protocol["transform"], "vq": protocol["vq"],
                   "microscaling": protocol["microscaling"],
                   "decoded_bf16": True, "dynamic_a4_inputs": True,
                   "native_low_bit_kernel": False, "closed_loop_episodes": 0,
                   "new_full_model_forwards": 0},
        "comparison_status": {"c80": "not_complete_at_request_time",
                              "c8_vs_c80_comparison": "not_available",
                              "original_only_vs_mixed_comparison": "not_available"},
    }
    summary = {"status": "complete", "phase": "c8_preliminary_analysis",
               "protocol": PROTOCOL, "pbs_jobid": result["analysis_pbs_jobid"],
               "case_count": 40, "query_count": 80, "target_linears": 614,
               "calibration_contexts": 8, "fitted_linears": 614,
               "source_jobs": result["source_jobs"], "outputs": ["analysis.json", "report.zh.md"]}
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "analysis.json", result)
    save_text(out / "report.zh.md", build_report(result))
    save_json(out / "summary.json", summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_compute_allocation()
    analyze(args.root, args.protocol, args.out)
    print(f"C8_ANALYSIS_COMPLETE out={args.out}", flush=True)


if __name__ == "__main__":
    main()
