"""Aggregate the frozen four-arm dual-KL endpoint panel on a PBS CPU node."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import statistics


ARMS = ("mse_nmse", "action_kl", "latent_kl", "dual_kl")
NEW_ARMS = ("action_kl", "latent_kl", "dual_kl")
METRICS = (
    "first10motor_mse", "first10motor_rmse", "full32motor_rmse",
    "late22motor_rmse", "first10gripper_rmse", "future_latent_nmse",
    "mixed_loss", "action_kl", "latent_kl", "objective_loss",
)
PRIMARY = ("first10motor_rmse", "future_latent_nmse")
MAX_JSON_BYTES = 200 * 1024


def guard():
    job_id = os.environ.get("PBS_JOBID", "")
    nodefile = os.environ.get("PBS_NODEFILE", "")
    host = socket.gethostname().split(".")[0]
    if not job_id or not nodefile or "login" in host.lower():
        raise RuntimeError("A real PBS compute allocation is required")
    nodes = {line.split("/")[0].split(".")[0] for line in Path(nodefile).read_text().splitlines() if line.strip()}
    if host not in nodes:
        raise RuntimeError("Host is outside the PBS allocation")
    return job_id


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


def finite_metric(metrics, name):
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"Metric {name} is missing or non-finite")
    if value < 0:
        raise RuntimeError(f"Metric {name} must be non-negative")
    return float(value)


def matrix(value, name):
    if not isinstance(value, list) or len(value) != 32:
        raise RuntimeError(f"{name} must contain 32 action rows")
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != 7:
            raise RuntimeError(f"{name} must have shape [32,7]")
        parsed = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, (int, float)) or not math.isfinite(float(cell)):
                raise RuntimeError(f"{name} contains a non-finite action")
            parsed.append(float(cell))
        result.append(parsed)
    return result


def motor_abs_chunk_bias(action, teacher):
    dims = [abs(sum(action[t][d] - teacher[t][d] for t in range(10)) / 10) for d in range(6)]
    return average(dims), dims


def verify_receipt(receipt, p, arm, expected_job=None, expected_protocol=None):
    require(receipt.get("complete") is True and receipt.get("updates") == 80,
            f"{arm} training receipt is not a complete 80-update run")
    require(receipt.get("protocol") == (expected_protocol or p["protocol"]),
            f"{arm} receipt protocol differs")
    order = receipt.get("update_order")
    require(order == p["calibration"]["update_order"], f"{arm} receipt update_order differs")
    require(bool(receipt.get("pbs_jobid")), f"{arm} receipt lacks pbs_jobid")
    if expected_job is not None:
        require(receipt["pbs_jobid"] == expected_job, f"{arm} receipt and run summary job ids differ")
    history = receipt.get("history")
    if history is not None:
        require([item.get("case_id") for item in history] == order,
                f"{arm} receipt history does not follow update_order")
    return receipt


def expected_queries(p, split):
    if split == "calibration":
        cases, seeds = p["calibration"]["case_ids"], [0]
    else:
        cases = p["evaluation"]["case_ids"]
        seeds = p["evaluation"].get("seed_indices", [0, 1])
    require(seeds == ([0] if split == "calibration" else [0, 1]),
            f"Frozen {split} seed indices differ")
    return [(case, seed) for case in cases for seed in seeds]


def validate_record(record, p, arm, split, case, seed, plan_rows, expected_job):
    row = plan_rows.get(case)
    require(row is not None, f"Plan row is missing for case {case}")
    expected = {
        "protocol": p["protocol"], "arm": arm, "split": split, "updates": 80,
        "case_id": case, "suite": row["suite"], "plan_row": row,
        "seed_index": seed, "sampler_seed": row["sampler_seeds"][seed],
    }
    require(all(record.get(key) == value for key, value in expected.items()),
            f"{arm}/{split}/{case}/{seed} input binding differs from the frozen plan")
    require(bool(record.get("pbs_jobid")), f"{arm}/{case}/{seed} record lacks pbs_jobid")
    if expected_job:
        require(record["pbs_jobid"] == expected_job,
                f"{arm}/{case}/{seed} record came from a different PBS job")
    values = record.get("metrics", record)
    require(isinstance(values, dict), f"{arm}/{case}/{seed} metrics object is missing")
    parsed = {name: finite_metric(values, name) for name in METRICS}
    action, teacher_action = matrix(record.get("action"), "action"), matrix(record.get("teacher_action"), "teacher_action")
    commands, teacher_commands = matrix(record.get("commands"), "commands"), matrix(record.get("teacher_commands"), "teacher_commands")
    action_bias, action_dims = motor_abs_chunk_bias(action, teacher_action)
    command_bias, command_dims = motor_abs_chunk_bias(commands, teacher_commands)
    return {
        "case_id": case, "seed_index": seed, "suite": row["suite"],
        "pbs_jobid": record["pbs_jobid"], "metrics": parsed,
        "action_motor_abs_chunk_bias": action_bias,
        "action_motor_abs_chunk_bias_by_dimension": action_dims,
        "command_motor_abs_chunk_bias": command_bias,
        "command_motor_abs_chunk_bias_by_dimension": command_dims,
    }


def average(values):
    return sum(values) / len(values) if values else None


def aggregate_arm(records, suites):
    by_case = {}
    for record in records:
        case = record["case_id"]
        by_case.setdefault(case, {"suite": record["suite"], "metrics": {}, "bias": {}})
        target = by_case[case]
        require(target["suite"] == record["suite"], "Suite changed within a case")
        for name, value in record["metrics"].items():
            target["metrics"].setdefault(name, []).append(value)
        for name in ("action_motor_abs_chunk_bias", "command_motor_abs_chunk_bias"):
            target["bias"].setdefault(name, []).append(record[name])
        for name in ("action_motor_abs_chunk_bias_by_dimension", "command_motor_abs_chunk_bias_by_dimension"):
            target["bias"].setdefault(name, []).append(record[name])

    case_values = {}
    for case, item in by_case.items():
        case_values[case] = {name: average(values) for name, values in item["metrics"].items()}
        case_values[case]["_suite"] = item["suite"]
        for name, values in item["bias"].items():
            if name.endswith("_by_dimension"):
                case_values[case][name] = [average([v[d] for v in values]) for d in range(6)]
            else:
                case_values[case][name] = average(values)

    by_suite = {}
    for suite in suites:
        selected = [case_values[c] for c in case_values if case_values[c]["_suite"] == suite]
        require(bool(selected), f"No cases belong to suite {suite}")
        by_suite[suite] = {name: average([row[name] for row in selected]) for name in METRICS}
        by_suite[suite]["cases"] = len(selected)
        for name in ("action_motor_abs_chunk_bias", "command_motor_abs_chunk_bias"):
            by_suite[suite][name] = average([row[name] for row in selected])
        for name in ("action_motor_abs_chunk_bias_by_dimension", "command_motor_abs_chunk_bias_by_dimension"):
            by_suite[suite][name] = [average([row[name][d] for row in selected]) for d in range(6)]
    overall = {name: average([by_suite[s][name] for s in suites]) for name in METRICS}
    for name in ("action_motor_abs_chunk_bias", "command_motor_abs_chunk_bias"):
        overall[name] = average([by_suite[s][name] for s in suites])
    for name in ("action_motor_abs_chunk_bias_by_dimension", "command_motor_abs_chunk_bias_by_dimension"):
        overall[name] = [average([by_suite[s][name][d] for s in suites]) for d in range(6)]
    return {"queries": len(records), "cases": len(by_case), "overall": overall,
            "by_suite": by_suite, "case_values": case_values}


def bootstrap_interval(values, seed):
    rng = random.Random(seed)
    n, reps = len(values), 2000
    means = sorted(average([values[rng.randrange(n)] for _ in range(n)]) for _ in range(reps))
    return [means[int(0.025 * (reps - 1))], means[int(0.975 * (reps - 1))]]


def paired_comparisons(split_stats, split, seed_base):
    control = split_stats["mse_nmse"]["case_values"]
    result, vectors = {}, {}
    for arm in NEW_ARMS:
        arm_cases = split_stats[arm]["case_values"]
        cases = sorted(control)
        require(set(cases) == set(arm_cases), f"{split} paired case sets differ for {arm}")
        result[arm] = {}
        vectors[arm] = {}
        for index, metric in enumerate(METRICS):
            deltas = [arm_cases[case][metric] - control[case][metric] for case in cases]
            vectors[arm][metric] = dict(zip(cases, deltas))
            loco = [average(deltas[:i] + deltas[i + 1:]) for i in range(len(deltas))]
            item = {
                "cases": len(deltas), "mean_delta": average(deltas),
                "median_delta": statistics.median(deltas),
                "improved": sum(delta < 0 for delta in deltas),
                "worsened": sum(delta > 0 for delta in deltas),
                "tied": sum(delta == 0 for delta in deltas),
                "leave_one_case_out_mean_range": [min(loco), max(loco)],
            }
            if split == "heldout":
                item["fixed_seed_bootstrap_95pct_ci"] = bootstrap_interval(
                    deltas, seed_base + NEW_ARMS.index(arm) * len(METRICS) + index)
            result[arm][metric] = item
    return result, vectors


def interactions(split_stats):
    cases = sorted(split_stats["mse_nmse"]["case_values"])
    result = {}
    for metric in METRICS:
        by_suite = {}
        for suite in split_stats["mse_nmse"]["by_suite"]:
            vals = []
            for case in cases:
                if split_stats["mse_nmse"]["case_values"][case]["_suite"] != suite:
                    continue
                values = {arm: split_stats[arm]["case_values"][case][metric] for arm in ARMS}
                vals.append(values["dual_kl"] - values["action_kl"] - values["latent_kl"] + values["mse_nmse"])
            by_suite[suite] = average(vals)
        result[metric] = {
            "definition": "dual_kl - action_kl - latent_kl + mse_nmse",
            "overall_equal_suite_mean": average(list(by_suite.values())),
            "by_suite": by_suite,
            "case_deltas": {
                case: (split_stats["dual_kl"]["case_values"][case][metric]
                       - split_stats["action_kl"]["case_values"][case][metric]
                       - split_stats["latent_kl"]["case_values"][case][metric]
                       + split_stats["mse_nmse"]["case_values"][case][metric])
                for case in cases
            },
        }
    return result


def validate_preparation(root, p):
    fit = read_json(root / "results/fit_stats/summary.json")
    require(fit.get("complete") is True and fit.get("protocol") == p["protocol"],
            "C80 fit_stats summary is incomplete or belongs to another protocol")
    require(fit.get("calibration_case_ids") == p["calibration"]["case_ids"]
            and fit.get("seed_indices") == [0] and fit.get("heldout_used") is False
            and fit.get("new_teacher_queries") == 0,
            "fit_stats must use only the original C80 calibration queries")
    scaling = read_json(root / "results/prepare/loss_scaling.json")
    initial_rows = scaling.get("initial_rows", [])
    require(scaling.get("protocol") == p["protocol"] and scaling.get("complete") is True
            and len(initial_rows) == 80
            and scaling.get("seed_indices") == [0]
            and scaling.get("case_ids") == p["calibration"]["case_ids"]
            and scaling.get("heldout_used") is False,
            "Loss scaling must be fitted on the frozen 80-query C80 calibration set")
    require([row.get("case_id") for row in initial_rows] == p["calibration"]["case_ids"]
            and all(row.get("seed_index") == 0 for row in initial_rows),
            "Loss scaling rows must be exactly the original C80 seed-0 queries")
    if "calibration_case_ids" in scaling:
        require(scaling["calibration_case_ids"] == p["calibration"]["case_ids"],
                "Loss scaling calibration cases differ")
    prepare = read_json(root / "results/prepare/summary.json")
    require(prepare.get("complete") is True and prepare.get("protocol") == p["protocol"],
            "prepare summary is incomplete or belongs to another protocol")
    require(prepare.get("initial_calibration_queries") == 80
            and prepare.get("control_evaluation_queries") == 160
            and prepare.get("new_teacher_queries") == 0,
            "prepare must replay the original control's 160 endpoints without new teacher queries")
    require(bool(prepare.get("pbs_jobid")), "prepare summary lacks pbs_jobid")
    original = read_json(Path(p["source_per_channel_root"]) / "results/train_eval/training_receipt.json")
    source_protocol = read_json(Path(p["source_per_channel_root"]) / "protocol.json")
    verify_receipt(original, p, "mse_nmse control", expected_protocol=source_protocol["protocol"])
    expected_source_job = p.get("source_per_channel_job")
    if expected_source_job:
        require(original["pbs_jobid"] == expected_source_job,
                "Reused MSE/NMSE control checkpoint job differs from the frozen source")
    return fit, scaling, prepare, original


def validate_new_runs(root, p):
    summaries, receipts = {}, {}
    for arm in NEW_ARMS:
        summary = read_json(root / "results" / arm / "summary.json")
        require(summary.get("complete") is True and summary.get("protocol") == p["protocol"]
                and summary.get("updates") == 80 and summary.get("evaluation_queries") == 160
                and summary.get("new_teacher_queries") == 0,
                f"{arm} must complete exactly 80 updates and 160 evaluations without new teacher queries")
        require(bool(summary.get("pbs_jobid")), f"{arm} summary lacks pbs_jobid")
        receipt_path = root / "results" / arm / "training_receipt.json"
        receipt = read_json(receipt_path)
        verify_receipt(receipt, p, arm, summary["pbs_jobid"])
        summaries[arm], receipts[arm] = summary, receipt
    return summaries, receipts


def read_panel(root, p, plan_rows, run_jobs):
    suites = []
    for row in plan_rows.values():
        if row["suite"] not in suites:
            suites.append(row["suite"])
    require(len(suites) == 4, "The frozen plan must contain exactly four suites")
    stats, record_job_ids = {}, {}
    for split in ("calibration", "heldout"):
        expected = expected_queries(p, split)
        split_stats = {}
        for arm in ARMS:
            records, seen = [], set()
            for case, seed in expected:
                record = read_json(root / "results" / split / arm / "queries" / f"{case}_{seed}.json")
                parsed = validate_record(record, p, arm, split, case, seed, plan_rows, run_jobs[arm])
                key = (case, seed)
                require(key not in seen, f"Duplicate {split}/{arm} query {key}")
                seen.add(key)
                records.append(parsed)
            require(len(seen) == len(expected), f"Incomplete {split}/{arm} query panel")
            record_job_ids[arm] = sorted({row["pbs_jobid"] for row in records})
            split_stats[arm] = aggregate_arm(records, suites)
            require(split_stats[arm]["cases"] == len(p["calibration"]["case_ids"] if split == "calibration"
                                                        else p["evaluation"]["case_ids"]),
                    f"{split}/{arm} case count differs from frozen cohort")
            expected_per_suite = 20 if split == "calibration" else 10
            require(all(split_stats[arm]["by_suite"][suite]["cases"] == expected_per_suite for suite in suites),
                    f"{split}/{arm} cases per suite differ from the frozen balanced panel")
            expected_per_suite = 20 if split == "calibration" else 10
            require(all(split_stats[arm]["by_suite"][suite]["cases"] == expected_per_suite for suite in suites),
                    f"{split}/{arm} cases per suite differ from the frozen balanced panel")
        stats[split] = split_stats
    return suites, stats, record_job_ids


def gpu_telemetry(root, prepare, summaries):
    jobs = [("mse_nmse_control_replay", root, prepare["pbs_jobid"])]
    jobs.extend((arm, root, summaries[arm]["pbs_jobid"]) for arm in NEW_ARMS)
    output = {}
    for label, job_root, job_id in jobs:
        path = Path(job_root) / "artifacts" / job_id / "job.log"
        if not path.is_file():
            output[label] = {"job_id": job_id, "log_found": False, "stages": {}}
            continue
        current_phase, stage_values = "unknown", {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "PROGRESS " in line and "{" in line:
                try:
                    payload = json.loads(line[line.index("{"):])
                    if payload.get("phase"):
                        current_phase = str(payload["phase"])
                except (ValueError, json.JSONDecodeError):
                    pass
            if not line.startswith("GPU_SAMPLE timestamp="):
                continue
            sample = line.split(" ", 2)[-1]
            util = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", sample)
            memory = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*MiB", sample)
            util_value = float(util.group(1)) if util else None
            memory_value = float(memory[0]) if memory else None
            stage_values.setdefault(current_phase, []).append((util_value, memory_value))
        stages = {}
        for phase, values in stage_values.items():
            utils = [u for u, _ in values if u is not None]
            memories = [m for _, m in values if m is not None]
            stages[phase] = {
                "samples": len(values),
                "gpu_utilization_percent_mean": average(utils),
                "gpu_utilization_percent_max": max(utils) if utils else None,
                "memory_used_mib_mean": average(memories),
                "memory_used_mib_max": max(memories) if memories else None,
            }
        output[label] = {"job_id": job_id, "log_found": True, "stages": stages}
    return output


def markdown_table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "\n".join(
        "| " + " | ".join(str(value) for value in row) + " |" for row in rows)


def report_text(p, split_stats, comparisons, interaction, decision, telemetry, fit, scaling):
    metrics = ("first10motor_rmse", "full32motor_rmse", "late22motor_rmse",
               "future_latent_nmse", "action_kl", "latent_kl")
    lines = [
        "# Fast-WAM 双 KL 2×2 固定输入分析", "",
        "本分析比较 `mse_nmse`、`action_kl`、`latent_kl`、`dual_kl` 四组。每组为一次 80-update 训练；单次训练结果不是独立训练复现。全部结果来自固定输入、decoded-BF16 endpoint 记录，不包含 rollout 或 closed-loop 证据。",
        "", "每个 case 先对 seed 求均值，再在 suite 内对 case 求均值，最后四个 suite 等权。Gripper RMSE 仅作诊断，未进入主指标或候选门槛。", "",
        "## Held-out：原 40 cases × 2 seeds", "",
        markdown_table(["Arm", "First10 motor RMSE", "Full32 motor RMSE", "Late22 motor RMSE", "Future latent NMSE", "Action KL", "Latent KL"],
                       [[arm] + [f"{split_stats['heldout'][arm]['overall'][metric]:.8g}" for metric in metrics]
                        for arm in ARMS]),
        "", "## Held-out：suite 等权分项", "",
        markdown_table(["Suite", "Arm", "First10 motor RMSE", "Full32 motor RMSE", "Late22 motor RMSE", "Future latent NMSE", "Action KL", "Latent KL"],
                       [[suite, arm] + [f"{split_stats['heldout'][arm]['by_suite'][suite][metric]:.8g}" for metric in metrics]
                        for suite in split_stats["heldout"]["mse_nmse"]["by_suite"] for arm in ARMS]),
        "", "## Held-out motor bias 诊断", "",
        "先对每个 query 的前 10 步逐维计算 `abs(mean_t(student − teacher))`，再对同一 case 的 seeds 求均值，最后按 case 和 suite 等权汇总；action 与 command 分开计算。", "",
        markdown_table(["Arm", "Action bias dims 0–5", "Mean abs bias", "Command bias dims 0–5", "Mean abs bias"],
                       [[arm,
                         ", ".join(f"{v:.6g}" for v in split_stats['heldout'][arm]['overall']['action_motor_abs_chunk_bias_by_dimension']),
                         f"{split_stats['heldout'][arm]['overall']['action_motor_abs_chunk_bias']:.6g}",
                         ", ".join(f"{v:.6g}" for v in split_stats['heldout'][arm]['overall']['command_motor_abs_chunk_bias_by_dimension']),
                         f"{split_stats['heldout'][arm]['overall']['command_motor_abs_chunk_bias']:.6g}"]
                        for arm in ARMS]),
        "", "## Calibration：原 C80 × 1 seed", "",
        markdown_table(["Arm", "First10 motor RMSE", "Full32 motor RMSE", "Late22 motor RMSE", "Future latent NMSE", "Action KL", "Latent KL"],
                       [[arm] + [f"{split_stats['calibration'][arm]['overall'][metric]:.8g}" for metric in metrics]
                        for arm in ARMS]),
        "", "## Calibration：suite 分项", "",
        markdown_table(["Suite", "Arm", "First10 motor RMSE", "Full32 motor RMSE", "Late22 motor RMSE", "Future latent NMSE", "Action KL", "Latent KL"],
                       [[suite, arm] + [f"{split_stats['calibration'][arm]['by_suite'][suite][metric]:.8g}" for metric in metrics]
                        for suite in split_stats["calibration"]["mse_nmse"]["by_suite"] for arm in ARMS]),
        "", "## Loss 与 gripper 诊断", "",
        markdown_table(["Split", "Arm", "First10 motor MSE", "Mixed loss", "Objective loss", "First10 gripper RMSE"],
                       [[split, arm] + [f"{split_stats[split][arm]['overall'][metric]:.8g}"
                                        for metric in ("first10motor_mse", "mixed_loss", "objective_loss", "first10gripper_rmse")]
                        for split in ("calibration", "heldout") for arm in ARMS]),
        "", "Calibration 用途仅限原 C80 的 KL 坐标和 loss scaling；held-out 用于冻结端点比较，未参与拟合。",
        f"Loss scaling 来源为 {len(scaling.get('initial_rows', []))} 个 C80 calibration queries；`heldout_used={scaling.get('heldout_used')}`。Fit stats 标记 `heldout_used={fit.get('heldout_used')}`。", "",
        "## Held-out 配对变化：候选减 `mse_nmse`", "",
        "负 delta 表示该指标下降。统计单位为 seed 平均后的 40 个 case；区间是固定 seed 的 case bootstrap 95% 区间，用于描述本 cohort 的波动，不作显著性结论。", "",
        markdown_table(["Arm", "Metric", "Mean Δ", "Median Δ", "Improved / worsened / tied", "LOCO mean range", "Bootstrap 95% CI"],
                       [[arm, metric, f"{item['mean_delta']:.7g}", f"{item['median_delta']:.7g}",
                         f"{item['improved']} / {item['worsened']} / {item['tied']}",
                         f"[{item['leave_one_case_out_mean_range'][0]:.7g}, {item['leave_one_case_out_mean_range'][1]:.7g}]",
                         f"[{item['fixed_seed_bootstrap_95pct_ci'][0]:.7g}, {item['fixed_seed_bootstrap_95pct_ci'][1]:.7g}]"]
                        for arm in NEW_ARMS for metric in PRIMARY for item in [comparisons[arm][metric]]]),
        "", "## 2×2 interaction", "",
        "Interaction 按 `dual_kl − action_kl − latent_kl + mse_nmse` 计算；正负号表示该指标上的交互方向，不单独证明机制。", "",
        markdown_table(["Metric", "Equal-suite interaction", "By-suite"],
                       [[metric, f"{interaction[metric]['overall_equal_suite_mean']:.7g}",
                         "; ".join(f"{suite}: {value:.7g}" for suite, value in interaction[metric]["by_suite"].items())]
                        for metric in PRIMARY]),
        "", "## 预设工程门槛与 Pareto 对比", "",
        f"Dual-KL 候选门槛：{decision['candidate_screen']['passes']}。motor 低于 control：{decision['candidate_screen']['motor_lower_than_control']}；latent 低于 control：{decision['candidate_screen']['latent_lower_than_control']}；motor case 中位 delta 为 {decision['candidate_screen']['motor_case_median_delta']:.7g}；40 cases 中改善 {decision['candidate_screen']['motor_improved_cases']} 个（至少 24 个）。",
        "", markdown_table(["对比 single-KL", "Dual-KL Pareto 优于", "Motor Δ", "Latent Δ"],
                           [[arm, str(decision["pareto_vs_single_kl"][arm]["pareto_better"]),
                             f"{decision['pareto_vs_single_kl'][arm]['motor_delta']:+.7g}",
                             f"{decision['pareto_vs_single_kl'][arm]['latent_delta']:+.7g}"]
                            for arm in ("action_kl", "latent_kl")]),
        "", "## 训练与 GPU 记录", "",
        "三组新 loss 各有一次训练；case bootstrap 只反映固定训练 checkpoint 下的输入 cohort 变化，不能替代独立训练复现。Loss 数值下降不代替动作质量或闭环验证。", "",
        markdown_table(["GPU job", "阶段", "Samples", "Mean util %", "Peak util %", "Mean memory MiB", "Peak memory MiB"],
                       [[label, phase, values["samples"],
                         "—" if values["gpu_utilization_percent_mean"] is None else f"{values['gpu_utilization_percent_mean']:.1f}",
                         "—" if values["gpu_utilization_percent_max"] is None else f"{values['gpu_utilization_percent_max']:.1f}",
                         "—" if values["memory_used_mib_mean"] is None else f"{values['memory_used_mib_mean']:.1f}",
                         "—" if values["memory_used_mib_max"] is None else f"{values['memory_used_mib_max']:.1f}"]
                        for label, job in telemetry.items() for phase, values in job["stages"].items()] or [["—", "No GPU_SAMPLE lines", "0", "—", "—", "—", "—"]]),
        "", "本结果是固定输入 endpoint 证据；未运行 rollout，不作 closed-loop 或统计显著性结论。",
    ]
    return "\n".join(lines) + "\n"


def aggregate(p, aggregate_job_id):
    root = Path(p["remote_root"])
    require(p.get("arms") == list(ARMS), "Protocol arm order or membership differs")
    require(p.get("new_training_arms") == list(NEW_ARMS), "Protocol new-training arms differ")
    plan = read_json(p["calibration"]["plan_file"])
    require(isinstance(plan.get("inputs"), list), "Frozen calibration plan lacks inputs")
    plan_rows = {}
    for row in plan["inputs"]:
        case = row["case_id"]
        require(case not in plan_rows, f"Duplicate plan row for case {case}")
        plan_rows[case] = row
    frozen_cases = p["calibration"]["case_ids"] + p["evaluation"]["case_ids"]
    require(len(p["calibration"]["case_ids"]) == 80 and len(p["evaluation"]["case_ids"]) == 40,
            "Frozen cohorts must contain 80 calibration and 40 held-out cases")
    require(len(p["calibration"]["case_ids"]) == 80 and len(p["evaluation"]["case_ids"]) == 40,
            "Frozen cohorts must contain 80 calibration and 40 held-out cases")
    require(len(frozen_cases) == len(set(frozen_cases)), "Calibration and held-out cases overlap")
    require(all(case in plan_rows for case in frozen_cases), "Frozen protocol case is absent from plan inputs")

    fit, scaling, prepare, control_receipt = validate_preparation(root, p)
    new_summaries, new_receipts = validate_new_runs(root, p)
    run_jobs = {"mse_nmse": prepare["pbs_jobid"]}
    run_jobs.update({arm: new_summaries[arm]["pbs_jobid"] for arm in NEW_ARMS})
    suites, split_stats, record_job_ids = read_panel(root, p, plan_rows, run_jobs)

    for arm in ARMS:
        worker = new_summaries.get(arm, prepare)
        training = new_receipts.get(arm, control_receipt)
        summary = {
            "protocol": p["protocol"], "arm": arm, "complete": True,
            "updates": 80, "evaluation_queries": 160, "new_teacher_queries": 0,
            "pbs_jobid": worker["pbs_jobid"],
            "training_receipt": {"pbs_jobid": training["pbs_jobid"],
                                 "updates": training["updates"],
                                 "update_order": training["update_order"]},
            "splits": {split: {k: v for k, v in split_stats[split][arm].items() if k != "case_values"}
                       for split in ("calibration", "heldout")},
        }
        save_json(root / "results" / arm / "comparison_summary.json", summary)

    paired, paired_vectors = {}, {}
    for split in ("calibration", "heldout"):
        paired[split], paired_vectors[split] = paired_comparisons(
            split_stats[split], split, 20261019 + (0 if split == "calibration" else 10000))
    interaction = {split: interactions(split_stats[split]) for split in ("calibration", "heldout")}
    heldout = split_stats["heldout"]
    dual_motor = paired["heldout"]["dual_kl"]["first10motor_rmse"]
    candidate = {
        "motor_lower_than_control": heldout["dual_kl"]["overall"]["first10motor_rmse"] < heldout["mse_nmse"]["overall"]["first10motor_rmse"],
        "latent_lower_than_control": heldout["dual_kl"]["overall"]["future_latent_nmse"] < heldout["mse_nmse"]["overall"]["future_latent_nmse"],
        "motor_case_median_delta": dual_motor["median_delta"],
        "motor_improved_cases": dual_motor["improved"],
        "required_motor_improved_cases": 24,
    }
    candidate["passes"] = (candidate["motor_lower_than_control"] and candidate["latent_lower_than_control"]
                            and candidate["motor_case_median_delta"] < 0
                            and candidate["motor_improved_cases"] >= 24)
    pareto = {}
    for arm in ("action_kl", "latent_kl"):
        motor_delta = heldout["dual_kl"]["overall"]["first10motor_rmse"] - heldout[arm]["overall"]["first10motor_rmse"]
        latent_delta = heldout["dual_kl"]["overall"]["future_latent_nmse"] - heldout[arm]["overall"]["future_latent_nmse"]
        pareto[arm] = {"pareto_better": motor_delta <= 0 and latent_delta <= 0 and (motor_delta < 0 or latent_delta < 0),
                       "motor_delta": motor_delta, "latent_delta": latent_delta}
    decision = {
        "protocol": p["protocol"], "complete": True, "pbs_jobid": aggregate_job_id,
        "candidate_screen": candidate, "pareto_vs_single_kl": pareto,
        "interpretation": "engineering_screen_only; no statistical significance, independent training replication, rollout, or closed-loop claim",
        "one_training_run_per_arm": True,
    }
    save_json(root / "results/decision.json", decision)

    detail_metrics = (*PRIMARY, "action_kl", "latent_kl")
    paired_details = {
        "protocol": p["protocol"], "complete": True, "pbs_jobid": aggregate_job_id,
        "case_unit": "seed mean within case; 40 held-out cases, not 80 independent queries",
        "calibration_case_count": len(p["calibration"]["case_ids"]),
        "heldout_case_count": len(p["evaluation"]["case_ids"]),
        "paired_vs_mse_nmse": {"calibration": paired["calibration"],
                                "heldout": {arm: {metric: paired["heldout"][arm][metric]
                                                  for metric in detail_metrics}
                                            for arm in NEW_ARMS}},
        "case_metrics": {}, "paired_case_deltas": {}, "interaction": {},
    }
    heldout_case_values = {arm: split_stats["heldout"][arm]["case_values"] for arm in ARMS}
    paired_details["case_metrics"] = {
        case: {arm: {**{metric: heldout_case_values[arm][case][metric] for metric in detail_metrics},
                     "action_motor_abs_chunk_bias": heldout_case_values[arm][case]["action_motor_abs_chunk_bias"],
                     "command_motor_abs_chunk_bias": heldout_case_values[arm][case]["command_motor_abs_chunk_bias"]}
               for arm in ARMS} for case in sorted(heldout_case_values["mse_nmse"])
    }
    paired_details["paired_case_deltas"] = {
        arm: {metric: paired_vectors["heldout"][arm][metric] for metric in detail_metrics}
        for arm in NEW_ARMS
    }
    paired_details["interaction"] = {
        split: {metric: interaction[split][metric] for metric in PRIMARY}
        for split in ("calibration", "heldout")
    }
    save_json(root / "results/paired_details.json", paired_details, MAX_JSON_BYTES)

    telemetry = gpu_telemetry(root, prepare, new_summaries)
    comparison = {
        "protocol": p["protocol"], "complete": True, "pbs_jobid": aggregate_job_id,
        "heldout_records": 320, "calibration_records": 320,
        "training_updates_per_arm": 80, "new_teacher_queries": 0,
        "queries_by_arm_per_split": 80, "suite_weighting": "mean cases after averaging seeds within case; four suites equally weighted",
        "one_training_run_per_arm": True,
        "evidence_scope": p.get("scope"),
        "loss_scaling": scaling,
        "splits": {split: {arm: {k: v for k, v in split_stats[split][arm].items() if k != "case_values"}
                           for arm in ARMS} for split in ("calibration", "heldout")},
        "paired_vs_mse_nmse": paired, "interaction": interaction,
        "decision": decision, "gpu_telemetry_by_phase": telemetry,
        "record_pbs_job_ids": record_job_ids,
        "claims": {"statistical_significance": False, "independent_training_replicates": False,
                   "closed_loop_evaluated": False, "rollouts": 0, "gripper_is_primary": False},
    }
    save_json(root / "results/comparison.json", comparison, MAX_JSON_BYTES)
    aggregate_summary = {
        "protocol": p["protocol"], "complete": True, "pbs_jobid": aggregate_job_id,
        "heldout_records": 320, "calibration_records": 320,
        "training_updates_per_arm": 80, "new_teacher_queries": 0,
        "arms": list(ARMS), "suites": suites,
        "one_training_run_per_arm": True,
    }
    save_json(root / "results/aggregate/summary.json", aggregate_summary)
    report = report_text(p, split_stats, paired["heldout"], interaction["heldout"],
                         decision, telemetry, fit, scaling)
    (root / "results/ANALYSIS.zh.md").write_text(report, encoding="utf-8")
    print("DUAL_KL_AGGREGATE_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    job_id = guard()
    p = read_json(args.protocol)
    aggregate(p, job_id)


if __name__ == "__main__":
    main()
