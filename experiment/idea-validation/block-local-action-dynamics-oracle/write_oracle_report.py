"""Create the Chinese report and decision JSON from completed oracle outputs."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def generate(out: Path, allocation: dict[str, Any]) -> dict[str, Any]:
    config = json.loads((out / "FREEZE.json").read_text(encoding="utf-8"))
    addon = json.loads((out / "GLOBAL16_FREEZE.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    mechanisms = json.loads((out / "mechanism_tests.json").read_text(encoding="utf-8"))["checks"]
    reproduction = json.loads((out / "reference_reproduction.json").read_text(encoding="utf-8"))
    runs = summary["runs"]
    if len(runs) != 30 or sum(r["arm"] != "oracle_global16" for r in runs) != 27:
        raise ValueError("oracle report requires the complete frozen 27+3 run matrix")
    if len(reproduction) != 39 or not all(r["original_episode_errors_reproduced"] for r in reproduction):
        raise ValueError("reference checkpoint/error reproduction is incomplete")
    if not all(r.get("test", {}).get("dense_quality_gate_pass") is not None for r in runs):
        raise ValueError("quality gate results missing")

    primary = [r for r in runs if r["arm"] != "oracle_global16"]
    addon_runs = [r for r in runs if r["arm"] == "oracle_global16"]
    arms = ["oracle_block", "oracle_global4", "oracle_local4"]
    table: list[dict[str, Any]] = []
    for condition in config["conditions"]:
        for arm in arms:
            rows = [r for r in primary if r["condition"] == condition and r["arm"] == arm]
            table.append({
                "condition": condition,
                "arm": arm,
                "h10": mean(r["test"]["primary_episode_mean"] for r in rows),
                "h20": mean(r["test"]["horizon_episode_aggregates"]["20"]["mean"] for r in rows),
                "action_response": mean(r["test"]["action_response_true_energy_normalized_mean"] for r in rows),
                "parameters": rows[0]["parameter_count"],
                "quality_pass": sum(bool(r["test"]["dense_quality_gate_pass"]) for r in rows),
                "b1_ms": mean(r["latency"]["complete_rollout_final_batch1_ms"]["median"] for r in rows),
                "b300_ms": mean(r["latency"]["complete_rollout_final_batch300_ms"]["median"] for r in rows),
                "eachstep_b300_ms": mean(r["latency"]["each_step_reconstruction_batch300_ms"]["median"] for r in rows),
                "speed_pass_b1": sum(bool(r["latency"]["complete_rollout_final_batch1_ms"]["speed_gate_pass"]) for r in rows),
                "speed_pass_b300": sum(bool(r["latency"]["complete_rollout_final_batch300_ms"]["speed_gate_pass"]) for r in rows),
            })

    contrasts = summary["paired_episode_bootstrap_contrasts"]
    contrast_lookup = {(r["condition"], int(r["training_seed"]), r["left_arm"], r["right_arm"]): r for r in contrasts}
    decision = {
        "status": "complete_diagnostic_pilot",
        "primary_runs_expected": 27,
        "primary_runs_completed": len(primary),
        "global16_addon_expected": 3,
        "global16_addon_completed": len(addon_runs),
        "mechanism_checks_passed": sum(item["status"] == "pass" for item in mechanisms),
        "reference_error_reproductions_passed": len(reproduction),
        "oracle_passes_dense_quality_gate_by_condition_arm": {
            f"{condition}/{arm}": sum(bool(r["test"]["dense_quality_gate_pass"]) for r in runs
                                        if r["condition"] == condition and r["arm"] == arm)
            for condition in config["conditions"] for arm in arms + ["oracle_global16"]
            if any(r["condition"] == condition and r["arm"] == arm for r in runs)
        },
        "oracle_passes_dense_speed_gate_by_condition_arm_batch": {
            f"{condition}/{arm}/B{batch}": sum(
                bool(r["latency"][f"complete_rollout_final_batch{batch}_ms"]["speed_gate_pass"])
                for r in runs if r["condition"] == condition and r["arm"] == arm
            )
            for condition in config["conditions"] for arm in arms + ["oracle_global16"]
            for batch in (1, 300)
            if any(r["condition"] == condition and r["arm"] == arm for r in runs)
        },
        "training_seed_population_inference": "not_supported; three seeds are descriptive pilot",
        "global16_parameter_confound": "message width and predictor input width both increase; bandwidth effect not isolated",
        "visual_escalation": "not evaluated by this toy oracle study",
    }
    (out / "DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    lines = [
        "# 已知正确坐标的 toy oracle 对照结果", "",
        "本轮在已分配 PBS compute node 完成 27 个主 runs 与 3 个 lowrank global16 附加 runs。模型直接使用生成器正交矩阵 `M`，固定 `s=(z-mean_z)@M`，只训练 predictor/message。它回答：第一轮的误差有多少来自坐标学习，已知正确坐标后局部模型、有限通信和 dense predictor 的差异如何。Oracle 得到生成器坐标，属于诊断对照，不是可部署方案。", "",
        "主实验的 train/dev/test 为 512/128/128 episodes，三个 seeds，各训练 1500 steps。误差使用 train-only delta energy 归一化；h10/h20 是 free-running 末步误差。action-response 是以真实 response energy 归一化的扰动响应差。延迟是在同一 allocation 重测 oracle 与原 dense checkpoint，计入初次坐标变换和最终重构。", "",
        "## 主实验", "",
        "| Condition | Arm | h10 | h20 | Action-response | Params | Dense quality seeds | B1 ms | B300 ms | Each-step B300 ms | Speed gate B1/B300 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['condition']} | {row['arm']} | {row['h10']:.5f} | {row['h20']:.5f} | {row['action_response']:.4f} | "
            f"{row['parameters']} | {row['quality_pass']}/3 | {row['b1_ms']:.3f} | {row['b300_ms']:.3f} | "
            f"{row['eachstep_b300_ms']:.3f} | {row['speed_pass_b1']}/3; {row['speed_pass_b300']}/3 |"
        )
    lines += [
        "", "质量门槛沿用第一轮 `h10 <= dense + max(0.1*dense, 0.02)`；速度门槛沿用完整 rollout 至少快 20%，B1 和 B300 分开判断。表中参数数不计入固定的 64×64 坐标矩阵。",
        "", "## 与第一轮 learned arm 的逐 seed 配对差分", "",
        "差分为 oracle 减原 learned arm；负值表示 oracle 误差较低。区间是每个训练 seed 内按 episode bootstrap 得到的 95% CI；它不是三个训练 seed 的总体区间。", "",
        "| Condition | Seed | Oracle arm - learned arm | Mean | Episode bootstrap 95% CI | Oracle - dense mean | Oracle - dense 95% CI |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for row in runs:
        condition, arm, seed = row["condition"], row["arm"], int(row["training_seed"])
        aligned = row["initializer_arm"]
        left = contrast_lookup[(condition, seed, arm, aligned)]
        dense = contrast_lookup[(condition, seed, arm, "dense")]
        lo, hi = left["bootstrap_95pct_mean_ci"]
        dlo, dhi = dense["bootstrap_95pct_mean_ci"]
        lines.append(f"| {condition} | {seed} | {arm} - {aligned} | {left['mean_paired_difference']:.6f} | [{lo:.6f}, {hi:.6f}] | {dense['mean_paired_difference']:.6f} | [{dlo:.6f}, {dhi:.6f}] |")

    lines += ["", "## global16 附加对照", "",
              "它只在 lowrank_coupled 做三个 seeds。message 从 4D 增至 16D，输入层也相应扩大，参数量随之上升；即使结果更好，也不能区分通信维数和容量的作用。", "",
              "| Seed | h10 | Params | Quality gate | B1 reduction | B300 reduction |", "|---:|---:|---:|---|---:|---:|"]
    for row in addon_runs:
        lines.append(f"| {row['training_seed']} | {row['test']['primary_episode_mean']:.5f} | {row['parameter_count']} | "
                     f"{row['test']['dense_quality_gate_pass']} | "
                     f"{row['latency']['complete_rollout_final_batch1_ms']['latency_reduction_vs_dense']:.1%} | "
                     f"{row['latency']['complete_rollout_final_batch300_ms']['latency_reduction_vs_dense']:.1%} |")
    lines += ["", "## 机制与原结果复现", "",
              f"机制检查通过 {sum(item['status'] == 'pass' for item in mechanisms)}/{len(mechanisms)}。复载并复算第一轮 learned/dense 与 global16 原 checkpoint 的全部 held-out horizon arrays 共 {len(reproduction)} 份；最大绝对差为 {max(float(x['max_abs_difference']) for x in reproduction):.3g}。逐模型值见 `reference_reproduction.json`。", "",
              "## 怎样解释这轮结果", "",
              "- 如果 oracle_block 明显优于第一轮 learned_block，且误差接近 dense，第一轮的重要瓶颈是从混合 latent 学坐标；这是已知 M 条件下的诊断上限。",
              "- 如果 oracle_block 的坐标已正确却仍落后 dense，局部 predictor 的函数形式、容量或当前优化预算仍不足。此结果不能简单归因于坐标学习。",
              "- 在正确坐标下，oracle_global4 与 oracle_local4 是等参数、同初始化的全局/局部消息范围对照，最直接检验跨块信息的价值。oracle_block 与 oracle_global4 之间还增加了消息参数和输入维数，因此只反映加入消息后的整体结构收益，不能单独归因于通信。仍需结合各 condition 的差异看，而不是把某个 seed 的优势泛化。",
              "- `dense_coupled` 是更广泛的交互压力测试，不证明真实视觉 dynamics 一定 dense 或不可分。",
              "- global16 是参数与通信同时增加的附加探索。全部结果均不支持视觉、C-SWM/LeWM、CEM、部署加速或闭环结论；是否进入视觉 pilot 需由并行视觉实验单独回答。", "",
              "全部逐 episode arrays、checkpoint、training.tsv 和 helper/source snapshot 保存在本 PBS run 目录。机制文件、JSON 摘要与本报告由 job 输出。GPU 利用率和显存按 30 秒采样写入 job.log。", "",
              "实验设计来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065", "",
              f"PBS job `{allocation['job_id']}`，compute host `{allocation['hostname']}`。", ""]
    (out / "REPORT.zh.md").write_text("\n".join(lines), encoding="utf-8")
    return decision


def main() -> None:
    import argparse
    from allocation_guard import ensure_allocation

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allocation = ensure_allocation(require_gpu=True)
    generate(args.output, allocation)


if __name__ == "__main__":
    main()
