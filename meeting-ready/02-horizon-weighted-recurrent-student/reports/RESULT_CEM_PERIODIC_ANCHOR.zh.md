# DINO-WM PushT：CEM periodic teacher anchoring 结果

## 结论

正式作业 `25158912.pbs101` 的计算阶段完成了冻结协议要求的六个 fixed-observation
PushT cases（`0, 1, 2, 4, 5, 7`）。`periodic_teacher_anchor_P5` 在 iteration 30
相对 `student_only` 提供了清晰的相对 correction：first-action RMS、trajectory
AUC 和 teacher-cost regret 的 median 都下降，三项均有 `4/6` cases 不变差；但
两个 absolute fidelity gates 都失败。`shuffled_anchor_P5` 没有同时复现这两项相对
改善，因此 negative-control gate 通过。

因此，本 frozen recipe 的决策为 **NO-GO**。P=5 的 periodic full-teacher anchoring
可以部分纠正 student 的 CEM 漂移，但不能把最终 proposal state 拉回到可替代
teacher 的绝对 fidelity。本结果只支持 fixed-observation 的六-case mechanism
diagnosis；不支持 closed-loop success、environment interaction 或 population-level
推断。按冻结边界，不放宽 gate、不用同一 recipe 重跑，也不推进 closed-loop。

## 实验范围与执行状态

- backend：official DINO-WM PushT；固定 observation；六个预选 case 为
  `0, 1, 2, 4, 5, 7`，对应 eval seeds `1, 100, 199, 397, 496, 694`。
- CEM：`30` iterations、每轮 `300` candidates、`top30`、horizon `H=5`、packed
  action dim `10`；诊断 checkpoints 为 `1/5/10/15/20/25/30`。
- periodic anchors：1-based rounds `5, 10, 15, 20, 25`，明确不包含 round `30`。
- 三臂为 `student_only`、`periodic_teacher_anchor_P5` 和
  `shuffled_anchor_P5`；三臂共享 common CPU innovations 与 teacher reference。
- teacher reference budget：`180` calls / `54000` candidates，共享于各 arm，且不
  计入 intervention-arm budgets。
- treatment mechanism budget：`30` teacher calls / `9000` candidates；另有
  diagnostics `12` calls / `3600` candidates。
- 六个 case JSON 均完整生成；summary 的 validity 为 all cases/rounds complete、
  all outputs finite、无 silent fallback，且 teacher reference zero drift。
- GPU 为 `NVIDIA A100-SXM4-40GB`；作业计算阶段 telemetry 为 `99–100%` GPU
  utilization（显存约 `23.1 GiB`，总显存 `40 GiB`）。

### Wrapper failure 与科学结果的区分

原 runner 在最后的 summary aggregation 阶段因 `record` 顶层缺少
`trajectory_auc` 而触发 `KeyError`，`runner_exit_status` 与 `final_exit_status` 均为 `1`。这不是
case 计算或科学 gate 的失败：日志已经记录六个 case 全部 `completed=true`，且
六份 case JSON 保留了完整的 arm-level trajectory 与 round-level measurements。
随后从这些已完成 case JSON、使用修复后的同一冻结 gate 逻辑离线恢复了
`cem_periodic_anchor_summary.json`。因此本文将 wrapper 的汇总 bug 作为执行
完整性的说明，不把它改写成实验失败。

## 主要 fixed-observation 结果

下表中的 treatment 数值均为六 case 的 case-level aggregation；trajectory AUC 按
protocol 定义为 30-step first-action RMS trajectory 的 case-level mean。括号中的
delta 为 `periodic_teacher_anchor_P5 - student_only`，负值表示 treatment 更好。

| 指标（iteration 30 或 trajectory） | `student_only` | `periodic_teacher_anchor_P5` | 相对结果 | 冻结 gate |
|---|---:|---:|---:|---|
| first-action RMS median | `0.7317022681` | `0.6429772973` | median delta `-0.1906259060`；`4/6` 不变差 | absolute `<=0.15`：**FAIL** |
| first-action coordinate abs max | — | `1.9762158394` | — | absolute `<=0.25`：**FAIL** |
| trajectory AUC median | `0.6152800977` | `0.5493363014` | median delta `-0.1207193355`；`4/6` 不变差 | median strictly lower 且 `>=4/6`：**PASS** |
| teacher-cost regret median | `0.0960179530` | `0.0786698610` | median delta `-0.0358193964`；`4/6` 不变差 | median `<=` 且 `>=4/6`：**PASS** |

因此，所有 treatment-vs-student 的 relative improvement gates 均通过，但两个
absolute fidelity gates 均失败，整体 decision 仍为 **NO-GO**。在 iteration 30，
treatment 的 teacher top-30 overlap median 只有 `0.0333333`（约 `1/30`），说明
anchor 之间重新发生了 proposal drift；round 30 本身没有 teacher anchor，不能把
前面 anchor round 的 correction 延伸为最终 fidelity。

## Negative control

`shuffled_anchor_P5` 使用相同的 teacher costs，但在 anchor round 打乱
cost-action correspondence。它没有同时复现 treatment 的两项主要相对下降：

- iteration-30 first-action RMS 只有 `3/6` cases 不变差；
- trajectory AUC 只有 `2/6` cases 不变差。

因此 negative-control gate 为 **PASS**（`negative_control_reproduces_both=false`）。
该结果支持 anchor 的 correction 依赖正确的 cost-action correspondence，但不能
抵消 treatment 自身绝对 fidelity gate 的失败。

## 决策与 claim boundary

本轮允许的 claim 只有：在冻结的 fixed observation、六个预选 PushT cases、官方
CEM semantics、共同 CPU innovations 和 P=5 periodic teacher-anchor mechanism 下，
该机制相对 student-only trajectory 可产生明确但不足以替代 teacher 的 correction。

本轮不支持：

- closed-loop PushT success、environment interaction 或 official planner deployment；
- 将六个 case、30 个 iterations 或 candidate 数量当作独立 statistical replicates；
- population-level generalization、native deployment latency/memory 或 teacher
  replacement claim。

`closed_loop` 按 scope 为 **NOT RUN**。absolute fidelity 已失败，所以没有理由把
这一路径直接推进到 environment evaluation。

## 下一步建议

当前证据更适合用于选择下一种不同 mechanism，而不是重复本 recipe。可将
trust-region/state correction 或显式 teacher fallback 作为后续候选方向；二者在
本报告中均尚未运行、尚未冻结 protocol，也不应被写成已验证结果。下一步仍应先在
fixed-observation 层定义 absolute fidelity 与 planner-facing gate，再决定是否有
资格进入 closed-loop；不应先用 closed-loop 结果替当前 absolute gate 失败辩护。

## 可复核输入

- [recovered summary](../artifacts/25158912.pbs101/cem_periodic_anchor_summary.json)
- [case 00](../artifacts/25158912.pbs101/case_00.json)、[case 01](../artifacts/25158912.pbs101/case_01.json)、[case 02](../artifacts/25158912.pbs101/case_02.json)、[case 04](../artifacts/25158912.pbs101/case_04.json)、[case 05](../artifacts/25158912.pbs101/case_05.json)、[case 07](../artifacts/25158912.pbs101/case_07.json)
- [freeze](../config/CEM_PERIODIC_ANCHOR_FREEZE.json)；[protocol](../config/PROTOCOL_CEM_PERIODIC_ANCHOR.zh.md)
- [job log](../artifacts/25158912.pbs101/job.log)；[job status](../artifacts/25158912.pbs101/job_status.txt)；[GPU telemetry](../artifacts/25158912.pbs101/gpu_usage.csv)

方法支持（非实验依据）：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv:2609.00065
