# Goal 结束记录

2026-09-27，按用户更新后的范围结束：取消 LpWM dense 和 PLDM formal，LpWM sparse 留给用户自行监测。结束不表示原定全部复现已完成。

| 工作 | 最终记录 |
|---|---|
| LeWM × FastLeWM × CEM 10/30 小实验 | 完成：50 tasks × 3 seeds；固定初始 observation 的 planning 计时下，FastLeWM 相对 LeWM 约 5×。两模型保留各自官方配置，不能将差异唯一归因于模型架构；质量 interaction 的置信区间跨零，尚无无损降预算结论。 |
| FastLeWM exact iteration cache | 完成：8 个 B=1 contexts，60 次 encode 请求压至 2 次实际计算，结果 byte-exact；matched median latency 降低 57.1%。这是有限范围工程结果，未验证 B=50 或 closed-loop，不能与上述 5×相乘。 |
| Goal-query compression pilot | NO-GO：现有排序/elite fidelity 与成本门槛未通过；未继续扩大。 |
| LpWM dense `25571463.pbs101` | 排队期间用户取消；qdel 成功，F + terminated；未开始正式训练。 |
| PLDM formal `25572927.pbs101` | 排队期间用户取消；qdel 成功，F + terminated；CPU preparation 和完整成本 calibration 保留，无正式复现质量结果。 |
| LpWM sparse `25571461.pbs101` | 用户通知完成后补录：F / Exit_status=0 / Stageout_status=1，walltime 08:07:51；2 epochs、61,930 updates 与原生 50-case MPC 完整完成，21/50=42%。单臂运行有效；dense 已取消，没有 sparse-over-dense 结论。 |

两个取消作业的终态查询未提供 Exit_status/Stageout_status，不推断其值。相关 agent 监控已停止，不再提交实验或自动跟进 sparse。

## Sparse 完成后补充的 insights（2026-09-27）

[论文 v1](https://arxiv.org/pdf/2608.22764v1) Appendix H.1 / Figure 9 对应 `MLP ∘ LTI(k),D=384` sparse tuned cell 的 closed-loop success 为 **31% ± 2%**（三个 planning seeds 的 std）；本轮单 seed 为 42%。这提供了小 predictor cell 的恰当描述性参照，不能与主实验约 95% 直接比较。单次高出论文均值 11 个百分点不能称改进，也未复现论文三 seed 均值；论文 seed IDs / 逐 task identity 未披露，差异来源未识别。

原生 MPC 1–10 次的成功任务数为 **15、20、20、20、21、21、21、21、21、21**。第 5 次之后没有新增成功，是研究失败任务“无进展”识别和外层预算分配的机制信号。它只是这一 seed / checkpoint 的事后观察；未测提前停止省时，不建立普遍无损减少 MPC 的结论，也不能推导减少 CEM 内层 steps 无损。

两轮 validation loss 从 0.0132 降至 0.0115；imagined-vs-real latent discrepancy 随执行前缀变长而增大（native global norm 29.28→128.80）。目前应优先诊断失败轨迹的长 rollout、action response 与 goal-cost 对齐，不能把 one-step loss 改善等同于规划质量。该 discrepancy 不是 per-task mean，也不是到 goal 的 objective；state-distance 混合位置、角度、速度，不能单独判定控制退化或提前停止。

本轮产生了一个可用的官方 `mlp_var,D=384` sparse checkpoint，但尚未提取分析官方训练日志的 active fraction，也未分析 support stability 或测量 sparse-kernel speedup。训练、goal selection 与 harness 不同，**42% 不能直接与主实验的 LeWM/FastLeWM 成功率比较**；dense 未运行也使 sparse gain 无法估计。下一步若重启研究，先分析这 29 条失败任务的真实动作响应与 cost 排序，而不凭平台期直接加大或削减预算；本次仅总结，未提交新实验，goal 保持结束。

细节、指标定义、原始结果及证据限制见 [LpWM sparse 结果](../reproduction/lpwm/RESULT.zh.md)。

详细证据：

- [Model × search-budget](lewm-fastlewm-paired/RESULT.zh.md)
- [Exact cache](fastlewm-exact-iteration-cache/RESULT.zh.md)
- [LpWM sparse 最终结果](../reproduction/lpwm/RESULT.zh.md)
- [PLDM 阶段记录](../reproduction/pldm/RESULT.zh.md)
