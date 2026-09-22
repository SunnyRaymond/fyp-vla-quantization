# LeWM Action-History Conditioner：实验结果

完整结果、paired block deltas、作业关系与 evidence boundary 见：

[lewm-transfer/conditioner/RESULT_LEWM_ACTION_HISTORY_CONDITIONER.zh.md](../lewm-transfer/conditioner/RESULT_LEWM_ACTION_HISTORY_CONDITIONER.zh.md)

一句话结论：zero-init latest-3 action-history AdaLN-style affine conditioner 让
Spearman median `0.415374 → 0.453163`、relative latent MSE median `0.012116 →
0.011190`，但 top-30 median `0.266667 → 0.250000`，仍为 `predictor_feasibility =
NO-GO`；没有明显、稳定的 planner-facing ranking 提升。正式 conditioner 参数量为
`845,888`，predictor latency `2.45914 ms`，相较原 baseline `1.85293 ms` 增加约
`32.72%`。本轮没有运行 official CEM、planner viability 或 closed-loop。
