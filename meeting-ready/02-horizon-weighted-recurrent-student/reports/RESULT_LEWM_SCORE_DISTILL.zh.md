# LeWM Bounded Score Distillation：实验结果

完整的三臂协议、step 500/1000/1500 snapshot、paired block deltas、screening/absolute
gates、GPU telemetry 与 scope boundary 见：

[lewm-transfer/score-distill/RESULT_LEWM_SCORE_DISTILL.zh.md](../lewm-transfer/score-distill/RESULT_LEWM_SCORE_DISTILL.zh.md)

一句话结论：`score_distill` 在相同 h256 recurrent student 和相同 held-out protocol
下，相对已验证的 `pairwise_rank` 把 step-1500 Spearman median 从 `0.701261` 提高到
`0.974011`、top-30 median 从 `0.433333` 提高到 `0.816667`，relative latent MSE
从 `0.025179` 降到 `0.013728`；paired deltas 分别为 `+0.268554`、`+0.300000` 和
`-0.007516`。不过仍有 `15/16` 而非 `16/16` 个 positive top-30 blocks，minimum
top-30 为 `0`，所以冻结 screening gate 与 absolute predictor gate 均为
**NO-GO**。本轮没有运行 official CEM、planner viability 或 closed-loop。

正式作业为 `24916520.pbs101`（`Exit_status=0`，walltime `00:03:44`）；`score_distill`
predictor-only latency 为 `1.852416 ms`，相对 teacher reduction `91.6772%`。
