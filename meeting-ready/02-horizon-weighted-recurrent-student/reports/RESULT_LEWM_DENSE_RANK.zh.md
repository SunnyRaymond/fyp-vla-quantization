# LeWM Dense Query + Rank Distillation：实验结果

完整的三臂协议、step 500/1000/1500 snapshot、paired block deltas、absolute gates、
GPU telemetry 与 scope boundary 见：

[lewm-transfer/dense-rank/RESULT_LEWM_DENSE_RANK.zh.md](../lewm-transfer/dense-rank/RESULT_LEWM_DENSE_RANK.zh.md)

一句话结论：`dense_rank` 在相同 h256 recurrent student 上把 64-candidate train
coverage 与 top-6-vs-7..12 teacher rank loss 结合后，相对 `dense_latent` 的 step-1500
top-30 median 提升 `+0.166667`（`14/16` blocks 改善），Spearman median 提升
`+0.122187`；shuffled-label control 没有复现收益。冻结的 dense-rank screening gate
为 **GO**，但 absolute ranking gate 仍 **FAIL**，因此 predictor feasibility 仍为
**NO-GO**。本轮没有运行 official CEM、planner viability 或 closed-loop。

正式作业为 `24908446.pbs101`（`Exit_status=0`，walltime `00:05:32`）；三臂参数量均
`775,872`，`dense_rank` predictor-only latency `1.645056 ms`，相对 teacher reduction
`91.7548%`。
