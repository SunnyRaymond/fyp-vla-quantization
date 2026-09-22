# LeWM Tail-Robust EMA Phase 3：实验结果

完整 frozen protocol、两个 EMA arms、snapshot trajectory、paired deltas、absolute
predictor gate、historical no-regression gate、GPU telemetry 与 scope boundary 见：

[lewm-transfer/tail-robust/RESULT_LEWM_TAIL_ROBUST.zh.md](../lewm-transfer/tail-robust/RESULT_LEWM_TAIL_ROBUST.zh.md)

一句话结论：`ema_tail_score` 通过 inherited absolute predictor gate，但 median
Spearman/top-30 `0.973554/0.816667` 低于 historical `512×3000` 的
`0.978317/0.866667`，因此预注册 primary gate 为 **NO-GO**。普通 `ema_score`
在 Spearman median/minimum 和 relative latent MSE 上均略优于 tail-emphasis；本轮
worst-block 改善来自 EMA，而不是 top-2 tail weighting。未运行 official CEM、planner
viability 或 closed-loop。

正式作业为 `24933682.pbs101`（`Exit_status=0`，walltime `00:03:39`）；job log
包含约 5 秒一次的 GPU utilization/VRAM telemetry。
