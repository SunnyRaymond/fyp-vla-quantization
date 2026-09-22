# LeWM temporal-balanced training

这是 Phase 3 fresh multi-anchor observation 之后的单次 confirmatory predictor-only
experiment。唯一改变是 512-context training anchor distribution：control 使用原
`anchor=0`，treatment 按 ordinal `mod 3` 平衡到 early/middle/late，counts
`171/171/170`；architecture、loss、candidate bank、initialization、schedule、
optimizer、batch 和 3000 updates 全部保持不变。

正式成功 job 为 `25152151.pbs101`：`Exit_status=0`、walltime `00:02:37`、
`exec_host=x1000c1s3b0n0/0*16`。Fresh confirmation 使用 selection 后的全新
`valid[528:536]`，8 episodes、3 anchors、2 seeds（`20300927/20300928`），共
48 blocks × 300 candidates。balanced treatment 的 secondary paired mechanism
gate 为 **GO**（median ΔSpearman `+0.108159`、Δtop-30 `+0.233333`、joint
`6/8`），但 absolute predictor gate、三 strata gate 和 primary gate 均
**NO-GO**：terminal overall Spearman/top-30 median `0.759380/0.483333`。

完整结果见 [RESULT_LEWM_TEMPORAL_BALANCED_TRAIN.zh.md](RESULT_LEWM_TEMPORAL_BALANCED_TRAIN.zh.md)。

## 文件

- [冻结协议](LEWM_TEMPORAL_BALANCED_TRAIN_FREEZE.json)
- [中文 protocol](PROTOCOL_LEWM_TEMPORAL_BALANCED_TRAIN.zh.md)
- [runner](run_lewm_temporal_balanced_train.py)
- [PBS wrapper](run_lewm_temporal_balanced_train.pbs)
- [summary JSON](artifacts/25152151.pbs101/lewm_temporal_balanced_train_summary.json)
- [job log / 5-second GPU telemetry](artifacts/25152151.pbs101/job.log)
- [job status](artifacts/25152151.pbs101/job_status)

首次 job `25150832.pbs101` 仅因 summary 序列化包含内部 tensors 失败；修复输出层后
按同一冻结设计重跑为 `25152151.pbs101`，不改变实验条件。未回传 prepared rows、
checkpoint 或 HDF5。

## Scope

本轮停在 predictor-level。official CEM、planner viability、closed-loop 均为
**NOT_RUN_BY_SCOPE**；predictor latency 只覆盖 cached latent + five-step predictor，
不代表 encoder、CEM 或 environment speedup。
