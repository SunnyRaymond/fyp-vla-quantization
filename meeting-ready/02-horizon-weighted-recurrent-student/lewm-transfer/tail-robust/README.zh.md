# LeWM Phase 3：tail-robust EMA

本目录保存冻结的 predictor-level tail-robust experiment。正式作业
`24933682.pbs101` 已完成且 `Exit_status=0`；两个 arm 的 absolute predictor gate
均为 `GO`，但 `ema_tail_score` 相对 historical `512×3000` 的 no-regression gate
为 `NO-GO`，所以 primary gate 为 **NO-GO**。

完整结果见：[RESULT_LEWM_TAIL_ROBUST.zh.md](RESULT_LEWM_TAIL_ROBUST.zh.md)。

## Evidence

- [freeze](LEWM_TAIL_ROBUST_FREEZE.json)
- [protocol](PROTOCOL_LEWM_TAIL_ROBUST.zh.md)
- [formal summary](artifacts/24933682.pbs101/lewm_tail_robust_summary.json)
- [job log](artifacts/24933682.pbs101/job.log)
- [job status](artifacts/24933682.pbs101/job_status)

Artifact scope 仅包含 summary、log、status；没有拉取 checkpoint、`prepared_rows` 或
manifest。Snapshot、gate、paired delta、latency、causality、convergence 和 GPU
telemetry 的数字均绑定到 formal summary/job log；paired delta 是按 summary 的
`pairing_key` 做的轻量派生统计。

## Boundary

本轮只比较 `ema_score` 与 `ema_tail_score` 的 LeWM predictor-level EMA training/evaluation。
official CEM、full CEM viability、planner viability 和 closed-loop 均为
**NOT_RUN_BY_SCOPE**。
