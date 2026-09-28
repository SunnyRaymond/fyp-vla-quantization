# LeWM PushT adaptive teacher schedule 结果

## 执行状态

实验按冻结配置完成，最终状态为 `PASS`。PBS 作业 `25432397.pbs101` 在 `x1000c1s3b0n1` 结束，历史 qstat 状态为 `F`、`Exit_status=0`，墙钟时间为 10 分 05 秒。作业使用 1 张 A100；GPU 利用率和显存每 5 秒记录到 `job.log`，监控进程在作业结束时由 wrapper 清理。

主 checkpoint 为 `cem-distribution-distill/artifacts/25239551.pbs101/treatment_step1000.pt`；summary 中的 provenance 与冻结来源一致：`cem_distribution_distill` / `treatment` / 额外更新 1000 步。fresh evaluation 使用 `valid[592:600]`，共 8 episodes、48 条配对 trajectory；四臂共享冻结的 innovations，均完成 30 rounds × 300 candidates × top-30。

## Quality gate

冻结 primary 是 round 30 的 episode-level paired teacher-objective gap。每个 schedule 的 episode delta 用对应 pair 的 teacher-only round-1 300-candidate population standard deviation 归一化；数值越低越好。gate 要求 median delta 不高于 `-0.10`，且至少 5/8 episodes 严格改善。

| Schedule | 相对 student-only 的 median delta | 严格改善 episodes | Quality gate |
|---|---:|---:|---|
| `uniform_teacher7` | -1.099 | 8/8 | PASS |
| `late_teacher7` | -2.316 | 8/8 | PASS |

两种 schedule 均满足质量门槛。episode-level `uniform_teacher7 - late_teacher7` 的 median delta 为 `+1.378`，因此在本次 primary 上 late schedule 的 teacher objective 更低。相对 teacher-only 的 round-30 median gap 分别为 student-only `3.924`、uniform `2.796`、late `0.554`；final first-action L2 drift median 分别为 `5.124`、`4.313`、`4.575`（teacher-only 为 0）。round 10/20 保留为诊断，不参与 primary gate。

## Latency gate

计时覆盖完整 30-round adaptive trajectory，包含候选生成、H2D、打分、top-30 和分布更新；CUDA 同步，3 次 warmup 后对 48 条 trajectory 各计时 5 次，每臂 240 个测量样本。

| Arm | Mean latency | P95 latency | 相对 teacher-only 减少 |
|---|---:|---:|---:|
| `student_only` | 62.49 ms | 71.86 ms | 89.43% |
| `uniform_teacher7` | 186.30 ms | 193.13 ms | 68.49% |
| `late_teacher7` | 186.04 ms | 195.67 ms | 68.54% |
| `teacher_only` | 591.30 ms | 607.09 ms | — |

两个 7-call schedule 都超过冻结的 30% latency reduction 门槛。schedule calls、finite 输出、shared provenance、48 条配对轨迹完整性和其余 validity checks 均通过；整体 gate 为 `PASS`。

## 结论边界与证据

结果支持：在本次固定 observation 的 fully adaptive CEM 机制评估中，uniform 与 late 两种 7-call teacher schedule 都降低了 student-only 的 round-30 teacher-objective gap，同时比 teacher-only 更快；late schedule 的 primary median 更低。该结果不证明 official CEM planner 部署效果、PushT closed-loop 成功率或实际任务成功。`official CEM` 和 `closed-loop` 均为 `NOT_RUN_BY_SCOPE`。

本地回收的原始证据位于 `local-status/results/25432397.pbs101/`：summary、job log、job/final status、GPU 信息、execution identity 和历史 qstat。模型、dataset、banks 与完整 checkpoint 保留在 cluster。
