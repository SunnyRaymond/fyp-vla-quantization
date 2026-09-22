# LeWM EMA temporal confirmatory predictor

本目录保存 LeWM recurrent student 的单次 confirmatory predictor-only 实验。实验在同一条
`512`-row training trajectory 上并行维护 `online` control 与 `EMA(decay=0.999)` primary，
并在全新的 episode-disjoint temporal anchors 上比较 ranking fidelity。实验不改变 h256
shared recurrent architecture，不测试 tail weighting，不运行 official CEM、planner viability
或 closed-loop。

## 结果

正式作业 `25142797.pbs101` 正常结束（`Exit_status=0`，walltime `00:03:22`，
`exec_host=x1000c1s3b0n0/0*16`）。EMA terminal `step_3000` 的整体 Spearman median/minimum
为 `0.815925/-0.344404`，top-30 median/minimum 为 `0.466667/0`，relative latent MSE
median 为 `0.147960`。因此 inherited absolute predictor gate 与 early/middle/late
stratum protection gate 均未通过，primary gate 为 **NO-GO**。

EMA 在 `early` stratum 通过局部 median threshold（Spearman `0.962553`、top-30
`0.800000`），但 `middle`（`0.486430/0.283333`）与 `late`
（`0.280386/0.266667`）明显失败。与 concurrent online 的 episode-level paired
delta 为 Spearman `-0.018833`、top-30 `-0.033333`，只有 `2/8` episodes 满足 joint
non-worsening improvement，不能支持 EMA 机制改善。

Predictor latency 为 EMA `1.8412 ms`、teacher `20.5635 ms`，reduction `91.0465%`；
这只覆盖 cached latent + packed action prefix → five-step predictor，不代表 encoder、
CEM、environment 或 closed-loop 加速。训练收敛、EMA isolation、same-trajectory pairing
与 causality checks 均通过；GPU telemetry 由作业内每 5 秒写入 `job.log`。

完整解释见 [RESULT_LEWM_EMA_TEMPORAL_CONFIRM.zh.md](../../reports/RESULT_LEWM_EMA_TEMPORAL_CONFIRM.zh.md)。

## 冻结 scope

- fresh selection 固定为 `valid[520:528]`，8 episodes × 3 anchors × 2 action-prefix
  seeds = 48 blocks，每 block 300 candidates；不使用结果选择 episode 或 snapshot。
- 复用 Phase 2 formal `512` training bank；fresh rows/teacher targets 只在 PBS compute
  node 生成，未返回 prepared rows 或 checkpoint。
- official CEM、planner viability、closed-loop：**NOT_RUN_BY_SCOPE**。

## Artifact

- [summary JSON](artifacts/25142797.pbs101/lewm_ema_temporal_confirm_summary.json)
- [job log / GPU telemetry](artifacts/25142797.pbs101/job.log)
- [job status](artifacts/25142797.pbs101/job_status)

方法与 reporting 组织参考已核实的 [Scientific Agent Skills, arXiv:2609.00065](https://arxiv.org/abs/2609.00065)。该文献只作为方法参考，不是本实验的结果证据。
