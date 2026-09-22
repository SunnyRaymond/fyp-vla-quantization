# LeWM state-action prefix GRU：正式结果

正式 job `25158955.pbs101` 正常完成（`Exit_status=0`、walltime `00:04:58`、
`exec_host=x1000c1s3b0n1/0*16`）。本轮为单一、两臂、predictor-only structure
experiment：`balanced_base` 复现 temporal-balanced h256 recurrent student，
`state_action_prefix_gru` 新增一个 shared 64-D state-action prefix GRU。两臂共享
512 balanced contexts、anchor counts `171/171/170`、initial state、optimizer、
candidate slates、schedule、loss 和 3000 updates。

Fresh test 是全新的 `valid[536:544]`，排除旧 heldout/train 与 `valid[520:536]`；
共 8 episodes × 3 anchors × 2 seeds = 48 blocks，每 block 300 candidates。

## Terminal gate

| arm | Spearman med/min | top-30 med/min | relative MSE | positive S/top-30 | absolute |
|---|---:|---:|---:|---:|---|
| balanced base | `0.824964/-0.624814` | `0.533333/0` | `0.100185` | `42/48, 46/48` | **NO-GO** |
| state-action prefix GRU | `0.841137/-0.639435` | `0.566667/0` | `0.094277` | `43/48, 46/48` | **NO-GO** |

Treatment 的 early/middle/late stratum Spearman medians 为
`0.921711/0.805193/0.847419`，top-30 medians 为 `0.650000/0.500000/0.633333`，
均未达到预注册 `0.95/0.75`，故 stratum gate 与 combined primary gate 均
**NO-GO**。

## Mechanism、latency 与完整性

以 episode 为 replicate、先聚合六个 nested blocks 后，GRU−base 的 median deltas
为 Spearman `+0.023855`、top-30 `+0.025000`，joint improvement/non-worsening
为 `5/8`，secondary mechanism gate **GO**，但不能替代 absolute gate。

Base / GRU predictor latency 分别为 `1.565696/2.216448 ms`，teacher reduction
分别为 `91.8845%/88.4171%`；GRU 相对 base overhead `41.5631% > 35%`，所以
secondary latency guard **NO-GO**。两臂 convergence、finite metrics、causality
（prefix 1–4 max difference `0`）均通过。参数量为 base `775,872`、GRU `844,096`，
新增 `68,224`。

GPU telemetry 记录 60 个约 5 秒 samples；PBS 记录无 GPU health/Xid/ECC error，
max VRAM 约 `684 MB`，SM utilization 峰值 `42%`。official CEM、planner viability
和 closed-loop 均为 **NOT_RUN_BY_SCOPE**。

完整冻结设计、snapshot、paired episode deltas、telemetry 与证据文件见
[lewm-transfer/state-action-prefix-gru/RESULT_LEWM_STATE_ACTION_PREFIX_GRU.zh.md](../lewm-transfer/state-action-prefix-gru/RESULT_LEWM_STATE_ACTION_PREFIX_GRU.zh.md)。
