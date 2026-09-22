# LeWM state-action prefix GRU

这是 temporal-balanced training 之后的单次、两臂、predictor-only structure
experiment。`balanced_base` 是原始 h256 recurrent student；`state_action_prefix_gru`
只加入一个 shared 64-D state-action prefix GRU，保留原始 transition/output/free-running
path。

正式 job `25158955.pbs101` 已完成：`Exit_status=0`、walltime `00:04:58`、
`exec_host=x1000c1s3b0n1/0*16`。Fresh confirmation 固定为 `valid[536:544]`，
8 episodes、3 anchors、2 seeds（`20300937/20300938`），共 48 blocks × 300
candidates。GRU terminal overall Spearman/top-30 median 为 `0.841137/0.566667`；
相对 balanced base 的 paired mechanism gate **GO**（median ΔSpearman `+0.023855`、
Δtop-30 `+0.025000`、joint `5/8`），但 treatment absolute、三个 stratum 和
primary gate 均 **NO-GO**。GRU 相对 base predictor overhead `41.56%`，超过冻结的
`35%` latency guard。

完整结果见 [RESULT_LEWM_STATE_ACTION_PREFIX_GRU.zh.md](RESULT_LEWM_STATE_ACTION_PREFIX_GRU.zh.md)。

## 文件

- [冻结协议](LEWM_STATE_ACTION_PREFIX_GRU_FREEZE.json)
- [中文 protocol](PROTOCOL_LEWM_STATE_ACTION_PREFIX_GRU.zh.md)
- [runner](run_lewm_state_action_prefix_gru.py)
- [PBS wrapper](run_lewm_state_action_prefix_gru.pbs)
- [summary JSON](artifacts/25158955.pbs101/lewm_state_action_prefix_gru_summary.json)
- [job log / 5-second GPU telemetry](artifacts/25158955.pbs101/job.log)
- [job status](artifacts/25158955.pbs101/job_status)

## Scope

本轮停在 predictor-level。official CEM、planner viability、closed-loop 均为
**NOT_RUN_BY_SCOPE**；latency 只覆盖 cached latent + five-step predictor，不代表
encoder、CEM、environment 或 closed-loop speedup。没有回传 checkpoint、prepared rows
或 HDF5。
