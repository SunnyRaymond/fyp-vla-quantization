# Anchor-aligned bank handoff

## 当前状态

- 实现与 PBS wrapper 已通过 peer review PASS；本地 `py_compile`、`bash -n` 与轻量 preflight 已通过。
- 首次提交 `25222660.pbs101` 已结束：`F`、`Exit_status=1`、`resources_used.walltime=00:00:01`。
- 权威 PBS comment 为 `Job run ... failed`；无 experiment `job.log`，说明未创建实验输出目录，也未进入 runner、模型加载或训练。
- 启动原因已定位：wrapper 默认 dataset 路径误写为 `$STABLEWM_HOME/pusht/pusht_expert_train.h5`；集群实际文件为 `$STABLEWM_HOME/pusht_expert_train.h5`。

## 修正与重试

- 已在本地 wrapper 修正 dataset 默认路径，并重新通过轻量 syntax checks。
- bounded retry 已完成：`25223859.pbs101`，`qstat -x` 为 `F`、`Exit_status=0`、walltime `00:06:09`。
- 远端输出目录：
  `/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/anchor-aligned-bank/artifacts/25223859.pbs101/`
- `job_status=EXIT_STATUS=0`、`final_exit_status=RUNNER_EXIT_STATUS=0`；summary 状态为 `PREDICTOR_LEVEL_COMPLETE`。
- Paired mechanism gate **FAIL**：median episode recall120 delta `+0.00833`，strict improvement `4/8`，bad blocks `17→16`，worst recall `0→0.3`，standardized regret delta `−0.000689`；absolute gate **FAIL**（Treatment minimum `0.3`，middle stratum median `0.8667`）。按冻结规则停止，不进入 CEM/closed-loop。

## 证据路径

- 首次 PBS 历史状态：`qstat -x -f 25222660.pbs101`；PBS stdout `/home/users/ntu/yguo017/lewm_anchor_bank.o25222660` 为空。
- 首次失败发生在 wrapper file guard，未触发模型/HDF5/训练 I/O。
- 重试证据：本地 `artifacts/25223859.pbs101/anchor_aligned_bank_summary.json`、`job.log`、`job_status`、`final_exit_status.txt`、`gpu_info.csv`；权威 PBS 状态为 `qstat -x -f 25223859.pbs101`。
- 结果只回传小 summary、job log、status、GPU info；Treatment checkpoint 和 training rows 留在 cluster。
