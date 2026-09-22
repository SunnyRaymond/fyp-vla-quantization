# Selective teacher correction handoff

状态：**已完成唯一 bounded PBS；INCONCLUSIVE_STOP**。作业正常完成，但 calibration 没有满足预冻结 `0 < call_rate <= .25` 的 midpoint threshold；未进入 test quality/timing，不重跑、不放宽阈值。

- 实际模型：`gpt-5.6-luna`，reasoning=`xhigh`。
- PBS job：`25269182.pbs101`；`normal`，1 GPU、16 CPU、110 GB、30 分钟；权威终态 `F`、`Exit_status=0`、walltime `00:02:54`；qstat 证据：`local-status/25269182.pbs101/qstat_final.txt`。
- 远端运行目录：`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction/artifacts/25269182.pbs101`。
- calibration 实际覆盖 24 contexts、48 context/seed CEM trajectories×3 saved rounds=`144` trajectory blocks；最低 midpoint call rate=`37/144=.256944444444444`，高于预设上限 `.25`。
- summary=`INCONCLUSIVE`，`reason=no calibration midpoint has 0 < call_rate <= 0.25`；因此没有 test[584:592]、quality gate、risk gate 或 timing 结果。
- 证据已取回：`local-status/25269182.pbs101/selective_teacher_correction_summary.json`、`job.log`、`job_status`、`final_exit_status.txt`、`gpu_info.csv`、`execution_identity.txt`、`qstat_final.txt`。checkpoint、rows、banks 留在 cluster。
- `job.log` 保留 direct 5 秒 GPU telemetry；`gpu_info.csv` 仅静态 GPU 信息。official CEM/closed-loop=`NOT_RUN_BY_SCOPE`。
