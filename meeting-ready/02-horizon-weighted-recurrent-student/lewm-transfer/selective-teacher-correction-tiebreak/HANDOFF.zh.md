# Selective teacher correction tie-break handoff

状态：runner 已完成；质量门控 FAIL。PBS 已到 F，但 wrapper 因 CRLF 最后一行报 Exit2；不重跑。

- 实际模型：`gpt-5.6-luna`，reasoning=`xhigh`。
- 提交时间（本地）：2026-09-22 20:04:43 +08:00
- PBS job：`25273781.pbs101`；`normal`，1 GPU、16 CPU、110 GB、30 分钟；提交脚本：`run_selective_teacher_correction_tiebreak.pbs`。
- 远端实验目录：`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction-tiebreak`。
- refinement source：exact `25269182.pbs101` summary 与 `fresh_calibration_cem_trajectory_banks.pt`；原 summary 的 `INCONCLUSIVE` 结论不被改写。
- routing：compute 内重算 calibration bank 全部 144 个 `u/g` 并逐项对齐 summary，再断言 `u=1` count=37、g36/g37/tau、calls=36；路由只用 main/sentinel student costs，teacher-free。
- test：独立 `valid[584:592]`，selection=`20300903`，prefix seeds=`20301101/20301102`，innovation base=`20301104`；144 trajectory blocks。test 未使用于 threshold。
- 上传内容仅 FREEZE、PROTOCOL、runner、PBS wrapper；checkpoint、rows、banks 留在 cluster。
- 终态：权威 `qstat -xf` 为 `job_state=F`、`Exit_status=2`、`Stageout_status=1`、walltime `00:04:24`；`job_status=EXIT_STATUS=0`、`final_exit_status=RUNNER_EXIT_STATUS=0`、summary `status=COMPLETE`。
- Exit2 原因：PBS wrapper 最后一行 `exit: 0\r: numeric argument required`，属于 CRLF 终止状态异常；runner 结果完整，未重跑。
- 关键结果：calls `33/144=.2291667`；primary delta `-.09117148`（阈值 `-.10`，FAIL）；strict improve `7/8`；catastrophic capture `14/50=.28`（FAIL）；analytic random PASS；latency reduction `.5735161` PASS；overall FAIL。
- 小证据已取回至 `local-status/25273781.pbs101/`：summary、`job.log`、`job_status`、`final_exit_status.txt`、`gpu_info.csv`、`execution_identity.txt`、`qstat_final.txt`。
- 旧 `25269182.pbs101` 的 `INCONCLUSIVE` 保留；不进入新实验或 closed-loop。
