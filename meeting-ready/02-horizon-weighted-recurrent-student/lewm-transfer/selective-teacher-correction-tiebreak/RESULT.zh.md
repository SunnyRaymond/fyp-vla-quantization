# Selective teacher correction tie-break：结果

状态：`runner COMPLETE`，质量门控 `FAIL`。本实验是 `25269182.pbs101` calibration 的 post-calibration refinement；旧作业的 `INCONCLUSIVE` 结论保持不变，没有覆盖。

## 终态与证据

- PBS job：`25273781.pbs101`；唯一 bounded job，运行于 `x1000c2s1b0n1`，walltime `00:04:24`。
- 权威 `qstat -xf`：`job_state=F`，`Exit_status=2`，`Stageout_status=1`。
- runner 证据：`job_status=EXIT_STATUS=0`、`final_exit_status=RUNNER_EXIT_STATUS=0`、summary `status=COMPLETE`。
- PBS wrapper 最后一行报：`exit: 0\r: numeric argument required`。这是上传脚本最后一行 CRLF 导致的 wrapper 终止状态不一致；因此不能把 PBS 记为正常 `Exit0`，但 runner 的完整 summary 已写出，未重跑。
- 本地 PBS wrapper 已将该末行规范为 LF，并重新通过 `bash -n`；这只修复后续复用入口，不改变本次作业或结果。
- 小型证据目录：`local-status/25273781.pbs101/`，含 `qstat_final.txt`、summary、`job.log`、`job_status`、`final_exit_status.txt`、`gpu_info.csv`、`execution_identity.txt`。GPU telemetry 在 `job.log` 中持续记录；`gpu_info.csv` 是静态 GPU 信息。

## Frozen design

Calibration source 是旧 `25269182.pbs101` 的 exact summary/bank，状态仍为 `INCONCLUSIVE`。本次在 compute 内重算 144 个 calibration trajectory blocks，验证 `u=1` 有 37 个，并得到 `g36=0.6875929732552584`、`g37=0.6465405171168568`、`tau_g=0.6670667451860576`，因此 calibration calls `36/144=0.25`。test 使用未见过的 `valid[584:592]`、selection seed `20300903`、144 trajectory blocks；路由仍是 `u>1` 或 `u=1 且 g>tau_g`，只用 main/sentinel student costs。

## Test metrics

- Teacher calls：`33/144 = 0.2291666667`，预算门控 `PASS`（上限 `.35`）。
- selective minus main 的 episode mean median：`-0.0911714809`，要求 `<= -0.10`，`FAIL`；严格改善 `7/8`，该项 `PASS`。
- catastrophic 定义为 main standardized regret `>=1.5`：`50` blocks，其中 teacher 捕获 `14`，capture `0.28`，要求 `>=.75`，`FAIL`。
- selective 对 analytic same-budget random：`PASS`。
- native latency reduction vs teacher300：`0.5735161101`，要求 `>=.30`，`PASS`。
- finite、shared pairing、provenance、calibration exact recompute：`PASS`。
- 总门控：`FAIL`，主要失败项是质量阈值略未达标和 catastrophic risk capture。

## Claim boundary

这只提供 independent fixed-observation post-calibration selective candidate-ranking evidence。`official CEM` 和 `closed-loop` 均为 `NOT_RUN_BY_SCOPE`；结果不支持部署或闭环成功声明。
