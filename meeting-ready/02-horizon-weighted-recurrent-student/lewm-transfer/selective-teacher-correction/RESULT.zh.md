# Selective teacher correction 结果

状态：**INCONCLUSIVE_STOP**。作业正常完成，但预冻结的 calibration threshold 条件没有可部署解，因此没有进入 test quality/timing，也没有放宽阈值或重跑。

## 执行证据

- PBS：`25269182.pbs101`，`job_state=F`，`Exit_status=0`，`resources_used.walltime=00:02:54`，compute host=`x1000c0s5b0n1`；权威记录见 `local-status/25269182.pbs101/qstat_final.txt`。
- 远端 runner summary：`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/selective-teacher-correction/artifacts/25269182.pbs101/selective_teacher_correction_summary.json`。
- 本地小证据：`local-status/25269182.pbs101/` 下的 `selective_teacher_correction_summary.json`、`job.log`、`job_status`、`final_exit_status.txt`、`gpu_info.csv`、`execution_identity.txt`、`qstat_final.txt`。
- `job.log` 记录了 direct `nvidia-smi` 5 秒 telemetry；`gpu_info.csv` 是静态 GPU 信息（A100-SXM4-40GB）。checkpoint、rows 和 CEM banks 留在 cluster。

## Calibration 结果

- calibration 使用 `valid[576:584]`，selection seed=`20300903`，排除旧 valid prefix `[0:576)`；8 episodes×3 anchors×2 action-prefix seeds，形成 24 contexts、48 条 context/seed CEM trajectories；保存 rounds=`10,20,30`，共 `144` trajectory blocks（每条 trajectory 3 个 round block），每 block 300 candidates。
- uncertainty 按冻结的 `u=1-|top30_main∩top30_sentinel|/30`，main/sentinel top30 均使用 pinned `torch.topk(..., largest=False, sorted=True)`。
- 19 个 distinct uncertainty levels 产生 midpoint candidates；严格规则是 `u > tau`，且 calibration call rate 必须满足 `0 < q <= .25`。最低 call-rate candidate 是 `tau=0.983333333333333`，调用 `37/144=0.256944444444444`，仍高于 0.25；因此没有 feasible threshold。
- summary status=`INCONCLUSIVE`，reason=`no calibration midpoint has 0 < call_rate <= 0.25`。此结果依照 freeze 的 `INCONCLUSIVE_STOP`，没有使用 test block、teacher labels 或跨 test 排名来修改阈值。

## 证据边界

本轮没有生成 `valid[584:592]` test bank，也没有执行 selective quality、risk、analytic-random comparison 或 native timing；因此 selective teacher correction 的 quality/latency gates 均为 **未评估**，不能宣称通过或失败。official CEM deployment 与 closed-loop 仍为 `NOT_RUN_BY_SCOPE`。
