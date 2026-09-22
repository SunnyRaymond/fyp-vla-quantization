# CEM elite-boundary pairwise ranking：Stage 结果

## 结论

PBS job `25263388.pbs101` 已正常完成，PBS 与 runner 的 `Exit_status=0`，训练 1000 updates、fresh CEM 48/48 blocks 均完成。冻结的 primary gate 为 **FAIL**：boundary pairwise loss（固定 `lambda=0.05`、temperature=1）没有改善 shared fresh CEM 上的 standardized teacher-cost regret，且整体表现略差于 historical control。按冻结协议停止后续实验；本结果只支持 fixed-observation candidate-ranking 结论，不支持 official CEM deployment 或 closed-loop 结论。

## Provenance 与执行有效性

- Treatment 起点：`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/anchor-aligned-bank/artifacts/25223859.pbs101/anchor_aligned_bank_step3000.pt`，`anchor_aligned_bank_treatment`，3000 updates。
- Historical control：`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/cem-distribution-distill/artifacts/25239551.pbs101/treatment_step1000.pt` 与同目录 `cem_distribution_train_rows.pt`、`cem_distribution_distill_summary.json`；summary 标明 `non_concurrent=true`，因此 control 是历史复用的 paired fresh evaluation，不是同时重训 control。
- Fresh evaluation：`valid[568:576]`、selection seed `20300903`、8 episodes、3 CEM rounds（10/20/30）、每轮 300 candidates、共 144 blocks；两臂共享 frozen-start student-driven CEM banks 与 teacher costs。
- Pairing：teacher ranks `[20,30)` 对 `[30,40)`，每 context 100 cross-pairs，teacher detached；raw teacher population std floor `1e-6`，finite/pairing 检查通过。
- 远端只保留 checkpoint/rows/trajectory bank；本地仅取回小型 summary、log、status、GPU telemetry。

## Primary 结果

Primary metric 是每 block 的
`(teacher mean cost of student top30 − teacher mean cost of teacher top30) / max(population std, 1e-6)`，先在每个 episode 的 18 个 nested CEM blocks 取 median，再比较 8 个 episode。

| 指标 | Historical control | Pairwise treatment | treatment−control / 说明 |
|---|---:|---:|---:|
| overall standardized regret median（144 blocks） | 0.828342 | 0.981975 | +0.153633 |
| overall recall@120 median | 0.766667 | 0.666667 | 下降 |
| overall full elite containment@120 | 0.173611 | 0.138889 | 下降 |
| overall Spearman median | 0.504075 | 0.445519 | 下降 |
| boundary inversion rate median | 0.450000 | 0.480000 | 下降 |

episode-level primary deltas（treatment−control）为：

`2347:+0.133861, 4453:-0.335796, 6110:+0.082441, 11044:+0.114653, 13208:+0.095930, 13379:+0.509788, 13906:-0.017160, 14916:-0.214446`。

因此 episode median delta 为 **+0.089186**，严格改善 episode 仅 **3/8**。每轮 episode-median delta 依次为 round10 **+0.066989**、round20 **+0.065361**、round30 **+0.065455**，均未达到 `<=0`。

## Forgetting guard 与 gate

共享 fresh current-anchor 300 bank 的 forgetting guard 通过：episode-median delta **−0.013334**（阈值 `<=+0.05`），最差 episode delta **+0.125912**（阈值 `<=+0.20`）。但 primary median、严格改善 episode 数、各 round median 三项均失败；finite 通过，故总 gate 为 **FAIL/STOP**，不进入下一阶段。

## 证据文件

- [冻结设计](./FREEZE.json)
- [实验协议](./PROTOCOL.zh.md)
- [本地小型 summary](./artifacts/25263388.pbs101/cem_boundary_ranking_summary.json)
- [PBS/runner log](./artifacts/25263388.pbs101/job.log)
- [job status](./artifacts/25263388.pbs101/job_status)
- [final exit status](./artifacts/25263388.pbs101/final_exit_status.txt)
- [权威 PBS 终态 qstat -xf](./artifacts/25263388.pbs101/qstat_final.txt)
- [GPU telemetry](./artifacts/25263388.pbs101/gpu_info.csv)

`qstat_final.txt` 确认 `job_state=F`、`Exit_status=0`、实际 walltime `00:03:48`（申请上限 `00:30:00`）。`gpu_info.csv` 是作业启动时的静态 GPU 摘要；运行期间的 5 秒 GPU 利用率/显存 telemetry 以 `job.log` 为准，记录完整且不影响本次结果有效性。

关键证据路径：
`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/cem-boundary-ranking/artifacts/25263388.pbs101/cem_boundary_ranking_summary.json`。
