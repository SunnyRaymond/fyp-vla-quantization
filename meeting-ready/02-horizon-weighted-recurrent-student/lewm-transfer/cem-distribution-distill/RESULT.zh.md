# CEM candidate distribution distillation：结果

## 终态

- PBS：`25239551.pbs101`，`gdev`，`F`，`Exit_status=0`；`resources_used.walltime=00:05:14`，运行节点 `x1000c2s1b0n0`。
- 作业从 `2026-09-22 17:34:11` 开始，`17:39:34` 更新为终态。5 秒 GPU telemetry 已写入远端 `job.log`；作业在 compute allocation 内完成。
- 远端 summary：`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/cem-distribution-distill/artifacts/25239551.pbs101/cem_distribution_distill_summary.json`。
- 本地小证据：`local-status/25239551.pbs101/` 下的 `cem_distribution_distill_summary.json`、`job.log`、`job_status`、`final_exit_status.txt`、`gpu_info.csv`、`qstat_final.txt`。

## 实验契约与 provenance

- 起点是 `anchor_aligned_bank_step3000.pt`，来源 `25223859.pbs101` 的 anchor-aligned treatment checkpoint；summary 标记 `source=anchor_aligned_bank_treatment`、`reconstructed=false`、`updates=3000`，没有把它声称为原 Phase 6 checkpoint。
- control/treatment 都从同一 state dict 各自重置 AdamW 并训练 1000 updates。control 使用原 64-candidate bank；treatment 使用原前 32 条加固定 CEM round 10/20/30 的 11/11/10 条 tail。
- CEM 为 30 rounds、每轮 300 candidates、top30；candidate-zero、unbiased `std`、无 clipping。512 个训练 context 共保存 16,384 个 teacher-labelled tail candidates；teacher 没有参与 CEM proposal 更新或训练样本选择。
- fresh evaluation 是 `valid[560:568]`，selection seed `20300903`，8 episodes×3 anchors×2 action-prefix seeds，共 48 standard blocks；primary 将 round 10/20/30 合并后按 episode 聚合 18 个 nested CEM blocks。所有结果均为 fixed-observation candidate-ranking evidence。

## 主要数字

144 个 primary CEM blocks 的描述性整体统计如下；正式 gate 使用 episode median，因此不能用下表的全 block median 替代 gate。

| 指标 | control | treatment |
|---|---:|---:|
| Spearman median | 0.262932 | 0.415128 |
| top30 overlap median | 0.116667 | 0.233333 |
| relative latent MSE median | 0.411915 | 0.313603 |
| recall@60 median | 0.233333 | 0.400000 |
| recall@120 median | 0.466667 | 0.683333 |
| raw elite mean-cost regret median | 13.820964 | 11.615292 |
| standardized elite mean-cost regret median | 1.237150 | 1.043333 |
| full elite containment@120 rate | 0.055556 | 0.083333 |

两臂的 1000-update terminal trace 也不同：control weighted latent MSE=`0.0233925`、score loss=`0.0467153`；treatment 分别为 `0.1027424`、`0.2110681`。两组训练目标分布不同，这些 terminal loss 不能单独证明任务难度差异，也不能替代 held-out ranking 结果。

## Gate

最终 gate=`FAIL`，finite 条件通过，但 primary 与 forgetting 的全部组合条件没有通过：

- primary episode standardized-regret delta（treatment−control）为：episode `835=+0.516450`、`1058=+0.421059`、`2682=-0.206108`、`2773=-0.722510`、`10658=-0.131125`、`11169=+0.068502`、`12612=-0.323892`、`17214=+0.321083`。
- primary 8-episode median delta=`-0.031312`，未达到预设 `<=-0.05`；严格改善 episode 数=`4/8`，未达到 `5/8`。
- 按 CEM round 的 episode median delta：round10=`-0.332304`（通过），round20=`+0.212484`、round30=`+0.204261`（均未通过逐 round `<=0`）。
- standard current-anchor 300-bank forgetting median delta=`-0.068779`（通过），但最差 episode delta=`+0.351707`，超过 `+0.20` 上限。

代表性 primary worst blocks：episode `10658` middle、round30、seed `20300987` 的 Spearman=`-0.737944`、recall@120=`0`、standardized regret=`2.810153`；episode `1058` late、seed `20300987` 的 round20/30 recall@120 都为 `0`。forgetting guard 的代表性 treatment standard blocks 包括 episode `11169` middle、seed `20300988`，recall@120=`0.2`、standardized regret=`2.804363`，以及 episode `12612` middle、seed `20300988`，recall@120=`0.133333`、standardized regret=`2.266151`。

## 结论边界

本次固定的 CEM candidate-distribution distillation 没有通过预注册 primary gate，也没有通过 worst-episode forgetting guard；因此当前 evidence 不支持继续把该 recipe 推进到下一轮、planner deployment、adaptive CEM 或 closed-loop。summary 明确记录 official CEM 与 closed-loop 均为 `NOT_RUN_BY_SCOPE`；CEM trajectory 是 frozen-start student-driven action bank with teacher shadow labels，不应称为 student on-policy environment data。
