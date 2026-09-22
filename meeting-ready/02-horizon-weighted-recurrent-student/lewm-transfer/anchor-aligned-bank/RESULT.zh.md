# Anchor-aligned action-bank 配对实验结果

## 结论

本轮 predictor-level 实验正常完成，但未通过预先冻结的 mechanism gate，因此停止在本阶段，不进入 official CEM 或 closed-loop。Anchor-aligned bank 在 48 个 paired blocks 上提高了整体中位 recall@120，并减少了低 recall block；改善没有稳定覆盖 5/8 个 episode，且 episode 12015 明显退化，故不能把本轮结果写成稳健改进或模型 replacement 证据。

## PBS 与来源

- 重试作业：`25223859.pbs101`；`qstat -x`：`job_state=F`、`Exit_status=0`、walltime `00:06:09`、host `x1000c2s1b0n0/2*16`。
- 初次作业 `25222660.pbs101` 在启动 file guard 阶段因 dataset 默认路径错误于 1 秒退出，未进入 runner、模型加载或训练；修正后仅进行这一次 bounded retry。
- Control 使用 job `25213164.pbs101` 的 reconstructed balanced_base terminal checkpoint 与 rows，provenance 保留为 `reconstructed`，不声称原 Phase 6 checkpoint。
- Treatment 从 seed `20300901` 的原始初始化重新训练 3000 updates；h256、AdamW、batch 8、64 candidates、loss 和 schedule 保持冻结。
- 512 个 training contexts 的 anchor counts 为 early/middle/late=`171/171/170`。CPU generator 按 `count1,count1,count64,randn(HORIZON,ACTION_DIM)` 重放，缓存 control bank 精确匹配，`paired_noise_exact=true`。
- Fresh evaluation 是固定 shuffle 的 `valid[552:560]`，episode IDs=`18307,15094,5479,12015,15229,9523,2819,17934`；48 blocks 使用 seeds `20300967/20300968`，两臂共享 candidate banks 和 full-teacher scores。

## 主要指标

| 指标（48 blocks median） | Control | Treatment |
|---|---:|---:|
| Spearman | 0.7857 | 0.8904 |
| top30 overlap | 0.5000 | 0.7167 |
| relative latent MSE | 0.1160 | 0.0703 |
| recall@60 | 0.7333 | 0.9167 |
| recall@120 | 0.9667 | 1.0000 |
| full elite containment@120 | 41.67% | 58.33% |
| standardized elite regret@120 | 0.01624 | 0 |

Episode-level recall@120 median（Treatment−Control）为：2819 `0.3167→0.4667`（`+0.1500`）、5479 `0.9500→1.0000`（`+0.0500`）、9523 `0.9667→0.9833`（`+0.0167`）、12015 `0.8667→0.5167`（`−0.3500`）、15094 `1.0000→1.0000`（`0`）、15229 `0.2500→0.4833`（`+0.2333`）、17934 `1.0000→1.0000`（`0`）、18307 `1.0000→1.0000`（`0`）。严格改善为 `4/8`，冻结要求为至少 `5/8`。

按 stratum 的 recall@120 median 为：early `1.0000→1.0000`、middle `0.9333→0.8667`、late `0.8500→1.0000`。Treatment 的 block minimum 为 `0.3000`；Control 为 `0`。低于 `0.8` 的 blocks 为 Control `17`、Treatment `16`，worst block 为 `0→0.3`。Episode-median standardized regret delta 为 `−0.000689`。因此 paired mechanism gate 为 **FAIL**：median delta、bad-block count、worst-block、regret、finite/interface/paired-noise 条件通过，但 strict improvement 条件失败。独立 absolute gate 也为 **FAIL**：Treatment median 通过，minimum 和 middle stratum 未通过。

最差 Treatment blocks 为：episode 2819 middle anchor 36 seed 20300967，recall@120=`0.30`、Spearman=`−0.148`；episode 2819 late anchor 72 seed 20300967，`0.30`、`0.149`；episode 12015 early anchor 0 seed 20300967，`0.30`、`−0.010`；episode 15229 early anchor 0 seed 20300967，`0.3333`、`−0.074`；episode 12015 middle anchor 52 seed 20300967，`0.40`、`0.164`。这些 block 保留在 summary 的 `worst_blocks` 和 `per_block` 中供复核。

## Timing

所有 48 blocks 使用 3 warmups、10 repeats、order seed `20300969`，记录 CUDA-synchronized wall-clock median（ms）：teacher300=`19.246`、teacher120=`17.622`、teacher60=`17.707`、student300 control=`1.767`、student300 treatment=`1.759`、hybrid120 control=`19.414`、hybrid120 treatment=`19.498`。本实现中 hybrid 没有转化为相对 full teacher 的加速；该 timing 仅作诊断，不改变 training mechanism gate，也不推断具体瓶颈原因。

## 证据文件

- 本地小 summary：[anchor_aligned_bank_summary.json](artifacts/25223859.pbs101/anchor_aligned_bank_summary.json)
- 本地 PBS 状态：[job_status](artifacts/25223859.pbs101/job_status)、[final_exit_status.txt](artifacts/25223859.pbs101/final_exit_status.txt)、[job.log](artifacts/25223859.pbs101/job.log)、[gpu_info.csv](artifacts/25223859.pbs101/gpu_info.csv)
- 集群 summary：`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/anchor-aligned-bank/artifacts/25223859.pbs101/anchor_aligned_bank_summary.json`
- 集群 Treatment checkpoint 与 rows 留在同一 artifacts 目录；未回传大文件。
- `official_cem=NOT_RUN_BY_SCOPE`，`closed_loop=NOT_RUN_BY_SCOPE`。
