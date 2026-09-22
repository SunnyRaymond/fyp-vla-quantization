# LeWM temporal-balanced training confirmatory predictor result

## 结论先行

本轮检验的是 Phase 3 fresh multi-anchor observation 导出的单一新 hypothesis：
原来的 `anchor=0` training distribution 可能不足以覆盖 middle/late temporal
states；在不改变 h256 architecture、objective、candidate bank、optimizer、batch
或 3000 updates 的条件下，按 early/middle/late 平衡 training anchors 是否能改善
fresh multi-anchor predictor fidelity。

结论是：`balanced_temporal_train` 相对 `step0_train` 的 paired mechanism
comparison **GO**，但预注册的 treatment absolute predictor gate、三个 temporal
stratum gate 和 combined primary gate 均 **NO-GO**。因此本轮保留为有信息量的
predictor-level negative result：balanced anchor distribution 带来相对改善，仍不
足以支持 predictor replacement claim。

## 作业与证据

成功正式 job 为 `25152151.pbs101`：

| 项目 | 值 |
|---|---|
| `qstat -xf` Exit_status | `0` |
| walltime | `00:02:37` |
| exec_host | `x1000c1s3b0n0/0*16` |
| GPU | NVIDIA A100-SXM4-40GB |
| training | 两臂各 512 contexts、3000 updates |
| evaluation | 8 episodes × 3 anchors × 2 seeds = 48 blocks；300 candidates/block |
| terminal snapshot | `step_3000` |

首次提交 `25150832.pbs101` 已完成计算，但在写 JSON 时误把内部 snapshot tensors
带入序列化，故 `Exit_status=1` 且没有正式 summary；修正为只写可序列化 training
trace 后，以同一冻结 protocol 重跑成功。该修正不改变数据、模型、loss、seed、
training 或 evaluation。

成功证据只保留小型 summary、job log/status：

- [freeze](LEWM_TEMPORAL_BALANCED_TRAIN_FREEZE.json)
- [protocol](PROTOCOL_LEWM_TEMPORAL_BALANCED_TRAIN.zh.md)
- [summary JSON](artifacts/25152151.pbs101/lewm_temporal_balanced_train_summary.json)
- [job.log 与 5-second GPU telemetry](artifacts/25152151.pbs101/job.log)
- [job_status](artifacts/25152151.pbs101/job_status)

未回传 prepared rows、checkpoint、HDF5 或 manifests。

## Frozen design

`step0_train` 复用 Phase 2 formal 的原 512 episode IDs，每 episode 固定
`anchor=0`；`balanced_temporal_train` 复用完全相同的 episode IDs 与 context
ordinals，仅按 `ordinal mod 3` 固定映射：`early=0`、`late=episode_length-26`、
`middle=floor(late/2)`，counts 为 `early/middle/late=171/171/170`。没有按
数据或结果选择 anchor。

两臂共享 initialization、training seed、context schedule、h256 shared recurrent
no attention/no conditioner architecture、AdamW、batch 8、3000 updates、原
horizon-weighted latent MSE + `0.1` score-distill objective。为保持 paired
comparison，control 的 64-candidate action bank 按 context ordinal 原样复用到
treatment；treatment 只在新的 balanced state 上重算 teacher targets/objectives。

Fresh confirmation 固定使用 selection shuffle 后全新的 `valid[528:536]`，排除旧
heldout `valid[:8]`、原 512 train `valid[8:520]` 和上一轮 fresh `valid[520:528]`。
episode IDs 为 `6767, 3851, 2615, 15265, 14960, 10302, 18347, 10045`；新
action-prefix seeds 为 `20300927/20300928`。本轮 test 未用于设计。

## Terminal metrics

| arm | Spearman median/min | top-30 median/min | relative latent MSE median | positive S/top-30 | absolute gate |
|---|---:|---:|---:|---:|---|
| `step0_train` | `0.720345 / -0.415650` | `0.433333 / 0.000000` | `0.190753` | `39/48`, `44/48` | **NO-GO** |
| `balanced_temporal_train` | `0.759380 / 0.108881` | `0.483333 / 0.133333` | `0.085880` | `48/48`, `48/48` | **NO-GO** |

Balanced treatment 的 relative latent MSE 与 positive-block counts 通过 inherited
conditions，但 Spearman median/minimum、top-30 median/minimum 均未通过
`0.95/0.80` 与 `0.75/0.50` thresholds。Control 也只是 concurrent descriptive
control，不能替 treatment gate。

### Descriptive snapshots

| arm / snapshot | Spearman median/min | top-30 median/min | relative MSE median | positive S/top-30 |
|---|---:|---:|---:|---:|
| control `500` | `0.303158 / -0.917199` | `0.150000 / 0` | `0.280543` | `31/48`, `37/48` |
| control `1000` | `0.555807 / -0.626769` | `0.283333 / 0` | `0.211900` | `34/48`, `39/48` |
| control `1500` | `0.556630 / -0.432235` | `0.383333 / 0` | `0.198345` | `37/48`, `41/48` |
| control `3000` | `0.720345 / -0.415650` | `0.433333 / 0` | `0.190753` | `39/48`, `44/48` |
| balanced `500` | `0.419588 / -0.663934` | `0.250000 / 0` | `0.099410` | `37/48`, `43/48` |
| balanced `1000` | `0.539763 / -0.464642` | `0.283333 / 0` | `0.089784` | `43/48`, `46/48` |
| balanced `1500` | `0.626132 / -0.171008` | `0.416667 / 0.033333` | `0.081405` | `45/48`, `48/48` |
| balanced `3000` | `0.759380 / 0.108881` | `0.483333 / 0.133333` | `0.085880` | `48/48`, `48/48` |

## Temporal stratum gate

Treatment stratum gate 要求每个 stratum 的 median Spearman/top-30 至少
`0.95/0.75`、positive blocks 各至少 `12/16`。三个 stratum 的 positive counts
均为 `16/16`，但 fidelity medians 均失败：

| stratum | Spearman median/min | top-30 median/min | relative MSE median | positive S/top-30 | gate |
|---|---:|---:|---:|---:|---|
| early | `0.677767 / 0.382707` | `0.450000 / 0.200000` | `0.032546` | `16/16`, `16/16` | **NO-GO** |
| middle | `0.900639 / 0.142431` | `0.600000 / 0.166667` | `0.145356` | `16/16`, `16/16` | **NO-GO** |
| late | `0.554081 / 0.108881` | `0.400000 / 0.133333` | `0.125017` | `16/16`, `16/16` | **NO-GO** |

Balanced training reduced the late/middle gap relative to its control, but it did not
raise any stratum to the absolute replacement threshold. In particular, this is not
evidence that balancing alone solves the temporal-transfer problem.

## Secondary paired mechanism comparison

Paired unit 是 episode；每个 episode 先对 6 个 nested blocks（3 anchors × 2 seeds）
取 median，再计算 balanced minus control：

| metric | median delta | gate |
|---|---:|---|
| Spearman | `+0.108159` | pass (`>=0`) |
| top-30 overlap | `+0.233333` | pass (`>=0`) |
| joint improvement/non-worsening | `6/8` episodes | pass (`>=5/8`) |

该 secondary gate 为 **GO**，但不能替代 absolute/stratum gate。每 episode 的
Spearman/top-30 deltas 为：

| episode | ΔSpearman | Δtop-30 | joint |
|---:|---:|---:|---|
| 2615 | `-0.498470` | `-0.350000` | no |
| 3851 | `+0.137315` | `+0.233333` | yes |
| 6767 | `+0.516314` | `+0.483333` | yes |
| 10045 | `-0.004349` | `-0.100000` | no |
| 10302 | `+0.079003` | `+0.100000` | yes |
| 14960 | `+0.049220` | `+0.233333` | yes |
| 15265 | `+0.608419` | `+0.300000` | yes |
| 18347 | `+0.324246` | `+0.400000` | yes |

## Latency, causality and convergence

| arm | student median | teacher median | reduction | causality | last10/first |
|---|---:|---:|---:|---|---:|
| control | `1.842688 ms` | `20.512256 ms` | `91.0166%` | PASS | `0.002646` |
| balanced | `1.848320 ms` | `20.579329 ms` | `91.0186%` | PASS | `0.008771` |

Causality 对 unchanged prefix lengths `1/2/3/4` 的 `max_abs` 均为 `0`，threshold
为 `1e-6`。两臂 convergence 均通过 `last10/first <= 0.8`，所有 terminal metrics
finite。Latency boundary 只包括 cached H=1 latent + five-step predictor rollout，
不包括 encoder、CEM、environment 或 closed-loop。

## GPU telemetry 与 scope

作业内以约 5 秒间隔记录了 32 个 `nvidia-smi` samples；GPU 为 A100-SXM4-40GB，
observed utilization 为 `0–60%`，显存为约 `1–681 MiB`。telemetry 仅用于确认
compute-node allocation 内的运行与资源记录，不改变 predictor metrics。

本轮明确：

- official CEM：**NOT_RUN_BY_SCOPE**；
- planner viability：**NOT_RUN_BY_SCOPE**；
- closed-loop：**NOT_RUN_BY_SCOPE**。

因此不能从本轮的 predictor latency 或 ranking delta 声称 CEM、planner、closed-loop
success、encoder speedup 或 native deployment benefit。

## 结果解释与下一步边界

本轮支持一个有限结论：在同一 candidate bank、same init/schedule 与相同 compute
budget 下，balanced temporal training 相对 anchor=0 control 的 episode-paired
ranking 有稳定方向的改善，并把所有 terminal blocks 的 positive Spearman/top-30
提升到 `48/48`。但是 absolute ranking 仍明显低于预注册 gate，且三个 stratum
均未通过；因此不得把该 recipe 升格为 replacement，也不应通过降低 thresholds、
选择 snapshot 或运行 planner 来补救该 NO-GO。

方法与 reporting provenance 参考 Scientific Agent Skills：Kassis, T., Agarwal,
V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A
Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065.
https://arxiv.org/abs/2609.00065 （当前 arXiv record 为 v2；引用按无版本后缀的
记录）。该方法 reference 不是本实验结果证据。
