# LeWM temporal-balanced training：Phase 5 结果

正式成功 job `25152151.pbs101`：`Exit_status=0`、walltime `00:02:37`、
`exec_host=x1000c1s3b0n0/0*16`。本轮在相同 512 episode IDs、相同 initialization/
schedule/optimizer/objective/candidate bank 与 3000 updates 下，比较原始 `anchor=0`
control 和按 ordinal `mod 3` 采用 early/middle/late anchors 的 balanced treatment
（`171/171/170`）。Fresh test 固定为全新 `valid[528:536]`，8 episodes、3 anchors、
2 action-prefix seeds（`20300927/20300928`），共 48 blocks × 300 candidates。

## Gate summary

| arm | Spearman median/min | top-30 median/min | relative MSE median | positive S/top-30 | absolute |
|---|---:|---:|---:|---:|---|
| `step0_train` | `0.720345 / -0.415650` | `0.433333 / 0` | `0.190753` | `39/48`, `44/48` | **NO-GO** |
| `balanced_temporal_train` | `0.759380 / 0.108881` | `0.483333 / 0.133333` | `0.085880` | `48/48`, `48/48` | **NO-GO** |

Treatment 的 early/middle/late Spearman medians 为 `0.677767/0.900639/0.554081`，
top-30 medians 为 `0.450000/0.600000/0.400000`，三个 stratum gate 均 **NO-GO**。
Secondary paired mechanism gate 为 **GO**：episode-median ΔSpearman `+0.108159`、
Δtop-30 `+0.233333`、joint improvement/non-worsening `6/8`。该 secondary 不能
替代 absolute gate，combined primary 仍 **NO-GO**。

Balanced/control predictor latency reduction 分别为 `91.0186%/91.0166%`；causality
`PASS`，convergence `PASS`（last10/first `0.008771/0.002646`），GPU telemetry
记录 32 个约 5-second samples（utilization `0–60%`、显存约 `1–681 MiB`）。

## 解释边界

结果支持“balanced anchor distribution 相对改善 paired ranking”的有限机制信号，
但不支持 predictor replacement；不得降低阈值或据此推进 planner。official CEM、
planner viability、closed-loop 均 **NOT_RUN_BY_SCOPE**。完整 evidence、snapshots、
paired episode table 与 scope 见
[完整中文结果](../lewm-transfer/temporal-balanced-train/RESULT_LEWM_TEMPORAL_BALANCED_TRAIN.zh.md)。

方法与 reporting provenance：Kassis et al. (2026), *Scientific Agent Skills: A
Library of Procedural Knowledge for Research Agents*, arXiv:2609.00065，
<https://arxiv.org/abs/2609.00065>。该 reference 不是实验结果证据。
