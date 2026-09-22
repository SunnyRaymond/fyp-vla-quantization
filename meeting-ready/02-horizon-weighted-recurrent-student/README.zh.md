# Horizon-weighted Recurrent Student：组会与后续实验包

这是一个可以独立拿去组会、并作为后续 predictor-level 实验起点的最小 bundle。它把 `shared recurrent latent-transition student` 和其唯一的 loss treatment——`horizon weighting`——放在同一个可运行闭包中；原始文件仍保留在 `experiment/idea-validation/jepa-action-prefix-compiler/`，本目录是复制出来的工作副本。

## 先看什么

1. [MEETING_CARD.zh.md](MEETING_CARD.zh.md)：组会前的 1 页版本。
2. [RESULT_HORIZON_WEIGHTED.zh.md](reports/RESULT_HORIZON_WEIGHTED.zh.md)：本 idea 的正式结果。
3. [RESULT_RECURRENT_STUDENT.zh.md](reports/RESULT_RECURRENT_STUDENT.zh.md)：为什么 recurrent student 是当前基座。
4. [RESULT_CLOSED_LOOP_PUSHT.zh.md](reports/RESULT_CLOSED_LOOP_PUSHT.zh.md)：已训练 student 对 official DINO-WM teacher 的 bounded PushT closed-loop comparison。
5. [RESULT_CEM_TRACE_DIAGNOSIS.zh.md](reports/RESULT_CEM_TRACE_DIAGNOSIS.zh.md)：6 个 teacher-only success cases 的 CEM elite-selection 与 feedback amplification diagnosis。
6. [CONTINUE_EXPERIMENTS.zh.md](CONTINUE_EXPERIMENTS.zh.md)：以后只从本目录继续实验的路径、外部资产和 PBS 用法。
7. [LeWM transfer result](lewm-transfer/RESULT_LEWM_RECURRENT_STUDENT.zh.md)：同一 best student 迁移到 official LeWM compact latent 的 predictor-level 结果。
8. [LeWM dense-rank result](reports/RESULT_LEWM_DENSE_RANK.zh.md)：64-candidate train coverage + elite-boundary rank distillation 的三臂结果。
9. [LeWM bounded score-distill result](reports/RESULT_LEWM_SCORE_DISTILL.zh.md)：用有限的 context-normalized teacher-score SmoothL1 检验能否保留 rank gain 并降低 latent drift。
10. [LeWM Phase 2 state/context coverage result](reports/RESULT_LEWM_STATE_COVERAGE.zh.md)：在固定 score-distill student 与 held-out protocol 下区分 train coverage 与 optimization steps。
11. [DINO-WM CEM-DAgger result](reports/RESULT_CEM_DAGGER.zh.md)：on-proposal teacher labels 对 matched replay 的 predictor/mechanism 对照结果。
12. [LeWM tail-robust EMA result](reports/RESULT_LEWM_TAIL_ROBUST.zh.md)：EMA 与 context-level top-2 tail emphasis 的 Phase 3 predictor gate。
13. [LeWM EMA temporal confirm result](reports/RESULT_LEWM_EMA_TEMPORAL_CONFIRM.zh.md)：fresh early/middle/late anchors 上的 EMA confirmatory predictor gate。
14. [DINO-WM CEM score-distill result](reports/RESULT_CEM_SCORE_DISTILL.zh.md)：proposal score distillation 的三臂 predictor-level 对照结果。
15. [DINO-WM CEM periodic teacher-anchor result](reports/RESULT_CEM_PERIODIC_ANCHOR.zh.md)：P=5 periodic full-teacher anchoring 的 fixed-observation mechanism diagnosis。
16. [Decision after periodic anchoring](reports/NEXT_AFTER_CEM_PERIODIC_ANCHOR.zh.md)：停止继续扩展 DINO special-purpose CEM recipe，并把后续主线收束回 LeWM + PushT。

## 一句话结论

在冻结的 DINO-WM PushT action-conditioned predictor 上，shared recurrent student 将 predictor-only latency 降到约 `10.7 ms`，相对 frozen teacher reduction 约 `99.69%`，参数量为 direct student control 的约 `30.35%`。在这个 recurrent 基座上给较远 horizon 更高 loss 权重，使 Spearman 在 `16/16` 个 paired blocks 上同向提高，但 effect size 很小，因此仍不能宣称可以 full replacement。

LeWM transfer 的后续 conditioner 增量结果见：[RESULT_LEWM_ACTION_HISTORY_CONDITIONER.zh.md](reports/RESULT_LEWM_ACTION_HISTORY_CONDITIONER.zh.md)。zero-init latest-3 action-history AdaLN-style affine conditioner 对 Spearman 有小幅改善，但 top-30 没有改善，仍为 predictor-level `NO-GO`。

第一优先的 rank-aware training 实验见：[RESULT_LEWM_DENSE_RANK.zh.md](reports/RESULT_LEWM_DENSE_RANK.zh.md)。在不改 student structure 的条件下，`dense_rank` 相对同预算 `dense_latent` 的 step-1500 top-30 median 提升 `+0.166667`（`14/16` blocks 改善），Spearman median 提升 `+0.122187`；shuffled-label control 未复现该收益，因此 screening gate 为 **GO**。但 absolute ranking 仍不足以替换 LeWM teacher，predictor feasibility 仍为 **NO-GO**。

后续 bounded closed-loop pilot 进一步验证了这个风险：在相同的 8 个 PushT cases 上，official teacher 为 `8/8` success，已训练的 horizon-weighted recurrent student 为 `2/8`。预先冻结的 progression gate 失败，因此停止扩到 50 cases。

针对 6 个 teacher-only success cases 的首轮 CEM trace 又把问题收窄到 elite selection：相同初始 candidate pool 上的 top-30 overlap median 只有 `0.766667`，到 iteration 30 的 student-pool teacher-shadow overlap 降至 `0.066667`，first-action RMS drift 从 `0.135645` 放大到 `0.731702`。这支持 local ranking mismatch + CEM feedback amplification，而不是单凭平均 latent fidelity 判断可替换性。

LeWM transfer（job `24564619.pbs101`）得到相同但更强的边界：student predictor
latency 为 `1.85293 ms`，teacher 为 `23.5295 ms`，reduction `92.1251%`；但 held-out
Spearman median/minimum 仅 `0.415374/-0.404356`，top-30 median/minimum
`0.266667/0`，因此 predictor gate 为 **NO-GO**。按用户要求未运行 official CEM。

Dense-rank transfer（job `24908446.pbs101`）是当前最明确的正向训练信号：
`dense_rank` 的 step-1500 Spearman median/minimum 为 `0.701261/-0.368217`，
top-30 median/minimum 为 `0.433333/0`，相对 `dense_latent` 的 paired top-30
median delta 为 `+0.166667`。它通过本轮 screening gate，但没有通过 absolute
predictor replacement gate；按 scope 仍未运行 official CEM、planner viability 或
closed-loop。

Bounded score-distill transfer（job `24916520.pbs101`）在完全相同的 student、candidate
bank、held-out protocol 和 training budget 下，把 step-1500 Spearman median 提高到
`0.974011`、top-30 median 提高到 `0.816667`，并把 relative latent MSE 降到
`0.013728`；相对 concurrent `pairwise_rank` 的 paired deltas 为
`+0.268554/+0.300000/-0.007516`。shuffled-score control 未复现 ranking 收益，说明
bounded score target 是更强的训练信号。但最后仍有 `1/16` block 的 top-30 overlap
为零，positive top-30 只有 `15/16`，因此冻结 screening gate 和 predictor feasibility
仍为 **NO-GO**；本轮没有运行 official CEM、planner viability 或 closed-loop。详见
[reports/RESULT_LEWM_SCORE_DISTILL.zh.md](reports/RESULT_LEWM_SCORE_DISTILL.zh.md)。

Phase 2 state/context coverage（job `24926383.pbs101`）复用上述 `256×1500` formal
reference，只新增 `256×3000`、`512×1500`、`512×3000`。512 manifest 的 held-out
rows、旧 256 train prefix、prepared-row prefix 和 held-out candidate/target bank 全部
通过 bitwise scope checks；但 primary `512×1500` 的 Spearman/top-30 median 为
`0.966364/0.800000`，低于冻结的 `0.974011/0.816667`，所以 coverage-at-fixed-updates
gate 为 **NO-GO**。`512×3000` 的 median 达到 `0.978317/0.866667`，但它混合了
coverage 与 extra exposure，且 worst-case absolute predictor gate 仍失败。没有运行
official CEM、planner viability 或 closed-loop。详见
[reports/RESULT_LEWM_STATE_COVERAGE.zh.md](reports/RESULT_LEWM_STATE_COVERAGE.zh.md)。

DINO-WM CEM-DAgger（job `24928207.pbs101`）中，treatment 的 held-out predictor
gate PASS，但 matched replay control 的 top-30 median 为 `0.783333 < 0.80`，所以
fixed-observation CEM 按冻结协议跳过。更关键的是，treatment 相对 control 的 paired
Spearman 为 `7/16` 改善、`9/16` 回退，top-30 为 `4/16` 改善、`6/16` 持平、
`6/16` 回退；该 recipe 因而记为机制层面 **NO-GO**，不通过放宽 gate 继续。
详见 [reports/RESULT_CEM_DAGGER.zh.md](reports/RESULT_CEM_DAGGER.zh.md)。

LeWM Phase 3 tail-robust EMA（job `24933682.pbs101`）中，primary
`ema_tail_score` 通过 inherited absolute predictor gate，但 Spearman/top-30 median
`0.973554/0.816667` 低于 historical `512×3000` 的 `0.978317/0.866667`，因此
预注册 no-regression gate 与 overall primary gate 为 **NO-GO**。普通 `ema_score`
在 median、minimum 和 relative latent MSE 上均略优于 tail-emphasis，说明 worst-block
改善来自 EMA，而不是 top-2 tail weighting。按 scope 未运行 official CEM、planner
viability 或 closed-loop。详见
[reports/RESULT_LEWM_TAIL_ROBUST.zh.md](reports/RESULT_LEWM_TAIL_ROBUST.zh.md)。

## 结果边界

- Recurrent architecture：Spearman median `+0.037226`、top-30 median `+0.066667`，两者方向正确但未达到冻结 effect gate。
- Horizon weighting：Spearman median `+0.004083`，top-30 median `+0.033333`；`latent non-inferiority`、`causality`、收敛和 latency 通过，但 effect gate 失败。
- Horizon-weighted student 的 absolute ranking 为 Spearman median/minimum `0.970596/0.894798`，top-30 median/minimum `0.833333/0.600000`，未达到 replacement gate。
- 计时边界是 cached native observation latent + action prefix → predictor rollout；不包含 `encode_obs`、CEM、environment 或 closed-loop success。
- 现已完成 LeWM predictor-level transfer；它是 `NO-GO`，仍不声称 Fast-LeWM comparison、official LeWM CEM viability 或 all-JEPA universality。
- 新增的 closed-loop 证据是 8-case、最多 12 MPC rounds 的 exploratory pilot，不是 official 50-case reproduction；其结果是 student 相对 teacher 的 bounded no-go signal。

后续 spatial/token mixer 等 NO-GO 分支不属于本包，也没有复制；它们只作为后续证据边界，不改变这里的两个 idea。

## 目录结构

```text
02-horizon-weighted-recurrent-student/
├─ README.zh.md
├─ MEETING_CARD.zh.md
├─ CONTINUE_EXPERIMENTS.zh.md
├─ src/                 # runner 及其最小本地 import 闭包（含 cache_core.py）
├─ config/              # freeze、protocol 和共享实验契约
├─ jobs/                # 可从 bundle 目录提交的 PBS wrapper
├─ artifacts/           # manifest、baseline、summary、log、status、telemetry
└─ reports/             # 中文结果解释
```

## 当前 bundle 自带的证据

- `24466744.pbs101`：256-context Dhigh manifest；
- `24373175.pbs101`：旧 128-context manifest，用于 manifest pairing 检查；
- `24477396.pbs101`：context-density summary，作为 Dhigh 支持证据；
- `24494756.pbs101`：query-slate direct student control；
- `24503839.pbs101`：uniform recurrent student；
- `24510395.pbs101`：horizon-weighted recurrent treatment。
- `24542653.pbs101`：2-case closed-loop engineering smoke；
- `24544733.pbs101`：8-case paired closed-loop pilot，teacher `8/8`、student `2/8`，progression gate `FAIL`。
- `24560503.pbs101`：6-case fixed-observation CEM trace diagnosis，支持 initial elite mismatch、iterative amplification 和 final shadow misranking。
- `24564619.pbs101`：LeWM predictor-level transfer，约 `12.70×` predictor speedup，但 frozen ranking gate `NO-GO`；Stage B 未运行。
- `24908446.pbs101`：LeWM dense query + elite-boundary rank distillation；screening gate `GO`，但 absolute predictor feasibility 仍 `NO-GO`；Stage B 未运行。
- `24916520.pbs101`：LeWM bounded context-normalized score distillation；整体 ranking 与 latent MSE 均优于 pairwise reference，但 screening/absolute predictor feasibility 仍 `NO-GO`；Stage B 未运行。
- `24926383.pbs101`：LeWM state/context coverage Phase 2；prefix/held-out scope checks `PASS`，`512×1500` treatment gate `NO-GO`；Stage B 未运行。
- `24933682.pbs101`：LeWM Phase 3 tail-robust EMA；两个 arm 的 absolute predictor gate `GO`，但 `ema_tail_score` 的 historical `512×3000` no-regression gate `NO-GO`，primary gate `NO-GO`；官方 CEM、planner viability 与 closed-loop 均 `NOT_RUN_BY_SCOPE`。详见 [tail-robust result](lewm-transfer/tail-robust/RESULT_LEWM_TAIL_ROBUST.zh.md)。
- `25145484.pbs101`：DINO-WM CEM proposal score distillation；`score_distill` absolute predictor gate `PASS`，但 paired gate `FAIL`（Spearman median delta `+0.001877`、top-30 median delta `0`），Stage 2 `SKIPPED`；不通过放宽 threshold 继续。详见 [score-distill result](reports/RESULT_CEM_SCORE_DISTILL.zh.md)。
- `25158912.pbs101`：DINO-WM CEM periodic teacher-anchor；relative RMS/AUC/regret gates 通过，但两个 absolute fidelity gates 失败，整体 **NO-GO**；六 case JSON 完整、summary 为 aggregation bug 后离线恢复；closed-loop `NOT RUN`。详见 [periodic teacher-anchor result](reports/RESULT_CEM_PERIODIC_ANCHOR.zh.md)。

GPU telemetry 和 job status 也保留在上述对应目录中；没有复制 checkpoint、`plan_targets.pkl`、数据集或完整 source repository。

## LeWM Phase 3 tail-robust 更新

正式 job `24933682.pbs101` 已正常结束（`Exit_status=0`，实际 walltime `00:03:39`，
`exec_host=x1000c0s1b0n0/1*16`）。`ema_score` terminal Spearman/top-30 median 为
`0.974618/0.816667`，`ema_tail_score` 为 `0.973554/0.816667`；两臂均通过
absolute predictor gate，但 tail arm 相对 historical `512×3000` 的 no-regression
Spearman/top-30 thresholds `0.978317/0.866667` 均失败，故 primary 为 **NO-GO**。
terminal paired `tail - mean` 的 Spearman median delta 为 `-0.001011`，top-30 为
`0`，relative MSE 为 `+0.000369`，未显示 tail emphasis 的稳定 ranking gain。

该分支仍严格停在 predictor-level；没有把 predictor latency 解读为 encoder/CEM/
environment speedup，也没有运行 official CEM、planner viability 或 closed-loop。
完整 snapshot、gate、paired delta、causality、convergence 与 GPU telemetry 见
[RESULT_LEWM_TAIL_ROBUST.zh.md](lewm-transfer/tail-robust/RESULT_LEWM_TAIL_ROBUST.zh.md)。

## LeWM EMA temporal confirm 更新

confirmatory job `25142797.pbs101` 正常完成（`Exit_status=0`，实际 walltime `00:03:22`，
`exec_host=x1000c1s3b0n0/0*16`）。本轮在同一条 `512`-row training trajectory 上比较
`online` 与 `EMA(decay=0.999)`，并把 evaluation 扩展到全新的 episode-disjoint
`early/middle/late` anchors（8 episodes、48 blocks、300 candidates/block）。EMA terminal
整体 Spearman/top-30 median 为 `0.815925/0.466667`，minimum 为 `-0.344404/0`，
relative latent MSE median 为 `0.147960`；因此 inherited absolute gate 与 temporal
stratum gate 均 **NO-GO**。EMA 相对 online 的 episode-median paired deltas 为
Spearman `-0.018833`、top-30 `-0.033333`，joint non-worsening improvement 仅 `2/8`，
secondary mechanism gate 也 **NO-GO**。

EMA predictor latency 为 `1.8412 ms`，teacher 为 `20.5635 ms`，reduction `91.0465%`；
该 boundary 只覆盖 cached latent + five-step predictor，不代表 encoder、CEM、planner 或
closed-loop speedup。convergence、causality、same-trajectory 与 EMA isolation 均通过。
本轮仍明确不运行 official CEM、planner viability、closed-loop；详见
[RESULT_LEWM_EMA_TEMPORAL_CONFIRM.zh.md](reports/RESULT_LEWM_EMA_TEMPORAL_CONFIRM.zh.md)。
报告组织参考已核实的 [Scientific Agent Skills, arXiv:2609.00065](https://arxiv.org/abs/2609.00065)；该引用不是实验结果证据。

## DINO-WM CEM score-distill 更新

正式 job `25145484.pbs101` 正常完成（`final_exit_code=0`，500 updates）。三臂的
Spearman median/mean/minimum 与 top-30 median/mean/minimum 分别为：
`latent_only` `0.967509/0.961855/0.906688`、`0.783333/0.779167/0.600000`；
`score_distill` `0.974295/0.964824/0.908604`、`0.816667/0.795833/0.600000`；
`shuffled_score` `0.964781/0.962079/0.896153`、`0.816667/0.785417/0.600000`。
其中 score arm 的 absolute gate **PASS**，但 paired gate **FAIL**：相对
`latent_only` 的 Spearman median delta 为 `+0.001877`（`13/16` 改善），top-30
median delta 为 `0`（`7/16` 改善、`7/16` 持平）；两项未达到冻结的相对阈值。
shuffled control 未复现该 paired gain，仍不足以把 recipe 判为机制 GO。故 Stage 2
fixed-observation CEM 为 `SKIPPED`，本轮只保留 fixed-observation predictor/mechanism
diagnosis，不声称 CEM、planner 或 closed-loop improvement。下一步优先转向
sequential/planner-aware correction；保持 no threshold loosening、no same-recipe rerun。
详见 [RESULT_CEM_SCORE_DISTILL.zh.md](reports/RESULT_CEM_SCORE_DISTILL.zh.md)。

DINO-WM CEM periodic teacher-anchor（job `25158912.pbs101`）在六个固定 observation
cases 上显示出明确但不足的相对 correction：`periodic_teacher_anchor_P5` 将
iteration-30 first-action RMS median 从 `0.7317022681` 降至 `0.6429772973`，
trajectory AUC median 从 `0.6152800977` 降至 `0.5493363014`，teacher-cost regret
median 从 `0.0960179530` 降至 `0.0786698610`，三项均为 `4/6` cases 不变差；但
absolute RMS `<=0.15` 与 coordinate abs max `<=0.25` 两个 gate 均失败，treatment
因此为 **NO-GO**。shuffled control 没有同时复现（RMS `3/6`、AUC `2/6` 不变差），
negative-control gate **PASS**。原 runner 只在最后 summary aggregation 因顶层
`trajectory_auc` `KeyError` 退出 `1`；六份 case JSON 已完整保留，summary 已用同一
冻结 gate 逻辑离线恢复。该轮严格停在 fixed-observation mechanism diagnosis，
closed-loop 为 `NOT RUN`；不放宽 gate、不重跑同一 recipe。详见
[RESULT_CEM_PERIODIC_ANCHOR.zh.md](reports/RESULT_CEM_PERIODIC_ANCHOR.zh.md)。

## LeWM state-action prefix GRU 更新

在 temporal-balanced training 之后，正式 job `25158955.pbs101` 完成了唯一一轮
两臂 structure experiment（`Exit_status=0`、walltime `00:04:58`、
`exec_host=x1000c1s3b0n1/0*16`）。`balanced_base` 保留原始 h256 recurrent student；
`state_action_prefix_gru` 只新增 shared 64-D state-action prefix GRU，base-owned
parameters bitwise equal，新增参数 `68,224`。Fresh test 是全新 `valid[536:544]`，
8 episodes、48 blocks、300 candidates/block。

| arm | terminal Spearman med/min | terminal top-30 med/min | relative MSE | absolute |
|---|---:|---:|---:|---|
| `balanced_base` | `0.824964/-0.624814` | `0.533333/0` | `0.100185` | **NO-GO** |
| `state_action_prefix_gru` | `0.841137/-0.639435` | `0.566667/0` | `0.094277` | **NO-GO** |

GRU−base 的 episode-paired mechanism gate 为 **GO**：median ΔSpearman `+0.023855`、
Δtop-30 `+0.025000`、joint `5/8`；但 treatment absolute、early/middle/late
stratum 和 combined primary gate 均 **NO-GO**。GRU predictor latency 相对 base
overhead 为 `41.56%`，超过 frozen `35%` guard；teacher-relative reduction 为
`88.42%`。本轮仍严格停在 predictor-level，official CEM、planner viability、
closed-loop 均 **NOT_RUN_BY_SCOPE**。完整结果见
[RESULT_LEWM_STATE_ACTION_PREFIX_GRU.zh.md](reports/RESULT_LEWM_STATE_ACTION_PREFIX_GRU.zh.md)。
