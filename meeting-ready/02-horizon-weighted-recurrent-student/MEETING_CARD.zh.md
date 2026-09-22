# 组会卡片：Horizon-weighted Recurrent Student

## 30 秒版本

我们的目标是把 DINO-WM 的 action-conditioned predictor 从昂贵的 teacher rollout 编译成轻量 student。当前最有效的结构不是继续增大 MLP，而是让 student recurrently consume 自己预测的 latent，并把 output feedback 到下一个 action step。

LeWM 上的 action-history conditioner 增量：[conditioner result](lewm-transfer/conditioner/RESULT_LEWM_ACTION_HISTORY_CONDITIONER.zh.md)。

第一优先的 rank-aware training 增量：[dense-rank result](reports/RESULT_LEWM_DENSE_RANK.zh.md)。

在这个 recurrent student 上，我们只改变训练 loss：五个预测 horizon 的误差使用

```text
[1/3, 2/3, 1, 4/3, 5/3]
```

让更远的预测受到更大梯度。结果是 ranking 方向稳定改善，但幅度没有达到预先冻结的 gate。

## 实验和结果

| 项目 | 结果 |
|---|---:|
| held-out paired blocks | 16 |
| recurrent architecture：median ΔSpearman | `+0.037226` |
| recurrent architecture：median Δtop-30 | `+0.066667` |
| horizon weighting：median ΔSpearman | `+0.004083` |
| horizon weighting：median Δtop-30 | `+0.033333` |
| horizon-weighted absolute Spearman median/min | `0.970596 / 0.894798` |
| horizon-weighted absolute top-30 median/min | `0.833333 / 0.600000` |
| horizon-weighted logged MSE ratio | `0.927539` |
| student predictor latency | `10.719 ms` |
| teacher → student reduction | `99.6936%` |
| recurrent student parameters | `732,554` |

## 应该怎样解释

1. 这是 predictor-level result，不是 end-to-end control result。
2. Student 已经足够快，且没有明显 latent fidelity、causality 或训练稳定性问题。
3. 主要瓶颈不是简单的容量或训练时长，而是 student 是否保留了 planner candidate 之间的局部 action-to-cost sensitivity。
4. Horizon weighting 是稳定但很小的正向信号，不能单独作为 ranking solution。

## 结论措辞

推荐说：

> A shared recurrent latent-transition student gives a large predictor-only speedup and a consistent but sub-threshold ranking improvement. Mean-normalized horizon weighting preserves this behavior and slightly improves the paired ranking trend, but does not pass the frozen replacement gate.

不要说：

- “已经超过 teacher”；
- “可以安全替换 teacher”；
- “CEM 或 closed-loop success 已改善”；
- “结果已经证明适用于 LeWM/Fast-LeWM 或所有 JEPA-style WM”。

## LeWM transfer 更新

同一个 horizon-weighted recurrent mechanism 已在 official LeWM compact latent 上完成
predictor-level 测试（job `24564619.pbs101`）：

| 项目 | 结果 |
|---|---:|
| student / teacher latency | `1.85293 / 23.5295 ms` |
| predictor latency reduction | `92.1251%`（约 `12.70×`） |
| Spearman median / minimum | `0.415374 / -0.404356` |
| top-30 median / minimum | `0.266667 / 0` |
| relative latent MSE median | `0.012116` |
| predictor feasibility | **NO-GO** |

这说明机制的速度优势能迁移到 LeWM，但 ranking fidelity 没有迁移成功；高 latent cosine
仍不足以支持 replacement。按实验边界未运行 official LeWM CEM。详见
[lewm-transfer/RESULT_LEWM_RECURRENT_STUDENT.zh.md](lewm-transfer/RESULT_LEWM_RECURRENT_STUDENT.zh.md)。

## Dense-rank 更新

在不改变 h256 shared recurrent student structure 的前提下，我们把 train query
coverage 提高到每个 context `64` 个 candidates，并加入 frozen teacher top-6 对
ranks 7–12 的 pairwise rank loss；另设 context-internal shuffled-label control。三臂
共享 initial state、context schedule、candidate bank 和 1500-step budget。

正式作业 `24908446.pbs101` 的 step-1500 结果为：

| arm | Spearman median/min | top-30 median/min | predictor feasibility |
|---|---:|---:|---|
| dense_latent | `0.514727 / -0.301539` | `0.266667 / 0` | **NO-GO** |
| dense_rank | `0.701261 / -0.368217` | `0.433333 / 0` | **NO-GO** |
| shuffled_rank | `0.086698 / -0.335159` | `0.100000 / 0.033333` | **NO-GO** |

`dense_rank - dense_latent` 的 paired top-30 delta median 为 `+0.166667`，`14/16`
blocks 改善；paired Spearman delta median 为 `+0.122187`，`14/16` blocks 改善。
Shuffled control 的 top-30 delta median 为 `-0.150000`（`2/12/2` improve/worse/tie），
因此本轮 screening gate 为 **GO**，说明正确的 teacher ordering 确实提供了有效训练
信号。但 absolute ranking 仍低于 replacement threshold，故仍不能安全替换 LeWM
teacher。完整结果见
[reports/RESULT_LEWM_DENSE_RANK.zh.md](reports/RESULT_LEWM_DENSE_RANK.zh.md)。

## Bounded score-distill 更新

为处理 pairwise margin 可能造成的 latent drift，我们在同一 frozen candidate bank 上
测试了全 64 candidates 的 context-normalized teacher-score SmoothL1，并保留
`pairwise_rank` 和 shuffled-score 两个 concurrent arms。正式作业
`24916520.pbs101` 的 step-1500 结果为：

| arm | Spearman median/min | top-30 median/min | relative latent MSE | feasibility |
|---|---:|---:|---:|---|
| pairwise_rank | `0.701261 / -0.368217` | `0.433333 / 0` | `0.025179` | **NO-GO** |
| score_distill | `0.974011 / -0.010504` | `0.816667 / 0` | `0.013728` | **NO-GO** |
| shuffled_score | `0.233274 / -0.168441` | `0.183333 / 0` | `0.011658` | **NO-GO** |

相对 `pairwise_rank`，`score_distill` 的 paired median deltas 为
`ΔSpearman=+0.268554`、`Δtop-30=+0.300000`、`Δrelative-MSE=-0.007516`；16 个
blocks 的 Spearman 和 relative MSE 全部改善，top-30 为 `15/16` 改善。这个结果支持
bounded score target 同时改善 ranking median 和 latent fidelity，且 shuffled labels
不能复现收益。

但 screening 仍为 **NO-GO**：positive top-30 只有 `15/16`（冻结要求 `16/16`），
minimum top-30 仍为 `0`。所以它是比 pairwise rank 更有希望的 predictor-level 训练
信号，不是已经通过的 replacement。Latency 为 `1.852416 ms`，相对 teacher reduction
`91.6772%`；本轮仍未运行 official CEM、planner viability 或 closed-loop。详见
[reports/RESULT_LEWM_SCORE_DISTILL.zh.md](reports/RESULT_LEWM_SCORE_DISTILL.zh.md)。

## State/context coverage Phase 2

在不改变 score-distill student、loss、64-candidate train bank 和 held-out 16 blocks
的条件下，复用 `256×1500` reference，只新增 `256×3000`、`512×1500`、`512×3000`
（job `24926383.pbs101`）。512 manifest 的 held-out rows、旧 train prefix、prepared
row prefix 和 held-out bank 均 bitwise **PASS**，所以没有 scope leakage。

| arm | Spearman median/min | top-30 median/min | relative MSE | gate |
|---|---:|---:|---:|---|
| `256×3000` | `0.977582 / -0.351171` | `0.850000 / 0` | `0.008170` | **NO-GO** |
| `512×1500` | `0.966364 / 0.790585` | `0.800000 / 0.433333` | `0.011604` | **NO-GO** |
| `512×3000` | `0.978317 / 0.656421` | `0.866667 / 0.366667` | `0.007499` | **NO-GO** |

Primary `512×1500` 没有达到冻结 median gate `0.974011/0.816667`，但把 positive
top-30 提到 `16/16`、minimum top-30 提到 `0.433333`；因此 coverage 在 fixed
updates 下改善 worst-case，却不足以宣称 GO。`512×3000` 同时改变 coverage 与
exposure，不能单独归因于 coverage。结论停在 predictor-level；没有运行 official
CEM、planner viability 或 closed-loop。完整结果见
[reports/RESULT_LEWM_STATE_COVERAGE.zh.md](reports/RESULT_LEWM_STATE_COVERAGE.zh.md)。

## Closed-loop 更新

已训练的 horizon-weighted `step1500` student 已接入 official DINO-WM PushT CEM + MPC + environment evaluator。冻结的 8-case、最多 12 MPC rounds paired pilot（job `24544733.pbs101`）得到：

| arm | success |
|---|---:|
| official DINO-WM teacher | `8/8` |
| horizon-weighted recurrent student | `2/8` |

paired table 为 `both success=2`、`teacher only=6`、`student only=0`。Progression gate 要求 student 至少 `7/8` 且最多落后 teacher 1 例，结果为 **FAIL**，因此停止扩到 50 cases。

这不是 official 50-case reproduction；它是 bounded exploratory no-go signal。Student 总 planner walltime 更长是因为 6 个失败例跑满 12 rounds，不能解释为 predictor forward 变慢。

## CEM trace diagnosis

对上述 6 个 teacher-only success cases，在相同 initial observation 上用 paired innovations 重跑 30-step CEM trace：

| metric（case-level median） | iter 1 | iter 30 |
|---|---:|---:|
| teacher-pool Spearman | `0.975736` | `0.421793` |
| teacher-pool top-30 overlap | `0.766667` | `0.083333` |
| student-pool teacher-shadow Spearman | `0.975736` | `0.238582` |
| student-pool shadow top-30 overlap | `0.766667` | `0.066667` |
| first-action RMS drift | `0.135645` | `0.731702` |

最可能的问题是 student 没有稳定保留 CEM 所需的 local elite ordering；第一次 top-30 selection 已分叉，之后 proposal feedback 将误差放大。这个结果仍是 fixed-observation mechanism diagnosis，不是新的 success-rate evidence，也没有定位到某个 architecture component。

## 关键材料

- 正式 horizon 结果：[reports/RESULT_HORIZON_WEIGHTED.zh.md](reports/RESULT_HORIZON_WEIGHTED.zh.md)
- recurrent 基座结果：[reports/RESULT_RECURRENT_STUDENT.zh.md](reports/RESULT_RECURRENT_STUDENT.zh.md)
- PushT closed-loop 结果：[reports/RESULT_CLOSED_LOOP_PUSHT.zh.md](reports/RESULT_CLOSED_LOOP_PUSHT.zh.md)
- CEM trace diagnosis：[reports/RESULT_CEM_TRACE_DIAGNOSIS.zh.md](reports/RESULT_CEM_TRACE_DIAGNOSIS.zh.md)
- 后续 continuation：[CONTINUE_EXPERIMENTS.zh.md](CONTINUE_EXPERIMENTS.zh.md)
- Dense-rank 结果：[reports/RESULT_LEWM_DENSE_RANK.zh.md](reports/RESULT_LEWM_DENSE_RANK.zh.md)
- Bounded score-distill 结果：[reports/RESULT_LEWM_SCORE_DISTILL.zh.md](reports/RESULT_LEWM_SCORE_DISTILL.zh.md)

## LeWM Phase 3 tail-robust EMA

正式 job `24933682.pbs101` 正常完成（`Exit_status=0`，实际 walltime `00:03:39`）。
两个 EMA arm 的 absolute predictor gate 均为 `GO`，但预注册 primary gate 为
**NO-GO**：`ema_tail_score` 没有达到 historical `512×3000` 的 no-regression
ranking thresholds。

| arm | terminal Spearman med/min | terminal top-30 med/min | relative MSE | absolute | primary |
|---|---:|---:|---:|---|---|
| `ema_score` | `0.974618 / 0.854591` | `0.816667 / 0.566667` | `0.008968` | GO | diagnostic |
| `ema_tail_score` | `0.973554 / 0.851060` | `0.816667 / 0.533333` | `0.009391` | GO | **NO-GO** |

terminal paired `ema_tail_score - ema_score` 为 Spearman median `-0.001011`、
top-30 median `0`、relative MSE median `+0.000369`。因此在当前冻结预算下，tail
emphasis 没有带来可确认的 ranking improvement。Predictor latency reduction 为
`90.9975%`（mean）和 `91.7772%`（tail），但计时只覆盖 cached latent + predictor
rollout，不包括 encoder、CEM 或 environment。

本轮结论停在 predictor-level：official CEM、planner viability、closed-loop 均
**NOT_RUN_BY_SCOPE**。详见 [tail-robust full result](lewm-transfer/tail-robust/RESULT_LEWM_TAIL_ROBUST.zh.md)。

## LeWM EMA temporal confirm 更新

在 tail-robust Phase 3 之后，正式 job `25142797.pbs101` 把 `EMA(decay=0.999)` 放到同一条
`512`-row training trajectory 上，并在全新、episode-disjoint 的 early/middle/late
anchors 上验证（8 episodes、48 blocks、300 candidates/block）。结果为 predictor-level
**NO-GO**：

| arm | terminal Spearman med/min | terminal top-30 med/min | relative MSE | gate |
|---|---:|---:|---:|---|
| online control | `0.828863 / -0.321581` | `0.550000 / 0` | `0.143349` | **NO-GO** |
| EMA primary | `0.815925 / -0.344404` | `0.466667 / 0` | `0.147960` | **NO-GO** |

EMA 的 early stratum 通过 local median thresholds（Spearman `0.962553`、top-30 `0.800000`），
但 middle/late 分别只有 `0.486430/0.283333` 与 `0.280386/0.266667`。EMA − online 的
episode-median paired delta 为 Spearman `-0.018833`、top-30 `-0.033333`，仅 `2/8`
episodes 满足 joint non-worsening improvement，因此机制 secondary gate 仍 **NO-GO**。

EMA predictor latency 为 `1.8412 ms`，teacher 为 `20.5635 ms`，reduction `91.0465%`；
计时只覆盖 cached latent + five-step predictor。training convergence、causality、EMA
isolation 与 same-trajectory checks 均通过；GPU telemetry 由作业内每 5 秒记录。该轮没有
运行 official CEM、planner viability 或 closed-loop。完整结果见
[RESULT_LEWM_EMA_TEMPORAL_CONFIRM.zh.md](reports/RESULT_LEWM_EMA_TEMPORAL_CONFIRM.zh.md)。
报告使用已核实的 [Scientific Agent Skills, arXiv:2609.00065](https://arxiv.org/abs/2609.00065) 作为方法参考，不把它当作本实验结果证据。

## LeWM state-action prefix GRU 更新

在 balanced temporal training 之后，我们做了唯一一轮固定设计的结构实验：control
是原始 h256 balanced recurrent student，treatment 只加入一个 shared 64-D
state-action prefix GRU。Fresh test 为全新的 `valid[536:544]`，8 episodes、48
blocks、300 candidates/block；两臂共享 init、balanced anchors、candidate slates、
schedule、loss 与 3000 updates。

正式 job `25158955.pbs101`：`Exit_status=0`、walltime `00:04:58`。

| arm | Spearman med/min | top-30 med/min | relative MSE | absolute |
|---|---:|---:|---:|---|
| balanced base | `0.824964/-0.624814` | `0.533333/0` | `0.100185` | **NO-GO** |
| state-action prefix GRU | `0.841137/-0.639435` | `0.566667/0` | `0.094277` | **NO-GO** |

GRU−base 的 paired mechanism gate 为 **GO**（median ΔSpearman `+0.023855`、
Δtop-30 `+0.025000`、joint `5/8`），但三个 temporal strata 与 combined primary
gate 均 **NO-GO**。GRU predictor 相对 base overhead `41.56%`，高于 `35%` guard；
teacher-relative reduction 仍为 `88.42%`。这支持“轻量 state-action memory 有小幅
paired ranking signal”，不支持 teacher replacement，也不触发 CEM/planner/closed-loop。

完整报告：[RESULT_LEWM_STATE_ACTION_PREFIX_GRU.zh.md](reports/RESULT_LEWM_STATE_ACTION_PREFIX_GRU.zh.md)。
