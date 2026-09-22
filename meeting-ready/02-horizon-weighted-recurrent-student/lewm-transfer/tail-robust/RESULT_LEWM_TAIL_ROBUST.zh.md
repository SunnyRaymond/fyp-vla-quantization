# LeWM PushT Phase 3 tail-robust EMA：实验结果

## 结论

正式作业 `24933682.pbs101` 正常完成（PBS `Exit_status=0`，walltime
`00:03:39`）。primary `ema_tail_score` 通过全部 inherited absolute predictor gates，
但没有通过预注册的 historical no-regression gate，因此总体结论是 **NO-GO**。

更具体地说，EMA 把 historical `512×3000` reference 的 worst-block Spearman/top-30
从 `0.656421/0.366667` 提高到 `0.85+/0.53+`，使两个新 arm 都达到 `16/16`
positive blocks；但 median Spearman/top-30 从 historical `0.978317/0.866667`
回退到约 `0.974/0.817`。这是 tail 与 median 的 trade-off，不是无回退改进。

此外，预注册的 top-2 context tail emphasis 没有优于普通 EMA：`ema_score` 在 terminal
Spearman median/minimum 和 relative latent MSE 上均略好。因此本轮的正向 tail signal
应归因于 EMA，而不是 tail-weighted score loss。

## 冻结设计

- 权威输入：Phase 2 formal job `24926383.pbs101` 的 512-context manifest、prepared
  rows 和 `512×3000` historical reference；
- 两个新 arm：`ema_score` 与 `ema_tail_score`；
- 相同 initialization、AdamW、3000 updates、context schedule、held-out 16 blocks；
- EMA decay `0.999`，terminal evaluation 固定使用 `EMA(step=3000)`；
- `ema_tail_score` 只对 score-distill component 使用
  `0.75×mean + 0.25×mean(top-2 context losses)`；
- predictor-level only；official CEM、planner viability 和 closed-loop 均未运行。

## Terminal metrics

| arm | Spearman median / min | top-30 median / min | relative latent MSE median | positive blocks |
|---|---:|---:|---:|---:|
| historical `512×3000` online | `0.978317 / 0.656421` | `0.866667 / 0.366667` | `0.007499` | `16/16`, `16/16` |
| `ema_score` | `0.974618 / 0.854591` | `0.816667 / 0.566667` | `0.008968` | `16/16`, `16/16` |
| `ema_tail_score` | `0.973554 / 0.851060` | `0.816667 / 0.533333` | `0.009391` | `16/16`, `16/16` |

`ema_tail_score` 的 absolute predictor gate 全部 PASS：ranking、latent MSE、finite、
convergence、causality 与 predictor-only latency 均通过；student `1.584 ms`、teacher
`19.265 ms`，predictor-boundary reduction `91.78%`。该 latency 不包含 encoder、CEM
或 environment。

## Paired tail-emphasis effect

在相同的 16 个 held-out blocks 上，`ema_tail_score - ema_score`：

- Spearman median delta `-0.001011`，`6/16` 改善、`10/16` 回退；
- top-30 median delta `0`，`4/16` 改善、`8/16` 持平、`4/16` 回退；
- relative latent MSE median delta `+0.000369`，`12/16` blocks 更差。

因此不能把 worst-block 改善归因于 tail emphasis；普通 EMA 是更简洁且整体更好的
new arm，但它同样没有通过 historical median no-regression thresholds。

## Snapshot trajectory

| EMA snapshot | `ema_score` Spearman / top-30 median | `ema_tail_score` Spearman / top-30 median |
|---|---:|---:|
| 500 | `0.255895 / 0.166667` | `0.201206 / 0.133333` |
| 1000 | `0.625460 / 0.433333` | `0.626113 / 0.416667` |
| 1500 | `0.878464 / 0.683333` | `0.900285 / 0.650000` |
| 3000 | `0.974618 / 0.816667` | `0.973554 / 0.816667` |

EMA 收敛较慢；step 3000 才通过 absolute predictor gate。不能根据 descriptive
snapshots 事后选择更有利的 checkpoint。

## 冻结 gate 与决策

primary `ema_tail_score`：

- inherited absolute predictor feasibility：**GO**；
- no-regression vs historical `512×3000`：**NO-GO**；
- overall primary gate：**NO-GO**。

决策：停止 top-2 tail weighting recipe；不因 absolute gate PASS 而越过预注册的
no-regression gate 去运行 official CEM 或 closed-loop。EMA 的 worst-block 改善保留为
后续设计依据，但任何新实验必须作为独立 freeze，而不能改写本轮 primary outcome。

## Claim boundary

本结果仅支持 predictor-level EMA/tail score-distillation comparison。它不支持 official
CEM viability、planner viability、closed-loop PushT success、encode_obs speedup 或
native deployment benefit。
