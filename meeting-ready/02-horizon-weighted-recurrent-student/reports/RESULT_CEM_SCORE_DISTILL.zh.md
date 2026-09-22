# DINO-WM PushT：CEM proposal score distillation 结果

## 结论

正式作业 `25145484.pbs101` 正常完成（`final_exit_code=0`），按冻结协议完成了
500 updates 的三臂 predictor-level training。`score_distill` 的 absolute predictor
gate 通过，但相对 `latent_only` 的 paired gate 失败；因此 Stage 1 总体为
**FAIL**，Stage 2 fixed-observation CEM 按协议为 **SKIPPED**。

这说明 context-normalized teacher-score supervision 在本次固定 proposal 数据上可以
达到绝对 ranking 门槛，但没有达到预先冻结的、足够大的相对 improvement 门槛。它不是
继续调同一路径或放宽 threshold 的依据，本 frozen recipe 记为 **NO-GO**。

## 实验范围与执行状态

- backend：official DINO-WM PushT；三臂为 `latent_only`、`score_distill` 和
  `shuffled_score`。
- 输入：复用 `24928207.pbs101` 已生成的 4800 行 proposal-label 数据；不重新采集
  teacher labels。
- training：相同 horizon-weighted step-1500 warm start、optimizer、schedule、
  500 updates 和 replay bank；每个 update 的三臂共享 candidate rows。
- Stage 1：8 个 held-out contexts、16 个 held-out blocks、每 block 300 candidates、
  top-30 ranking；candidate 不是独立 statistical replicate，统计单位是 block。
- validity：所有输出 finite，causality checks 通过，未发生 silent fallback；没有
  environment interaction、student closed loop 或 official planner deployment。

## Stage 1 absolute predictor gate

| arm | Spearman median / mean / min | top-30 median / mean / min | absolute gate |
|---|---:|---:|---:|
| `latent_only` | `0.967509 / 0.961855 / 0.906688` | `0.783333 / 0.779167 / 0.600000` | **FAIL** |
| `score_distill` | `0.974295 / 0.964824 / 0.908604` | `0.816667 / 0.795833 / 0.600000` | **PASS** |
| `shuffled_score` | `0.964781 / 0.962079 / 0.896153` | `0.816667 / 0.785417 / 0.600000` | **PASS** |

`score_distill` 满足冻结的 absolute thresholds：Spearman median/minimum 至少
`0.95/0.85`，top-30 median/minimum 至少 `0.80/0.50`，causality maximum absolute
delta 为 `0`，且 outputs finite、没有 silent fallback。`latent_only` 只在 top-30
median 上为 `0.783333`，低于 `0.80`；这不改变 score arm 的 paired gate 结论。

## 为什么 score_distill absolute PASS 仍然是 paired FAIL

冻结协议要求 score arm 同时满足两类相对条件：Spearman median delta 至少 `+0.005`
且至少 `10/16` blocks 改善；top-30 median delta 至少 `+1/30` 且至少 `10/16`
blocks 改善。`score_distill - latent_only` 的结果为：

| paired metric | median delta | mean delta | improve / worse / tie | 冻结要求 | 结果 |
|---|---:|---:|---:|---:|---:|
| Spearman | `+0.001877` | `+0.002970` | `13 / 3 / 0` | median `>=+0.005`，improve `>=10` | **FAIL**（delta 太小） |
| top-30 overlap | `0` | `+0.016667` | `7 / 2 / 7` | median `>=+0.033333`，improve `>=10` | **FAIL**（两项均未满足） |

所以这里的关键不是 score arm 绝对表现差，而是它相对同一 warm start、同一 candidate
schedule 的 gain 不够大且不够广泛：Spearman 虽有 `13/16` blocks 改善，median
improvement 仍只有 `+0.001877`；top-30 则有 `7/16` 改善、`7/16` ties，median
仍为零。作为补充，terminal relative latent MSE 的 `score_distill - latent_only`
median/mean 为 `-0.001105/-0.001823`，`10/16` blocks 改善、`6/16` 回退；该指标
不是本轮 paired gate 的放行条件，也不能替代 planner-facing ranking evidence。

## Shuffled negative control

`shuffled_score` 没有复现 score-distill 的 paired gain（summary 中
`shuffled_reproduces_paired_gain=false`）：

- Spearman：median/mean delta `-0.000391/+0.000224`，`6/16` 改善、`10/16`
  回退；
- top-30：median/mean delta `0/+0.006250`，`5/16` 改善、`4/16` 回退、`7/16`
  持平。

这支持一个较弱的判断：score target 的对应关系不是完全可置换的噪声；但 negative
control 不复现并不能补偿 score arm 自身未达到 paired thresholds 的事实。因此本轮
仍是 Stage 1 **FAIL**，而不是 mechanism GO。

## Stage 2 与 claim boundary

Stage 2 状态为 **SKIPPED**，原因是“Stage 1 score absolute/paired/negative-control
gate failed”。没有产生 case-level CEM trace，也没有运行新的 fixed-observation
CEM、environment rollout 或 closed-loop evaluation。

本报告允许的 claim 只有：在冻结的 fixed-observation CEM-proposal 数据和 held-out
predictor protocol 下，对 `score_distill` 训练 objective 的 predictor/mechanism
diagnosis。本报告不支持：

- CEM ranking、iteration-30 drift 或 planner performance improvement；
- new closed-loop success rate、student environment rollout 或 official planner
  deployment；
- population-level inference、LeWM/Fast-LeWM transfer 或 native low-bit deployment。

## 下一步建议

不建议继续调同一路径的 `score_loss_weight`、top-k 权重或 threshold，也不建议为本次
结果补跑相同 recipe。当前证据更适合把研究重点转向直接的 **sequential/planner-aware
correction**：

1. 在已有固定 CEM trace 上构造按 iteration/prefix 对齐的 correction target，直接
   约束 candidate ranking、elite membership 与 first-action stability，而不是只回归
   单个 proposal 的 context-normalized scalar score。
2. 用同一 frozen observation 和同一 innovation 复用 `latent_only`、sequential
   correction、以及 time/prefix-shuffled negative control；先做 predictor-level
   paired gate，再决定是否允许固定观测 CEM mechanism gate。
3. 将 per-step correction 与 rollout-consistency/teacher-shadow signal 分开报告，
   重点观察误差是否在 CEM feedback loop 中被放大；不要把静态 top-30 或 latent MSE
   改写成 closed-loop 证据。

该方向仍须预先冻结 gate、保持 candidate/block 配对，并继续遵守“no threshold
loosening / no same-recipe rerun”。若未来需要新的 teacher trace 或大文件操作，应在
获批的 PBS compute allocation 中完成，不能回退到 login node。

## 可复核输入

- [summary](../artifacts/25145484.pbs101/cem_score_distill_summary.json)
- [freeze](../config/CEM_SCORE_DISTILL_FREEZE.json)
- [protocol](../config/PROTOCOL_CEM_SCORE_DISTILL.zh.md)
