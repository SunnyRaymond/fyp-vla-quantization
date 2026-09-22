# LeWM PushT：Phase 2 state/context coverage 结果

## 结论先行

正式作业 `24926383.pbs101` 在 official LeWM PushT backend 上完成了冻结的单作业
predictor-level 2×2 新增 arms（`Exit_status=0`，PBS walltime `00:06:05`）。
Phase 1 已完成的 `256×1500 score_distill` 作为 reference，没有重跑；本轮只新增
`256×3000`、`512×1500`、`512×3000`。

primary treatment `512×1500` 的 coverage gate 为 **NO-GO**：Spearman median
`0.966364`（要求 `>=0.974011`）、top-30 median `0.800000`（要求 `>=0.816667`），
relative latent MSE median `0.011604`（PASS），positive top-30 blocks `16/16`
（PASS），minimum top-30 `0.433333`（PASS，但仍未达到 inherited absolute
predictor gate 的 `0.50`）。因此扩大 train context coverage 在固定 1500 updates 下
改善了最差 block，但没有达到冻结的 primary median gate；不支持 GO 或 planner-facing
替换。

`512×3000` 在增加 exposure 后达到更高 median ranking（Spearman `0.978317`、
top-30 `0.866667`、relative MSE `0.007499`），但 inherited absolute predictor
gate 仍为 **NO-GO**（minimum Spearman `0.656421`、minimum top-30 `0.366667`）。
`256×3000` 的 Spearman/top-30 median 也上升，但 minimum top-30 仍为 `0`，不能把
extra steps 单独解释为可靠 coverage replacement。

## 1. 冻结公平性与 scope checks

- architecture、loss、optimizer、batch size、64-candidate train bank、held-out
  8 contexts × 2 seeds（16 blocks × 300 candidates）均保持 Phase 1 不变；
- 512 manifest 的 held-out 8 rows 与旧 manifest 逐字相同；旧 train 256 rows 是新
  train 的 prefix；新增 suffix episode-disjoint；所有 temporal anchors 仍为 step 0；
- 512 prepared rows 的旧 256 train prefix 与旧 prepared rows bitwise 相同；held-out
  candidate actions、teacher targets、teacher objectives、latent/goal inputs
  bitwise 相同；
- scope leakage：`PASS`；candidate 不是独立 statistical replicate。

| check | result |
|---|---|
| manifest prefix invariance | **PASS** |
| prepared-row train prefix | **PASS** |
| held-out bank equivalence | **PASS** |
| temporal anchor unchanged | **PASS** |

## 2. Terminal metrics

| arm | updates | Spearman median / min | top-30 median / min | relative latent MSE median | positive top-30 | absolute predictor gate |
|---|---:|---:|---:|---:|---:|---|
| `256×1500` reference | 1500 | `0.974011 / -0.010504` | `0.816667 / 0` | `0.013728` | `15/16` | Phase 1 **NO-GO** |
| `256×3000` | 3000 | `0.977582 / -0.351171` | `0.850000 / 0` | `0.008170` | `14/16` | **NO-GO** |
| `512×1500` | 1500 | `0.966364 / 0.790585` | `0.800000 / 0.433333` | `0.011604` | `16/16` | **NO-GO** |
| `512×3000` | 3000 | `0.978317 / 0.656421` | `0.866667 / 0.366667` | `0.007499` | `16/16` | **NO-GO** |

All new arms passed integrity/convergence and causality. Predictor-only latency reduction
was `93.42%` (`256×3000`), `91.96%` (`512×1500`) and `92.01%` (`512×3000`); this is only
the cached latent + five-step predictor boundary and excludes encoder/CEM/environment.

## 3. Snapshot trajectory

| arm | step 500 Spearman / top-30 / relMSE | step 1000 | step 1500 | step 3000 |
|---|---|---|---|---|
| `256×3000` | `0.736084 / 0.533333 / 0.020829` | `0.952318 / 0.733333 / 0.015060` | `0.974011 / 0.816667 / 0.013728` | `0.977582 / 0.850000 / 0.008170` |
| `512×1500` | `0.769849 / 0.583333 / 0.018843` | `0.920935 / 0.700000 / 0.018028` | `0.966364 / 0.800000 / 0.011604` | — |
| `512×3000` | `0.769849 / 0.583333 / 0.018843` | `0.920935 / 0.700000 / 0.018028` | `0.966364 / 0.800000 / 0.011604` | `0.978317 / 0.866667 / 0.007499` |

`512×1500` 和 `512×3000` 的前 1500-step snapshots 完全一致，符合相同 context
schedule seed 与 exposure extension 的冻结设计。

## 4. Paired deltas vs `256×1500` reference

paired unit 是同一 held-out context × 同一 fresh action-prefix seed；以下为 16-block
median delta，candidate 不是 replicate。

| arm | Spearman delta (improve/worse) | top-30 delta (improve/worse/tie) | relative MSE delta (improve/worse) |
|---|---:|---:|---:|
| `256×3000` | `+0.003107` (`10/6`) | `0.000000` (`7/5/4`) | `-0.003312` (`12/4`) |
| `512×1500` | `+0.000471` (`8/8`) | `0.000000` (`7/5/4`) | `-0.001237` (`11/5`) |
| `512×3000` | `+0.007966` (`14/2`) | `+0.033333` (`9/3/4`) | `-0.004482` (`14/2`) |

这些 deltas 说明 3000 updates 带来更一致的 median/latent-MSE 改善，512×1500 的
coverage-only arm 则主要改善 worst-case block（minimum top-30 从 `0` 到 `0.433333`），
但其 median 仍略低于冻结 threshold。由于 512×3000 同时改变 coverage 与 exposure，
不能把它的收益归因于 coverage 单一因素。

## 5. 决策与 claim boundary

| gate | result |
|---|---|
| `512×1500` frozen treatment gate | **NO-GO** |
| all manifest/prepared-row scope checks | **PASS** |
| all new-arm integrity/convergence | **PASS** |
| all new-arm causality | **PASS** |
| all new-arm absolute predictor fidelity | **FAIL / NO-GO** |
| official CEM viability | `NOT_RUN_BY_SCOPE` |
| planner viability | `NOT_RUN_BY_SCOPE` |
| closed-loop PushT | `NOT_RUN_BY_SCOPE` |

本轮支持的结论仅是：在冻结的 LeWM PushT predictor-level protocol 下，扩大 train
context coverage 到 512 并不能在固定 1500 updates 通过预注册 median gate；增加到
3000 updates 对 median ranking 和 latent MSE 更有利，但仍有 worst-case failure，且
512×3000 混合了 coverage 与 exposure。不得声称 planner-facing replacement、end-to-end
control improvement、encode_obs speedup 或 closed-loop success。下一步应停止本 recipe，
保留 NO-GO；只有重新冻结明确的新机制/对照后才可继续。

## 6. Artifacts

- [freeze](LEWM_STATE_COVERAGE_FREEZE.json)
- [protocol](PROTOCOL_LEWM_STATE_COVERAGE.zh.md)
- [formal summary](artifacts/24926383.pbs101/lewm_state_coverage_summary.json)
- [job log with 5-second GPU telemetry](artifacts/24926383.pbs101/job.log)
- [job status](artifacts/24926383.pbs101/job_status)
- [256-context reference summary](../score-distill/artifacts/24916520.pbs101/lewm_score_distill_summary.json)
