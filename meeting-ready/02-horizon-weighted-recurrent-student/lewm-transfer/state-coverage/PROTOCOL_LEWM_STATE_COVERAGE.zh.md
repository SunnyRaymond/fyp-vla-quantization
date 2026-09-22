# LeWM PushT：Phase 2 state/context coverage 冻结协议

状态：`frozen / predictor-level only / pending one formal GPU job`

## 1. 问题与 2×2 设计

本轮只区分 train state/context coverage 与 optimization step budget。student
仍是 Phase 1 `score_distill` 的原始 `LeWMCompactRecurrentTransitionStudent`：
192-D latent、packed action 10-D、shared recurrent h256、predicted-latent
free-running feedback；architecture、horizon-weighted latent loss、context-normalized
teacher-score SmoothL1、optimizer、batch size、candidate bank、held-out protocol
全部保持不变。

已完成的 `256 contexts × 1500 updates` score-distill 正式结果作为 reference，
本轮只新增三个 arm：

| arm | train contexts | updates | 作用 |
|---|---:|---:|---|
| `256x1500` | 256 | 1500 | 已完成 reference，不重跑 |
| `256x3000` | 256 | 3000 | step-budget control |
| `512x1500` | 512 | 1500 | coverage at fixed updates；primary treatment |
| `512x3000` | 512 | 3000 | coverage + matched exposure；interaction/exposure aid |

所有新 arm 使用同一 initialization state、AdamW、batch size 8、score-distill loss、
context schedule seed 20300904 和冻结的 64-candidate train bank construction。
`512x1500` 与 reference 的正式 treatment gate 在读取结果前已写入
`LEWM_STATE_COVERAGE_FREEZE.json`。

## 2. Context manifest 与 prefix invariance

旧 256 manifest 和 held-out 8 rows 原样保留。`build_context_manifest` 的确定性规则
是一次性 shuffle valid episode IDs，然后取 `heldout=valid[:8]`，再取
`train=valid[8:8+train_target]`。因此 512 manifest 必须满足：

1. held-out 8 rows 与旧 manifest 逐字相同；
2. 新 train 的前 256 rows 与旧 train 逐字相同；
3. 新增 train rows 为旧 train 之后的 episode-disjoint suffix；
4. 所有 anchors 仍为 episode step 0，future action start 仍为 0，goal offset 仍为 25；
5. 512 prepared rows 的旧 train prefix 与旧 prepared rows bitwise 相同；
6. 512 prepared rows 的 held-out actions、teacher targets、teacher objectives、latent
   history、goal embedding 与旧 held-out bank bitwise 相同。

若任一 prefix/held-out/candidate bank 一致性检查失败，作业必须在训练前退出，不能
解释结果或自行更换 manifest。512 manifest、HDF5 读取、teacher preparation 和
benchmark 只能在 PBS compute allocation 中完成；本机和 login node 不读取新大数据。

## 3. Frozen evaluation gates

统计单位是 held-out context × fresh action-prefix seed，共 16 blocks；candidate 不是
独立 statistical replicate。评估完全复用 Phase 1：每个 block 300 candidates、top-k
30、seeds `[20300907, 20300908]`，报告 snapshot 500/1000/1500，并对 3000 arms
增加 snapshot 3000。

`512x1500` 相对已完成 `256x1500` reference 必须同时满足：

- Spearman median `>= 0.974011`；
- top-30 median `>= 0.816667`；
- relative latent MSE median `<= 0.0175`；
- positive top-30 blocks **恰好 16/16**；
- minimum top-30 overlap `> 0`。

报告四个 cell 的 paired context-level deltas（Spearman、top-30、relative latent
MSE），并独立报告每个新 arm 的原 absolute predictor gate。任何 absolute gate 失败
均为 predictor-level `NO-GO`，不因 phase gate 结果放宽。

## 4. Scope 与运行边界

只运行 predictor-level training/evaluation、causality 和 predictor-only latency。
不运行 official CEM、planner viability、closed-loop PushT、environment interaction，
也不把 predictor latency 解读为 end-to-end control speedup。candidate ranking 的
改善不等于 planner-facing replacement；所有失败或 inconclusive 结果原样保留。

正式作业必须是单个 bounded GPU PBS job，job log 每约 5 秒记录 GPU utilization 与
VRAM；完成后只回传 summary、job log、job status 和小型 manifest，不回传 checkpoint
或 `prepared_rows.pt`。
