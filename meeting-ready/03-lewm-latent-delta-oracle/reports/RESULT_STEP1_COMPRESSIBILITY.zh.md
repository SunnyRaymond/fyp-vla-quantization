# Step 1 结果：LeWM latent-delta compressibility

状态：**完成；primary reusable model-PCA structure gate FAIL（rank ≤64）**。PBS job：`24844057.pbs101`，退出码 `0`。权威结果：[summary.json](../artifacts/24844057.pbs101/summary.json)；运行记录：[job.log](../artifacts/24844057.pbs101/job.log)。旧 job `24841804.pbs101` 是 superseded initial run，缺少 per-horizon 字段；核心 candidate/CEM 数值在 corrective job 中复核一致。

## 实验边界

Baseline 是 official LeWM PushT，planner-facing latent 为 `192-D CLS vector`。calibration / held-out 为 episode-disjoint 的 `256 / 8` contexts；observed offsets 为 `0/5/10/15/20/25`。PCA 只用 calibration delta 拟合；本结果不表示 cheap predictor、runtime speedup 或 closed-loop success。

## 1. Reusable PCA：observed 与 model 必须分开看

下表是 calibration basis 的 centered cumulative variance；held-out 指标用同一 basis 在 held-out delta 上的结果。

| basis | fit samples | rank 32 | rank 64 | rank 96 | rank 128 | rank 192 |
|---|---:|---:|---:|---:|---:|---:|
| observed | 1,280 | 0.7885 | 0.9574 | 0.9950 | 0.9990 | 1.0000 |
| model（primary） | 5,120 | 0.7800 | 0.9375 | 0.9843 | 0.9954 | 1.0000 |

held-out same-basis 的关键 worst-case 值如下；括号内为 `(centered-variance retained minimum, relative-MSE maximum)`：

| reusable basis | rank 64 | rank 96 | rank 128 | rank 160 |
|---|---:|---:|---:|---:|
| observed-on-observed | `(0.7606, 0.2450)` | `(0.9614, 0.0395)` | `(0.9936, 0.0066)` | `(0.9975, 0.0025)` |
| model-on-model（primary） | `(0.5728, 0.4059)` | `(0.7734, 0.2167)` | `(0.8969, 0.0987)` | `(0.9342, 0.0636)` |

### Rank-64 的 per-horizon tail

以下直接取 corrective `summary.json` 的 `step1.frontier.heldout_by_horizon`；每格为
`(retained minimum, relative-MSE maximum)`。这是按 horizon 的记录，不另作 horizon-independent
推断。

| horizon | model-on-model | observed-on-observed |
|---:|---:|---:|
| h1 | `(0.5728, 0.4059)` | `(0.9005, 0.1053)` |
| h2 | `(0.7760, 0.2588)` | `(0.9238, 0.0736)` |
| h3 | `(0.7512, 0.3115)` | `(0.8819, 0.1191)` |
| h4 | `(0.7442, 0.3349)` | `(0.8436, 0.1565)` |
| h5 | `(0.7524, 0.3158)` | `(0.7606, 0.2450)` |

冻结 structure gate 要求 rank `≤64` 且 retained `≥0.95`、relative MSE `≤0.05`。corrective artifact 的 `step1.structure_gate.evaluation` 明确记录 observed/model 的 `passing_ranks=[]` 与 `rank_le_64_overall_pass=false`。因此：

- **model reusable PCA：正式 gate FAIL，passing ranks 为空**。rank 64 同时未达到两个 worst-case 要求；若仅作忽略 rank 上限的 reconstruction 诊断，首个满足这两个 worst-case 条件的是 rank **192**（rank 160 仍为 `0.9342 / 0.0636`），这不叫 gate pass。
- **observed reusable PCA：正式 gate FAIL；rank 96 是超出上限的弱证据**。rank 64 仍为 `0.7606 / 0.2450`；若忽略 rank 上限，rank **96** 首个满足 worst-case reconstruction 条件，但 corrective evaluation 仍将其记为 `passed=false`。

这说明 observed 与 model delta 的 global subspace 不是同一个可直接复用的低秩结构；尤其 model-PCA 在 rank 64 的 held-out tail 误差仍明显超阈值。per-horizon 表不改变 formal gate：最大允许 rank 仍为 `64`，因此没有正式 passing rank。

## 2. Cross-basis control

rank 64 的 held-out cross reconstruction 仍不满足上述结构阈值：

| reconstruction | retained minimum | relative-MSE maximum | 结论 |
|---|---:|---:|---|
| observed delta → model basis | 0.7200 | 0.2803 | FAIL / cross-basis 不稳 |
| model delta → observed basis | 0.6022 | 0.5945 | FAIL / cross-basis 不稳 |

作为参考，rank 64 的 held-out median relative MSE 为 `0.0681`（model-on-model）、`0.0510`（observed-on-observed）、`0.0712`（observed-on-model）和 `0.0951`（model-on-observed）；median 接近阈值不能抵消 worst-case gate 的失败。

## 3. Per-trajectory `5×192` oracle SVD（非 reusable）

这是每个 full-model trajectory/candidate 自己拟合的 hindsight SVD，不是 global basis，也没有 runtime/acceleration evidence。summary 记录的是 held-out `8 × 2 × 300 = 4,800` 个 model-rollout candidate stacks；没有单独记录 observed-trajectory oracle aggregate，故不将两者混写。

| oracle rank | retained-energy median | retained-energy minimum | relative-MSE median |
|---:|---:|---:|---:|
| 1 | 0.6753 | 0.4187 | 0.3245 |
| 2 | 0.9275 | 0.7311 | 0.0725 |
| 3 | 0.9848 | 0.9208 | 0.0152 |
| 4 | 0.9967 | 0.9779 | 0.0033 |
| 5 | 1.0000 | 1.0000 | ~0 |

该 oracle 只能说明“知道每条 trajectory 自己的最佳方向时”的信息下界；不能把 rank 3/4 写成 reusable subspace 或 predictor speedup。Step 2 会单独报告其 planner diagnostic。

## 结论

在冻结 `rank≤64` structure gate 下，**primary model-delta reusable low-rank hypothesis 暂不成立（FAIL；formal passing ranks 为空）**。observed rank 96 与 model rank 192 只是在忽略 rank 上限时的 reconstruction diagnostics，不是 gate pass；per-trajectory oracle 的低误差不改变 global reusable gate 的结论。
