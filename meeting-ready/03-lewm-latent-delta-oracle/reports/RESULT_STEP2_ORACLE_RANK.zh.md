# Step 2 结果：oracle projection 与 planner decision

状态：**完成；candidate diagnostic 通过的 model reusable ranks 为 64、96，但 fixed-observation CEM action gate 均 FAIL**。PBS job：`24844057.pbs101`，退出码 `0`。权威结果：[summary.json](../artifacts/24844057.pbs101/summary.json)；运行记录：[job.log](../artifacts/24844057.pbs101/job.log)。

## 实验边界

每条 approximate path 都先完整运行 LeWM predictor，再对 `5×192` full-model delta 做 projection/reconstruction，并接回 unchanged official criterion。candidate 评估为 `16` 个 held-out blocks（`8` contexts × `2` banks × `300` candidates）。CEM 只在相同 observation/seed 下做 fixed-observation official diagnostic：`300` samples、`30` iterations、`topk=30`、horizon `5`。没有 environment interaction；不声称 predictor speedup 或 closed-loop success。

## 1. Candidate ranking diagnostic

表中 `Spearman` 与 `top-30` 均为 `median / minimum`；`first L2` 与 teacher-objective regret 为 `median / maximum`。`gate` 使用冻结 candidate gate 的 summary 判定。

### Per-trajectory oracle SVD（hindsight lower bound，非 reusable）

| rank | Spearman | top-30 | argmin match | first L2 | teacher regret | gate |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.9112 / 0.6536 | 0.8500 / 0.3667 | 0.6250 | 0 / 2.0490 | 0 / 11.1083 | FAIL |
| 2 | 0.9927 / 0.9136 | 0.9667 / 0.7333 | 0.8125 | 0 / 1.6469 | 0 / 2.0102 | FAIL |
| 3 | 0.9982 / 0.9967 | 0.9667 / 0.9000 | 0.9375 | 0 / 0.6360 | 0 / 0.2726 | PASS |
| 4 | 0.9998 / 0.9994 | 1.0000 / 0.9667 | 1.0000 | 0 / 0 | 0 / 0 | PASS |
| 5 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 | 0 / 0 | 0 / 0 | PASS |

这些 PASS 只表示 candidate diagnostic 在 hindsight oracle 分支下满足记录的 candidate gate；它们不能用于 CEM replacement，也不是可复用 basis。

### Reusable model-PCA（primary）

| rank | Spearman | top-30 | argmin match | first L2 | teacher regret | gate |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 0.6618 / −0.7514 | 0.4167 / 0.0000 | 0.1250 | 0 / 4.2780 | 0 / 58.5062 | FAIL |
| 8 | 0.8821 / 0.0247 | 0.7667 / 0.0333 | 0.5625 | 0 / 7.6945 | 0 / 49.8023 | FAIL |
| 16 | 0.9836 / 0.1797 | 0.8500 / 0.1000 | 0.5625 | 0 / 1.1048 | 0 / 49.8023 | FAIL |
| 32 | 0.9915 / 0.4850 | 0.9333 / 0.3000 | 0.6875 | 0 / 2.0490 | 0 / 25.6543 | FAIL |
| **64** | **0.9992 / 0.9867** | **0.9667 / 0.8667** | **0.8750** | **0 / 0.6360** | **0 / 0.4546** | **PASS** |
| **96** | **1.0000 / 0.9982** | **1.0000 / 0.9667** | **0.9375** | **0 / 0.6360** | **0 / 0.2726** | **PASS** |
| 128 | 1.0000 / 0.9999 | 1.0000 / 0.9667 | 1.0000 | 0 / 0 | 0 / 0 | PASS |
| 160 | 1.0000 / 1.0000 | 1.0000 / 0.9667 | 1.0000 | 0 / 0 | 0 / 0 | PASS |
| 192 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 | 0 / 0 | 0 / 0 | PASS |

因此 candidate-level primary gate 的最小 reusable model-PCA rank 是 **64**。这只是 fixed candidate-bank diagnostic；它不推翻 Step 1 的 reconstruction structure gate FAIL，也不等于 CEM action gate 通过。

### Observed-PCA cross-basis

| rank | Spearman | top-30 | argmin match | first L2 | teacher regret | gate |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 0.9998 / 0.9988 | 1.0000 / 0.9667 | 1.0000 | 0 / 0 | 0 / 0 | PASS |
| 96 | 1.0000 / 0.9998 | 1.0000 / 0.9667 | 1.0000 | 0 / 0 | 0 / 0 | PASS |
| 192 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 | 0 / 0 | 0 / 0 | PASS |

这里的 `PASS` 是把 observed-PCA 当作 candidate cross-basis control 的结果，不能把它改写成 primary reusable model basis 的 structure PASS。

### Negative/full-rank controls

- deterministic random rank-64 basis：Spearman `0.9941 / 0.8695`，top-30 `0.9333 / 0.7333`，argmin match `0.75`，teacher regret `0 / 2.4225`，candidate gate **FAIL**。这排除了“任意 rank-64 basis 都等价”的解释。
- model rank-192 full-rank control：Spearman/top-30/argmin 均为 `1.0`，first-action L2 与 teacher regret 均为 `0`；artifact 标记 `approximately_exact=true`（tolerance `1e-5`），control **PASS**。

## 2. Fixed-observation CEM：selected ranks 与 action drift

由于 model-PCA rank 64、96 是最小两个 candidate-gate passing ranks，冻结 policy 选择它们进入 CEM；另有 baseline 和 rank-192 exact control。每行是同一 observation/seed 下相对 baseline 的 first-action drift；冻结 planner gate 为 normalized L2 `≤0.15` 且 absolute difference `≤0.25`。

| observation | seed | model rank 64: L2 / abs-max | model rank 96: L2 / abs-max |
|---|---:|---:|---:|
| `pusht_obs_00` | 20300921 | 0.4136 / 3.5279 | 0.2924 / 1.6887 |
| `pusht_obs_00` | 20300922 | 0.2446 / 1.4809 | 0.1037 / 0.6444 |
| `pusht_obs_01` | 20300921 | 0.9573 / 4.9527 | 0.6034 / 3.3842 |
| `pusht_obs_01` | 20300922 | 0.6982 / 2.2021 | 0.7699 / 2.9027 |

- rank 64：4/4 records 同时超过至少一个阈值，且最大 drift 为 `0.9573 / 4.9527`；**planner gate FAIL**。
- rank 96：4/4 records 的 absolute difference 都超过 `0.25`（normalized L2 也有 3/4 超过 `0.15`）；**planner gate FAIL**。
- rank-192 full control：4/4 records 的 first-action normalized L2 与 absolute difference 均为 `0`；**PASS**。

## 结论

candidate ranking 在 reusable model-PCA rank 64/96 上看起来通过，但相同 ranks 接入 official fixed-observation CEM 后，first action 漂移明显超出冻结 planner gate。因此当前结果只支持：**rank 64/96 是 candidate-level oracle diagnostic 的通过点，不支持 fixed-observation CEM action preservation，更不支持 speedup 或 closed-loop success。**
