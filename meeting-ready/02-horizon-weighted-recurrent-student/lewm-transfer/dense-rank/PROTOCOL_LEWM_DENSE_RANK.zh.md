# LeWM PushT：Dense Query + Elite-Boundary Rank Distillation 冻结协议

状态：`frozen / predictor-level only`

## 1. 目的与设计

本实验只改变 train action-query coverage 与 training supervision，不改变 student
inference structure。三臂都使用原始 `LeWMCompactRecurrentTransitionStudent`：
`192-D` latent、`10-D` packed action、h256 shared recurrent、predicted-latent
free-running feedback、无 attention、无 conditioner、无 hidden-size expansion。

同一批 256 个 episode-disjoint train contexts 各使用同一 frozen 64-candidate
teacher bank；同一 8 个 held-out contexts 使用同一 2×300 candidate bank。三臂从同一
initial state、同一 context schedule、同一 1500-step budget 开始。

| arm | latent loss | rank loss |
|---|---|---|
| `dense_latent` | horizon-weighted free-running latent MSE | none |
| `dense_rank` | same | teacher top 6 vs ranks 7–12 |
| `shuffled_rank` | same | same pair count, fixed context-internal label shuffle |

Rank loss 使用 external official LeWM criterion 对 student predicted latent 计算 cost；
goal 只进入该 external loss，绝不输入 student。每个 context 的 teacher cost 先按
其 standard deviation 做 normalization，pairwise loss 固定 `temperature=0.5`、
`lambda=0.1`。这些值在收集结果前冻结，不做 sweep 或 result-dependent retuning。

## 2. Screening gate

正式使用 step 1500 的 held-out 16 blocks，primary 为 top-30 overlap，secondary 为
Spearman 与 relative latent MSE。`dense_rank` 相对 `dense_latent` 必须同时满足：

- top-30 median delta `>= +0.10`；
- 至少 `10/16` paired blocks 改善；
- minimum top-30 不下降；
- positive top-30 block count 不下降；
- Spearman median 不下降；
- `shuffled_rank` 不能满足同一收益条件。

原有 absolute predictor feasibility gate 仍对每个 arm 单独报告；任意未达成只保留
相应 NO-GO，不放宽门槛。统计单位为 held-out block，不能把 candidates 当独立样本。

## 3. 评估与边界

报告每个 arm 的 step 500/1000/1500 metrics、step1500 paired deltas、relative
latent MSE、causality、predictor-only latency、training telemetry 与 absolute gate。
本实验停在 predictor-level；不运行 official CEM、planner viability 或 closed-loop。
