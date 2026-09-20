# 组会卡片：LeWM PushT iteration cache

## 一句话

CEM 的每一轮都在重复编码同一个 observation 和 goal；把这两个固定 embedding
移到一次 `solve` 开始时缓存，可以在不改变 planner 行为的情况下减少约 29.4%
的 full-planner latency。

## 改了什么

```text
官方 CEM solve
  ├─ 每轮 get_cost：initial encode + goal encode + action-conditioned rollout
  └─ CEM update

iteration-cache solve
  ├─ solve 开始：initial encode + goal encode（各一次）
  ├─ 每轮 get_cost：复用 fixed embeddings
  │              + 原 action encoder/predictor/criterion
  └─ 原 CEM update
```

只缓存 detached `initial_embedding` 和 `goal_embedding`。不缓存 action embedding、
predicted embedding、mutable info state 或 CEM distribution，因此 candidate ranking
和最终 action 仍由原始路径决定。

## 结果

| 指标 | 结果 |
|---|---:|
| paired exactness | 30/30 units，bitwise exact |
| full planner latency reduction | 29.4091% |
| plan section reduction | 29.4358% |
| inner CEM cost reduction | 29.7539% |
| peak memory max ratio | 0.97083 |
| overall frozen gate | PASS |

## 该结果说明什么

这是一个很干净的 systems-level speedup：省掉的是重复的 fixed-context
encoding，action-conditioned predictor 没有被近似或替换。因为行为 trace 完全
一致，可以把 latency 收益归因到 cache，而不是 planner stochasticity 或输出变化。

## 不要过度外推

证据边界是官方 LeWM PushT CEM、两个固定 observation、固定 seed/solver 配置；
没有测试其他 WM、其他 planner 或 closed-loop task success。下一步若扩展，应先
在另一个 planner/model 上复制 exactness + latency gate。

## 明天之后从哪里继续

从 `lewm_pusht_iteration.pbs` 开始；freeze 和 verifier 在同一目录。
只需要补齐 external LeWM source、官方 checkpoint/dataset 和远端 Python 环境。
