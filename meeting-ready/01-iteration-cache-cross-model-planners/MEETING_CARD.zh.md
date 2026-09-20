# 组会卡片：Iteration cache 的跨模型验证

## 一句话

在 planner 的重复调用中，只缓存不随 candidate action 改变的 observation/goal
prefix；action-conditioned encoder、predictor、objective、ranking 和 planner
update 保持原路径。结果取决于 planner 与 model 的重复计算边界。

## 四个 case 的结果

| 组合 | 结论 | 关键数字 |
|---|---|---:|
| DINO-WM Wall + CEM | decision PASS，system FAIL | `14.35%` reduction；gate `20%` |
| DINO-WM Wall + GD | **PASS** | bitwise exact；`10.80%` reduction |
| DINO-WM PushT + CEM | decision PASS，system FAIL | `4.67%` reduction；gate `10%` |
| LeWM PushT + CEM | **PASS** | bitwise exact；`29.41%` reduction |

## 最重要的解释

这不是一个“只要加 cache 就必然提速”的结论。Cache 能否形成有意义收益，取决于：

1. 同一个 fixed prefix 是否在 planner loop 中重复计算；
2. cache 是否能在不切断 action-conditioned suffix 的情况下复用；
3. setup/goal/preprocessing 开销是否吃掉了省下来的编码时间。

DINO-WM Wall GD 和 LeWM PushT CEM 说明机制确实可迁移；DINO-WM 两个 CEM case
则提醒我们，exact decision equivalence 与 system-level latency gain 是两个独立
问题。

## 组会上不要说过头

- 不把 DINO-WM PushT 的 `4.6681%` 写成通过；
- 不把 DINO-WM Wall CEM 的 `14.3485%` 写成通过 20% gate；
- 不把早期 PushT transfer job `23986275.pbs101` 的 FAIL 改写成成功；
- 不把这些 planner-call benchmark 当成 closed-loop task success。

## 入口

- 详细矩阵：`README.zh.md`
- 结果与证据边界：`RESULT.zh.md`
- 明天继续实验：`CONTINUE_EXPERIMENTS.zh.md`
- 各组合的代码和 freeze：`cases/*/`
