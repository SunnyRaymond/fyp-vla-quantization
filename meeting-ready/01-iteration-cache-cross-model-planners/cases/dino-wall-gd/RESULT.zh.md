# DINO-WM Wall GD：结果卡

| 项目 | 结果 |
|---|---:|
| job | `23926079.pbs101` |
| decision equivalence | PASS；bitwise exact |
| paired median full-plan reduction | `10.7966%` |
| peak-memory ratio | `1.00054`，PASS |
| progression gate | **PASS** |
| benchmark boundary | 2 observations，direct GD planner calls |

结论：在 Wall GDPlanner 上，复用 planner-call 内固定 observation prefix 同时保持
action-gradient suffix，达到了冻结的数值、latency 和 memory gate。这是 planner
portability evidence，不是跨模型或 closed-loop 证明。
