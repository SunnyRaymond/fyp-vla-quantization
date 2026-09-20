# Iteration cache：综合结果与证据边界

## 机制

一次 planner call 内，observation/goal prefix 对 candidate action 不变，因此可
在 loop 外计算一次；candidate-dependent 的 action encoder、predictor rollout、
objective、ranking、elite/update 或 action-gradient suffix 继续走官方实现。

```text
baseline:  each iteration = fixed prefix encode + action-conditioned suffix
cached:    call start   = fixed prefix encode once
           each iteration = cached prefix + same suffix + same planner update
```

## 结果摘要

| Case | Decision gate | System gate | Latency | Memory |
|---|---|---|---:|---:|
| DINO-WM Wall CEM | PASS；10/10 screen observations | **FAIL** | `14.3485%`（要求 `20%`） | `1.00007` |
| DINO-WM Wall GD | PASS；bitwise exact | **PASS** | `10.7966%` | `1.00054` |
| DINO-WM PushT CEM | PASS；30/30 exact | **FAIL** | `4.6681%`（要求 `10%`） | `1.00470` |
| LeWM PushT CEM | PASS；30/30 exact | **PASS** | `29.4091%` | `0.97083` |

## 结论

最稳妥的 research claim 是：

> Planner-call-local fixed-prefix caching can preserve the action-conditioned
> planner computation exactly in the validated cases, while the realized
> latency gain is model/planner dependent.

当前 strongest positive evidence 是 DINO-WM Wall GD 与 LeWM PushT CEM。DINO-WM
Wall CEM 和 PushT CEM 的结果不是实现错误意义上的失败：它们保持了 decision
behavior，但省下的时间不足冻结 system gate。

早期 DINO-WM PushT transfer `23986275.pbs101` 还出现 30/30 非 exact 且 latency
`4.8150%` 的失败结果；它保留为 negative evidence，不应写成成功或与
`24054689.pbs101` 合并成一个 PASS。

## 证据边界

- benchmark 是固定 observations、seeds、checkpoint 和 planner settings；
- 不包含 environment interaction 或 closed-loop task success；
- 没有证明所有 JEPA-style WM、所有 planner 或所有 task 都有相同收益；
- 大型 source/runtime/checkpoint/dataset/plan_targets 留在外部 staging，bundle
  仅保留可继续运行所需代码、freeze、PBS、verifier 和小型结果。

逐 case 的 frozen protocol、runner、verifier、job handle 和结果卡在
`cases/dino-wall-cem/`、`cases/dino-wall-gd/`、`cases/dino-pusht-cem/` 和
`cases/lewm-pusht-cem/`。
