# Iteration cache：跨模型、任务与 planner 验证 bundle

这是 iteration/shared-prefix cache idea 的独立材料包。四个 case 的代码、freeze、
PBS、verifier、中文结果卡和关键小型 artifacts 都在这里；继续实验时不需要回到
旧的 `experiment/idea-validation/...` 目录拼文件。

## 先看什么

1. 组会总览：`MEETING_CARD.zh.md`
2. 四个 case 结果卡：各 case 下的 `RESULT.zh.md`
3. 详细继续方式：`CONTINUE_EXPERIMENTS.zh.md`
4. 机制与 full-planner boundary：`RESULT.zh.md`

## 验证矩阵

| Case | Model / task | Planner | Job | Decision | Latency / memory | 总结 |
|---|---|---|---|---|---|---|
| `cases/dino-wall-cem` | DINO-WM Wall | CEM | `23435258.pbs101` | PASS；10/10 | `14.3485%`；ratio `1.00007` | system gate FAIL（要求 20%） |
| `cases/dino-wall-gd` | DINO-WM Wall | GD | `23926079.pbs101` | PASS；bitwise exact | `10.7966%`；ratio `1.00054` | progression PASS |
| `cases/dino-pusht-cem` | DINO-WM PushT | CEM | `24054689.pbs101` | PASS；30/30 exact | `4.6681%`；ratio `1.00470` | system gate FAIL（要求 10%） |
| `cases/lewm-pusht-cem` | LeWorldModel PushT | CEM | `24382364.pbs101` + `24389764.pbs101` recovery | PASS；30/30 exact | `29.4091%`；ratio `0.97083` | system gate PASS |

这里的 PASS/FAIL 是分 gate 记录：DINO-WM PushT 的 exactness PASS 不等于 latency
PASS；DINO-WM Wall CEM 的 decision PASS 也不等于 system gate PASS。

## 如何理解整体结论

目前最强的证据是：

- DINO-WM Wall GD：固定 observation prefix 的跨 planner-call cache 已通过完整
  数值、latency 和 memory gate；
- LeWM PushT CEM：固定 initial/goal embedding 的 iteration cache 已通过 exactness
  和 full-planner gate，并达到 `29.4091%` median reduction；
- DINO-WM Wall CEM 与 PushT CEM：行为可保持，但当前收益未达到各自冻结的
  system-level gate。

所以可讲的故事是“cache boundary 在不同 action-conditioned planner 中具有可复用
机制”，而不是“所有 model/planner 都已经获得同样的 speedup”。

## 目录

```text
01-iteration-cache-cross-model-planners/
├── README.zh.md
├── MEETING_CARD.zh.md
├── RESULT.zh.md
├── CONTINUE_EXPERIMENTS.zh.md
├── common/
│   └── cache_core.py
└── cases/
    ├── dino-wall-cem/
    ├── dino-wall-gd/
    ├── dino-pusht-cem/
    └── lewm-pusht-cem/
```

不复制 checkpoint、dataset、runtime、plan_targets 或大型 summary。它们的外部
位置、准备约束和默认变量写在各 case README/PBS 中。
