# 组会卡片：Horizon-weighted Recurrent Student

## 30 秒版本

我们的目标是把 DINO-WM 的 action-conditioned predictor 从昂贵的 teacher rollout 编译成轻量 student。当前最有效的结构不是继续增大 MLP，而是让 student recurrently consume 自己预测的 latent，并把 output feedback 到下一个 action step。

在这个 recurrent student 上，我们只改变训练 loss：五个预测 horizon 的误差使用

```text
[1/3, 2/3, 1, 4/3, 5/3]
```

让更远的预测受到更大梯度。结果是 ranking 方向稳定改善，但幅度没有达到预先冻结的 gate。

## 实验和结果

| 项目 | 结果 |
|---|---:|
| held-out paired blocks | 16 |
| recurrent architecture：median ΔSpearman | `+0.037226` |
| recurrent architecture：median Δtop-30 | `+0.066667` |
| horizon weighting：median ΔSpearman | `+0.004083` |
| horizon weighting：median Δtop-30 | `+0.033333` |
| horizon-weighted absolute Spearman median/min | `0.970596 / 0.894798` |
| horizon-weighted absolute top-30 median/min | `0.833333 / 0.600000` |
| horizon-weighted logged MSE ratio | `0.927539` |
| student predictor latency | `10.719 ms` |
| teacher → student reduction | `99.6936%` |
| recurrent student parameters | `732,554` |

## 应该怎样解释

1. 这是 predictor-level result，不是 end-to-end control result。
2. Student 已经足够快，且没有明显 latent fidelity、causality 或训练稳定性问题。
3. 主要瓶颈不是简单的容量或训练时长，而是 student 是否保留了 planner candidate 之间的局部 action-to-cost sensitivity。
4. Horizon weighting 是稳定但很小的正向信号，不能单独作为 ranking solution。

## 结论措辞

推荐说：

> A shared recurrent latent-transition student gives a large predictor-only speedup and a consistent but sub-threshold ranking improvement. Mean-normalized horizon weighting preserves this behavior and slightly improves the paired ranking trend, but does not pass the frozen replacement gate.

不要说：

- “已经超过 teacher”；
- “可以安全替换 teacher”；
- “CEM 或 closed-loop success 已改善”；
- “结果已经证明适用于 LeWM/Fast-LeWM 或所有 JEPA-style WM”。

## 关键材料

- 正式 horizon 结果：[reports/RESULT_HORIZON_WEIGHTED.zh.md](reports/RESULT_HORIZON_WEIGHTED.zh.md)
- recurrent 基座结果：[reports/RESULT_RECURRENT_STUDENT.zh.md](reports/RESULT_RECURRENT_STUDENT.zh.md)
- 后续 continuation：[CONTINUE_EXPERIMENTS.zh.md](CONTINUE_EXPERIMENTS.zh.md)
