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

## LeWM transfer 更新

同一个 horizon-weighted recurrent mechanism 已在 official LeWM compact latent 上完成
predictor-level 测试（job `24564619.pbs101`）：

| 项目 | 结果 |
|---|---:|
| student / teacher latency | `1.85293 / 23.5295 ms` |
| predictor latency reduction | `92.1251%`（约 `12.70×`） |
| Spearman median / minimum | `0.415374 / -0.404356` |
| top-30 median / minimum | `0.266667 / 0` |
| relative latent MSE median | `0.012116` |
| predictor feasibility | **NO-GO** |

这说明机制的速度优势能迁移到 LeWM，但 ranking fidelity 没有迁移成功；高 latent cosine
仍不足以支持 replacement。按实验边界未运行 official LeWM CEM。详见
[lewm-transfer/RESULT_LEWM_RECURRENT_STUDENT.zh.md](lewm-transfer/RESULT_LEWM_RECURRENT_STUDENT.zh.md)。

## Closed-loop 更新

已训练的 horizon-weighted `step1500` student 已接入 official DINO-WM PushT CEM + MPC + environment evaluator。冻结的 8-case、最多 12 MPC rounds paired pilot（job `24544733.pbs101`）得到：

| arm | success |
|---|---:|
| official DINO-WM teacher | `8/8` |
| horizon-weighted recurrent student | `2/8` |

paired table 为 `both success=2`、`teacher only=6`、`student only=0`。Progression gate 要求 student 至少 `7/8` 且最多落后 teacher 1 例，结果为 **FAIL**，因此停止扩到 50 cases。

这不是 official 50-case reproduction；它是 bounded exploratory no-go signal。Student 总 planner walltime 更长是因为 6 个失败例跑满 12 rounds，不能解释为 predictor forward 变慢。

## CEM trace diagnosis

对上述 6 个 teacher-only success cases，在相同 initial observation 上用 paired innovations 重跑 30-step CEM trace：

| metric（case-level median） | iter 1 | iter 30 |
|---|---:|---:|
| teacher-pool Spearman | `0.975736` | `0.421793` |
| teacher-pool top-30 overlap | `0.766667` | `0.083333` |
| student-pool teacher-shadow Spearman | `0.975736` | `0.238582` |
| student-pool shadow top-30 overlap | `0.766667` | `0.066667` |
| first-action RMS drift | `0.135645` | `0.731702` |

最可能的问题是 student 没有稳定保留 CEM 所需的 local elite ordering；第一次 top-30 selection 已分叉，之后 proposal feedback 将误差放大。这个结果仍是 fixed-observation mechanism diagnosis，不是新的 success-rate evidence，也没有定位到某个 architecture component。

## 关键材料

- 正式 horizon 结果：[reports/RESULT_HORIZON_WEIGHTED.zh.md](reports/RESULT_HORIZON_WEIGHTED.zh.md)
- recurrent 基座结果：[reports/RESULT_RECURRENT_STUDENT.zh.md](reports/RESULT_RECURRENT_STUDENT.zh.md)
- PushT closed-loop 结果：[reports/RESULT_CLOSED_LOOP_PUSHT.zh.md](reports/RESULT_CLOSED_LOOP_PUSHT.zh.md)
- CEM trace diagnosis：[reports/RESULT_CEM_TRACE_DIAGNOSIS.zh.md](reports/RESULT_CEM_TRACE_DIAGNOSIS.zh.md)
- 后续 continuation：[CONTINUE_EXPERIMENTS.zh.md](CONTINUE_EXPERIMENTS.zh.md)
