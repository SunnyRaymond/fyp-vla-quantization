# Continuous B32 scaling + clipping80：结果与下一步

结论：当前 continuous B32 配方没有超过 clipping-only，不建议直接扩展 per-channel。接下来优先检验 scale 的优化方式；这不等于否定所有 continuous scaling 方法。

## 原80个 held-out queries

原40 contexts ×2 seeds，每个 suite 10 contexts；先在 case 内平均 seeds，再对四 suites 等权。

| 配置 | First10 motor RMSE ↓ | Future latent NMSE ↓ |
|---|---:|---:|
| 原 baseline | 0.044296774 | 0.060877371 |
| C80 clip_only80 | 0.043848437 | 0.054848124 |
| 离散 B32 + clip80 | 0.043986743 | 0.054151555 |
| 连续 B32 + clip80 | 0.044588993 | 0.060690613 |

- vs baseline：motor +0.6597%，latent −0.3068%。
- vs clip_only：motor +1.6889%，latent +10.6521%。
- vs 离散 B32：motor +1.3692%，latent +12.0755%。
- 相对 clip_only，case 内平均 seeds 后，motor 18/40 cases 改善、22/40变差；latent 15/40改善、25/40变差。80 queries 不被当作80个独立 contexts。

| Suite：连续版本 vs clip_only | First10 motor 变化 | Future latent 变化 |
|---|---:|---:|
| libero_spatial | +1.5554% | +7.9211% |
| libero_object | +0.9675% | +7.5861% |
| libero_goal | +1.6987% | +16.3871% |
| libero_10 | +2.4859% | +10.9004% |

因此结果并非只由一个 suite 拉差。late22、full32 和 gripper 未参与 loss 或推进门槛；即使 gripper RMSE 改善，也不能替代当前的两项主目标。

## Calibration 结果：不只是 held-out 过拟合问题

| 配置 | First10 motor RMSE ↓ | Future latent NMSE ↓ |
|---|---:|---:|
| 原 baseline | 0.044146961 | 0.054735372 |
| C80 clip_only80 | 0.041858108 | 0.047223291 |
| 离散 B32 + clip80 | 0.043173617 | 0.048167793 |
| 连续 B32 + clip80 | 0.043243457 | 0.048459245 |

连续版本 vs clip_only 的 calibration motor +3.3096%、latent +2.6173%，原 mixed loss 也高4.6889%。它并没有先在 calibration 上拟合得更好，因此不宜把相对 clip_only 的失败主要归结为过拟合或训练数据量不足。相对 baseline，calibration 收益在 held-out 上明显衰减，仍存在泛化问题；目前应同时考虑优化质量，不能只靠扩大数据解释。

## 连续 scaling 确实参与了计算

- 614个 Linear、56,466个 groups，612个 Linear 的 scales 发生变化。
- 42,677个 groups 的 D 偏离1超过1%；19,226个超过5%。最终 D 范围0.563690至1.422935，中位数1.000096。
- 相比离散版本只有13个 groups 非单位值，连续版本已解决“多数 group 完全不改变实际 forward”的问题。但更多调整没有转化为当前目标的收益。
- group/ratio 梯度全部连接且有限；612个模块有非零 group 梯度。单 group 的共同缩放存在理想量化的冗余，层内梯度中心化会使其梯度为0。

这轮结果不支持通过增加约32倍的 per-channel 自由度直接改善当前配方。

## 工程一致性与额外误差

D=1、ratio=1 的原 baseline 完整 action parity 为0。最终 uncached 训练 forward 与 cached 正式评估 forward 的完整 action/future query max_abs 均为0。GPU 小型 selfcheck 通过，覆盖 Smooth0.5/H128、D=1 与已有 clipping 的 forward/gradient 对照、fractional log2D 不取整、显式 compensated weight 对照、declared surrogate、zero/tail/bounds。

Codebooks、indices、row gains、原 integer exponents 均保持冻结。连续补偿另外保存 FP32 log2 scales，其数值 payload 为225,864 bytes（约220.57 KiB，未计键名及序列化开销）。

将实际 BF16 compensated weights 除以 D 还原至原坐标后，相对原 baseline decoded BF16 weights 的新增 rounding NMSE 为2.745702e-6，最大单层 NMSE 为3.601206e-6。这不是原 VQ 相对完整精度 weights 的总误差；小的 weight NMSE 也不保证最终 action 影响小。没有单独隔离这一 rounding 因素，因此不能把它完全排除。

上述检查未发现训练/评估 forward 不一致；它们验证实现和声明的 surrogate 一致，不证明 surrogate 是有效下降方向或已优化到最优。

## 为什么先不扩展 per-channel

当前 backward 沿用 LSQ surrogate，将 dynamic absmax 视为常数，并将 group 梯度在层内中心化。对于 clipping ratio，这个近似直接作用于量化 scale；对于 D，改变某个 group 会改变 Z/D，也可能改变该 token 的 absmax，从而联动所有 groups 的量化分辨率。层内减均值不等同于把这种联动的链式项归给实际决定 absmax 的 group。

这是一项明确尚未纳入的耦合，而非已经证明的失败主因。更有针对性的下一步是：保持 B32、C80、80 updates、原 loss、seeds 和 teacher，先检验加入该 absmax 耦合项的 scale surrogate，并用真实量化 forward 检查它是否提供有效下降方向。这样先改变优化方式，避免同时增加 granularity 或改变 loss。必要时再考虑限制 D 偏离1、少量层的 scaling 或更丰富的 calibration，但不同时把这些变量一起改掉。

只有 B32 的优化方式能稳定超过 clipping-only 后，再讨论更细 granularity。当前 retained reference 仍为 clip_only80；离散 B32 的 latent 更低，但 motor 略高，属于已有另一项取舍。未自动启动任何后续训练。

## 完成证据与范围

GPU `25856825.pbs101`：01:09:53；CPU aggregate `25857520.pbs101`：00:00:25。两者 scheduler F/exit0、artifact exit0、PHASE_COMPLETE、matching complete summary 均通过。完成80 updates、新方法 calibration80+原 held-out80，复用480条对照，共640条端点；新增 teacher=0。

Parity 用途的3个 queries 中，cached query 复用为一个 calibration endpoint，因此单独增加2个 parity queries，endpoint/parity 合计162次新 forward，另有80次 training forward/backward（含 checkpoint 内部重算）。GPU utilization/VRAM 每15秒记录；gpt-6-luna 每10分钟监控，现已结束。

仅单次训练、固定 cohort、decoded-BF16 VQ + fake A4 的数值比较。没有运行 per-channel、native kernel、closed-loop 或新的 development set；不能作为统计显著性或未见任务泛化结论。
