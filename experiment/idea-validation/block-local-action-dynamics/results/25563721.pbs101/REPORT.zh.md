# global16 附加对照

该对照由主实验低秩 dev 的预设门槛触发，只增加 lowrank_coupled 的三个 seeds；同 data、loss、1500 steps 和最后 checkpoint。45 个 primary runs 保留。

| Seed | Dense h10 | Global4 h10 | Global16 h10 | Global16 params | B1: dense / g4 / g16 ms | B300: dense / g4 / g16 ms |
|---:|---:|---:|---:|---:|---|---|
| 1103 | 0.12336 | 0.19762 | 0.10258 | 36416 | 1.437 / 2.552 / 3.007 | 1.525 / 2.797 / 3.327 |
| 1102 | 0.12899 | 0.19382 | 0.10567 | 36416 | 1.211 / 2.567 / 2.526 | 1.287 / 2.784 / 2.771 |
| 1101 | 0.12647 | 0.19942 | 0.10791 | 36416 | 1.208 / 2.531 / 2.527 | 1.283 / 2.785 / 2.791 |

原始 dense/global4 checkpoints 在本 allocation 重新计时，且逐 episode h10 error 与主实验记录一致。延迟包含初始坐标变换、native rollout 和最后重构。

## 配对差分

| Seed | Global16 - Reference | Mean difference | Episode bootstrap 95% CI |
|---:|---|---:|---|
| 1103 | dense | -0.020774 | [-0.025185, -0.016266] |
| 1103 | learned_global4 | -0.095037 | [-0.102537, -0.088402] |
| 1103 | learned_local4 | -0.173804 | [-0.184834, -0.162781] |
| 1103 | learned_block | -0.156102 | [-0.166857, -0.146659] |
| 1102 | dense | -0.023320 | [-0.028319, -0.018558] |
| 1102 | learned_global4 | -0.088145 | [-0.095913, -0.080332] |
| 1102 | learned_local4 | -0.175120 | [-0.188312, -0.163272] |
| 1102 | learned_block | -0.146956 | [-0.157843, -0.138081] |
| 1101 | dense | -0.018556 | [-0.023920, -0.013022] |
| 1101 | learned_global4 | -0.091505 | [-0.098729, -0.084456] |
| 1101 | learned_local4 | -0.172441 | [-0.184156, -0.161778] |
| 1101 | learned_block | -0.177320 | [-0.190301, -0.165571] |

## 解释边界

global16 同时增加通信维数和参数量，没有训练 local16 等参对照，不能单独归因为通信带宽。其结果是附加探索，不替换 global4/local4 primary 对照，也不支持视觉任务或 CEM 结论。

训练时长、Torch allocator peak memory、Q overlap、同 allocation latency 和逐 seed 配对差分见 global16_comparison.json。

实验设计引用：Kassis et al. (2026). Scientific Agent Skills. https://doi.org/10.48550/arXiv.2609.00065
