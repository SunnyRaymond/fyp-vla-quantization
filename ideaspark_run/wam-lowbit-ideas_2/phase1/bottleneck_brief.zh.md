# Phase 1 bottleneck brief

**State: proceed.** 用户方向足够具体，31 篇检索记录中有 20 篇属于 WAM、低比特 diffusion 或 robot-policy quantization 的直接/可迁移证据。Overall closest anchor 为 Q-WAM（arxiv:2609.33269v1）；另对照 SteerQuant、QuantWAMs 与 PreDE。

1. **执行对象错位（共享前提待检验）**：现有分数以完整 action chunk 或固定 execution condition 为准；候选交付应在固定 controller 和 W4A4/W4A8 下提出并检验能保留已执行 command 行为的方法，同时不预定内部机制。
2. **多点误差交互**：Q-WAM 将交叉项置零，SteerQuant 用 full-precision 输入逐区域评分；候选交付应以联合扰动证据检验 W4A4/W4A8 方法对多处残差合成误差的判断，不预定估计或调节机制。
3. **闭环历史迁移**：FP/reference replay 与固定 observation log 给出的 proxy，尚未证明能在量化动作改变后续 observation history 时保持校准；候选交付应在固定 evaluator 下证明其质量判断可迁移到候选诱发的多轮 history。

三个 gap 都保持在 failure/assumption 层，不指定算法、损失或补偿位置。四篇 WAM 论文全文已读；检索窗外的 MPC/quantized-control 节点仅作 lineage awareness，不作为新颖性证据。用户未指定硬件或预算，因此后续只可给小规模 pilot 建议，不假设 150 GPU-days。
