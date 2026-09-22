# Meeting Card｜Compiled World Model

- **问题**：context compiler 能否把 LeWM PushT 的 state computation 移出 candidate loop，并不被 dense-prefix baseline 支配？
- **设计**：B1 dense prefix；B2 global basis；B3 context basis，`r=32/96/192`；B4 exact quadratic contraction。512 train contexts，development 选 rank，独立 `valid[592:600]` final test。
- **结果**：selected `r=192`。B3 final Spearman/top-30/relMSE=`0.0534/0.0833/0.1389`；absolute quality FAIL。
- **机制**：B3−B2 episode-median Spearman `+0.0596`，top-30 `−0.00833`，joint improvement `4/8`；context-dependence gate FAIL。
- **效率**：30-query p50 B1/B3/B4=`14.349/14.536/16.821 ms`；B3 未快于 B1，B4 未快于 B3。
- **正信号**：B4 与该训练模型的 official terminal cost 数值一致，max abs `5.34e-5`，argmin exact match。
- **决策**：**predictor-level NO-GO / STOP**。不进入 official CEM、closed-loop 或 joint encoder training。
- **边界**：B4/teacher 的 `35.89×` 只是低质量 surrogate 的 predictor boundary，不能作为有效 replacement speedup。
