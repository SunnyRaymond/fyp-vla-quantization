# Null-input columns：root disposition

2026-09-13。保留独立审查，最终记为 **insufficiently_differentiated / engineering-only，GPU0**。这不是经验 no-go，也不是“已找到 exact prior”的断言。

静态 zero-input columns 不应支配 active-column 的量化范围，这个命题成立时是合理的实现优化；当前候选未提出超出 dead-input removal 与常规 row-wise quantization 的研究问题。Deep Compression / Network Slimming 是相关组合和 pruning 先例，不能单独证明它们已经研究本案的 exact mechanism。

历史 metadata 的 nominal config6 与实际8维 state需按真实 processor 路径区分；它不是本次科学失败的证据。因为此候选在研究切面 gate 已停止，没有为它加载模型、检查 zero columns 或追加 action MSE。若将来已有独立工程任务需要处理 inactive columns，可在那个任务内做对应自检；本轮不自动重开。
