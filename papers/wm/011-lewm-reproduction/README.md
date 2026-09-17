# 18. LeWorldModel 独立复现与协议检查

**The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom**  
本地版本：**arXiv:2608.10145v1** · 26 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：必读 companion / 不计入 SOTA

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 版本页](https://arxiv.org/abs/2608.10145v1) · [代码或官方项目入口](https://github.com/joyjeet-singh/tinylab) · [返回总指南](../README.md)

## 背景与要解决的问题

2026-08 的独立 TwoRoom reproduction。将它与 #01 配套阅读，是为了避免把 evaluation implementation 差异误判为 quantization damage。

## 方法与核心机制

独立实现、用作者 checkpoint 在统一 episodes 上比较，检查 frameskip action gathering、action encoder dimensions、pixel/action normalization、goal construction 和 evaluation budget。

## 结果：带着条件读证据

PDF pp.2–3 明确限定 scope：TwoRoom、单 seed，比较样本量有限。文中显示不同公开协议和 goal construction 可显著改变同一 checkpoint 的表现，并说明部分发现不能推广为 method-level properties。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

它的 replication claims 尚未由本次运行独立复核，也不评价 LeWM 其他三种环境。不要把它概括成原工作“错误/无效”。应把具体 configuration differences 列为复现前核对项。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 当前官方 repo 是否已经修正文中指出的差异？
2. Frameskip 应采样单 action 还是拼接 action block？
3. 哪种 goal construction 是目标论文真实采用的？
4. 相同 paired episodes 上 FP 与 quantized 的差异是否仍出现？
5. 文中 BatchNorm 问题影响谁的 checkpoint，是否明确排除了作者原权重？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

