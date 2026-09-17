# 24. Where Bits Matter in World Model Planning

**Where Bits Matter in World Model Planning: A Paired Mixed-Bit Study for Efficient Spatial Reasoning**  
本地版本：**arXiv:2602.11882v1** · 12 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：PTQ companion / paired mixed-bit study

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 版本页](https://arxiv.org/abs/2602.11882v1) · [代码或官方项目入口](https://github.com/suraj-ranganath/DINO-MBQuant) · [返回总指南](../README.md)

## 背景与要解决的问题

2026-02 workshop submission，以 DINO-WM / Wall 检验 bits 分配位置与 planner budget 的关系；是 mixed-precision novelty 必须考虑的相邻工作。

## 方法与核心机制

相同 checkpoint、paired goals，比较 uniform、mixed、asymmetric、layerwise weight-only quantization；在两种 planning budgets 下验证。

## 结果：带着条件读证据

PDF p.2 写明 quantize/dequantize nn.Linear weights；p.3–4 与 appendix 的 paired comparisons 包含有限样本与预算依赖。原文明确部分 INT4 差异不具统计确定性。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

只覆盖 Wall 和 weight-only；M4 MacBook Pro 实验及 dequantized inference 不等于 NVIDIA packed INT4 acceleration。某些 mixed/uniform 对比仍有 size confounding，不能直接宣称普适最优 bits allocation。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 两个 planner budgets 下 effect 的方向是否一致？
2. 总模型 bytes 是否真正匹配？
3. 哪些结论只是 directional evidence？
4. 把 Wall 扩展到 PushT/OGBench 后假设如何被证伪？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

