# 21. WorldScore

**WorldScore: A Unified Evaluation Benchmark for World Generation**  
本地版本：**arXiv:2504.00983v2** · 21 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：Benchmark / world generation

[本地 PDF](paper-arxiv-v2.pdf) · [arXiv 版本页](https://arxiv.org/abs/2504.00983v2) · [代码或官方项目入口](https://haoyi-duan.github.io/WorldScore/) · [返回总指南](../README.md)

## 背景与要解决的问题

把 world generation 分解为沿指定 camera trajectory 的 next-scene generation，用统一协议比较不同生成方法。

## 方法与核心机制

3,000 个测试样例，按 controllability、quality、dynamics 评估，包括多种场景和风格。

## 结果：带着条件读证据

读 metric definitions 与对应失败样例，再看 PDF p.7 的总结果；官方 project page 提供 dataset、evaluation code、leaderboard。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

WorldScore 主要面向 world generation，不是把 robotic actions 放进 physics simulator 后的成功率。LeWM 不生成 RGB，不能不加适配就用同一套图像生成分数。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 你的模型是否支持 benchmark 的 camera control interface？
2. 3D consistency 与 camera control 的下降能否分离？
3. 是否要增加 quantized-vs-FP 的 paired trajectory comparison？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

