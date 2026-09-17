# 22. Physics-IQ

**Do generative video models understand physical principles?**  
本地版本：**arXiv:2501.09038v3** · 13 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：Benchmark / generative physical prediction

[本地 PDF](paper-arxiv-v3.pdf) · [arXiv 版本页](https://arxiv.org/abs/2501.09038v3) · [代码或官方项目入口](https://github.com/google-deepmind/physics-IQ-benchmark) · [返回总指南](../README.md)

## 背景与要解决的问题

论文标题 Do generative video models understand physical principles?，其核心是视频视觉真实感不等于物理规律掌握。

## 方法与核心机制

以需要特定物理原理的视频 continuation 测试生成模型，覆盖多种真实物理现象；用可评估的预测结果检验 physical understanding。

## 结果：带着条件读证据

读 paper 中测试构造和指标，结合官方 benchmark code。它适合检测低精度是否破坏物理预测，而不是只比较美学。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

生成式 continuation 与 IntPhys2 的 violation-of-expectation 任务不同。原文模型排名是当时的结果；对新的模型应重新运行匹配版本协议。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 正确结果允许多少 stochastic variation？
2. 模型是物理错误还是时间/视角对齐错误？
3. 如何证明量化损失集中在物理结构而非一般图像质量？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

