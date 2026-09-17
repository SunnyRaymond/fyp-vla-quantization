# 19. WorldArena

**WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models**  
本地版本：**arXiv:2602.08971v2** · 22 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：Benchmark 核心 / functionality

[本地 PDF](paper-arxiv-v2.pdf) · [arXiv 版本页](https://arxiv.org/abs/2602.08971v2) · [代码或官方项目入口](https://github.com/tsinghua-fib-lab/WorldArena) · [返回总指南](../../README.md)

## 背景与要解决的问题

评估 embodied world models 的 perceptual quality 与 functional utility，是补足“视频好看但任务失败”这一缺口的重要 benchmark。

## 方法与核心机制

16 个视觉指标、data engine / policy evaluator / action planner 等功能角色及 human evaluation，并提供综合 EWMScore。

## 结果：带着条件读证据

读 benchmark 设计、数据划分与功能评测，不先看总榜。官方 repo 使用 RoboTwin 2.0 Clean-50 子集，说明了 train/test episode split；数据版本有更新记录。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

综合分依赖各指标的权重，不能替代任务级 success。作为 learned evaluator 的模型可能与 policy 共享偏差；需 physics simulator 或真实机器人 ground truth。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 评测 WM 作为哪种组件，其外部 IDM/policy 固定了吗？
2. 量化后的 generated data 是否真的训练出更好的 policy？
3. EWMScore 的聚合是否隐藏某种严重失败？
4. Train/calibration/test trajectories 如何隔离？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
