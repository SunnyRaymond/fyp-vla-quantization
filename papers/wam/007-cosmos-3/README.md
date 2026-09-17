# 13. Cosmos 3

**Cosmos 3: Omnimodal World Models for Physical AI**  
本地版本：**arXiv:2606.02800v4** · 139 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：前沿大模型 / 平台扩展

[本地 PDF](paper-arxiv-v4.pdf) · [arXiv 版本页](https://arxiv.org/abs/2606.02800v4) · [代码或官方项目入口](https://github.com/NVIDIA/cosmos) · [返回总指南](../README.md)

## 背景与要解决的问题

截至本次检索，NVIDIA 已发布 Cosmos 3，不能再把 Cosmos Predict 2.5 说成最新家族。本地是 139 页的 arXiv v4 technical report，适合按 robotics/efficiency 问题跳读。

## 方法与核心机制

Unified Mixture-of-Transformers 处理 language、image、video、audio、action，兼顾 reasoning、generation、forward/inverse dynamics 和 robot policy。

## 结果：带着条件读证据

先看 report 的 architecture 与任务地图，再定位 robot policy、RoboArena 及 efficiency。官方仓库与 HF collection 公开模型和工作流；README 的不同 model roles、sizes 和 supported backends 需分别核对。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

官方“最佳开放模型”的判断限于报告写作时、相应榜单与任务，不能当作本日所有 WM benchmark 的统一冠军。代码/模型采用 OpenMDW-1.1 条款；当前部分 FP8/NVFP4 表项仍标 coming soon。特定 policy checkpoint 能否用于你的 simulator 需要单独验证。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 与 Cosmos Policy 的训练目标、action representation 有何变化？
2. WAM 功能使用哪个具体 checkpoint，而不是哪个总品牌？
3. 总 parameters、active parameters、resident memory 各是多少？
4. 量化是否需要跨 modality 分别校准？
5. 公开 benchmark 是否覆盖你的 closed-loop task？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

