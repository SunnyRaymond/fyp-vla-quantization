# 07. DreamZero

**World Action Models are Zero-shot Policies**  
本地版本：**arXiv:2602.15922v1** · 36 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：核心 / 大规模 joint WAM

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 版本页](https://arxiv.org/abs/2602.15922v1) · [代码或官方项目入口](https://github.com/dreamzero0/dreamzero) · [返回总指南](../../README.md)

## 背景与要解决的问题

论文标题是 World Action Models are Zero-shot Policies。用于理解 video-action 联合建模和跨任务、跨 embodiment generalization；不是 Dreamer 系列。

## 方法与核心机制

在 pretrained video diffusion backbone 上联合生成 future video 与 continuous actions，使用 autoregressive generation，并配合系统优化实现闭环控制。

## 结果：带着条件读证据

读模型结构、zero-shot/few-shot 划分和 real robot protocol。PDF pp.9–10、16–17 的结果与 pp.23 附近的效率说明应分开读；作者报告 14B 模型与 7Hz 控制方案。当前官方 README 的 server 路线要求多 GPU，测试 GB200/H100，并给出不同 inference timing。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

论文中的高频结果与当前 README 的 server timing 不应直接互换；7Hz 不是任何 GPU 上的单次完整推理延迟倒数。当前不建议把它作为第一项低成本部署实验。其效率方案包含多种因素，不能把整体 speedup 归给量化。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. Video/action 是 jointly denoise 还是先后生成？
2. Zero-shot 的任务、场景和 embodiment 各自怎么定义？
3. 本文 PTQ、step reduction、caching、parallelism 各贡献多少？
4. 完整 robot cycle 的关键路径是什么？
5. 如果只量化 video branch，action quality 会如何变化？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
