# 14. DreamerV3

**Mastering Diverse Domains through World Models**  
本地版本：**arXiv:2301.04104v2** · 40 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：基础支线 / Model-based RL

[本地 PDF](paper-arxiv-v2.pdf) · [arXiv 版本页](https://arxiv.org/abs/2301.04104v2) · [代码或官方项目入口](https://github.com/danijar/dreamerv3) · [返回总指南](../../README.md)

## 背景与要解决的问题

理解 RL world model 的基础。它使用 learned dynamics 在 imagination 中学习 actor/critic，与 LeWM 的 test-time CEM goal planning 不同。

## 方法与核心机制

学习 latent dynamics、reward/continuation，再在 imagined trajectories 上优化行为；通过归一化、loss balancing 等机制提高跨任务训练稳定性。

## 结果：带着条件读证据

本地 arXiv v2 标题 Mastering Diverse Domains through World Models；Nature 2025 发表标题为 Mastering diverse control tasks through world models。阅读 arXiv 的 model/actor training 和 benchmark suite，不把其数字冒充 Nature 版本。官方 code 为 danijar/dreamerv3。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

本库保存的是 arXiv 作者稿，不是 Nature 最终 PDF。DreamerV3 是成熟基础，不声明为截至今天所有 RL benchmark 的最新 SOTA。量化对象若只是 imagined training world model，不一定减少部署 actor 的关键路径。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. World model 在训练和部署阶段各承担什么？
2. Discrete latent categories 与 low-bit weight quantization 是一回事吗？
3. 哪些任务用 pixels，哪些用 low-dimensional states？
4. 要研究量化，是 inference-only PTQ 还是 quantized model-based training？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
