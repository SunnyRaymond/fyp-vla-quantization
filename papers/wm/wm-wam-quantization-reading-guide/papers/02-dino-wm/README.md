# 02. DINO-WM

**DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning**  
本地版本：**arXiv:2411.04983v2** · 21 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：优先核心 / 已有量化 baseline

[本地 PDF](paper-arxiv-v2.pdf) · [arXiv 版本页](https://arxiv.org/abs/2411.04983v2) · [代码或官方项目入口](https://github.com/gaoyuezhou/dino_wm) · [返回总指南](../../README.md)

## 背景与要解决的问题

另一篇 Yann LeCun 合著工作，也是 #03 QuantWM 与 #24 mixed-bit study 的直接基础。想与已有 WM PTQ 结果建立可比关系，应优先熟悉它。

## 方法与核心机制

冻结 pretrained DINOv2，保留 spatial patch features，训练 action-conditioned dynamics predictor 预测未来 feature maps。测试时对 candidate action sequences 做 latent rollout，以目标图像特征为规划目标。它不要求把每个 imagined state 解码为 RGB 来做 planning。

## 结果：带着条件读证据

读方法部分的 patch-level features、temporal context、action/proprioception conditioning，再读 PDF p.6 的主表、p.8 的比较与 appendix 的 planning 配置。原论文涉及六类环境；官方 README 当前明确给出 PointMaze、PushT、Wall 的 pretrained checkpoints 和 planning configs。原论文覆盖范围不等于全部 checkpoint 都一键可用。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

Zero-shot 指训练 world model 后对目标执行 planning，不代表完全没有环境数据。不同后续工作会使用 DINO-WM-no-proprioception 或不同视觉 backbone；不能把它们当作相同 baseline。PointMaze、Wall、TwoRoom 的 dataset、goal construction 也不可互换。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 保留 patch tokens 相对 CLS 有什么 planning 收益和计算代价？
2. 哪些 ablation 使用了 proprioception？
3. Encoder 对当前观测和目标观测的误差如何同时进入 cost？
4. QuantWM 使用的是哪一份 checkpoint 和 planning 配置？
5. CEM iterations、rollout horizon、executed action chunk 是三个不同变量吗？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：
