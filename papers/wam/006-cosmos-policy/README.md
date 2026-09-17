# 12. Cosmos Policy

**Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning**  
本地版本：**arXiv:2601.16163v1** · 22 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：核心 / 视频模型到机器人 policy

[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 版本页](https://arxiv.org/abs/2601.16163v1) · [代码或官方项目入口](https://github.com/NVlabs/cosmos-policy) · [返回总指南](../README.md)

## 背景与要解决的问题

将 Cosmos-Predict2 改为 action/state/value generation 的代表工作，ICLR 2026 有公开论文记录。对已有 LIBERO 流程的 FYP，是值得优先评估可行性的另一条路线。

## 方法与核心机制

将 robot actions、future images、values 编码为 latent frames，在同一 diffusion process 中生成；单阶段 post-training 后既可作 direct policy，也可用 value/world predictions 做 best-of-N planning。

## 结果：带着条件读证据

读方法的 latent frame allocation，区分 base policy 与 planning enhancement，再看 LIBERO、RoboCasa、ALOHA 实验。作者报告 LIBERO 98.5%、RoboCasa 67.1%，仅对应其 paper task/protocol。官方 README 明确列 code、weights、training data 和各 benchmark 指南。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

Base policy 与 planning 模式不是同一成本。README 目前给出的 base inference 显存约 6.8GB（LIBERO）和 8.9GB（RoboCasa），这些是作者条件下的说明，不是本地验证；训练预算明显更大。本文使用 Cosmos-Predict2，不是 Cosmos 3，RoboCasa paper subset 也不是自动等于 RoboCasa365 全套。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. Action/value latent frames 是否共享尺度和 quantizer？
2. Base policy 与 model-based planning 各调用模型几次？
3. Value ranking 的误差是否比 video reconstruction 更影响成功？
4. Quantization 后 best-of-N 的 N 增加是否仍值得？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

