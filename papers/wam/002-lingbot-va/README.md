# 08. LingBot-VA 1.x

**Causal World Modeling for Robot Control**  
本地版本：**arXiv:2601.21998v2** · 31 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：核心 / 可用 joint WAM baseline

[本地 PDF](paper-arxiv-v2.pdf) · [arXiv 版本页](https://arxiv.org/abs/2601.21998v2) · [代码或官方项目入口](https://github.com/Robbyant/lingbot-va) · [返回总指南](../README.md)

## 背景与要解决的问题

Causal World Modeling for Robot Control，官方标记 RSS 2026。这是 QuantWAMs 已评估的模型之一，应与 #23 对读。

## 方法与核心机制

Autoregressive diffusion、video/action MoT、ground-truth observation re-grounding 与 asynchronous inference 共同实现长时闭环。注意视频未来与机器人 action 的 conceptual distinction，即使二者在同一序列或架构中交互。

## 结果：带着条件读证据

读模型 attention/temporal ordering、闭环图和 PDF pp.13–15 的结果。当前仓库明确提供 base、RoboTwin post-trained、LIBERO-Long post-trained 模型，以及 RoboTwin/LIBERO evaluation 路线。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

当前公开 LIBERO checkpoint 的名称是 Long；不能直接当作四个 LIBERO suites 都有对应权重。异步机制、KV cache 和动作执行 overlap 会改变量化的有效收益。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 哪些 token 能看见哪些未来 token？有没有 action leakage？
2. 每轮新 observation 替换了哪些 predicted latents？
3. KV cache 的 lifetime 与 precision 是什么？
4. QuantWAMs 的 LingBot-VA 结果究竟覆盖哪些 LIBERO tasks？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

