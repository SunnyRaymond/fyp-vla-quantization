# 15. TD-MPC2

**TD-MPC2: Scalable, Robust World Models for Continuous Control**  
本地版本：**arXiv:2310.16828v2** · 31 页 · 来源核验：2026-09-08 · 阅读状态：`unread`  
分类：基础支线 / 连续控制 planning

[本地 PDF](paper-arxiv-v2.pdf) · [arXiv 版本页](https://arxiv.org/abs/2310.16828v2) · [代码或官方项目入口](https://github.com/nicklashansen/tdmpc2) · [返回总指南](../README.md)

## 背景与要解决的问题

ICLR 2024 的强 model-based RL baseline。用于理解 decoder-free dynamics、reward/value 和 online planning 的结合。

## 方法与核心机制

在 latent space 做短 horizon trajectory optimization，以 learned reward 和 terminal value 评估候选；结合 policy prior 和稳定训练设计扩展到多任务。

## 结果：带着条件读证据

论文报告 104 online RL tasks、317M 多任务 agent；看单任务结果与 scaling，再读具体 task domains、observations、train/eval budgets。官方代码有 pretrained models、data、results。

以上为原文/官方公开材料的报告，本次未运行训练、量化或完整 benchmark。页码按本地 PDF 的物理页编号，可能不同于印刷页码。

## 边界与对量化选题的意义

它依赖 reward/online learning 的设置与 reward-free LeWM 不等价。SimNorm、critic、policy prior 和 dynamics 的数值敏感性不同，不应只量化 encoder 后称为完整 TD-MPC2 量化。

## 分时阅读路线

- **20 分钟定位**：abstract → method overview → 本页列出的结果与限制，写出模型实际输入和输出。
- **75 分钟理解**：方法 25 分钟，实验协议与结果 25 分钟，appendix/官方入口 15 分钟，填写 Meeting Card 10 分钟。
- **2 小时深入**：沿一个具体 failure mode 核对模型、data split、precision、planner/inference budget，记录哪些内容只有论文、哪些已有代码、哪些还缺证据。

## Reading Questions（读完自己回答）

1. 模型选择候选轨迹依靠 reward、value 还是 goal distance？
2. Critic quantization 会怎样放大 ranking error？
3. 用相同 environment steps 与相同 GPU hours 得到的比较是否不同？
4. 哪种观测设置与你的 latent WM 实验相容？

## Meeting Card（留空）

- 我要解决的具体问题：
- 模型的训练 / 测试输入和输出：
- 最关键机制与证据位置：
- 数据、checkpoint、benchmark 与 precision 条件：
- 一个重要局限或未复现点：
- 与我的量化 idea 重合的部分：
- 我准备验证的不同假设：
- 想向导师/师兄确认的问题：

