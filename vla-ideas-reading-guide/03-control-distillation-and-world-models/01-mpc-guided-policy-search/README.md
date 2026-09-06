# 01. MPC-Guided Policy Search

**Learning Deep Control Policies for Autonomous Aerial Vehicles with MPC-Guided Policy Search**  
Zhang, Tianhao, Kahn, Gregory, Levine, Sergey, Abbeel, Pieter  
**历史基础 · 本次补充** · ICRA 2016  
本地版本：**arXiv:1509.06791v2，8 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/1509.06791v2)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

MPC 计算昂贵且依赖 state estimation，目标是训练一个由 onboard observations 直接输出控制的 neural policy。

## Method / Key Innovation

训练时用完整 state 的 MPC 生成指导数据，通过 guided policy search 学 partial-observation policy；推理时直接运行 neural network。

## Results：重点核对的证据

Sections III–V、Algorithms 1–2（pp.3–6）；simulated quadrotor obstacle avoidance，无 test-time explicit state estimation。作者项目页对应 ICRA 2016。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

已实现“训练用 MPC，推理用 neural policy”的机器人先例；不能把这一结构本身写成方向三的首次贡献。原文主要是 simulation，不是通用 VLA 或 safety theorem。

## 与师兄方向的关系

为方向三提供最清楚的历史起点，也帮助区分 teacher imitation 与通过 model 直接求 policy gradient。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section III GPS → Section IV MPC-guided data generation → Section V。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. teacher 和 student 在训练时分别观察什么？
2. 为什么数据采集用 MPC，而不是直接 rollout 当前 student？
3. policy matching 如何改变 teacher 的 objective？
4. 换成 contact-rich VLA，最大的 distribution shift 来自哪里？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
