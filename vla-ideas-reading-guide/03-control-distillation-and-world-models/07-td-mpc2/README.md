# 07. TD-MPC2

**TD-MPC2: Scalable, Robust World Models for Continuous Control**  
Hansen, Nicklas, Su, Hao, Wang, Xiaolong  
**control-oriented world model 基础 · 本次补充** · ICLR 2024  
本地版本：**arXiv:2310.16828v2，31 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2310.16828v2)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

训练 world model 不必重建每个 pixel；控制需要的是对 transition、reward 和 value 有用的表示。

## Method / Key Innovation

学习 implicit、decoder-free latent world model，在 latent space 做 trajectory optimization，联合 temporal-difference learning 与 reward / representation objectives。

## Results：重点核对的证据

Section 3（pp.3–4）、Section 4（pp.5–8）、Table 1（p.7）；104 online RL tasks 与 multi-task scaling。ICLR 2024。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

这是已有 control-oriented latent modeling 的强先例，但 learned latent 未必等于可解释 pose / contact state；inference 仍有 planning，不是已完成 VLA control distillation。

## 与师兄方向的关系

方向三“world model 不预测图像”的基础已经存在。更细的新增点可能是 task-state sufficiency、uncertainty calibration、contact transitions 和 VLA transfer。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section 2 → Section 3 losses → single/multi-task experiments → Section 5 risks。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. 没有 image reconstruction 时，latent 被什么目标约束？
2. reward/value sufficiency 与物理 state identifiability 有何区别？
3. model capacity scaling 与 planning budget 如何分开？
4. 要把 latent 换成 contact / pose prediction，可能损失什么？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
