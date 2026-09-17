# 03. Differentiable MPC

**Differentiable MPC for End-to-end Planning and Control**  
Amos, Brandon, Rodriguez, Ivan Dario Jimenez, Sacks, Jacob, Boots, Byron, Kolter, J. Zico  
**可微优化基础 · 原总结列出** · NeurIPS 2018  
本地版本：**arXiv:1810.13400v3，16 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v3.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/1810.13400v3)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

希望 cost / dynamics 能通过最终控制目标训练，需要对 MPC solver 的输出求导。

## Method / Key Innovation

在 controller fixed point 对 convex approximation 的 KKT conditions 做 differentiation，学习 MPC 的 cost 和 dynamics；MPC 本身作为 policy class。

## Results：重点核对的证据

Sections 3–4（pp.3–6）、Figures 2–3 和 Section 5（pp.7–9）；pendulum / cartpole imitation experiments。NeurIPS 2018。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

可微 MPC 不等于把 MPC 蒸馏掉；该 policy class 仍求解 MPC。fixed-point convergence 和 approximation assumptions 也影响 gradient 的意义。

## 与师兄方向的关系

方向三的 differentiable teacher / planner 技术基础，必须与 student-only inference 的工作区分。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Figure 1 → differentiable LQR → Section 4 fixed-point MPC → Section 5。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. 求导对象是 optimizer iterations 还是 optimality conditions？
2. 没有收敛到 fixed point 会怎样？
3. learned cost 与 learned dynamics 是否可辨识？
4. 要移除 inference MPC，还必须增加哪一步学习？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

