# 01. Shielding-Aware Dual Control

**Active Uncertainty Reduction for Safe and Efficient Interaction Planning: A Shielding-Aware Dual Control Approach**  
Hu, Haimin, Isele, David, Bae, Sangjae, Fisac, Jaime F.  
**理论基础 · 本次补充** · IJRR journal work; local arXiv v2  
本地版本：**arXiv:2302.00171v2，25 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2302.00171v2)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

其他 agent 的 hidden goals / parameters 会影响交互结果，控制动作同时改变环境和未来可获得的信息。

## Method / Key Innovation

把未知参数作为 stochastic hidden states，通过 scenario sampling 近似 stochastic dynamic programming，再用 MPC 求解；结合 runtime shielding 与 shielding-aware planning。

## Results：重点核对的证据

Sections 3–6（pp.4–11）与后续 driving / 1:10 hardware experiments。重点读 belief update、scenario tree 和 dual effect 的条件。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

这里的 uncertainty 主要是 agent behavior，不是物体质量；推理保留 MPC 与 safety filter。不能把它当作已完成 VLA probing distillation。

## 与师兄方向的关系

给方向二提供严谨的 exploration–exploitation formulation；也说明信息收集不一定必须由显式 information-gain bonus 实现。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Problem statement → scenario-tree Algorithms 1–2 → Section 6 shielding → experiments。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. dual effect 与普通 receding-horizon replanning 有何区别？
2. 哪些 action 改变 belief，而不只是改变物理 state？
3. explicit information gain 与 implicit dual control 的区别是什么？
4. 把 hidden intent 改成 friction class 会改变哪些 model assumptions？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
