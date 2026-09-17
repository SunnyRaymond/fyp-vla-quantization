# 06. PI-VLA

**PI-VLA: Adaptive Symmetry-Aware Decision-Making for Long-Horizon Vision–Language–Action Manipulation**  
Yina Jian, Di Tian, Xuan-Jing Chen, Zhen-Yuan Wei, Chen-Wei Liang, Mu-Jiang-Shan Wang  
**不确定性对照 · 原总结列出** · Symmetry 2026  
本地版本：**Symmetry 18(3), 394; publisher final, 2026-02-24，26 页** · 阅读状态：`unread`

- [本地 PDF](paper-publisher.pdf) · [官方来源 / 版本记录](https://doi.org/10.3390/sym18030394)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

长 action chunk 在 prediction error 或 action disagreement 增大时可能继续执行过久，需要调整 execution horizon。

## Method / Key Innovation

Cognitive–Motor Synergy 同时输出 discrete / continuous action 与 predicted state；AURD 结合 action consensus discrepancy 与 prediction error，触发 horizon adjustment / replanning。

## Results：重点核对的证据

Publisher PDF：Section 3（pp.5–10）、Algorithm 1（p.10）与 Sections 5–6。特别核对 AURD 的实际执行逻辑，而非仅依据其名称。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

Active Uncertainty-Resolving Decider 的命名不自动表示选择了 information-gathering physical action；当前机制主要支持 adaptive execution / replanning。不同 paper 的 LIBERO averages 不可直接横比。

## 与师兄方向的关系

为方向二提供“不确定时重规划”的 baseline，明确与“采取动作来测量 hidden physical variable”的差异。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section 3.4 / Algorithm 1 → main experiment protocol → AURD ablation。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. uncertainty signal 是 calibrated probability 还是 heuristic score？
2. AURD 输出了新探索动作，还是改变 chunk 执行长度？
3. 105.2 Hz 的计算对象与 physical feedback frequency 是否一致？
4. probe-and-act 如何在相同 wall-clock budget 下与 replanning 比较？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

