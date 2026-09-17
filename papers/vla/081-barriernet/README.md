# 05. BarrierNet

**BarrierNet: A Safety-Guaranteed Layer for Neural Networks**  
Xiao, Wei, Hasani, Ramin, Li, Xiao, Rus, Daniela  
**可学习安全层基础 · 修正原链接** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2111.11277v1，23 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2111.11277v1)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

传统 CBF layer 可能保守且难与 perception network 联训，希望安全约束可随环境信息调整。

## Method / Key Innovation

构造 differentiable higher-order CBF，把环境相关参数放入 differentiable QP，使安全层与 neural controller 端到端训练。

## Results：重点核对的证据

Sections 4–5（pp.5–9）、Algorithm 1（p.9）及 Section 6 的 numerical evaluations。本地是作者主页链接的 2021 preprint，不是 2023 T-RO final。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

原总结 arXiv:2003.09140 是 Coq proof-assistant 论文，已纠正为 arXiv:2111.11277；2023 journal title / DOI 另列在文献核对表。安全层依然在 inference path 中。

## 与师兄方向的关系

方向三的重要对照：network 学习 CBF 参数与整个网络独立输出安全动作是不同实现；应把 solver 是否保留写明。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section 4 → Section 5 的 QP → Algorithm 1 → Section 6。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. 哪些 safety parameters 可训练，哪些由设计者指定？
2. 控制量在 inference 时是否仍由 QP 决定？
3. CBF guarantee 要求哪些 dynamics / feasibility 条件？
4. 若直接模仿 BarrierNet 输出，student 自动继承 guarantee 吗？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

## 版本对应

原总结引用的 journal work 是 [BarrierNet: Differentiable Control Barrier Functions for Learning of Safe Robot Control](https://doi.org/10.1109/TRO.2023.3249564)，T-RO 2023。作者主页在该条目下提供两个早期 arXiv links，本目录采用其中 [2111.11277](https://arxiv.org/abs/2111.11277)。这里明确保留前身论文自己的 title、author list 与版本；尚未取得 journal final binary，不能把 23-page preprint 当成 T-RO final。[作者主页](https://people.csail.mit.edu/weixy/)。

