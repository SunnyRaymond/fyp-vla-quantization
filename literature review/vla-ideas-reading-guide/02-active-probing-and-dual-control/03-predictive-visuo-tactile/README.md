# 03. Predictive Visuo-Tactile Interactive Perception

**Predictive Visuo-Tactile Interactive Perception Framework for Object Properties Inference**  
Dutta, Anirvan, Burdet, Etienne, Kaboli, Mohsen  
**直接物理试探 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2411.09020v1，21 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2411.09020v1)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

要处理 homogeneous、heterogeneous 与 articulated objects，仅靠一种 push 或单步 observation 可能不够。

## Method / Key Innovation

结合 active shape perception、pushing / pulling、GNN-based dual differentiable filtering 和 N-step information gain，主动挑选用于估计物性的数据。

## Results：重点核对的证据

Section III（p.4 起）与后续 real-robot experiments：看 object property inference、goal-driven task 和 change detection 的不同 protocol。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

估计收益仍依赖感知与交互模型；goal-driven demo 不等价于通用 VLA 学到了可迁移 probing policy。

## 与师兄方向的关系

把方向二从“会推一下”推进到多种探索动作与任务联系；非常适合对照师兄提出的小维度 unknown vector。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Framework → active shape perception → GNN filtering → information-gain action choice → goal-driven examples。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. pushing 与 pulling 能辨识的参数是否相同？
2. N-step information gain 是对哪个 posterior 定义的？
3. 物理参数变化时 estimator 如何忘记旧证据？
4. 能否用 vision/proprioception 替代部分 tactile sensing，代价是什么？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
