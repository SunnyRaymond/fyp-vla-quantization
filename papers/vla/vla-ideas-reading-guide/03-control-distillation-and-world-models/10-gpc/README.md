# 10. Generative Predictive Control (GPC)

**Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling**  
Qi, Han, Yin, Haocheng, Zhu, Aris, Du, Yilun, Yang, Heng  
**online world-model planning 对照 · 原总结列出** · RA-L 2026  
本地版本：**arXiv:2502.00622v4，8 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v4.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2502.00622v4)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

已有 behavior-cloning policy 可生成 plausible proposals，但缺少按未来结果选优的能力。

## Method / Key Innovation

保持 pretrained diffusion policy frozen，另学 action-conditioned world model；部署时通过 model-based look-ahead 排序和修正候选动作。

## Results：重点核对的证据

Algorithm 1（p.2）、Sections III–V（pp.3–6）与 Section VII limitations（p.7）；本地是 RA-L 2026 accepted manuscript 的 arXiv version。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

不 retrain 原 policy 不等于零训练成本，world model 仍要训练；online planning 是方法组成部分。移除 planner 后不应假定 frozen policy 自动保留增益。

## 与师兄方向的关系

方向三的直接 teacher / inference-time baseline：比较 planner-on、student-only 和原 policy，而不只比较平均 success。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Algorithm 1 → world model data → ranking/refinement → experiments / limitations。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. random exploration data 解决哪种 model coverage 问题？
2. ranking 与 gradient refinement 分别需要什么预测？
3. model exploitation 怎样让 planner 选到错误动作？
4. 用 GPC 产生 distillation labels 会引入什么 distribution shift？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
