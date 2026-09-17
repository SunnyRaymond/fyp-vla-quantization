# 07. Perturbation-Based Failure Detection

**Perturbation-Based Epistemic Uncertainty for Failure Detection in Vision-Language-Action Models**  
Lee, Yousung, Har, Dongsoo  
**不确定性对照 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2606.20754v2，8 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2606.20754v2)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

continuous / flow VLA 不天然提供可直接使用的 action probability，普通 stochastic sampling 也未必反映 epistemic uncertainty。

## Method / Key Innovation

对选定 weights 注入 low-rank perturbations，以 perturbed action predictions 的 disagreement 形成 training-free failure-detection signal。

## Results：重点核对的证据

Section IV（p.3 起）和 Section V（pp.3–7）；LIBERO-PRO 的 AUROC / balanced accuracy 与 real-world unseen-object detection。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

failure detection 不等于 failure prevention，更不等于识别具体 friction / mass；多次扰动 forward pass 的成本必须纳入。

## 与师兄方向的关系

方向二需要区分“何时值得 probing”的触发器与“怎么 probing”的 policy；这篇可作触发信号 baseline。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section IV perturbation scheme → score aggregation → Section V distribution shifts。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. perturbation scale 如何影响 uncertainty score？
2. epistemic disagreement 能否区分感知失败与物理未知？
3. AUROC 改善能否推出 task success 改善？
4. 若把 score 用作 probing trigger，false positive 会浪费多少时间？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

