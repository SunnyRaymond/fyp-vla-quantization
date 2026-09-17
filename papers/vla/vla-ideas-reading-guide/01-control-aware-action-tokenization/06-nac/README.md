# 06. NAC

**NAC: Neural Action Codec for Vision-Language-Action Models**  
Jawaid, Ahad, Xiang, Yu  
**方法扩展 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2606.21372v1，15 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2606.21372v1)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

动作也是多通道 time series，neural audio codec 的 compression 机制能否迁移到 action tokenization？

## Method / Key Innovation

采用 multi-scale RVQGAN、offset codebooks 与 Vocos-style ISTFT decoder；用 time-domain 和非 mel spectral losses 替换不适合 kinematics 的 audio objectives。

## Results：重点核对的证据

Section 3 与 Algorithm 1（pp.3–5），Tables 1–3（pp.7–8）：分别检查 reconstruction、downstream success、compression 和 latency。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

spectral fidelity 仍是 surrogate；GAN decoder 的 plausible motion 不等于保留每次精细接触所需的 exact control。

## 与师兄方向的关系

直接对应师兄的 rate–distortion 视角：可将其作为保持 codec architecture、只改变 distortion 或 budget 的对照。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Figure 2 → loss definitions → Table 1 ablations → Tables 2–3。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. 为什么 mel-spectrogram objective 不适合动作？
2. RVQ 层数、token 数和 bit rate 怎样换算？
3. offset codebooks 如何影响 AR 可预测性？
4. 应如何构造低 MSE 却导致抓取失败的测试片段？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
