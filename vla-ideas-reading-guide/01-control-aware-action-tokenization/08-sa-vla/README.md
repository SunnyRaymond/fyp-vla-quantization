# 08. SA-VLA

**SA-VLA: State-aware tokenizer for improving Vision-Language-Action Models' performance**  
Jiang, Tengyue, Xu, Chunpu, Kang, Jiayue, Mu, Yao  
**直接相关 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2606.30113v1，15 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2606.30113v1)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

同一个 action code 在不同 proprioceptive states 下，可能需要对应不同 continuous action；固定 prototype 会限制解码表达能力。

## Method / Key Innovation

把 robot state 注入 detokenizer，比较 cross-attention 与轻量 state adapter 的 action-wise modulation；用有限 codebook 表达 state-dependent action families。

## Results：重点核对的证据

Methods（pp.3–5），Tables 4.1–4.3（pp.6–8）。作者报告 RoboTwin 与小规模 zero-shot sim-to-real 改善。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

state-conditioned reconstruction 是实现机制，不等同于 risk-conditioned bit allocation；仿真结果和 real-world 结果的任务数及 trials 需分别记录。

## 与师兄方向的关系

方向一必须考虑的 conditioning baseline：新方法若同时加入 state，收益不应全归因于 consequence-aware loss。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Figure 1 → 两种 state injection → Table 4.2 → Tables 4.1、4.3。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. state adapter 在 token generation 之前还是之后起作用？
2. fixed codebook 能表达多少 state-dependent variation？
3. 怎样隔离 state information 与新 distortion 的作用？
4. state noise 会如何影响 detokenization 和 contact stability？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
