# 07. X-Tokenizer

**X-Tokenizer: A Multimodal Action Tokenizer for Vision-Language-Action Pretraining**  
Kang, Miracle, Shi, Lights, Liang, Lucy, Gan, Roy, Liu, Dongxiu, Zhang, Pushi et al.  
**直接相关 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2606.14752v2，27 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2606.14752v2)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

action tokens 除了描述轨迹，还承担连接 multimodal reasoning 与 continuous control 的语义监督。

## Method / Key Innovation

Semantic Residual Quantization 对 RVQ 层施加不对称职责：首层用 Masked Action Modeling 学 coarse intent，后续层补 reconstruction residual；再加入 foundation-feature alignment 和 future-feature prediction。

## Results：重点核对的证据

Section 3 / Figure 1（pp.3–5）、Section 4（p.6 起）、Table 1（p.9）。分别看 semantic alignment、tokenizer ablation 与 VLA downstream。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

semantic consistency 并不等价于低 task risk；pretraining scale 与 mixed discrete-continuous policy 都可能贡献提升。

## 与师兄方向的关系

方向一的 semantic-interface 对照。新 objective 要说明保护的是 task outcome 的哪一部分，而不只是让 visual/action embeddings 更相似。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Figure 1 → SRQ 与两个 alignment objectives → tokenizer ablation → downstream tasks。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. 首层和后续 RVQ 层分别保留什么？
2. action tokens 用于直接执行还是训练监督？
3. semantic alignment 高时是否可能遗漏微小但关键的姿态误差？
4. 比较新 distortion 时如何控制 pretraining data 与 backbone？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
