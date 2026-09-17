# 03. OAT

**OAT: Ordered Action Tokenization**  
Liu, Chaoqi, Han, Xiaoshen, Gao, Jiawei, Zhao, Yue, Chen, Haonan, Du, Yilun  
**直接相关 · 优先精读 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2602.04215v2，16 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2602.04215v2)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

低 reconstruction error 不足以让离散 latent 适合 next-token prediction；同时希望短 prefix 就能生成可执行的完整 action chunk。

## Method / Key Innovation

使用带 registers 的 Transformer、Finite Scalar Quantization 和 ordering-inducing training；让 early tokens 提供 coarse information、later tokens 逐步补细节，支持 prefix-based detokenization。

## Results：重点核对的证据

Algorithms 1–2（pp.4–5）、Section V（pp.5–7）和 Section VI（p.8）；重点追踪 prefix budget 改变时的 reconstruction、success 与 latency。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

OAT 已支持 anytime / 可变 prefix budget，因此原总结“全部固定码率”不成立。但可截断 prefix 与按 contact risk 自动分配预算是不同问题。

## 与师兄方向的关系

这是方向一最应先排除重合的工作。新增机制应清楚落在 consequence-based distortion、risk estimator 或 online budget policy 上，不能只声称 variable-length tokens。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section III → Algorithm 1 → Algorithm 2 → Section V 中 prefix/action-horizon 对照。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. total decodability 是否意味着任意 token prefix 都对应安全动作？
2. ordering 是靠 architecture、loss 还是训练时 token dropout 获得？
3. 谁决定 inference 时使用多少 tokens？
4. 若用 rollout consequence 决定 prefix length，公平 baseline 应如何设？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

## 同一研究线的扩展稿

已附 [Ordered Action Tokens for Visuomotor Policy Learning](companion-2607.21670-v1.pdf)，**arXiv:2607.21670v1，2026-07-23，45 页**。它增加 token co-training / flow-based expert 等应用与更多任务。两份是不同 arXiv records，前者是原总结指定稿；本目录只算一个 numbered reading entry，另记一份 companion PDF。不要把 2 月稿的 20+ tasks 与 7 月稿的 60+ tasks 混作同一实验。

先读扩展稿 Abstract、Introduction 和 policy interface 部分，自己列出相对 2 月稿新增了哪些实验，再判断哪个版本适合引用。[扩展稿官方来源](https://arxiv.org/abs/2607.21670v1)。

