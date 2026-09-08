# 04. ActionCodec

**ActionCodec: What Makes for Good Action Tokenizers**  
Dong, Zibin, Liu, Yicheng, Zhang, Shiduo, Ye, Baijun, Yuan, Yifu, Ni, Fei et al.  
**直接相关 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2602.15397v1，18 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2602.15397v1)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

action reconstruction 做得好，未必让 VLA 的 token prediction 更容易优化。需要从 downstream optimization 反推 tokenizer design。

## Method / Key Innovation

分析 temporal token overlap、vocabulary redundancy、multimodal mutual information 与 token independence，并把这些原则落实到 learned action tokenizer。

## Results：重点核对的证据

Section 4 的设计分析、Section 5（p.5）、Tables 1–5（pp.6–8）。LIBERO 的 95.5% 与带 architecture enhancements 的 97.4% 是不同配置，不能都归因于 tokenizer。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

信息论代理指标与任务成功有关，但未由此证明它就是 closed-loop consequence distortion。benchmark success 也受 backbone、pretraining 和 architecture changes 影响。

## 与师兄方向的关系

帮助将“换一个 distortion”与“改善 token modelability”拆开，避免把现有 design principles 重新命名为 control awareness。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section 4 → Section 5 → Table 3 的 compression/efficiency → Table 5 ablations。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. 每个信息论设计原则对应哪个可计算量？
2. 哪个 ablation 最直接支持 token overlap 的作用？
3. 95.5 与 97.4 的模型配置差在哪里？
4. 如何证明新 objective 的收益来自控制后果而非更好预测的 codebook？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
