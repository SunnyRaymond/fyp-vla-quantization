# 05. FASTer

**FASTer: Toward Efficient Autoregressive Vision Language Action Modeling via Neural Action Tokenization**  
Liu, Yicheng, Zhang, Shiduo, Dong, Zibin, Ye, Baijun, Yuan, Tianyuan, Yu, Xiaopeng et al.  
**方法扩展 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2512.04952v2，24 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2512.04952v2)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

希望同时提高 action reconstruction 与 autoregressive inference efficiency，但 tokenizer 和 policy decoding 往往同时影响结果。

## Method / Key Innovation

FASTerVQ 把 action chunk 组织为 single-channel image 来学习时空依赖；FASTerVLA 再结合 block-wise autoregressive decoding 与轻量 action expert。

## Results：重点核对的证据

Section 3、Algorithm 1（pp.4–6），Table 1（p.7）与 inference timing Table 2（p.10）。分别读 tokenizer-only 和完整 VLA 的结果。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

完整 FASTer 同时改变 representation、decoding 和 action expert；其 speedup 不能直接当成某个 tokenizer 的纯收益。

## 与师兄方向的关系

为方向一提供联合设计的 baseline，同时提醒新方案需要在相同 policy architecture 上隔离 loss / allocation 的贡献。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Figures 2–3 → Algorithm 1 → Tables 1–2 → tokenizer ablations。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. action-as-image representation 引入什么结构假设？
2. block-wise decoding 怎样改变 latency 与 action dependence？
3. tokenizer-only ablation 是否保持 action expert 一致？
4. 在接触阶段自适应预算会不会破坏 block structure？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
