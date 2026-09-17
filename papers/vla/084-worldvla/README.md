# 09. WorldVLA

**WorldVLA: Towards Autoregressive Action World Model**  
Cen, Jun, Yu, Chaohui, Yuan, Hangjie, Jiang, Yuming, Huang, Siteng, Guo, Jiayan et al.  
**VLA / WAM bridge · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2506.21539v1，14 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2506.21539v1)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

action prediction 与 future-image modeling 能否在同一 autoregressive framework 中互相改善？

## Method / Key Innovation

统一 action / image understanding 和 generation，并用 action attention masking 减轻 autoregressive action sequence 的误差问题。

## Results：重点核对的证据

Section 3 / Figures 2–3（pp.4–5），Table 2（p.6）、Table 3（p.7）与 Table 4（p.9），分别核对 action-model 和 world-model ablation。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

联合模型能提高某些 benchmark 指标，不表示已学到足够准确、校准良好的 contact dynamics；图像预测质量也不是 safety guarantee。

## 与师兄方向的关系

方向三的 pixel-generation 对照，帮助说明为什么选择 compact task state，以及移除 world-model inference 时要测试哪些性能。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Figure 2 → attention masks → Tables 2–4 → limitations。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. action model 和 image model 的 token / attention 如何共享？
2. world-model objective 带来的改善能否与额外训练算力分开？
3. future images 是否真的用于 online planning？
4. 若训练时保留 image auxiliary loss、推理时移除，怎样核验收益来源？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

