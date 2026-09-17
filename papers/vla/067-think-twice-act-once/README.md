# 09. FlashVLA / Think Twice, Act Once

**Think Twice, Act Once: Token-Aware Compression and Action Reuse for Efficient Inference in Vision-Language-Action Models**  
Tan, Xudong, Yang, Yaoxin, Ye, Peng, Zheng, Jialin, Bai, Bizhe, Wang, Xinyi et al.  
**效率对照 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2505.21200v1，16 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2505.21200v1)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

视觉 tokens 与连续 action steps 都存在冗余，可否不 retrain 就降低 VLA 推理成本？

## Method / Key Innovation

结合 information-guided visual-token selection 与 token-aware action reuse，在稳定片段复用动作、减少重复 decoding。

## Results：重点核对的证据

Method 与 Experiments（本地正文 pp.3–8）。摘要报告 LIBERO 下 FLOPs 减少 55.7%、latency 减少 36.0%、success 下降 0.7%；应核对各自 denominator。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

这里是 arXiv:2505.21200，和主阅读库 arXiv:2608.27384 的 streaming FlashVLA 不是同一篇。训练免调并不表示在所有 contact transitions 下可无风险复用动作。

## 与师兄方向的关系

方向一的 efficiency baseline，帮助把 action-token compression、visual pruning 与 action reuse 三种成本节省分开。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Method 的 reuse criterion → visual pruning → main results → ablation / failure cases。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. reuse decision 依赖哪些当前或历史信号？
2. reuse 能否错过突然发生的接触事件？
3. 0.7 是 absolute percentage points 还是 relative drop？
4. 如何在同一时间预算下比较 token allocation 与 action reuse？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

