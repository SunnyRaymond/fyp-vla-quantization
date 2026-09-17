# 08. LaWAM

**LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies**  
Chen, Jialei, Wang, Kai, Chen, Kang, Chen, Shuaihang, Gao, Feng, Tang, Wenhao et al.  
**直接 VLA / latent-world-model prior art · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2606.15768v1，23 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2606.15768v1)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

VLA 缺少动作后果的预见，而 pixel-space WAM 的 future video generation 成本较高。

## Method / Key Innovation

在 pretrained vision-feature space 学 latent-action-conditioned world model，用预测的 compact visual subgoals 条件化 action generation，避免重建 future video。

## Results：重点核对的证据

Section 3 / Figures 2–3（pp.2–5），Tables 1–3（pp.6–7），Section 5（p.8）。Table 1 明确 latency 为 model-only wall-clock time。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

187 ms/action chunk 与最高 24× 对比受 reference WAM、hardware 和 inference protocol 影响；latent visual subgoal 不是直接可解释的 mass/contact state。

## 与师兄方向的关系

虽在原总结方向二引用，这里放方向三主目录，因为它最直接约束 compact predictive world model 的新颖性；方向二保留 cross-link。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Figure 2 → latent world-model losses → Figure 3 execution → Tables 1–3。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. latent action、robot action 与 visual subgoal 是三种什么变量？
2. world model 的预测在哪个阶段被 action policy 使用？
3. latency 是否包含 sensing、decoding、controller 与 communication？
4. 预测很准是否足以处理静态不可辨识的物理变量？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

