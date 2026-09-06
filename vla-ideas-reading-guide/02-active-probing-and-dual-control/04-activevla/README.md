# 04. ActiveVLA

**ActiveVLA: Injecting Active Perception into Vision-Language-Action Models for Precise 3D Robotic Manipulation**  
Liu, Zhenyang, Gu, Yongchong, Wang, Yikai, Xue, Xiangyang, Fu, Yanwei  
**VLA bridge · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2601.08325v1，17 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2601.08325v1)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

固定视角和分辨率可能遮住操作所需信息；需要先定位重要区域，再选择更有用的 observation。

## Method / Key Innovation

coarse-to-fine 3D critical-region localization，结合 active view selection 与 3D zoom-in，以 relevance、diversity 和 occlusion 为视角选择依据。

## Results：重点核对的证据

Section 3 / Figure 2（pp.3–5），Tables 1–4（pp.6–8）；包括 RLBench、COLOSSEUM、GemBench 和 real-world transfer。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

“active view” 可能涉及从已有 3D input 中投影/选择视角，必须查清是否需要真实移动相机。它不直接证明通过接触辨识隐藏 mass 或 friction。

## 与师兄方向的关系

方向二的观测选择对照：要分别评价 occlusion uncertainty 与 visual observation 本身无法辨识的物理 uncertainty。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Figure 2 → view selection / zoom-in → Table 4 ablation → simulation / real-world protocol。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. 新的 viewpoint 是真实传感动作还是 3D reprojection？
2. view selection 优化了哪个可计算指标？
3. 已经可见但质量未知的物体，方法能得到新信息吗？
4. 如何公平计算 active sensing 的时间成本？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
