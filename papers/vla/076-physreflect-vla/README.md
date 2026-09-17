# 08. PhysReflect-VLA

**PhysReflect-VLA: Physical Feasibility and Self-Reflective Regulation for Reliable Vision-Language-Action Policies**  
Yang, Jiayu, Yang, Tao, Li, Weijun, Chang, Xiang, Chao, Fei, Shang, Changjing et al.  
**可靠性对照 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2606.27146v1，8 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2606.27146v1)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

contact-rich long-horizon tasks 会遇到物理不可行 transition 与执行偏差，需要在线诊断和纠正。

## Method / Key Innovation

在 VLA execution loop 中加入 Feasibility Operator、Action Explanation Operator 和 LLM-based Reflection Module；用 two-stage training 准备 feasibility model 与 reflection。

## Results：重点核对的证据

Algorithms 1–2（p.5）与 Section IV（pp.5–7），阅读 real-world multi-stage tasks 和模块 ablation。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

在线 feasibility checking / reflective correction 本身不是 information-gain-directed probing；也不能仅凭平均 success 的提高获得 formal safety guarantee。

## 与师兄方向的关系

方向二和方向三的跨方向对照：区别事后纠正、执行前筛选、主动辨识，以及 inference-time 开销。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Operator definitions → Algorithm 2 execution → ablation / failure cases。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. feasibility 与 explanation 两个 operator 的监督来自哪里？
2. reflection 在失败前还是失败后触发？
3. 一次 corrective action 是否被显式优化为信息收集？
4. 关掉 reflection module 后，哪些收益仍保留在 policy 中？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

