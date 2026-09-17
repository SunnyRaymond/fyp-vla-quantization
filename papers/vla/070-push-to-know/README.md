# 02. Push to know!

**Push to know! -- Visuo-Tactile based Active Object Parameter Inference with Dual Differentiable Filtering**  
Dutta, Anirvan, Burdet, Etienne, Kaboli, Mohsen  
**直接物理试探 · 优先精读 · 原总结列出** · IROS 2023  
本地版本：**arXiv:2308.01001v1，8 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2308.01001v1)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

仅由静态 observation 很难判断 mass、friction、center of mass 等；主动 pushing 可以制造可辨识的交互数据。

## Method / Key Innovation

使用 visuo-tactile active dual differentiable filtering 学 object–robot interaction，并以 N-step active formulation 选择下一次 push 的位置与方式。

## Results：重点核对的证据

Section III（pp.2–4）与 Section IV（pp.5–7），包含 simulation 和 real robot；核对 parameter error、exploration efficiency 和 sensing setup。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

这是 structured estimation / active exploration，不是通用 language-conditioned VLA。learned differentiable interaction model 也说明不能把整条旧路线都概括成依赖解析模型。

## 与师兄方向的关系

方向二最直接的 prior art：physical probing 和 active parameter inference 已有，新增贡献应落在 task-conditioned probing、sensor restrictions 或 VLA integration。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section III 的 state/parameter estimator → N-step action selection → experiments。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. dual filtering 中两个被估计的对象是什么？
2. 不同推力与接触点分别帮助辨识哪些参数？
3. 需要什么 tactile / force information？
4. 如何证明 trial action 提供信息，而不只是物理上把任务变简单？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

