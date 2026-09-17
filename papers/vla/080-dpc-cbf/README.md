# 04. DPC with CBF

**Differentiable Predictive Control with Safety Guarantees: A Control Barrier Function Approach**  
Cortez, Wenceslao Shaw, Drgona, Jan, Tuor, Aaron, Halappanavar, Mahantesh, Vrabie, Draguna  
**安全边界必读 · 原总结重点** · CDC 2022  
本地版本：**arXiv:2208.02319v1，7 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2208.02319v1)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

离线训练 predictive neural controller 很快，但仅用 soft constraint penalties 不自动带来 deterministic safety。

## Method / Key Innovation

DPC 通过 differentiable dynamics / MPC objective 直接训练 neural policy，不要求模仿 expert action labels；加入 sampled-data barrier conditions，并在 online safe-set 边界附近干预。

## Results：重点核对的证据

Section III 的 barrier theory，Section IV-A/B/C（pp.4–5）分别讲 DPC、offline barrier training、online safety；Section V（pp.5–6）是 simulation。CDC 2022。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

原总结称其“证明 MPC 可蒸馏进网络”过于简化：这里有无监督直接 policy optimization，严格 guarantee 还涉及 online CBF。关闭所有 protection 后不能照搬 theorem。

## 与师兄方向的关系

方向三应区分“减少 online intervention”“不用 iterative MPC”“完全无 safety layer”三个目标。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section IV-A → IV-B → IV-C → Section III theorem assumptions → Section V。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. DPC 需要 expert demonstrations 吗？
2. offline penalty 与 online barrier enforcement 分别保证什么？
3. sampling interval 与 model mismatch 怎样进入 safety 条件？
4. 若完全关掉 protection，需要补充什么验证或新的证明？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

