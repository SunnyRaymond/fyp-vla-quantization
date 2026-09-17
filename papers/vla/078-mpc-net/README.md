# 02. MPC-Net

**MPC-Net: A First Principles Guided Policy Search**  
Carius, Jan, Farshidian, Farbod, Hutter, Marco  
**控制蒸馏基础 · 优先精读 · 本次补充** · RA-L 2020  
本地版本：**arXiv:1909.05197v2，8 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/1909.05197v2)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

普通 behavior cloning 只最小化 action distance，可能没把 optimal-control structure 与 constraint information 传给 student。

## Method / Key Innovation

用 MPC 解指导 policy search，但 training loss 来自 control Hamiltonian 的 optimality condition；配合 mixture-of-experts 表达多模式控制。

## Results：重点核对的证据

Algorithm 1 / Section II（pp.2–4）、Section III（pp.5–7）；包含 real quadruped gait control。RA-L 2020 的 accepted manuscript。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

已在 robot hardware 上展示 learned fast policy，直接限定方向三的宽泛新颖性。约束改善的 numerical evidence 不能泛化成任意 unseen state 的 safety guarantee。

## 与师兄方向的关系

非常贴近“控制知识进入网络”：比较 imitation loss、Hamiltonian loss 与 differentiable-rollout loss，找出你真正要引入的新信息。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Algorithm 1 → Hamiltonian objective → mixture-of-experts → robot evaluation。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. Hamiltonian loss 比 action MSE 多使用了哪些 teacher 信息？
2. constraint terms 如何进入训练？
3. policy 学到了控制结构还是仅更好拟合 teacher？
4. 若 world model 有误差，Hamiltonian supervision 是否仍可靠？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

