# 05. CoMe-VLA / Act, Sense, Act

**Act, Sense, Act: Learning Active Perception from Large-Scale Egocentric Human Data**  
Li, Jialiang, Qiao, Yi, Guo, Yunhan, Chen, Changwen, Lian, Wenzhao  
**关键补充 · VLA 主动感知** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2602.04600v2，27 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2602.04600v2)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

长程操作需要 history-dependent perception–action loop，单张图像无法表达之前探索了什么、还有哪些信息未获得。

## Method / Key Innovation

利用 human egocentric data 学 active perception priors；通过 cognitive auxiliary head 管理 sub-task transition，以 dual-track memory 结合 proprioceptive 与 visual history。

## Results：重点核对的证据

Sections 3–6（pp.4–10 起），重点看 active perception taxonomy、data curriculum 和 wheel-based humanoid 的实验。当前采用 arXiv v2，其标题已更新。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

这是主动感知 VLA 的真实相邻工作，削弱“所有 VLA 不确定性方法都是被动的”这一概括；是否覆盖显式 mass/friction posterior 与 contact probing，需要更细的任务核对。

## 与师兄方向的关系

方向二不能只与 ActiveVLA 比较；CoMe-VLA 是本次新增的必要 boundary reading。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section 3 problem → Section 4 data → Section 5 memory / cognitive head → Section 6。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. 哪些行为真正改变了可获得的信息？
2. memory 保存 belief、raw history 还是 learned features？
3. human active-perception prior 如何对齐 robot embodiment？
4. 若任务要求试探重量，现有训练数据是否包含可辨识的 interaction？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

