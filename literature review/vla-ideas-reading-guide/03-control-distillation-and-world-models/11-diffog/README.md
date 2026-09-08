# 11. DiffOG

**DiffOG: Differentiable Policy Trajectory Optimization with Generalizability**  
Xu, Zhengtong, Miao, Zichen, Qiu, Qiang, Zhang, Zhe, She, Yu  
**trajectory refinement 对照 · 原总结列出** · arXiv preprint；本次未独立核验 venue  
本地版本：**arXiv:2504.13807v5，20 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v5.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2504.13807v5)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

visuomotor policy 的动作可能 jerky 或违反 constraints，需要在改善轨迹质量时保持原 policy 的任务分布。

## Method / Key Innovation

结合 Transformer trajectory representation 与 differentiable optimization layer，学习可泛化的 trajectory refinement。

## Results：重点核对的证据

Section III（pp.3–7）、Section IV（pp.7–10 起），包含 11 simulation tasks、2 real-world tasks，以及 smoothness / constraints / inference time。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

优化层是系统的一部分；更平滑不必然提高任务 success，尤其 contact-rich transitions。原总结标注 T-RO 2025；本次本地采用 arXiv manuscript，未据此冒充 journal final。

## 与师兄方向的关系

方向三需要的 trajectory-quality baseline，帮助定义 student-only 测试中的 jerk、constraint violation 与 task success。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Framework / training stages → differentiable optimization → trajectory metrics → latency。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. 优化目标怎样避免 refined action 离开 demonstrations 的分布？
2. 哪些 constraints 是硬约束、哪些是 penalty？
3. smoothness 是否可能伤害快速接触转换？
4. 移除 optimization layer 后，base policy 的输出会变化吗？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
