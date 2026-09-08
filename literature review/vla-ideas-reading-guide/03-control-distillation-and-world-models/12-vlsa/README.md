# 12. VLSA / AEGIS

**VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer**  
Hu, Songqiao, Liu, Zeyi, Liu, Shuang, Cen, Jun, Meng, Zihan, Wang, Shihefeng et al.  
**VLA safety baseline · 原总结列出** · IROS 2026 accepted  
本地版本：**arXiv:2512.11891v2，8 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v2.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2512.11891v2)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

VLA instruction following 与 collision avoidance 需要同时评价，普通 manipulation benchmark 不足以反映安全约束。

## Method / Key Innovation

通过 CBF-based plug-and-play Safety Constraint layer 形成 AEGIS，并构建 SafeLIBERO 评测障碍与不同空间复杂度下的执行。

## Results：重点核对的证据

Sections III–IV（pp.2–4），Section V simulation（pp.5–6）、Section VI real-world（p.7）。arXiv comments 标注 IROS 2026 accepted。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

formal guarantee 依赖 dynamics、state estimation、CBF feasibility 等条件；SafeLIBERO 的 collision avoidance 不等于覆盖所有 contact failure。SC layer 保留在 inference。

## 与师兄方向的关系

方向三必须包含的 VLA-specific safety-layer baseline；适合做 teacher-on、student-only、student+fallback 三组清晰比较。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Problem formulation → SC layer → SafeLIBERO protocol → real-world / ablations。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. 哪些安全约束可由该 CBF formulation 表达？
2. perception error 是否被纳入 guarantee？
3. obstacle avoidance rate 和 task success 的 denominator 是否一样？
4. 蒸馏后应记录哪些罕见但严重的失败，才能发现平均 success 掩盖的问题？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
