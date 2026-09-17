# 10. BEAST

**BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning**  
Hongyi Zhou, Weiran Liao, Xi Huang, Yucheng Tang, Fabian Otto, Xiaogang Jia et al.  
**解析 tokenizer 对照 · 原总结列出** · NeurIPS 2025  
本地版本：**NeurIPS 2025 proceedings final，26 页** · 阅读状态：`unread`

- [本地 PDF](paper-neurips2025.pdf) · [官方来源 / 版本记录](https://proceedings.nips.cc/paper_files/paper/2025/hash/fc75cfcec16170f2b54d00283c739928-Abstract-Conference.html)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

希望同时缩短 action sequence、支持 parallel decoding，并控制连续轨迹的平滑性，而不增加独立 tokenizer training。

## Method / Key Innovation

用 B-spline control points 表示 action sequence，可形成 discrete 或 continuous tokens；给定配置下输出 uniform-length tokens，以解析结构连接相邻轨迹片段。

## Results：重点核对的证据

NeurIPS 2025 proceedings final：先读 Section 3 的 B-spline formulation，再读 simulation / real-world evaluation 与 architecture comparisons；官方摘要列出 166 simulated tasks 与 8 real-world tasks。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

固定数量 control points 与 task-adaptive token budget 是不同机制。B-spline smoothness 不意味着 task safety，contact discontinuity 也可能是任务必要部分。

## 与师兄方向的关系

方向一需要的 analytic baseline；相比 learned codecs，更容易隔离“trajectory representation”与“训练学出的 distortion”的收益。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section 3 B-splines → discrete / continuous interfaces → parallel decoding → experiments。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. B-spline degree、control-point count 与 token budget 有何关系？
2. 边界 continuity 通过什么条件保证？
3. 如果关键 contact event 落在两个 control points 之间会怎样？
4. 怎样在相同物理误差与实时预算下比较 BEAST 和 risk-adaptive tokenizer？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

