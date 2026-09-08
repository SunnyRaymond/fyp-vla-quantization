# 01. FAST

**FAST: Efficient Action Tokenization for Vision-Language-Action Models**  
Pertsch, Karl, Stachowicz, Kyle, Ichter, Brian, Driess, Danny, Nair, Suraj, Vuong, Quan et al.  
**基础必读 · 原总结列出** · RSS 2025  
本地版本：**arXiv:2501.09747v1，19 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2501.09747v1)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

逐 timestep、逐 dimension 的 binning 会产生很长的 action sequence，高频 dexterous control 尤其明显。先理解 action chunk、sampling frequency 与 token length 的区别。

## Method / Key Innovation

把 action time series 变到 DCT frequency domain，再 quantize coefficients、用 BPE 压缩；FAST+ 用大规模 robot trajectories 训练通用 tokenizer。创新是把成熟的 signal compression 接到 autoregressive VLA interface。

## Results：重点核对的证据

Section IV 的 tokenizer 与 Section V 的 robot experiments；本地 pp.4–8。作者报告 dexterous / high-frequency tasks 和训练效率改善，读图时分清 FAST、FAST+ 与 π0-FAST。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

固定 quantization recipe 不意味着每个 chunk 都输出同样多 tokens；BPE 输出长度可随内容变化。训练速度收益不等同于机器人 closed-loop frequency 提升。

## 与师兄方向的关系

方向一的起点：已有 compression interface，但主要 distortion 仍是 action reconstruction；control consequence 是否应进入 quantization objective 才是下一层问题。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section IV → Section V 的 compression / task performance 对照 → discussion。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. DCT、coefficient quantization 与 BPE 分别消除了哪种冗余？
2. 同一个 recipe 下，不同 chunk 的 token length 是否相同？
3. contact event 在 frequency domain 中可能分散在哪里？
4. 怎样区分 training speed、decoding latency 与 physical control rate？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
