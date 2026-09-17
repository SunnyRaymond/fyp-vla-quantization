# 02. VQ-VLA

**VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers**  
Wang, Yating, Zhu, Haoyi, Liu, Mingyu, Yang, Jiange, Fang, Hao-Shu, He, Tong  
**基础必读 · 原总结列出** · ICCV 2025  
本地版本：**arXiv:2507.01016v1，10 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v1.pdf) · [官方来源 / 版本记录](https://arxiv.org/abs/2507.01016v1)
- [返回本方向](../README.md) · [三个方向总览](../README.md)

## Background / Problem

需要一个跨任务可复用的 learned action vocabulary，单靠少量真实轨迹训练 codebook 容易限制覆盖。

## Method / Key Innovation

扩大 vector-quantized action tokenizer 的 trajectory training corpus，并利用大量 synthetic trajectories；tokenizer 学习连续动作的时空结构，再作为 downstream VLA action interface。

## Results：重点核对的证据

Method 与 Experiments（本地 pp.3–8）：关注 tokenizer data scaling、synthetic/real gap、simulation 与 real-robot long-horizon 对照。ICCV 2025 身份由 arXiv comments 确认。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

synthetic data 的有效性是特定数据生成方式和任务上的经验结果；smooth reconstruction 不自动保证 contact timing 或 safety。注意这是 action vector quantization，不是压缩 VLA weights。

## 与师兄方向的关系

方向一的 learned-tokenizer baseline：若改变 distortion，需要与同训练数据、同 codebook budget 的 VQ baseline 比较。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Method 的 tokenizer architecture → data construction → scaling 与 real-world experiments。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. synthetic trajectories 覆盖的是几何轨迹还是接触动力学？
2. pretraining corpus 的扩大与 tokenizer architecture 哪个贡献更大？
3. 如何在固定 vocabulary / token budget 下加入 task cost？
4. smoothed action 会不会抹掉必要的接触切换？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：

