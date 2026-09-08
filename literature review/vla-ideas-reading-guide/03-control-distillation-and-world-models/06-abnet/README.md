# 06. ABNet

**ABNet: Adaptive explicit-Barrier Net for Safe and Scalable Robot Learning**  
Wei Xiao, Tsun-Hsuan Wang, Chuang Gan, Daniela Rus  
**安全结构扩展 · 本次补充** · ICML 2025  
本地版本：**ICML 2025 / PMLR 267 proceedings final，19 页** · 阅读状态：`unread`

- [本地 PDF](paper-icml2025.pdf) · [官方来源 / 版本记录](https://proceedings.mlr.press/v267/xiao25f.html)
- [返回本方向](../README.md) · [三个方向总览](../../README.md)

## Background / Problem

iterative differentiable QP 可能限制训练规模、数值稳定性与速度，需要把 barrier 约束更直接地编码进模型。

## Method / Key Innovation

ICML 2025 的 Adaptive explicit-Barrier Net 使用 explicit barrier 的 closed-form structure 和 multi-head aggregation，兼顾 scaling 与 safety analysis。

## Results：重点核对的证据

Section 3、Algorithm 1（pp.2–4）、Figure 3（p.5）、Tables 1–3（pp.6–8）；包含 robot manipulation、noisy-input closed-loop 和 visual driving。

阅读时自己填写：`model / training data / task / baseline / metric / inference budget / simulation or real robot / failure cases`。这里概述作者报告并提供定位，不表示已独立复现，也不标记为你已读。

## Limitations / Evidence Boundary

本地选用 PMLR proceedings final；较早 arXiv:2406.13025 标题为 Attention BarrierNet，不能把二者细节混用。closed-form safety layer 仍是推理结构，不等于 unconstrained student 获得保证。

## 与师兄方向的关系

方向三在“关闭优化器”的表述上必须比较 explicit solver / barrier architecture 与真正移除安全结构。 这是阅读组织与研究判断，不是原论文已经完成的新实验。

## 阅读路线

- **20 minutes**：先读 Abstract 与 overview figure，再按此顺序扫描：Section 3 explicit barrier → aggregation guarantee → Figure 3 runtime → Tables 1–3。 最后只记录一个最有说服力的结果与其条件。
- **60–90 minutes**：逐项追踪 input、latent / belief / action、loss 和 inference path；把上面的关键结果同其 baseline 放到同一张表；完成下面 Reading Questions。
- **2 hours**：检查 appendix 的 data、hardware、trials 和 ablations；填写 Meeting Card，并写出一个能区分本篇机制与师兄构想的最小对照。此步是阅读练习，不要求立即申请 GPU。

## Reading Questions（留给你阅读后回答）

1. closed-form QP solution 需要什么约束形式？
2. multi-head aggregation 为什么仍安全？
3. “无 iterative optimization”与“无 safety layer”是否相同？
4. 如果只保留 distilled VLA，ABNet proof 的哪一步会失效？
5. 最强结论依赖哪些实验或理论条件？超出条件后最可能怎样失败？
6. 如果只能保留一个 matched ablation 来判断它对本方向的贡献，你会选择什么？

## Meeting Card（留空）

- Problem：
- 核心机制（一句话 + 一个公式或流程）：
- 最强证据及 PDF 定位：
- 最重要的限制 / alternative explanation：
- 对本方向已覆盖的部分：
- 仍想验证的问题：
