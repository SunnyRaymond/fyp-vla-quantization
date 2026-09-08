# 24. GRACE：VLM Quantization-Aware Distillation

**Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs**  
Yanlong Chen, Amirhossein Habibian, Luca Benini, Yawei Li · **ICML 2026**  
本地版本：**arXiv:2601.22709v5，2026-06-26，35 页** · 阅读状态：`unread`

- [本地 PDF](paper-arxiv-v5.pdf)
- [arXiv 与版本记录](https://arxiv.org/abs/2601.22709v5) · [官方 code](https://github.com/ForeverBlue816/GRACE) · [ICML 官方条目](https://icml.cc/virtual/2026/poster/60706) · [OpenReview](https://openreview.net/forum?id=xM4iQX34JA)
- [返回 Reading List](../../README.md) · [师兄的原始 report](../../../report_icml26.pdf)

## 身份与范围

这篇与 report 中 “GRACE / QAT” 的研究主题和 Yawei Li 作者信息相符。PDF 同时列出 ETH Zurich、NTU 等 affiliations；这支持论文身份匹配，但原 report 没有署名，不能据此独立确认师兄本人是否为作者。

这里的 GRACE 是 **VLM weight-only QAT + knowledge distillation**，不是 GRACE gradient-communication framework，也不是 VLA action-head QAT 或通用 QAT optimizer 理论。当前官方 repository 重点展示 Qwen3-VL；本地 v5 同时含 LLaVA、Qwen2-VL 与 Qwen3-VL 实验，应按具体表格区分。

## Background / Problem

低位宽 student 容量有限，而 teacher 的监督并非处处可靠。阅读问题是：在 quantization noise 下，应保留哪些 teacher 信息，如何避免把不可靠预测一并蒸馏进去？先分清较小 base model 的收益、distillation 的收益和 quantization 后的保留程度。

## Method / Key innovation

按四个模块读 Section 3：confidence-gated decoupled KD（GDKD）、visual-token relational CKA（RCKA）、adaptive Information Bottleneck controller、group-wise learned step-size quantization。训练 objective 为 `L_CE + β L_GDKD + ω L_RCKA`；β 用 smoothed loss 的 projected dual update 调整。

Section 3.4 中 weight 与 log-scale 联合更新，scale 使用独立 optimizer group、无 weight decay，并设置较大的 learning rate。它提供可参考的 QAT recipe，但不等价于对 Adam/AdamW 不收敛的普遍解释。IB 是方法的理论视角及 surrogate formulation，不能把全部 mutual information 当成已被精确计算。

## Results：去哪里找证据

| 要核对的结论 | 本地 PDF 定位 | 关键条件 |
|---|---|---|
| INT4 LLaVA student 的 benchmark retention | Table 2，p.7 | 7B base、13B teacher；同表平均值 66.5 → 67.2 为 0.7 percentage points |
| QAT 与 naive KD / GRACE 的差异 | Table 4，p.8 | Qwen2-VL 7B → 2B；三项 benchmark，不能混用另一张表的 average |
| Qwen3-VL 的扩展 | Appendix A.2，Table 6 | BF16 / INT8 / INT4 student 分开 |
| 实际 INT4 deployment | Appendix A.4，Figure 7 | A100、TinyChat；throughput / per-token latency / peak memory 各有独立口径 |

这是作者报告，未本地复现。vision encoder 保持 BF16；不能把模型称为全模块 W4A4。正文与 repository 对 projector 的 quantization scope 描述应以所用实现逐层检查，不能由 “INT4 model” 自动推断。

## Limitations / 对 FYP 的意义

Teacher quality、额外训练数据和 training compute 都会影响结果；高于未经同等训练的 base model 不说明 quantization 本身提高准确率。VLM benchmark accuracy 与 tokens/s 也不能替代 VLA closed-loop success、action-chunk latency 和 control frequency。

值得带着读的问题是：RCKA 对 visual tokens 的保持能否保护 action-critical features，以及 continuous flow action expert 应采用何种 distillation target。原文没有回答这些 VLA 问题。

## 阅读路线

- **20 minutes**：Figure 1 → Section 3 overview → Tables 2、4 → Appendix A.4。
- **90 minutes**：逐项重写 objective 和 β update；核对 Section 3.4 的 quantizer、STE 与 scale update；读 Tables 3–7 的 ablation，区分 KD-only、QAT-only、joint training。
- **3 hours**：读 Appendix D 的训练配置，对照官方 code 中实际 quantized modules、precision 和 optimizer groups，再设计一个不混淆额外训练预算的 VLA 对照。

## Reading Questions（读完自己回答）

1. GDKD gate 衡量 teacher 的什么不确定性？高 confidence 是否等于正确？
2. RCKA 保留关系结构而非逐点 feature，有什么利弊？
3. β update 与直接固定一个 KD weight 的差异在哪里？
4. 哪些量真实 quantize，哪些仍用 BF16？training fake quant 与 deployment packing 是否一致？
5. Table 4 是否已隔离不同 optimizer 的稳定性？缺少哪个实验？
6. 将 teacher entropy 换成 action-risk signal，会不会把“难”与“不可靠”混为一谈？
7. 如果 teacher 和 student 的 action heads 不同，应该蒸馏 velocity、trajectory 还是 rollout outcome？
8. 对 flow expert，scale LR、STE noise 与 ODE discretization 应如何分别测量？

## Meeting Card（留空）

- Problem：
- 四个模块各解决的问题：
- 最有说服力的 matched ablation：
- Quantization scope 与 deployment 条件：
- 可移植到 VLA 的一个机制：
- 想向师兄确认的问题：
