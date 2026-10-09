# 03 — AQLM

## 基本信息

- 完整标题：*Extreme Compression of Large Language Models via Additive Quantization*
- 作者：Vage Egiazarian、Andrei Panferov、Denis Kuznedelev、Elias Frantar、Artem Babenko、Dan Alistarh
- 发表会议：ICML 2024
- 本地来源：[官方 PMLR PDF](paper-icml-2024.pdf)
- 在本阅读路线中的定位：面向 2–3 bits per parameter 的 learned additive codebooks

## 一句话要点

AQLM 将每个 weight group 表示为多个 learned codewords 之和，再跨完整 Transformer blocks 调整 codebooks，从而在极端的 sub-3-bit regime 中改善 accuracy–size Pareto frontier。

## 先理解 representation

AQLM 并不是把每个 scalar weight 放到单一 grid 上，而是把 weights 分组成 vectors。每个 vector 由多个 codebooks 中各选出的一个 codeword 相加来近似。因此，实际存储的 model 包含 discrete codes、codebooks、scales，以及其余未 quantize 的 parameters。

Nominal bits per parameter 必须计入所有这些组成部分，而不能只计算 code indices。

## 方法脉络

1. 为每个 weight matrix 初始化多个 codebooks。
2. 为每个 weight group 从每个 codebook 分配一个 code，使这些 codewords 的和能够 reconstruct 该 group。
3. 使用该 layer 的 calibration activations，让 assignment 具有 input-adaptive 特性。
4. 固定 discrete codes，并使用 output reconstruction loss 跨一个 Transformer block 联合优化 codebooks、scales 和 non-quantized parameters。
5. 使用融合 codebook lookup 与 matrix multiplication 的 custom CPU/GPU kernels 执行。

## 重点检查的证据

- 在 Llama 2 7B、2.02 bits per parameter 条件下，AQLM 报告 WikiText-2 perplexity 6.59、C4 perplexity 8.54，以及 average zero-shot accuracy 57.28。论文引用的 QuIP# 对照结果分别为 8.22、11.01 和 52.23；FP16 则为 5.12、6.63 和 62.35。
- 在 Llama 2 70B、2.07 bits 条件下，AQLM 报告 WikiText-2 3.94、C4 5.72、average zero-shot accuracy 68.75；FP16 对应为 3.12、4.97 和 70.17。
- 最强的 accuracy–size operating point 往往在约 2.5 bits，而不是恰好 2.0 bits。
- Table 5 报告主要 configuration 在 NVIDIA RTX 3090 上约有 1.20–1.31× layer-level GPU speedup；部分更大的数值来自牺牲 accuracy 或采用不同 representation choices 的 configurations。
- Appendix Table 14 报告在 NVIDIA RTX 3090 上、batch-1、128-token 的 end-to-end generation throughput：Llama 2 7B、13B 和 70B 的 FP16 分别为 54.2、29.5 和 5.8 tokens/s；一种 AQLM 2×8 configuration 则达到 114.1、68.1 和 14.3 tokens/s。

## 局限与证据边界

- Compression 的 computational cost 高于 RTN 或 GPTQ，因为 encoding 是一个困难的 combinatorial problem。
- Custom kernels 和 codebook memory access 是 method practicality claim 的组成部分。
- 最快的 configuration 不一定是最准确的 configuration。
- 论文主要研究 LLM weight compression；activations 和 KV cache 仍保持 high precision。
- Calibration-set size 和 block-level tuning 会影响结果，因此 comparison 必须匹配这些 budgets。

## 对你的 FYP 有何意义

AQLM 清楚区分了 numerical precision 与 model representation。它很适合用于设计这样的实验：报告 actual checkpoint bytes、codebook/scale metadata、peak memory、quantization time 和 kernel throughput，而不是只写“2-bit”。

## 阅读路线

### 25 分钟

阅读 Abstract、Figure 1、representation definition、Table 1、Table 5 和 Limitations。

### 90 分钟

沿着从 initial encoding 到 block-level fine-tuning 的 pipeline，确认哪些 variables 是 discrete 的、哪些仍可训练，以及 inference time 实际存储了什么。

## 阅读问题——阅读后自行回答

1. 在 representation level，additive quantization 与 scalar GPTQ 有何不同？
2. 为什么 block-level tuning 可以修复 independent per-layer reconstruction 遗漏的 errors？
3. 是什么使 discrete codes 无法通过普通 gradient descent 优化？
4. 为证明一个 bits-per-parameter 数值，需要计入哪些 bytes？
5. 在什么情况下，2.5-bit operating point 比作为 headline 的 2-bit result 更实用？

## 讨论速记

- 需要解释的核心论点：learned additive codebooks 在低于 3 bits per parameter 时尤其有价值。
- 需要绘制的示意图：一个 weight group 如何由多个 codewords 之和 reconstruct。
- 需要引用的结果：Llama 2 7B at 2.02 bits 和 Appendix Table 14。
- 应主动说明的局限：quantization cost 与 lookup-kernel behavior 是核心因素，而非次要细节。

