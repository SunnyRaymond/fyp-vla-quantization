# 02 — OPTQ / GPTQ

## 基本信息

- 会议正式发表版本的完整标题：*OPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*
- 作者：Elias Frantar、Saleh Ashkboos、Torsten Hoefler、Dan Alistarh
- 发表会议：ICLR 2023
- 本地来源：[官方发表版本 PDF](paper-iclr-2023.pdf)
- 命名说明：method 和 public implementation 目前通常称为 GPTQ；OPTQ 是该 conference version 中印刷的标题。
- 在本阅读路线中的定位：LLM-scale second-order weight-only PTQ

## 一句话要点

GPTQ 延续 OBC/OBQ 的 second-order reconstruction 思路并重组其 updates，使超大规模 Transformer layers 能够在数小时内完成 3–4-bit quantization；配套的 custom kernels 则把 compression 转化为 generation speedup。

## 问题与目标

Layer target 仍为 `||WX - W_hat X||²_F`，approximate Hessian 为 `H = 2XXᵀ`。挑战不再只有 accuracy：一个实用的 algorithm 必须能够处理 OPT-175B 这样规模的 model，而不必存储或更新不切实际的 full model-scale Hessian。

## 方法脉络

1. 按固定且共享的顺序 quantize weights，从而高效处理不同 rows。
2. 每次作出 quantization decision 后，使用 inverse Hessian 补偿剩余 weights。
3. 延后高成本的 matrix updates，再通过 lazy batch updates 按 blocks 执行。
4. 使用数值稳定的 Cholesky-based representation 表示 inverse Hessian。
5. 逐 layer 进行 quantization，使 working set 保持在有限范围内。

核心洞见是 systems-algorithm co-design：多项小型重排在保留 second-order correction 的同时，让计算能够在 GPU 上实际运行。

## 重点检查的证据

- Calibration 使用来自 C4 的 128 个 segments，每段长度为 2,048 tokens。
- 论文报告在一张 NVIDIA A100 上约 4.2 小时完成 OPT-175B quantization。
- OPT-175B 在 WikiText-2 上的 FP16 perplexity 为 8.34；4-bit 为 8.37；3-bit 为 8.68；采用 group size 128 的 3-bit 为 8.45。
- Table 6 报告在 batch-1、sequence-length-128 条件下，使用作者的 custom kernel 时，per-token latency 在 NVIDIA A100 上从 230 ms 降至 71 ms，在 NVIDIA A6000 上从 589 ms 降至 130 ms。

这些都是 paper-specific conditions，不能视为普适的 3–4× speedup guarantee。

## 局限与证据边界

- 该 method 是 weight-only：activations 和 KV cache 不属于主要 quantization target。
- Accuracy 依赖 calibration distribution、damping、group size 和 ordering choices。
- Runtime gains 依赖兼容的 packed format 和 custom kernel。
- Perplexity 与 zero-shot tasks 无法证明所有 downstream 或 interactive behavior。
- 该工作早于较新的 Llama-family architectures 和当前 serving stacks。

## 对你的 FYP 有何意义

GPTQ 是此后几乎所有 LLM quantization paper 都会涉及的 reference baseline。为了进行公平实验，应记录 implementation、group size、symmetric/asymmetric setting、calibration corpus、sequences 的数量与长度、kernel、batch size，以及 prompt/decode lengths。

## 阅读路线

### 25 分钟

阅读 Abstract、Sections 3.1–3.4、Tables 2 和 6，以及 Conclusion。

### 90 分钟

从 OBC 重新推导 compensation rule，再明确 shared order、lazy updates、blocking 和 Cholesky 分别如何改变 computational cost。

## 阅读问题——阅读后自行回答

1. 为什么 GPTQ 可以在多个 rows 之间使用同一个 quantization order，而不采用原始 OBQ selection rule？
2. Lazy batch updates 延后了哪些工作？这为什么有利于 GPU execution？
3. 为什么 group size 会同时影响 metadata/scale overhead 和 perplexity？
4. 论文中的哪项 comparison 匹配得足够严格，能够支撑 speed claim？
5. 若要把该 method 从 weight-only W4A16 改为 W4A4，需要改变什么？

## 讨论速记

- 需要解释的核心论点：second-order error compensation 可以在 175B scale 上实际运行。
- 需要绘制的机制：一个 quantization block，以及针对剩余 columns 的 delayed update。
- 需要引用的结果：OPT-175B 4-bit perplexity 和 Table 6 latency。
- 应主动说明的局限：compressed model size、quantization time 和 generation latency 是三个不同的 metrics。
