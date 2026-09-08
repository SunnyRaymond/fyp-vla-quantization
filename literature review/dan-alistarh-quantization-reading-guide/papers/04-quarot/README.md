# 04 — QuaRot

## 基本信息

- 完整标题：*QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*
- 作者：Saleh Ashkboos、Amirkeivan Mohtashami、Maximilian L. Croci、Bo Li、Pashmina Cameron、Martin Jaggi、Dan Alistarh、Torsten Hoefler、James Hensman
- 发表会议：NeurIPS 2024
- 本地来源：[官方会议论文 PDF](paper-neurips-2024.pdf)
- 在本阅读路线中的定位：通过 function-preserving rotations 实现 end-to-end W4A4KV4 inference

## 一句话要点

QuaRot 利用 Transformer invariances 插入 randomized Hadamard rotations，以分散 outlier energy，使 weights、activations 和 KV cache 更容易被 quantize 到 4 bits，同时不改变 full-precision function。

## 关键区别

QuaRot 不只是另一种 rounding rule，它改变了执行 quantization 的 coordinate system。Orthogonal rotations 在被正确吸收到其他运算或正确配对时能够保持 full-precision computation 不变，却可以显著改变 low-bit quantizer 所看到的 distribution。

## 方法脉络

1. 对 residual stream 应用 function-preserving rotations。
2. 尽可能将兼容的 rotations fold 进 weight matrices，使其不产生额外 runtime operation。
3. 在 rotation 无法完全 fuse 的 operations 前后放置 online Hadamard transforms，包括 attention 和 feed-forward paths。
4. 将 weights、activations 和 KV cache quantize，以实现目标 W4A4KV4 execution。
5. 主要 4-bit weight path 使用 GPTQ，执行时使用兼容的 integer kernels。

## 重点检查的证据

- 比较 weight-only、weight-and-activation 和完整 W4A4KV4 settings，不要只引用一个 aggregate number。
- 检查不同 Llama 2 model sizes 和 tasks 下的 performance 变化。
- 将 rotations 的效果与 GPTQ、clipping、group size 和 kernel choice 的效果分开。
- 核对 online Hadamard transforms 所报告的 overhead，以及它们在何处被 fuse。

论文默认的强 4-bit weight path 使用 GPTQ calibration；不要把整套 approach 概括成 calibration-free。

## 局限与证据边界

- Rotations 会减少 outliers，但不能消除所有 quantization error。
- 一些 Hadamard transforms 仍需 online 执行，因此存在取决于 shape 和 hardware 的 overhead。
- W4A4KV4 support 需要兼容的 integer kernels 和 model graph changes。
- Evaluation 与特定的 Llama-family models、calibration settings、tasks 和 hardware 绑定。
- Full precision 下的 rotation equivalence 并不意味着 finite-precision arithmetic 下仍然等价。

## 对你的 FYP 有何意义

QuaRot 是从 weight-only compression 走向 end-to-end low-bit inference 的重要桥梁。它还提供了一份实用的 evaluation checklist：weight memory、activation peak memory、KV-cache growth、prefill latency、decode latency、batch size、context length 和 online-transform overhead。

## 阅读路线

### 25 分钟

阅读 Abstract、Figure 1、rotation identities、主要 W4A4KV4 results，以及 runtime discussion。

### 90 分钟

画出一个 Transformer block 中的每个 rotation，并标记它是 offline fused 还是 online executed；随后追踪每个位置被 quantize 的 tensor。

## 阅读问题——阅读后自行回答

1. 哪些 Transformer invariances 允许插入 rotation 而不改变 full-precision function？
2. 为什么 Hadamard rotations 能够降低 outlier channels 的影响？
3. 哪些 rotations 可以 fuse 进相邻 weights，哪些必须保持 online？
4. GPTQ 在 pipeline 的什么位置介入？它使用什么 calibration data？
5. 哪项 measurement 能揭示 online rotations 在某个 device 上抵消了 W4A4 的收益？

## 讨论速记

- 需要解释的核心论点：更合适的 coordinate system 能够实现实用的 W4A4KV4 quantization。
- 需要绘制的示意图：一个经过 rotation 的 attention 和 MLP block，并区分 offline transforms 与 online transforms。
- 需要进行的对比：GPTQ 改变 quantized weights；QuaRot 在 quantization 前改变 basis。
- 应主动说明的局限：deployment result 依赖完整的 kernel 和 transform support。
