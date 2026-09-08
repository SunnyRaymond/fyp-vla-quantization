# 05 — HIGGS

## 基本信息

- 完整标题：*HIGGS: Pushing the Limits of Large Language Model Quantization via the Linearity Theorem*
- 作者：Vladimir Malinovskii、Andrei Panferov、Ivan Ilin、Han Guo、Peter Richtárik、Dan Alistarh
- 发表会议：NAACL 2025
- 本地来源：[官方 ACL Anthology PDF](paper-naacl-2025.pdf)
- 在本阅读路线中的定位：theory-guided data-free quantization 与 non-uniform bit allocation

## 一句话要点

HIGGS 证明 layer-wise squared quantization error 与 end-to-end perplexity increase 之间存在局部线性关系，并据此设计 Hadamard-rotated MSE-optimal grids，在给定 model-size budget 下全局分配各 layer 的 bitwidth。

## 真正的新意

较早的 methods 会最小化 layer proxy，再通过经验方式验证 end-to-end behavior。HIGGS 进一步追问这个 proxy 在什么条件下成立。它为每个 layer 分配一个 sensitivity coefficient，使 local MSE estimate 能够在 medium-error regime 内转化为对 global perplexity change 的近似。

## 方法脉络

1. 对用 quantized version 替换单个 layer 所产生的局部影响进行建模。
2. 在 smoothness 和 local-minimum assumptions 下，推导 layer MSE 与 perplexity increase 之间的 first-order relationship。
3. 应用 randomized Hadamard transforms，使 weight groups 近似 Gaussian 且具有 incoherent 特性。
4. 使用 MSE-optimal scalar 或 vector grids 进行 quantization，得到 data-free HIGGS representation。
5. 使用 synthetic noise 估计各 layer 的 sensitivity coefficients，再通过 dynamic programming 求解 fixed-size layer assignment。
6. 使用改造后的 FLUTE kernels 和 vLLM integration 执行受支持的 grid families。

## 重点检查的证据

- 在 Llama 3.1 8B、约 3.25 bits 条件下，随着 grid dimension 增加，fixed HIGGS variants 报告的 WikiText-2 perplexity 从 7.110 降至 6.643；dynamic data-free HIGGS 报告 6.388。FP16 为 5.607，3.25-bit GPTQ 为 7.133。
- 在约 4.0 bits 条件下，dynamic data-free HIGGS 报告 WikiText-2 perplexity 5.910、MMLU 63.86；GPTQ 则为 6.238 和 62.96。
- Model sweep 包括 Llama 3.2 1B/3B、Llama 3.1 8B/8B Instruct/70B，以及 Qwen 2.5 7B。
- 论文在受支持的 GPU-kernel configurations 中报告相对 FP16 的 2–3× speedup；在部分 batch/bitwidth cases 下，throughput 接近 3×。

不要混淆 Table 1 与 Table 3：前者是在特定 kernel conditions 下的 throughput，后者是 model quality。

## 重要限定

- Linear approximation 在约 3 bits 以上最可靠；bitwidth 更低、quantization error 增大时，它可能出现偏离。
- 并非每个理论上可用的 grid 都有高效 kernel。用于 fixed-bitwidth quality 的 configurations 与 runtime-supported dynamic configurations 并非同一集合。
- Dynamic “data-free” procedure 仍会使用 synthetic noise 进行 model evaluations；calibrated variant 使用 287k WikiText-2 training tokens。
- Hadamard transforms 若无法 fuse，可能增加 runtime overhead。
- 作者指出，对其他 architecture types（包括 Mixture-of-Experts）的验证仍然有限。

## 对你的 FYP 有何意义

HIGGS 为讨论 mixed precision 提供了有原则的方式：不要只根据 weight magnitude 分配 bits，而应测量 layer sensitivity，并求解一个 global resource allocation problem。它也表明，theory result 必须在其 local-error regime 之外接受检验。

## 阅读路线

### 30 分钟

阅读 Abstract、theorem statement、Algorithm 1、dynamic bitwidth section、Tables 1–3 和 Limitations。

### 120 分钟

写下每个 theorem assumption，追踪它如何产生 dynamic programming 所使用的 coefficient，并比较用于最佳 accuracy 的 grid set 与 FLUTE 支持的 subset。

## 阅读问题——阅读后自行回答

1. 在哪些 assumptions 下，perplexity increase 与 layer MSE 近似呈线性关系？
2. 当 quantization 超出 theorem 的 local regime 时，什么会失效？
3. 为什么 Hadamard-rotated weights 会引出 Gaussian MSE-optimal grids？
4. Dynamic data-free variant 究竟为何可以称为 data-free？
5. 哪些 grid configurations 有高效 kernels，哪些只存在于 quality experiments 中？
6. 如何扩展 bit-allocation problem，使其纳入 measured latency 或 energy，而不只考虑 model size？

## 讨论速记

- 需要解释的核心论点：在明确定义的 regime 内，local MSE 可以预测 global perplexity change。
- 需要复现的算法：bit budget 下的 layer-choice dynamic programming。
- 需要引用的结果：Llama 3.1 8B Table 3。
- 应主动说明的局限：theorem、quantizer 与 kernel support 各有不同的 applicability boundaries。
