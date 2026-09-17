# LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

> **Reading-list role**: BitVLA companion — paper cited for the bitsandbytes PTQ baselines  
> **Verification**: `verified-full-text`; local file is NeurIPS 2022 venue final  
> **Recommended effort**: **Targeted read**，重点区分 library、algorithm 与 BitVLA native quantization

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer |
| Venue | NeurIPS 2022 |
| Primary source | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html) · [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) |
| Software | [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) · [official documentation](https://huggingface.co/docs/bitsandbytes/index) |

## 2. First clarify the names

`bitsandbytes` 是 PyTorch low-bit software library，不是一种单独 quantization algorithm。Current library 主要包含：

- `LLM.int8()`：8-bit model inference；
- 4-bit `FP4/NF4` layers 与 QLoRA path；
- 8-bit optimizer states；
- quantized linear/embedding kernels 与 Hugging Face integrations。

BitVLA Sec. IV-B citation `[9]` 指向本篇 **LLM.int8()**。所以这是 paper text 最直接支持的 “original paper”；但它只正式定义 8-bit path，不能替 BitVLA Table II 的 4-bit configuration 补全缺失细节。

## 3. One-sentence takeaway

LLM.int8() 发现 billion-scale Transformer 的少数 systematic outlier feature dimensions 会破坏 naive INT8 quantization，于是用 vector-wise INT8 处理 99.9% 以上 values、把 outlier columns 拆到 FP16 matmul，再合并结果，从而把 model-weight memory 大约减半并保持 up to 175B model 的 measured quality。

## 4. Problem

- Naive per-tensor/per-row INT8 在 model scale 增大后出现 severe perplexity degradation。
- 根因不是 outlier values 数量多，而是它们集中在极少、跨 tokens/layers 反复出现的 hidden feature dimensions。
- 单一 absmax scale 为保留这些 extreme values 会浪费 regular values 的 quantization resolution。

## 5. Method

### 5.1 Vector-wise quantization

对于 `X @ W`，分别按 input rows 与 weight columns 计算 normalization constants，将大多数 values quantize 到 INT8；INT8 matmul 以 INT32 accumulate，再用 scale outer product dequantize。

### 5.2 Mixed-precision decomposition

定义 outlier feature set `O`，paper 使用 magnitude threshold `α=6.0`。矩阵乘法被拆成：

`XW ≈ Σ_(h∈O) X_h^FP16 W_h^FP16 + S · Σ_(h∉O) X_h^INT8 W_h^INT8`

在 tested models 中，outlier set 最多约 7 个 feature dimensions；超过 99.9% values 仍走 INT8 path。最后 FP16 outlier result 与 dequantized regular result 相加。

### 5.3 What it is not

- 不是 pure all-INT8 compute；存在 FP16 outlier path、scales、dequantization 与 FP16 output accumulation。
- 不是 QAT；checkpoint 可在 load time quantize，目标是 inference。
- 不是 4-bit NF4/FP4 的理论来源；那条路线主要由 later QLoRA work 支撑。

## 6. Main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Naive INT8 scaling breaks at large scale | 13B C4 perplexity：FP32 12.45；vector-wise INT8 16.48 | Table 1, PDF p. 6 | specific model/data family |
| Mixed precision recovers quality | 13B LLM.int8 absmax perplexity 12.45，matching FP32 12.45 | Table 1, p. 6 | “no degradation” is measured-task claim, not universal guarantee |
| Scales to very large models | OPT models up to 175B retain reported zero-shot accuracy trend | Fig. 1 / Sec. 3.4 | runtime/hardware implementation matters |
| Memory reduction enables access | 8-bit lets larger models fit on listed GPU configurations | Table 2, p. 10 | model fit is not the same as end-to-end speedup |

Paper notes small models can be slower than FP16 because quantization/outlier handling overhead dominates；runtime benefit is scale- and kernel-dependent。

## 7. How BitVLA uses bitsandbytes

BitVLA does **not** use bitsandbytes for its native W1.58A8 model. Its two paths are：

| Path | Purpose | Precision / runtime |
|---|---|---|
| BitVLA native model | proposed model | ternary weights + INT8 activations；custom BitBLAS kernel |
| OpenVLA/OpenVLA-OFT baselines | post-hoc comparison | backbone PTQ to “INT8/INT4” with bitsandbytes |

Sec. IV-B says public fine-tuned checkpoints are loaded and their backbones quantized with bitsandbytes, then LIBERO is evaluated. This is a convenient PTQ baseline, not a matched native-low-bit training baseline。

### 7.1 Important reproducibility gap

The paper reports an `INT4` row but does not specify：

- `FP4` vs `NF4` quantization type；
- group/block size；
- compute dtype；
- double/nested quantization；
- modules skipped or kept high precision；
- exact bitsandbytes/Transformers versions。

Current `BitsAndBytesConfig(load_in_4bit=True)` exposes both FP4/NF4-related options；therefore “INT4” in BitVLA should not be treated as a fully specified numeric format. Reproducing Table II requires code/config beyond the paper text。

Similarly，bitsandbytes `load_in_8bit=True` usually means LLM.int8-style mixed precision，not every operation and activation being plain INT8。

## 8. Why it matters for fair quantization evaluation

- PTQ library choice changes not only bit-width，also scale granularity、outlier handling、module coverage 与 kernel path。
- BitVLA 与 bnb-quantized OpenVLA-OFT 的 comparison simultaneously changes parameter count、pretraining data、native/QAT vs PTQ、vision backbone、action mask/head 与 kernels。
- Table II supports “BitVLA has a strong memory/accuracy trade-off”，but does not isolate “1.58-bit is intrinsically better than 4-bit”。

## 9. Limitations

- LLM.int8 paper studies inference, not fine-tuning/training；attention operation is not quantized。
- Primary experiments are language models/perplexity/zero-shot tasks, not VLM/VLA closed-loop control。
- Mixed FP16 outlier path makes speedup dependent on hardware, model size, sequence shape and kernel fusion。
- The original paper does not cover modern 4-bit NF4/QLoRA behavior；bitsandbytes library scope has grown beyond this paper。

## 10. How to read it

### 20-minute route

1. Fig. 2（PDF p. 3）：画出 regular INT8 path 与 outlier FP16 path。
2. Sec. 3.2（p. 5）：理解 outliers 位于 feature dimensions 而不是 random values。
3. Table 1（p. 6）：比较 naive vector-wise 与 +decomposition。
4. Sec. 6（p. 10）：读 inference-only、attention-not-quantized limitations。
5. 回到 BitVLA Table II：列出 paper 未报告的 bnb config。

### 60-minute route

1. 手算 absmax quantization 与 dequantization。
2. 解释为什么 row-wise scales 仍无法保护 column-wise systematic outliers。
3. 对照 SmoothQuant：LLM.int8 将 outliers split 到 FP16；SmoothQuant 重新分配 activation/weight difficulty。
4. 为 BitVLA baseline 写出完整 reproducibility checklist。

## 11. Reading questions

1. 若 outlier feature path 比例随 VLM image tokens 增加，LLM.int8 latency 会怎样变化？
2. BitVLA Table II 的 4-bit row 若是 FP4 与 NF4，结果可能有多大差异？
3. Quantize only backbone 而保留 vision connector/action head high precision，memory 与 success 分别由谁主导？
4. 为什么 model memory reduction 不必然等于 action-query latency reduction？

## 12. Weekly meeting card

- **Problem**：large Transformer 中 naive INT8 被 systematic outlier features 破坏。
- **Key idea**：vector-wise INT8 for regular values + FP16 decomposition for outlier dimensions。
- **Best evidence**：13B C4 perplexity recovered from 16.48 to 12.45，matching FP32 12.45（Table 1）。
- **Biggest limitation**：mixed precision、inference-only、language-model evidence；speedup 依赖 kernels/hardware。
- **Connection to BitVLA**：这是 bitsandbytes INT8 baseline 的 cited source，不是 BitVLA native W1.58A8 method。

## 13. Evidence boundary

- **Source claims**：LLM.int8 algorithm/results/limitations 来自 NeurIPS final。
- **Current software scope**：bitsandbytes feature map 来自 official documentation；library capabilities exceed this single paper。
- **My critique**：BitVLA “INT4” baseline configuration is underspecified in full text。
- **Primary links**：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html) · [arXiv](https://arxiv.org/abs/2208.07339) · [bitsandbytes docs](https://huggingface.co/docs/bitsandbytes/index)
