# Microscaling Data Formats for Deep Learning

> **Reading-list role**: Format-family support for MXFP8 / MXFP4  
> **Verification**: `verified-full-text` + `official-technical-material`  
> **Recommended effort**: Core read  
> **Identity/status**: 主要 paper 是 MX 的 empirical evaluation；**OCP Microscaling Formats (MX) Specification v1.0 才是 normative format definition**。两者不能互换。

> **Local files**: [paper.pdf](paper.pdf) · [MX specification](specification.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Title | *Microscaling Data Formats for Deep Learning* |
| Authors | Bita Darvish Rouhani et al. (33 authors; multi-vendor collaboration) |
| Year / version | arXiv v3, 2023-10-19; v1 submitted 2023-10-16 |
| Venue / status | arXiv technical preprint, `arXiv:2310.10537` |
| Primary paper | [arXiv abstract and full text](https://arxiv.org/abs/2310.10537) |
| Normative companion | [OCP Microscaling Formats (MX) Specification v1.0, September 2023](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) |
| Current MXFP8 implementation reference | [NVIDIA Transformer Engine 2.18 MXFP8 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html) |
| Supplementary training evidence | [Recipes for Pre-training LLMs with MXFP8, arXiv:2506.08027v2](https://arxiv.org/abs/2506.08027) |
| Code / project | No standalone code artifact was verified in Phase 2; paper experiments include emulation, while current native MXFP8 behavior is documented by the vendor implementation above. |

### Paper vs specification

| Question | Authority |
|---|---|
| `MXFP8/MXFP4` 的 element type、block size、shared-scale encoding、special-value behavior 是什么？ | **OCP MX Specification v1.0** |
| 这些 formats 在 tested inference/training tasks 上表现如何？ | **Microscaling Data Formats for Deep Learning** |
| Blackwell/Transformer Engine 当前怎样 layout、quantize 与 support MXFP8？ | **NVIDIA Transformer Engine documentation** |
| Long-run LLM pretraining 怎样选择 MXFP8 conversion recipe？ | **Recipes for Pre-training LLMs with MXFP8** |

## 2. One-sentence takeaway

MX formats 通过给每 32 个 narrow elements 配一个 `E8M0` shared scale，把 per-tensor dynamic-range problem 缩小到 local block，并在 broad evaluation 中显示 MXFP8 通常比 direct-cast MXFP4 稳健得多，但 4-bit success 依赖 tensor role、mixed-format recipe 与模型分布。

## 3. Background and prerequisites

需要先理解：

- OFP8 `E4M3/E5M2` 与 FP4 `E2M1` element encodings。
- per-tensor scaling 的失败模式：一个 outlier 会迫使整个 tensor 使用较大 scale，压扁小值。
- block quantization、scale overhead、reduction dimension 与 matrix layout。
- PTQ direct cast 与 mixed-precision training recipe 的区别。

OCP 定义的 shared-scale block：

\[
v_i = X\,P_i,\qquad i=1,\ldots,k,
\]

其中 `P_i` 是 narrow element，`X` 是 shared `E8M0` scale，MXFP8/MXFP4 的 `k=32`。对非-NaN E8M0 code：

\[
X=2^{E-127}.
\]

`E8M0` 有 8 exponent bits、0 mantissa bits，只表示 powers of two；它没有 zero/Inf，并保留一个 NaN code `0xFF`。若 `X` 为 NaN，whole block represents NaN。

Scale storage overhead：

\[
\text{overhead per element}=\frac{8}{32}=0.25\text{ bits}.
\]

因此 MXFP4 nominal payload 是 `4 + 0.25 = 4.25 bits/element`，尚未计算 padding、metadata 与 layout overhead。

## 4. Problem

- **Target setting**: inference direct cast、mixed-precision training 与 hardware-friendly low-precision GEMM。
- **Bottleneck**: one scale per tensor 无法同时照顾 outliers 与 bulk values；更低 element precision 虽节省 bandwidth，却增加 saturation/rounding error。
- **Why previous methods are insufficient**: regular FP8 scaling granularity 太粗时 backward 可能需要 E5M2；naive E2M1 direct cast 在 MobileNetV2 和 LLaMA-7B 等模型上可能灾难性退化。
- **Design objective**: 在 fixed block size 与 compact scale encoding 下，让 hardware 能高吞吐处理，并把用户 recipe 改动控制在可接受范围。

## 5. Method

### 5.1 System view

```text
higher-precision tensor
      ↓ partition along GEMM reduction dimension
32-element blocks
      ↓ compute/encode one E8M0 scale per block
MX elements: FP8 E4M3/E5M2 or FP4 E2M1
      ↓ MX GEMM / emulated evaluation
higher-precision accumulation/output
```

在 current NVIDIA MXFP8 implementation 中，rowwise 与 columnwise tensors 分别量化：

- rowwise block: `1×32`；
- columnwise block: `32×1`；
- 两者必须从 higher-precision tensor 独立产生，不能只转置 quantized representation。

### 5.2 Core formats

| Property | MXFP8 | MXFP4 |
|---|---|---|
| Element | OCP FP8 `E4M3` or `E5M2` | FP4 `E2M1`, bias 1 |
| Block size | 32 | 32 |
| Shared scale | one `E8M0` per block | one `E8M0` per block |
| Element max | 448 for E4M3; 57,344 for E5M2 | 6 |
| Element Inf/NaN | inherited from selected OFP8 subtype | no element Inf/NaN; signed zero |
| Block NaN | E8M0 scale NaN makes whole block NaN | same |
| Exact non-negative E2M1 values | not applicable | `0, 0.5, 1, 1.5, 2, 3, 4, 6` |

OCP specification 不规定 exact scale-computation algorithm。evaluation paper 与 vendor recipe 对 rounding、amax handling、transpose/layout 的选择属于 method/implementation，不是 normative format field。

### 5.3 Key Innovation

1. **Microscaling abstraction**: narrow element + compact per-block shared scale，统一 FP8/FP6/FP4 families。
2. **Hardware-aware regularity**: fixed 32-element blocks 与 power-of-two E8M0 scale，减少 arbitrary-scale multiply complexity。
3. **Broad empirical map**: 不只报告成功案例，也暴露 naive MXFP4 direct-cast 的明显 failure cases。
4. **Mixed-format training**: sub-8-bit weights 可以与更高 MX activations/gradients 组合，而不必把所有 tensors 强行降到同一 bit-width。

## 6. Experiments and main results

### Direct-cast inference

| Model / metric | FP32 | MXFP8 E4M3 | MXFP8 E5M2 | MXFP4 | Locator | Caveat |
|---|---:|---:|---:|---:|---|---|
| BERT-Large reported task score | 93.47 | 93.42 | 93.32 | 90.97 | Paper direct-cast inference table; find row `BERT-Large` | task-specific score；不要与其他 rows 当成同一 metric 平均。 |
| DeiT-Small top-1 (%) | 80.54 | 79.83 | 79.00 | 71.35 | Same table; row `DeiT-Small` | 4-bit drop 明显。 |
| MobileNetV2 top-1 (%) | 72.14 | 65.74 | 53.50 | 0.25 | Same table; row `MobileNetV2` | 强 failure case，反驳 universal drop-in claim。 |
| LLaMA-7B WikiText perplexity | 9.488 | 9.768 (E4M3) | — | 27.201 | Paper LLM direct-cast table; row `LLaMA-7B` | perplexity lower is better；pure MXFP4 direct cast 不稳。 |

### Training and later MXFP8 evidence

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Sub-8-bit training can remain close to FP32 in one GPT setup | 1.5B GPT final loss: **2.74 FP32** vs **2.76** with MXFP4 weights + MXFP6-E3M2 activations/gradients | Paper training-results table; find `1.5B` and `2.76` | 这是 mixed MX precision，不是 W4A4G4，也不是 pure MXFP4。 |
| MXFP8 can support long-run pretraining with careful conversion | models up to **8B**, data up to **15T tokens** | *Recipes for Pre-training LLMs with MXFP8*, Abstract | NVIDIA-authored preprint；不是 independent replication。 |
| Long-run MXFP8 stays close to BF16 in reported experiment | validation-perplexity difference under **0.50%** | Recipes paper validation-perplexity discussion/figure; find `0.50%` | 依赖 specific conversion/rounding recipe。 |
| MoE experiment remains close to BF16 | **16B total / ~2.5B active**, **1T tokens**, final loss within **0.1%** | Recipes paper MoE subsection; find `0.1%` | model/data/recipe-specific。 |

Broad MX paper 使用 emulator，因此不能从这些 accuracy tables 推导 native Blackwell throughput。current Transformer Engine 2.18 只证明 MXFP8 有 Blackwell `SM100+` native path；它不把 paper accuracy 自动变成 end-to-end system speedup。

## 7. Limitations

### Authors' stated limitations / evidence boundary

- OCP spec 明确把 application、scale-computation algorithm 与部分 arithmetic choices 留在 scope 外。
- Evaluation paper 的 broad claim 是基于 emulator 与 selected benchmarks；它没有 native hardware end-to-end throughput evaluation。
- Training success 常使用 mixed formats，不应被压缩成“MXFP4 可以无损训练所有 tensors”。

### My critique

- **Internal validity**: 多任务表很有价值，但不同 metric 与 sensitivity 不能被一个平均 degradation 概括；MobileNetV2 failure 说明 distribution/model architecture 是关键 moderator。
- **External validity**: 没有 VLA、robot policy、closed-loop rollout 或 action error evaluation。
- **Systems validity**: scale computation、two orientations、padding 与 requantization 产生额外 memory/compute；emulation accuracy 不说明 native kernel 速度。
- **Reproducibility**: OCP intentionally leaves scale algorithm open，结果对 rounding choice 敏感；只写 “MXFP8” 不足以复现实验。
- **Conflict of evidence**: later MXFP8 long-run results来自 vendor-authored preprint，应视为 strong engineering evidence，但不能替代 independent replication。

## 8. Why it matters for this project

### VLA relevance

- Per-block scale 可能比 per-tensor FP8 更适合 multimodal activations，因为 image patches、language tokens 与 proprioceptive/action features 的 local ranges 不同。
- 但 block boundary 与 tensor layout 会影响 error：如果 32-element block 跨越 semantic groups，scale sharing 可能伤害 rare action features。
- MXFP4 direct-cast failure 提醒我们：weight-only、activation、gradient、KV cache 与 action head 必须分开做 precision assignment。
- VLA evaluation 应同时测 task success、action L2/error tails、control frequency、latency、VRAM 与 energy；只测 perplexity/top-1 不够。

### Professor Li relevance

这条线直接体现 software–hardware co-design：OCP format 定义 regular blocks，Blackwell Tensor Cores 规定可高效执行的 layout，training recipe 决定 accuracy。研究问题应从 “几 bits” 升级为 “哪种 granularity、layout 与 tensor-role allocation 在 target hardware 上得到真实 latency gain”。

## 9. How to read it

### 20-minute route

1. 先读 OCP spec scope 与 MX block/table，确认 paper 不是 normative authority。
2. 读 paper Abstract 与 method overview，写出 `v_i = X P_i`。
3. 看 direct-cast table 的 `BERT-Large`、`MobileNetV2`、`LLaMA-7B` 三行。
4. 看 1.5B GPT training row，圈出 MXFP4/MXFP6 的 tensor-role difference。
5. 写一句话解释为什么 “MXFP4 works” 是不完整命题。

### 60–90-minute route

1. 从 E8M0 encoding 推导 scale range 与每 element 0.25-bit overhead。
2. 对照 MXFP8、MXFP4 special values，解释 block-scale NaN 与 element NaN 的区别。
3. 逐行重算 direct-cast degradation 或 perplexity increase。
4. 读 paper training setup，标出 emulator、accumulation precision 与 tensor formats。
5. 读 later MXFP8 recipes paper，找出 rounding/conversion 选择为何会导致 divergence。
6. 对照 Transformer Engine rowwise/columnwise layout，画出一个 Linear layer 的 two-copy dataflow。

## 10. Reading questions

1. 为什么 scale 用 E8M0，而不是 E4M3 或 FP16？power-of-two constraint 带来什么 hardware benefit 与 quantization cost？
2. OCP 允许 MXFP8-E5M2，为什么 current NVIDIA recipe 默认两向都 E4M3？
3. MobileNetV2 的 catastrophic degradation 更可能来自 architecture、activation distribution，还是 scale algorithm？paper evidence 能否区分？
4. MXFP4 weights + MXFP6 activations/gradients 的成功，能支持多强的 “4-bit training” claim？
5. 对 VLA，block 应沿 hidden dimension、token dimension，还是 modality boundary 划分？
6. Native kernel peak rate 与 end-to-end speedup 之间，scale generation和two-orientation storage占多少比例？

## 11. Weekly meeting card

- **Problem**: per-tensor scaling 让 outliers 支配整个 tensor，降低 narrow-format resolution。
- **Key idea**: 每 32 elements 共享一个 E8M0 power-of-two scale，形成规则的 Microscaling block。
- **Best evidence**: MXFP8 在多项 direct-cast tasks 中接近 FP32；但 MobileNetV2 与 pure MXFP4 LLaMA-7B 提供重要 failure cases。
- **Biggest limitation**: broad paper 依赖 emulation；training example 多为 mixed formats，不能声称 universal native MXFP4 success。
- **Question for the group**: 我们是否应优先在 VLA backbone 上测 MXFP8，再把 MXFP4 限制为 weights，并让 action head 保持 higher precision？

## 12. Evidence boundary

- **Source claim**: block size、E8M0、E2M1 values 与 special behavior 来自 OCP spec；accuracy/loss numbers 来自 papers。
- **My interpretation**: MX 对 VLA 最重要的是 local scale granularity，而不是单纯从 8 bits 降到 4 bits。
- **Open question**: native Blackwell VLA workload 上的 end-to-end latency、power 与 closed-loop robustness 尚未被这些 sources 回答。

## 13. Primary links

- [Microscaling Data Formats for Deep Learning — arXiv:2310.10537](https://arxiv.org/abs/2310.10537)
- [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [Recipes for Pre-training LLMs with MXFP8 — arXiv:2506.08027](https://arxiv.org/abs/2506.08027)
- [NVIDIA Transformer Engine 2.18 — MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)
