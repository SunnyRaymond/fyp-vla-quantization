# Pretraining Large Language Models with NVFP4

> **Reading-list role**: Format-family support for NVFP4  
> **Verification**: `verified-full-text` + `official-technical-material`; performance evidence is vendor-authored  
> **Recommended effort**: Core read  
> **Identity/status**: NVIDIA-authored arXiv technical report + NVIDIA-defined format/recipe；不是 OCP/IEEE industry specification，也不是 “fully 4-bit end-to-end training”。

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Title | *Pretraining Large Language Models with NVFP4* |
| Authors | NVIDIA-authored multi-author technical-report team; use the full author list in the arXiv record when preparing a formal citation |
| Year / version | arXiv v2, revised 2026-03-04; v1 submitted 2025-09-29 |
| Venue / status | arXiv technical report, `arXiv:2509.25149`; vendor-authored, not peer-reviewed venue metadata in the verified packet |
| Primary source | [arXiv abstract and full text](https://arxiv.org/abs/2509.25149) |
| Defining implementation material | [NVIDIA Transformer Engine 2.18 NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html) |
| Inference companion | [NVIDIA: Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/) |
| Hardware scope | native NVIDIA Blackwell, compute capability `>=10.0`; Hopper supports FP8 but not native NVFP4 Tensor Core math |

### Identity rule

`NVFP4` 不是 “E2M1 的另一个名字”。完整 recipe 至少包括：

- E2M1 4-bit tensor elements；
- one E4M3 scale per 16-element microblock；
- one FP32 global scale per tensor；
- 1-D scaling for activations/gradients，2-D scaling for weights；
- stochastic rounding、Hadamard transforms 与 selective higher precision。

OCP MXFP4 同样使用 E2M1 elements，但它是 32-element block + E8M0 scale；不能把两者结果互换。

## 2. One-sentence takeaway

NVFP4 用 E2M1 elements、16-element E4M3 microblock scales 与 tensor-level FP32 scale 组成 two-level scaling，再配合 2-D weight scaling、stochastic rounding、Hadamard transforms 和 selective higher precision，使 12B model 的 10T-token pretraining 在报告中接近 FP8 accuracy，但并非所有 layers、operations 或 optimizer states 都运行在 4 bits。

## 3. Background and prerequisites

读前需要：

- FP4 `E2M1`：non-negative values 为 `0, 0.5, 1, 1.5, 2, 3, 4, 6`，无 element Inf/NaN。
- block scaling 与 global scaling 的不同职责。
- stochastic rounding 为什么能减少 repeated low-precision gradient updates 的 bias。
- Hadamard transform 怎样把 outlier-heavy distribution 旋转为更易 quantize 的形状。
- mixed-precision training 中 GEMM inputs、accumulation、optimizer state 与 sensitive layers 可以使用不同 precision。

核心 representation：

\[
\hat{x}=q_{\mathrm{E2M1}}\;s_{\mathrm{block,E4M3}}\;s_{\mathrm{global,FP32}}.
\]

这里 `q_E2M1` 提供 compact payload；`s_block` 在 local 16-element region 内保留 range/resolution；`s_global` 防止 E4M3 scale hierarchy 自身 range 不够。

## 4. Problem

- **Target setting**: large-language-model pretraining 与 low-precision inference。
- **Bottleneck**: BF16/FP8 仍消耗大量 matrix-multiply bandwidth、activation memory 与 compute；naive E2M1 quantization 对 outliers、transpose inconsistency 与 gradient bias 非常敏感。
- **Why MXFP4/direct FP4 is insufficient**: 32-element E8M0 scale 的 granularity 较粗，scale 又没有 mantissa；paper 的 ablation 显示 MXFP4 relative loss error 高于 NVFP4，并需要更多 tokens 才追上 compared loss。
- **Design objective**: 利用 Blackwell native FP4 throughput，同时把 accuracy gap 控制在可接受范围，并让 pretraining recipe 能扩展到 trillion-token regime。

## 5. Method

### 5.1 System view

```text
BF16 / higher-precision tensor
        ↓ optional random Hadamard transform
global FP32 normalization
        ↓
16-element local blocks → E4M3 block scale
        ↓
E2M1 quantization
        ↓ Blackwell NVFP4 GEMM
higher-precision accumulation / selected layers and ops
        ↓
training update (gradient cast uses stochastic rounding)
```

### 5.2 Core mechanism

#### Two-level scaling

- element: `E2M1`, 4 bits, max magnitude 6；
- microblock: one `E4M3` scale per **16 elements**；
- tensor: one `FP32` global scale。

相比 MXFP4 的 one E8M0 scale per 32 elements，NVFP4 用更细 block、带 mantissa 的 E4M3 scale，再用 FP32 global scale 补足 overall range。

#### Tensor-role-specific layout

- activations/gradients: 1-D `1×16` blocks；
- weights: one scale shared across a **16×16 2-D block**，再复制为 hardware-compatible 1×16 scale layout；
- reason: weight 与 weight transpose 都参与不同 GEMMs，单向 1-D quantization 会造成 transpose-dependent values，2-D scale 提供一致性。

#### Error-control recipe

1. **Stochastic rounding** for gradients，减少 quantization bias 累积。
2. **Random Hadamard transforms**，平滑 outliers，使 E2M1 resolution 更有效。
3. **Selective higher precision**，让最后若干 sensitive layers/operations 使用 MXFP8/BF16 等更高 precision。
4. **Higher-precision state/ops**，attention、embedding、nonlinearities、optimizer states 等并未全部变成 NVFP4。

### 5.3 Key Innovation

1. 把 FP4 accuracy 问题从单一 element format 转成 **hierarchical scaling + tensor-role layout + training recipe** 的联合设计。
2. 对 weights 使用 2-D scaling，直接处理 transpose inconsistency。
3. 将 stochastic rounding 与 Hadamard outlier smoothing 纳入 trillion-token pretraining recipe。
4. 用长程 pretraining evidence 展示 NVFP4 与更简单 MXFP4 recipe 的差异，而不只做短 calibration 或 inference benchmark。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Large-scale training reaches close downstream accuracy | 12B hybrid Mamba–Transformer, **10T tokens**: MMLU-Pro **62.62 FP8** vs **62.58 NVFP4** | Paper 12B downstream-evaluation table; find row `MMLU-Pro` | vendor-authored result；一个 benchmark row 不能代表所有 capabilities。 |
| Stable-phase loss remains close to baseline | validation relative-loss error stays below **1%** in stable phase and rises slightly above **1.5%** during decay | Paper 12B validation-loss figure/discussion; find `1.5%` | relative-error definition与phase boundary需按 paper 复述。 |
| NVFP4 outperforms MXFP4 in an 8B ablation | at **1T tokens**, about **1.5%** relative loss error for NVFP4 vs **2.5%** for MXFP4 | Paper 8B precision-ablation figure; find `2.5%` | comparison is recipe-specific, not proof every NVFP4 implementation wins。 |
| MXFP4 needs more tokens to match compared loss | reported **1.36T tokens**, or **36% more** than the 1T-token reference | Same 8B ablation discussion; find `1.36T` | token-equivalence is not automatically wall-clock or energy equivalence。 |
| Blackwell exposes much higher peak FP4 math rate | Table 1 lists NVFP4 peak ratios vs BF16 of **4× on GB200** and **6× on GB300** | Paper Table 1 | peak Tensor Core ratio，not end-to-end training speedup。 |
| Recipe is not fully 4-bit | about **16% of linear layers** in the 12B recipe remain high precision | Paper 12B model-recipe section/table; find `16%` | attention, embedding, nonlinearities, optimizer states 等也保持 BF16/FP32。 |
| Inference memory claim | NVIDIA blog reports **3.5×** model-memory reduction vs FP16 and **1.8×** vs FP8, with <**1%** degradation on selected LM tasks | NVIDIA inference blog, results/accuracy discussion | `VENDOR CLAIM`; selected models/tasks，且不是本 training report 的系统实验。 |

这些 numbers 应分成三类汇报：accuracy/loss、peak hardware capability、workflow/system result。把三类合成一个“NVFP4 既 6× 又无损”的 headline 会越过证据。

## 7. Limitations

### Authors' stated limitations / evidence boundary

- Report 明确以 algorithm/recipe 为中心，而不是 optimized runtime/system study。
- 12B recipe 保留约 16% linear layers in higher precision；最后 sensitive layers 需要 selective precision。
- attention、embeddings、nonlinearities、optimizer states 与其他 components 仍使用 BF16/FP32。
- 报告比较的是 specific model/data/training configuration，不给 universal convergence guarantee。

### My critique

- **Internal validity**: recipe 同时引入 finer scales、2-D scaling、stochastic rounding、Hadamard transforms 与 selective precision；需要逐组件 ablation 才能判断每项贡献及 interaction。
- **External validity**: evidence 聚焦 language pretraining，没有 VLA、vision encoder、action decoder 或 closed-loop control。
- **Systems validity**: Table 1 是 peak math ratio。scale generation、Hadamard transform、higher-precision layers、communication 与 optimizer cost 会降低 realized speedup。
- **Reproducibility**: native execution依赖 Blackwell 与 rapidly changing CUDA/PyTorch/Transformer Engine stack；software version 是 dated snapshot。
- **Conflict of interest**: format、implementation 与 evaluation 都来自 NVIDIA ecosystem。结果有工程价值，但仍需要 independent replication。
- **Claim precision**: “4-bit training” 容易误导；更准确的说法是 NVFP4-accelerated mixed-precision training。

## 8. Why it matters for this project

### VLA relevance

- NVFP4 可能显著压缩 large backbone 的 weights/linear activations，帮助 edge/robot GPU memory；但 vision stem、action head 与 control-critical layers 可能要保留 MXFP8/BF16。
- Hadamard transforms 对 multimodal outliers 可能有帮助，但会增加 kernel/layout complexity，需要量实际 latency。
- VLA 的 rare actions 与 contact transitions 对 tail error 敏感。MMLU-Pro 接近不代表 action distribution 或 success rate 接近。
- 2-D weight scaling 与 transpose consistency 对 training 很关键；对 inference-only policy，则应重新判断哪些 recipe components仍必要。

### Professor Li relevance

NVFP4 是典型 software–hardware co-design case：Blackwell Tensor Core capability、hierarchical scales、tensor layout 与 error-control algorithm 必须一起设计。适合用来训练一种研究习惯：不要从 peak FLOPS 直接推 energy/latency，也不要从 average accuracy 直接推 embodied robustness。

## 9. How to read it

### 20-minute route

1. 读 Abstract，确认 model size、token scale 与 baseline。
2. 看 format figure，写出 `E2M1 × E4M3 block scale × FP32 global scale`。
3. 看 Table 1，给 peak ratio 加上 “not end-to-end” 标签。
4. 找 12B MMLU-Pro row 与 8B MXFP4 ablation。
5. 找 selective higher-precision paragraph，解释为什么不是 fully 4-bit。

### 60–90-minute route

1. 对照 OCP MXFP4，列出 block size、scale type、global scale 三项差异。
2. 画 1-D activation scaling 与 16×16 weight scaling dataflow。
3. 阅读 stochastic rounding 与 Hadamard sections，分别写出它们解决的 error source。
4. 核对 8B/12B experiments 的 model、tokens、metric 与 baseline。
5. 把 accuracy results、peak hardware rates、vendor inference claims 分三张小表。
6. 设计 VLA selective-precision ablation：backbone NVFP4，action head/last layers MXFP8/BF16。

## 10. Reading questions

1. E4M3 block scale 相比 E8M0 增加 mantissa 后，hardware 与 storage cost 是什么？
2. 为什么 weights 需要 2-D scaling，而 activations/gradients 仍用 1-D blocks？
3. stochastic rounding 与 Hadamard transform 分别减少 bias 还是 outlier range？它们能互相替代吗？
4. 16% high-precision linear layers 对 final accuracy 和 realized throughput 各贡献多少？
5. 1.36T token-equivalence 能否转成更低 cost，还是 extra FP4 overhead/throughput 会改变结论？
6. VLA 中哪些 layers 应按 sensitivity 保持 BF16/MXFP8，如何用 closed-loop evidence 决定？

## 11. Weekly meeting card

- **Problem**: naive FP4 对 outliers、gradient bias 与 transpose layout 太敏感，难以稳定 long-run pretraining。
- **Key idea**: 16-element E4M3 block scale + FP32 global scale，再加 2-D weights、stochastic rounding、Hadamard transforms 与 selective precision。
- **Best evidence**: 12B/10T-token report 中 MMLU-Pro 62.62 FP8 vs 62.58 NVFP4；8B ablation 中 NVFP4 relative loss error 低于 MXFP4。
- **Biggest limitation**: vendor-authored、没有 end-to-end runtime study，而且相当多 layers/ops/states 仍是 higher precision。
- **Question for the group**: 对 VLA，NVFP4 应用于整个 backbone，还是只用于 latency-dominant linear layers并保留 action-critical path 的 higher precision？

## 12. Evidence boundary

- **Source claim**: equation components、block sizes、recipe 与 reported numbers 来自 NVIDIA report/docs。
- **My interpretation**: NVFP4 的核心不是 E2M1，而是 hierarchical scaling 与 selective precision 的 joint design。
- **Open question**: independent reproduction、VLA closed-loop accuracy 与真实 robot/edge energy gain 尚未确认。

## 13. Primary links

- [Pretraining Large Language Models with NVFP4 — arXiv:2509.25149](https://arxiv.org/abs/2509.25149)
- [NVIDIA Transformer Engine 2.18 — NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [NVIDIA FP8/FP4 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

