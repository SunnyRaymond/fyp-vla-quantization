# FP8 Formats for Deep Learning

> **Reading-list role**: Format-family support / Core reading-list item  
> **Verification**: `verified-full-text` + `official-technical-material`  
> **Recommended effort**: Core read  
> **Identity/status**: 这是 arXiv technical preprint，不是 OCP specification；E4M3/E5M2 的 normative interchange definition 以后续 OCP OFP8 Revision 1.0 为准。

> **Local files**: [paper.pdf](paper.pdf) · [OFP8 specification](specification.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Title | *FP8 Formats for Deep Learning* |
| Authors | Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, Hao Wu |
| Year / version | arXiv v2, 2022-09-29; v1 submitted 2022-09-12 |
| Venue / status | arXiv technical preprint, `arXiv:2209.05433`; multi-company authorship |
| Primary source | [arXiv abstract and full text](https://arxiv.org/abs/2209.05433) |
| Normative companion | [OCP 8-bit Floating Point Specification (OFP8), Revision 1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf) |
| Implementation reference | [NVIDIA Transformer Engine 2.18 FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) |

### Identity rule

读这篇时必须同时保留两层身份：

1. **Paper** 提出 E4M3/E5M2、给出 mixed-precision usage recommendation，并提供 training/PTQ evidence。
2. **OCP specification** 定义 bit-level interchange behavior。它明确不规定 application、arithmetic 或 scaling factor，所以不能把 NVIDIA 的 per-tensor scaling recipe 写成 OFP8 标准的一部分。

## 2. One-sentence takeaway

这篇论文用互补的 `E4M3` 与 `E5M2` 8-bit encodings，将 forward 的 precision need 与 backward 的 dynamic-range need 分开处理，并在作者测试的 CNN、RNN、Transformer 与最大 175B language model 上展示接近 16-bit baseline 的 result quality。

## 3. Background and prerequisites

读前需要掌握：

- IEEE 754 的 sign、biased exponent、mantissa/significand、normal、subnormal、NaN、Inf。
- mixed-precision training：storage precision、multiply input precision 与 accumulation precision 可以不同。
- quantization scale：format 的 finite range 不够覆盖 tensor range 时，需要把 tensor 映射进可表示范围。
- forward/backward 的分布差异：weights/activations 往往更需要 resolution，gradients 往往更需要 dynamic range。

核心表示式。对 normal finite value，可把 binary floating-point 写成：

\[
x=(-1)^s\,2^{e-b}\left(1+\frac{m}{2^M}\right),
\]

其中 `s` 是 sign bit，`e` 是 exponent field，`b` 是 exponent bias，`m` 是 mantissa integer，`M` 是 mantissa bit count。对 subnormal，implicit leading one 消失：

\[
x=(-1)^s\,2^{1-b}\left(\frac{m}{2^M}\right).
\]

Format 与 scaling recipe 要分开。一个常见 implementation abstraction 是：

\[
q=Q_{\mathrm{FP8}}(x/s_x),\qquad \hat{x}=s_x q,
\]

但 `s_x` 的 granularity、history、rounding 与 update rule 不属于 OFP8 interchange encoding。

## 4. Problem

- **Target setting**: deep-learning training 与 PTQ/inference。
- **Bottleneck**: FP16/BF16 仍占较高 memory bandwidth 与 Tensor Core cost；单一 FP8 encoding 很难同时给 forward tensors 足够 resolution、给 gradients 足够 dynamic range。
- **Why INT8 or one FP8 layout is insufficient**: fixed-point INT8 对 outlier-rich language-model tensors 可能不稳；若把 8 bits 全用于更大 exponent，mantissa resolution 不够，反之则 gradient overflow/underflow 风险增加。
- **Design objective**: 用最小的 non-IEEE deviation 构造两个互补 encoding，并让训练 recipe 在不重调大量 hyperparameters 的前提下接近 16-bit baseline。

## 5. Method

### 5.1 System view

```text
higher-precision tensor
        ↓ choose tensor scale / recipe
FP8 cast: E4M3 or E5M2
        ↓ Tensor Core GEMM / convolution
higher-precision accumulation and selected operations
        ↓
updated activation / gradient / weight state
```

Paper 的典型 hybrid assignment：

- forward weights and activations → `E4M3`；
- backward gradients → `E5M2`；
- accumulation、optimizer states、部分 reductions/nonlinearities → higher precision。

### 5.2 Core mechanism: two encodings

| Property | E4M3 | E5M2 |
|---|---:|---:|
| Sign / exponent / mantissa | 1 / 4 / 3 | 1 / 5 / 2 |
| Exponent bias | 7 | 15 |
| Max finite magnitude | 448 | 57,344 |
| Min positive normal | `2^-6` | `2^-14` |
| Min positive subnormal | `2^-9` | `2^-16` |
| Infinity | none | `±Inf` |
| NaN | one magnitude pattern, either sign | IEEE-style exponent-all-ones with non-zero mantissa |

`E4M3` 用多一位 mantissa 提高 resolution，并通过取消 infinity、压缩 NaN encodings 把 max finite 扩到 448。`E5M2` 保留 IEEE-like Inf/NaN，并用五个 exponent bits 覆盖 gradients 所需的更大 dynamic range。

### 5.3 Key Innovation

真正的新点不是“把 FP16 截到 8 bits”，而是：

1. 用 **two-format interchange family** 显式表达 precision–range trade-off；
2. 给出与 tensor role 对应的 hybrid training recipe；
3. 通过 large-model training 与 PTQ evidence 说明 FP8 不只适合 small vision models；
4. 设计被后续 OCP OFP8 specification 标准化为 interoperable element encodings。

Standard engineering choices 包括 per-tensor scaling、delayed scaling、higher-precision accumulation 与 kernel selection；它们决定实际稳定性和速度，但不是 E4M3/E5M2 bit layout 本身。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Evaluation scale reaches 175B parameters | Abstract states training experiments include language models up to **175B parameters** | Paper Abstract; search `175B` | 这是规模边界，不是单独的 accuracy metric。 |
| Tested tasks effectively match 16-bit result quality | Authors report comparable result quality across tested image/language tasks and CNN/RNN/Transformer architectures | Paper Abstract; task-specific result sections/tables | 不能扩写成“所有 FP8 training 都无损”；各 task metric 需逐表核对。 |
| E4M3/E5M2 cover complementary numeric ranges | max finite **448** vs **57,344**; different special-value policies | Paper format-definition section; OCP OFP8 encoding tables | 这是 representation property，不是 experiment result。 |
| Hopper FP8 can produce a large system speedup in one benchmark | NVIDIA reports **4.5×** on BERT high-accuracy in MLPerf Inference v2.1 | [NVIDIA 2022 FP8 blog](https://developer.nvidia.com/blog/?p=54825), “High-accuracy training and inference” | `VENDOR CLAIM` for one system submission; not a result of this paper and not an intrinsic 4.5× datatype speedup。 |

读 result table 时先问三件事：baseline 是 FP16 还是 BF16？metric direction 是 higher-better 还是 lower-better？FP8 覆盖了哪些 tensors/ops，而哪些仍是 higher precision？

## 7. Limitations

### Authors' stated limitations / evidence boundary

- v2 没有一个独立、完整的 “Limitations” section；authors 通过实验范围把结论限定在所测 architectures、tasks 与 recipes。
- Paper 主要论证 numerical feasibility，不给出一个可跨硬件复用的 end-to-end throughput model。
- OCP companion specification 只规范 interchange encoding，明确不覆盖 scaling-factor selection、arithmetic 或 application。

### My critique

- **Internal validity**: “matching 16-bit” 汇总了多种不同 metrics，必须逐 task 检查 variance、seed 与 tolerance；abstract-level conclusion 太容易掩盖小而系统性的 degradation。
- **External validity**: VLA 中 action head、proprioception、rare contact events 和 closed-loop error accumulation 未被直接测试。
- **Systems validity**: 8-bit storage、Tensor Core peak rate 与 end-to-end speedup 是三件事。scaling reductions、casts、transpose、communication 和 non-FP8 ops 都可能吃掉收益。
- **Reproducibility**: 大模型训练的 exact hardware、kernel、scale history 与 selective precision policy 比 bit layout 更影响复现；paper 不能替代 current implementation documentation。
- **Standard evolution**: paper 应按 arXiv v2 解释 empirical claims，bit-level statement 则优先引用 OCP Rev. 1.0，避免 version mixing。

## 8. Why it matters for this project

### VLA relevance

- VLA 的 vision encoder、language backbone 与 action decoder 可能具有不同 activation distributions；E4M3/E5M2 的 role separation 是设计 mixed-precision policy 的起点。
- Closed-loop control 对 rare but large errors 更敏感。offline benchmark 中微小平均 degradation，不保证 rollout success rate 不变。
- FP8 可降低 model/activation bandwidth，但 camera preprocessing、tokenization、KV cache、action postprocessing 与 environment latency 仍可能成为 bottleneck。
- 真正需要的 VLA experiment 是 per-component precision map：vision tokens、language tokens、action head、KV cache、gradients 各用什么 format，并测 success rate、control latency 与 tail error。

### Professor Li relevance

这篇把 model compression 与 software–hardware co-design 连接起来：format 只给 potential，scale recipe、kernel availability、memory traffic 和 target GPU generation 决定 realized latency/energy。汇报时应把 “numeric format efficiency” 与 “deployment efficiency” 分开。

## 9. How to read it

### 20-minute route

1. 读 Abstract，写下 E4M3 与 E5M2 各自解决的 tensor need。
2. 读 format-definition 部分，对照上面的 bit/range table。
3. 选一张 language-model result table，只核对一个 strongest claim。
4. 读 OCP spec 的 scope 与 encoding tables，标记哪些内容不属于 paper recipe。
5. 用一句话回答：为什么 E4M3 没有 infinity？

### 60–90-minute route

1. 手算 E4M3 的 max finite、min normal 与 min subnormal。
2. 画出 forward/backward mixed-precision dataflow，标出 accumulation 与 optimizer state precision。
3. 逐项核对一个 vision task 和一个 language task 的 baseline、metric、difference。
4. 比较 paper v2 与 OCP Rev. 1.0 的角色，避免把 normative 和 empirical evidence 混写。
5. 阅读 Transformer Engine primer 的 scaling discussion，区分 current/delayed scaling 与 element encoding。
6. 写下一个 VLA layer-wise FP8 ablation 方案。

## 10. Reading questions

1. E4M3 通过取消 infinity 换来的 range，在哪些 overflow-handling assumptions 下才安全？
2. 为什么 gradients 更偏好 E5M2，而 block scaling 出现后 MXFP8 又常能让 backward 使用 E4M3？
3. Paper 的 “unchanged hyperparameters” 是否足以证明 recipe 易迁移，还是只说明所选 tasks 不敏感？
4. 哪些 result 支持 element format，哪些 result 其实支持 scaling/implementation recipe？
5. 如果 VLA action head 保持 BF16、backbone 用 FP8，应该如何判断 latency gain 是否来自真正的 critical path？
6. Closed-loop success rate 对 quantization error 的 tolerance，可能与 language perplexity 有什么根本差异？

## 11. Weekly meeting card

- **Problem**: 一个 8-bit floating-point layout 难以同时满足 forward resolution 与 gradient dynamic range。
- **Key idea**: E4M3 给 forward precision，E5M2 给 backward range；selected operations/state 保持 higher precision。
- **Best evidence**: authors 在多类 architecture 上测试，并覆盖最大 175B language model，报告接近 16-bit result quality。
- **Biggest limitation**: numerical feasibility 不等于 VLA closed-loop robustness，也不等于 end-to-end speedup。
- **Question for the group**: 我们的 VLA 应按 tensor role 使用 E4M3/E5M2，还是直接采用 Blackwell 上的 MXFP8 block scaling？

## 12. Evidence boundary

- **Source claim**: bit layout/range 由 paper 和 OCP tables 支持；175B 与 tested-quality statement 来自 paper Abstract。
- **My interpretation**: FP8 对 VLA 的价值主要在 bandwidth/compute，但 action head 和 rare-event sensitivity 可能需要 selective precision。
- **Open question**: 目标 VLA、robot benchmark 与 hardware 上的 success-rate/latency trade-off 尚无直接 evidence。

## 13. Primary links

- [FP8 Formats for Deep Learning — arXiv:2209.05433](https://arxiv.org/abs/2209.05433)
- [OCP OFP8 Specification — Revision 1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf)
- [NVIDIA Transformer Engine 2.18 — Using FP8 and FP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [NVIDIA FP8 standardization technical blog — 2022-09-14](https://developer.nvidia.com/blog/?p=54825)

