# Phase-2 Evidence Packet: Numeric Formats and Researcher Watchlist

> Scope: `ReadingList.md` 中的 **FP8 Formats for Deep Learning**、**MXFP8 / MXFP4 / NVFP8 / NVFP4**，以及 **MIT Prof. Song Han**、**ISTA Prof. Dan Alistarh**。  
> Snapshot date: **2026-08-22 (Asia/Singapore)**。  
> 本文件是 Phase-2 source-verification packet，不是最终 paper README。术语保留 English。任何 performance number 都附到可定位的 primary/official source；vendor-authored results 不当作 independent validation。

## 1. Verification legend

| Status | Meaning |
|---|---|
| `VERIFIED — primary full text` | 已检查 specification、paper full text 或 official technical documentation。 |
| `VERIFIED — official metadata/profile` | 身份、venue 或 publication metadata 由学校、venue、作者主页核验。 |
| `VENDOR CLAIM` | 数字来自 vendor blog、vendor model card 或 vendor-authored report；需保留 benchmark context。 |
| `UNRESOLVED LABEL` | official materials 使用了名称，但没有找到足以定义独立 datatype/format 的 bit-level primary document。 |

## 2. Executive classification: 不要把 format name 全部当成 paper

| Reading-list item | 它实际上是什么 | 最权威的 defining primary document | 是否应建立 “paper README” |
|---|---|---|---|
| **FP8 Formats for Deep Learning** | 一篇 industry-authored arXiv technical preprint；提出 E4M3/E5M2 interchange encodings | [Micikevicius et al., arXiv:2209.05433v2 (2022-09-29)](https://arxiv.org/abs/2209.05433)；正式 interchange definition 应同时看 [OCP OFP8 Specification, Revision 1.0](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf) | **是**。一个 README 以 paper 为主，OCP spec 作 normative companion。 |
| **MXFP8 / MXFP4** | 两个 concrete Microscaling formats，不是两篇同名论文 | [OCP Microscaling Formats (MX) Specification v1.0, September 2023](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | **不是一名一篇**。建议一个 MX-family specification note，再配 evaluation paper。 |
| **NVFP4** | NVIDIA-defined format + training/inference recipe；不是开放 industry specification | [NVIDIA Transformer Engine 2.18 NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)；训练 evidence 为 [Pretraining Large Language Models with NVFP4, arXiv:2509.25149v2 (2026-03-04)](https://arxiv.org/abs/2509.25149) | **是，但题名应按 technical report**，并清楚标为 vendor-defined/vendor-authored。 |
| **NVFP8** | official NVIDIA pages 中出现的 optimization/checkpoint label；目前未核验到独立 numeric-format specification | [NVIDIA GeForce CES 2026 product article](https://www.nvidia.com/en-us/geforce/news/gfecnt/20261/rtx-ai-garage-ces-2026-open-models-video-generation/) 与 [NVIDIA Technical Blog](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs) 只描述 product/checkpoint usage；没有定义独立 layout | **否**。只建一个 short clarification note；不能杜撰 “NVFP8 paper”。 |
| **Song Han / Dan Alistarh** | researcher watchlist，不是老师明确点名的 core papers | official institutional profile + official venue/publication records | 只把选中的 canonical papers 建成 **optional** README；不计入 core paper count。 |

## 3. Format comparison matrix

| Format | Element encoding | Scale granularity / block | Scale encoding | Special values | Defining authority | NVIDIA native/support snapshot | Typical use |
|---|---|---|---|---|---|---|---|
| **OFP8 E4M3** | `S1 E4 M3`, bias 7 | interchange format 本身**不定义 scaling**；NVIDIA “regular FP8” recipe 常用 one FP32 scale per tensor | implementation-specific | signed zero；finite max ±448；无 infinity；仅 `S.1111.111` 为 NaN | OCP OFP8 Rev. 1.0 | Hopper H100 introduced FP8 Tensor Core support；Blackwell continues support | training forward weights/activations；PTQ/inference |
| **OFP8 E5M2** | `S1 E5 M2`, bias 15 | 同上 | implementation-specific | IEEE-style ±Inf；多种 NaN；finite max ±57,344 | OCP OFP8 Rev. 1.0 | Hopper and later | training backward/gradients；需要更大 dynamic range 的 tensor |
| **MXFP8** | OCP 允许 FP8 `E4M3` **或** `E5M2` element | 1-D block of **32** elements；current NVIDIA recipe: `1×32` rowwise / `32×1` columnwise, independently quantized | one `E8M0` 8-bit power-of-two scale per block | FP8 element specials 继承 OFP8 subtype；若 block scale 为 E8M0 NaN，则 whole block represents NaN | OCP MX v1.0 | native Blackwell `SM100+` in Transformer Engine 2.18；default E4M3 both passes, optional hybrid E4M3/E5M2 | training and inference；blockwise scaling |
| **MXFP4** | `S1 E2 M1`, bias 1 | 1-D block of **32** elements | one `E8M0` scale per block | element encoding has signed zero, **no element Inf/NaN**；max ±6；min normal ±1；subnormal ±0.5；scale NaN can make block NaN | OCP MX v1.0 | OCP spec hardware-agnostic；NVIDIA NVFP4 report lists Blackwell native MXFP4 Tensor Core peak capability, but Transformer Engine 2.18 primer exposes MXFP8/NVFP4 recipes rather than promising an MXFP4 training recipe | inference evaluation；training evidence commonly mixes MXFP4 weights with higher MX formats for other tensors |
| **NVFP4** | `S1 E2 M1` values | **16-element microblock** for activations/gradients；training recipe uses **16×16 2-D scaling** for weights | one FP8 `E4M3` scale per microblock + one FP32 global scale per tensor | E2M1 element has signed zero, no element Inf/NaN；NVFP4 overall special-value propagation is vendor recipe, not a separate open standard | NVIDIA documentation/report | native NVIDIA Blackwell, compute capability `>=10.0` | inference and mixed-precision pretraining; sensitive layers/operations remain BF16/FP32 |
| **NVFP8** | **not independently defined**；某些 NVIDIA FP8 model cards use E4M3, but this cannot be generalized to every item marketed as “NVFP8” | **not specified under this label** | **not specified under this label** | **not specified under this label** | no bit-level primary spec found | official pages advertise RTX/ComfyUI checkpoints/optimizations, not a new Tensor Core datatype | observed official usage is inference/checkpoint optimization |

### Important distinction: element format ≠ scaling recipe

- `E4M3`, `E5M2`, `E2M1` describe an **element encoding**.
- `OFP8` is an **interchange format**; its OCP scope explicitly excludes application, arithmetic and scaling-factor choices.
- `MXFP8`/`MXFP4` add an OCP-standardized **shared-scale block structure**.
- `NVFP4` adds a NVIDIA-specific **two-level scaling recipe** and training practices. Calling it merely “E2M1” loses the most important part of the format.
- `NVFP8` cannot currently be placed in this hierarchy as a distinct format because the necessary definition is absent.

## 4. FP8: paper proposal and normative OCP specification

### Source passport

1. **Proposal/evidence paper:** Paulius Micikevicius et al., *FP8 Formats for Deep Learning*, arXiv:2209.05433v2, revised **2022-09-29**.  
   Source type: arXiv technical preprint, multi-company authors.  
   Status: `VERIFIED — primary full text`.
2. **Normative definition:** *8-bit Numerical Formats for Deep Neural Networks (OFP8)*, OCP Specification **Revision 1.0**, submitted **2023-05-26**, approved **2023-06-20**; revision history records later correction of exponent biases on **2023-12-01**.  
   Source type: industry specification.  
   Status: `VERIFIED — primary full text`.
3. **Implementation/hardware snapshot:** [NVIDIA Transformer Engine 2.18 FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).  
   Source type: vendor technical documentation, version **2.18.0**.  
   Status: `VERIFIED — primary full text`.

### What the paper actually contributes

- 提出两个 8-bit binary interchange encodings: `E4M3` and `E5M2`。
- 推荐 training 中 forward weights/activations 用 `E4M3`，backward gradients 用 `E5M2`；这是 numerical recipe，不等于所有 implementation 必须如此。
- `E4M3` 牺牲 infinity 和大部分 NaN code points 换更大 finite range；`E5M2` 保留 IEEE-like special-value behavior。
- 论文覆盖 CNN、RNN、Transformer；training experiment 包含最大 **175B parameters** 的 language model，并包含 PTQ experiments。

### Bit-level facts to retain

| Property | E4M3 | E5M2 |
|---|---:|---:|
| Sign / exponent / mantissa | 1 / 4 / 3 | 1 / 5 / 2 |
| Exponent bias | 7 | 15 |
| Max finite magnitude | 448 | 57,344 |
| Min positive normal | 2^-6 | 2^-14 |
| Min positive subnormal | 2^-9 | 2^-16 |
| Infinity | none | ±Inf |
| NaN | one magnitude pattern with either sign | IEEE-style exponent-all-ones with non-zero mantissa |

### Claims that are safe to quote

- Paper abstract: across its tested image/language tasks, FP8 “effectively” matches 16-bit result quality; exact model-by-model comparison must be taken from the individual paper tables, not generalized into a universal no-loss claim.
- Paper scale: training models up to **175B parameters**.
- NVIDIA’s 2022 blog separately reported **4.5×** speedup for Hopper on BERT high-accuracy in MLPerf Inference v2.1. This is a **vendor benchmark for a specific system submission**, not a throughput result established by the FP8 paper and not an intrinsic 4.5× property of the datatype. Source: [NVIDIA FP8 standardization blog, 2022-09-14](https://developer.nvidia.com/blog/?p=54825).

### Hardware/use boundary

- NVIDIA H100 introduced native FP8 Tensor Core support; NVIDIA documentation uses FP8 for both training and inference.
- “8-bit storage” does **not** mean the whole training state is 8-bit: accumulation, optimizer states, nonlinearities and some layers commonly remain higher precision.
- OFP8 spec deliberately does not define a scale granularity. NVIDIA’s regular FP8 recipe commonly uses one FP32 scale per tensor, but that is implementation behavior, not an OFP8 encoding field.

## 5. MXFP8 and MXFP4: one industry specification, then evaluation papers

### Defining source passport

**Open Compute Project, *Microscaling Formats (MX) Specification v1.0*, September 2023.**  
Source type: industry specification; hardware/software agnostic.  
Status: `VERIFIED — primary full text`.

The spec defines concrete 8-, 6- and 4-bit MX formats and basic operations. It explicitly leaves applications, scale-computation algorithms and some implementation decisions out of scope. Therefore, a paper that evaluates MX is not the defining authority for the bit layout.

### Shared MX block model

An MX block represents values `v_i = X × P_i`, where:

- `X` is one shared scale;
- `P_i` are `k` same-type element encodings;
- `k = 32` for both MXFP8 and MXFP4 in OCP v1.0;
- `X` is `E8M0`: 8 exponent bits, 0 mantissa bits, bias 127, power-of-two values, no zero/Inf, one NaN code (`0xFF`); exponent range is -127 to +127;
- if `X` is NaN, the entire block represents NaN. Element Inf/NaN propagates when the element format supports it.

### MXFP8

- Element can be OFP8 `E4M3` or `E5M2`; the OCP spec does **not** restrict MXFP8 to E4M3.
- Current NVIDIA Transformer Engine 2.18 implementation defaults to E4M3 for forward and backward, supports hybrid E4M3-forward/E5M2-backward, and does not expose pure-E5M2 training.
- Scale granularity is one E8M0 per 32 consecutive values. NVIDIA Blackwell hardware requires scale blocks along the reduction dimension, so rowwise `1×32` and columnwise `32×1` copies are quantized independently from higher-precision input.
- Native Transformer Engine path requires **Blackwell SM100 or later**.

### MXFP4

- Element is `E2M1`: one sign, two exponent and one mantissa bit, bias 1.
- Exactly representable non-negative values are `0, 0.5, 1, 1.5, 2, 3, 4, 6` (and sign-reflected negatives).
- No element NaN or Inf. The format has signed zero; max magnitude 6; min positive normal 1; the only positive subnormal is 0.5.
- One E8M0 scale per 32 E2M1 elements. The nominal storage is 4 bits/element plus 8/32 = **0.25 scale bits per element** before padding/metadata/layout overhead.

### Evaluation paper 1: broad MX feasibility

**Bita Darvish Rouhani et al., *Microscaling Data Formats for Deep Learning*, arXiv:2310.10537v3, 2023-10-19.**  
Source type: arXiv technical preprint; multi-vendor authors.  
Status: `VERIFIED — primary full text`.

Useful table-level evidence, not universal claims:

| Task/model | FP32 | MXFP8 E4M3 | MXFP8 E5M2 | MXFP4 | Interpretation |
|---|---:|---:|---:|---:|---|
| BERT-Large task score reported in paper | 93.47 | 93.42 | 93.32 | 90.97 | MXFP8 direct cast is close in this case; MXFP4 drops more. |
| DeiT-Small top-1 (%) | 80.54 | 79.83 | 79.00 | 71.35 | direct-cast degradation grows at 4-bit. |
| MobileNetV2 top-1 (%) | 72.14 | 65.74 | 53.50 | 0.25 | strong counterexample: a “drop-in” 4-bit claim must not be universalized. |
| LLaMA-7B WikiText perplexity | 9.488 | 9.768 (E4M3) | — | 27.201 | pure direct-cast MXFP4 is not robust here. |

Training evidence in the same paper is **mixed MX precision**, not “everything MXFP4”: for a 1.5B GPT experiment, FP32 final loss is **2.74**, while MXFP4 weights with MXFP6-E3M2 activations/gradients reaches **2.76**. The experiments use emulation on existing GPUs; they do not establish hardware end-to-end throughput for native MXFP4.

### Evaluation paper 2: current MXFP8 pretraining recipe

**Asit Mishra et al., *Recipes for Pre-training LLMs with MXFP8*, arXiv:2506.08027v2, revised 2025-08-18.**  
Source type: NVIDIA-authored arXiv technical preprint.  
Status: `VERIFIED — primary full text`; performance/accuracy remains vendor-authored evidence.

- Studies models up to **8B parameters** and datasets up to **15T tokens**.
- Reported MXFP8-vs-BF16 validation-perplexity difference stays under **0.50%** in the long-run experiment.
- For a **16B total / ~2.5B active** MoE trained on **1T tokens**, final loss is reported within **0.1%** of BF16.
- The important methodological lesson is scale computation/conversion: an apparently minor rounding choice can cause divergence. “Same element encoding” is insufficient to reproduce training.

## 6. NVFP4: vendor format/recipe with a technical report

### Source passport

1. [NVIDIA Transformer Engine 2.18 NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html).  
   Source type: vendor technical documentation, version 2.18.0.  
   Status: `VERIFIED — primary full text`.
2. **NVIDIA et al., *Pretraining Large Language Models with NVFP4*, arXiv:2509.25149v2**, revised **2026-03-04**.  
   Source type: vendor-authored arXiv technical report, not an open industry standard.  
   Status: `VERIFIED — primary full text`.
3. [Introducing NVFP4 for Efficient and Accurate Low-Precision Inference](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/), **2025-06-24**.  
   Source type: vendor technical blog for inference.  
   Status: `VENDOR CLAIM`.

### Defining structure

For a tensor element:

`x = x_E2M1 × s_block_E4M3 × s_global_FP32`

- `x_E2M1`: 4-bit E2M1 element, max magnitude 6, no element Inf/NaN.
- `s_block_E4M3`: one FP8 E4M3 scale per **16 elements**.
- `s_global_FP32`: one FP32 scale per tensor to extend overall dynamic range.
- Training recipe: activations/gradients use 1-D 16-element blocks; weights use a **16×16 2-D scale** and replicate it into hardware-compatible 1×16 blocks.
- Accuracy recipe also uses stochastic rounding for gradients, random Hadamard transforms for outliers, and keeps sensitive final layers/higher-level operations in higher precision.

### Hardware and software boundary

- Native support is a **NVIDIA Blackwell (compute capability >=10.0)** feature. Hopper supports OFP8, not native NVFP4 Tensor Core math.
- The arXiv report’s Table 1 lists theoretical/native Tensor Core peak speedup relative to BF16 of **4× on GB200** and **6× on GB300** for NVFP4 (and MXFP4). These are peak math-rate ratios, **not end-to-end training speedups**.
- Current software constraints can change. Treat CUDA/PyTorch/Transformer Engine version requirements as a dated implementation snapshot, not part of the numeric format.

### Locatable training and inference claims

- 12B hybrid Mamba-Transformer, **10T tokens**: MMLU-Pro **62.62 (FP8)** vs **62.58 (NVFP4)** in the report.
- Validation relative-loss error stays below **1%** during the stable phase and rises slightly above **1.5%** in the decay phase.
- 8B, 1T-token ablation: MXFP4 has about **2.5%** relative loss error versus about **1.5%** for NVFP4; MXFP4 is reported to need **1.36T tokens**, i.e. **36% more tokens**, to reach the compared loss.
- Critical caveat: in the 12B recipe, about **16% of linear layers** remain high precision; attention, embeddings, nonlinearities, optimizer states and other components also remain BF16/FP32. Do not call this “fully 4-bit training.”
- Inference blog claims **3.5× model-memory reduction vs FP16** and **1.8× vs FP8**, with less than **1%** degradation on selected language-model tasks. These are `VENDOR CLAIM`, model/task dependent, and the ratio includes scale/metadata effects rather than idealized 4-bit arithmetic alone.
- The training report explicitly focuses on algorithms rather than optimized runtime/system measurement; it should not be cited as an end-to-end throughput benchmark.

## 7. NVFP8: unresolved product/checkpoint label, not a verified distinct datatype

### What was verified

- NVIDIA official CES 2026 and technical-blog materials use “NVFP8” for ComfyUI/video-generation checkpoints and optimizations.
- The same technical blog’s implementation bullets describe **FP8** fused quantization/dequantization kernels, and linked NVIDIA model artifacts are commonly named `...-FP8`, not accompanied by an NVFP8 bit-level specification.
- No official OCP/NVIDIA specification, CUDA datatype definition, Transformer Engine recipe, or arXiv paper was found that independently defines all of: E/M layout, bias, block size, scale hierarchy, special-value behavior, and arithmetic rules for a distinct “NVFP8” format.

### Therefore

- Classification: `UNRESOLVED LABEL` — currently best treated as a NVIDIA-optimized FP8 **checkpoint/product label**, not a proven new numeric format alongside OFP8/MXFP8/NVFP4.
- Do **not** copy E4M3, per-tensor scaling or a block size into a generic NVFP8 table unless the specific checkpoint’s model card states them.
- Example of checkpoint-specific evidence: [NVIDIA Llama-3.3-70B-Instruct-FP8 model card](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP8) says its linear weights/activations use E4M3 PTQ and reports approximately **50%** disk/GPU-memory reduction versus 16-bit. It reports MMLU **83.3 BF16 / 83.2 FP8**, GSM8K-CoT **95.3 / 94.3**, ARC-Challenge **93.7 / 93.2**, and IFEval **92.1 / 92.2**. This verifies that **that checkpoint** is E4M3 FP8; it does not define “NVFP8” globally.
- NVIDIA CES material reports **2× performance** and **40% VRAM reduction** for the named NVFP8 ComfyUI workflow. This is a workflow-level `VENDOR CLAIM`, not an intrinsic datatype ratio.

### Clarification question for advisor

> “NVFP8” 是否指 NVIDIA-optimized standard FP8/E4M3 checkpoint（通常 per-tensor scaling），还是一份尚未附上的 NVIDIA internal/product format document？

Until clarified, create only a short `NVFP8-identity-and-evidence-gap` note, not a paper README and not a core-paper count entry.

## 8. Researcher watchlist — optional, not core count

## 8.1 MIT Prof. Song Han

### Identity resolution

- Official [HAN Lab profile](https://hanlab.mit.edu/songhan) identifies **Song Han**, Associate Professor with tenure at **MIT EECS**, HAN Lab PI.
- The official profile explicitly groups Deep Compression, EIE, Once-for-All, SmoothQuant and AWQ under his efficient AI / model compression trajectory.
- Status: `VERIFIED — official metadata/profile`.
- Identity ambiguity: low. “Song Han” is a common name, but the reading-list qualifier “MIT Prof.” and official MIT/HAN Lab page resolve the intended researcher.

### Canonical 5-paper watchlist

| Paper | Venue/version | Why canonical here | Reading-list status |
|---|---|---|---|
| [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) | ICLR 2016; arXiv v5, 2016-02-15 | foundational three-stage compression pipeline; connects algorithmic compression to storage and energy | **optional new note** |
| [EIE: Efficient Inference Engine on Compressed Deep Neural Network](https://research.nvidia.com/publication/2016-06_eie-efficient-inference-engine-compressed-deep-neural-network) | ISCA 2016 | shows why compressed sparse models need co-designed hardware/dataflow to obtain real speed/energy gains | **optional new note** |
| [Once-for-All: Train One Network and Specialize it for Efficient Deployment](https://openreview.net/pdf?id=HylxE1HKwS) | ICLR 2020 | hardware-aware specialization; useful bridge from compression to heterogeneous edge deployment | **optional new note** |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html) | ICML 2023 | canonical W8A8 activation-smoothing PTQ | **already core elsewhere; cross-link only** |
| [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) | MLSys 2024 Best Paper | canonical weight-only, activation-aware LLM PTQ/system | **already core elsewhere; cross-link only** |

`HAQ` is also already explicitly named in the core reading list. The clean follow line is:

`Deep Compression → EIE → HAQ [core] → SmoothQuant [core] → AWQ [core]`

Follow **algorithm–hardware co-design for deployable quantization**: distinguish nominal bit-width/FLOPs from measured latency, memory traffic and energy. `Once-for-All` is a useful side branch if the project emphasizes heterogeneous edge devices.

## 8.2 ISTA Prof. Dan Alistarh

### Identity resolution

- Official [ISTA Alistarh Group](https://www.ista.ac.at/en/research/alistarh-group/) identifies Professor Dan Alistarh and the Deep Algorithms and Systems Lab.
- Official [ISTA Research Explorer profile](https://research-explorer.ista.ac.at/person/4A899BFC-F248-11E8-B48F-1D18A9856A87) gives the full bibliographic name **Dan-Adrian Alistarh**, ORCID `0000-0003-3650-940X`, and Google Scholar ID `75q-6ZQAAAAJ`, exactly matching the reading-list link.
- Status: `VERIFIED — official metadata/profile`.
- Identity ambiguity: “Dan Alistarh” and “Dan-Adrian Alistarh” are the same person. The supplied Scholar ID removes residual ambiguity.

### Canonical 5-paper watchlist

| Paper | Venue | Why canonical here | Status |
|---|---|---|---|
| [QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding](https://proceedings.neurips.cc/paper/2017/hash/6c340f25839e6acdc73414517203f5f0-Abstract.html) | NeurIPS 2017 | foundational stochastic gradient quantization; separates communication compression from inference PTQ | optional |
| [Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1caf09c9f4e6b0150b06a07e77f2710c-Abstract-Conference.html) | NeurIPS 2022 | scalable approximate-second-order framework unifying one-shot quantization and pruning | optional |
| [OPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://openreview.net/pdf?id=tcbBPnfwxS) | ICLR 2023 | 3/4-bit one-shot LLM weight PTQ; direct descendant of second-order compression | optional; see naming ambiguity below |
| [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://proceedings.mlr.press/v202/frantar23a.html) | ICML 2023 | extends the same curvature-aware lineage from quantization to large-scale pruning | optional |
| [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html) | NeurIPS 2024 | rotation-based end-to-end 4-bit weights/activations/KV-cache; strong conceptual neighbor to SpinQuant | optional |

Recommended follow line:

`QSGD → Optimal Brain Compression → OPTQ/GPTQ → SparseGPT → QuaRot`

Follow **principled compression error control**: first stochastic/communication quantization, then approximate-second-order one-shot compression, finally rotation-based LLM quantization. For the current core list, `OPTQ/GPTQ + QuaRot` have the closest direct payoff; `SparseGPT` is the pruning comparison, and `QSGD` supplies optimization foundations.

### GPTQ/OPTQ naming ambiguity

- The accepted ICLR 2023 paper and ISTA publication record use **OPTQ**.
- The official implementation repository is named **GPTQ**, and later literature commonly cites the method as GPTQ.
- The final README should title it `OPTQ (commonly known as GPTQ)` and cite the accepted title; do not treat GPTQ and OPTQ as two papers.

## 9. Handoff: identity ambiguities and final note split

### Ambiguities that must survive into Phase 3

1. **NVFP8:** no distinct defining specification/paper found. Keep as product/checkpoint label pending advisor clarification.
2. **GPTQ/OPTQ:** one ICLR 2023 paper/method lineage with two names, not two papers.
3. **Dan Alistarh / Dan-Adrian Alistarh:** same verified person.
4. **FP8 scaling:** OCP OFP8 defines element interchange encodings, not per-tensor/block scaling. Do not attribute a vendor recipe to the standard.
5. **MXFP4 training:** the broad MX paper’s strong training example uses MXFP4 weights together with MXFP6 activations/gradients; do not summarize it as pure W4A4G4 training.
6. **NVFP4 training:** not fully 4-bit end to end; significant operations/layers/states stay BF16/FP32.

### Recommended final-note count from this packet

**12 notes total if the full watchlist is materialized, of which only 4 are format/core-support artifacts and 8 are optional watchlist notes:**

1. `FP8-Formats-for-Deep-Learning` — paper README + OFP8 normative companion.
2. `OCP-Microscaling-MXFP8-MXFP4` — one specification/family note; include the broad MX paper as evaluation.
3. `Pretraining-LLMs-with-NVFP4` — vendor technical-report README + Transformer Engine format definition.
4. `NVFP8-identity-and-evidence-gap` — short clarification note, **not a paper README**.
5–7. Song Han optional: `Deep Compression`, `EIE`, `Once-for-All`. `SmoothQuant`, `AWQ` and `HAQ` should cross-link their already-core notes rather than be duplicated.
8–12. Dan Alistarh optional: `QSGD`, `Optimal Brain Compression`, `OPTQ/GPTQ`, `SparseGPT`, `QuaRot`.

If time is tight, a **7-note minimum practical route** is: notes 1–4, plus `Deep Compression`, `OPTQ/GPTQ`, and `QuaRot`. The remaining watchlist entries stay in the total README as future reading and do not enter the core-paper count.

Optional thirteenth note only if a weekly report specifically focuses on native MXFP8 pretraining: `Recipes for Pre-training LLMs with MXFP8`. Otherwise keep it inside the MX-family note to avoid splitting a format name into an artificial paper count.

## 10. Evidence limitations

- Specifications define encodings and interchange/block behavior; they do not prove training accuracy or system throughput.
- Vendor documentation establishes current support and recipes, but version requirements and available kernels can change.
- Vendor-authored accuracy/throughput numbers are retained only with model, hardware or experiment context available in the source. No cross-paper number is normalized into a universal ranking.
- No independent system evaluation was found that turns “NVFP8” into a separately defined format. The absence is reported as an evidence gap rather than filled by inference.
