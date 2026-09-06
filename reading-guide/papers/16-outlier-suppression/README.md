# Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models

> **Reading-list role**: Core #16 - structured activation-outlier diagnosis and low-bit Transformer quantization foundation  
> **Verification**: `verified-full-text` - NeurIPS 2022 official proceedings final + official supplemental  
> **Recommended effort**: Core read; complete the 20-minute route before revisiting SmoothQuant

> **Local full text**: [paper.pdf](paper.pdf)  
> **Local supplemental**: [supplemental.pdf](supplemental.pdf)

## 1. Paper identity and version boundary

| Field | Value |
|---|---|
| Authors | Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, Xianglong Liu |
| Canonical publication | NeurIPS 2022 Main Conference Track, *Advances in Neural Information Processing Systems 35*, 17402-17414 |
| DOI | `10.52202/068431-1265` |
| arXiv record | arXiv:2209.13325; v1 2022-09-27, v2 2022-10-14, v3 2023-02-21 |
| Primary sources | [NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6f6db140de9c9f111b12ef8a216320a9-Abstract-Conference.html) · [official paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/6f6db140de9c9f111b12ef8a216320a9-Paper-Conference.pdf) · [official supplemental](https://proceedings.neurips.cc/paper_files/paper/2022/file/6f6db140de9c9f111b12ef8a216320a9-Supplemental-Conference.pdf) · [arXiv](https://arxiv.org/abs/2209.13325) |
| Code | [official `wimh966/outlier_suppression` repository](https://github.com/wimh966/outlier_suppression) |
| Local main artifact | NeurIPS proceedings final, 13 PDF pages, SHA-256 `320b3958f960cb93c5778225197fcecbe39ff69f5c3bbc6bcb7d5236bc83aef2` |
| Local supporting artifact | NeurIPS supplemental, 10 PDF pages, SHA-256 `8c4d62b9f6a5cf6351c4aabad7c14623a2fb8ec39524656e16af31b4af854163` |

**Version boundary.** 本地 `paper.pdf` 与 `supplemental.pdf` 是 official NeurIPS proceedings artifacts，不按 arXiv v3 标记。arXiv v3 是同一 work cluster 的 later preprint version，不另计 paper。Official main paper 把 proof、quantization nodes、additional ablations 与 implementation details 放进 separate supplemental；因此两份本地文件合起来才是这次的完整阅读包。

## 2. One-sentence takeaway

Outlier Suppression 先把 `LayerNorm` 的 learned scale $\gamma$ 识别为 structured outliers 的 amplifier，再用 function-preserving **Gamma Migration** 把 $\gamma$ 移进后续 branches/weights，并用 **Token-Wise Clipping** 从 per-token extrema 中高效寻找 clipping range；在 paper 的 `BERT`/`RoBERTa`/`BART` setup 中，它显著改善 6-bit `PTQ` 与 4-bit `QAT`，但后来的 `SmoothQuant` evidence 显示这套 recipe 不能直接外推到 100B-scale `LLM`。

## 3. Background and prerequisites

- **Uniform quantization**：知道 scale、zero point、rounding、clipping，以及 `PTQ` 与 `QAT` 的差别。
- **Quantization granularity**：分清 per-layer/per-tensor、per-channel、per-token 与 group-wise quantization；granularity 越细，通常 accuracy 越好，但 metadata 与 kernel path 越复杂。
- **Transformer block**：熟悉 `LayerNorm`、`Multi-Head Attention`、`FFN`、`GELU` 与 residual connection。
- **Structured outliers**：这里不是孤立 random noise，而是集中在少数 embedding dimensions，并由 `[SEP]`、`[CLS]`、punctuation 或 frequent tokens 进一步放大的 activation pattern。
- **Function-preserving reparameterization**：real-valued function 相同，不代表经过 nonlinear quantizer 后 error 相同。
- **Bit notation**：paper 的 `W-E-A` 分别表示 weight、embedding、activation bit-width，例如 `6-6-6`。

建议先快速看本地 [LLM.int8 / bitsandbytes note](../06d-llm-int8-bitsandbytes/README.md)，再读本篇，最后进入 [SmoothQuant](../10-smoothquant/README.md)。三篇分别代表 mixed-precision outlier routing、outlier suppression、以及 large-scale activation-to-weight smoothing。

## 4. Problem

Paper 针对的是一个很具体的 failure mode：

1. `LayerNorm` 与 `GELU` outputs 中存在大幅 structured outliers。
2. Coarse per-tensor/per-layer activation quantization 的 range 被少数 outliers 拉宽，普通 values 只能使用很少 quantization bins。
3. 早期解决方案会用 finer granularity 或绕开 problematic values，但这可能增加 computation、scale metadata 与 irregular kernel cost。
4. 直接按 local reconstruction error 选 clipping range，也不一定对应 final-task performance；某些很大的 outliers 可以安全截断，另一些较小 values 一旦被截断就会让 accuracy 突降。

作者因此把问题拆成两个 research questions：

- outliers 为什么在特定 embedding dimensions 被放大？
- 怎样利用 token-range structure，快速找到对 final output loss 更合适的 clipping range？

## 5. Method

### 5.1 Gamma Migration

Standard `LayerNorm` 对 token $t$、dimension $j$ 的 output 是：

$$
\widetilde X_{t,j}=\frac{X_{t,j}-\mu_t}{\sqrt{\sigma_t^2+\epsilon}}\gamma_j+\beta_j.
$$

作者把 learned scale $\gamma_j$ 从 `LayerNorm` 中抽出，定义 Non-scaling LayerNorm：

$$
X'_{t,j}=\frac{X_{t,j}-\mu_t}{\sqrt{\sigma_t^2+\epsilon}}+\frac{\beta_j}{\gamma_j},
\qquad
\widetilde X_{t,j}=X'_{t,j}\gamma_j.
$$

接着把 $\gamma$ 迁移到 residual branch 与后续 Linear weights。对 column vector $x$：

$$
W(x\odot\gamma)=(W\odot[\gamma,\gamma,\ldots]^\top)x.
$$

因此在 full precision 中，这只是 exact reparameterization；quantizer 改在较温和的 $X'$ 上执行。Main paper Figure 3 与 supplemental Figures 6-7 展示 `MHA-LN`、`FFN-LN` 与 Cross-Attention 的 migration path。

关键边界：

- `FP equivalence` 只说明 real-valued function 不变。
- Quantization error 会变，因为 $X'$ 与 absorbed-$\gamma$ weight 的 ranges 改变。
- “No extra inference time”是作者对 fused/equivalent path 的 method-level claim；paper 没有提供 modern end-to-end production latency benchmark 来独立证明 every backend 都无开销。

### 5.2 Token-Wise Clipping

目标是选 quantization step size $s$，最小化 final quantized output 与 real output 的 Frobenius distance：

$$
\mathcal L(s)=\lVert \hat f(s)-f\rVert_F^2.
$$

**Coarse stage** 先用每个 token 的 maximum/minimum 作为 representatives：

$$
o_u=\{\max(x_1),\max(x_2),\ldots,\max(x_T)\},
$$

再对 $o_u/o_l$ 做 quantile grid search，得到 clipping bounds $c_u/c_l$ 与 initialization $s_0$。因为 long-tail outliers 只来自少数 tokens，这个 token-wise search 可以跳过很大一段不重要 value range。

**Fine stage** 从 $s_0$ 出发，用 gradient descent 继续优化：

$$
s\leftarrow s-\eta\frac{\partial\mathcal L(s)}{\partial s}.
$$

Supplemental Algorithm 1 给出完整流程。`PTQ` 使用 coarse + fine stages；`QAT` 与 `LSQ+` 结合时，coarse stage 负责 initialization，后续 step-size learning 交给 `LSQ+`。

### 5.3 Quantization recipe

- Weight：symmetric per-channel quantization。
- Activation：asymmetric per-layer quantization。
- Calibration：256 samples；`GLUE`/`SQuAD` batch size 32，summarization batch size 4。
- `PTQ`：Token-Wise Clipping fine stage 统一训练 3 epochs，learning rate $10^{-5}$。
- `QAT`：与 `LSQ+` 结合，并对 learning rate/batch size 做 search。
- Quantization nodes：paper 采用接近 `FasterTransformer` 的 hardware-oriented placement；supplemental Figure 8/Table 8 列出 Input Embedding、Query、Key、Value、Attention probabilities、Context、`MHA-LN`、`GELU` 与 `FFN-LN`。

## 6. What is actually new

这篇的 novelty 不是第一次观察 Transformer outliers，也不是第一次做 clipping。真正的新组合是：

1. 把 `LayerNorm` $\gamma$ 明确解释为 activation-outlier amplifier，并通过 exact parameter migration 改变 quantization geometry。
2. 把 outlier importance 与 token-level range variance 联系起来，用 per-token extrema 做 coarse search。
3. 把 two-stage suppression 作为 plug-in，分别接到 `PTQ` calibration 与 `QAT` initialization。

不要把它与 `SmoothQuant` 合并成同一个 method。两者都使用 function-preserving scaling，但 Outlier Suppression 专门迁移 `LayerNorm` $\gamma$ 并配合 Token-Wise Clipping；`SmoothQuant` 依据 activation/weight channel statistics 构造可调 smoothing factor $s_j$，目标是 regular large-scale `W8A8` kernels。

## 7. Experiments and main results

### 7.1 Authors' reported evidence

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Non-scaling LayerNorm 明显降低 activation quantization error | Across 12 `BERT-SST-2` `MHA-LN` tensors，6-bit cosine similarity 从约 **93.31-98.57%** 提升到 **98.70-99.44%** | Table 1, main PDF p. 4 | Tensor similarity 是 mechanism evidence，不等于 downstream-task accuracy。 |
| 两个 components 都贡献改善 | `RoBERTa` 6-bit `PTQ`：`QNLI` 从 MinMax **62.13** 到 full method **86.82**；`MRPC` 从 **71.64/67.40** 到 **87.50/83.33** | Table 3, main PDF p. 8 | Full-precision `QNLI` 是 **92.68**；仍有明显 gap。 |
| 6-bit `BERT PTQ` 接近 paper 的 full-precision average | `BERT` `32-32-32` average **83.83**；ours `6-6-6` **81.19**，absolute difference **2.64 points** | Table 4, main PDF p. 8 | 同一 table 的 `RoBERTa` 与 `BART` gaps 更大：**86.40→79.62**、**84.61→79.10**。Headline 不能推广为 all models full-precision parity。 |
| 4-bit `BERT QAT` 明显优于 baseline | `BERT` `4-4-4`：ours **81.13** average vs `LSQ+` **71.49**；ours + `KD` **83.56** vs FP **83.83** | Table 5, main PDF p. 9 | `KD`、`LSQ+` 与 proposed initialization 共同作用；不能把 full gain 全归因于 Gamma Migration。 |
| Scope 不只 classification | `BERT` `SQuAD v1.1` full precision **88.28/80.82 F1/EM**，ours 6-bit **84.48/75.53**；`BART` `SQuAD v2.0` **80.82/77.41→74.44/70.36** | Table 6, main PDF p. 9 | Quality 保留并不统一；model/task 对低 bit-width 的 sensitivity 不同。 |
| Coarse token-wise search 比比较对象更快 | 256-sample activation calibration：Token-Wise Clipping coarse **135.73 s**；OMSE grid **1754 s**、golden-section **439.29 s**、Percentile **301.49 s** | Table 15, supplemental PDF p. 9 / printed p. 22 | Calibration microbenchmark 不是 inference latency；hardware/software environment 没在 table 中完整绑定。 |

### 7.2 Later direct comparison reported by SmoothQuant

以下不是 Outlier Suppression authors 的 claim，而是后来的 `SmoothQuant` 在自己的 matched `W8A8` evaluation 中报告的 baseline result：

| Model / metric | Full precision | Outlier Suppression baseline | SmoothQuant-O3 | Locator |
|---|---:|---:|---:|---|
| `OPT-175B` zero-shot average / WikiText PPL | **66.9 / 10.99** | **36.0 / 96151** | **66.8 / 11.17** | SmoothQuant Table 3, local PDF p. 6 |
| `OPT-175B` four-task average | **71.6** | **31.7** | **71.1** | SmoothQuant Table 4, local PDF p. 6 |
| `BLOOM-176B` four-task average | **68.2** | **54.1** | **67.4** | SmoothQuant Table 4, local PDF p. 6 |
| `GLM-130B` four-task average | **73.8** | **63.5** | **72.8** | SmoothQuant Table 4, local PDF p. 6 |

这个 later comparison 支持一个重要 scope boundary：Outlier Suppression 在 original `BERT`/`RoBERTa`/`BART` experiments 中有效，但不能据此声称已经解决 modern autoregressive large-`LLM` activation quantization。它也只是 `SmoothQuant` authors 对 baseline 的 implementation/evaluation，不是 independent multi-lab replication。

## 8. Limitations

### Authors' stated limitations

- Conclusion 明确把 computer vision extension 留作 open problem。
- Authors 认为还需要从 pretraining process 深入理解 outlier formation。
- NeurIPS checklist 说明没有报告 random-seed error bars，理由是跨 models/datasets 重复实验成本高。

### My critique

- **Model-era boundary**：主体是 encoder-style `BERT`/`RoBERTa` 与较早的 `BART`；没有 current decoder-only `LLM`、`VLM` 或 `VLA` evidence。
- **External validity**：frequent text tokens、`[SEP]`/`[CLS]` 与 punctuation 的 outlier pattern，不自动对应 vision patches、proprioceptive states 或 action channels。
- **Systems validity**：paper 讨论 hardware-friendly nodes 与 “no extra burden”，但没有报告 end-to-end latency、peak memory、power、P50/P99 或 production kernel profiling。
- **Calibration boundary**：256 examples 可能漏掉 rare tokens/states；VLA 中 near-failure trajectories 可能正是最危险的 missing calibration modes。
- **Statistical boundary**：没有 error bars/seed variance；多个 small `GLUE` tasks 对 hyperparameter search 敏感。
- **Baseline boundary**：`QAT` results 同时涉及 `LSQ+`、`KD`、learning-rate/batch search 与 proposed initialization；必须按 ablation 而不是 headline 归因。
- **No universal clipping semantics**：一个 value 在 full-precision text benchmark 上可安全 clipping，不代表它在 closed-loop control 中不承载 rare corrective action。

## 9. Why it matters for this project

这篇值得单开 `#16`，因为它补上目前 quantization route 中缺失的一环：**先诊断 outlier 的来源与重要性，再决定迁移、截断或旋转**。

对 `VLA`，可以把它转成三个可验证 hypotheses：

1. **Source hypothesis**：vision encoder、language backbone、multimodal projector 与 action expert 的 outliers 是否同样由 `LayerNorm` $\gamma$ 放大？
2. **Importance hypothesis**：large action-channel activation 是否真能安全 clipping，还是代表 contact transition、recovery 或 rare corrective behavior？
3. **Deployment hypothesis**：Gamma Migration 是否能 fuse 到目标 runtime 的 weights/residual path，并在 matched kernels 上真正改善 latency/control frequency，而不只是 offline accuracy？

最小实验需要同时报告：

`closed-loop success/progress + action error + clipping/saturation rate + P50/P99 latency + control frequency + peak memory + power/energy`。

Calibration data 至少比较：random trajectories、task-balanced trajectories、failure-heavy/recovery trajectories。若只用 text/token statistics，无法验证 embodied safety boundary。

## 10. Reading lineage

### 10.1 Historical route

`Understanding and Overcoming the Challenges of Efficient Transformer Quantization`  
→ structured outliers + finer granularity / PEG  
→ **Outlier Suppression**  
→ `LayerNorm` $\gamma$ migration + token-wise clipping  
→ **SmoothQuant**  
→ more general activation-to-weight channel smoothing for large `LLM W8A8`  
→ **QuaRot / SpinQuant**  
→ change outlier geometry by fixed/random or learned rotations

`LLM.int8()` 是同一时期的 parallel route：保留 outlier dimensions 的 higher-precision path，而不是先消除它们。

### 10.2 Method comparison

| Method | Main object | Outlier treatment | Typical target | System consequence |
|---|---|---|---|---|
| LLM.int8() | activations/weights in matrix multiplication | outlier features走 FP16 mixed-precision path | 8-bit large-model inference | quality strong; irregular mixed-precision path can add latency |
| Outlier Suppression | `LayerNorm`/`GELU` activations | migrate $\gamma$ + token-wise clipping | 6-bit `PTQ`, 4-bit `QAT` on `BERT` family | intends hardware-friendly per-layer activation path; large-LLM transfer fails in later evidence |
| SmoothQuant | Linear input channels + weights | calibrated per-channel scaling | regular `W8A8` large-`LLM` inference | scales fuse offline; designed for dense INT8 kernels |
| QuaRot | hidden states, FFN/attention, `KV cache` | fixed/randomized Hadamard rotation | `W4A4KV4` | some rotations fuse; others need online Hadamard kernels |
| SpinQuant | same rotated paths | learned rotations under quantized loss | `W4A4KV4` | learned/fused rotations plus remaining online transforms |

## 11. How to read it

### 20-minute route

1. **3 min** - Abstract + Introduction (main PDF pp. 1-2)：写出两个 findings 与两个 method components 的对应关系。
2. **4 min** - Figure 1 + Table 1 (pp. 3-4)：解释为什么 $\gamma$ 被叫作 amplifier，而不只是相关变量。
3. **5 min** - Section 4.1 + Figure 3 (p. 5)：手写 $\widetilde X=X'\odot\gamma$ 与 absorbed-weight equivalence。
4. **3 min** - Section 4.2 + Figure 4 (pp. 6-7)：说明 token-wise coarse search 为什么比 value-wise search 快。
5. **3 min** - Tables 3-5 (pp. 8-9)：选一条 `PTQ` 与一条 `QAT` result，并保留 FP/baseline/bit setting。
6. **2 min** - Conclusion + NeurIPS checklist (pp. 10, 13)：记录 authors' limitations 与 missing error bars。

### 90-minute route

1. **0-12 min** - Review quantizer equation、granularity、structured outliers 与 `LayerNorm`。
2. **12-28 min** - 读 Sections 3.1-3.2，分开 outlier inducement、token frequency conjecture 与 clipping-impact observation。
3. **28-43 min** - 推导 Gamma Migration；再看 supplemental Section A 的 proof 与 three structure diagrams。
4. **43-57 min** - 推导 Token-Wise Clipping coarse/fine stages；核对 supplemental Algorithm 1。
5. **57-69 min** - Audit Tables 3-7；分别标记 mechanism evidence、task quality 与 calibration-time evidence。
6. **69-77 min** - 读 supplemental Figure 8、Tables 8-11，确认 exact quantization nodes 与 hardware-oriented assumptions。
7. **77-84 min** - 对比 SmoothQuant Tables 2-4，写出 original scope 与 large-`LLM` transfer gap。
8. **84-90 min** - 设计一个 `VLA` experiment：component-wise Gamma Migration × calibration-set composition × closed-loop metrics。

## 12. Reading questions - answer these yourself

1. 为什么 $\gamma$ 与 activation outliers 出现在相同 dimensions，还不足以单独证明 causal amplification？Eq. (3)-Eq. (5) 补上了什么 evidence？
2. Gamma Migration 为什么在 full precision 中 exact equivalent，却会改变 quantization error？
3. Residual branch 与 next Linear weight 分别怎样吸收 $\gamma$？哪些 operations 让 migration 不能随意 fuse？
4. Token-Wise Clipping 为什么用 per-token maximum/minimum，而不是直接对所有 activation values 排序？
5. Eq. (6) 优化的是 output reconstruction loss。它与 final task accuracy 之间还有什么 gap？
6. Table 4 的 `BERT 6-6-6` 支持什么 claim？为什么同表的 `RoBERTa/BART` rows 阻止你说“6-bit universally reaches FP level”？
7. Table 5 中 `LSQ+`、`KD` 与 proposed initialization 各贡献什么？现有 ablation 能否完全分离？
8. SmoothQuant Table 4 的 large-`LLM` failure 说明 original method 的哪一个 assumption 没有 scale？
9. Outlier Suppression 与 SmoothQuant 都是 equivalent scaling；它们的 scale source、placement、quantized object 与 deployment target 有何不同？
10. 在 `VLA` 中，哪些 rare tokens/states 可能不能安全 clipping？你会怎样构造 failure-heavy calibration set？
11. 如果 Gamma Migration 不改变 nominal FLOPs，为什么仍需要测 P50/P99 latency、peak memory、power 与 control frequency？

建议把回答直接写在每题下方，并为每个答案添加 `Evidence: main/supplemental Section/Figure/Table` locator。

## 13. Weekly meeting card

- **Problem**：structured activation outliers 让 coarse low-bit Transformer quantization 崩溃。
- **Key idea**：migrate `LayerNorm` $\gamma$ into subsequent paths，再用 token-wise coarse-to-fine clipping 找 range。
- **Best original evidence**：`BERT` 6-bit `PTQ` average **83.83→81.19**；`BERT` 4-bit `QAT` + `KD` **83.83→83.56**。
- **Strongest boundary**：later `SmoothQuant` evaluation 中，Outlier Suppression on `OPT-175B` reports **36.0 average / 96151 PPL** vs FP **66.9 / 10.99**。
- **Question for the group**：在 `VLA` 中，大 activation 是 removable outlier，还是 rare but safety-critical control signal？

## 14. Evidence boundary

- **Source claim**：outlier analysis、equations、component ablations、`BERT`/`RoBERTa`/`BART` results 与 calibration details 来自 local NeurIPS main/supplemental artifacts。
- **Direct later comparison**：100B-scale `LLM` baseline numbers 来自 local SmoothQuant venue final Tables 2-4，已与 original-paper claims 分开。
- **My synthesis**：historical lineage、`VLA` hypotheses、failure-heavy calibration proposal 与 deployment metric design。
- **Open question**：modern `LLM`/`VLM`/`VLA` architectures、closed-loop control、current kernels 与 edge hardware 上是否有效，需要 new experiments。
- `verified-full-text` 表示 identity、method 与 stated results 已追到 local primary artifacts；不表示 independent experimental replication，也不表示 user 已读完。
