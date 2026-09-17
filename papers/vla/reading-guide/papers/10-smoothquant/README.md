# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

> **Reading-list role**: Core — activation-aware PTQ / efficient LLM inference 入口  
> **Verification**: `verified-full-text` — ICML 2023 official PMLR final  
> **Recommended effort**: Core read；第一次组会可先走 20-minute route

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han |
| Year / version | 2023; arXiv:2211.10438 |
| Venue / status | ICML 2023, PMLR 202:38087–38099; peer-reviewed venue final |
| Primary source | [PMLR record](https://proceedings.mlr.press/v202/xiao23c.html) · [official PDF](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) · [arXiv](https://arxiv.org/abs/2211.10438) |
| Code / project | [official repository](https://github.com/mit-han-lab/smoothquant) |

## 2. One-sentence takeaway

SmoothQuant 用一个 FP function-preserving 的 per-channel scaling，把难量化的 activation outliers 离线迁移到相对容易量化的 weights，从而让 LLM 的 Linear 和 attention BMM 能走 regular `W8A8` INT8 kernels，并在 OPT-175B 上将 average accuracy 从 66.9 保持到 66.8。

## 3. Background and prerequisites

- 先理解 symmetric uniform quantization、scale、clipping、static vs dynamic quantization。
- 区分 weight-only quantization 与 `W8A8`：前者节省 weight bandwidth，但 matrix multiplication 仍处理 FP16 activation；后者才可完整使用 INT8 tensor cores。
- 熟悉 Transformer 的 residual stream、LayerNorm、Linear、attention `BMM`；知道 activation outlier 是少数 channel 的异常大幅值，而不只是随机 single-element noise。
- Symbols：$X$ 是 activation，$W$ 是 weight，$s$ 是 channel-wise smoothing factor，$\alpha$ 控制 quantization difficulty 在 activation 与 weight 间的迁移方向。

## 4. Problem

- **Target setting**：无需 retraining 的 post-training quantization，目标是 LLM inference 的 weights 与 activations 均为 INT8。
- **Bottleneck**：activation outliers 扩大 per-tensor dynamic range，使普通 INT8 quantizer 把多数正常值挤到很少 bins 中，accuracy/perplexity 崩溃。
- **Why previous methods are insufficient**：weight-only 路径没有把 activation compute 变成 INT8；mixed-precision outlier routing 可以保 accuracy，却引入 irregular storage、kernel branching 和额外 system complexity；naive W8A8 在 Table 3 的 OPT-175B PPL 达 93080。

## 5. Method

### 5.1 System view

`calibration sentences → collect per-channel activation maxima → choose α → construct smoothing scale s → fuse inverse/forward scales into neighboring parameters → quantize Linear/BMM → dense INT8 inference`

主要 element-wise operations 仍保留 FP16。O1 是 per-token dynamic，O2 是 per-tensor dynamic，O3 是 per-tensor static；O3 最接近 deployment-friendly full static path。

### 5.2 Core mechanism

Base symmetric quantizer，Section 2, Eq. (1)：

$$
\bar X_{\mathrm{INT8}}=\operatorname{round}(X_{\mathrm{FP16}}/\Delta),\qquad
\Delta=\frac{\max(|X|)}{2^{N-1}-1}.
$$

关键 equivalent transformation，Section 4, Eq. (3)：

$$
Y=XW=(X\operatorname{diag}(s)^{-1})(\operatorname{diag}(s)W)=\hat X\hat W.
$$

同一 input channel $j$ 的 activation 除以 $s_j$、对应 weight row 乘以 $s_j$，所以在 real-valued arithmetic 中 $Y$ 不变；但是重新分配后的 ranges 会改变 quantization error。

Section 4, Eq. (4)：

$$
s_j=\frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}.
$$

$\alpha$ 越大，activation 被压得越强、weight range 越大。Paper 对 OPT/BLOOM 用 $0.5$，GLM-130B 用 $0.75$，LLaMA experiment 用 $0.8$。Calibration 使用 512 条 random Pile sentences。

### 5.3 What is actually new

真正新增的是 `quantization difficulty migration`：不是设计更复杂的 quantizer，也不是给 outliers 单独开 mixed-precision branch，而是以可 fuse 的 exact scaling 把 tensor 变成现有 dense INT8 kernels 容易处理的 distribution。Grid-search $\alpha$、普通 uniform quantizer 和 INT8 GEMM 属于实现该机制的 standard choices。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| 175B accuracy 基本保持 | OPT-175B FP16 average/PPL **66.9/10.99**；SmoothQuant O3 **66.8/11.17**；naive W8A8 **35.5/93080**；ZeroQuant **35.8/84648**。 | Table 3, paper p. 38092 / PDF p. 6 | `lossless` 是 benchmark-level near-equivalence，不是 bit-exact。 |
| 跨 LLaMA scale | 7B/13B/30B/65B WikiText-2 PPL：FP16 **11.51/10.05/7.53/6.17**；W8A8 **11.56/10.08/7.56/6.20**。 | Table 6, paper p. 38093 / PDF p. 7 | 仍是 language modeling，不代表 VLA action quality。 |
| Latency 与 memory | OPT-30B batch 1, seq 512：**422→314 ms**, **57→30 GB**。OPT-175B batch 16, seq 512：**2212→1628 ms**, **50→30 GB**。 | Table 7, paper p. 38094 / PDF p. 8 | Speedup 随 batch、sequence、kernel/backend 改变。 |
| 530B deployment | MT-NLG 530B average **73.1→73.1**；GPU count **16→8**；seq 1024 memory **1095→570 GB**, latency **1707→1689 ms**。 | Tables 8–9, paper p. 38094 / PDF p. 8 | Latency 改善小于 memory/GPU-count 改善，不能只说“全面 2×”。 |

读结果时先区分 metric direction：accuracy 越高越好、PPL/latency/memory 越低越好。然后把 quality preservation 与 system speedup 视作两条独立 evidence chain。

## 7. Limitations

### Authors' stated limitations

- 没有独立 `Limitations` section。Appendix 说明与 GPTQ weight-only 的 speed comparison 不完全公平，因为 implementations/workloads 不同；batch-1 autoregressive generation 中 GPTQ 可以更快。
- 主体是 W8A8；更 aggressive 的 W4A4 integration 被留作 future work。

### My critique

- **Internal validity**：$\alpha$ 依赖 model family，且 calibration maxima 是否覆盖 rare states 决定 scaling quality；论文没有证明唯一/普适最优的 $\alpha$。
- **External validity**：512 Pile sentences 与 text benchmarks 没覆盖 vision tokens、proprioception 或 rare action failures。
- **Systems validity**：Table 7 speedup 依赖 batch、sequence、INT8 kernel 与 GPU；不能直接外推到 robot SoC。
- **Reproducibility**：official code 可用，但 VLA 复现仍需重新定义 calibration distribution 和 continuous-action metrics。

## 8. Why it matters for this project

- **VLA**：可直接作用于 VLM/language backbone、multimodal projector 和 action-token decoder 的 Linear/BMM；对 batched prefill 和高 throughput 较有吸引力。
- **Embodied risk**：continuous action head 的少数 sensitive dimensions 可能并不产生大 activation，因此不能只用 text max statistics；需测试 success rate、action L2 error、temporal drift 和 P99 latency。
- **Professor Li's direction**：它把 numerical transformation 与 commodity INT8 hardware path 对齐，是 model compression、latency、edge deployment 和 software-hardware co-design 的典型交叉点。

## 9. How to read it

### 20-minute route

1. 读 Abstract 与 Figure 1，先说清 activation outlier 为什么阻止 W8A8。
2. 读 Section 4 的 Eq. (3)–(4)，手写一次 $XW=(X/s)(sW)$。
3. 核对 Table 3 的 OPT-175B 四条 baseline 数字。
4. 看 Table 7，只选一个 row 分别解释 latency、memory、batch/sequence。
5. 写一句：`FP equivalent does not mean quantization-error equivalent`。

### 90-minute route

1. **0–15 min**：补 uniform quantization、static/dynamic scale 与 outlier prerequisites。
2. **15–35 min**：读 Sections 2–4，推导 Eq. (3)，解释 $\alpha\to0$ 与 $\alpha\to1$ 两个极端。
3. **35–50 min**：画出 calibration/fusion/inference dataflow，标出哪些操作仍为 FP16。
4. **50–65 min**：核对 Tables 3、6，确认 quality claim 跨 OPT/LLaMA scale。
5. **65–78 min**：核对 Tables 7–9，区分 memory、latency 与 GPU-count claim。
6. **78–90 min**：设计一个 VLA calibration ablation：generic text vs multimodal trajectory vs failure-heavy trajectory。

## 10. Reading questions

1. Eq. (3) 为什么在 FP 中 exact equivalent，却会改变 quantization 后的 error？
2. $\alpha$ 增大时，activation error 与 weight error 分别如何变化？
3. 为什么 O3 static path 对 deployment 更重要，也更难？
4. 如果 rare action state 不在 calibration set，per-channel maximum 会如何误导 scaling？
5. Table 7 的 speedup 为什么不能与 Table 3 的 accuracy preservation 合成一个单一 claim？
6. SmoothQuant 与 AWQ 都用 scaling，它们量化的 object、objective 和 bit setting 有何不同？

### 10.1 My answers

#### 1. 为什么 FP exact equivalent 不等于 quantized equivalent？

令 $S=\operatorname{diag}(s)$。在 real-valued arithmetic 中：

$$
XW=(XS^{-1})(SW),
$$

因为中间的 $S^{-1}S=I$，所以只是 reparameterization，并没有改变 function。但 quantizer $Q(\cdot)$ 是带 rounding、clipping 和有限 range 的 nonlinear map，一般不满足：

$$
Q(XS^{-1})Q(SW)=XW.
$$

把 quantization error 写出来更清楚。设：

$$
Q(XS^{-1})=XS^{-1}+E_X,\qquad Q(SW)=SW+E_W,
$$

那么 quantized output error 为：

$$
E_XSW+XS^{-1}E_W+E_XE_W.
$$

$S$ 虽然在 FP path 中完全抵消，却会改变两个 tensor 的 range、quantization step、clipping probability，以及 error 被另一个 operand 放大的方式。因此 SmoothQuant 真正优化的是 **quantization 后的 error allocation**，不是 FP function 本身。

#### 2. $\alpha$ 增大时，两侧 error 怎样变化？

记 $a_j=\max|X_j|$、$w_j=\max|W_j|$，则：

$$
s_j=\frac{a_j^\alpha}{w_j^{1-\alpha}},\qquad
\max|X_j/s_j|=(a_jw_j)^{1-\alpha},\qquad
\max|s_jW_j|=(a_jw_j)^\alpha.
$$

所以 $\alpha$ 增大时，activation range 通常缩小，activation quantization 变容易；weight range 通常增大，weight quantization 变难。两个极端可这样记：

- $\alpha\to0$：几乎不替 activation 承担 outlier difficulty，weights 更容易量化。
- $\alpha\to1$：强力压 activation outliers，但把更大 dynamic range 推给 weights。

这只是 range-level tendency，不保证 downstream error 严格 monotonic，因为 group/tensor granularity、rounding boundary、clipping 和后续 layer amplification 也会参与。最佳 $\alpha$ 因而是 balance，而不是“越大越好”。

#### 3. 为什么 O3 static path 更重要，也更难？

O3 在 calibration 后固定 activation scale。deployment 时不再为每个 token 计算 `absmax`，因此可以：

1. 去掉 runtime reduction 和 scale-computation overhead；
2. 构建 fixed-shape、predictable 的 graph；
3. 更容易做 operator fusion、ahead-of-time compilation 和 dense INT8 kernel dispatch；
4. 得到更稳定的 P50/P99 latency。

代价是 scale 无法适应当前 input。如果 calibration set 没覆盖某类 prompt、vision observation 或 robot failure state，runtime activation 可能超出固定 range 并被 clipping；若为了防止 clipping 把 static range 设得太宽，大部分正常值又会只占很少 quantization bins。O1/O2 的 dynamic scale 能跟随当前 token/tensor，因此 quality 较稳，但需要 runtime statistics，system path 也更复杂。

#### 4. rare action state 缺失时，per-channel maximum 会怎样误导 scaling？

若 calibration 只观察到较小的 $a_j$，它会低估 channel $j$ 的真实 peak，于是得到过小的 $s_j$。这意味着该 channel 的 activation 没被充分缩小、相应 difficulty 也没有足够迁移到 weights。部署时 rare state 产生更大 activation 后：

- 对 O3 static quantization，最直接的结果是 saturation/clipping；
- 对 dynamic per-tensor quantization，新的 outlier 会突然拉宽整 tensor scale，使其他普通 channels 的 resolution 下降；
- 即使单步 action error 不大，也可能通过 closed-loop feedback 累积成 trajectory drift。

因此 VLA calibration 不应只随机采样常见 frames；至少要覆盖 contact transitions、occlusion、recovery、near-failure states，并比较 random calibration 与 failure-heavy calibration。也可以同时记录 maximum、high percentile 和 clipping rate，避免让一个不稳定 maximum 成为唯一 statistic。

#### 5. 为什么 Table 7 与 Table 3 是两条 evidence chain？

Table 3 回答“quantized model 的 benchmark quality 是否保持”，measurement object 是 accuracy/PPL；Table 7 回答“特定 hardware、batch、sequence length 和 kernel implementation 下是否减少 latency/memory”。两者的 setup、metric 和 causal factors 都不同。

因此不能把它们压成“SmoothQuant 在保持 accuracy 的同时统一加速 1.5×”这种无条件 claim。可能出现：quality 保持但 backend 没有合适 INT8 kernel，因而不加速；也可能 latency 降低但某个 task quality 已受损。正确表述必须同时带上 quality setting 与 systems setting，例如：`OPT-175B O3 在 Table 3 的 text benchmarks 接近 FP16；另在 Table 7 的 A100 workload 上降低 latency/memory`。

#### 6. SmoothQuant 与 AWQ 的 scaling 到底有什么不同？

| Dimension | SmoothQuant | AWQ |
|---|---|---|
| Quantized object | weights + activations | weights only；activations 保持 FP16/BF16 |
| Typical setting | `W8A8` | `W4A16` / `W3A16`, usually group-wise |
| Primary bottleneck | activation outliers 阻止 regular INT8 GEMM/BMM | low-bit weight rounding 会破坏 salient input channels |
| Statistic | per-channel activation maximum 与 weight maximum | per-channel average activation magnitude，作为 weight saliency proxy |
| Objective | 平衡 activation/weight quantization difficulty，服务 dense INT8 path | 保护 output-sensitive weight channels，同时保留 regular packed low-bit layout |
| Deployment gain | 同时减少 weight bandwidth 与 INT8 compute cost；static path 最容易 fuse | 主要减少 autoregressive decoding 的 weight memory traffic；依赖 dequantization/packing kernel |

两者都利用 FP-equivalent scaling，但不能因为公式长得相似就视为同一个 method。SmoothQuant 问的是“怎样把 activation outlier difficulty 搬到 weights”；AWQ 问的是“怎样利用 activation signal 保护最重要的 low-bit weights”。

## 11. Weekly meeting card

- **Problem**：activation outliers 使 regular LLM W8A8 accuracy 崩溃。
- **Key idea**：用可 fuse 的 per-channel scaling，把 quantization difficulty 从 activations 迁移到 weights，保持 FP function。
- **Best evidence**：OPT-175B average 66.9→66.8；batch-16/seq-512 latency 2212→1628 ms、memory 50→30 GB。
- **Biggest limitation**：text calibration/benchmarks 未覆盖 VLA rare action channels，system gain 又依赖 hardware workload。
- **Question for the group**：trajectory-aware calibration 能否在保持 INT8 kernels 的同时保护 rare control dimensions？

## 12. Status & evidence boundary

- **Status**：peer-reviewed ICML 2023 venue final；metadata、equations、tables 来自 official PMLR full text。
- **Source claim**：Eq. (1)/(3)/(4)、512 Pile calibration、Tables 3/6/7/8/9 的数字。
- **My interpretation**：它对 VLA backbone 的适用性，以及用 trajectory-aware calibration 改造的建议。
- **Open question**：W8A8 对 continuous actions、closed-loop success、edge SoC P99 latency 的影响，原文不能确认。
