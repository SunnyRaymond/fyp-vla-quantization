# TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

> **Reading-list role**: Core — online vector/KV-cache quantization with rate-distortion guarantees  
> **Verification**: `verified-full-text` — ICLR 2026 official proceedings final; arXiv v1 also full-text checked  
> **Recommended effort**: Deep read；理论主线与 KV-cache evidence 分两遍读

> **Local full text**: [ICLR 2026 proceedings final](paper-iclr-2026-final.pdf) · [archived arXiv v1](paper-arxiv-v1.pdf)  
> **Critical companion**: [Revisiting RaBitQ and TurboQuant](../026-rabitq-turboquant-comparison/README.md)
> **Foundational RaBitQ reading**: [original paper + multi-bit extension](../027-rabitq/README.md)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni |
| Year / version | ICLR 2026 final; earlier arXiv:2504.19874v1 submitted 2025-04-28 |
| Venue / status | **Published as a conference paper at ICLR 2026**；official OpenReview ID `tO3ASKZlok` |
| Primary source | [ICLR proceedings record](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5c802ef38ab6e366c2ea06eee554c088-Abstract-Conference.html) · [official proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/5c802ef38ab6e366c2ea06eee554c088-Paper-Conference.pdf) · [OpenReview](https://openreview.net/forum?id=tO3ASKZlok) · [arXiv](https://arxiv.org/abs/2504.19874) |
| Code / project | No code link was established in the core evidence packet; do not infer one |

### Version drift to remember

The reading-list evidence packet was first built from arXiv v1; the official ICLR 2026 proceedings final is now local and canonical. In final Table 1, Llama-3.1-8B TurboQuant 2.5-bit average is **49.74**, whereas arXiv v1 reported **49.44**; 3.5-bit remains **50.06**. The proceedings final also removes arXiv v1 Table 2 containing the disputed PQ/RaBitQ/TurboQuant quantization times. Any presentation must label the version.

## 2. One-sentence takeaway

TurboQuant 用 random rotation 把 worst-case vector coordinates 转成已知 spherical/Beta distribution，再用 Lloyd–Max scalar codebooks 获得近 information-theoretic optimal 的 MSE rate；对 residual 再加 1-bit QJL 则得到 unbiased inner-product estimator，并在 ICLR final 的 Llama-3.1-8B Table 1 中以 3.5-bit KV 达到和 16-bit full cache 相同的 50.06 average。

## 3. Background and prerequisites

- Shannon source coding / rate-distortion：bit budget $b$ 与最小 achievable distortion 的关系。
- Random orthogonal rotation、unit hypersphere、coordinate Beta distribution 及高维 Gaussian approximation。
- Lloyd–Max scalar quantization / continuous 1-D k-means。
- Johnson–Lindenstrauss 与 1-bit QJL；unbiased estimator、variance、MSE 与 inner-product distortion 的差异。
- Transformer attention 与 KV cache：queries 和 cached keys/values 通过 inner products 交互；cache 随 context length 增长。
- Symbols：$Q$ 是 randomized quantizer，$b=B/d$ 是 average bits per coordinate，$\Pi$ 是 random orthogonal rotation，$S$ 是 Gaussian QJL matrix，$r$ 是 MSE-stage residual。

## 4. Problem

- **Target setting**：data-oblivious online vector quantization，重点应用于 streaming KV cache 与 nearest-neighbor vectors。
- **Bottleneck**：dataset-specific Product Quantization 需要 k-means/codebook preprocessing，不适合生成时不断到来的 KV；简单 online quantizers 的 distortion-rate 次优。
- **Metric mismatch**：最小 reconstruction MSE 不自动意味着 inner-product estimate unbiased；attention 更关心 inner-product geometry。
- **Goal**：对 worst-case vectors 同时给 computationally usable algorithms、$4^{-b}$-rate upper bounds，以及与 Shannon lower bound 对照的 near-optimality。

## 5. Method

### 5.1 System view

**MSE path**：`unit-normalize/store norm → random rotate Πx → per-coordinate nearest Lloyd–Max centroid → store b-bit indices → centroid lookup → inverse rotate Πᵀ → rescale norm`

**Inner-product path**：`(b−1)-bit MSE path → residual r → 1-bit sign(Sr) + store ||r|| → QJL reconstruction → add to MSE reconstruction → unbiased inner product`

**KV deployment**：`streaming K/V → split outlier/regular channels → apply separate bit allocations → cache compact indices/side information → dequantize for attention`。2.5-bit example 为 32 outlier channels 用 3 bits、96 regular channels 用 2 bits。

### 5.2 Core mechanism

Quantizer $Q:\mathbb R^d\to\{0,1\}^{B}$，$b=B/d$。Objectives：

$$
D_{mse}=\mathbb E_Q\|x-Q^{-1}(Q(x))\|_2^2,
$$

$$
D_{prod}=\mathbb E_Q\left|\langle y,x\rangle-\langle y,Q^{-1}(Q(x))\rangle\right|^2.
$$

#### Algorithm 1: MSE TurboQuant

Gaussian random matrix 经 QR 得 orthogonal $\Pi$，对 unit vector $x$ 计算 $z=\Pi x$。每个 coordinate density：

$$
f_X(t)=\frac{\Gamma(d/2)}{\sqrt\pi\Gamma((d-1)/2)}(1-t^2)^{(d-3)/2},
$$

高维趋近 $\mathcal N(0,1/d)$。预先对该 1-D distribution 做 Lloyd–Max，online 时只存每个 $z_j$ 最近 centroid 的 index，再以 $\Pi^\top$ 还原。

Theorem 1：对 $\|x\|=1$，

$$
D_{mse}\le\frac{\sqrt3\pi}{2}4^{-b}.
$$

$b=1,2,3,4$ 的 tighter numerical distortions 约为 **0.36, 0.117, 0.03, 0.009**。非 unit vector 需额外存 FP norm。

#### Algorithm 2: inner-product TurboQuant

先给 MSE stage $b-1$ bits，得到 $\tilde x_{mse}$ 与 $r=x-\tilde x_{mse}$；最后 1 bit：

$$
q=\operatorname{sign}(Sr),\qquad S_{ij}\sim\mathcal N(0,1),
$$

$$
\tilde x=\tilde x_{mse}+\frac{\sqrt{\pi/2}}{d}\|r\|_2S^\top q.
$$

Theorem 2：

$$
\mathbb E\langle y,\tilde x\rangle=\langle y,x\rangle,
\qquad
D_{prod}\le\frac{\sqrt3\pi^2\|y\|_2^2}{d}4^{-b}.
$$

$b=1,2,3,4$ 的 refined bounds 约 **1.57/d, 0.56/d, 0.18/d, 0.047/d**。

#### Information-theoretic lower bound

Theorem 3 用 Shannon lower bound + Yao's minimax principle 得 hard instances：

$$
D_{mse}\ge4^{-b},\qquad
D_{prod}\ge\frac{\|y\|_2^2}{d}4^{-b}.
$$

所以 rate dependence 与 lower bound 同为 $4^{-b}$，MSE constant factor 最多 $\sqrt3\pi/2\approx2.7$。

### 5.3 What is actually new

真正新的是两段式 theoretical design：random rotation 将 distribution-free worst-case input 转成 distribution-known coordinates，使 simple scalar codebook 获得 near-optimal vector rate；residual QJL 则专门修复 MSE quantizer 对 inner products 的 bias。Outlier-channel split、Lloyd–Max solver、KV evaluation 是支撑 deployment 的 choices，不应与核心 theorem 混为一个 claim。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Long-context retrieval | LLaMA-3.1-8B-Instruct, context 4k–104k, memory ratio 0.25：Full **0.997**, TurboQuant **0.997**, PolarQuant **0.995**, KIVI **0.981**, PyramidKV **0.895**, SnapKV **0.858**。 | ICLR final Figure 3, Section 2.2 | Figure score equality 不等于所有 token/query error 为零；scorer/protocol 不可与后续 Gao et al. absolute scores 直接混比。 |
| ICLR-final LongBench aggregate | LLaMA-3.1-8B full cache 16-bit average **50.06**；TurboQuant 3.5-bit **50.06**；TurboQuant 2.5-bit **49.74**。 | **ICLR 2026 final Table 1** | 2.5-bit average 是 final correction；arXiv v1 为 49.44，引用必须标 version。 |
| Aggregate neutrality 的边界 | 3.5-bit aggregate 与 full 都是 **50.06**；categories 仍有上下波动，例如 Summarization **26.55→26.00**。 | ICLR final Table 1 | `absolute quality neutrality` 是 rounded aggregate equality，不是 per-task identity，也没有 statistical equivalence test。 |
| Disputed quantization time, archived version only | arXiv v1 Table 2 reported, for $d=200/1536/3072$, PQ **37.04/239.75/494.42 s**；RaBitQ **597.25/2267.59/3957.19**；TurboQuant **0.0007/0.0013/0.0021**。 | **arXiv v1 Table 2, pp. 20; absent from ICLR proceedings final** | Gao et al. report that RaBitQ used translated Python on single-core CPU while TurboQuant used A100, and that neither timing reproduced under symmetric settings. See the critical companion. |

Version discipline：报告最终 paper 时以 ICLR proceedings 为准；arXiv-v1 numbers 只用于显式标注的 version and controversy audit，不应无标签混入同一 row。

## 7. Limitations

### Authors' stated limitations

- 没有独立 `Limitations` section；theory 以 unit-norm input 陈述，一般 vectors 需要存 FP norm 并 rescale。
- Paper 明确说明 MSE-optimized quantizer 对 inner product 有 bias，所以必须增加 residual/QJL stage。

### My critique

- **Internal validity**：theorem 是对 quantizer randomness 的 expected distortion guarantee，不是每个 token、每次 rotation 或每条 trajectory 的 deterministic guarantee。
- **External validity**：KV evidence 仍是少数 LLM/long-context tasks，无 multimodal/action/closed-loop VLA。
- **Systems validity**：algorithm 按定义包含 dense $d\times d$ random $\Pi$ 和 $S$；final Figure 2(c) 给的是 QK computation speedup，Appendix 讨论 fused kernels，但没有完整报告 shared matrix storage、online KV quant/dequant 或 end-to-end serving throughput。
- **Representation overhead**：non-integer bits 需要 outlier split，norm/residual state 和其他 side information 会使 nominal `16/b` compression ratio 与实际 memory ratio 不同。
- **Version/reproducibility**：arXiv v1→ICLR final 的 2.5-bit average 已变化，说明任何复现/汇报都必须固定 official version。
- **RaBitQ dispute**：arXiv v1 timing comparison 使用不对称 implementation/hardware 的指控有具体 numerical evidence；official final 删除 timing table，但仍保留 `RaBitQ lacks vectorized implementation and GPU support` 的描述。把这项争议与 TurboQuant 的 MSE theorem 是否成立分开判断。

## 8. Why it matters for this project

- **VLA**：long-horizon multimodal/action rollouts 会快速增长 KV cache；inner-product preservation 比单纯 reconstruction MSE 更贴近 attention computation。
- **Embodied deployment**：online/data-oblivious 特性适合不断生成的新 states/tokens，但仍需验证 modality-specific outliers、nonstationary norms、temporal error accumulation、success rate 与 P99 control latency。
- **Professor Li's direction**：该工作把 information-theoretic compression、online algorithm 和 accelerator-friendly deployment 连接起来，直接契合 model compression、latency、edge deployment 与 software-hardware co-design。

## 9. How to read it

### 20-minute route

1. Abstract + Section 1.1，分别写下 $D_{mse}$、$D_{prod}$ 和 unbiasedness。
2. 看 Algorithm 1，只理解 `rotate → scalar quantize → inverse rotate`。
3. 看 Algorithm 2，只理解为什么 residual 用最后 1 bit。
4. 读 Theorems 1–3 的 statement，不展开 proof。
5. 核对 ICLR final Table 1：16-bit 50.06、3.5-bit 50.06、2.5-bit 49.74，并记下 version drift。

### 90-minute route

1. **0–15 min**：补 rate-distortion、hypersphere/Beta、Lloyd–Max、unbiased estimator prerequisites。
2. **15–32 min**：读 problem definition，解释 MSE 与 inner-product objectives 为何不同。
3. **32–48 min**：逐步执行 Algorithm 1，推导 coordinate distribution 与 $4^{-b}$ upper bound 的意义。
4. **48–63 min**：执行 Algorithm 2，证明 conditional expectation 如何恢复 unbiasedness，并理解 variance 与 residual norm 的关系。
5. **63–73 min**：对照 Theorem 3 lower bound，区分 optimal rate 与 2.7 constant gap。
6. **73–83 min**：核对 Figure 4、final Table 1、Table 2；拆分 task quality 与 indexing speed。
7. **83–90 min**：列出实际 KV kernel 仍缺的 measurements，并设计一个 VLA cache experiment。

## 10. Reading questions

1. Random rotation 如何把 arbitrary worst-case vector 转成已知 marginal distribution？
2. 为什么 MSE-optimal reconstruction 仍可能产生 biased inner-product estimate？
3. Residual QJL 为什么用 1 bit/dimension 就能恢复 unbiasedness？
4. “Near-optimal” 指 $4^{-b}$ rate、2.7 constant，还是 empirical benchmark equality？三者如何区分？
5. Dense $\Pi,S$ 在实际 attention kernel 中如何存储/生成；Table 2 是否回答了 online serving latency？
6. 为什么 final 2.5-bit average 从 arXiv v1 的 49.44 变为 49.74；复现应固定哪些 protocol/version fields？
7. Expected inner-product distortion 如何转化成 VLA closed-loop safety metric？

## 11. Weekly meeting card

- **Problem**：online KV/vector quantization 要同时有低 distortion、unbiased inner products 和低 preprocessing cost。
- **Key idea**：random rotation + optimal scalar codebook 获得 near-optimal MSE；1-bit residual QJL 恢复 unbiased inner products。
- **Best evidence**：ICLR final LLaMA-3.1-8B LongBench average：16-bit 50.06、3.5-bit 50.06、2.5-bit 49.74。
- **Biggest limitation**：theoretical expectation 不等于 per-trajectory guarantee；完整 transform/kernel/side-info system cost 未报告。
- **Question for the group**：structured fast rotations + fused attention 能否把 $4^{-b}$ theory 变成 VLA 的可测 KV memory/latency/success frontier？

## 12. Status & evidence boundary

- **Status**：**Published as a conference paper at ICLR 2026**；local official proceedings PDF is canonical。
- **Version boundary**：arXiv:2504.19874v1 是较早 primary full text；final Table 1 将 Llama-3.1-8B 2.5-bit average 从 **49.44** 更新为 **49.74**，3.5-bit 保持 **50.06**；arXiv v1 Table 2 timing comparison 不在 proceedings final。
- **Source claim**：problem definitions、Algorithms 1–2、Theorems 1–3、Figures 1–4、final Table 1；旧 timing numbers 只能定位到 archived arXiv v1 Table 2。
- **My interpretation**：对 VLA KV cache、structured transforms、fused kernels 与 closed-loop metrics 的建议。
- **Open question**：multimodal/action task quality、side-information-adjusted compression、real end-to-end serving/control latency。

