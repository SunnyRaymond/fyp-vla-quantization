# SpinQuant: LLM Quantization with Learned Rotations

> **Reading-list role**: Core — learned rotations for W/A/KV low-bit quantization  
> **Verification**: `verified-full-text` — ICLR 2025 official proceedings/OpenReview final  
> **Recommended effort**: Deep read；先完成 [QuaRot 20-minute bridge](../12a-quarot/README.md#20-minute-bridge-route-for-your-current-spinquant-reading)，再看 manifold optimization

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort |
| Year / version | 2025; arXiv:2405.16406 |
| Venue / status | ICLR 2025; peer-reviewed venue final |
| Primary source | [ICLR record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html) · [official PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf) · [OpenReview](https://openreview.net/forum?id=ogO6DGE6FZ) · [arXiv](https://arxiv.org/abs/2405.16406) |
| Code / project | [official repository](https://github.com/facebookresearch/SpinQuant) |

## 2. One-sentence takeaway

SpinQuant 不接受任意/random function-preserving rotation，而是在 Stiefel manifold 上学习使 quantized loss 最小的 orthogonal rotations；LLaMA-2 7B 在 `W4A4KV4` 下由 FP16 average 66.9 变为 64.0，而 Mistral-7B learned rotation 比 random Hadamard 高 16.2 average points。

## 3. Background and prerequisites

- Orthogonal matrix：$R^\top R=I$，保持 $L_2$ norm 与 inner product；paired $R,R^\top$ 可在 FP graph 中保持 function。
- Hadamard transform：快速、结构化的 outlier spreading；理解它为何可降低 low-bit uniform quantizer 的 dynamic-range pressure。
- [QuaRot](../12a-quarot/README.md)：fixed/randomized Hadamard rotation predecessor；先区分其 rotation construction 与默认 GPTQ calibration，再理解 SpinQuant 为什么学习 $R_1,R_2$。
- Stiefel manifold、Riemannian/constrained optimization、Cayley transform。
- Fake quantization、straight-through gradient、GPTQ；区分 rotation learning 与随后 weight quantization。
- Transformer residual stream、attention value/output projections、KV cache、FFN down projection。

## 4. Problem

- **Target setting**：frozen LLM 的 aggressive `W4A4`, `W4A8`, `KV4/KV8` quantization。
- **Bottleneck**：function-preserving rotation 并不保证 quantization error 相同；random/Hadamard choice 可使 accuracy 相差很多。
- **Why previous methods are insufficient**：fixed QuaRot-style rotations 去 outliers，但没有利用 task/calibration loss 选择 quantization-optimal orientation；random rotation 的 variance 令结果不可预测。

## 5. Method

### 5.1 System view

`WikiText calibration → insert function-preserving rotations → fake-quantize activations → backprop quantized loss → Cayley updates preserve orthogonality → freeze/fuse R1/R2 → GPTQ weights → optionally execute online Hadamard R3/R4 → W/A/KV low-bit inference`

$R_1$ 旋转 residual stream，可 fuse；$R_2$ 是 attention head 内 value/output paired rotation，可 fuse；$R_3$ 在线作用于 KV path；$R_4$ 在线作用于 FFN/down-projection。`SpinQuant_no_had` 只学/fuse $R_1,R_2$；`SpinQuant_had` 加 online $R_3,R_4$。

### 5.2 Core mechanism

Section 3, Eq. (2)：

$$
\min_{R_1,R_2\in\mathrm{Stiefel}}\mathcal L_Q(R_1,R_2\mid W,X),
$$

$W$ frozen，$\mathcal L_Q$ 通过 simulated quantization 反向传播。Cayley update，Eqs. (3)–(4)：

$$
R'=(I-\eta Y/2)^{-1}(I+\eta Y/2)R,
$$

$$
Y=\hat G-\hat G^\top,\qquad
\hat G=GR^\top-\tfrac12RR^\top GR^\top.
$$

$Y$ 是 skew-symmetric，update 保持 $R'^\top R'=I$。Calibration/optimization 使用 800 WikiText-2 samples、100 iterations、learning rate 1.5 linearly decay 到 0。之后 GPTQ 使用 128 条 length-2048 WikiText-2 sequences。$R_1,R_2$ parameters 约为 model weights 的 0.26%。

### 5.3 What is actually new

新机制是 `learn the rotation under a quantized objective while remaining on the orthogonal manifold`。Function-preserving placement 和 Hadamard outlier spreading 是 lineage；GPTQ 是随后产生 low-bit weights 的 standard component；headline quality 不是单独由 GPTQ 或 random rotation得到。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Rotation choice 非无关 | LLaMA-2 7B W4A4 的 100 random rotations，best–worst 最多约 **13 average points**；random Hadamard variance 可达约 **6**。 | Figure 4, Section 2.2, PDF pp. 4–5 | Figure-level approximate claim，不应写成精确 13.0。 |
| Aggressive W4A4KV4 | LLaMA-2 7B FP16 average/PPL **66.9/5.5**；SpinQuant_had **64.0/5.9**；LLM-QAT **44.9/14.9**；SmoothQuant **39.0/698.7**。 | Table 1, PDF p. 7 | 与 FP16 gap 2.9 points；baseline setting 要逐列对齐。 |
| Learned beats random | Mistral-7B W4A4KV4 random Hadamard **52.4**, learned **68.6** (+16.2)；LLaMA-3 8B **63.9→65.5**。 | Table 2, PDF p. 8 | Gain 对 model 很不均匀，不能统一说“+16 points”。 |
| Beats QuaRot rows | LLaMA-3 70B FP16 **74.5/2.8**；QuaRot+GPTQ W4A4KV4 **65.1/20.2**；SpinQuant **69.3/5.5**。8B：**63.3/8.0→65.5/7.3**。 | Table 5, PDF p. 9 | Abstract 的 “45.1% gap reduction” 无法由主表唯一重算，优先引用 absolute scores。 |
| Runtime trade-off | Mac M1, LLaMA-3 8B：FP16 **177.15 ms/token**；no_had W4A8 **58.88** (~3.01×)；had **63.90** (~2.77×)。 | Table 6, PDF p. 10 | 这是 W4A8 runtime，不是 W4A4KV4 quality row。 |

## 7. Limitations

### Authors' stated limitations

- 没有独立 `Limitations` section；runtime discussion 明确承认 online Hadamard 带来额外 compute。
- Theoretical optimal rotation 被留作 future work；`no_had` 与 `had` 是 accuracy/overhead trade-off。

### My critique

- **Internal validity**：不是 training-free PTQ；需要 backprop、calibration、manifold optimization，headline W4 weights 还依赖 GPTQ。应把 rotation gain 与 GPTQ gain 分离。
- **External validity**：主要是 text zero-shot/PPL，无 vision encoder、multimodal projector 或 continuous control。
- **Systems validity**：可 fuse 的 $R_1,R_2$ 与 online $R_3,R_4$ 成本不同；Mac speed claim 的 precision setting 与 strongest quality setting 不同。
- **Reproducibility**：rotation optimization 随 model scale 增长；Table 15 报 LLaMA-2 70B 约 **3.5 h**，并依赖 calibration/optimizer details。

## 8. Why it matters for this project

- **VLA**：vision/action tokens 和 policy KV cache 也可能有 outliers；learned rotation 可望保护 multimodal/action decoder 的 low-bit geometry，$R_3$ 对 long-horizon cache 尤其相关。
- **风险**：rotation objective 若只用 language calibration，可能优化 text loss 却伤害 action tails；应加入 action-aware or multi-objective loss。
- **Professor Li's direction**：它将 model reparameterization、quantization 和 kernel overhead 放在同一设计中，连接 model compression、latency 与 software-hardware co-design。

## 9. How to read it

### 20-minute route

1. Abstract + Figure 1，定位 $R_1$–$R_4$ 在 Transformer 的位置。
2. 读 Figure 4，先接受“rotation choice matters”。
3. 读 Eq. (2)–(4)，只回答 objective 与 constraint 分别是什么。
4. 核对 Table 1 的 LLaMA-2 7B W4A4KV4 row。
5. 对比 Table 6 的 no_had/had latency，写出 accuracy/overhead trade-off。

### 90-minute route

1. **0–15 min**：补 orthogonal/Hadamard/Stiefel/Cayley prerequisites。
2. **15–30 min**：画 $R_1$–$R_4$ placement，标注 fuse vs online。
3. **30–48 min**：逐行解释 Eq. (2)–(4)，说明 Cayley update 为什么保持正交。
4. **48–60 min**：还原 calibration→optimization→GPTQ pipeline，避免误称 training-free。
5. **60–73 min**：核对 Figure 4、Tables 1–2，验证 rotation sensitivity 与 learned gain。
6. **73–82 min**：读 Tables 5–6、15，区分 quality、runtime、optimization cost。
7. **82–90 min**：定义 VLA multi-objective rotation loss，并列出 action-specific metrics。

## 10. Reading questions

1. 为什么所有 orthogonal rotations 都保持 FP function，却产生不同 quantized accuracy？
2. $R_1$–$R_4$ 中哪些能 fuse，哪些必须在线执行，为什么？
3. Cayley update 如何保证 orthogonality；普通 SGD 后 re-normalization 有何不同？
4. Mistral +16.2 而 LLaMA-3 8B +1.6，说明 rotation benefit 依赖什么？
5. SpinQuant 与 TurboQuant 都 rotation，它们是 learned vs random、task loss vs rate-distortion 的什么差异？
6. VLA 应用 language loss、action loss，还是 risk-sensitive multi-objective loss 学 rotation？

## 11. Weekly meeting card

- **Problem**：random/function-preserving rotation 的 quantized quality 方差很大。
- **Key idea**：在 Stiefel manifold 上直接学习最小化 quantized loss 的 rotations，并把可 fuse 与 online rotations 分开。
- **Best evidence**：LLaMA-2 7B W4A4KV4 66.9→64.0；Mistral learned 68.6 vs random Hadamard 52.4。
- **Biggest limitation**：需要 calibration/backprop/GPTQ，且没有 VLM/VLA evidence；online Hadamard 有 latency。
- **Question for the group**：action-aware rotation loss 能否保护 rare control dimensions而不牺牲 language capability？

## 12. Status & evidence boundary

- **Status**：peer-reviewed ICLR 2025 venue final；arXiv/OpenReview/ICLR 是同一 work cluster。
- **Source claim**：$R_1$–$R_4$ placement、Eqs. (2)–(4)、calibration counts、Figure 4、Tables 1/2/5/6/15。
- **My interpretation**：VLA action-aware rotation objective 与对 long-horizon policy KV 的意义。
- **Open question**：multimodal/action calibration、closed-loop quality、online rotations 在 robot backend 的真实 overhead。
