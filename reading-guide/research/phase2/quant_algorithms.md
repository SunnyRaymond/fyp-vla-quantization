# Quantization Algorithms — Phase-2 Evidence Packet

> Scope: `SmoothQuant`, `SpinQuant`, `AWQ`, `HAQ`, `TurboQuant`。这是给后续 per-paper README 使用的 evidence packet，不是最终综合 README。术语保留 English；实验数字只记录可在 primary/official full text 定位的 claims。

## Evidence Protocol, Search, and Deduplication

- 检索日期：2026-08-22。按 `ReadingList.md` exact title/arXiv ID 检索，再到 official venue proceedings、arXiv、OpenReview、official repository 核对；未用 blog、新闻稿或第三方综述作 evidence。
- 五篇 official PDF 均逐页提取并检查；TurboQuant 公式另以 official arXiv HTML 对照，排除了 PDF text extraction 对 $\sqrt3\pi/2$ 的歧义。
- SmoothQuant arXiv 2211.10438 与 ICML 2023 是同一 cluster，以 PMLR final 为 canonical；SpinQuant arXiv 2405.16406 与 ICLR 2025/OpenReview `ogO6DGE6FZ` 同 cluster，以 ICLR final 为 canonical。
- AWQ venue title 增加 `On-Device`，authors/arXiv/method 相同，不是另一篇；HAQ arXiv 1811.08886 与 CVPR 2019 同 cluster，以 CVF final 为 canonical。
- TurboQuant 的 first-pass evidence 来自 arXiv:2504.19874v1（2025-04-28）；parent verification 随后核验到 official OpenReview final `tO3ASKZlok`，状态为 ICLR 2026 conference paper。下文已标出发生变化的 Table 1 数字。
- Labels：`verified — official peer-reviewed full text`；或 `verified — official preprint full text; peer review unverified`。

---

## 1. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

### Canonical Metadata

- Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han. ICML 2023, PMLR 202:38087–38099; arXiv:2211.10438.
- [PMLR record](https://proceedings.mlr.press/v202/xiao23c.html) · [official PDF](https://proceedings.mlr.press/v202/xiao23c/xiao23c.pdf) · [arXiv](https://arxiv.org/abs/2211.10438) · [official code](https://github.com/mit-han-lab/smoothquant)
- **Verification:** verified — official peer-reviewed full text.

### Background / Problem

Weight-only quantization 降低 weight traffic，却仍以 FP16 activation 做 GEMM。真正的 `W8A8` 可用 INT8 tensor cores，但 Transformer activations 有少量 channel-wise outliers，per-tensor INT8 dynamic range 会被极端值主导。目标是在不 retraining、无 irregular outlier branch 的条件下，把 linear layers 与 attention BMM 都变成 INT8，并保持 accuracy。

### Method

Section 2, Eq. (1) 的 symmetric uniform quantizer：

$$\bar X_{INT8}=\operatorname{round}(X_{FP16}/\Delta),\qquad \Delta=\max(|X|)/(2^{N-1}-1).$$

关键是 Section 4, Eq. (3) 的 exact equivalent transformation：

$$Y=XW=(X\operatorname{diag}(s)^{-1})(\operatorname{diag}(s)W)=\hat X\hat W.$$

同一 input channel 的 activation 除以 $s_j$，对应 weight row 乘以 $s_j$，FP function 不变；$s$ 可 fuse 到前一层 parameter。Eq. (4)：

$$s_j=\frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}.$$

$\alpha$ 越大，越多 quantization difficulty 从 activation 迁移到 weight。OPT/BLOOM 用 0.5、GLM-130B 0.75、LLaMA experiment 0.8。Flow：`512 random Pile sentences → activation maxima → grid-search α → fuse scales → INT8 Linear/BMM; light element-wise ops stay FP16`。O1 是 per-token dynamic，O2 per-tensor dynamic，O3 per-tensor static。

### Key Innovation

不是新 quantizer，而是用 algebraically exact、hardware-friendly 的 channel scaling 把 activation outliers 转成较易处理的 weight variation，使 regular dense INT8 kernels 可直接使用。

### Main Results — Exact Evidence

| Claim | Exact result | Locator |
|---|---|---|
| OPT-175B quality | FP16 average/PPL **66.9/10.99**；SmoothQuant O3 **66.8/11.17**；naive W8A8 **35.5/93080**；ZeroQuant **35.8/84648**。 | Table 3, paper p. 38092 / PDF p. 6. |
| LLaMA 7B–65B | WikiText-2 PPL FP16 **11.51/10.05/7.53/6.17**；W8A8 **11.56/10.08/7.56/6.20**。 | Table 6, paper p. 38093 / PDF p. 7. |
| Decoding system result | OPT-30B, batch 1, seq 512: **422→314 ms** (1.35×), memory **57→30 GB** (1.91×). OPT-175B, batch 16, seq 512: **2212→1628 ms** (1.36×), **50→30 GB**。 | Table 7, paper p. 38094 / PDF p. 8. |
| 530B deployment | MT-NLG 530B average **73.1→73.1**；GPU count **16→8**。Seq 1024 memory **1095→570 GB**, latency **1707→1689 ms**。 | Tables 8–9, paper p. 38094 / PDF p. 8. |

### Limitations

**Authors/scope:** 没有独立 `Limitations`。Appendix 说明和 GPTQ weight-only speed comparison 不完全公平；batch-1 generation 中 GPTQ 可更快。`W4A4` 留作 future work，主 claim 是 W8A8。

**Our critique:** `lossless` 是 benchmark-level near-equivalence，不是 bit-exact；calibration maxima 和 $\alpha$ 依赖 model/data；kernel speedup 对 architecture、batch、prefill/decode 很敏感。512 Pile sentences 未覆盖 multimodal/action outliers，实验也没有 VLA、continuous action、closed-loop safety。

### VLA Relevance

可用于 Transformer VLM/backbone、multimodal projector、action-token decoder 的 Linear/BMM；但应以 vision/action trajectories 重新 calibration，并单独测 `success rate`, `action L2 error`, `temporal drift`, `P99 latency`，不能只看 LM perplexity。

### Prerequisites / Reading Questions / 组会 3 点

- Prerequisites: symmetric quantization；static/dynamic scale；Transformer residual/LayerNorm/Linear/BMM；activation outlier；INT8 GEMM。
- Questions: Eq. (3) 为什么 FP exact、quantized error 却改变？$\alpha$ 如何交换 activation/weight error？O3 为何更适合 deployment？rare action state 未进 calibration 会怎样？
- 组会：① `offline channel scaling transfers activation outliers into weights`；② OPT-175B 66.9→66.8 且 2212→1628 ms、50→30 GB；③ 研究 trajectory-aware calibration 如何保护 rare action channels。

---

## 2. SpinQuant: LLM Quantization with Learned Rotations

### Canonical Metadata

- Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort. ICLR 2025; arXiv:2405.16406.
- [ICLR record](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html) · [official PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf) · [OpenReview](https://openreview.net/forum?id=ogO6DGE6FZ) · [arXiv](https://arxiv.org/abs/2405.16406) · [official code](https://github.com/facebookresearch/SpinQuant)
- **Verification:** verified — official peer-reviewed full text.

### Background / Problem

Orthogonal rotation 可保持 norm/inner product；在相邻 matrices 中配对 $R,R^\top$，FP function 不变。Hadamard rotation 能扩散 outliers，但 random/function-preserving rotation 不保证相同 quantized accuracy。目标是在 `W4A4`, `W4A8`, `KV4/KV8` 中学习最小化 quantized loss 的 orthogonal rotations。

### Method

- $R_1$：residual stream，fuse into weights；$R_2$：head-wise value/output pair，可 fuse；$R_3$：KV path online Hadamard；$R_4$：FFN/down-projection online Hadamard。
- `SpinQuant_no_had` 只学/fuse $R_1,R_2$；`SpinQuant_had` 加 $R_3,R_4$，accuracy 更强但有 online overhead。

Section 3, Eq. (2)：

$$\min_{R_1,R_2\in\mathrm{Stiefel}}\mathcal L_Q(R_1,R_2\mid W,X),$$

其中 $W$ frozen，fake quantization 参与 backprop。Cayley update，Eqs. (3)–(4)：

$$R'=(I-\eta Y/2)^{-1}(I+\eta Y/2)R,$$
$$Y=\hat G-\hat G^\top,\qquad \hat G=GR^\top-\tfrac12RR^\top GR^\top,$$

保持 $R'^\top R'=I$。Flow：`800 WikiText-2 samples → 100 iterations, LR 1.5→0 → freeze/fuse rotations → GPTQ using 128 WikiText-2 sequences × 2048 tokens → inference`。$R_1,R_2$ 约为 model weights 的 **0.26%**。

### Key Innovation

把 rotation 从 fixed preprocessing 变为 Stiefel-manifold optimization variable：`function invariance in real arithmetic ≠ quantization invariance`，应直接优化 quantization geometry。

### Main Results — Exact Evidence

| Claim | Exact result | Locator |
|---|---|---|
| Rotation sensitivity | LLaMA-2 7B W4A4 的 100 random rotations，best–worst 最多约 **13 average points**；random Hadamard variance 可达约 **6**。 | Figure 4, Section 2.2, PDF pp. 4–5. |
| LLaMA-2 7B W4A4KV4 | FP16 average/PPL **66.9/5.5**；SpinQuant_had **64.0/5.9**；LLM-QAT **44.9/14.9**；SmoothQuant **39.0/698.7**。 | Table 1, PDF p. 7. |
| Learned vs random | Mistral-7B W4A4KV4 **52.4→68.6** (+16.2)；LLaMA-3 8B **63.9→65.5**。 | Table 2, PDF p. 8. |
| QuaRot comparison | LLaMA-3 70B FP16 **74.5/2.8**；QuaRot+GPTQ W4A4KV4 **65.1/20.2**；SpinQuant **69.3/5.5**。8B: **63.3/8.0→65.5/7.3**。 | Table 5, PDF p. 9. |
| Runtime | Mac M1, LLaMA-3 8B: FP16 **177.15 ms/token**；no_had W4A8 **58.88** (~3.01×)；had **63.90** (~2.77×)。 | Table 6, PDF p. 10. |

Abstract 的 “LLaMA-3 8B gap reduced by up to 45.1% vs QuaRot” 无法由主表唯一重算；汇报应引用 Table 5 absolute scores。

### Limitations

**Authors/scope:** 无独立 `Limitations`；承认 online Hadamard compute overhead，把 theoretical optimal rotation 留作 future work。

**Our critique:** 不是 training-free：要 backprop/calibration/manifold optimization，W4 headline 还依赖 GPTQ。Optimization 从小模型十几分钟到 LLaMA-2 70B **3.5 h**（Table 15）。主要是 language zero-shot/PPL，无 VLM/VLA；Mac `3×` 是 W4A8，不能等同于 W4A4KV4 quality setting。

### VLA Relevance

可能改善 multimodal/action decoder outliers 和 long-horizon KV；但 calibration 必须含 vision/action trajectories，需测 head/RoPE layout、continuous action tail error 与 online $R_3/R_4$ latency。

### Prerequisites / Reading Questions / 组会 3 点

- Prerequisites: orthogonal/Stiefel/Cayley；Hadamard；fake quantization；GPTQ；Transformer tensor flow。
- Questions: FP-preserving rotations 为何 quantized accuracy 不同？$R_1$–$R_4$ 哪些可 fuse？Cayley 如何保持正交？VLA objective 应是 language loss 还是 action loss？
- 组会：① `learn function-preserving rotations because quantization error is rotation-dependent`；② LLaMA-2 7B W4A4KV4 66.9→64.0、Mistral learned 68.6 vs random 52.4；③ 用 action-aware loss 学 rotations。

---

## 3. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

### Canonical Metadata

- Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han. MLSys 2024, *Proceedings of Machine Learning and Systems* 6:87–100; arXiv:2306.00978. Venue title includes `On-Device`.
- [MLSys record](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) · [official PDF](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) · [arXiv](https://arxiv.org/abs/2306.00978) · [official code](https://github.com/mit-han-lab/llm-awq)
- **Verification:** verified — official peer-reviewed full text.

### Background / Problem

Autoregressive generation 常 memory-bound。Weight-only W3/W4 可降 bandwidth/capacity，但 RTN accuracy 差；保留少量 FP16 salient weights 会破坏 regular packing。目标是在无 backprop/昂贵 reconstruction 下找到 salient channels，并以 hardware-friendly regular low-bit format 保护它们。

### Method

Activation magnitude 用来识别 important input channels；保留 activation-selected 0.1%–1% weights 的诊断实验显著优于 weight magnitude/random，但 final AWQ 不采用 mixed precision。

Equivalent scaling：$WX=(W\operatorname{diag}(s))(\operatorname{diag}(s)^{-1}X)$。若 scaling 后 group scale $\Delta'$ 未同比增长，salient channel relative error 约乘 $(\Delta'/\Delta)/s$。

Section 3, Eq. (4)：

$$s^*=\arg\min_s\|Q(W\operatorname{diag}(s))\operatorname{diag}(s)^{-1}X-WX\|.$$

Eq. (5) 降成 1-D search：

$$s=s_X^\alpha,\qquad \alpha^*=\arg\min_{\alpha\in[0,1]}\mathcal L(s_X^\alpha),$$

$s_X$ 是 per-input-channel average activation magnitude，20-point grid search；随后 clipping。Flow：`small Pile calibration → stats → per-layer α search → scaling+clipping → W3/W4 group size 128 → packing → TinyChat on-the-fly dequant, SIMD layout, kernel fusion`。

### Key Innovation

用 activation 作为 weight saliency proxy，并用 regular scaling 代替 irregular mixed precision；既保护重要 channels，又保留统一 packed low-bit layout。

### Main Results — Exact Evidence

| Claim | Exact result | Locator |
|---|---|---|
| Saliency diagnostic | OPT-6.7B W3-g128 PPL: RTN **23.54**；keep 1% activation-selected **11.39**，weight-selected **22.37**，random **24.23**。 | Table 1, PDF p. 4. |
| OPT W3-g128 | 1.3/2.7/6.7/13/30B AWQ **16.32/13.58/11.39/10.56/9.77**；RTN **119.47/298.00/23.54/46.04/18.80**。 | Table 3, PDF p. 4. |
| Llama quality | W4-g128 Llama-2 13B/70B: FP16 **4.88/3.32**, AWQ **4.97/3.41**, GPTQ **4.98/3.42**。LLaMA-7B: **5.68/5.78/6.22**。 | Table 4, PDF p. 7. |
| Multimodal | OpenFlamingo-9B COCO CIDEr 32-shot: FP16 **81.70**, W4 AWQ **80.53**, RTN **77.13**, GPTQ **74.98**；W3 AWQ **74.47** vs RTN/GPTQ **64.79/64.77**。 | Table 6, PDF p. 8. |
| Throughput | VILA-7B tokens/s A100 **81.6→155.3**, 4090 **58.5→168.1**, Orin **11.5→35.6**；VILA-13B **48.5→102.1**, `OOM→99.0`, **6.1→17.5**。 | Table 10, PDF pp. 10–11. |

Table 10 是 `AWQ + TinyChat system`，不能归因于 scaling algorithm alone。

### Limitations

**Authors/scope:** Section 3.1 明确指出 naive mixed-precision keep-set 难获 hardware speedup；final paper 无独立 `Limitations`。

**Our critique:** weight-only 不压 activations/KV；average activation magnitude 可能漏掉 low-frequency/high-consequence action channels；`lossless` 不是每项相同（VILA-7B VizWiz Table 7: **59.6→57.8**）。OpenFlamingo 主要量化 language component；speedup 还包含 TinyChat，且 group size 128/packing hardware-specific。

### VLA Relevance

已有 VLM/VILA evidence，比纯 text PTQ 更接近 VLA；适合 edge robot memory/bandwidth。但 vision encoder、proprioception、continuous action head、KV cache 仍需单测，calibration 应覆盖 contact transitions 与 rare failures。

### Prerequisites / Reading Questions / 组会 3 点

- Prerequisites: group-wise weight-only quantization；RTN/GPTQ；memory-bound generation；saliency；packing/dequant kernel。
- Questions: activation magnitude 为何优于 weight magnitude？$s$ 太大为何损害同 group weights？AWQ 与 SmoothQuant scaling 的 target/objective 有何不同？TinyChat 贡献如何剥离？
- 组会：① `activation-aware scaling protects salient channels in a regular W3/W4 format`；② COCO 81.70→80.53、Orin 11.5→35.6 tokens/s；③ 用 action-loss sensitivity 替代 average activation saliency。

---

## 4. HAQ: Hardware-Aware Automated Quantization With Mixed Precision

### Canonical Metadata

- Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin, Song Han. CVPR 2019, pp. 8612–8620; arXiv:1811.08886.
- [CVF record](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.html) · [official PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_HAQ_Hardware-Aware_Automated_Quantization_With_Mixed_Precision_CVPR_2019_paper.pdf) · [arXiv](https://arxiv.org/abs/1811.08886) · [official code](https://github.com/mit-han-lab/haq)
- **Verification:** verified — official peer-reviewed full text.

### Background / Problem

Uniform bits 通常非 Pareto-optimal；mixed precision 要在 accuracy 与 latency/energy/model-size 间分配。FLOPs 不能可靠预测硬件 latency，因为 depthwise conv、memory access、parallelism 不同。目标是为 target accelerator 自动选择每层 weight/activation bits 并满足硬 constraint。

### Method

DDPG actor-critic 每层做 weight、activation 两次 decision。Conv state，Eq. (1)：

$$O_k=(k,c_{in},c_{out},s_{kernel},s_{stride},s_{feat},n_{params},i_{dw},i_{w/a},a_{k-1}),$$

FC 用 Eq. (2) analog。Continuous $a_k\in[0,1]$ 映射整数 bits，Eq. (3)：

$$b_k=\operatorname{round}(b_{min}-0.5+a_k(b_{max}-b_{min}+1)),$$

$b_{min}=2,b_{max}=8$。若超 budget，顺序降 bits 直到满足。

Quantizer，Eq. (4)：

$$q(w,a_k,c)=\operatorname{round}(\operatorname{clamp}(w,c)/s)s,\qquad s=c/(2^{a_k-1}-1),$$

$c$ 由 Eq. (5) KL-divergence minimization 选，activation range $[0,c]$。Hardware simulator 直接返回 latency/energy；constraint 已强制，reward 只看 accuracy：

$$R=\lambda(acc_{quant}-acc_{origin}),\qquad\lambda=0.1.$$

每 episode fine-tune 1 epoch；actor/critic hidden 400/300，exploration $\sigma=0.5$ ×0.99 decay；ImageNet-100 搜索，full ImageNet final fine-tune。

### Key Innovation

用 target hardware simulator 的 `direct latency/energy feedback` 替代 FLOPs/bit-ops proxy，使同一 network 在 edge/cloud 学到不同 mixed-precision policy。

### Main Results — Exact Evidence

| Claim | Exact result | Locator |
|---|---|---|
| BISMO edge/cloud | MobileNet-V1 original 8-bit **70.82**, edge/cloud **96.20/151.09 ms**。HAQ edge **70.58/57.70** (1.67×)；cloud **69.97/77.49** (1.95×)。 | Table 3, PDF p. 6. |
| BitFusion latency | Original 8/8 **70.82/20.08 ms**；HAQ flexible **70.40/11.09** (~1.81×)，或 **70.90/19.98**。 | Table 4, PDF p. 7. |
| BitFusion energy | Original **70.82/31.03 mJ**；HAQ **70.37/16.30** (~1.90×)，或 **70.90/26.67**。 | Table 5, PDF p. 7. |
| Tight model size | ~2-bit: MobileNet-V1 Deep Compression **37.62/1.09 MB**, HAQ **57.14/1.09**；MobileNet-V2 **58.07/0.96** vs **66.75/0.95**；ResNet-50 **68.95/6.32** vs **70.63/6.30**。 | Table 6, PDF p. 8. |

Table 5 exact HAQ 是 **16.30 mJ**；相邻 prose 的 **16.57** 对应 PACT row，`2×` 只是 rounded summary。

### Limitations

**Authors/scope:** 无独立 `Limitations`；policy 明确 hardware-specific，换 accelerator 需重搜。

**Our critique:** 只测 2019 CNN/ImageNet；RL search 反复 fine-tuning，未充分报告 total compute、seed variance/CI。Feedback 来自 simulator，受 fidelity 限制；sequential budget repair 不保证 global optimum。State 缺 Transformer/KV/context/batch/token phase；static budget 也不表达 robot P99 latency、thermal/battery dynamics。

### VLA Relevance

价值主要是 hardware-software co-design：为 vision encoder、projector、language/action decoder、KV、control head 分配 precision，并在 robot SoC 优化 success + P99 latency + energy。原 policy 不可直接复用。

### Prerequisites / Reading Questions / 组会 3 点

- Prerequisites: mixed precision；MDP/DDPG；accelerator simulator；KL clipping；Pareto/hard constraint。
- Questions: FLOPs 为何不等于 latency？哪些 state feature 造成 edge/cloud policy 分化？sequential repair 会错过什么？VLA reward 如何组合 success/P99/energy？
- 组会：① `DDPG allocates bits from target-hardware latency/energy feedback`；② cloud 151.09→77.49 ms、accuracy 70.82→69.97；③ 将 state 扩展成 `module × modality × context × hardware state`。

---

## 5. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

### Canonical Metadata

- Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni. ICLR 2026; earlier version arXiv:2504.19874v1 [cs.LG], 2025-04-28.
- [ICLR final / OpenReview PDF](https://openreview.net/pdf?id=tO3ASKZlok) · [arXiv](https://arxiv.org/abs/2504.19874) · [arXiv v1 HTML](https://arxiv.org/html/2504.19874)
- **Verification:** verified — official ICLR conference PDF plus earlier arXiv full text. The ICLR final is canonical for publication status and changed Table 1 values.

### Background / Problem

Streaming KV cache 和 vector search 都依赖 inner product。Dataset-specific PQ 要 k-means/codebook training；online scalar quantization 又缺 optimal rate。MSE-optimal reconstruction 还可能产生 inner-product bias。定义 $Q:\mathbb R^d\to\{0,1\}^{B}$，$b=B/d$，分别最小化：

$$D_{mse}=\mathbb E_Q\|x-Q^{-1}Q(x)\|_2^2,$$
$$D_{prod}=\mathbb E_Q|\langle y,x\rangle-\langle y,Q^{-1}Q(x)\rangle|^2,$$

并要求 inner-product estimator unbiased、data-oblivious/online、接近 Shannon lower bound。

### Method

**MSE TurboQuant (Algorithm 1).** Gaussian random matrix QR 得 orthogonal $\Pi$，$z=\Pi x$。Unit-sphere coordinate density：

$$f_X(t)=\frac{\Gamma(d/2)}{\sqrt\pi\Gamma((d-1)/2)}(1-t^2)^{(d-3)/2},$$

高维趋近 $\mathcal N(0,1/d)$。预计算 1-D Lloyd–Max 的 $2^b$ centroids；online 每个 $z_j$ 只存最近 centroid index，dequant 后乘 $\Pi^\top$。Theorem 1：

$$D_{mse}\le\frac{\sqrt3\pi}{2}4^{-b},$$

$b=1,2,3,4$ tighter values **0.36, 0.117, 0.03, 0.009**。一般 vector 另存 FP norm。

**Inner-product TurboQuant (Algorithm 2).** 先用 $b-1$ bit MSE stage，residual $r=x-\tilde x_{mse}$；最后 1 bit QJL：

$$q=\operatorname{sign}(Sr),\ S_{ij}\sim\mathcal N(0,1),$$
$$\tilde x=\tilde x_{mse}+\frac{\sqrt{\pi/2}}{d}\|r\|_2S^\top q.$$

Theorem 2：$\mathbb E\langle y,\tilde x\rangle=\langle y,x\rangle$，且

$$D_{prod}\le\frac{\sqrt3\pi^2\|y\|_2^2}{d}4^{-b}.$$

$b=1,2,3,4$ 分别约 **1.57/d, 0.56/d, 0.18/d, 0.047/d**。

**Lower bound (Theorem 3).** Shannon lower bound + Yao minimax 给 hard instances：

$$D_{mse}\ge4^{-b},\qquad D_{prod}\ge(\|y\|_2^2/d)4^{-b}.$$

所以 MSE rate 匹配 $4^{-b}$，差的 constant 最多 $\sqrt3\pi/2\approx2.7$。

KV flow：`streaming K/V → split outlier/regular channels → two TurboQuant bit allocations → store indices/norm/residual state → attention dequant`。2.5-bit example 是 32 channels ×3 bits + 96×2 bits。

### Key Innovation

Random rotation 把 worst-case input 变成 distribution-known coordinates，用 scalar codebook 达到近 optimal vector rate；residual QJL 再修复 MSE quantizer 的 inner-product bias。

### Main Results — Exact Evidence

| Claim | Exact result | Locator |
|---|---|---|
| Needle retrieval | LLaMA-3.1-8B-Instruct, 4k–104k, memory ratio 0.25: Full **0.997**, Turbo **0.997**, Polar **0.995**, KIVI **0.981**, Pyramid **0.895**, Snap **0.858**。 | Figure 4, Section 4.2. |
| LongBench | LLaMA-3.1-8B full 16-bit average **50.06**；Turbo 3.5-bit **50.06**；2.5-bit **49.74**；Polar 3.9-bit **49.78**；KIVI 3-bit **48.50**。 | ICLR final Table 1, Section 2.3, PDF p. 9. Earlier arXiv v1 reported 49.44 for 2.5-bit. |
| Equality is aggregate only | Full→3.5 bit: SingleQA **45.29→45.01**, MultiQA **45.16→45.31**, Summarization **26.55→26.00**, Few-shot **68.38→68.63**, Synthetic **59.54→59.95**, Code **46.28→46.17**。 | Table 1, PDF p. 20. |
| Quantization time | 4-bit, 100k vectors, d=200/1536/3072: PQ **37.04/239.75/494.42 s**；RabitQ **597.25/2267.59/3957.19**；Turbo **0.0007/0.0013/0.0021**。All experiments: one A100. | Table 2 and Sections 4, 4.4, PDF p. 20. |

ICLR final Section 2.3 prose 写 `LongBench-E`，Table 1 caption 写 `LongBench-V1`，paper 内部不一致。`absolute quality neutrality` 只指 aggregate 50.06 恰相同，没有 per-category identity/statistical equivalence。

### Limitations

**Authors/scope:** 无独立 `Limitations`；unit-norm assumption 要额外 FP norm；MSE estimator 的 inner-product bias 要 residual/QJL stage 修复。

**Our critique:** 即使已有 ICLR 2026 publication record，Algorithm 按文中定义仍使用 dense $d\times d$ random $\Pi,S$；论文给 indexing time，却没完整报告 shared-matrix storage、online KV quant/dequant latency、fused attention 或 end-to-end serving throughput。Non-integer setup 又加 outlier split/side metadata；nominal 4.5×/5× 需扣除 overhead。Theory 是 expectation distortion，不是 per-token/action safety guarantee。只测两个 LLM family，无 VLA；near-neighbor Figure 无 exact recall table。

### VLA Relevance

Long-horizon multimodal/action tokens 令 KV cache 增长，inner-product guarantee 比单纯 MSE 更贴近 attention。但需验证 modality outliers、nonstationary norms、temporal accumulation、action success 和 P99 control latency，并把 transforms fuse 到真实 attention kernel。

### Prerequisites / Reading Questions / 组会 3 点

- Prerequisites: rate-distortion/Shannon；random rotation/hypersphere Beta；Lloyd–Max；JL/QJL；unbiased/variance；attention/KV。
- Questions: rotation 如何处理 worst-case input？MSE-optimal 为何 inner-product biased？QJL 为何恢复 unbiasedness？$4^{-b}$ 与 2.7 constant 如何解读？dense $\Pi,S$ 的实际成本是什么？
- 组会：① `random rotation + scalar codebook gives near-optimal MSE; 1-bit residual QJL gives unbiased inner products`；② LongBench average 50.06=50.06，但 summarization 26.55→26.00；③ 用 structured rotations + fused attention 做 VLA system test。

---

## Suggested Dimensions for the Later Algorithm Comparison

| Dimension | Comparison prompt |
|---|---|
| Quantized object | weights / activations / KV / per-layer W+A 是否覆盖真实 bottleneck？ |
| Precision and granularity | W4-g128、W8A8、layer-wise 2–8 bits、2.5/3.5 average bits 能否直接比较？ |
| Transformation | saliency scaling、outlier smoothing、learned/random rotation、bit allocation 改变了什么？ |
| Objective | output reconstruction、task loss、hardware cost、MSE、unbiased inner product 各保护什么？ |
| Calibration/training | data-oblivious；light calibration；backprop+GPTQ；RL+fine-tuning 的成本/可移植性。 |
| FP function preservation | SmoothQuant/AWQ/SpinQuant 的 FP algebra equivalent 不等于 quantized behavior equivalent。 |
| Runtime integration | TinyChat packing、Hadamard overhead、INT8 kernels、Turbo transforms 是否真正 fused？ |
| Hardware specificity | device/backend 换了是否需重搜、重调、重 benchmark？ |
| Evidence domain | CNN、LLM、VLM、KV-cache 与 closed-loop VLA 的 evidence distance。 |
| Claim type | theorem、accuracy benchmark、memory、latency、energy 不应混成一个 `better`。 |
| Failure mode | rare activation/action channels、rotation sensitivity、simulator mismatch、side-info overhead。 |
| VLA metrics | success rate、action error、temporal drift、P99 latency、energy、thermal stability。 |

核心比较问题：**方法改变的是 value distribution、bit allocation，还是 execution path；优化的是 reconstruction、task behavior、inner-product geometry，还是 measured device cost？** 这比只比较 “W4/W8” 更能判断方法能否组合，以及结论能否迁移到 VLA。
