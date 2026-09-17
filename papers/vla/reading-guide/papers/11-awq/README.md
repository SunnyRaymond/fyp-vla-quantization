# AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

> **Reading-list role**: Core — weight-only PTQ / on-device LLM and VLM deployment  
> **Verification**: `verified-full-text` — MLSys 2024 official final  
> **Recommended effort**: Core read；优先和 SmoothQuant 对照

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han |
| Year / version | 2024; arXiv:2306.00978 |
| Venue / status | MLSys 2024, *Proceedings of Machine Learning and Systems* 6:87–100; venue title adds `On-Device` |
| Primary source | [MLSys record](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html) · [official PDF](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) · [arXiv](https://arxiv.org/abs/2306.00978) |
| Code / project | [official repository](https://github.com/mit-han-lab/llm-awq) |

## 2. One-sentence takeaway

AWQ 用 activation magnitude 找到影响输出的 salient input channels，再以 per-channel equivalent scaling 降低这些 weights 的 relative quantization error，同时保持 regular packed W3/W4 layout；在 OpenFlamingo-9B 的 32-shot COCO 上，W4 CIDEr 为 80.53、接近 FP16 的 81.70。

## 3. Background and prerequisites

- 理解 weight-only `W4A16/W3A16`、group-wise quantization、group size、RTN 与 GPTQ。
- 理解 autoregressive decode 常是 memory-bound：每生成一个 token 都要搬运 weights，因此压 weights 会直接减少 bandwidth。
- 区分 three signals：weight magnitude、activation magnitude、output reconstruction error。
- 理解 packed low-bit weights、on-the-fly dequantization、SIMD-aware layout、kernel fusion。
- Symbols：$W$ 是 weight，$X$ 是 calibration activation，$s_X$ 是 per-input-channel average activation magnitude，$s$ 是 scaling，$\alpha$ 是 1-D search variable。

## 4. Problem

- **Target setting**：无需 backprop 的 weight-only PTQ，通常是 W4/W3、activation FP16、group size 128。
- **Bottleneck**：RTN 在 3–4 bit 下误差大；保留少量 salient weights 为 FP16 虽可恢复 quality，却形成 irregular mixed precision，难以得到 hardware speedup。
- **Why previous methods are insufficient**：按 weight magnitude 选择“重要 weights”并不等于对当前 activation/output 重要；GPTQ/block reconstruction 更重，且 calibration/generalization 成本更高。

## 5. Method

### 5.1 System view

`small Pile calibration set → collect activation statistics → search per-layer α → scale salient input channels → clip weights → W3/W4 group quantization → pack → TinyChat dequant/layout/fusion → on-device inference`

Algorithm 与 system 应分开：AWQ 决定 scale/clipping/quantized weights；TinyChat 才把 packed representation 转成设备上的 throughput。

### 5.2 Core mechanism

Diagnostic observation：只保留 0.1%–1% activation-selected salient weights 为 FP16，效果显著优于 weight-selected/random；但 final method 不采用 irregular keep-set。

Equivalent scaling：

$$
WX=(W\operatorname{diag}(s))(\operatorname{diag}(s)^{-1}X).
$$

若 channel weight 乘 $s>1$ 后，所在 group 的 quantization step 从 $\Delta$ 变为 $\Delta'$ 且增长较慢，则该 channel 的 relative error 约乘 $(\Delta'/\Delta)/s$，从而被保护。

Section 3, Eq. (4)：

$$
s^*=\arg\min_s\left\|Q(W\operatorname{diag}(s))\operatorname{diag}(s)^{-1}X-WX\right\|.
$$

Eq. (5) 把高维搜索 parameterize 为：

$$
s=s_X^\alpha,\qquad
\alpha^*=\arg\min_{\alpha\in[0,1]}\mathcal L(s_X^\alpha).
$$

Paper 使用 20-point grid search，并在 scaling 后进行 clipping；inverse scale 可 fuse 到前一 operation。

### 5.3 What is actually new

真正新增的是 `activation-aware weight saliency + regular scaling`：用当前 layer input 的 statistics 估计哪些 weight channels 值得保护，同时不破坏 packed low-bit format。W4 grouping、uniform quantizer、dequant kernel 和 fusion 是 supporting engineering；TinyChat speedup 不能全部归给 saliency algorithm。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Activation 是有效 saliency signal | OPT-6.7B W3-g128 PPL：RTN **23.54**；keep 1% activation-selected **11.39**，weight-selected **22.37**，random **24.23**。 | Table 1, PDF p. 4 | 这是 diagnostic mixed-precision experiment，不是 final regular AWQ execution path。 |
| W3 OPT gains | OPT 1.3/2.7/6.7/13/30B AWQ PPL **16.32/13.58/11.39/10.56/9.77**；RTN **119.47/298.00/23.54/46.04/18.80**。 | Table 3, PDF p. 4 | PPL 越低越好；不同 model scale 的 RTN failure 不均匀。 |
| LLaMA/Llama-2 quality | W4-g128 Llama-2 13B/70B：FP16 **4.88/3.32**, AWQ **4.97/3.41**, GPTQ **4.98/3.42**。LLaMA-7B：**5.68/5.78/6.22**。 | Table 4, PDF p. 7 | 是 language PPL，不是 task success。 |
| Multimodal evidence | OpenFlamingo-9B COCO CIDEr 32-shot：FP16 **81.70**, W4 AWQ **80.53**, RTN **77.13**, GPTQ **74.98**；W3 AWQ **74.47** vs RTN/GPTQ **64.79/64.77**。 | Table 6, PDF p. 8 | 主要量化 language component，不能代表 full VLA stack。 |
| On-device throughput | VILA-7B tokens/s：A100 **81.6→155.3**, RTX 4090 **58.5→168.1**, Jetson Orin **11.5→35.6**；VILA-13B **48.5→102.1**, `OOM→99.0`, **6.1→17.5**。 | Table 10, PDF pp. 10–11 | 是 AWQ + TinyChat 的 end-to-end system result。 |

## 7. Limitations

### Authors' stated limitations

- Section 3.1 明确指出 naive mixed-precision keep-set 即使只保留 0.1% salient weights，也难转换为 hardware speedup；这是 final regular scaling 要解决的限制。
- Final paper 没有独立 `Limitations` section。

### My critique

- **Internal validity**：average activation magnitude 是 proxy，不一定捕捉 low-frequency/high-consequence channel；需要 action-loss sensitivity ablation 才能验证 VLA saliency。
- **External validity**：VLM evidence 有价值，但 OpenFlamingo 主要压 language side；vision encoder、proprioception、continuous action head 未覆盖。
- **Systems validity**：Table 10 同时包含 TinyChat packing、layout 和 fusion；换 backend 后必须重测。Weight-only 也没有压 activation/KV cache。
- **Reproducibility**：official code 存在，但 group size 128、W3/W4 packing 与 device SIMD/tensor-core capability strongly coupled。
- `lossless` 不是每一 metric 恒等；例如 Table 7 的 VILA-7B VizWiz **59.6→57.8**。

## 8. Why it matters for this project

- **VLA**：已有 OpenFlamingo/VILA evidence，是这组 paper 中最接近 multimodal deployment 的 weight PTQ；适合在 edge robot 上降低 backbone memory bandwidth。
- **需要补的部分**：KV cache、vision encoder、action head 仍可能成为 bottleneck；calibration set 必须包含 contact transitions、occlusions、rare failure states，而非只用 Pile text。
- **Professor Li's direction**：AWQ 将 model-side saliency、regular low-bit representation 与 TinyChat system kernels连起来，直接对应 model compression、latency、edge deployment 和 software-hardware co-design。

## 9. How to read it

### 20-minute route

1. 看 Figure 1/overview，明确 generation 为什么 memory-bound。
2. 读 Table 1：只问“activation-selected 为什么比 weight-selected 好”。
3. 读 Eq. (4)–(5)，画出 $W\times s$ 与 $X/s$。
4. 核对 Table 6 的 OpenFlamingo 32-shot row。
5. 看 Table 10，同时在笔记上写 `algorithm + system`。

### 90-minute route

1. **0–15 min**：补 group-wise quantization、RTN、GPTQ、memory-bound decoding。
2. **15–30 min**：读 saliency observation 与 Table 1，区分 diagnostic keep-set 和 final AWQ。
3. **30–48 min**：推导 scaling 对 relative quantization error 的影响，解释 $\Delta'/\Delta$。
4. **48–60 min**：读 $\alpha$ grid search、clipping 与 calibration generalization。
5. **60–73 min**：核对 Tables 3、4、6，按 text→multimodal 排 evidence strength。
6. **73–82 min**：读 TinyChat implementation 与 Table 10，拆分 algorithm/system attribution。
7. **82–90 min**：写一个 VLA saliency experiment：mean activation vs action-gradient sensitivity。

## 10. Reading questions

1. Activation magnitude 为什么可能比 weight magnitude 更能预测 output sensitivity？
2. $s$ 太大时，为什么同一 quantization group 的其他 weights 可能受损？
3. Eq. (5) 为什么能把高维 $s$ 搜索降成单一 $\alpha$，代价是什么？
4. AWQ 和 SmoothQuant 都用 equivalent scaling，但 target object/precision/objective 有什么差异？
5. Table 10 中有多少收益来自 AWQ representation，有多少来自 TinyChat kernels？
6. VLA 中 rare-but-critical action channel 不一定 activation 大，应该用什么 saliency signal？

### 10.1 My answers

#### 1. 为什么 activation magnitude 比 weight magnitude 更接近 output sensitivity？

Linear layer 的 quantization error 可以写成：

$$
\delta Y=(Q(W)-W)X=\Delta W X.
$$

只看 weight magnitude，只能说明 parameter 本身有多大；它没有告诉我们该 parameter 对应的 input channel 在真实 data 中是否经常被激活。对 input channel $j$，它对 output error 的直接贡献近似为：

$$
\delta y_j=\Delta W_{:,j}x_j,
$$

所以在相同 weight error 下，$|x_j|$ 越大，error 被传到 output 的幅度通常也越大。若暂时忽略 channel correlation，expected squared error 中会出现：

$$
\mathbb E[x_j^2]\|\Delta W_{:,j}\|_2^2.
$$

这解释了为什么 activation statistics 比单纯的 $\|W_{:,j}\|$ 更接近实际 usage-weighted sensitivity。Table 1 也提供了 empirical evidence：OPT-6.7B W3-g128 中，保留 1% activation-selected weights 得到 PPL 11.39，而 weight-selected 与 random 分别是 22.37 和 24.23。

但 average activation magnitude 仍只是 proxy。它忽略 activation correlation、后续 nonlinear layers、error cancellation 和 task loss；“activation 大”不等于“对所有任务都关键”。

#### 2. 为什么 $s$ 太大会伤害同一 group 的其他 weights？

Group-wise quantization 的多个 weights 共用一个 step：

$$
\Delta=\frac{\max_{w\in G}|w|}{2^N-1}.
$$

将 salient channel 的 weight 乘以较大的 $s$ 后，如果它成为 group 中新的 maximum，整个 group 的 step 会从 $\Delta$ 增大到 $\Delta'$. Salient channel 自己还有 inverse activation scaling $x_j/s_j$，所以其 error ratio 近似为：

$$
\frac{\Delta'}{\Delta}\frac{1}{s_j}.
$$

只要 $\Delta'/\Delta$ 增长得比 $s_j$ 慢，它仍然获益。但是其他 non-salient channels 没有同等强度的 $1/s_j$ compensation，却必须使用变粗的 $\Delta'$，其 rounding error 会变大。

Table 2 正好展示这个 trade-off：$s=2$ 时 PPL 最好，为 11.92；继续增加到 $s=4$ 后，21.2% groups 的 $\Delta$ 被改变，PPL 反而退化到 12.36。AWQ 因而不能只最大化 salient-channel protection，还必须控制 collateral group error。

#### 3. Eq. (5) 如何把高维搜索降成单一 $\alpha$？代价是什么？

原始 Eq. (4) 要为每个 input channel 独立搜索一个 $s_j$，维度等于 hidden size，且 quantizer 不可微，直接 optimization 很困难。AWQ 先用 activation magnitude $s_X$ 固定一个“哪些 channels 应更受保护”的方向，再限制：

$$
s_j=(s_{X,j})^\alpha,qquad \alpha\in[0,1].
$$

等价地说，所有可选 scale 都被限制在 log-space 的一条 one-dimensional path 上：

$$
\log s_j=\alpha\log s_{X,j}.
$$

于是每层只需用 20-point grid search 选择一个 $\alpha$：$\alpha=0$ 不做 scaling，$\alpha=1$ 采用最 aggressive 的 activation-proportional scaling。这样无需 backprop，搜索稳定且 calibration cost 很小。

代价是 expressivity：所有 channels 共用同一个 aggressiveness，无法独立处理 unusual weight distribution、channel correlation、rare feature 或 task-specific sensitivity；搜索空间也不允许某些 channels 采用与 activation magnitude 相反的排序。因此 Eq. (5) 是高效 inductive bias，不是 Eq. (4) 的 exact solution。

#### 4. AWQ 与 SmoothQuant 的 equivalent scaling 有什么本质差异？

| Dimension | AWQ | SmoothQuant |
|---|---|---|
| Quantized object | weights only | weights + activations |
| Typical setting | `W4A16` / `W3A16` | `W8A8` |
| Activation role | saliency signal；activation inference precision 仍是 FP16/BF16 | activation 本身也是 quantized object |
| Statistic | per-channel average activation magnitude | per-channel activation maximum + weight maximum |
| Scaling | $s=s_X^\alpha$ | $s_j=a_j^\alpha/w_j^{1-\alpha}$ |
| Objective | 保护 output-sensitive weight channels，最小化 weight-quantized layer output reconstruction error | 把 activation outlier difficulty 迁移到更容易量化的 weights |
| Main deployment path | packed low-bit weights + on-the-fly dequantization，减少 decode weight traffic | regular INT8 GEMM/BMM，同时降低 weight traffic 与 activation compute cost |

两者都保持 FP function 不变，也都可能把一个 tensor 的 difficulty 转移到另一个 tensor；但 AWQ 的 activation 是 **importance estimator**，SmoothQuant 的 activation 是 **需要被量化和 smoothing 的 target**。

#### 5. Table 10 的 speedup 有多少来自 AWQ，又有多少来自 TinyChat？

仅凭 Table 10 无法给出精确百分比，因为它比较的是：

`FP16 model/backend` vs `AWQ W4A16 representation + TinyChat packed kernels/layout/fusion`。

这里至少有三个不同贡献：

1. **AWQ saliency/scaling algorithm**：主要贡献是让 W4/W3 weights 保持 quality；它基本不减少额外 operation count，也不比相同 layout 的 RTN/GPTQ W4 weights 天然跑得更快。
2. **Dense W4 representation**：理论上将 weight traffic 降到约 FP16 的 1/4，为 memory-bound batch-1 decode 提供 speedup upper bound。
3. **TinyChat implementation**：通过 fused on-the-fly dequantization、SIMD-aware packing、MM/MV kernels、LayerNorm/QKV fusion 等，把 theoretical memory saving 变成 measured throughput。

Figure 9 用 `HuggingFace FP16 → TinyChat FP16 → TinyChat AWQ W4A16` 部分拆出了 generic FP16 kernel optimization 与 low-bit path，但仍没有比较 `TinyChat + RTN/GPTQ/AWQ` 的 matched kernels。因此最稳妥的 attribution 是：**AWQ 负责在 regular W4 layout 下守住 quality；W4 representation 创造 bandwidth opportunity；TinyChat 兑现实际 speedup。** Table 10 的 2–3× gain 是 combined system claim，不是 saliency algorithm 单独的 speedup。

#### 6. rare-but-critical VLA channel 应使用什么 saliency signal？

仅用 $\mathbb E|x_j|$ 会偏向常见、大幅 activation，却可能漏掉低频但决定 contact、recovery 或 safety 的 channel。更合理的 first-order signal 是 action-loss-gradient-weighted activation：

$$
S_j=mathbb E_{(\tau,t)\sim\mathcal D}
\left[
\omega(\tau,t)
\left|x_{t,j}\frac{\partial \mathcal L_{\text{action}}}{\partial x_{t,j}}\right|
\right].
$$

其中 $\omega(\tau,t)$ 对 contact transition、near-failure、recovery、high action curvature 或 low safety-margin state 提高权重。它同时回答“channel 是否被激活”和“扰动它是否会改变 action loss”。也可以用 Fisher/gradient-square、action Jacobian norm 或 quantization perturbation 后的 action-chunk deviation 作为更昂贵的替代。

对 VLA，我会采用 two-stage design：

1. 用 `mean activation + tail-aware action-gradient sensitivity` 选 bit-width/scale candidates；
2. 再用 closed-loop success、failure recovery、P99 control latency 验证，而不只看 offline action MSE。

这是面向 VLA 的 research proposal，不是 AWQ paper 已验证的结论。原文只验证了 language/VLM activation-magnitude proxy；它没有 continuous action 或 closed-loop experiment。

## 11. Weekly meeting card

- **Problem**：W3/W4 RTN accuracy 差，而 irregular FP16 salient-weight keep-set 难加速。
- **Key idea**：用 activation statistics 识别 salient channels，再以 equivalent scaling 保护它们且保留 regular packed format。
- **Best evidence**：OpenFlamingo 32-shot CIDEr 81.70→80.53；VILA-7B Jetson Orin 11.5→35.6 tokens/s。
- **Biggest limitation**：weight-only 不压 KV/activation；system speedup 与 TinyChat implementation 绑定。
- **Question for the group**：把 saliency 换成 action-loss sensitivity，是否更能保护 VLA rare events？

## 12. Status & evidence boundary

- **Status**：peer-reviewed MLSys 2024 venue final；reading-list 的短 title 与 venue `On-Device` title 是同一 work cluster。
- **Source claim**：Eqs. (4)–(5)、20-point search、Tables 1/3/4/6/7/10 的数字。
- **My interpretation**：对 edge VLA backbone 的价值、action-aware saliency 和 calibration 建议。
- **Open question**：完整 VLA stack 的 memory/latency/success trade-off，以及 AWQ 与 KV/activation quantization 的组合效果。
