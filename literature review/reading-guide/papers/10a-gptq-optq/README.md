# OPTQ / GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

> **Reading-list role**: Companion — second-order weight-only PTQ；SmoothQuant/AWQ 的重要 predecessor and baseline  
> **Verification**: `verified-full-text` — ICLR 2023 published version  
> **Recommended effort**: Deep read；优先读 Section 3–5

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh |
| Year / version | 2023; arXiv:2210.17323 v2, 2023-03-22 |
| Venue / status | ICLR 2023, peer-reviewed conference paper |
| Published title | **OPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** |
| Common later name | **GPTQ**；arXiv v2、official code 和后续 ecosystem 普遍使用此名 |
| Primary source | [ICLR OpenReview](https://openreview.net/forum?id=tcbBPnfwxS) · [ISTA published-version record](https://research-explorer.ista.ac.at/record/17378) · [arXiv](https://arxiv.org/abs/2210.17323) |
| Code | [official GPTQ repository](https://github.com/IST-DASLab/gptq) |

**Identity boundary.** `OPTQ` 与 `GPTQ` 不是两篇论文。ICLR published version 使用 `OPTQ`；authors 后来把 method/code 改称 `GPTQ`。本地 PDF 是 ISTA repository 标记的 ICLR `Published Version`，所以正文写 `OPTQ`；本 note 用 `OPTQ/GPTQ` 方便和现代 toolchain 对接。

## 2. One-sentence takeaway

OPTQ/GPTQ 把 OBQ 的 approximate second-order error compensation 改造成同一 column order、lazy block update 和 numerically stable Cholesky implementation，使 175B LLM 的 3/4-bit weight-only PTQ 能在单张 A100 上约 4 小时完成，并在 batch-1 memory-bound decoding 中通过 custom kernels 获得 end-to-end speedup。

## 3. Background and prerequisites

- 理解 uniform asymmetric quantization、per-row/group-wise scale、RTN 与 weight-only `W4A16/W3A16`。
- 复习 layer-wise output reconstruction：目标不是让每个 weight 独立最接近，而是让 $\hat W X$ 接近 $WX$。
- 知道 `Hessian`、inverse Hessian、Cholesky decomposition 和 damping 的基本作用。
- 区分 algorithmic quantization 与 inference kernel：GPTQ 决定 quantized weights；custom packed kernel 才能把更少 memory traffic 变成 latency gain。
- Technical lineage：Optimal Brain Surgeon → OBQ / Optimal Brain Compression → OPTQ/GPTQ → SparseGPT / later Hessian-aware LLM compression。

## 4. Problem

- **Target setting**：不 retrain 巨型 LLM，用少量 calibration data 将 weights 压到 3/4 bit。
- **Bottleneck**：RTN 足够快但 3-bit 常崩溃；AdaRound、BRECQ、OBQ 等 accurate PTQ 在 billion-scale model 上计算太慢。
- **Core systems tension**：真正 generative decoding 常为 memory-bound；weight compression 有机会减少 bandwidth，但 mainstream hardware 当时没有直接的 FP16-activation × INT3/INT4-weight compute path，必须边读取边 dequantize。

## 5. Method

### 5.1 Layer-wise reconstruction objective

对一个 Linear layer，calibration inputs 为 $X$，full-precision weights 为 $W$。目标是：

$$
\hat W^*=\arg\min_{\hat W}\|WX-\hat WX\|_2^2.
$$

这里的 “second-order” 不是计算整个 language-model loss 的 full Hessian。该 layer reconstruction objective 本身是 quadratic，per-row Hessian 可写为：

$$
H=2XX^\top.
$$

所以 calibration data 通过 $X$ 决定哪些 weight directions 对 observed layer outputs 更敏感。

### 5.2 OBQ compensation intuition

当 weight $w_q$ 被迫 round 到 `quant(w_q)` 时，不只是接受这一个误差；算法利用 $H^{-1}$ 调整尚未 quantize 的 weights $F$：

$$
\delta_F=-\frac{w_q-\operatorname{quant}(w_q)}{[H_F^{-1}]_{qq}}(H_F^{-1})_{:,q}.
$$

直觉是：当前 weight 被离散化后，沿 calibration input covariance 允许的方向重新分配剩余 weights，让 layer output reconstruction error 尽量小。这是 **local compensation**，不是重新训练整个 network，也不能保证 downstream task loss 最优。

### 5.3 GPTQ 对 OBQ 的三个关键 scale-up changes

1. **Arbitrary fixed order across rows.** 大 layer 中，greedy “每次选最小 error weight” 的收益很小。所有 rows 使用相同 column order 后，共享同一个 remaining-set Hessian，把复杂度从 $O(d_{row}d_{col}^3)$ 降到 $O(\max(d_{row}d_{col}^2,d_{col}^3))$。
2. **Lazy batch updates.** 以 $B=128$ columns 为 block，block 内逐 column quantize/compensate，block 完成后再一次更新所有 remaining columns。总 FLOPs 没有理论性减少，但把许多 low-arithmetic-intensity vector updates 变成 GPU-friendly matrix operations；paper 报告 very large models 上约一个数量级加速。
3. **Cholesky reformulation.** 直接反复更新 $H^{-1}$ 会积累 numerical error，甚至使 matrix indefinite。算法给 Hessian diagonal 加平均 diagonal 的 1% damping，并预先用 Cholesky form 稳定地保存后续需要的 inverse-Hessian rows。

### 5.4 Calibration and execution details

`128 random C4 segments × 2048 tokens → collect layer inputs → build damped H → quantize one Transformer block at a time → rerun quantized block to generate inputs for next block`

- Quantizer：standard uniform per-row asymmetric min-max grid。
- Quantization hardware：single NVIDIA A100 80GB。
- 每次只把一个包含 6 layers 的 Transformer block 放入 GPU memory。
- 后一 block 使用 **前面 blocks 已量化后的 actual inputs**，而不是始终使用 pure FP16 activations；这是对 sequential error propagation 的有限补偿。
- group-wise quantization 可进一步提高 3-bit/2-bit accuracy，但会为每组 scale/zero-point 增加 metadata bits。

### 5.5 What is actually new

真正贡献不是“第一次使用 Hessian”，而是把 OBQ 的 second-order compensation 变成能在 175B LLM 上执行的 algorithm/system recipe：shared order、blocked lazy updates、stable Cholesky 和 layer-streaming implementation。Uniform quantizer、C4 calibration 和 packed GPU kernel 是实现与验证这条路线的 supporting choices。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| 175B quantization 可在 hours 内完成 | OPT-175B **4.2 h**；BLOOM-176B **3.8 h**，single A100 80GB | Table 2, PDF p. 7 | 是 quantization preparation time，不是 inference latency。 |
| 4-bit/3-bit quality 明显优于 RTN | OPT-175B WikiText-2 PPL：FP16 **8.34**；GPTQ-4bit **8.37**；GPTQ-3bit **8.68**；RTN-3bit **7.3e3** | Table 3, PDF p. 7 | 大 model 较容易量化的 trend 不应外推到 every architecture/task。 |
| group-wise 3-bit 接近 FP16 | OPT-175B GPTQ-3/g128：Wiki2 **8.45**、PTB **12.37**、C4 **10.36**；FP16 为 **8.34/12.01/10.13** | Table 5, PDF p. 8 | group size 128 约增加 0.15 bits/weight 的 metadata overhead。 |
| 3-bit model 可显著减少 GPU count | OPT-175B：FP16 需要 5×A100；3-bit + FP16 embedding/output + KV cache 可放入 1×A100 80GB | Sec. 5 Practical Speedups, PDF p. 8 | 这是 2048-token KV cache、paper-era model layout 下的 memory accounting。 |
| batch-1 decoding latency 下降 | A100：**230→71 ms/token, 3.24×**；A6000：**589→130 ms/token, 4.53×** | Table 6, PDF p. 8 | sequence length 128、batch 1、custom kernel；不是 large-batch GEMM speedup。 |

## 7. Limitations

### Authors' stated limitations

- speedup 来自减少 weight memory movement，而不是减少 mathematical multiplication count。
- 不量化 activations；target 是 low-batch generative inference。
- mainstream hardware 缺少直接的 mixed-precision FP16 × INT4 compute support，必须 dynamic dequantization。（Sec. 1/6, PDF pp. 2, 9）

### My critique

- **Objective limitation**：最小化 layer output MSE 不等于最小化 final token loss、reasoning quality 或 VLA success；Hessian 只来自 calibration $X$。
- **Calibration limitation**：128 random C4 segments 很小且是 generic text。后续工作已经表明 GPTQ 对 calibration distribution、ordering、group size 和 model family 有 sensitivity。
- **Systems limitation**：headline speedup 绑定 paper 的 batch-1 custom kernel 与 A100/A6000 memory hierarchy；现代 TensorRT-LLM、Marlin、ExLlama、vLLM 中的 “GPTQ support” 可能使用不同 packing/kernel，不能直接继承 Table 6 数字。
- **Architecture limitation**：experiments 主要是 OPT/BLOOM decoder-only dense LLM；没有 modern GQA/MoE/long-context/KV-cache quantization，更没有 vision/action modules。
- **Naming/ecosystem limitation**：checkpoint 标成 `GPTQ` 往往只说明 weight format/quantization family，未必说明 exact algorithm revision、symmetry、group size、activation order 或 kernel。

## 8. GPTQ 与 SmoothQuant / AWQ 的位置

| Method | Primary object | Typical precision | Calibration signal | Optimization target | Main deployment bottleneck |
|---|---|---|---|---|---|
| GPTQ | weights | W4A16 / W3A16 | layer inputs $X$ → approximate Hessian | layer output reconstruction + error compensation | batch-1 weight bandwidth |
| SmoothQuant | weights + activations | W8A8 | activation/weight channel maxima | migrate activation outlier difficulty to weights | dense INT8 GEMM/BMM |
| AWQ | weights | W4A16 / W3A16 | activation magnitude as saliency | protect salient weight channels with regular scaling | packed weight bandwidth + dequant kernel |

它们可以组合，但不是简单叠加：SmoothQuant/AWQ 先改变 weights 的 scale/distribution 后，GPTQ 的 quantization grid、Hessian-weighted rounding error 和 compensation trajectory 都会改变，必须 joint calibration 并重新评估。

## 9. Why it matters for this project

- GPTQ 是 weight-only LLM PTQ 的 canonical baseline；后续 AWQ、QuIP、AQLM、OmniQuant、QServe 等经常与它比较。
- 对 VLA，最直接用法是压 language/VLM backbone 的 Linear weights；但 action expert/head 的 calibration 必须来自 multimodal trajectories，不能直接复用 C4。
- 它提供一个很好的 FYP ablation axis：`RTN vs GPTQ vs AWQ`，在相同 W4 group size、kernel 和 calibration budget 下比较 `closed-loop success + action error + P99 latency + memory`。
- second-order compensation 也启发 component-wise sensitivity：vision encoder、LLM backbone、action expert 可以使用不同 bit-width/group size，而不是整网一刀切。

## 10. How to read it

### 20-minute route

1. 读 identity boundary，确认 `OPTQ = GPTQ`。
2. 读 Eq. (1)（PDF p. 3）：说清 objective 是 $WX$ reconstruction，不是 weight MSE。
3. 看 Eq. (2) 与 Figure 2（p. 4）：理解 quantize one weight/column 后为何更新 remaining weights。
4. 读 Algorithm 1（p. 5）：圈出 `B=128`、lazy update、Cholesky。
5. 核对 Tables 3、5、6（pp. 7–8）：分别对应 accuracy、metadata trade-off、system speedup。

### 90-minute route

1. **0–15 min**：复习 RTN、asymmetric group-wise quantization 与 layer reconstruction。
2. **15–32 min**：推导 $\|WX-\hat WX\|^2$ 为什么产生 $2XX^\top$。
3. **32–50 min**：读 OBQ Eq. (2)–(3)，用 two-weight toy example 理解 compensation。
4. **50–62 min**：读 three scale-up changes，分别标出 algorithmic complexity、GPU utilization、numerical stability。
5. **62–75 min**：核对 calibration/setup 与 Tables 2–5。
6. **75–84 min**：读 Table 6 + Appendix A.2.2，确认 batch 1、memory-bound 与 kernel boundary。
7. **84–90 min**：设计 VLA calibration ablation：C4 text vs random trajectories vs failure-heavy trajectories。

## 11. Reading questions

1. 为什么 $H=2XX^\top$ 能表示 layer reconstruction sensitivity，却不能代表 full task-loss sensitivity？
2. arbitrary fixed order 为什么在 large layers 上几乎不损失 accuracy，却能大幅降复杂度？
3. lazy block update 为什么不改变当前 columns 的 rounding decisions？
4. group size 变小时，accuracy 与 metadata/kernel efficiency 分别怎样变化？
5. Table 6 的 speedup 为什么主要适用于 batch-1 autoregressive decoding？
6. 若把 GPTQ 用于 VLA action expert，calibration objective 应继续用 hidden-output MSE，还是加入 action/trajectory-aware weighting？

## 12. Weekly meeting card

- **Problem**：accurate second-order PTQ 太慢，RTN 在 3/4 bit 又不可靠。
- **Key idea**：用 calibration Hessian 做 error compensation，并以 shared order + lazy blocks + Cholesky 把它扩展到 175B。
- **Best evidence**：OPT-175B Wiki2 PPL 8.34→8.37（4-bit）；single A100 quantization 4.2 h；batch-1 latency 230→71 ms/token。
- **Biggest limitation**：weight-only layer reconstruction + paper-specific memory-bound kernel，不等于 universal task/system gain。
- **Question for the group**：closed-loop VLA 中，trajectory-aware Hessian weighting 是否优于 generic hidden-state reconstruction？

## 13. Status & evidence boundary

- **Status**：ICLR 2023 peer-reviewed paper；local PDF 为 ISTA `Published Version`。
- **Source claim**：Eqs. (1)–(5)、Algorithm 1、128×2048 C4 calibration、Tables 2/3/5/6。
- **My interpretation**：GPTQ/SmoothQuant/AWQ design-space mapping，以及 VLA calibration/ablation 建议。
- **Open question**：modern VLA architectures、current kernels、long context 和 closed-loop metrics 下能否复现 paper-era trade-offs。
