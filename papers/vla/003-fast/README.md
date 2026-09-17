# FAST: Efficient Action Tokenization for Vision-Language-Action Models

> **Reading-list role**: Bridge / Action representation — 从 π₀ 的 continuous flow actions 过渡到 π₀.₅ 的 discrete action pretraining  
> **Verification**: `verified-full-text` + RSS 2025 proceedings  
> **Recommended effort**: Deep read before revisiting π₀.₅ Sec. IV

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Title | *FAST: Efficient Action Tokenization for Vision-Language-Action Models* |
| Authors | Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, Sergey Levine |
| Year / version | 2025; arXiv v1 (2025-01-16), latest checked 2026-08-23 |
| Venue / status | Robotics: Science and Systems (RSS) 2025, paper 012; peer reviewed |
| Primary source | [RSS proceedings](https://www.roboticsproceedings.org/rss21/p012.html) · [arXiv:2501.09747](https://arxiv.org/abs/2501.09747) · [DOI](https://doi.org/10.15607/RSS.2025.XXI.012) |
| Project / code | [Physical Intelligence project](https://www.pi.website/research/fast) · [FAST+ tokenizer](https://huggingface.co/physical-intelligence/fast) |

## 2. One-sentence takeaway

FAST 先用 `Discrete Cosine Transform (DCT)` 把一段连续 action trajectory 变成按 frequency 排列的整数 coefficients，再用 `Byte Pair Encoding (BPE)` 压缩高频出现的 coefficient patterns，使 autoregressive VLA 不必逐 timestep、逐 action dimension 预测大量高度冗余的 tokens。

## 3. Background and prerequisites

- **为什么这篇夹在 π₀ 与 π₀.₅ 之间**：π₀ 用 flow matching 直接生成 continuous action chunk；FAST 说明 continuous trajectory 也能被有效地离散化；π₀.₅ 因而可在 pretraining 阶段把 text、bounding boxes、high-level subtasks 与 robot actions 全部变成 next-token prediction。
- **读前知识**：action chunking、autoregressive next-token prediction、DCT、lossy quantization、BPE、flow matching。
- **关键区分**：DCT 是固定的 analytic transform；coefficient rounding 是 lossy；BPE 对已经量化的 integer stream 做 lossless compression；只有 BPE vocabulary 需要从 data 学习。

## 4. Problem

### 4.1 Naive action tokenization 为什么低效

假设 robot 以 `f` Hz 控制、action dimension 是 `D`，一秒 action chunk 就有 `f × D` 个 scalar values。若每个 scalar 独立分桶为 token：

- 相邻 timestep 的动作高度相关，模型只复制上一个 token 就能得到很低 training loss；
- 高 control frequency 会线性增加 sequence length，却没有线性增加 information；
- action tokens 挤占 context 与 training compute；autoregressive inference 也必须逐 token 解码；
- local smoothness 成为容易的 shortcut，而整段 trajectory shape 才是 policy 真正需要学习的结构。

论文把这个问题描述为：当 sampling frequency 增大时，下一个 naive action token 的 marginal information 接近零。核心问题不是 “continuous actions 不能 tokenized”，而是原来的 tokenization 没有去除 temporal redundancy。

### 4.2 Design target

一个实用 action tokenizer 需要同时满足：

1. reconstruction error 足够小，不破坏 dexterous control；
2. token sequence 足够短，训练与解码才会变快；
3. 能跨 action dimension、control frequency 与 robot embodiment 使用；
4. 可以直接接入现有 VLM 的 categorical vocabulary 与 cross-entropy objective。

## 5. Method

### 5.1 End-to-end pipeline

对 action chunk `A ∈ R^(H×D)`，FAST 的 encoding path 是：

`A → percentile normalization → per-dimension DCT → scale by γ → integer rounding → low-frequency-first flatten → BPE → action tokens`

decoding path 完全反向：

`action tokens → inverse BPE → reshape → divide by γ → inverse DCT → denormalization → reconstructed action chunk Â`

### 5.2 Step-by-step technical details

#### Step 1 — Robust normalization

每个 action dimension 用 training data 的 `q1` 与 `q99` 映射到约 `[-1, 1]`，而不是直接使用 minimum/maximum。这样可降低 outlier 对有效 resolution 的影响，并让不同物理 units 的 dimensions 处在可比较尺度。

#### Step 2 — Per-dimension DCT

沿时间轴对每个 action dimension 分别应用 DCT。平滑 trajectory 的主要能量集中在少量 low-frequency coefficients；high-frequency coefficients 往往很小。DCT 没有在预测未来，它只是把同一条 trajectory 从 time domain 改写到 frequency domain。

#### Step 3 — Coefficient quantization

把 coefficients 乘以 scale `γ` 后 round 到 integer。论文默认 `γ=10`。`γ` 越大，保留的 numerical precision 越高，但 non-zero integers 更多，BPE 后 sequence 可能更长；`γ` 越小，compression 更强，但 reconstruction error 更大。这一步构成 FAST 的主要 lossy trade-off。

#### Step 4 — Low-frequency-first serialization

将 coefficient matrix 按 frequency column 展平：先输出所有 dimensions 的 lowest-frequency coefficients，再输出更高 frequencies。这样 autoregressive model 先决定整段 trajectory 的 coarse/global shape，再补充 fine details。论文报告这种 ordering 比逐 dimension 展平产生更稳定的 rollouts。

#### Step 5 — BPE compression

在 integer sequences 上训练 BPE vocabulary，合并反复出现的 coefficient subsequences。许多 high-frequency coefficients 被 quantize 为零，BPE 可把长 zero runs 与常见跨 dimension patterns 压成少量 tokens。默认 vocabulary size 是 1024。

BPE 本身不再丢失信息；它只 losslessly 编码 Step 3 已经量化的 integers。去掉 BPE 后仍比 naive tokenization 好，但 sequence 更长、zero tokens 稀释 learning signal，policy performance 也下降。

### 5.3 FAST 与 FAST+ 的区别

| Variant | BPE vocabulary 从哪里来 | 适用场景 | Trade-off |
|---|---|---|---|
| FAST | 当前 dataset 的 action chunks | 有足够 target data、允许为 dataset 单独训练 tokenizer | compression 最贴合当前 data，但每个 dataset 都要拟合 vocabulary |
| FAST+ | 约 1 million 个 one-second real-robot action chunks，涵盖多种 embodiments、action spaces 与 frequencies；统一 pad 到 32 dimensions | 想直接复用一个 universal tokenizer | 接近 dataset-specific FAST，但 “universal” 主要由 compression 与 downstream evidence 支持，不等于已经证明对所有 robots 都有 policy generalization |

### 5.4 What is actually new

DCT、integer quantization 与 BPE 都不是新算法。FAST 的贡献是识别 VLA action tokenization 的真正 bottleneck 是 high-frequency temporal redundancy，并把三者组合成一个适合 autoregressive VLA 的 trajectory tokenizer；论文进一步用跨 robot data 训练 FAST+，验证 tokenizer 可以在多种 action spaces 与 frequencies 之间复用。

## 6. Experiments and main results

### 6.1 Compression scales with control frequency

| Dataset / setting | Naive tokens per 1 s | FAST tokens per 1 s | Compression |
|---|---:|---:|---:|
| BridgeV2, 7D, 5 Hz | 35 | 20 | 1.75× |
| DROID, 7D, 15 Hz | 105 | 29 | 3.6× |
| Bussing, 7D, 20 Hz | 140 | 28 | 5.0× |
| Shirt Fold, 14D, 50 Hz | 700 | 53 | 13.2× |

这些是 Table I（PDF p. 7）的 exact values。关键现象不是固定 compression ratio，而是 FAST token count 约保持在每个 arm、每秒几十个 tokens；control frequency 增长时，naive sequence length 继续线性增长，FAST 则主要为新增的真实 trajectory detail 付费。

### 6.2 Policy learning evidence

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| Compression-aware tokenizers 优于 naive bins | FAST 与 FSQ 在 simulation/real-robot tasks 上总体提高 training efficiency；dexterous real-robot settings 中 FAST 更强 | Fig. 6 | 多个 tokenizer 的 reconstruction、vocabulary 与 sequence length 同时变化，不能把全部收益只归因于 DCT |
| FAST+ 接近 dataset-specific FAST | universal vocabulary 在多组 tasks 上接近单 dataset vocabulary | Fig. 6 and Sec. IV | evaluation robots/tasks 仍有限；不等于 arbitrary embodiment zero-shot control |
| Large-data regime 中训练更快 | large table-bussing task 达到高 performance 所需 training steps 约少 3× | Fig. 9, PDF p. 9 | 是特定 task/backbone 的 empirical result，不是通用 complexity guarantee |
| Scale-up 可匹配 flow policy | 在与 π₀ 相同的约 10k-hour / 903M-timestep mixture 上，π₀-FAST average performance 与 flow-based π₀ 相近，GPU hours 约少 5× | Sec. IV-E / Fig. 1 | 两种 output heads 的 inference path 不同；training-efficiency parity 不代表 deployment parity |
| Language following 可能更好 | DROID experiments 中 FAST policy 的 language following 优于 flow policy | Fig. 9 | authors 明确把 causal explanation 留作 future work，不能直接断言是 discrete tokens 必然保留 language ability |

论文覆盖 6 个 real-robot evaluations 与 1 个 simulation evaluation，并在 π₀/PaliGemma 与 OpenVLA-style backbones 上测试。这里最值得相信的是 “冗余压缩可显著减少 action-token training burden”；最需要保留边界的是把个别 task 的 gains 推广成所有 robots 的 universal result。

## 7. FAST 与 flow matching 到底是什么关系

它们解决同一个 output problem，但优化目标和 deployment profile 不同：

| Dimension | FAST autoregressive policy | π₀ flow action expert |
|---|---|---|
| Action representation | quantized DCT coefficients + BPE tokens | continuous noisy action chunk |
| Training objective | categorical next-token cross-entropy | conditional flow matching regression |
| 与 text/web outputs 的统一程度 | 高：都进入同一 token vocabulary/objective | 低：non-action data 没有 flow target，需要另一条 objective/path |
| Training efficiency | sequence 压缩后较高；paper reports up to 5× fewer GPU hours at scale | 需要对 sampled noise/time 训练 vector field |
| Inference | sequentially decode about 30–60 tokens | action expert parallel-refines whole chunk for about 10 steps |
| Reported chunk latency | π₀-FAST 约 750 ms | π₀ flow <100 ms on RTX 4090 |

因此 FAST 不是证明 flow matching “过时”。它揭示的是一个反直觉 trade-off：**FAST 更适合 scalable heterogeneous pretraining；flow action expert 更适合 low-latency continuous control**。π₀.₅ 的 two-stage recipe 正是把两者分别放到各自更擅长的阶段。

## 8. Limitations

### Authors' stated / directly evidenced limitations

- FAST 通过 coefficient rounding 做 lossy compression；`γ` 必须在 reconstruction fidelity 与 token length 之间取舍。
- autoregressive inference 仍是 sequential。论文报告 π₀-FAST 生成一秒 chunk 约 750 ms，而 π₀ 的 smaller flow expert 约用 10 steps、低于 100 ms（RTX 4090；PDF p. 9）。
- π₀-FAST 每个 token 都经过约 2B backbone；flow inference 主要经过约 300M action expert，所以 shorter token sequence 不自动等于 lower latency。

### My critique

- FAST+ 的 pretraining mixture 与部分 evaluation task families 有覆盖关系；held-out compression evidence 可以说明 codec transfer，但不能单独证明 downstream policy 的 fully unseen embodiment generalization。
- 主要 closed-loop policy evidence 来自 static manipulators；mobile manipulation 与 humanoid settings 更多是 offline compression evidence。
- 与 flow matching 的比较同时改变 action head、loss、inference algorithm 与 compute allocation，不能视为只替换 tokenizer 的 perfectly controlled ablation。
- action reconstruction error 是必要但不充分 metric；closed-loop error 会经 environment feedback 累积，最终仍应看 task success、latency 与 recovery behavior。

## 9. Why it matters for this project

- **Action representation 决定 compute shape**：参数量不变，output sequence length 与 decoding path 也会显著改变 training cost 和 P99 latency。
- **对 π₀.₅ 的解释关键**：FAST 让 heterogeneous pretraining 可以用单一 next-token objective；它不是 π₀.₅ deployment 时最终 continuous controller 的替代品。
- **对 quantization project 的提醒**：FAST 的 “quantization” 是 action-space lossy coding，不是 weight/activation low-bit quantization。两者都影响 error，但传播路径完全不同。
- **潜在研究问题**：能否把 FAST 的 global-to-local representation 与 parallel decoder 结合，保留 training efficiency，同时避免 autoregressive latency？

## 10. How to read it

### 20-minute route

1. Figure 1：先理解 training speed 与 inference speed 是相反的 trade-off。
2. Figure 3/4 + Algorithm 1（PDF pp. 4–5）：手工复述 normalization → DCT → rounding → flatten → BPE。
3. Table I（p. 7）：观察 frequency 越高 compression ratio 越大。
4. Figure 9 + inference-speed paragraph（p. 9）：比较 FAST 与 flow matching。
5. 回到 π₀.₅ Figure 3：解释为什么 Stage 1 用 FAST、Stage 2 才加 flow action expert。

### 60–90-minute route

1. 用一条 1D smooth trajectory 手算/画出 DCT coefficients，确认 low frequencies 表示 coarse shape。
2. 分别标出三个概念：`lossy rounding`、`lossless BPE`、`autoregressive CE`。
3. 核对 Figure 6：区分 reconstruction/compression quality 与 closed-loop policy quality。
4. 核对 Figure 9：把 small-data parity、large-data convergence 与 language following 分成三个 claims。
5. 比较 π₀-FAST 与 π₀ flow 的 training compute、head size、inference steps 与 latency，不只比较 task score。
6. 最后读 FAST+：问 universal tokenizer 迁移的是 codec statistics，还是 control knowledge？答案主要是前者。

## 11. Reading questions

1. 若 DCT 后不做 BPE，只对 coefficients 单独建 token，training 与 inference 分别会损失什么？
2. low-frequency-first ordering 为什么可能减少 closed-loop jitter？它是否也可能延迟关键 high-frequency correction？
3. `γ` 应按 reconstruction MSE、task success，还是 action dimension 的 physical sensitivity 选择？
4. FAST+ 共享 vocabulary 时，如何避免不同 embodiments 中相同 integer pattern 表示不同 physical semantics？
5. 为什么 FAST 在 large-data regime 的 training advantage 更明显，而 small-data tasks 与 flow matching 接近？
6. 能否用 non-autoregressive/parallel token decoder 缩小 750 ms 与 <100 ms 的 inference gap？
7. 对 VLA compression，action-token compression 与 model-weight quantization 的 errors 会相加、相乘，还是被 closed-loop feedback 部分修正？

## 12. Weekly meeting card

- **Problem**：high-frequency robot actions 被 naive tokenization 变成大量高度相关 tokens，拖慢 autoregressive VLA training。
- **Key idea**：DCT 去 temporal redundancy，rounding 控制 precision，low-frequency-first serialization 保留 global-to-local structure，BPE 压缩 repeated integer patterns。
- **Best evidence**：50 Hz、14D Shirt Fold 从 700 naive tokens/s 降到 53 FAST tokens/s（13.2×；Table I, p. 7）；scale experiment 报告约 5× fewer GPU hours。
- **Biggest limitation**：training 更快但 autoregressive action inference 更慢；π₀-FAST 约 750 ms/chunk，π₀ flow <100 ms/chunk。
- **Question for the group**：能否设计一个 training 时 token-based、deployment 时 parallel continuous decoding 的 distillation route？

## 13. Evidence boundary

- **Source claims**：algorithm、default `γ=10`、vocabulary size 1024、Table I、policy experiments 与 latency 均来自 FAST full text。
- **My interpretation**：FAST 的本质是 trajectory coding / temporal redundancy reduction，而不是普通 scalar discretization。
- **Open question**：FAST 的 language-following gain、FAST+ 的 fully unseen embodiment generalization、以及 FAST 与 low-bit model compression 的 interaction 尚未被充分隔离。
- **Primary links**：[RSS proceedings](https://www.roboticsproceedings.org/rss21/p012.html) · [arXiv](https://arxiv.org/abs/2501.09747) · [Physical Intelligence project](https://www.pi.website/research/fast)

