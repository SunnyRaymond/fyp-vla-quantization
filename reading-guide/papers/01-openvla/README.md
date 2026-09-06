# OpenVLA: An Open-Source Vision-Language-Action Model

> **Reading-list role**: Core — open-source generalist VLA baseline  
> **Verification**: `verified-full-text` + peer-reviewed proceedings  
> **Recommended effort**: Core read

> **Local full text**: [paper.pdf](paper.pdf)

## 1. Paper identity

| Field | Value |
|---|---|
| Authors | Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P. Foster, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn |
| Year / version | arXiv 2024, v3 (2024-09-05); proceedings published 2025 |
| Venue / status | CoRL 2024, PMLR 270:2679–2713; peer reviewed |
| Primary source | [PMLR](https://proceedings.mlr.press/v270/kim25c.html) · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) · [DOI](https://doi.org/10.48550/arXiv.2406.09246) |
| Code / project | [Official project](https://openvla.github.io/) |

**Identity note.** arXiv 与 PMLR 是同一 work，不重复计数；canonical citation 采用 PMLR author list 与 venue record。

## 2. One-sentence takeaway

OpenVLA 用 fused visual encoders、Llama-2 7B 与跨 embodiment 的 970k demonstrations 构成公开 generalist VLA，在 BridgeData V2 达到 70.6% success，并证明 LoRA 只更新 1.4% parameters 也能接近 full fine-tuning。

## 3. Background and prerequisites

- **Technical lineage**：RT-1/RT-2 → Open X-Embodiment → generalist VLA；OpenVLA 可以理解为“可下载、可拆解的 RT-2-style baseline”。
- **读前知识**：Transformer/VLM、`SigLIP`、`DINOv2`、`Llama-2`、behavior cloning、cross-entropy、Open X-Embodiment、LoRA、quantization。
- **关键术语**：`action tokenization`、`cross-embodiment pretraining`、`autoregressive decoding`、`control frequency`。

## 4. Problem

- **Target setting**：从 image + language instruction 直接预测 continuous robot control，并让一个 model 跨 robot、task 与 downstream embodiment 迁移。
- **Bottleneck（更精确的说法）**：RT-2/RT-2-X 已公开论文中的 architecture 与 high-level training description，Open X-Embodiment data 也公开；但可下载的 model weights、可重训的 end-to-end implementation、完全可核对的 mixture/preprocessing recipe，以及面向新 robot 的 reproducible adaptation path 并未公开。研究者因此无法对 architecture、data cleaning、fine-tuning 与 deployment choices 做 matched ablation。
- **Why previous methods are insufficient**：小型 imitation policy 缺少 internet-scale semantic priors；closed VLA 无法复现；从头训练每个 robot 又不能利用 cross-embodiment data。

## 5. Method

### 5.1 System view

`224×224 image + language` → `SigLIP + DINOv2` fused visual features → 2-layer MLP projector → `Llama-2 7B` → autoregressive action tokens → per-dimension bin centers → robot action。

### 5.2 Core mechanism

- 每个 action dimension 用训练集的 1st–99th percentile 分成 256 bins。
- 复用 Llama vocabulary 中最少使用的 256 tokens 表示 action。
- objective 的数学形式是 next-token cross-entropy，但 loss mask 只保留 action-token positions；官方实现默认还监督 sequence-ending stop token。
- 全参数训练，包括 visual encoder；约 970k demonstrations，27 epochs，learning rate `2e-5`，batch size 2048。
- 64×A100 约 14 天、约 21,500 A100-hours；bf16 在 RTX 4090 约 6 Hz。

核心式可以写成：对 action-token sequence `a₁:T`，最小化 `−Σₜ log pθ(aₜ | image, instruction, a₍<t₎)`。注意它学习的是离散 categorical distribution，而不是 continuous density。

#### 5.2.1 “Next-token objective, action-only loss” 到底是什么意思？

训练 sample 可以抽象成：

```text
input:  [BOS, prompt/instruction tokens ..., OUT:, a_x, a_y, a_z, a_roll, a_pitch, a_yaw, a_gripper, EOS]
label:  [-100, -100, ...               , -100, a_x, a_y, a_z, a_roll, a_pitch, a_yaw, a_gripper, EOS]
```

其中 `-100` 是 Hugging Face cross-entropy 的 `ignore_index`。model 仍然读到 image 与完整 instruction；只是不会因为“能否复述 prompt”而被计分。设完整 token sequence 为 `x₁:N`，action/stop position 的 mask 为 `mᵢ∈{0,1}`，实际 objective 是：

`L(θ) = −(1 / Σᵢmᵢ) Σᵢ mᵢ log pθ(xᵢ | image, x₍<i₎)`。

这里有四个关键点：

1. **Conditioning 不等于 supervision**：instruction token 虽然没有 label loss，仍通过 attention 影响 action logits。
2. **Visual encoder 仍会收到 gradient**：action loss 会沿 `action logits → Llama → projector → SigLIP/DINOv2` 反向传播；“不在 image/text positions 算 loss”不等于“冻结 image/text pathway”。
3. **训练使用 teacher forcing**：预测第 `d` 个 action dimension 时会看到 ground-truth `a₍<d₎`；inference 时看到的是自己先前生成的 token，因此存在 within-action autoregressive exposure bias。
4. **为什么要 mask prompt**：prompt 通常比 7 个 action tokens 长。如果也让固定 prompt 参与 loss，容易让 optimization 被复述 template/language 的简单目标主导，而真正需要学习的 robot action signal 被稀释。

官方 code 中先令 `labels = input_ids`，随后把 action 之前的 labels 设为 `-100`；model 内部再做 one-token shift。因此，logit at `OUT:` 预测第一个 action token，logit after `a_x` 预测 `a_y`，依此类推。参见 [official dataset transform](https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/datasets.py) 与 [official autoregressive inference](https://github.com/openvla/openvla/blob/main/prismatic/models/vlas/openvla.py)。

### 5.3 What is actually new

真正的贡献是 open recipe 的系统集成：强 open VLM + 跨 embodiment robot data + action tokenization + weights/code + adaptation/deployment study。`Transformer`、cross-entropy、LoRA、INT4 本身都不是本文发明。

### 5.4 从 2026 的视角看，主要贡献是什么？

1. **Open research substrate**：它第一次把较强的 generalist VLA checkpoint、training code、Open X mixture support、fine-tuning 与 deployment path 放进同一套可下载、可修改的系统。今天它最重要的地位更像“公共实验底座”，而不是永远保持 SOTA 的单一 model。
2. **证明 data/recipe 可以比 parameter count 更关键**：7B OpenVLA 在作者的 evaluation suite 中 match or outperform 55B RT-2-X；但作者也明确把差距归因于 970k vs 350k trajectories、data cleaning 与 fused visual encoder 的组合，而不是一个被隔离验证的新 module。
3. **把 downstream adaptation 变成 first-class research question**：论文不只做 out-of-the-box evaluation，还系统比较 full fine-tuning、partial fine-tuning、LoRA 与 quantized inference。
4. **暴露 VLA 是 systems problem**：bit-width、latency、control cadence 与 closed-loop success 相互耦合。这个 insight 对 model compression 与 software-hardware co-design 比单独的 `INT4 success rate` 更有价值。
5. **一个有用但已显过渡性的 action formulation**：256-bin autoregressive tokens 让 VLM 很容易被改造成 robot policy；但 OpenVLA authors 自己后来的 OpenVLA-OFT 已经用 parallel continuous action head 取代了它。OpenVLA 的持久贡献因此主要是 open pretrained representation 与 ecosystem，而不是“action 必须做成 language token”。

## 6. Experiments and main results

| Claim | Evidence | Locator | Caveat |
|---|---|---|---|
| OpenVLA 在 Bridge 明显超过其他 generalist baselines | BridgeData V2，17 tasks / 170 rollouts：OpenVLA **70.6±3.2%**；RT-2-X **50.6±3.5%**；Octo **20.0±2.6%**；RT-1-X **18.5±2.7%** | Table 4, PDF p. 23 | +20.0 points 是本表口径，不能与 abstract 的 29-task aggregate +16.5 混用 |
| 在 Google Robot 也保持强结果 | 12 tasks / 60 rollouts：OpenVLA **85.0±4.6%**；RT-2-X **78.3±5.4%**；RT-1-X **33.3±6.1%**；Octo **26.7±5.8%** | Table 6, PDF p. 25 | OpenVLA 与 RT-2-X error bars overlap；作者结论是 **comparable**，不是显著胜出 |
| Downstream adaptation 优于 Diffusion Policy | Franka-tabletop **67.2±4.0%** vs **48.5±4.9%**；Franka-DROID **58.3±7.2%** vs **35.0±8.0%** | Table 7, PDF p. 29 | 不是统一公开 leaderboard；task/rubric 由论文定义 |
| LoRA 大幅降 adaptation cost | rank-32 LoRA **68.2±7.5%** vs full FT **69.7±7.2%**；97.6M trainable params（**1.4%**）；59.7 GB vs 163.3 GB | Table 1, PDF p. 8 | “约 8× compute reduction”依赖该 training setup |
| INT4 可把 memory 降到 7.0 GB | bf16 **71.3±4.8%**, 16.8 GB；INT4 **71.9±4.7%**, 7.0 GB；INT8 **58.1±5.1%**, 10.2 GB | Table 2, PDF p. 8; Table 5, PDF p. 23 | blocking-control ablation 表明 latency 是 confound：bf16/INT8/INT4 为 70.0/74.4/68.8（Table 11, p. 33） |

### 6.1 为什么 performance 更高仍不消除 “closed RT-2-X” bottleneck？

先纠正 premise：Figure 3 的 caption 写的是 OpenVLA 在各 category 中领先，**except semantic generalization**；Google Robot 的 aggregate number 虽然是 85.0 vs 78.3，但 error bars overlap，正文称两者 comparable。真正明显的 gap 主要在 BridgeData V2 与 29-task aggregate。

更重要的是，**performance claim 与 openness claim 是正交的**：

- `OpenVLA > RT-2-X on this suite` 回答的是“这次 deployment 谁成功得更多”。
- `RT-2-X is closed` 回答的是“研究者能否下载、重训、替换 preprocessing、做 matched ablation，并把同一 model 适配到新 robot”。

而且 Bridge comparison 本身恰好说明了 closed baseline 的问题。OpenVLA 使用 970k trajectories、更仔细的 cleaning，并过滤 Bridge demonstrations 开头的 all-zero transition；RT-2-X 使用约 350k trajectories，作者只能通过 private API evaluation，不能用 cleaned Bridge data 重训它。由于 RT-2-X 容易输出 all-zero action，他们最终采用“总是取 second-most-likely action”的 workaround（Appendix E, PDF pp. 29–30）。所以这组实验支持的是：

> 在作者可获得的两个完整 systems 下，OpenVLA rollout performance 更强；它不构成 matched data/matched recipe 下 architecture superiority 的证明。

因此原来的 `Bottleneck` 方向没有错，但“data recipe 不开放”写得太绝对；现在已改成“high-level description/data 可见，但 trainable artifact 与 end-to-end reproducible path 不可得”。

## 7. Limitations

### Authors' stated limitations

- 只处理 single image；没有原生 multi-view、proprioception 或 observation history。
- throughput 不足以直接满足 50 Hz control；多数 real-world reliability 仍低于 90%。
- base VLM size、co-training 与 visual feature choices 仍未系统穷尽。（Sec. 5, PDF p. 8）

### My critique

- **Internal validity**：quantization 同时改变 latency/control cadence，“bit-width 导致成功率变化”不是隔离好的因果结论。
- **External validity**：real-robot rollout 数有限，custom tasks 难代表不同 lab/robot。
- **Systems validity**：INT4 的 headline result 与 controller blocking/throughput 强耦合。
- **Reproducibility**：OpenVLA 本身开放，但 RT-2-X 无法以 matched compute/data 重训。

## 8. Official follow-up: OpenVLA-OFT

同一研究线在 peer-reviewed [OpenVLA-OFT (RSS 2025)](https://www.roboticsproceedings.org/rss21/p017.html) 中直接针对 original OpenVLA 的慢速、single-step、discrete autoregressive action head 做了重构：

| Design choice | Original OpenVLA | OpenVLA-OFT / OFT+ | 作用 |
|---|---|---|---|
| Decoding | 7 action dimensions sequentially autoregressive | empty action embeddings + bidirectional attention，所有 action parallel decode | 从 `D` 或 `K×D` decoder passes 降为 1 pass |
| Temporal output | single action | action chunking（LIBERO `K=8`；ALOHA `K=25`） | 提升 throughput，也减少 long-horizon compounding error |
| Action representation | 256-bin discrete tokens | continuous MLP action head | 避免 binning precision loss |
| Objective | action-only next-token cross-entropy | mean L1 regression；diffusion 是受控 ablation | L1 与 diffusion performance 接近，但 training/inference 更简单更快 |
| Inputs | one third-person image | optional wrist image(s) + proprioceptive state | 支持 richer robot setup |
| Language grounding | ordinary VLM attention | OFT+ 在 visual encoder blocks 加 FiLM | ALOHA language-dependent tasks 中减少对 spurious visual correlations 的依赖 |

结果需要分口径读：在相同 third-person image + language input 下，original fine-tuned OpenVLA 为 **76.5%**，加 parallel decoding/action chunking 为 **90.2%**，再换 continuous L1 head 为 **95.3%**；加入 wrist image/proprio 后达到 headline **97.1%**。action throughput 在 LIBERO 上约 **26×**；在 three-camera ALOHA setup 上从 **1.8 Hz** 提升到 **77.9 Hz**，约 **43×**。OFT+ 去掉 FiLM 后，两个 language-following tasks 都降到 **33% chance level**。参见 [paper full text](https://arxiv.org/html/2502.19645v2)、[project page](https://openvla-oft.github.io/) 与 [official OpenVLA updates](https://github.com/openvla/openvla)。

最有意思的 retrospective result 是：即使彻底换掉 action representation、decoding 与 loss，移除 OpenVLA robot pretraining 仍让 LIBERO average success 下降 **5.2 absolute points**。这说明可迁移资产主要存在于 pretrained visual-language-robot representation，不只存在于原来的 token head。

## 9. Why it matters for this project

- 它是后续 VLA architecture、fine-tuning、quantization 的公共参照系；读 π₀、BitVLA 前要先知道它怎样把 action 变成 token。
- 对 model compression 最直接的启示是：LoRA 可把 trainable parameters 降到 1.4%，INT4 可降 memory，但 robot performance 必须和 latency/controller 一起测。
- 对 software-hardware co-design 的启示是：robot policy 的“模型精度”不是独立 metric；control loop frequency 会改变 closed-loop behavior。

## 10. How to read it

### 20-minute route

1. Abstract + Figure 1：记下 inputs、backbone、action tokens。
2. Sec. 3（PDF pp. 3–5）：只追 256-bin action representation 与 action-only loss。
3. Table 4（p. 23）：核对 Bridge headline number。
4. Table 1/2 与 Sec. 5（均在 p. 8）：看 LoRA、quantization 与 authors' caveats。

### 60-90-minute route

1. 先复习 RT-2、Open X-Embodiment、quantile binning。
2. 画出 `image/language → fused encoder → LLM → tokens → action` 数据流。
3. 对照 Table 4、6、7，分清 pretraining evaluation 与 downstream adaptation。
4. 联读 Table 2、5、11，解释为什么 INT8 失败可能是 latency 而非 quantization error。
5. 比较 full FT 与 LoRA 的 parameters、memory、success，写出实际 adaptation recommendation。
6. 写下一点尚未相信的主张：fused vision encoder 的收益是否由充分 ablation 支撑？

## 11. Reading questions — answers

### Q1. 256-bin action tokenization 的主要误差来自 quantization，还是 autoregressive exposure bias？

**结论：原论文没有 factorial experiment 可以把两者严格分开；如果看 OpenVLA-OFT 的后续证据，最大的 practical bottleneck 更像“sequential decoding + 缺少 action chunking + resulting compounding error/latency”，而不是单独的 256-bin quantization。**

- 256 bins 在归一化 action range 上给出约 `2/255 ≈ 0.0078` 的 bin width，precision loss 存在，但通常不会单独解释 20-point 级的 rollout gap。
- original OpenVLA 每个 timestep 只 autoregress 7 个 dimensions，并在下一 control step 重新观察 image，所以 exposure bias 主要发生在 **within-action dimension sequence**，不是无限增长的 text-like sequence。
- OpenVLA-OFT 的受控序列显示：`autoregressive discrete` 76.5 → `parallel discrete + action chunking` 90.2（+13.7）；再换 `continuous L1` 95.3（+5.1）。但前一个 +13.7 同时混合了 parallel decoding 与 action chunking，不能全归因为 exposure bias。

所以更准确的回答是：**quantization 是较小但真实的 error source；original formulation 更大的系统性问题是 sequential generation 造成的 latency，以及无法高效进行 action chunking。**

### Q2. `SigLIP + DINOv2` 的 semantic/spatial 分工有多少直接证据？

**证据偏弱。** Appendix F.2 只做了 `SigLIP + DINOv2` vs `SigLIP-only`，而且是在 Bridge-only model 的 8 个 tasks 上：mean success **45.6±5.6% → 40.6±5.5%**（Table 9, PDF pp. 31–32）。

这能说明“加入 DINOv2 可能有约 5-point benefit”，但不能充分证明清晰的 `SigLIP=semantic`、`DINOv2=spatial` functional decomposition：没有 DINOv2-only baseline，没有专门隔离 semantic vs spatial demand 的 matched tasks，error bars 也高度重叠。论文对分工的解释更像 plausible mechanism，而不是被强 ablation 识别出的 causal fact。

### Q3. cross-embodiment gain 来自 data volume、task diversity 还是 robot diversity？

**无法从本文识别。** `OpenVLA` vs `OpenVLA-Bridge` 是 **76.3±4.8% vs 45.6±5.6%**（Table 9, p. 31），说明 full Open X training mixture 很重要；但这个 ablation 同时改变了 trajectory count、robot count、task/object/scene diversity 与 dataset weighting。它没有 matched-volume、matched-task 或 leave-one-embodiment-out design。

要回答这个问题，需要至少做 `volume × task diversity × embodiment diversity` factorial，或固定总 transitions 后分别增加 tasks/robots，并报告 held-out robot 与 held-out task 的 transfer。当前最稳妥的表述只能是：**mixture diversity as a bundle 有明显收益，各个 diversity axis 的独立贡献仍是 open question。**

### Q4. 为什么 LoRA 能接近 full fine-tuning；哪些 layers 真正需要适配？

最合理的解释是：pretrained model 已有大量 semantic、visual 与 robot prior，downstream adaptation 需要的是较低 intrinsic-rank 的 distributed correction，而不是重学整个 representation。rank-32 LoRA 被加到 **all linear layers**，因此能在整个 visual-language pathway 上做小幅协调更新。

本文真正支持的 layer-level 结论只有：**visual encoder 必须能适配。** Table 1 中 `frozen vision` 只有 **47.0±6.9%**，`sandwich` 为 **62.1±7.9%**，LoRA 为 **68.2±7.5%**，full FT 为 **69.7±7.2%**；Table 10 也显示 fine-tuned vision 明显强于 frozen vision。它没有做 `vision-only LoRA vs LLM-only LoRA vs projector-only LoRA`，所以不能进一步声称 attention、MLP 或 projector 中哪一组是真正必要层。

### Q5. 若把 action head 换成 flow matching，OpenVLA 哪些 component 仍值得保留？

应保留：

- `SigLIP + DINOv2` visual backbone、projector 与 language-conditioned Llama representation；
- Open X-Embodiment robot pretraining 所形成的 transferable representation；
- per-dataset normalization/unnormalization、RLDS/OXE data pipeline 与 LoRA adaptation infrastructure；
- OFT 中的 multi-image/proprio input fusion，以及必要时的 FiLM language grounding。

应替换：256-bin tokenizer、7-token autoregressive action decoding 与 action-only cross-entropy。新的 head 需要 continuous action chunks、noise/time conditioning 与 flow-matching objective。OpenVLA-OFT 的 pretraining ablation 已经给出关键证据：即使 action formulation 已大改，OpenVLA pretraining 仍贡献约 **5.2 absolute points**；所以值得迁移的是 backbone/representation，不是旧 action token interface。

### Q6. 怎样设计 hardware-controlled quantization experiment，排除 control cadence confound？

做一个 `precision × cadence` 2-factor experiment，而不是只比较三条 end-to-end success numbers：

| Axis | Levels | 目的 |
|---|---|---|
| Precision | bf16 / INT8 / INT4，同一 checkpoint | 改变 numerical representation |
| Cadence | matched fixed-rate / native fastest-rate | 分离 action quality 与 deployment speed |

最低限度 protocol：

1. 固定 GPU、CUDA/kernel、batch size、power/clock、controller gains、camera pipeline、task initial-state seeds 与 model checkpoint。
2. **Offline action-fidelity test**：对同一批 frozen observations 比较 token agreement、dequantized action L1/L∞、gripper error；这一步完全不让 robot dynamics 参与。
3. **Matched-cadence closed-loop test**：所有 precision 都以最慢 variant 的 fixed control period 运行；快 variant 人为等待，在相同 observation-to-actuation deadline 提交 action。这样主要测 quantization-induced action change。
4. **Native-cadence deployment test**：各 variant 以实际最快 rate 运行，测真实 system utility。
5. 对每个 task 使用 paired initial states，随机化 method order；报告 success、p50/p95 latency、deadline misses、GPU memory、energy/action 与 trajectory smoothness。
6. 用 two-way model 检查 `precision main effect`、`cadence main effect` 与 `precision×cadence interaction`。若只在 native cadence 下出现差异，主要原因更可能是 timing；matched cadence 仍有差异，才支持 numerical quantization effect。

原论文 Table 11 的 blocking control 已接近第 3 步，但更完整的设计还应加入 fixed-rate deadline、offline fidelity 与 randomized paired rollouts。

## 12. Weekly meeting card

- **Problem**：怎样建立一个公开、强、可适配的 generalist VLA baseline？
- **Key idea**：把 continuous action 量化成 language tokens，用 7B VLM 在 970k cross-embodiment demonstrations 上全参数训练。
- **Best evidence**：Bridge 70.6±3.2% vs RT-2-X 50.6±3.5%（Table 4, p. 23）；rank-32 LoRA 只更新 1.4% parameters 仍达 68.2±7.5%（Table 1, p. 8）。
- **Biggest limitation**：single-image、低 control throughput，且 quantization result 被 latency/controller confound。
- **Question for the group**：我们评估 compressed VLA 时，怎样把 model accuracy、latency 与 closed-loop success 拆开测？

## 13. Evidence boundary

- **Source claim**：original architecture/training 来自 OpenVLA Sec. 3；OpenVLA 数字来自本地全文的 Table/Figure locator，OpenVLA-OFT 数字来自 RSS proceedings/arXiv full text。
- **My interpretation**：OpenVLA 的最大贡献是“可拆解的 RT-2-style baseline”；OpenVLA-OFT 进一步说明 pretrained representation 比原始 discrete action head 更持久，而 quantization 必须做 system-level evaluation。
- **Open question**：fused visual representation、data diversity 与 model scale 各自贡献尚未被完全隔离。
- **Primary links**：[OpenVLA PMLR](https://proceedings.mlr.press/v270/kim25c.html) · [OpenVLA arXiv](https://arxiv.org/abs/2406.09246) · [OpenVLA Project](https://openvla.github.io/) · [OpenVLA-OFT RSS](https://www.roboticsproceedings.org/rss21/p017.html) · [OpenVLA-OFT Project](https://openvla-oft.github.io/)
- **Evidence status**：OpenVLA 与 OpenVLA-OFT 均为 peer-reviewed proceedings；关于“主要贡献”“误差主要来源”与 layer necessity 的文字已显式区分 source claim、controlled evidence 与 interpretation。
