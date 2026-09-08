# 李老师 Reading List：论文阅读与组会汇报指南

> **Evidence snapshot:** 2026-09-05  
> **Core inventory:** 24 篇 paper / technical report；另有 28 篇 companion papers、2 份 normative/vendor specifications 与 1 份 researcher watchlist。  
> **Local full text:** 24/24 core papers + 28/28 companions 已保存；`Outlier Suppression` 的 official supplemental 也已本地保存；详见 [PDF inventory](PDF_INVENTORY.md)。  
> **目标:** 不是机械“读完”，而是每周能讲清 `Problem → Mechanism → Evidence → Limitation → Research Question`。

## 先从哪里开始

### 2026-09-08：WM / WAM 量化专题

[独立阅读库](../wm-wam-quantization-reading-guide/README.md)：24 份本地 PDF 与中文导读，覆盖 LeWorldModel / DINO-WM / V-JEPA、直接 quantization prior art、开源 WAM 和 benchmark 地图；不加入本主库 core count。

如果下次组会就在一周内，先完成：

### 师兄 2026-09-05 report：新增 #23 / #24

1. [WaterSIC #23](papers/23-watersic/README.md)：先理解 column-wise rate allocation、SIC / GPTQ、entropy coding 与 rate–distortion theorem 的条件。
2. [GRACE #24](papers/24-grace/README.md)：再理解 confidence-gated KD、RCKA、adaptive IB controller 与 weight-only QAT。
3. [VLA_idea 三方向独立阅读库](../vla-ideas-reading-guide/README.md)：30 个 numbered entries、31 份 PDF；按你的要求单独组织，不加入本 guide core count。

### 李老师 2026-09-04 新增的三篇

1. [HBVLA](papers/20-hbvla/README.md) 的 **20-minute route**：先抓 action-aware Hessian、Haar-domain W1 PTQ 与 Mobile ALOHA absolute drop。
2. [WorldCache — Content-Aware Caching](papers/21-worldcache-content-aware/README.md) 的 **20-minute route**：把 `CFC / SWD / OFA / ATS` 各自解决的 failure mode 分开。
3. [WorldCache — Heterogeneous Token Caching](papers/22-worldcache-heterogeneous-token-caching/README.md) 的 **20-minute route**：解释 stable/linear/chaotic token predictor 与 CAS refresh budget。
4. 最后读 [three-paper targeted scan](research/THREE_PAPER_SCAN_2026-09-04.md)，只填一张 `approximation unit / risk signal / compute saved / evidence boundary` 对照表。

### 原有主线

1. [OpenVLA](papers/01-openvla/README.md) 的 **20-minute route**，建立 open VLA baseline。
2. [Flow Matching](papers/17-flow-matching/README.md) 的 **20-minute route**，先理解 `conditional probability path → velocity target → ODE sampling`。
3. [REPA / iREPA → VLA representation alignment](papers/19-repa-irepa-vla-alignment/README.md) 的 **20-minute route**，分清 generic `REPA`-style feature loss、`iREPA` spatial mechanisms 与已有 `VLA/WAM` prior art。
4. [Self-Flow](papers/18-self-flow/README.md) 的 **20-minute route**，分开 `Flow Matching loss`、`Dual-Timestep Scheduling`、internal `EMA teacher` 与后续 data-augmentation mechanism challenge。
5. [π₀](papers/02-pi0/README.md) 的 **20-minute route**，再理解 `flow matching action expert` 怎样把该 objective 特化到 continuous action chunks。
6. [FAST](papers/02a-fast/README.md) 的 **20-minute route**，理解 compressed action tokens，以及 π₀.₅ 为何把 discrete pretraining 与 flow post-training 分开。
7. [FlashVLA](papers/03a-flashvla/README.md) 的 **20-minute route**，理解 `isolated flow chunks → staggered streaming buffer`，并分清 policy latency、control frequency 与 closed-loop time per step。
8. [FlashSAC](papers/18a-flashsac/README.md) 的 **20-minute route**，分开 sample efficiency、wall-clock efficiency、UTD ratio、policy frequency 与 low-level control frequency。
9. [LoRA](papers/15-lora/README.md) 的 **20-minute route**，把 trainable parameters、task-specific storage 与 base-model inference cost 分开。
10. [Outlier Suppression](papers/16-outlier-suppression/README.md) 的 **20-minute route**，再回到 `SmoothQuant`，分清 `LayerNorm γ migration` 与 general per-channel smoothing。
11. 用一页纸比较 `naive discrete action tokens`、`FAST compressed tokens`、`continuous flow chunks` 与 `streaming flow chunks`。
12. 按 [8-week plan](WEEKLY_PLAN.md) 的 5-slide template 准备汇报；读不完时明确标记 `unread / skimmed / deep-read`。

完整路线按 [8-week reading and weekly-meeting plan](WEEKLY_PLAN.md) 执行。每周最低投入约 3 小时，standard route 约 6 小时。

## 这份 Reading List 实际包含什么

### A. Vision-Language-Action Models

| # | Reading | 你要抓住的主线 | Status | 建议 |
|---:|---|---|---|---|
| 1 | [OpenVLA](papers/01-openvla/README.md) | open generalist VLA、`action tokenization`、adaptation/deployment baseline | CoRL 2024 | **Core read** |
| 2 | [π₀](papers/02-pi0/README.md) | pretrained VLM + `flow matching action expert` + continuous action chunks | RSS 2025 | **Core read** |
| 2A | [FAST](papers/02a-fast/README.md) | DCT + quantization + BPE，压缩 high-frequency action trajectories | RSS 2025 | **Bridge / Deep read** |
| 3 | [π₀.₅](papers/03-pi05/README.md) | heterogeneous co-training、hierarchical inference、open-world generalization | CoRL 2025 oral | **Core read** |
| 4 | [π*₀.₆](papers/04-pistar06/README.md) | `RECAP`、value/advantage conditioning、offline RL 与 human correction | technical report | **Core read** |
| 5 | [ω-0](papers/05-omega0/README.md) | latent future prediction 作为 training-time auxiliary、humanoid VLA | arXiv preprint | **Critical scan** |
| 6 | [BitVLA](papers/06-bitvla/README.md) | native 1-bit VLA、ternary weights、memory/latency claims | arXiv; “Work in progress” | **Critical scan** |
| 20 | [HBVLA](papers/20-hbvla/README.md) | `HB-VLA` action-aware rectified Hessian、Haar-domain 1-bit PTQ、simulation 与 Mobile ALOHA retention | arXiv v2 preprint | **Core / Critical read** |

#### VLA systems companion route（不增加 core count）

| Reading | 为什么现在读 | Status | 建议 |
|---|---|---|---|
| [FlashVLA](papers/03a-flashvla/README.md) | 把 `π₀.₅` 的 10-step isolated flow decoding 改为 staggered streaming buffer，并联合考察 asynchronous continuity、reaction latency 与 closed-loop success | arXiv:2608.27384 v1 preprint; official code/checkpoints released | **Deep / Critical read** |

#### Robot RL systems companion route（不增加 core count）

| Reading | 为什么现在读 | Status | 建议 |
|---|---|---|---|
| [FlashSAC](papers/18a-flashsac/README.md) | 理解 off-policy replay、large model/batch/buffer、very low UTD、critic stabilization 与 sim-to-real control stack；同时训练如何审计 wall-clock、reward、hardware 和 controller confounds | RSS 2026 `Outstanding Paper Award`; local arXiv v2; official code released | **Deep / Critical read** |

#### PI-series companion route（不增加 core count）

| Reading | 为什么现在读 | Status | 建议 |
|---|---|---|---|
| [MEM](papers/04a-mem/README.md) | 理解 `π₀.₆` 如何用 short-term video/proprioceptive history 与 long-term text summary 处理 partial observability、failure history 和 long-horizon tasks | Physical Intelligence technical report / arXiv v2 | **Deep read** |

#### BitVLA companion route（不增加 core count）

| Reading | 为什么现在读 | Status | 建议 |
|---|---|---|---|
| [Apprentice](papers/06a-apprentice-quantization-distillation/README.md) | 核对 Quantize-then-Distill 的 prior art；Scheme-C 是 direct conceptual predecessor | ICLR 2018 | **Targeted read** |
| [LLaVA](papers/06b-llava/README.md) | 理解 BitVLA multimodal alignment → instruction-tuning curriculum | NeurIPS 2023 Oral | **Core companion** |
| [OpenVLA-OFT](papers/06c-openvla-oft/README.md) | 理解 parallel action queries、continuous L1 chunks、FiLM 与 three-camera ALOHA setup | RSS 2025 | **Deep read** |
| [LLM.int8 / bitsandbytes](papers/06d-llm-int8-bitsandbytes/README.md) | 区分 bitsandbytes PTQ baseline 与 BitVLA native BitBLAS W1.58A8 path | NeurIPS 2022 | **Targeted read** |

### B. Numeric formats and quantization

| # | Reading | 方法层级 | 你要抓住的主线 | Status | 建议 |
|---:|---|---|---|---|---|
| 7 | [FP8 Formats for Deep Learning](papers/07-fp8-formats/README.md) | datatype | `E4M3` / `E5M2` 的 precision–range trade-off | technical preprint + OCP companion | **Core read** |
| 8 | [Microscaling Data Formats for Deep Learning](papers/08-microscaling-formats/README.md) | datatype/scaling | `MXFP8` / `MXFP4` 的 block-level shared scale | technical preprint + OCP MX spec | **Core read** |
| 9 | [Pretraining Large Language Models with NVFP4](papers/09-nvfp4-pretraining/README.md) | datatype/recipe | two-level scaling、native low-precision pretraining、hardware dependence | vendor-authored technical report | **Targeted read** |
| 10 | [SmoothQuant](papers/10-smoothquant/README.md) | equivalent transform | 把 activation outliers 的 quantization difficulty 平滑迁移到 weights | ICML 2023 | **Core read** |
| 11 | [AWQ](papers/11-awq/README.md) | weight-only PTQ | activation-aware salient weights 与 per-channel scaling | MLSys 2024 | **Core read** |
| 12 | [SpinQuant](papers/12-spinquant/README.md) | learned transform | learned rotations 改善 outlier geometry，同时保持 full-precision function | ICLR 2025 | **Core read** |
| 13 | [HAQ](papers/13-haq/README.md) | mixed-precision search | 用 real hardware feedback 分配 layer-wise bit-width | CVPR 2019 | **Core read** |
| 14 | [TurboQuant](papers/14-turboquant/README.md) | vector quantization | random rotation、scalar codebook、unbiased inner-product estimator 与 distortion bound | ICLR 2026 | **Targeted read** |
| 16 | [Outlier Suppression](papers/16-outlier-suppression/README.md) | outlier diagnosis / equivalent transform | `LayerNorm γ` amplifier、Gamma Migration、Token-Wise Clipping 与 6-bit `PTQ` boundary | NeurIPS 2022 | **Core read** |
| 23 | [WaterSIC](papers/23-watersic/README.md) | rate–distortion / weight-only PTQ | unequal column rates、SIC error compensation、entropy coding 与 asymptotic optimality boundary | ICML 2026; local arXiv v2 | **Core / Critical read** |
| 24 | [GRACE](papers/24-grace/README.md) | VLM QAT + distillation | GDKD、RCKA、adaptive IB controller 与 group-wise LSQ | ICML 2026; local arXiv v5 | **Core / Critical read** |

#### Quantization companion route（不增加 core count）

| Reading | 为什么现在读 | Status | 建议 |
|---|---|---|---|
| [OPTQ (commonly known as GPTQ)](papers/10a-gptq-optq/README.md) | 补齐 `RTN → second-order error compensation → W3/W4 weight-only PTQ` 主线 | ICLR 2023 | **Deep read** |
| [QuaRot](papers/12a-quarot/README.md) | 先理解 fixed/randomized Hadamard rotation、function preservation 与 end-to-end `W4A4KV4`，再读 SpinQuant 的 learned rotation | NeurIPS 2024 | **Bridge / Deep read** |
| [A Survey of Quantization in LLM](papers/10b-llm-quantization-survey-2026/README.md) | 2026 dedicated snapshot：training、PTQ/QAT、KV cache 与 kernel generation | JCST 2026 | **Map read** |
| [A Survey of Low-bit Large Language Models](papers/10c-low-bit-llm-survey/README.md) | 更全面的 basics、systems、algorithms 与 toolkit reference | Neural Networks 2025; local arXiv v3 | **Reference read** |
| [RaBitQ](papers/14b-rabitq/README.md) | 先读 original 1-bit estimator、high-probability error bound 与 bitwise/SIMD path | SIGMOD 2024; local arXiv v1 | **Foundational companion** |
| [Practical and Asymptotically Optimal Quantization...](papers/14b-rabitq/README.md#4-what-the-multi-bit-extension-changes) | 补齐 multi-bit integer-grid extension、flexible bit rate 与 asymptotic space–error result | SIGMOD 2025; local arXiv v1 | **Deep companion** |
| [Revisiting RaBitQ and TurboQuant](papers/14a-rabitq-turboquant-comparison/README.md) | 审计 TurboQuant 对 RaBitQ 的 method/theory 描述、baseline symmetry 与 reproducibility | VecDB@VLDB 2026 workshop paper; local arXiv v2 | **Critical companion** |
| [A Note on TurboQuant and the Earlier DRIVE/EDEN Line of Work](papers/14a-rabitq-turboquant-comparison/README.md#7-the-separate-drive--eden-lineage-claim) | 区分 `TurboQuant_mse = EDEN with S=1` 的 lineage claim 与 RaBitQ baseline dispute | arXiv preprint | **Prior-art companion** |

### C. Parameter-Efficient Fine-Tuning

| # | Reading | 方法层级 | 你要抓住的主线 | Status | 建议 |
|---:|---|---|---|---|---|
| 15 | [LoRA](papers/15-lora/README.md) | Parameter-Efficient Fine-Tuning | `W₀ + ΔW = W₀ + BA`、mergeable low-rank update、training-state / task-storage / serving-cost boundary | ICLR 2022 | **Core read** |

LoRA 单开整数编号，因为它不是 OpenVLA 的一个 implementation detail，而是理解 VLA downstream adaptation、QLoRA-like training 与 component-wise tuning 的 foundational method。它减少 trainable update 和 per-task checkpoint，不自动压缩 frozen base model 的 inference memory、latency 或 power。

`Outlier Suppression` 也单开整数 `#16`：它位于 structured outlier diagnosis 与后续 `SmoothQuant`/rotation methods 之间，是理解“outlier 从哪里来、哪些能 clip、何时应迁移或旋转”的 foundational quantization paper。编号是按本 guide 的新增顺序，不代表 chronological reading order；实际应在 `SmoothQuant` 前读。

### D. Generative and representation-learning foundations

| # | Reading | 方法层级 | 你要抓住的主线 | Status | 建议 |
|---:|---|---|---|---|---|
| 17 | [Flow Matching for Generative Modeling](papers/17-flow-matching/README.md) | generative objective / probability path | marginal `Flow Matching`、tractable `Conditional Flow Matching`、Gaussian paths 与 conditional `Optimal Transport` interpolation | ICLR 2023, Top 25% | **Foundational core read** |
| 18 | [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](papers/18-self-flow/README.md) | internal representation learning / multi-modal flow | `Dual-Timestep Scheduling`、cleaner-view `EMA teacher`、joint velocity/feature objective，以及 image/video/audio 到 `SIMPLER` joint video-action transfer | ICML 2026 poster; local arXiv v1 | **Core / Critical read** |

`Flow Matching` 单开整数 `#17`，因为它不是 `π₀` architecture 的附属说明，而是后续 flow-based VLA action heads 的基础 objective。编号按加入 guide 的顺序；实际阅读位置应在 `π₀` 之前。

`Self-Flow` 单开整数 `#18`，因为它把 `Flow Matching`、internal self-supervised representation learning、multi-modal scaling 与 joint video-action transfer 接到同一条主线。它是 user-promoted core，不是 retroactive advisor assignment。

#### Representation-alignment companion route（不增加 core count）

| Reading | 为什么现在读 | Status | 建议 |
|---|---|---|---|
| [REPA / iREPA → VLA representation alignment](papers/19-repa-irepa-vla-alignment/README.md) | 先理解 diffusion/flow transformer 的 frozen-teacher feature alignment，再核对 `FLARE`、`Spatial Forcing`、`VEGA`、`AGRA` 等已覆盖边界；重点判断 exact `iREPA` spatial recipe 是否仍有研究空间 | REPA: ICLR 2025 Oral; iREPA: ICLR 2026; 10 direct/adjacent robot papers versioned locally | **Deep / Critical read** |

#### Self-Flow mechanism companion route（不增加 core count）

| Reading | 为什么现在读 | Status | 建议 |
|---|---|---|---|
| [From SRA to Self-Flow: Data Augmentation or Self-Supervision?](papers/18-self-flow/README.md#83-direct-post-publication-mechanism-challenge) | 用 `Attention Separation` 阻断 cross-noise token interaction，检验 `Dual-Timestep Scheduling` 的收益主要来自 self-supervision 还是 noise-state data augmentation | arXiv:2607.02508 v1; local 10-page preprint | **Critical companion** |

### E. World-model inference acceleration

| # | Reading | 方法层级 | 你要抓住的主线 | Status | 建议 |
|---:|---|---|---|---|---|
| 21 | [WorldCache: Content-Aware Caching for Accelerated Video World Models](papers/21-worldcache-content-aware/README.md) | deep-block / feature caching | content change、spatial saliency、optimal feature approximation 与 late-step relaxation | ECCV 2026 accepted; local arXiv v1 | **Core / Critical read** |
| 22 | [WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](papers/22-worldcache-heterogeneous-token-caching/README.md) | model-output token caching | curvature-based stable/linear/chaotic predictors 与 `Chaotic-prioritized Adaptive Skipping` | ICML 2026; local arXiv v2 | **Core / Critical read** |

两篇 `WorldCache` 单开整数编号，因为它们是两项不同工作。`#21` 在 H200 上围绕 probe/deep-block cache 优化“何时复用、如何修正”；`#22` 在 A800 上围绕 output token dynamics 优化“哪些 token 如何预测、何时整体刷新”。两者 headline speedup 不可直接横比。

## 九个容易读错的地方

1. **`π₀.₆` 的映射。** Reading list 中的 `pi_0.6` 最合理地对应 [π*₀.₆](papers/04-pistar06/README.md)。这篇同时定义 base `π₀.₆` 与经 `RECAP` 改进的 `π*₀.₆`；[MEM](papers/04a-mem/README.md) 是同一 base 的独立 memory extension，现作为 companion reading 单列，不能与 `π*₀.₆` 合并。
2. **Format name 不等于独立 paper。** `MXFP8` / `MXFP4` 由 OCP MX specification 定义，paper-level reading 归到 *Microscaling Data Formats for Deep Learning*。`NVFP4` 则是 NVIDIA-defined format/recipe，不是开放 industry standard。
3. **`NVFP8` 暂不建 paper entry。** 未核验到一个独立、权威、同时定义 bit layout、scale hierarchy 与 arithmetic semantics 的 `NVFP8` specification/paper；见 [NVFP8 clarification](NVFP8_CLARIFICATION.md)。不能因为 checkpoint/product label 出现这个词就虚构一种新 datatype。
4. **`FlashVLA` 有 name collision。** 本地 companion 是 [arXiv:2608.27384](papers/03a-flashvla/README.md)，方法是 streaming action decoding；arXiv:2505.21200 的 *Think Twice, Act Once* 也把其 action-reuse/token-pruning 方法称为 `FlashVLA`。引用时必须写 arXiv ID 或 full title。
5. **`Outlier Suppression` 不等于 `SmoothQuant`。** 前者迁移 `LayerNorm γ` 并做 Token-Wise Clipping，original evidence 是 `BERT`/`RoBERTa`/`BART`；后者用 calibrated per-channel scaling 面向 large-`LLM W8A8`。`SmoothQuant` 的 matched tables 还显示 original Outlier Suppression recipe 在 `OPT-175B` 等 setup 中失效。
6. **`Flow Matching` 的 simulation-free 指 training，不是 sampling。** 训练 loss 不需要先积分 ODE；生成时仍需 numerical ODE solver。Conditional `OT` pair 是 straight path，也不保证 learned marginal flow 全局最优、完全笔直或可以 one-step sampling。
7. **`FlashSAC` 的 official award 名称是 `Outstanding Paper Award`，而且它不是 VLA paper。** RSS abstract、arXiv v2 与 live repository 分别使用 `50+ / 60+ / 100+ tasks` wording，不能把 live repository capability 改写成 paper evidence；sim-to-real comparison 还应核对 Table 14 的 algorithm-specific reward coefficients。
8. **`Self-Flow` 不等于 *From SRA to Self-Flow*。** ICML 2026 paper 是 arXiv:2603.06507，提出 `Dual-Timestep Scheduling + EMA feature target`；arXiv:2607.02508 是后续 mechanism critique，用 `Attention Separation` 主张主要收益来自 data augmentation。后者不能被写成原 paper 的作者自我修正，也不能据此抹掉原 paper 的 multi-modal performance evidence。
9. **两篇 `WorldCache` 不是同一论文。** arXiv:2603.22286 是 `Content-Aware Caching`，采用 `CFC/SWD/OFA/ATS`；arXiv:2603.06331 是 `Heterogeneous Token Caching`，采用 `CHTP/CAS`。引用时必须带 subtitle 或 arXiv ID，且 H200 与 A800 speedup 不能直接比较。

两位教授的名字是研究方向提示，而不是“把其全部论文都读完”的指令。去重后的 optional 跟进列表见 [Researcher watchlist: Song Han and Dan Alistarh](RESEARCHER_WATCHLIST.md)。

新增 quantization branch：`GPTQ → WaterSIC（rate allocation）`；`QAT + KD → GRACE（teacher information selection）`。两者的 entropy / information-budget 概念不等于已实现 control-aware VLA。

## 阅读主线与方法关系

### VLA 主线

`OpenVLA: discrete autoregressive action-token baseline`  
↔ `Flow Matching: simulation-free Conditional Flow Matching + linear conditional OT path`  
↳ `REPA / iREPA: frozen-teacher alignment → spatial projector and target normalization → VLA/WAM prior-art boundary`  
↳ `Self-Flow: dual-timestep tokens + internal EMA teacher → image/video/audio representation learning → SIMPLER joint video-action transfer`  
&nbsp;&nbsp;↳ `From SRA to Self-Flow: Attention Separation → mechanism challenge from cross-noise interaction to data augmentation`  
→ `π₀: 把该 objective condition on VLM context，生成 continuous action chunks`  
→ `FAST: compressed trajectory tokens for scalable autoregressive training`  
→ `π₀.₅: FAST pretraining + flow post-training + hierarchical open-world generalization`  
&nbsp;&nbsp;↳ `FlashVLA: staggered streaming flow chunks + chunk-wise causal attention for fast asynchronous execution`  
→ `base π₀.₆` 分成两条互补路线：  
&nbsp;&nbsp;↳ `π*₀.₆: learning from experience with value/advantage conditioning`  
&nbsp;&nbsp;↳ `MEM: remembering experience with short-term video + long-term text memory`  
→ `ω-0: latent future prediction auxiliary for humanoid control`  
→ `BitVLA: architecture/training redesigned for native ultra-low-bit weights`  
→ `HB-VLA: action-aware Hessian + Haar-domain 1-bit PTQ for existing VLA checkpoints`

这不是简单的 leaderboard 顺序。它分别改变 `action representation`、training data、generalization mechanism、experience/reward loop、temporal context、world-model-like auxiliary 和 deployment precision。

BitVLA companion dependency 可以单独记成：

`LLaVA multimodal curriculum + Apprentice low-precision distillation prior + OpenVLA-OFT action adaptation + bitsandbytes PTQ baseline → BitVLA synthesis`

### Robot RL systems companion branch

`PPO: reliable on-policy high-throughput baseline`  
→ `SAC: replay-based maximum-entropy off-policy learning`  
→ `FlashSAC: large replay/model/batch + few updates + bounded critic dynamics + correlated exploration`  
→ `FYP question: can sample reuse, critic stability, quantization, and closed-loop deployment cost be evaluated under matched data/hardware/control protocols?`

FlashSAC 是独立 Robot RL systems companion，不属于 VLA architecture lineage。它的价值是训练你把 environment steps、wall-clock、GPU/simulator throughput、policy frequency、low-level controller rate、reward shaping 与 real-world behavior 分开。

### Quantization 主线

`FP8 / MX / NVFP4: 先决定可表示的 number format 与 scaling granularity`  
→ `GPTQ: 用 approximate second-order information 补偿 weight-only quantization error`  
→ `Outlier Suppression: 诊断 LayerNorm γ amplifier，并用 Gamma Migration + Token-Wise Clipping 抑制 structured outliers`  
→ `SmoothQuant / AWQ: 重新分配 weights 与 activations 的 quantization difficulty`  
→ `QuaRot: 用 fixed/randomized Hadamard rotation 分散 outliers`  
→ `SpinQuant: 在 quantized loss 下学习更好的 rotation`  
→ `HAQ: 在 hardware constraints 下搜索 mixed precision`  
→ `RaBitQ: corrected inner-product estimator + high-probability error bound`  
→ `TurboQuant: 直接优化 vector distortion / inner-product estimation`

### World-model caching branch

`Full DiT world-model denoising at every step`  
→ `WorldCache #21: probe/deep-block reuse + content/saliency-aware drift + online alignment + timestep scheduling`  
→ `WorldCache #22: stable/linear/chaotic token prediction + chaotic-error refresh budget`  
→ `FYP question: how do quantization noise, cache detection, temporal error and closed-loop action choice interact under one matched runtime?`

要持续问：一个方法是在改变 **representation**、**bit allocation**、**number format**、**kernel/runtime**，还是同时改变了多个层级？只报 nominal bit-width 会掩盖 scale、metadata、outlier path 与 kernel availability 的成本。

### Parameter-Efficient Fine-Tuning 主线

`Full fine-tuning: 为每个 task 更新并保存完整 model`  
→ `LoRA: freeze W₀，学习 low-rank ΔW=BA，并在 single-adapter inference 时 merge`  
→ `OpenVLA: 把 LoRA 扩到 visual-language-action adaptation，但改变 rank、target modules 与 evaluation target`

要把三项成本分开：`training states`、`task-specific storage`、`base-model inference cost`。LoRA 直接降低前两项；quantization、kernel 和 runtime optimization 才处理第三项。

## 跨论文比较框架

| 维度 | VLA 阅读时问 | Quantization 阅读时问 |
|---|---|---|
| Target | generalist pretraining、downstream adaptation 还是 closed-loop deployment？ | weights、activations、KV cache、gradients 还是 optimizer states？ |
| Representation | discrete tokens、continuous chunks、diffusion/flow，是否需要 history？ | integer、floating point、block floating point、vector codebook？ |
| Evidence | real-robot success、progress、task breadth、rollout 数量？ | perplexity/accuracy、distortion、latency、memory、power、kernel benchmark？ |
| Confound | model/data/robot/controller cadence 是否同时变化？ | algorithm 与 kernel optimization、scale metadata、outlier handling 是否混在一起？ |
| Reproducibility | data、reward、robot setup 与 evaluator 是否公开？ | calibration data、hardware、fused kernel 与 exact format semantics 是否公开？ |
| Deployment | P50/P99 latency、control frequency、memory、power？ | nominal compression 是否转化为 end-to-end speedup？ |

## 对 Final Year Project 最重要的 synthesis

[李老师当前公开研究方向](https://ofsoundof.github.io/)覆盖 efficient AI、model compression/deployment、software–hardware co-design，以及 efficient `LLM` / `VLM` / `VLA`。因此这份 list 的真正交叉点不是“给 VLA 套一个 LLM quantizer”，而是：

> 在不破坏 closed-loop behavior 的前提下，如何让 VLA 的 memory、latency、control frequency 与 power 满足实际 hardware constraints？

可以把后续研究假设收敛为：

1. **Component sensitivity:** vision encoder、language backbone、action expert 与 action head 对 low precision 的敏感性是否不同？
2. **Temporal robustness:** 相近的 offline action error 是否会产生不同的 closed-loop compounding error？
3. **Mixed-precision policy:** hardware-aware bit allocation 能否同时优化 success rate、P99 latency、memory 与 power，而不是只优化 model size？
4. **Evaluation protocol:** 至少同时报告 `task success/progress + action error + latency/control frequency + peak memory + power/energy`。

LoRA 为这里增加一个关键实验轴：在 matched data、hardware 与 rollout protocol 下，比较 `full fine-tuning`、language-backbone LoRA、visual/projector LoRA、action-expert LoRA 与 component-wise rank allocation。Trainable parameter count 只能解释 adaptation cost，不能替代 closed-loop deployment metrics。

Outlier Suppression 为 quantization path 增加另一个 experiment axis：先分 component 测量 outlier source、token/state importance 与 saturation，再比较 `Gamma Migration`、`SmoothQuant`、rotation 与 mixed-precision routing。对 `VLA`，large activation 可能是 removable noise，也可能是 rare corrective action；必须用 failure-heavy trajectories 与 closed-loop evidence 决定，不能只看 text calibration loss。

Flow Matching 为 action-generation path 增加第三个 experiment axis：在 matched checkpoint、data 与 hardware 下，交叉比较 probability path、timestep sampling 与 solver steps，并同时报告 endpoint action error、closed-loop success、P50/P99 latency、control frequency、peak memory 与 power。Image-generation `NFE` 不能直接替代这些 control/deployment measurements。

FlashSAC 为 RL/deployment path 增加第四个 experiment axis：在 matched simulator、environment steps、hardware、reward、controller cadence 与 seeds 下，比较 on-policy/off-policy data reuse、UTD、critic stability 与 wall-clock。若再加入 quantization，必须同时检查 critic norms/return stability 与 inference cost，不能假设 lower precision 只改变 latency。

两篇 WorldCache 为 inference path 增加第五个 axis：在 matched world/VLA model 上分别控制 precision 与 cache budget，并把 video/depth fidelity、action change、closed-loop success、cache hit、P50/P99 latency、peak memory 与 energy 放在同一 Pareto table。该 joint hypothesis 尚未被这三篇论文直接验证。

这些是由论文证据推导出的研究方向，属于 **inference / proposal**，不是任何单篇论文已经证明的结论。

## 每篇 README 怎么用

每份独立笔记都按同一结构整理：

- `Paper identity` 与 publication status
- `Background and prerequisites`
- `Problem`
- `Method` 与关键 equation / pipeline
- `Key innovation`
- `Experiments and main results`，保留 Table/Figure/Section locator
- `Authors' limitations` 与 `My critique`
- `Why it matters`，尤其是 VLA deployment 与李老师方向
- `20-minute route`、`90-minute route`
- `Reading questions`
- `Weekly meeting card`

读完后不要删掉问题；把自己的回答直接写在对应 README 下，组会前再提炼到 slides。

## Evidence boundary

- 所有 24 个 core entries 已用 primary full text 与 official metadata 核验；同一 work 的 arXiv / proceedings / project page 已合并，不重复计数。
- 24 篇 canonical core full texts、1 份 Outlier Suppression official supplemental、1 份 TurboQuant archived arXiv version、28 篇 companion papers（1 篇 VLA systems + 1 篇 Robot RL systems + 1 篇 PI series + 4 篇 BitVLA + 8 篇 quantization + 12 篇 representation-alignment route + 1 篇 Self-Flow mechanism critique）与两份 format specifications 已保存在对应 paper folder；文件版本、page count、既有 SHA-256（本次新增两篇仅做结构检查）与 acquisition source 见 [Local PDF inventory](PDF_INVENTORY.md)。
- 24 篇中，17 篇有 verified peer-reviewed / accepted conference record，7 篇是 arXiv preprint / official technical report；venue identity 与 local binary version 分开记录。
- VLA systems、Robot RL systems、PI series、BitVLA、quantization、representation-alignment 与 Self-Flow mechanism companions 是 extension/dependency/prior-art/map/critical readings，不改变 advisor-assigned core count。
- `verified-full-text` 表示 claim 已追到原文，不表示已经 independent replication。
- PI series 的 real-robot tasks/data、vendor format 的 hardware claims、以及近期 preprints 都需要保留 institutional/evidence caveat。
- 完整核验表见 [Source verification report](research/SOURCE_VERIFICATION.md)，范围与检索规则见 [Scope and method](research/SCOPE_AND_METHOD.md)，交付检查结果见 [Delivery validation](research/VALIDATION.md)。

2026-09-05 新增的 #23 / #24 来源与验证结果见 [师兄报告交付检查](research/SENIOR_REPORT_DELIVERY_2026-09-05.md)。此前日期的 source verification / validation 是历史快照。

## Suggested progress marks

在每篇 README 顶部自行加一行：

```text
My status: unread | 20-min skim | 90-min read | deep-read | presented
```

再记录三件事：`date`、`one unresolved question`、`one slide-ready evidence item`。这样即使没有读完，也能准确说明完成边界。

## AI-assisted preparation disclosure

本 guide 由 AI-assisted literature workflow 生成，并对 primary sources、publication identity 与 result locators 做了结构化核验。它适合做 reading map 与 first-pass analysis，但不能替代你本人阅读原文；组会上引用任何 numerical claim 前，应再次打开对应 README 中链接的 Table/Figure/Section。
