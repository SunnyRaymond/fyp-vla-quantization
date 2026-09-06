# Phase-2 Evidence Packet — Vision-Language-Action

> 用途：给后续每篇论文的独立 README 和总 README 提供可追溯证据，不是最终阅读笔记。  
> 检索/核验日期：2026-08-22（Asia/Singapore）。  
> 术语、模型名、objective、metric 均保留英文。所有结果都注明来自 paper 的 `source claim`，我的解释或质疑则显式标为 `inference / critique`。

## 0. Scope、身份消歧与证据等级

| ReadingList 条目 | 本 packet 的 canonical mapping | 数量口径 | Verification label |
|---|---|---:|---|
| OpenVLA | *OpenVLA: An Open-Source Vision-Language-Action Model* | 1 paper | **verified — peer-reviewed venue + original full text** |
| Physical Intelligence `pi_0` | *π₀: A Vision-Language-Action Flow Model for General Robot Control* | 1 paper | **verified — RSS 2025 + original full text** |
| Physical Intelligence `pi_0.5` | *π₀.₅: a Vision-Language-Action Model with Open-World Generalization* | 1 paper | **verified — CoRL 2025 oral + original full text** |
| Physical Intelligence `pi_0.6` | *π*₀.₆: a VLA That Learns From Experience* | 1 paper | **verified metadata/full text; preprint, identity mapping requires note** |
| `ω-0` | *ω-0: A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation* | 1 paper | **verified metadata/full text; preprint** |
| BitVLA | *BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation* | 1 paper | **verified metadata/full text; arXiv says “Work in progress”** |

**最终 core VLA 论文数量：6。** `pi_0.6` 是唯一实质身份歧义：Physical Intelligence 没有找到一篇 canonical title 恰为“π₀.₆”的独立论文；最直接的 official research paper 是 `π*₀.₆`，其 paper 同时定义 `π₀.₆` base model 与经过 RECAP reinforcement learning 的 `π*₀.₆`。2026 年另有 *MEM: Multi-Scale Embodied Memory for Vision Language Action Models* 使用 `π₀.₆-MEM`，但它是 memory extension，不与本条合并。MEM 于 2026-08-25 被单独补为 companion reading，不改变六篇 core count。

### Verification 约定

- `verified — peer-reviewed venue + original full text`：venue/proceedings metadata 与原论文全文相互核验。
- `verified metadata/full text; preprint`：arXiv metadata、version 与 PDF 已核验，但没有发现可确认的 peer-reviewed venue。
- 所有页码均为 PDF viewer 的 1-based page；当 PDF 内印刷页码不同，优先写 `PDF p.` 并同时给 `Table/Figure/Section`。
- 图中没有数值标签的柱形高度只写 `approx. plot-read`，不可在汇报中冒充 exact number。

## 1. OpenVLA

### Canonical metadata

- **Title:** *OpenVLA: An Open-Source Vision-Language-Action Model*
- **Authors:** Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P. Foster, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, Chelsea Finn.
- **Year/version:** arXiv 2024, `v3` (2024-09-05); peer-reviewed proceedings published 2025.
- **Venue/status:** Conference on Robot Learning (CoRL 2024), PMLR 270:2679–2713. Peer reviewed.
- **Identifiers / official sources:** [PMLR proceedings](https://proceedings.mlr.press/v270/kim25c.html) · [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) · [arXiv DOI](https://doi.org/10.48550/arXiv.2406.09246) · [official project](https://openvla.github.io/)
- **Verification label:** **verified — peer-reviewed venue + original full text**.
- **Version decision:** 用 PMLR author list 与 venue citation 作为 canonical citation；arXiv 版本只记录版本历史，不重复计数。

### Background

RT-2 证明了把 robot action 表示为 language token、在 pretrained VLM 上做 co-fine-tuning 可以得到通用 VLA，但模型和 training recipe 不开放。Open X-Embodiment 随后聚合跨 robot 的 trajectory，却仍缺少一个可下载、可 fine-tune、可部署并能系统研究 VLA design choice 的强 baseline。读本文前应理解 `VLM`, `behavior cloning`, `action tokenization`, `cross-embodiment data`, `LoRA`, `quantization`。

### Problem

目标是训练一个 open-source 7B VLA，使单一 model 能从 image + language instruction 直接预测 continuous robot control，并回答两个实用问题：

1. 大规模跨 embodiment pretraining 是否能在多 robot、多 task 上达到接近/超过 closed model 的表现？
2. 这样的大模型能否用 parameter-efficient fine-tuning 与 low-bit quantization 在普通研究硬件上适配和部署？

### Method

**Components.** `Prismatic-7B` VLM：fused `SigLIP` + `DINOv2` visual encoders（合计约 600M parameters），2-layer MLP projector，`Llama-2 7B` language backbone；输入为 224×224 image 与 language instruction。

**Action representation / objective.** 每个 action dimension 依据训练集的 1st–99th percentile 独立分成 256 bins；用 Llama vocabulary 中最少使用的 256 个 token 表示 action。训练采用 standard next-token cross-entropy，但 loss 只施加在 action tokens 上。这是 autoregressive discrete-action policy，不是 diffusion/flow policy。（Sec. 3, PDF pp. 4–6）

**Training.** 从 Open X-Embodiment 筛出约 970k robot demonstrations；all parameters，包括 visual encoder，均更新。27 epochs，learning rate `2e-5`，batch size 2048；64×A100 训练约 14 天，约 21,500 A100-hours。（Sec. 3.2, PDF pp. 5–6）

**Inference.** 逐 action-token autoregressive decode；bf16 checkpoint 约 15 GB，在 RTX 4090 约 6 Hz。论文也测试 LoRA、INT8、INT4，但 downstream controller 是否 blocking 会显著影响表观成功率。（Sec. 4.2/Appendix, PDF pp. 9–10, 35）

### Key Innovation

`source claim`：OpenVLA 的核心不是某个全新的 Transformer block，而是把强 open VLM、跨 embodiment 数据、action tokenization、开放权重/代码与系统性的 adaptation/deployment study 集成成第一个可复现实用的 7B generalist VLA。

`inference`：它最重要的研究价值是成为“可拆解的 RT-2”：后续论文可以把 action head、visual representation、fine-tuning 和 quantization 的改动放到同一个公开基准上比较。

### Main Results

| Evidence type | Benchmark / metric | Result | Locator |
|---|---|---|---|
| `source claim; exact` | BridgeData V2, average success, 17 tasks / 170 rollouts | OpenVLA **70.6±3.2%**; RT-2-X **50.6±3.5%**; Octo **20.0±2.6%**; RT-1-X **18.5±2.7%** | Table 4, PDF p. 26 |
| `source claim; exact` | Google Robot, average success, 12 tasks / 60 rollouts | OpenVLA **85.0±4.6%**; RT-2-X **78.3±5.4%**; RT-1-X **33.3±6.1%**; Octo **26.7±5.8%** | Table 6, PDF p. 28 |
| `source claim; exact` | Downstream adaptation, average success | Franka-tabletop: OpenVLA **67.2±4.0%** vs Diffusion Policy **48.5±4.9%**; Franka-DROID: **58.3±7.2%** vs **35.0±8.0%** | Table 7, PDF p. 32 |
| `source claim; exact` | LoRA adaptation | rank-32 LoRA **68.2±7.5%** vs full fine-tuning **69.7±7.2%**; 97.6M trainable params (**1.4%**), 59.7 GB vs 163.3 GB; claimed ~8× lower compute | Table 1, PDF p. 10 |
| `source claim; exact, platform-dependent` | Quantization on Bridge | bf16 **71.3±4.8%**, 16.8 GB; INT4 **71.9±4.7%**, 7.0 GB; INT8 **58.1±5.1%**, 10.2 GB | Table 2, PDF p. 10; Table 5, PDF p. 26 |

**Interpretation boundary.** `inference`：BridgeData 的 +20.0 percentage points 是 Table 4 上的 directly computed difference，不等于 abstract 在 29 tasks 汇总口径下写的 +16.5。INT4 “不掉点”也不能单纯归因于 quantization regularization；blocking-control ablation 中 bf16/INT8/INT4 分别为 70.0/74.4/68.8%，说明 controller timing 是 confound（Table 11, PDF p. 35）。

### Limitations

**Authors' stated limitations（Sec. 6, PDF p. 11）**

- 只处理 single image；不原生处理 multi-view、proprioception 或 observation history。
- inference throughput 对 50 Hz control 明显不足。
- real-world reliability 多数仍低于 90%，离 unattended deployment 尚远。
- base VLM size、co-training 与 visual feature choice 等 design axes 没有穷尽。

**Critical reading（inference / critique）**

- closed RT-2-X 不能在完全相同 recipe 下重训，comparison 不是 compute/data-matched；BridgeData 还需兼容不同 no-op conventions。
- real-robot task 数与 rollout 数有限，且不少 metric 是研究组自定义，不是统一公开 leaderboard。
- quantization evaluation 同时改变 latency 与 control cadence，因此“bit-width → task performance”的因果结论需要 hardware/controller controlled experiment。
- 逐 token decode 是 2024 年设计的主要 latency bottleneck；后续 OpenVLA-OFT 的 chunked/parallel action decoding 应作为更新阅读。

### Why It Matters

这是理解 open-source VLA lineage 的锚点：后面的 efficiency work（例如 BitVLA）都在回答“怎样比 OpenVLA 更小、更快”；flow/diffusion 路线（π₀）则在回答“为什么不要把 continuous action 硬离散成 autoregressive token”。

### Prerequisites

- Transformer/VLM forward path；`SigLIP`, `DINOv2`, `Llama-2` 的角色。
- behavior cloning 与 cross-entropy；continuous-to-discrete quantile binning。
- Open X-Embodiment、LoRA、weight-only quantization、robot control frequency。

### Reading Questions

1. 用 256 个 bins 离散每个 action dimension，误差来自 quantization 还是 autoregressive exposure bias？
2. fused SigLIP+DINOv2 分别提供 semantic 与 spatial information 的证据充分吗？
3. 跨 embodiment training 的增益来自数据量、task diversity，还是 embodiment diversity？
4. LoRA 几乎追平 full fine-tuning 时，应该更新 visual encoder 还是只更新 projector/action-related layers？
5. 如果把 action head 换成 flow matching，哪些 OpenVLA component 仍值得保留？

### 组会可讲的 3 个重点

1. **Open baseline 的意义大于 architecture novelty**：7B VLM + 970k demonstrations + open weights/code，建立可复现 VLA 基线。
2. **结果要按实验口径说**：Bridge 70.6%、Google 85.0%；不能把不同 task aggregation 的 improvement 混用。
3. **Efficiency insight**：LoRA 只训练 1.4% parameters 几乎追平 full fine-tuning；INT4 memory 降至 7.0 GB，但 latency/control confound 必须说明。

---

## 2. π₀

### Canonical metadata

- **Title:** *π₀: A Vision-Language-Action Flow Model for General Robot Control*
- **Authors:** Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, Laura Smith, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, Ury Zhilinsky.
- **Year/version:** arXiv 2024; latest checked `v4` (2026-01-08).
- **Venue/status:** Robotics: Science and Systems (RSS) 2025, paper 010. Peer reviewed.
- **Identifiers / official sources:** [RSS proceedings](https://www.roboticsproceedings.org/rss21/p010.html) · [DOI:10.15607/RSS.2025.XXI.010](https://doi.org/10.15607/RSS.2025.XXI.010) · [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) · [PI official paper](https://www.pi.website/download/pi0.pdf) · [PI official release](https://www.pi.website/blog/pi0)
- **Verification label:** **verified — RSS 2025 + original full text**.

### Background

OpenVLA 把 action 当作 token，但 high-frequency dexterous control 是 continuous、多峰且 temporal-correlated 的。Diffusion Policy 能生成 continuous action chunks，却通常缺少 internet-scale semantic knowledge。π₀ 将 pretrained VLM 与 conditional flow matching action expert 接起来，尝试同时保留 semantic reasoning 与 expressive continuous control。

### Problem

训练一个 single generalist policy，使其跨 7 种 robot configurations、68 个 tasks 工作，并通过 pretraining + task-specific fine-tuning 处理从 tabletop manipulation 到 laundry folding、mobile manipulation 等高灵巧任务。

### Method

**Components.** `PaliGemma 3B` VLM + 约 300M parameter `action expert`，总计约 3.3B。输入 2–3 个 RGB views、language、proprioception；VLM token 与 noisy action token 在 joint attention 中交互，但 action token routed through 独立 expert。（Sec. III, PDF pp. 4–6）

**Objective.** 预测 horizon `H=50` 的 continuous action chunk。给 clean action `A_t` 加 Gaussian noise，通过 conditional flow matching 学 velocity field；loss 是预测 velocity 与 target vector field 的 MSE。与 OpenVLA 的 categorical cross-entropy 相比，它保留 continuous geometry。

**Training.** PI 自采数据覆盖 7 robot configurations、68 tasks，约 903M timesteps、约 10,000 robot-hours；再混合 22 种 robots 的 Open X-Embodiment 数据（open-source mixture 约占 9.1%）。task-robot group 采用近似 `n^0.43` weighting，之后用每个 downstream task 约 5 到 100+ hours 数据 fine-tune。（Sec. IV, PDF pp. 6–7）

**Inference.** 从 Gaussian action noise 出发，用 10-step Euler integration（`δ=0.1`）沿 learned flow 得到 50-step action chunk；执行前一部分再 replan。可缓存 VLM context。20/50 Hz platforms 通过 chunk execution 支撑高频 control。（Sec. III-E; Appendix B, PDF pp. 6, 16）

### Key Innovation

`source claim`：用 pretrained VLM 提供 semantic/visual knowledge，用 separate action expert + flow matching 生成 high-frequency continuous action chunks，并在 heterogeneous robot data 上统一训练。

`inference`：π₀ 奠定 PI 后续模型族的 low-level motor interface；π₀.₅ 主要增加 open-world/hierarchical semantics，π*₀.₆ 则主要增加 reinforcement learning，而 flow action expert 这条主线延续下来。

### Main Results

| Evidence type | Benchmark / metric | Result | Locator |
|---|---|---|---|
| `source claim; approximate plot-read` | Base-model normalized task progress, 10 episodes/task | π₀ roughly shirt folding **1.00**, bussing easy **0.97**, bussing hard **0.88**, groceries **0.78**, toast **0.75**; best/near-best across shown tasks | Fig. 7, PDF p. 8 |
| `source claim; approximate plot-read` | Language following, 10 trials/task | π₀ approx. bussing **0.94**, groceries **0.70**, table setting **0.90** vs π₀-small approx. **0.70/0.32/0.32** | Fig. 9, PDF p. 9 |
| `source claim; approximate plot-read + exact textual threshold` | Fine-tuned dexterous/mobile tasks, 10 trials each | full pretrain+fine-tune approx. normalized progress: laundry **0.82**, table bussing **0.88**, mobile laundry **0.92**, dryer **0.72**, box **0.64**, to-go box **0.70**, eggs **0.84**; authors state **>50% of maximum score on every task** | Fig. 13 + Sec. VI-C, PDF p. 11 |
| `source claim; exact` | End-to-end inference timing, RTX 4090 | **73 ms** on-board and **86 ms** off-board for the reported pipeline | Table I, PDF p. 16 |

**Interpretation boundary.** 多数 robotics result 是 bar chart，不提供 tabulated exact values；上面的 decimal 只能作为读图近似，正式汇报应以“趋势 + 10 trials/task”讲，不要给虚假的小数精度。

### Limitations

**Authors' stated limitations（Sec. VII, PDF pp. 11–12）**

- 哪些 pretraining data 以及怎样的 mixing weight 最有效仍不清楚。
- 某些 task 仍不可靠，且新任务需要多少数据很难预先估计。
- 相距很远的 robot/task 之间能否稳定 positive transfer 尚未建立。
- 只用 high-quality demonstrations 会让 policy 缺少 failure recovery；zero-shot behavior 也不够 fluent。

**Critical reading（inference / critique）**

- 主要结果多为 custom normalized progress bar，且每 task 约 10 trials，统计不确定性有限。
- baseline 的 training steps/epochs 并非全部严格匹配；不能把差异全部归因于 flow matching。
- 绝大多数 PI training data 与模型权重不公开，复现 data mixture 和 scaling law 很困难。
- `progress` 不等于 strict task success；高 partial progress 可能掩盖关键最后一步失败。

### Why It Matters

π₀ 是 VLA 从 discrete token policy 转向 continuous generative action model 的关键节点，也是阅读 π₀.₅ 和 π*₀.₆ 的必修前置。它把 action chunking、flow matching 与 VLM backbone 组合成可扩展 generalist controller。

### Prerequisites

- conditional flow matching / diffusion basics；ODE sampling 与 Euler solver。
- PaliGemma/VLM tokenization、KV cache、mixture sampling。
- action chunking、receding-horizon control、proprioception。

### Reading Questions

1. separate action expert 为什么比直接让 VLM 输出 flow field 更稳定？
2. `H=50` 和 executed chunk length 怎样影响 latency、reactivity 与 temporal consistency？
3. flow matching 的优势来自 continuous representation 还是 parallel chunk prediction？
4. `n^0.43` mixture weighting 在少数高质量 task 与大规模低质量 task 之间作了什么折中？
5. progress metric 是否系统性高估“几乎完成但最终失败”的 policy？

### 组会可讲的 3 个重点

1. **Architecture shift**：PaliGemma 负责 semantic context，300M action expert 用 flow matching 一次生成 50-step continuous chunk。
2. **Scale**：7 robot configurations、68 tasks、约 10,000 hours，是结果的重要组成而不只是 model design。
3. **Evidence caveat**：论文显示跨高灵巧任务的强趋势，但很多数值来自 10-trial bar plots；不要把 normalized progress 当 success rate。

---

## 3. π₀.₅

### Canonical metadata

- **Title:** *π₀.₅: a Vision-Language-Action Model with Open-World Generalization*
- **Authors:** Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, Ury Zhilinsky (arXiv 另列 corporate author `Physical Intelligence`).
- **Year/version:** 2025, arXiv `v1` (2025-04-22)；checked 2026-08-22 时未见更新版本。
- **Venue/status:** CoRL 2025 oral, PMLR 305:17–40. Peer reviewed.
- **Identifiers / official sources:** [PMLR volume](https://proceedings.mlr.press/v305/) · [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) · [arXiv DOI](https://doi.org/10.48550/arXiv.2504.16054) · [PI official paper](https://www.pi.website/download/pi05.pdf) · [PI official release](https://www.pi.website/blog/pi0.5)
- **Verification label:** **verified — CoRL 2025 oral + original full text**.

### Background

π₀ 展示了多 robot、多 skill pretraining，但 evaluation 仍大多发生在 training-like lab setup。真正的 household robot 必须进入 unseen home，理解抽象任务、分解 subtask，并把 web-scale semantic knowledge 与 precise continuous action 对齐。π₀.₅ 把 high-level text subtask prediction 与 low-level flow action generation 放进一个 model，并加入 heterogeneous data co-training。

### Problem

在没有为目标 home 收集 demonstrations 的条件下，执行约 10–15 minute long-horizon household task；核心是 `open-world generalization`：novel environment、novel object layout 与 language-level task decomposition 同时变化。

### Method

**Components / factorization.** 基于 π₀/PaliGemma，将 high-level subtask text `l̂` 与 low-level actions `a` 联合建模：`π(a,l̂|o,l)=π(a|o,l̂)π(l̂|o,l)`。同一 Transformer 既 autoregressively 生成 subtask text/FAST action tokens，又用 continuous flow action expert 生成 low-level chunk。（Sec. IV, PDF pp. 5–7）

**Objective.** Pretraining 时以 cross-entropy 学 text、object detection 与 FAST-discretized actions（flow coefficient `α=0`）；post-training 加 continuous flow-matching loss，joint objective 为 CE + `α·flow loss`，论文设 `α=10`。

**Data.** 约 400 hours mobile-manipulator data，覆盖约 100 homes；另混合 multi-environment non-mobile robot data、cross-embodiment lab data、high-level subtask labels/bounding boxes、web caption/VQA/localization data 与 verbal-instruction demonstrations。约 11% mobile examples 含人工 high-level verbal instruction。（Sec. III/IV, PDF pp. 4–7）

**Training / inference.** 280k pretraining steps + 80k post-training steps；action normalization 用 1st–99th percentile。部署时 high-level policy 生成/更新 textual subtask，low-level flow head 以约 10 denoising steps 预测约 49–50-step chunk；mobile platform command rate 为 50 Hz。

### Key Innovation

`source claim`：将 heterogeneous co-training、explicit high-level semantic prediction 与 low-level flow control 合为一个 VLA，使 model 能把 web knowledge 与多 environment robot experience 迁移到 unseen homes。

`inference`：它不是简单“π₀ 加更多数据”；关键变量是 cross-environment (`CE`) 与 multi-embodiment (`ME`) diversity 加上 intermediate high-level language。其 ablation 显示 open-world behavior 更依赖 diversity，而非仅依赖 web data。

### Main Results

| Evidence type | Benchmark / metric | Result | Locator |
|---|---|---|---|
| `source claim; approximate plot-read` | Open-world real-home tasks, 10 trials/task | Across 3 real homes + mock environments, task progress roughly spans **65–94%**; individual examples include Home 1 kitchen drawer ~90%, bedroom laundry ~80%, Home 2 dishes ~94%, laundry ~90% | Fig. 7, PDF p. 8 |
| `source claim; approximate plot-read` | Number of training environments vs avg. progress | ~3 locations **14%**, 12 **47%**, 22 **60%**, 53 **76%**, 82 **66%**, 104 **87%**; same-domain baseline ~83%; 104-location no-pretraining ~5% | Fig. 8, PDF p. 9 |
| `source claim; plot-read, exact p-values` | Data-mixture ablation, avg. progress | full π₀.₅ ~**78%**; no web data ~72%, `p=.385` (not significant); no cross-environment ~51%, `p<.001`; no multi-embodiment ~53%, `p<.001`; neither ~40%, `p<.001` | Fig. 10, PDF p. 10 |
| `source claim; plot-read, exact p-values` | High-level ablation | full ~**78%**; implicit high-level ~71%, `p=.144`; no high-level ~62%, `p=.011`; no verbal instruction ~60%, `p=.009`; no web data ~60%, `p=.008`; GPT-4 high-level ~58%, `p=.002`; human high-level ~63%, `p=.016` | Fig. 13, PDF p. 11 |

**Interpretation boundary.** 结果柱形没有逐项数值表；percentage 仅为 approximate plot-read，p-values 是正文/图注中的 exact report。`noWD` 在不同 ablation context 下不能横向当成完全相同 intervention。

### Limitations

**Authors' stated limitations（Sec. VI, PDF p. 11）**

- unfamiliar handle/cabinet geometry 会造成失败。
- occlusion 与 partial observability 仍突出。
- high-level planner 有时被 distractor 影响、重复或选择错误 subtask。
- language prompt 较简单，能力边界受 training distribution 限制。
- context window / memory 较弱，长任务会忘记先前状态。
- 不同 data source 的最佳组合仍未系统解决。

**Critical reading（inference / critique）**

- 核心 quantitative evaluation 只覆盖少量 canonical long-horizon tasks，每 policy/task 约 10 trials。
- custom progress rubric 容许 partial credit；不能替代 strict success 与 time-to-completion。
- 数值主要是 plot-read，缺少 confidence interval table；小样本下 p-value 对 test choice 很敏感。
- data、weights、annotation pipeline 主要不公开，外部团队难以验证“data diversity”与 architecture 的独立贡献。

### Why It Matters

π₀.₅ 把 VLA 讨论从“一个 lab 里会多少 skill”推进到“能否进入没见过的 home”。它也给出一个重要 empirical lesson：cross-environment 与 multi-embodiment experience 比简单添加 web data 更直接地决定 physical generalization。

### Prerequisites

- π₀ 的 flow action expert 与 action chunking。
- hierarchical policy / task decomposition；FAST action tokenization。
- domain generalization、co-training、ablation 与 paired significance test。

### Reading Questions

1. explicit subtask text 是 causal bottleneck，还是只提供 auxiliary supervision？
2. 为什么 `noWD` 不显著，而 `noCE`/`noME` 显著？这对 data collection budget 有什么含义？
3. high-level planner 的错误怎样传播到 low-level flow controller？
4. 100 homes 的 diversity 应按 geometry、object、lighting 还是 task 分层度量？
5. 加入 episodic memory 后，哪些 π₀.₅ failure mode 应最先改善？

### 组会可讲的 3 个重点

1. **Open-world 目标**：在 unseen home 做约 10–15 minute task，不只是 lab benchmark transfer。
2. **Unified hierarchy**：同一 model 预测 textual subtask 与 continuous action chunk。
3. **Ablation takeaway**：去掉 cross-environment 或 multi-embodiment data，平均 progress 约从 78% 降到 51–53%；web data 单独移除未显著。

---

## 4. π*₀.₆（ReadingList `pi_0.6` 的官方最合理映射）

### Canonical metadata

- **Title:** *π*₀.₆: a VLA That Learns From Experience*
- **Authors:** Physical Intelligence; Ali Amin et al.（作者数很大，完整 canonical list 以 arXiv metadata / official PDF 首页为准；本 packet 不为压缩篇幅重排 corporate author list。）
- **Year/version:** 2025; arXiv `2511.14759`, latest checked `v2` (2025-11-19); PI official release 2025-11-17.
- **Venue/status:** arXiv technical report / preprint；截至检索日未确认 peer-reviewed venue。
- **Identifiers / official sources:** [arXiv:2511.14759](https://arxiv.org/abs/2511.14759) · [arXiv DOI](https://doi.org/10.48550/arXiv.2511.14759) · [PI official paper](https://www.pi.website/download/pistar06.pdf) · [PI official release](https://www.pi.website/blog/pistar06)
- **Verification label:** **verified metadata/full text; preprint; identity mapping requires note**.

### Identity resolution（必须在总 README 保留）

- `source claim`：paper 定义 `π₀.₆` base VLA，以及用 `RECAP` 做 reinforcement learning 后的 `π*₀.₆`。
- `inference`：ReadingList 的 bare `pi_0.6` 最可能指这一 official model generation/release；因此 canonical paper 映射为 π*₀.₆，而不是虚构一篇题为“π₀.₆”的论文。
- **Core-count exclusion / companion inclusion：** [MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://arxiv.org/abs/2603.03596)（official [research page](https://www.pi.website/research/memory)）使用 `π₀.₆-MEM`，但这是 2026 年 memory extension，不能与 π*₀.₆ 合并计数；它现已作为独立 companion reading 保存。

### Background

π₀/π₀.₅ 主要依赖 expert demonstrations 与 behavior cloning。BC 会复现成功 trajectory，却较少学习“哪些状态差、怎样从失败恢复”，也不能自然利用 autonomous rollout 的 success/failure signal。π*₀.₆ 把已有 generalist VLA 作为 prior，再用 deployment experience 做 offline/batch reinforcement learning。

### Problem

在真实机器人上提升 long-horizon、contact-rich task 的 reliability 与 throughput，同时避免从头 online RL 的高风险和高 sample cost；并让一个 multi-task VLA 从 autonomous rollout、binary success label 与可选 human correction 中持续改进。

### Method

**Base model (`π₀.₆`).** `Gemma 3 4B` backbone + 约 860M action expert；结合 continuous flow actions 与 FAST discrete action tokens，并用 `Knowledge Insulation`（stop-gradient）减少 action learning 对 pretrained knowledge 的破坏；同时输出 high-level subtask 与 50 Hz action chunks。（Sec. 3, PDF pp. 3–5）

**RECAP loop.** (1) 收集 autonomous rollouts，可加入 teleoperated corrections；(2) 用 Monte-Carlo return 训练 multi-task distributional value function `pφ(V|o,l)`，value discretized into `B=201` bins；(3) 根据 `A=return−V` 构造 positive/negative advantage condition，并以 text condition `Advantage: positive/negative` 训练 policy；部署时固定 positive condition。（Sec. 4, PDF pp. 5–7）

**Objective.** policy training 采用 binarized advantage filter `I(A>ε)`；保留 base VLA 的 flow/FAST objectives，只让 high-advantage action 更可能在 positive condition 下出现。human correction 被强制标作 positive。训练中 advantage condition dropout 30%，避免完全依赖该 token。

**Training / inference.** Pretraining 阶段 positive threshold 约选 top 30% demonstrations，task fine-tuning 多数约 top 40% evaluation rollouts（diverse laundry 约 10%）；以 batch offline iteration 重训。Inference 是 greedy positive-conditioned policy，不在执行时做 online value search。

### Key Innovation

`source claim`：把 heterogeneous real-world experience 转成可复用的 advantage-conditioned VLA training recipe，value model 同时服务数据筛选与 policy improvement；可迭代加入 autonomous data 与 human corrections。

`inference`：相对传统 task-specific offline RL，RECAP 的价值在于把 generalist VLA 的 language/task prior 与真实 deployment failure data 接起来；它更像“经验驱动的 post-training layer”，不是替代 π₀ 系列 pretraining。

### Main Results

| Evidence type | Benchmark / metric | Result | Locator |
|---|---|---|---|
| `source claim; exact text + plot-read` | Four task families after RECAP | authors report throughput **more than doubles** on diverse laundry and espresso vs offline-RL+SFT; failure rate roughly halves; all tasks except diverse laundry exceed **90% success**. Plot-read throughput approx. simple laundry **60**, diverse laundry **8.4**, espresso **29**, box assembly **13.3 successes/hour** | Figs. 7–8 + Sec. 5.2, PDF p. 9 |
| `source claim; exact study size/text` | Iterative laundry/box improvement | laundry: two iterations, **300 trajectories on four robots per iteration**, overall throughput **+50%**; box: **600 autonomous + 360 correction trajectories/iteration**, after iteration 2 throughput about **2×** | Figs. 9–10 + Sec. 5.3, PDF p. 10 |
| `source claim; exact` | Strict one-shirt laundry failure mode | two RECAP iterations, **600 trajectories each**, final **97% success** | Fig. 12 + Sec. 5.4, PDF p. 11 |
| `source claim; comparative trend` | Algorithm ablation | RECAP outperforms AWR and PPO variants under the shown real-robot setup; PPO uses trust-region coefficient `η=0.01` | Fig. 11, PDF p. 10 |

**Interpretation boundary.** “2× throughput”是相对该实验中 authors' offline-RL+SFT 或 iteration-0 policy 的 within-system comparison，不等同于跨论文 universal speedup。多数图无完整 confidence interval table。

### Limitations

**Authors' stated limitations（Sec. 7, PDF p. 11）**

- pipeline 仍不 fully autonomous：需要 human success labels、interventions/corrections 与 environment resets。
- exploration 基本是 greedy deployment，可能看不到能突破局部最优的 trajectory。
- training 是 batch offline iterations，而非 concurrent online update。
- real-robot RL 仍有昂贵的 sample collection 与 operational logistics。

**Critical reading（inference / critique）**

- PI base model/data 不公开，外部很难区分 RECAP gain 与 proprietary pretraining strength。
- success/throughput 涉及人工 task judgement，且许多结果是 plot values，统计透明度有限。
- per-task advantage threshold、classifier-free guidance、correction labeling 都引入 tuning degrees of freedom。
- benchmark 主要是 PI 自定义 household/industrial tasks，没有公共、跨团队的 RL-VLA benchmark。
- 截至检索日为 preprint，尚无已确认 peer review。

### Why It Matters

它标志 PI 路线从“学 demonstrations”转为“学 deployment experience”。对项目研究尤其重要的是：模型压缩/部署不只影响 latency，还影响能收集多少 autonomous experience，进而改变 RL data flywheel。

### Prerequisites

- π₀/π₀.₅ architecture；behavior cloning 与 flow matching。
- offline RL、advantage、Monte-Carlo return、distributional value function。
- AWR/PPO、human correction、batch policy iteration。

### Reading Questions

1. binary positive/negative advantage condition 丢失了多少 magnitude information？
2. value model error 会怎样系统性污染 filtered policy training？
3. human corrections 强制 positive 是否会引入 intervention bias？
4. RECAP gain 中有多少来自更多 data，而非 advantage conditioning？
5. 如果 policy 量化/压缩，value calibration 与 exploration behavior 会如何变化？

### 组会可讲的 3 个重点

1. **身份先讲清**：ReadingList 的 π₀.₆ 对应 canonical paper π*₀.₆；星号代表 RECAP-trained variant。
2. **RECAP 三步**：rollout/correction → distributional value → positive-advantage-conditioned policy。
3. **结果与代价**：若干任务 throughput 达 2×、strict laundry 达 97%，但依然需要人工 label/reset，且证据来自 proprietary preprint setting。

---

## 5. ω-0

### Canonical metadata

- **Title:** *ω-0: A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation*
- **Authors:** Zhe Li, Zhenzhe Zhang, Yangyang Wei, Wenjie Zhang, Xichen Yuan, Peiyuan Zhi, Gen Li, Xinying Guo, Fengjie Gao, Jianfei Yang, Shanghang Zhang.
- **Year/version:** 2026; arXiv `2608.06375`, `v1` 2026-08-06, latest checked `v2` 2026-08-09.
- **Venue/status:** arXiv preprint；截至检索日未确认 peer-reviewed venue。
- **Identifiers / official sources:** [arXiv:2608.06375](https://arxiv.org/abs/2608.06375) · [arXiv DOI](https://doi.org/10.48550/arXiv.2608.06375)
- **Verification label:** **verified metadata/full text; preprint**.

### Background

多数 VLA/WAM 重点是 arm-centric tabletop manipulation；humanoid loco-manipulation 还要求 locomotion、torso、balance、arms 与 dexterous hands 并发协调。直接预测 future video 再从 video 取 action 容易放大 temporal inconsistency；只预测 robot action 又缺少 task-progress 与 scene-evolution supervision。ω-0 选择在 latent space 联合预测 future visual state 与 controller-compatible whole-body action。

### Problem

给定 language、当前 visual observation 和 proprioception，让 Unitree G1 在 household long-horizon task 中边移动边操作，并通过 latent future prediction 学到 action-relevant dynamics，而不是在 inference 时显式生成 pixel video。

### Method

**Components.** `Qwen3-VL-2B-Instruct` whole-body action VLM，`V-JEPA2.1` current-image encoder，`T5` text encoder，frozen `Wan` encoder 提供 future-video latent targets，future-aware video/action queries，state encoder，action `DiT`，以及 `SONIC` low-level whole-body controller。（Sec. 3, PDF pp. 4–8）

**Three-stage training.** (1) 用 whole-body `FAST` tokenizer 将 unified SMPL motion 离散化，训练 VLM 从 ego/exo observation、language、view token 预测 action tokens；(2) 把 public human motion 经 SONIC simulation replay 转成 robot state/action latents，联合学习 future-video latent prediction 与 action-latent denoising；(3) 在 real-world ω-HOME data 上 fine-tune，训练时加入 `RTC` clean prefix 约束相邻 chunks 连续。V-JEPA、Wan、VLM 在后两阶段冻结，优化 query/state/fusion/action-DiT modules。

**Objectives / inference.** action DiT 做 clean latent `x0` prediction，同时以 weighted video latent loss训练 query；inference 用少步 `DDIM` 采样，只生成 action latent，不生成 video。单次约 0.14 s（>7 Hz）；chunk `H=25`，只执行前 `K=8` 后 replan，并把上个 chunk 的 prefix 用于 RTC consistency。（Appendix B, PDF p. 23）

**Data.** 新建 `ω-HOME`：40.3 hours、4,827 episodes、24 tasks、30 Hz，含 synchronized language、egocentric RGB、exocentric RGB-D、proprioception、whole-body motion 与 SONIC-compatible action latents。下游 11 tasks 共 2,220 trajectories、约 200 demonstrations/task；用于 pretraining 的 ω-HOME pool 排除这 11 tasks 以避免 task leakage。（Sec. 4–5, PDF pp. 8–10）

### Key Innovation

`source claim`：future video prediction 是 training-time latent auxiliary objective，为 action query 注入 task progress/scene evolution；execution 则直接输出 SONIC-compatible whole-body latent，避免 test-time pixel generation 的误差和延迟。

`inference`：这是一种“world modeling as representation learning”，而不是可独立 rollout 的 explicit world simulator；组会不应把它与 planning-by-video-generation 混为一谈。

### Main Results

| Evidence type | Benchmark / metric | Result | Locator |
|---|---|---|---|
| `source claim; exact` | 11 real-world household loco-manipulation tasks, 10 trials/task/method | ω-0 Ego: **79.1% SR / 35.8 of 41 score / 88.7% progress**；ω-0 Omni: **81.8 / 36.7 / 90.3**。最佳 baseline ψ-0 的 SR 为 **44.5%**，DiT4DiT 的 progress 为 **61.0%** | Table 2, PDF p. 13 |
| `source claim; exact ablation` | Remove future video query | no video query: **64.5% SR / 30.6 / 77.9%** vs full Ego **79.1 / 35.8 / 88.7**；其余 ablations：no state 60.9% SR、no VLM prefix 66.4%、no RTC 71.8%、Wan current encoder 63.6% | Table 4, PDF p. 14 |

补充 generalization evidence：cross-object/no-video vs video 为 **66.7→83.3% SR**，cross-scene **15.0→79.5%**，human-to-humanoid transfer **20.0→60.0%**（Table 5, PDF p. 18）。这是同一论文的 ablation，不能视作 external replication。

### Limitations

**Authors' disclosure.** 正文没有单列 limitations section；作者在实验讨论中承认 egocentric observation 对全身 displacement/stepping 可见性有限，因此 Omni 版本在 5 个 locomotion-heavy tasks 改用 exocentric view（Sec. 6.4, PDF pp. 13–14）。

**Critical reading（inference / critique）**

- 结果来自单一 Unitree G1 platform、自建 dataset/task suite，且每 task/method 10 trials；尚不能说明跨 humanoid/controller transfer。
- Omni 对部分任务使用 room-mounted exocentric camera，信息条件强于纯 onboard deployment；与 Ego 或其他只用 onboard view 的比较需要明确 sensing assumption。

### Why It Matters

ω-0 把 VLA、latent world model 与 whole-body humanoid controller 接到同一 training pipeline，尤其适合讨论“world prediction 的价值究竟在 test-time imagination，还是 training-time representation”。它也提供一个非常新的 humanoid WAM data/evaluation reference，但结论目前仍属 preprint/self-benchmark evidence。

### Prerequisites

- V-JEPA / video latent representation；Diffusion Transformer 与 DDIM。
- FAST tokenization、SMPL/SMPL-X、human-motion retargeting。
- whole-body control、SONIC、receding-horizon control 与 RTC。

### Reading Questions

1. video-query ablation 的 +14.6 SR points 是否足以证明 learned representation 真包含 causal dynamics？
2. frozen Wan target 是否会把 video-generation bias 带入 action representation？
3. simulation replay 丢弃 SONIC 无法追踪的 motion，会造成什么 selection bias？
4. Omni 使用 exocentric view 的 gain 与 world-model gain 如何解耦？
5. 若换 low-level controller，action latent 是否仍 transferable？

### 组会可讲的 3 个重点

1. **不是先生成 video 再执行**：future video latent 只在 training 提供 auxiliary supervision，inference 直接 denoise action latent。
2. **Whole-body interface**：human motion → SONIC replay → controller-compatible latent，把 locomotion 与 manipulation 统一。
3. **强结果但外部有效性有限**：Omni 81.8% SR vs baseline ≤44.5%，但来自单一平台、自建 11 tasks、10 trials/task。

---

## 6. BitVLA

### Canonical metadata

- **Title:** *BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation*
- **Authors:** Hongyu Wang, Chuyan Xiong, Ruiping Wang, Xilin Chen.
- **Year/version:** 2025–2026; arXiv `2506.07530`, `v1` 2025-06-09, latest checked `v2` 2026-03-01.
- **Venue/status:** arXiv preprint；arXiv comments 明示 **“Work in progress”**。
- **Identifiers / official sources:** [arXiv:2506.07530](https://arxiv.org/abs/2506.07530) · [arXiv DOI](https://doi.org/10.48550/arXiv.2506.07530) · [official code](https://github.com/ustcwhy/BitVLA)
- **Verification label:** **verified metadata/full text; preprint/work in progress**.

### Background

OpenVLA/OFT 级 VLA 的 memory 与 latency 仍不适合 edge robot。常见 post-training INT8/INT4 会在训练完成后才施加 quantization constraint，容易发生 accuracy drop。BitVLA 从 native ternary LLM 开始，并在 vision encoder 上做 quantization-aware distillation，主张把 efficiency 当作 training-time co-design，而非 deployment 后处理。

### Problem

训练一个几乎全 native low-bit 的 VLA，在 LIBERO 与 real-world manipulation 中接近 full-precision OpenVLA-OFT，同时显著降低 model memory 和 end-to-end latency。

### Method

**Architecture.** `BitNet b1.58 2B4T` 作为 ternary LLM；`SigLIP-L` vision encoder；lightweight connector、proprioceptive state token projector 与 continuous action head。weights 量化到 `{−1,0,1}`（平均 1.58 bits），activations 为 symmetric INT8；使用 `BitBLAS` custom kernel。（Sec. III-A, PDF pp. 3–4）

**Three-stage pipeline.** (1) LLaVA-style multimodal training：先对 558k image-caption data 训练 connector，再用 10M MammoTH-VL subset 做 instruction tuning；(2) `Quantize-then-Distill`：frozen BF16 teacher 指导 1.58-bit vision student，只有 student encoder 更新，loss 为 task CE 加 intermediate-feature alignment `Laux`；(3) 在约 1M Open X-Embodiment samples 上 robotics pretraining，再做 downstream SFT。（Fig. 2/Sec. III-B, PDF pp. 3–5）

**Objective / inference.** robot pretraining 先用 256-bin autoregressive next-action objective；downstream 采用 OpenVLA-OFT 式 parallel action-chunk decoding 与 `L1` trajectory regression。LIBERO 设 chunk `K=8`；efficiency test 统一 `K=25`，一次 forward 预测 chunk，保留 causal attention mask。

**Compute.** VLM curriculum 约 7 days on 8×H800；robotics pretraining 200k steps、batch 2048，约 14 days on 16×H800。（Sec. IV-A, PDF p. 5）

### Key Innovation

`source claim`：首个 fully native 1-bit/ternary VLA；`Quantize-then-Distill` 将 vision encoder 也压到 W1.58A8，同时以 teacher feature alignment 保留 multimodal competence。

`inference`：“1-bit”不等于所有运算都是 1-bit。weights 是 ternary，activations 是 INT8，scaling/dequantization 与若干 module 仍有高精度操作；汇报应说 **W1.58A8 native low-bit backbone**，而不是“整机只做 binary arithmetic”。

### Main Results

| Evidence type | Benchmark / metric | Result | Locator |
|---|---|---|---|
| `source claim; exact` | LIBERO success / memory | BitVLA (3.0B, **1.4 GB**) Spatial **96.6**, Object **99.0**, Goal **95.4**, Long **92.8**, avg **96.0%**；OpenVLA-OFT (7.7B, **15.4 GB**) avg **97.1%**。即 11.0× smaller memory、1.1 points lower average | Table I, PDF p. 5 |
| `source claim; exact` | A100 inference, 100 queries, 3×224² images + 14-D state, `K=25` | BitVLA **73 ms / 341.1 Hz**；OpenVLA-OFT+ **321 ms / 77.9 Hz**，约 **4.4×** lower latency/higher reported throughput；π₀ **86 ms / 291.6 Hz** | Fig. 6 + text, PDF p. 7 |

补充 quantization evidence：BitVLA 1.4 GB / 96.0% avg vs INT4 OpenVLA-OFT 4.7 GB / 96.9%，vs INT4 OpenVLA 4.4 GB / 72.7%（Table II, PDF p. 5）。Vision encoder 的 BF16→W1.58A8 memory 为 0.8→0.1 GB，five-benchmark avg 53.0→51.5%；去掉 representation alignment 后为 42.4%（Table III, PDF p. 8）。

### Limitations

**Authors' stated limitations（Sec. VI, PDF p. 8）**

- quantization-aware training 产生与 full-precision 不同的 distribution，因此 BitVLA 不是任意 pretrained VLA 的 drop-in 1-bit conversion recipe；且 robotics pretraining 只有约 1M samples，作者认为更强 generalization 需要更大规模。

**Critical reading（inference / critique）**

- `341.1 Hz throughput` 与 73 ms single-query latency 不是同一概念（前者含 parallel/chunk action accounting）；不能把 throughput 当 control-loop frequency。baseline numbers 又来自 OpenVLA-OFT paper，而非全部同代码重测。
- real-world evidence 只有 Franka + 3 base tasks/OOD variants，图中样本规模与 confidence interval 不充分；arXiv 仍标 “Work in progress”，不可把 edge deployment claim 当成熟验证。

### Why It Matters

BitVLA 直接对齐导师可能关心的 efficient AI：memory、bit-width、kernel 与 robot latency 被纳入 model design。它也提供一个好反例：post-training quantization 的结论不能直接外推到 native low-bit pretraining。

### Prerequisites

- ternary/1.58-bit quantization、STE、absmean weight quantizer、per-token absmax activation quantizer。
- knowledge distillation、representation alignment、BitNet/BitBLAS。
- OpenVLA-OFT action chunking、LIBERO evaluation。

### Reading Questions

1. 96.0% LIBERO 中多少增益来自 OpenVLA-OFT head/chunking，而非 native low-bit backbone？
2. 为什么 vision encoder 需要 teacher alignment，而 1.58-bit LLM 可直接 pretrained？
3. memory table 是否包含 KV cache、activations 与 runtime workspace，还是只含 weights/model allocation？
4. A100 custom-kernel speedup 能否迁移到真正 edge hardware？
5. low-bit policy 的 calibration/noise 会不会在 long-horizon closed loop 中累计？

### 组会可讲的 3 个重点

1. **Native low-bit 而非 PTQ**：ternary BitNet + W1.58A8 vision encoder，distillation 在训练期吸收 quantization constraint。
2. **核心数字**：1.4 GB、LIBERO 96.0%，相对 OpenVLA-OFT 约 11× smaller、只低 1.1 points；A100 latency 73 ms vs 321 ms。
3. **不要过度宣传**：throughput≠control frequency，real-world task 很少，且论文仍明确为 work in progress。

---

## 7. Search log、version merging 与排除决定

### Search log（2026-08-22）

只使用 primary/official sources；检索式包括：

- `site:arxiv.org OpenVLA 2406.09246`
- `OpenVLA PMLR CoRL proceedings`
- `site:pi.website pi0 pi0.5 pi0.6`
- `site:arxiv.org "π0: A Vision-Language-Action Flow Model"`
- `site:arxiv.org "π0.5: a Vision-Language-Action Model with Open-World Generalization"`
- `site:arxiv.org "π*0.6: a VLA That Learns From Experience"`
- `site:pi.website "π0.6"` 与 `site:pi.website "π*0.6"`
- exact-title query: `ω-0: A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation`
- exact-title query: `BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation`

### Version / dedup decisions

1. **OpenVLA:** arXiv 与 PMLR/CoRL 是同一 work，合并为一篇；citation 采用 peer-reviewed PMLR record。
2. **π₀:** arXiv technical report 与 RSS 2025 paper 合并为一篇；venue citation 采用 RSS record，method/result 读取 PI official/latest paper PDF。
3. **π₀.₅:** arXiv、PI official PDF/blog 与 CoRL 2025/PMLR 是同一 work，合并为一篇。
4. **π₀.₆:** 未找到同名独立 paper；映射至 official `π*₀.₆` paper，保留 identity caveat。`π₀.₆-MEM` 是后续不同 paper，从 core count 排除、作为 companion 单列。
5. **ω-0 / BitVLA:** 只计最新 arXiv version，不把 v1/v2 当多篇；截至检索日没有 verified peer-reviewed venue。
6. **Excluded sources:** 不用搜索聚合站、新闻转述、GitHub issue、社交媒体或二手 benchmark table 作为学术 claim 证据；official project/code 只用于 release/status 辅证。

### Full-text audit trail

所有六篇的 original PDF 均已本地结构检查：PDF header 正常、非加密、page count 可解析、全文可抽取；`OpenVLA 37 pp / π₀ 17 pp / π₀.₅ 19 pp / π*₀.₆ 18 pp / ω-0 39 pp / BitVLA 15 pp`，preflight 均为 **PASS**。临时 PDF/抽取文本保存在 workspace 外的 session temp directory，未作为最终 deliverable，也未修改其他 workspace 文件。

### Companion update — MEM（2026-08-25）

- **Canonical identity:** *MEM: Multi-Scale Embodied Memory for Vision Language Action Models*, arXiv:2603.03596 v2 / Physical Intelligence technical report.
- **Primary sources:** [official project page](https://www.pi.website/research/memory), [official project PDF](https://www.pi.website/download/Mem.pdf), [arXiv record](https://arxiv.org/abs/2603.03596).
- **Local provenance:** `papers/04a-mem/paper.pdf` is the 15-page Physical Intelligence project PDF, not the binary arXiv artifact; SHA-256 `63413aa32bb2b070ae0a0cbc5d5a2eb77686308725e48f15bcaa338bd3d644aa`.
- **Release boundary:** no peer-reviewed venue, official code, weights, or dataset release was verified as of the snapshot date.
- **Identity decision:** `π₀.₆-MEM` and `π*₀.₆` share base `π₀.₆` but change different axes—memory/context versus RECAP-based RL—so they remain separate entries.

### Companion update — FlashVLA（2026-08-31）

- **Canonical identity:** *FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference*, arXiv:2608.27384 v1, submitted 2026-08-27.
- **Primary sources:** [arXiv record](https://arxiv.org/abs/2608.27384), [arXiv v1 PDF](https://arxiv.org/pdf/2608.27384v1), and [official repository](https://github.com/z-lab/flashvla).
- **Exact search:** `FlashVLA Streaming Action Decoding Fast Asynchronous VLA Inference`, 2025-2026, across arXiv, DBLP, OpenAlex, OpenReview, Semantic Scholar, and Crossref. The search returned 13 unique records after two duplicate merges; full-text screening retained only the user-specified arXiv ID. OpenReview was unavailable because its client was not installed, and Semantic Scholar ended with HTTP 429 after bounded retries.
- **Local provenance:** `papers/03a-flashvla/paper-arxiv-v1.pdf` is the 17-page arXiv v1 artifact; SHA-256 `74d35658a10c7a57fdf06a59e4bc021bb8de1bb4ea9b882f38f45e94b3abfa66`; PDF header, page count, encryption state, first-page title, full-text extraction, and selected-page rendering were checked.
- **Publication boundary:** no peer-reviewed venue record or independent reproduction was verified. Released code/checkpoints improve reproducibility but do not replace peer review or matched reruns.
- **Identity decision:** arXiv:2505.21200 is a separate paper, *Think Twice, Act Once*, whose action-reuse/token-pruning method is also called `FlashVLA`. The two work clusters must not be merged.
- **Repository drift:** official repository `main` at commit `5227b039ebd4f6b5cad0c27d2d6098932f0f7ed3` (2026-08-30) reports different LingBot-VLA success/time-per-step values from arXiv v1 Table 5. Paper claims remain frozen to the local PDF; repository values are a separate post-submission evidence surface.

## 8. Cross-paper synthesis（供总 README 使用，属 inference）

| Paper | Action representation | 主要扩展轴 | 最应该警惕的 evidence issue |
|---|---|---|---|
| OpenVLA | autoregressive discrete tokens | open cross-embodiment VLA + adaptation | latency/control 与 quantization confound |
| π₀ | continuous flow action chunks | dexterity + heterogeneous robots | proprietary data；多为 plot-read progress |
| π₀.₅ | high-level text + flow chunks + FAST | open-world/home generalization | 10-trial custom progress，公开复现困难 |
| FlashVLA | staggered streaming flow chunks | fast asynchronous action decoding + chunk-wise causal continuity | very recent preprint；speedup denominators、8-H200 fine-tuning、limited real-world trials |
| π*₀.₆ | advantage-conditioned π₀.₆ | real-world RL/post-training | human-in-loop、preprint、自建 task |
| ω-0 | diffusion whole-body latent + video-latent auxiliary | humanoid loco-manipulation/world representation | single platform、自建 dataset、exocentric sensing |
| BitVLA | W1.58A8 VLA + parallel continuous chunks | native low-bit efficiency | throughput definition、少量 real-world tests |

一句主线：**OpenVLA 把 generalist VLA 开源；π₀ 把 action 从 discrete token 改为 continuous flow；π₀.₅ 加入 open-world hierarchy；FlashVLA 把 isolated flow decoding 改为 streaming asynchronous execution；π*₀.₆ 让 policy 从 deployment experience 学习；ω-0 把 latent future modeling 接到 whole-body humanoid control；BitVLA 则把 memory/latency constraint 放回 training design。**
