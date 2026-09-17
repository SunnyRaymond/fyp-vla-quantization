# 现有 Ideas 的 Efficient VLA / WM / WAM Taxonomy 审查

日期：2026-09-17  
依据：Yu et al., *A Survey on Efficient Vision-Language-Action Models*, arXiv:2510.24795v2；本地 `idea/`、`experiment/idea-validation/` 的当前方案、最终结果与 registry。  
目的：回答现有 ideas 从哪个效率细分环节产生，并筛选在 2–4×A100 预算下尚可探索的其他细分方向。本文不生成新的 novelty claim，也不把小 screen 的结果提升为完整方法结论。

## 1. 结论先行

现有 portfolio **高度集中在一个细分格子**：

> `Efficient Model Design → Model Compression → Model Quantization → frozen-model PTQ calibration / allocation / rounding / recovery`

RankCal、OTC-PTQ、CEM-Update、FRT、PRR、TR-PVQ，以及 14 个 bounded empirical screens，虽然用了 planner ranking、flow geometry、uncertainty、rounding coupling、temporal persistence 等不同语言，但大部分仍是在回答同一类问题：**固定或近似固定模型结构后，如何选择量化位置、量化参数、校准信号或量化误差耦合方式。**

因此，当前 ideas 并没有覆盖 survey 的整个 Efficient VLA 设计空间。明显空缺包括：

1. `Layer Pruning / Early Exit`；
2. `Token Optimization / Caching / Shared-prefix Reuse`；
3. `Efficient Action Decoding / Adaptive Compute`；
4. `Efficient Action Representation`；
5. `Data-efficient Calibration / Failure-focused Data Selection`；
6. 少量可承受的 `PEFT / Post-training`。

## 2. Taxonomy 如何从 VLA 迁移到 WM / WAM

这篇 survey 的三支柱可以迁移，但要把对象改写成“成本发生在哪里”，而不是照搬 VLA 模块名。

| Survey 支柱 | VLA 中的典型对象 | WM / WAM 中的对应对象 |
|---|---|---|
| Efficient Model Design | vision encoder、LLM/VLM backbone、action decoder | state/vision encoder、latent dynamics/predictor、reward/value/policy head、video/action expert、planner-facing decoder |
| Efficient Training | VLM-to-action pre-training、SFT、RL、action representation | dynamics/action pre-training、latent/action representation、distillation、PEFT、offline/online policy refinement |
| Efficient Data Collection | teleoperation、simulation、internet video、augmentation | environment rollouts、planner candidate traces、failure/contact transitions、world-model-generated data、counterfactual/perturbed trajectories |

适用原则：

- `weight/activation quantization` 属于 **Model Compression**；即使 calibration 使用优化器，也不自动变成 Efficient Training。
- `action discretization/tokenization` 属于 **Efficient Action Representation**；它不是 model quantization。
- `latent VQ` 要看量化对象：量化 model weights 是 Model Compression；量化 state/action representation 是 Efficient Action Representation。
- `cache/prune/reuse` 主要属于 Token Optimization 或 Efficient Architecture；只有学习 cache policy 的训练过程才同时涉及 Efficient Training。
- WM/WAM 的 planner、rollout horizon、denoising schedule 与 repeated candidate evaluation 是 survey 没有细分展开、但必须额外加入的 control/planning 轴。

## 3. 主线 Ideas 的逐项归类

| Idea | 一级位置 | 更细的机制位置 | 当前证据状态 |
|---|---|---|---|
| RankCal | Model Design → Model Compression → Quantization | planner-aware mixed-precision allocation；用 candidate-order influence 选择 W4/W8 site | 冻结 recipe no-go；不能继续只换 ranking score |
| OTC-PTQ | Model Design → Model Compression → Quantization | WAM single-site sensitivity / calibration；用 paired one-step action/observation consequence 排序 site | 当前配方 no-go；observation 主导且未区别于 Local MSE |
| CEM-Update PTQ | Model Design → Model Compression → Quantization | planner-update-aware mixed-precision allocation；保持 CEM elite mean/std update | Stage B mechanism no-go；不进入闭环 Stage C |
| Frozen-Residual Transport PTQ | Model Design → Model Compression → Quantization | quantizer calibration objective；匹配量化 residual 的 finite-difference transport | mechanism no-go；fresh residual 上未优于 random same-norm |
| Paired-Rollout Recovery | Model Design → Model Compression → Quantization；次级交叉 Efficient Post-training | fixed-W4 recovery target；quantizer-only / quantizer+LoRA | 当前 recipe mechanism no-go；平均改善不稳定跨 episode/seed |
| TR-PVQ / Goal-Anchored Trajectory VQ | Model Design → Model Compression → Quantization | **weight VQ** codebook calibration；goal/trajectory relational objective | conditional proposal；未实验、novelty 未认证 |

这里最容易误分类的是 PRR 与 TR-PVQ：PRR 使用 LoRA 不表示主问题变成 PEFT；它的主目标仍是 W4 recovery。TR-PVQ 使用 VQ 不表示 action tokenization；它量化的是 weights/codebook。

## 4. 14 个已实测 Bounded Screens 的归类

全部结果以 [SCREEN_REGISTRY](../experiment/idea-validation/v100-new-angles/SCREEN_REGISTRY.json) 和各自 RESULT 为准。

### 4.1 纯 Quantizer / Rounding Mechanics

| Screen | 细分位置 | 状态 |
|---|---|---|
| Antithetic Rounding Pairs | Quantization → correlated/stochastic rounding coupling | mechanism signal；two-W4 相对 W8 practical no-go |
| Semantic Input Scales | Quantization → activation granularity / interface scaling | mechanism no-go |
| Existing-Q Stratified Rounding | Quantization → ensemble/member rounding coupling | inconclusive binding |
| Value-head Gauge | Quantization → function-preserving parameter preconditioning | implementation inconclusive |
| Broadcast Activation Coupling | Quantization → activation rounding correlation across broadcast consumers | scope-limited preliminary go；generic novelty no-go |
| Reference-branch Rounding Coupling | Quantization → branch-wise correlated rounding | mechanism no-go |

### 4.2 Quantization × Action Decoding / Temporal Dynamics

| Screen | 细分位置 | 状态 |
|---|---|---|
| Flow Geometry under PTQ | Quantization diagnostic × flow action decoder geometry | backbone sub-result preliminary go；overall provenance inconclusive；expert no-go |
| Padded-coordinate Feedback | Quantization × analytic flow-coordinate update | method no-go |
| Conditional Action Distribution | Quantization evaluation × conditional action marginal | statistical inconclusive |
| Denoising-call Rounding Persistence | Quantization × iterative decoder temporal persistence | scope-limited preliminary go；generic method novelty no-go |
| Euler Jacobian | Quantization × numerical solver / action-decoder geometry | mechanism no-go |

### 4.3 Quantization × Planner / Dynamics Objective

| Screen | 细分位置 | 状态 |
|---|---|---|
| Action-gradient Geometry | Quantization diagnostic × planner action sensitivity | mechanism no-go |
| Recorded-future Error Cancellation | Quantization calibration target × dynamics prediction | mechanism no-go |
| Policy-prior Proposal Support | Quantization locus × actor proposal / FP scorer | mechanism no-go |

### 4.4 审查判断

14 个 screens 并不是 14 个彼此独立的宏观研究领域。它们仍主要位于 Quantization 这一格，区别在于：

- **what is quantized**：weight、activation、value head、actor、branch；
- **how error is coupled**：independent、antithetic、stratified、broadcast、temporal persistence；
- **what signal judges it**：MSE、ranking、gradient、flow geometry、distribution、planner support；
- **where it is measured**：single step、denoising call、two-step dynamics、planner candidates。

这解释了为什么继续“换一个更 planner-aware loss”容易重复旧失败模式，而不是进入 survey 的新细分领域。

## 5. Prior / Structure / Resource Gate 候选的完整归类

以下候选没有全部进入经验实验；这里按 idea 本身归类，不把 gate 状态改写成实证 no-go。

### 5.1 Model Compression → Quantization

- Batch Scale Coupling；TD-MPC2 Ensemble Uncertainty；MOPO Dynamics Uncertainty；Instruction Contrast；Gripper Decision Margin。
- Q-ensemble Common-U；Simplex Latent Quantization；Bellman Consistency；Residual Branch Interaction；Hidden-basis Permutation Invariance。
- Q Decoder-tail Rounding；Camera Redundancy；Shared Goal/Current Quantization；Physical-state Readout；Null-input Columns。
- Success-conditioned Masking；Discrete-tokenizer WM 中涉及 weight/latent quantization 的部分。

这些 idea 虽然借用了 uncertainty、language grounding、Bellman error 或 camera geometry，本质仍是分析 **quantization changes which downstream quantity**。

### 5.2 Token Optimization / Caching

- Recency-Split Latent-History Cache。
- Prefix KV Reuse vs Suffix A8。
- Timestep Constant Folding。
- Action-token Reinjection 中的 repeated-token reuse 部分。

这是目前少数真正离开“换 quantization loss”的分支，但现有候选多在 source/identifiability/prior gate 停止，没有形成系统的 native latency/quality study。

### 5.3 Efficient Action Decoding / Action Representation

- OFT Continuous/Discrete Interface。
- Flow Step Refinement。
- Action-chunk Suffix Feedback。
- Action-token Support。
- Discrete-tokenizer WM 中量化 latent/action tokens 的部分。
- Padded-coordinate Feedback、Euler Jacobian 也与 action decoder 相交，但现有实验仍以 PTQ perturbation 为主。

### 5.4 Data / Training 交叉但尚未成为主线

- Temporal Residual Correction、Recorded-future、PRR 使用了 paired trajectories / recovery targets，但目标仍是 quantizer fitting。
- Success-conditioned Masking 使用 outcome labels，但尚未形成 data selection / augmentation 方法。
- 现有 portfolio 没有把 calibration trajectory 的 **数量、覆盖度、采集策略或信息密度** 当成主要自变量。

## 6. Portfolio Coverage Map

| Survey 细分领域 | 当前覆盖 | 评价 |
|---|---:|---|
| Efficient Attention | 低 | 几乎没有独立 idea |
| Transformer Alternatives / SSM | 无 | 需要重训练 backbone，不适合当前预算 |
| Efficient Action Decoding | 中低 | 有 flow/solver/temporal diagnostics，但多附着于 PTQ |
| Lightweight Components | 低 | 没有系统比较 small backbone/component replacement |
| Mixture-of-Experts | 无 | 训练与 routing 验证成本较高 |
| Hierarchical Systems | 低 | planner/policy 分层被当作评测接口，不是效率方法 |
| Layer Pruning / Early Exit | 无 | 当前最明显的可行空缺之一 |
| Model Quantization | **极高** | portfolio 主体，已有多个 no-go 与 bounded signals |
| Token Optimization / Caching | 低 | 有零散 gated idea，没有完整性能验证 |
| Data-efficient Pre-training | 无 | full pre-training 不适合当前预算；小规模 data selection 可做 |
| Efficient Action Representation | 低 | 有阅读与 interface probes，没有形成主要实验线 |
| Efficient Post-training / PEFT | 低到中 | PRR LoRA 是交叉尝试；没有独立 PEFT 研究线 |
| RL-based Post-training | 无 | 完整 RL campaign 不适合当前预算 |
| Human-in-the-loop Collection | 无 | 缺真实 robot collection infrastructure |
| Simulation Collection | 中 | 用于 evaluation/data source，但没有把 collection efficiency 当研究对象 |
| Cross-domain / Internet Data | 无 | 数据整理与 alignment 成本过高 |
| Self-exploration | 无 | online RL / safety / sample cost 不适合当前预算 |
| Data Augmentation | 低 | failure/contact perturbation 尚未成为主变量 |

## 7. 2–4×A100 下值得考虑的其他细分领域

预算解释：假设可同时使用 2–4 张 A100，但不是进行 foundation-model pre-training 的长期 GPU-month 预算。所有建议都从 1×A100 smoke 开始，总并发最多 4；GPU-hour 上限需在首个 smoke 后冻结。DINO-WM、Fast-WAM、OpenVLA-OFT 的现有资源证据见 [feasibility audit](FEASIBILITY_AUDIT_2026-09-12.zh.md)。

### A. Data-efficient PTQ Calibration（首选）

- **Taxonomy**：Efficient Training / Data Collection × Model Compression。
- **核心问题**：在相同 calibration episode 数与相同 quantizer 下，按 transition/contact/failure/planner-state coverage 选取轨迹，是否比 random 或 frame-level diversity 更稳定地保持 held-out closed-loop behavior？
- **为什么不是旧 idea**：不再发明新的 planner-aware loss；把主要自变量从 `loss/rounding` 改成 `which episodes are worth collecting or retaining`。
- **最小实验**：DINO-WM Wall；固定 all-W4 或一个标准 AdaRound-like recipe；比较 random、visual diversity、planner-state diversity、failure/near-boundary selection。episode 是独立单位，CAL budget 完全相等；DEV/TEST 使用新锁定 episodes。
- **资源判断**：高可行。可大量复用现有 trajectories/metadata，只有入选集合的 calibration 与闭环验证需要 GPU。建议先设 1×A100、6–10 allocated GPU-hour 的硬 cap，再决定是否并行重复。
- **主要风险**：active/calibration data selection 已有广泛 prior；novelty 必须落在 WM/WAM 的 closed-loop、planning-state coverage 与 equal-episode-cost 证据，而不是“挑更重要的数据”。

### B. Training-free Layer Pruning / Early Exit

- **Taxonomy**：Efficient Model Design → Model Compression → Layer Pruning。
- **核心问题**：WM/WAM 的 encoder/predictor/action expert 是否存在随 rollout horizon、state difficulty 或 planner iteration 改变的冗余层？
- **最小实验**：先对 DINO-WM 做静态 block drop / block bypass sensitivity；用 equal-FLOPs / equal-wall-clock 的 fixed patterns 建立 Pareto curve。只有静态证据成立后，才训练极小 router 或研究 state-conditioned early exit。
- **资源判断**：高可行。无需从头训练，1×A100 即可 screen；2–4 卡只用于独立 patterns/episodes 并行。建议 cap 6–12 GPU-hours。
- **主要风险**：EfficientVLA、MoLe-VLA、DeeR-VLA 等已覆盖 VLA layer pruning。单纯“把 pruning 用到 WM”不足以构成 novelty；必须证明 planner rollout / temporal horizon 带来不同的可识别结构。

### C. Shared-prefix / Temporal Feature Caching

- **Taxonomy**：Efficient Model Design → Token Optimization / Caching。
- **核心问题**：CEM candidates、receding-horizon replanning 或 iterative denoising 中，哪些 feature 在输入未改变时可严格复用；何时必须 invalidate？
- **最小实验**：优先 DINO-WM 的 shared observation/history encoder 与 CEM candidate batch；其次 Fast-WAM 的 static condition / denoising intermediates。先做 exact-cache identity 与 profiler，之后才加 approximate cache。
- **资源判断**：高可行、训练需求低；主要成本是实现与 native profiling。1×A100 足以建立 latency/memory/quality 曲线，建议 cap 4–8 GPU-hours。
- **主要风险**：VLA-Cache、EfficientVLA、WorldCache 等直接 prior 很强。贡献必须是 WM/WAM-specific reuse boundary 或 invalidation rule；必须报告真实 wall-clock、peak memory 与 task quality，不能只报 FLOPs。

### D. Adaptive Inference Budget / Efficient Action Decoding

- **Taxonomy**：Efficient Architectures → Efficient Action Decoding；可与 Hierarchical Systems 相交。
- **核心问题**：能否按 state difficulty 动态选择 CEM iterations、rollout horizon、denoising steps 或 action-chunk refresh frequency？
- **最小实验**：从 DINO-WM 的 CEM5 选择 `1/3/5 iterations` 或 short/full horizon，使用一个不训练的 uncertainty/disagreement gate；对比固定相同平均 compute 的 schedule。若转 Fast-WAM，先只研究 denoising-step budget，不同时改 quantization。
- **资源判断**：中高可行。1–2×A100 足够 pilot，建议 cap 8–16 GPU-hours；Fast-WAM 分支必须先过单卡 memory/time gate。
- **主要风险**：adaptive computation、early exit 与 diffusion step reduction prior 丰富。要避免把“少跑几步”包装成 novelty；必须用 matched-average-compute 和 closed-loop failure recovery 证明 WM/WAM 特有价值。

### E. Action Representation Rate–Distortion

- **Taxonomy**：Efficient Training → Efficient Action Representation。
- **核心问题**：DCT/BPE、scalar bins、small VQ codebook 或 latent action 在相同 bitrate 下，哪种 distortion 与 closed-loop consequence 对齐？
- **最小实验**：不训练完整 VLA。先对已有 LIBERO/action trajectories 训练小 tokenizer 或直接使用 FAST-like transform；固定 frozen policy/world model，以 reconstruction、first-action error、chunk smoothness、closed-loop replay 分层评价。
- **资源判断**：中等可行。tokenizer 本身可在 1×A100 训练；完整 policy re-training 不在首轮范围。建议 cap 12–24 GPU-hours。
- **主要风险**：FAST、ActionCodec、VQ-VLA 等 prior 密集。更适合作为严谨的 rate–distortion / evaluation study；若要做方法，必须先完成 novelty search。

### F. Failure-focused Simulation Augmentation

- **Taxonomy**：Efficient Data Collection → Simulation / Data Augmentation。
- **核心问题**：少量 near-contact、near-goal、recovery/failure perturbations 能否比等量普通 rollouts 更有效地支持 calibration、robustness evaluation 或轻量 adaptation？
- **最小实验**：只在已有 simulator 中围绕已记录失败状态生成局部 perturbations；先服务于 calibration/evaluation，不启动完整 RL。
- **资源判断**：中等。2–4×A100 可并行 rollout，但 simulator/debug 与数据治理成本可能超过 GPU 成本；建议 cap 12–24 GPU-hours，并限制为单 task。
- **主要风险**：若同时改变生成方法、quantizer 和 training loss，会无法归因。首轮只改变 data distribution。

### G. PEFT Recovery（低优先级）

- **Taxonomy**：Efficient Training → Efficient Post-training / SFT，与 Quantization 交叉。
- **判断**：算力上可做，但 PRR 的 LoRA 交叉已经显示“不稳定平均改善”；QAT/LoRA recovery prior 也很密集。除非先找到明确的 episode-conditioned failure mechanism，否则不建议把更多 2–4×A100 预算投入另一个 recovery loss。

## 8. 当前预算下不建议的领域

| 领域 | 不建议原因 |
|---|---|
| 从头训练 Transformer alternative / SSM VLA/WM | backbone training 与 benchmark 规模超过当前预算；缺少公平共同训练条件 |
| 新建 Mixture-of-Experts VLA/WAM | routing、load balance、training 与 deployment 都需较大 campaign |
| Foundation-scale pre-training | 2–4×A100 不足以复现 survey 中的大规模数据与训练设置 |
| Full online RL / self-exploration | environment throughput、stability 与 confirmatory samples 成本高 |
| Internet-scale human-video alignment | 数据清洗、retargeting 与 annotation 是主要瓶颈，不是几张 GPU 能解决 |
| Hardware-software co-design / native accelerator | 可做 kernel profiling，但不能在当前条件下验证 custom hardware claim |
| 继续为 RankCal/OTC/CEM/FRT/PRR 换 loss 或放宽 gate | 已有 no-go 的共同教训是 proxy 改善未稳定传递到 held-out behavior；需要换主要干预对象 |

## 9. 推荐优先级

### Priority 1 — Data-efficient Calibration

最贴近现有 FYP，又真正进入 survey 中尚未覆盖的 Training/Data 轴；能复用失败轨迹和量化基础设施，也直接回应当前 no-go 的 episode instability。

### Priority 2 — Layer Pruning / Early Exit 或 Shared-prefix Caching

如果目标是从“quantization idea”扩展为“efficient WM/WAM”，这两条最自然。Layer pruning 更像算法论文；caching 更像 systems / deployment paper。二者首轮不要合并。

### Priority 3 — Adaptive Inference Budget

最能利用 WM/WAM 的 planner/rollout/denoising 特性，但 prior 密集，必须先做窄 novelty audit。

### Priority 4 — Action Representation Rate–Distortion

适合建立长期新支线；与现有 model quantization 清晰正交，但若追求完整 VLA re-training，预算会迅速上升。

## 10. 建议的最经济执行顺序

1. 只读复用现有 results，建立统一 episode registry、failure/contact/planner-state metadata。
2. 对 Priority 1 做 CPU selection + 1×A100 tiny calibration/closed-loop gate。
3. 同时只做 Priority 2 的 source/interface audit，不立即写第二套大实验。
4. Priority 1 通过后，最多用 2–4 张 A100 并行独立 seeds / held-out episodes；不让同一 episode 的 frames 冒充样本量。
5. 若 Priority 1 no-go，转向 Layer Pruning 或 Caching；不要再从 DEV 派生第六个 quantization proxy。
6. 任何 native efficiency claim 都必须实测 wall-clock、peak memory、packed bytes 与 task quality；fake quant 结果只支持 numerical/behavioral claims。

## 11. Evidence Boundary

- Survey 的 taxonomy 是组织工具，不是完整或互斥的 ontology；一个 idea 可以跨两个细分领域，但应按**主要被改变的对象**归类。
- 本报告中的可行性来自本地已有运行证据与 conservative resource caps，不是完成时间保证。
- “当前空缺”只指本地 portfolio，不等于全球 literature gap。
- 将 VLA 方法迁移到 WM/WAM 本身不构成 novelty；必须证明 WM/WAM 的 rollout、planning、temporal dynamics 或 decoder structure 引入新的、可识别的机制。
- 推荐方向均需在实施前做 primary-source novelty audit；本文没有替代该步骤。

## 12. 关键入口

- [Efficient VLA survey 导读](../papers/vla/105-efficient-vla-survey/README.md)
- [历史 ideas 总览](IDEAS_OVERVIEW.zh.md)
- [14 项 empirical screen registry](../experiment/idea-validation/v100-new-angles/SCREEN_REGISTRY.json)
- [新切面总表](../experiment/idea-validation/v100-new-angles/INDEX.zh.md)
- [2–4×A100 可行性审查](FEASIBILITY_AUDIT_2026-09-12.zh.md)
- [Quantization no-go 审查](v100-new-angles/PRIOR_NO_GO_AUDIT.zh.md)

## 13. 外部直接先例

- [EfficientVLA](https://arxiv.org/abs/2506.10100)：training-free layer pruning、visual-token selection、diffusion-feature caching。
- [VLA-Cache](https://arxiv.org/abs/2502.02175)：相邻 observation 的 adaptive visual-token caching。
- [FAST](https://arxiv.org/abs/2501.09747)：DCT/BPE-based action tokenization；属于 action representation，不是 low-bit model quantization。

