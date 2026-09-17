# World Model / World Action Model × Quantization：两个研究 idea

> 2026-09-17：现有主线、14 个 bounded screens 与 prior-only 候选的 Efficient VLA / WM / WAM taxonomy 映射，以及 2–4×A100 下的可行空缺，见 [taxonomy 审查](EFFICIENT_TAXONOMY_AUDIT_2026-09-17.zh.md)。

2026-09-08 · ResearchStudio-Idea / IdeaSpark · 当前状态：两条 pipeline 均已到达 DONE；六份阅读卡已生成。

以下是待实验检验的研究假设。pipeline 的文献与模型审查不构成新颖性认证，也没有证明机器人成功率、显存或速度收益。本次未执行 GPU、SSH 或机器人实验；数值检查仅使用 CPU toy examples。

## 1. World Model：RankCal——用候选动作排序校准量化

**问题。** World model 的单层数值误差很小，经过多步 imagined rollout 后，仍可能改变 planner 对候选动作序列的排序。对 planner 来说，选错动作往往比平均 latent MSE 稍大更重要。

**核心机制。** 固定一个 FP16 action-conditioned latent world model、起始 latent state 和 planner 生成的候选动作集合。每次只把一个 weight/activation site 量化，其他 site 保持 FP16，测量候选序列相对 FP16 的排序改变。将跨状态、horizon 的排序影响作为 precision allocation 的信号，并用实际测得的系统级 PeakBytes 约束配置。最后在联合量化配置和分离的测试 episodes 上评估。

**真正需要证明的假设。** 相比根据局部重建误差分配 precision，这个排序影响信号能在相同实际资源条件下更好地保住 planner 的动作选择，并迁移到闭环 return/success。排序改善本身不是最终答案。

**与已有工作的距离。** [QuantWM](https://arxiv.org/abs/2602.02110v1) 已经研究低比特下 planning objective 与 task success 的失配；[Where Bits Matter](https://arxiv.org/abs/2602.11882v1) 已研究 mixed-bit 与 planner budget。[VAML](https://proceedings.mlr.press/v54/farahmand17a.html) 等早期工作也已有 decision-aware model-loss 原则。RankCal 的候选差异是具体的 **site-only 数值量化干预 → candidate-order influence → 实测内存约束下的 allocation**，不是“第一次考虑决策”。

**怎样推翻它。** 如果排序更接近 FP16，但闭环表现没有改善；或者打乱影响分数与 site 的对应关系、保持 bit histogram 和实际 bytes 后收益仍在，核心解释就不成立。双 site interaction diagnostic 还要检查单点影响相加能否代表联合配置；无法构造精确匹配时应明确记录 unavailable。

**起点与主要限制。** pipeline 推断的起点是 DreamerV3/TD-MPC2 式模型与 DMC visual control，尚需锁定具体可用 checkpoint。单点 FP16-background 的影响不等于联合量化的边际收益。完整 nuisance grid 可达到 324 个组合，初版 12 GPU-days 只是未实测估计，不应直接作为开跑预算。数值 fake quant 也不等于 A100 上已有真实 W4A4 kernel。

[中文完整卡](<D:/Downloads/Final Year Project/idea/world-model-quantization/phase4/idea.std.zh.md>) · [English](<D:/Downloads/Final Year Project/idea/world-model-quantization/phase4/idea.std.en.md>) · [Reviewer version](<D:/Downloads/Final Year Project/idea/world-model-quantization/phase4/idea.detail.en.md>)

## 2. World Action Model：OTC-PTQ——用配对的一步后果校准量化

**问题。** 在联合预测 future video 与 action 的 WAM 中，某个 site 的局部量化误差不一定能反映它对动作的影响，尤其是接触发生前后。

**核心机制。** 从同一个可恢复的 simulator/controller/history/RNG/cache 状态分别运行 FP16 和单 site 量化分支。比较两边的完整 action chunk，再分别执行第一个 action，比较下一观测。用这两类经过归一化的差异构造平均分数 Cᵣ，并优先保护分数较高的 sites。校准时其余 sites 保持 FP16；最终测试联合 low-bit 配置。

**真正需要证明的假设。** Cᵣ 能比局部 tensor discrepancy 更好地识别 contact-transition sensitivity，而且这种区别能迁移到 held-out manipulation outcomes。Cᵣ 是 **one-control-transition proxy**，不是 task loss、完整 action chunk 的物理后果或 full-horizon marginal gain。

**与已有工作的距离。** [QuantWAMs](https://arxiv.org/abs/2607.28405v1) 已有 joint video/action saliency 与 closed-loop schedule auditing；[SQIL](https://arxiv.org/abs/2505.15304) 已保护 mission-critical states。这里的候选差异是 **单个 numerical W/A intervention、外部 simulator 的配对恢复、完整动作输出与下一观测组成的一步 proxy**。宽泛的 task-aware/critical-state 原则不是新贡献。

**怎样推翻它。** 如果 Cᵣ 排序没有比 local-score allocation 更好地预测 held-out contact-sensitive failures；或者置换 Cᵣ 与 site 的对应关系后优势不消失，这个机制就缺乏支持。必须保持相同评估设置，并区分 site-count matching 与真实 bytes/latency matching。

**起点与主要限制。** 候选载体为 Fast-WAM Optional IDM 的 `idm` 路径与 LIBERO/RoboTwin 2.0，需要锁定 checkpoint、split 和可执行的完整状态恢复。一步 pixel/action discrepancy 可能与真实任务重要性不一致；单点影响还可能在联合量化时失效。保护相同数量的 sites 不保证相同内存或运行时间。原文中的 1.2 GPU-day、80GB-class equivalent 未经测量，不能直接换算为 A100 40GB 的执行时长。

[中文完整卡](<D:/Downloads/Final Year Project/idea/world-action-model-quantization/phase4/idea.std.zh.md>) · [English](<D:/Downloads/Final Year Project/idea/world-action-model-quantization/phase4/idea.std.en.md>) · [Reviewer version](<D:/Downloads/Final Year Project/idea/world-action-model-quantization/phase4/idea.detail.en.md>)

## 如何理解这两个 idea 的关系

| | RankCal | OTC-PTQ |
|---|---|---|
| 校准时观察什么 | planner 内部的候选动作序列排序 | 完整 action chunk 与执行一步后的外部观测 |
| 关键结构 | 多步 latent rollout 与 planning | future/action 耦合、配对状态恢复 |
| 主要依赖 | 可访问的 planner、候选池、latents | 能可靠 fork/replay 的 simulator 与 controller |
| 共同风险 | 单点分数不一定代表联合量化效果 | 单点分数不一定代表联合量化效果 |

它们属于相关的方法家族，而非两个完全无关的原理。研究价值最终取决于各自能否发现可复现的 failure mode，并在公平对照下得到下游证据。

## 证据与运行记录

- [直接与跨领域前作核对](<D:/Downloads/Final Year Project/idea/DIRECT_PRIOR_REVIEW.md>)
- [初版候选的独立检查](<D:/Downloads/Final Year Project/idea/INITIAL_CANDIDATE_REVIEW.md>)
- [RankCal 修订范围的独立复核](<D:/Downloads/Final Year Project/idea/WM_REVISION_CONTRACT_RESOLUTION.json>)

单张 A100 40GB 是本轮采用的可行性约束；你没有给定 GPU-day 总额度。pipeline 的 factory default 150 GPU-days / $10,000 不能被当作你的预算。两条 pipeline 的 falsification_prediction 与 compute_budget 均与 Phase 2 原文一致。文献检索、独立批评、修订与阅读卡生成已完成；具体 checkpoint、真实低比特 backend、实验 manifest 等仍需在实施前锁定。原始估计、检索限制、未解决的 implementation annotations 和修正记录保留在各运行目录中。


## 2026-09-12 新候选入口

上文保留为 2026-09-08 原始方案记录。后续 RankCal / OTC 原配方已有 no-go screen；CEM-Update 是既有 handoff，本地尚未见最终结论。

本轮新增 weight VQ 与 non-VQ 两条候选，见 [新方案总览](<D:/Downloads/Final Year Project/idea/NEW_IDEAS_2026-09-12.zh.md>) 与 [旧实验复核](<D:/Downloads/Final Year Project/idea/EXISTING_IDEAS_REVIEW_2026-09-12.zh.md>)。新流程为精简 ResearchStudio-Idea，不能继承旧流程 DONE 或旧实验支持。
