# 独立 idea 最终审查（2026-09-12）

本版复审了既有 `EXISTING_IDEAS_REVIEW_2026-09-12.zh.md`、`PRIOR_SCOUT_2026-09-12.zh.md`、`FEASIBILITY_AUDIT_2026-09-12.zh.md`，两分支的最新 `STATUS/IDEA/REVISION_RESPONSE`、Phase 1/selection 与已保存的 primary-source 近邻记录，并按 `ponytail`、Idea Spark critique/implementability/anti-pattern 约束执行。没有运行模型、实验、集群任务或重 I/O；不改候选目录。该版是精简 ResearchStudio-Idea 审查，不宣称 canonical pipeline DONE。

## 最终判定

| 候选 | 判定 | novelty | 建议 |
|---|---|---|---|
| 历史 STRC/TRFC | **abandon** | 否 | 不再修补；保留为 rejected attempt |
| Frozen-Residual Transport PTQ（FRT） | **revise / pilot-ready conditional** | 未认证 | **优先做最小 pilot**；先过 hard-W4 与 full-history interface gate |
| Trajectory-Relational Product VQ（TR-PVQ） | **revise / pilot-ready conditional** | 未认证 | 后做；先完成 relational-VQ collision 与 fixed-byte identity gate |

“pilot-ready conditional”只表示假设已经足够具体，可以在硬门通过后做小规模证伪；不表示方法新颖、有效或 native deployable。若必须选一条先试，选 FRT。它的目标、变量和 stop rule 比 TR-PVQ 更窄，且不需要先构建 VQ codebook/kernel 路径。

## STRC：abandon

STRC 的阻断问题仍是结构性的。`K_r(ell)=E[e_(t+ell)r_t^T]` 是 paired trajectory association，不是 causal transfer；`sum_ell ||K_r(ell)||^2` 丢掉 lag sign，不能解释 cancellation。更严重的是当前 `A(q)` 来自自然 q→FP drift，而 `A_pi(q)` 来自 FP path 注入 shuffled residual，二者不是同一 intervention，`Delta_coh` 混入 model/controller path 差异。若 `e` 是 transition increment，`gamma=1` 又退化到普通 terminal displacement。single-site score 换成 trajectory score、fake logical bytes 或另加 observer 都不能修复这些 load-bearing 问题。因此 STRC 不应继承到 FRT，也不应为凑两个 idea 放行。

## FRT：revise，优先 pilot

FRT 的核心对象现在自洽：固定 `Q0`（symmetric per-output-channel weight-only W4 RTN）在同一 history/action/RNG 上产生
`delta=stopgrad(F_Q0(X,a)-F_FP(X,a))`，只放入下一预测 latent 槽；再用同一 `x`、`delta`、`a'` 比较
`T_theta=F_theta(x+delta,a')-F_theta(x,a')` 与 `T_FP`。clean latent reconstruction 加 transport finite difference，固定 W4 logical bytes，不作 causal、严格 Jacobian 或部署时 error-feedback 声明。相较 STRC，这已经移除了错误的 signed-lag/phase null。

当前仍不能认证 novelty。这个 loss 可能只是“选一个 Q0 方向的局部 sensitivity/Jacobian 对齐”：QDrop/input-noise、PD-Quant、Sobolev/JVP/GAD 与 direct two-step 已覆盖相邻原则。候选的 random same-norm control 若确实使用完全相同的 finite-difference target、优化器、W4 参数自由度、CAL/DEV 预算和 endpoint，是关键且足够强的第一对照；显式 JVP/multi-direction finite difference 可作 diagnostic，不必成为首轮硬性全量 pilot。若 Q0 direction 在 fresh DEV 与 held-out closed-loop endpoint 不优于 random control，FRT 应 **abandon**，不能改称 WM-specific novelty。

必须保留三项硬检查：

1. **Hard W4 数学和边界。** 写死 `q_min/q_max`、scale 的 per-output-channel 轴、zero/tie rounding、clamp 顺序、scale/metadata 是否计入 actual-byte ceiling。soft `h(alpha)` 只能用于 CAL，最终必须硬化为 `{0,1}`、丢弃 `alpha`，并从 materialized checkpoint 重新运行 transport；不能把 STE/软值或 logical bytes 当部署证据。
2. **完整 history 槽。** 明确 `X` 的 slot 顺序、shift/concat、normalization、cache/history fields、latent shape 与 RNG replay；`delta` 只能替换新预测槽，不能 broadcast 到全 history。`x+delta` 必须是 runner 接受的合法状态，且 `T_FP`/`T_Q` 使用同一 current state 和下一 action。
3. **冻结方向是否迁移。** CAL 的 frozen `delta_Q0` 不能自证最终部署模型。DEV 必须额外记录一次 fresh `Q_theta` residual，与 `delta_Q0` 比较 norm、cosine/alignment 和 transport ranking；若最终 residual 与 frozen direction 明显失配且 fresh-Q0/FRT 不传递，只能判 Q0-proximal overfit，随后 **abandon**。

physical `xy` 只作为独立 environment closed-loop endpoint，不需要也不应假设 latent→`xy` 线性 `P`。但 full history 和 one-step map 仍是 interface gate；若只能获得不稳定 latent 或不能恢复 history，则停止整个 physical claim。FRT 没有 observer、FP teacher、online correction 或 bit allocator，所以与 Feedback World Model 的 online residual observer、RPIQ 的 residual compensation/error-feedback、DA-PTQ 的 virtual Jacobian/motion surrogate、QuantWAMs 的 Fisher/reachable-state schedule repair 保持可检查的差异。AdaRound 只提供 rounding materialization，不能构成 FRT novelty。

FRT no-go：random same-norm（以及后续 diagnostic JVP）等效；只改善 CAL、不改善 fresh Q0/Q_theta DEV；`delta_Q0` 与最终 residual 严重失配；full-history/state contract 或 hard-W4 map 不成立；或差异仅由 clean-MSE/额外参数自由度造成。FRT 只有在这些门通过后才可从 conditional pilot 进入结果审查。

## TR-PVQ：revise，暂不优先

TR-PVQ 已从“固定 action Gram”修正为：用量化 DINO-WM 对 fixed action probes 重新 rollout，得到 predicted trajectory relation；以 goal-anchored absolute term 加 off-diagonal pairwise Gram term 拟合 codebook/index，并锁定 actual-byte ceiling、group map，不做 rate allocator、dynamic precision、MotionVQ schedule 或 custom accelerator。该修正使目标确实依赖 quantized model，删除 Gram diagonal 并保留 absolute/planner score anchor，也回应了旋转、平移和 diagonal-as-goal-score 风险。

但关系量化的直接碰撞尚未关闭。RKD（CVPR 2019）已经以 pairwise distance/angle relation 做 relational distillation；`QATMA/TPSD`（arXiv:2603.05964）已有 text-anchored pairwise similarity 与 QAT/OVOD 组合。它们不等于本候选的 DINO-WM weight-VQ protocol，却足以禁止“首次关系量化原理”表述。VPTQ/AQLM 已覆盖 codebook/index/packed lookup，RSAVQ 覆盖 FIM/natural-gradient/sensitivity allocation，MotionVQ/VQVLA 覆盖 motion-aware precision/centroid reuse/hardware，DA-PTQ 覆盖 Jacobian/motion drift，QuantWAMs 覆盖 reachable-state/Fisher/schedule，Feedback World Model 覆盖 online residual observer。TR-PVQ 的可辩护范围只能是固定 bytes 下的 WM imagined-trajectory relation objective；能否独立成立必须由实验决定。

必须证明 `lambda_rel>0` 改变了 codebook/index 选择并在 held-out action neighborhood、planner first action 和 return/success 上超过 `L_abs`、local-MSE/VPTQ-like、scalar 和 independent-codebook；pair-label shuffle、diagonal/goal-score-only、same-byte random/index perturbation 也要保留。若 relation 只改善 CAL/intermediate geometry，或与 `L_abs` 产生同一 map，降级为 negative/ablation；若 RKD/QATMA/TPSD 或其他 exact relational/task-aware VQ protocol subsumes objective，直接 **abandon**。

TR-PVQ 的 Gram/absolute anchor 现在比初稿合理，但仍须在 source output coordinate 中冻结 normalization、goal target、horizon、pooling 与 CEM objective；不能把 off-diagonal report 改写成原始 CEM score。VPTQ/AQLM layout 只代表 logical storage。actual codebook/index/scale/metadata bytes、packed loader、Peak VRAM、同步 throughput 和 action/return 必须分别记录；fake quant、nominal bits 或 inherited format 不能推出 native compression/latency。

## 4×A100 与停止边界

已有 feasibility audit 的 DINO-WM Wall CEM 约 4.06 GiB、probe sampled device memory 约 12,820 MB，支持单卡 inference-only 的 bounded screen；并不证明 backward、完整 imagined-trajectory/codebook search、native W4 或 runtime。FRT 的 ≤12 A100 GPU-hours、TR-PVQ 的约 8–16 A100-hours 都是规划上限而非实测 runtime/统计保证；40GB 与 80GB 分开报告，80GB 不能成为超 cap fallback。先用一张 A100 做 FRT interface/hard-map/fresh-direction gate，再决定是否扩大；TR-PVQ 只有在 tiny codebook identity gate 和 held-out relation check 有信息时才扩展到最多四卡。

未来所有模型加载、重 I/O、hash、解压、编译和实验均须在真实 PBS allocation 中同时核验非空 `PBS_JOBID`、非-login hostname 与 GPU allocation；login node 只做轻量控制。实际执行前不得把本审查的 feasibility 数字写成实测结果，也不得写入 credentials。

最终建议是：交付两条诚实的 conditional/pilot-ready research hypotheses；报告中把 FRT 置于优先 pilot，把 TR-PVQ 置于后续受控验证。两者当前都不能写成已认证 novelty、已证实有效或 native low-bit deployment。

编辑校正（root，2026-09-12）：arXiv:2603.05964 正式标题为 QATMA: Quantization-Aware Training with Multimodal Alignment for Open-Vocabulary Object Detection；早期检索误标 CR-QAT。最终 VQ pilot 固定 indices、仅更新 codebook entries，正文中 codebook/index 的泛称不代表首轮更新 index。具体优化预算和 FRT action-slot/hard-W4 规格以最终 IDEA 卡为准；这些编辑未改变独立审查的 conditional 判定。
