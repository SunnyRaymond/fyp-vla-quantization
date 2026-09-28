# Efficient JEPA world model：下一步切入点

日期：2026-09-26。以当前 LeWM＋PushT 为落点，同时检索更广的 JEPA planning 文献。

这是精简 ResearchStudio-Idea 分析：文献 grounding、已有失败约束、机制筛选与独立反证。不是完整 IdeaSpark 的 DONE，也没有新实验结果或 novelty 认证。所有建议均是待检验假说。

## 判断

优先研究 **goal-query compression**：能否用比完整 latent dynamics 更便宜的计算，保留实际 goal 查询所需的候选代价差？

第二顺位是 **decision-ambiguity selective scoring**：只有廉价模型无法判定是否进入 elite set 的候选，才调用完整 teacher。它的难点是自适应搜索下的可信误差范围，以及真实 GPU batch 成本。

**scalar cost geometry** 可以作为一个短成本 gate，重新判断是否值得使用梯度；它不能仅凭“没有完整 Jacobian”就成为独立研究贡献。

目前没有证据表明这三个方案能替换 teacher。这里排序依据是机制清晰度、与本地 NO-GO 的区别和最小证伪成本，不是预测成功率。

**同日 pilot 结果更新：** 后续授权的 [Goal-query preliminary screen](../../experiment/idea-validation/lewm-goal-query-pilot/RESULT.zh.md) 已有效完成，job `25568103.pbs101` 为 `NO_GO_STOP`。四个训练臂都未通过 dev 质量门；exact-y oracle 的 tail signal 为 PASS，但 fully learned projected＋tail 的 structural signal 为 FAIL，且 test scoring 比 matched scalar_aux 慢约39.4%。原 r32/h256/1500-update recipe 至此停止，不推进 CEM；未测试 affine goal-family 变体，也没有新增 novelty 依据。下文保留执行前的假说、来源与证伪路线，不能读成对该已失败 recipe 的继续推荐。

**同日 follow-up：gap 判断收窄。** Goal-query compression 是本次讨论的工作名；原 projected＋tail 机制直接复用 2026-09-22 的本地 Goal-Sufficient Cost 提案，goal-family 公式是本轮的代数展开。它不是 pipeline 已认证的文献空白。补查经典 value/goal-conditioned approximation 后，宽泛的“只保留规划查询所需信息”已有明确 prior art；当前只能称具体可证伪假说，不能称已经确认的 research gap。

最接近的经典与 JEPA prior：

- [Value Equivalence，NeurIPS 2020](https://papers.nips.cc/paper/2020/file/3bb585ea00014b0e3ebe4c6dd165a358-Paper.pdf)：§3 Definition 1 按规划使用的 policies/functions 定义等价；§6 明确包含 open-loop action-sequence operators。因此“保留查询而非完整状态”，甚至“改成 open-loop 查询”，本身都不足以主张 novelty。
- [UVFA，ICML 2015](https://proceedings.mlr.press/v37/schaul15.html) 和 [Bilinear Value Networks，2204.13695v3](https://arxiv.org/html/2204.13695v3)：goal-conditioned value approximation 与 state/action–goal 的分解已有工作；BVN §3 Eq. (5) 为 Q(s,a,g)=f(s,a)ᵀφ(s,g)。共享 goal-family 的双线性形式不是新原则。
- [TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning，2510.00739](https://arxiv.org/html/2510.00739)：§3.2 区分 state/task encoders，predictor 预测 policy-conditioned successor features；§4 分析低秩 successor measures。这与 Temporal-Distance-JEPA 是两篇不同论文。JEPA 中 task-relevant prediction 的宽泛方向也已有工作。
- [Traj-LeWM，2608.14125v1](https://arxiv.org/html/2608.14125v1)：Eq. (5)/(14) 的 learned trajectory cost 读取完整 predicted trajectory 并加入 endpoint score；与本提案模仿冻结 teacher 的原 terminal query 有具体差别，但差别本身不证明 novelty。

上述论文是 follow-up 中补查的近邻，不能倒写为最初生成提案的实际灵感来源。当前待验证问题是：在原 teacher cost、未见 contexts/goals 和 adaptive CEM proposals 上，projected＋tail 或受限 goal-family scorer 是否比 direct scalar head、small full-latent prefix predictor 和 cached teacher 提供更好的 native quality–latency frontier。若 matched scalar baseline 支配结构化方案，应承认该结构没有独立价值。

Follow-up 检索日期为 2026-09-26，检索词包括 goal-conditioned cost/value model、value equivalence、successor features、bilinear value networks；读取 Value Equivalence、BVN、TD-JEPA、Fast-LeWM、Temporal-Distance-JEPA 和 Traj-LeWM 的相关方法段，UVFA/SF 补核官方摘要。范围限制：不是穷尽所有 cost distillation／surrogate MPC 工作，也未核查上述方法所有实现与复现结果；未发现完全相同方法不能用作不存在 prior art 的证据。

## 1. 现有文献已经占据什么

| 方向 | 已有 primary source | 对下一步的约束 |
|---|---|---|
| action-prefix 并行预测 | [Fast-LeWM，2606.26217v1](https://arxiv.org/html/2606.26217v1) | causal action-prefix encoder、dense prefix supervision、直接并行 future latent；还包括 optional prefix-decomposition consistency。简单“把递归 rollout 改并行”已不是空白。 |
| 用 learned controller 替代在线搜索 | [Latent Geometry Beyond Search，2605.08732v2](https://arxiv.org/html/2605.08732v2)；[INTACT 官方代码](https://github.com/zju3dv/INTACT-JEPA) | goal-conditioned inverse dynamics、actor-assisted CEM 和 direct control 都已有工作。新 proposal head 应先作为对照，不能只靠 amortization 名称主张 novelty。INTACT 代码还记录了 2026-09-14 的 history 修复，比较时须固定版本。 |
| 更适合优化的 latent geometry | [Temporal Straightening，2603.12231v3](https://arxiv.org/html/2603.12231v3) | 泛泛“轨迹更直，梯度规划更快”已有工作；时间轨迹与反事实动作方向仍是不同对象，但需要具体方法和成本证据。 |
| 保留 action-dependent differences | [AD-WM，2609.30264v1，2026-09-24](https://arxiv.org/html/2609.30264v1) | residual dynamics＋predictor-level action recovery；用 elite regret 诊断规划。action-aware loss、MSE 不等于 ranking、elite 比全 bank Spearman 更关键，都不能直接当新发现。其 MPC 在线流程保持原样，留下压缩计算的具体问题。 |
| learned goal/progress cost | [Temporal-Distance-JEPA，2607.25337v2](https://arxiv.org/html/2607.25337v2) | 已从 reward-free trajectories 学 temporal cost 并用于规划。这里的窄问题是低成本模仿锁定 teacher cost、减少 dynamics 调用；不能只说“另训一个 goal cost head”。 |
| 更少 latent tokens | [CompACT，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_Planning_in_8_Tokens_A_Compact_Discrete_Tokenizer_for_Latent_CVPR_2026_paper.pdf) | compact planning representations 已是成熟方向。其 token 数收益不能直接套到本来就使用单个 global token 的 LeWM。 |
| 给冻结 LeWM 加层级规划 | [Mind the Gap，2607.12547v2](https://arxiv.org/html/2607.12547v2) | naive hierarchy 可变差；宏动作 support、subgoal 可执行性与 execution mode 是具体障碍。不要以“加 hierarchy”本身作为下一步。 |

更广的 query-conditioned abstraction 也已有 [Physically Viable World Models](https://arxiv.org/abs/2605.30542) 和 [Planning-Query-Guided Model Generation](https://arxiv.org/abs/2508.19199)。所以“只建任务需要的模型”是研究原则，具体计算机制及验证才可能构成贡献。

上表是近邻工作筛选，不能推出任何候选已通过全面 prior-art audit。

## 2. 本地结果带来的约束

- [Iteration cache](../../meeting-ready/01-iteration-cache-cross-model-planners/cases/lewm-pusht-cem/RESULT.zh.md) 已在两个固定 observations 的冻结协议上保持 bitwise-exact CEM trace，full-planner 时间降低约 29.4%。后续效率比较应使用 cached teacher；这不是闭环任务加速结论。
- [Terminal-response＋late7](../../meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/terminal-response-late7/RESULT_25550701.zh.md) 在该新 checkpoint 的固定 50-task 评估上为 student 2/50、late7 12/50、teacher 49/50；late7 比 teacher 更慢。少调用 teacher 并不等于加速，也不保证恢复已偏移的 proposal。
- [Compiled model](../../meeting-ready/01-compiled-world-model/RESULT.zh.md) 和 [straightening pilot](../../meeting-ready/03-action-response-straightening/EXPERIMENT_DECISION.zh.md) 排除了对应的冻结 recipe：低 latent MSE 没有保住排序；完整 Jacobian 的成本高于 batched scoring。它们没有排除所有 query compression 或 scalar-gradient 方法。
- [CEM depth](../../experiment/idea-validation/lewm-pusht-cem-depth/RESULTS_25546847.zh.md) 与 [population decay](../../experiment/idea-validation/lewm-pusht-icem/RESULTS_25547844.zh.md) 已给出具体预算取舍。继续换递减 schedule，优先级低于寻找明确的新计算机制。

核心误差链是：latent error → cost differences → elite selection → proposal mean/std → later candidate distribution → executed action。只验证链条前端不足以接受替换。

## 3. 第一顺位：保留 goal 查询，而不是恢复全部 latent

已有 [Goal-Sufficient Cost 提案](../../meeting-ready/02_Goal_Sufficient_Cost_LeWM_PushT.md) 尚未训练、验证，可以复用。

设 terminal latent 为 z，goal 为 g，P 的列正交，R=I−PPᵀ。平方欧氏 cost 有精确分解：

\[
J(z,g)=\|P^T(z-g)\|^2+\|R(z-g)\|^2.
\]

以前低秩 dynamics 的问题是：被删掉的方向仍会改变评分。这里保留显式 projected latent，再用一个 goal-dependent scalar head 预测 tail 的评分贡献。不能直接丢 tail，也不能遗漏其与 goal 的交叉项。

输出变小不等于网络变快；scalar head 也可能比完整 latent 更难学。它回答的是 planning query，不是可递归替代 LeWM 的低维 Markov state。

### 一个更具体的分支：共享一组 goal 查询

如果部署 goal 家族可以写成 g=g₀+Pq，则：

\[
\boxed{J(z,g)=b(z)-2y(z)^Tq+\|q\|^2},
\qquad b=\|z-g_0\|^2,\quad y=P^T(z-g_0).
\]

同一 candidate 的 r+1 个输出可以支持整个 goal 家族，无需给每个 goal 重新预测 tail。这与压缩完整可递归 dynamics 的要求不同。

当前每次 CEM solve 通常只查询一个 goal，因而跨 goal 的共享不直接带来在线加速。结构化输出只有在训练泛化、可缩小的 predictor 或真实多-goal 工作负载上胜过 direct scalar head，才有实际价值；这一点是独立 critique 提出的关键限制。

但这取决于实际 goals 是否具有可利用的结构。对未覆盖的 goal，令 g=g₀+Pq+r_g，则遗漏项是：

\[
\|r_g\|^2-2\langle R(z-g_0),r_g\rangle.
\]

小 goal reconstruction MSE 不能保证这个遗漏项不改变候选排序。若所有 goal 差分张成完整 latent 空间，精确回答这些距离查询就能恢复完整 z，因为：

\[
J(z,g_1)-J(z,g_2)
=-2z^T(g_1-g_2)+\|g_1\|^2-\|g_2\|^2.
\]

因此没有“任意 goal 都可无损压成几个数”的一般承诺。上述是基础线性代数，不是 novelty 声明。

**最小证伪路线：**

1. 不训练新模型，先用 teacher 已有 terminal labels 检查 goal-family projection 的结构上限。P、g₀ 只在 training goals 拟合；在未见 contexts/goals 和 early/middle/late proposals 上检查 elite、proposal 变化。对比 goal-PCA、dynamics-PCA 和 orthogonal random basis。若有用的低维 goal 家族不存在，停止这个分支。
2. 对原 goal-dependent-tail 方案，先看 exact projected latent＋learned tail 的乐观条件能否优于 tail-zero／constant-tail。这个 oracle 辅助诊断付了 teacher 成本，不能声称部署加速。
3. 只有结构与学习 gate 通过才测试完整小模型；匹配 label access 和在线成本，对比 direct scalar cost head、small full-latent predictor、原 projected＋goal-dependent tail。进一步检验完整自适应 CEM 与 paired closed loop。

**负对照：** 打乱 goal 对应关系，检查正确 goal geometry 带来的 held-out elite/proposal 优势是否消失。它只是 signal-source 对照，还需上述强基线。

**停止条件：** 低维结构不支持筛选；tail candidate-dependent error 失控；被匹配成本的 scalar head 支配；或达到需要的精度后 native latency 没有降低。

潜在贡献应是“特定 goal-query family 的可压缩性、误差如何影响 CEM，以及实际质量—延迟 frontier”，而不是把代数分解命名成新 world model。

## 4. 第二顺位：只精算无法确定 elite 身份的候选

廉价 scorer 给当前 bank 中每个 candidate 提供区间 [Lᵢ,Uᵢ]。设 q 是所有 Uᵢ 的第 K 小值。如果每个真实 teacher cost 都同时在区间里，那么 Lᵢ>q 的候选不可能进入 top K；所有剩余候选再用 cached teacher 精算。

等号不能剪。此证明保证的是有条件的 elite membership；tie ordering、floating-point reductions、原 argsort 行为与 bitwise trace 还需要独立核查，不能从集合证明直接推出。

它与固定 last7 teacher 的区别是按候选筛选歧义付费，与简单 cheap-top-K reranking 的区别是给被廉价模型漏排的潜在好候选留出恢复通道。

**真正难点：**

- IID validation residual quantile 不是自适应 CEM 中整批候选同时覆盖的保证。低 confidence、ensemble disagreement 或 learned error head 也可能一起错。
- 区间过宽时几乎所有候选都要精算；区间过窄时会把真正 elite 剪掉。
- 实际 break-even 是 Tcheap(N)+Texact(M)+Tgate < Texact(N)。小 teacher batch 的 GPU 利用率可能让调用数收益消失。

先冻结 validation 误差 envelope，检验 held-out adaptive proposals 上的 coverage、错误剪枝、survivor 数与真实 native 成本。比较 same-teacher-budget uniform/random correction、cheap-top-K correction 和 cached full teacher。没有有效可验证的 bounds 时，只能称 empirical selector，不能称 certified rank-safe。

**停止条件：** 大多数候选仍然 ambiguous；漏剪 elite；或 gate 与小 batch 开销吃掉 savings。不能为了获得好结果事后收紧区间。

Multi-fidelity optimization 与 interval pruning 都是已有方法家族。研究空间在 JEPA/CEM 的自适应查询与可靠筛选之间，普通 confidence gate 本身不构成 novelty。

## 5. 第三顺位：先测 scalar cost 梯度是否有便宜入口

完整输出 Jacobian NO-GO 不等于一个 scalar goal-cost gradient 同样不可行：

\[
\nabla_u J=2\left(\frac{\partial z_H}{\partial u}\right)^T(z_H-g).
\]

reverse-mode 可以直接算 vector-Jacobian product，无需显式构造完整 latent-output Jacobian。但 backward、HVP、exact checks 都要计价；少生成一个矩阵不自动比优化后的 batch forward 更快。

先比较 scalar forward/backward＋拟议 exact checks 与 cached batched scoring 的成本，再在真实 CEM proposal 半径上检查局部 scalar surrogate 的 cost/elite fidelity。接触边界、early-round 大半径与重启次数都可能把收益消掉。

若通过，也只能先称已有 action-response 思路的 implementation salvage。需要与 standard gradient MPC、局部 trust-region／quadratic 方法和 Temporal Straightening 比较，不能把标准 VJP 当新算法。

另一条更面向架构与规划共同设计的切口，是问：**更好的 action discrimination 能否让同样任务质量只需更少的搜索预算？** AD-WM 已经改善 action discrimination，但保持 CEM 在线流程；可以把具体研究对象改为 representation × candidate/iteration budget 的质量—延迟 frontier。先用可复现现有方法检验这条曲线，而不是发明新的 inverse-dynamics loss。若只有高预算成功率提升、低预算时仍失败，就没有效率证据。这是补充假说，尚未经过独立 candidate critique；单纯复现 AD-WM 后减少预算属于应用研究，若要方法贡献，仍需揭示并利用具体的误差／搜索结构。

## 6. 执行顺序与证据边界

先做 A 的结构上限 gate；优先复用原 projected＋tail 提案，goal-family 变体只有在目标结构存在时保留。B 的关键是成本与可靠区间共同可行；C 只值得短成本探针。

每一步保持固定 encoder、goal criterion、数据划分和明确的候选协议，避免同时更换任务目标、架构与优化器。质量与速度均以 cached teacher 为对照，并匹配实际 active-environment count；teacher labeling、oracle features、校准、gate、fallback 和重新启动都要计成本。

先后分清：结构／predictor gate → adaptive planner gate → native timing → paired closed loop。可以报告 NO-GO 或 INCONCLUSIVE；不凭更低 MSE、调用数减少或离线 oracle 成功声称 task success。

上述分析阶段未提交训练、benchmark 或 PBS 作业。用户随后授权的初步验证已单独冻结于 [Goal-query pilot 协议](../../experiment/idea-validation/lewm-goal-query-pilot/PROTOCOL.zh.md)，2026-09-26 提交 PBS job `25568103.pbs101`，仅检验原 projected＋goal-dependent-tail 路线，未检验 affine goal-family 变体。模型／数据 I/O、推理、训练与 benchmark 只放在真实 guarded compute allocation；GPU 实验记录利用率和显存。禁止使用 banked reset。

## 7. Pipeline 与检索范围

使用 idea-spark 的 bottleneck、pattern selection 和 falsification 思路，重点采用 C04 的要求：先证明资源／误差的异质性，比较 uniform 方案和简单替代，最后衡量 native deployment 指标。检查过但没有强行绑定不符合本任务的 attribution、identifiability、graph-expressivity cards。

Phase0 尝试使用缩小 connector pool；当前 arXiv connector 返回 HTTP 406，title-named anchors 未解析。通过官方 arXiv HTML／论文与官方 repositories 补充 primary-source grounding，保留 connector log。该路径不满足完整 canonical Phase0→Phase1 gate，不应报告正式 phases 全部通过；也没有完成 exhaustive collision search。

独立 [candidate critique](../../tmp/efficient-jepa-critique.md) 对前三个机制给出有条件 small-screen／implementation-salvage 判断；没有通过 novelty 的结论。它最初阅读了 Temporal Straightening v2、GC-IDM v1；主报告与 [literature summary](literature_summary.md) 已补核当前 v3/v2 元数据和全文。所有实测结论仍以链接的本地 frozen result 为准。
