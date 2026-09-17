# RankCal：面向动作条件潜在世界模型规划的反事实影响量化

**方法名称：** RankCal (Counterfactual Influence Quantization)

## 研究动机
Phase 1 识别出一个结构性瓶颈：在 world-model planning 中，post-training quantization（PTQ）的影响会随 bit-width、granularity、module 和 planning horizon 改变。因此，planning objective 可能表面正常，但 task success 仍会下降。在这个设置中，planner 使用 action-conditioned latent dynamics model 为想象的 action sequence 打分。一次局部扰动可能重新排列候选序列、改变被选中的 action，并在 receding-horizon closed-loop execution 中改变后续 latent state。tensor-MSE heuristic 衡量的是重建误差；这里下游真正需要保留的是排序，而不是重建目标。

近期工作让这个 gap 可以在不改变 planner 的情况下进行实际测试。全文支持的 `arxiv:2608.24855v1` 研究 latent trajectory generation 和 rollout reranking；已由 connector 验证、但只有 abstract 的 `arxiv:2607.28405v1` 暴露了与部署不匹配的 WAM PTQ 问题；全文支持的 `arxiv:2602.02110v1` 提供了最接近的系统性 world-model quantization sweep。Public visual continuous-control tooling 和 local inference 使 site-only counterfactual probe 具有可行性。明确的 horizon-wise planning 和 backend-level byte accounting 把问题变成范围受控的 decision-level audit。实际的 mixed weight/activation（W/A）backend support 和测得的 `PeakBytes` 仍是实现条件，不是预先假定的条件。

`arxiv:2602.02110v1` 是最接近的 empirical anchor。它系统扫描了不同 methods、granularity、modules 和 planning horizons 下的 weight-only 与 joint W/A PTQ，并报告 planning objectives 与 task success 之间存在不匹配。但它没有定义一种 frozen-model、site-only 的 counterfactual candidate-order influence estimand，也没有用这个 profile 求解受测量的 `PeakBytes` 约束的 mixed W/A allocation。它的 measurement axes 和 aggregate outcomes 没有包含 one-site intervention，因此没有把 imagined rollouts 中的 candidate-order discordance 接到 bit allocator 上。

`arxiv:2608.24855v1` 用 rectified-flow latent trajectory generation、inverse-dynamics decoding 和 frozen-LeWM rollout reranking 替代了反复的 CEM search。它没有利用多步 quantized rollouts 的 site-specific ranking influence，去改变 frozen latent planner 的 PTQ calibration 或 bit allocation。它重新设计的是 plan generation；它没有根据 decision-level sensitivity 来测量和分配 numerical precision。

`arxiv:2603.12231v3` 使用 curvature regularizer 拉直 local latent trajectories，并改善 latent distance 和 planning objectives 的 conditioning。它没有估计 quantized action-conditioned rollouts 下的 site-only counterfactual ranking influence，也没有把这个估计用于 mixed-precision PTQ。它在 training 阶段改变 representation geometry；RankCal 需要的是面向 fixed planner 和 latent dynamics model 的 post-training intervention 与 allocation mechanism。

已由 connector 验证、但只有 abstract 的 `arxiv:2607.28405v1` 为 World Action Model PTQ 提出了 coordinate-compatible activation evidence、joint video-action saliency 和 fixed-intervention auditing。它没有针对 imagined action-sequence latent-dynamics planner rankings，也没有从 site-only rollout-ranking influence profile 推导 weight/activation bits。它关注 iterative joint video/action denoising 和 WAM coupling；因此，它的 intervention target 不同于 imagined action-sequence planning 在 closed loop 中产生的后果。

如果 gap 被填补，mixed-precision PTQ 就可以成为 imagined action-sequence planners 的 decision-preservation tool。它可以暴露这样的情况：两个 allocation 具有相近的 nominal bit-width，但 candidate-order 和 closed-loop consequence 不同。它还会为 `arxiv:2602.02110v1` 的 broad sweep 提供 site-level diagnostic interpretation 和 deployable allocator interface；同时，它仍与 `arxiv:2608.24855v1` 的 planner redesign 正交，也与 `arxiv:2607.28405v1` 的 WAM joint video/action denoising branch 分开。

## 方法
### M1_background
*建立 RankCal 所依托的 frozen imagined-rollout probe。*

1. 从冻结的 FP16（16-bit floating-point，16 位浮点）动作条件潜变量世界模型（action-conditioned latent world model）开始。建立 `site_manifest`：为每个可量化的权重/激活位置（weight/activation, W/A）分配稳定的 site ID，并记录模块或算子路径（module/operator path）、张量形状与布局（tensor shape/layout）、校准数据（calibration data）来源、允许的位宽组合（bit pairs）和执行状态。冻结规划器输入（planner inputs）、带稳定 candidate ID 的候选动作序列池（candidate action-sequence pool）、潜变量初始状态（latent initial states）、rollout 时域（rollout horizons）、滚动时域调度（receding-horizon schedule）、planner 已使用的标量排序分数（ranking score）以及 tie rule；评估器（evaluator）对每个候选项（candidate）和时域（horizon）输出该分数及潜变量状态轨迹（latent-state trace）。选择并记录具体的目标推理后端（target inference backend，即真正执行 kernels 的 runtime）及 support manifest：只有实际部署的 kernel 和存储布局（storage layout）执行该 bit pair 时，才标记为 `backend_native`；如果 PyTorch 的 `torch.ao` 只模拟量化算术，则标记为 `emulation_only`。 【作者需决定：明确 target backend 及支持的 W/A kernel/layout 矩阵，并决定哪些 unsupported pairs 只能作为 emulation】 对每个 site 和允许的 bit pair，只量化该 site，其他 sites 保持 FP16；使用完全相同的输入、顺序、已声明的随机种子（random seeds）、校准流程（calibration procedure）和 tie rule 重放。记录量化器参数（quantizer parameters）、运行时来源（runtime provenance）和执行模式（execution mode）。输出按 site ID、bit pair、horizon 和 candidate ID 对齐的 FP16 与单 site 结果；不要合并 native 与 emulated rows。

*在冻结的 action-conditioned latent dynamics 中，对每条候选 action sequence 进行 imagined rollout。*
$$ z_{t+1}=F_{\theta}(z_t,a_t),\quad \tau(a_{0:H-1})=(z_0,a_0,z_1,\ldots,a_{H-1},z_H) \tag{1} $$

   - _为什么：_ 这样可以在 action-conditioned latent-dynamics branch 中隔离 site-level perturbation，也能避免把 planner redesign 或 WAM denoising changes 误认为 quantization effects。

### M2_rankcal
*测量逐 site、逐 bit pair 的 rollout-ranking influence，并把它转成受 memory 约束的 mixed W/A assignment。*

2. 对每个量化位置（site ID）和允许的位宽组合（bit pair），在完全相同的候选池（candidate pool）、潜变量初始状态、已声明时域和随机种子上重放冻结探针（frozen probe）。把固定候选池中不同候选项（candidate ID）的所有无序组合定义为比较集合 `P`；FP16 和单 site 运行都使用 planner 原有的标量分数（scalar score）。按照 E2 的两两排序一致性定义比较两条分数差，并为每个时域保留由不一致度得到的 I_g(b) 排序影响（ranking influence）。分数差为零时保持 tie；只有 planner 必须选择 action 时才使用 第1步 的 tie rule，不加入随机扰动（jitter）或新分数。若 rollout 执行含随机性，保留每个已声明随机种子的分数，并对所有记录行使用同一聚合规则（aggregation rule）。 【作者需决定：选择 single-trace 还是 seed-set aggregation，并固定 aggregation 发生在 pairwise comparison 之前还是之后】 每一行按 site、bit pair、horizon 和 execution mode 记录排序一致性（agreement）、I_g(b)、分数来源（score provenance）、校准版本（calibration revision）与后端状态（backend status）。native 与 `torch.ao` emulation rows 分开保存；只有得到 the bit-allocation support manifest 接受的 rows 才能进入 deployable assignment。

*只量化 site g 为 bit pair b，比较候选 action sequence 的两两排序一致性，并把不一致度转成 influence。*
$$ A_g(b)=\frac{1}{|\mathcal{P}|}\sum_{(i,j)\in\mathcal{P}}\mathbf{1}\left[\operatorname{sgn}(s_i-s_j)=\operatorname{sgn}(s_i^{(g,b)}-s_j^{(g,b)})\right],\quad I_g(b)=1-A_g(b) \tag{2} $$

   - _为什么：_ ranking estimand 把 tensor reconstruction error 与 planner 的 decision-level sensitivity 连接起来，也包含 selected action 改变后产生的下游后果。
3. 使用 第2步 的 site-by-bit influence table、允许的位宽组合列表（bit-pair list）、后端支持清单（backend support manifest）、固定的 target histogram 和给定的 `PeakBytes` 上限。为每个可量化 site 分配一组权重/激活位宽。先定义 histogram 并记录它按 site 还是 tensor elements 计数。 【作者需决定：选择按 site 计数还是按 element 加权的 histogram 语义，并在观察结果前固定 exact byte equality 或预先声明的 matching tolerance】 对每个完整分配方案（assignment），在目标运行时（target runtime）中构建完整的训练后量化（post-training quantization, PTQ）checkpoint，并用同一个已声明的测量工具链（measurement harness）测量端到端 `PeakBytes`：从加载 checkpoint 开始到完整 planner trace 结束，包含模型存储（model storage）、量化器元数据（quantizer metadata）、运行时工作区（runtime workspaces）、planner/candidate state、潜变量滚动缓冲区（latent rollout buffers）和临时分配（temporary allocations）。 【作者需决定：固定端到端测量的起止边界，以及采用 device allocated 还是 reserved bytes 作为计数器】 不得用 nominal bit count 的算术估算或 `torch.ao` emulation 的 bytes 替代实际后端的 `PeakBytes`。枚举完整分配方案，只保留满足 fixed histogram 和 measured limit 的方案，并选择 E3 定义的 aggregate site influence 最小者。记录每个 site 的 assignment、support mode、measured peak 和 histogram。使用相同的 calibration data、runtime、histogram 和 measurement harness 构建统一位宽基线（uniform-bit comparator）与局部张量重建基线（local tensor-reconstruction comparator）。 【作者需决定：定义 local reconstruction scalar 及 weight/activation normalization，并规定 exact histogram 或 measured-PeakBytes match 无法取得时如何表示 uniform comparator】只有比较对象拥有相同 histogram 且遵守相同的 measured-byte matching rule 时，才标记为 matched；否则标记为 unmatched。

*在 measured `PeakBytes` 预算和固定 bit histogram 约束下，选择 aggregate ranking influence 最小的 mixed W/A assignment。*
$$ \mathbf{b}^{\star}=\arg\min_{\mathbf{b}}\sum_{g\in\mathcal{G}}I_g(b_g)\quad\text{s.t.}\quad\operatorname{PeakBytes}(\mathbf{b})\le B,\;\operatorname{Hist}(\mathbf{b})=\mathbf{h} \tag{3} $$

   - _为什么：_ 这样可以把 diagnostic profile 转成 deployable mixed W/A PTQ assignment，同时不改变 latent planner，也不把 measured memory 换成 nominal byte assumption。

### M3_validation
*在 planner 不变的情况下，评估 candidate ordering、first-action agreement 和 closed-loop task success。*

4. 将 `b*` 应用到完整的可量化位置集合（quantizable-site set），在实际目标后端 checkpoint 上运行未改变的 planner。使用版本化 `evaluation_manifest`，列出任务与 episode IDs、初始状态、候选池、已声明时域、replanning interval、stopping rule、random seeds 和既有 success predicate；对 FP16、RankCal、uniform-bit、local-reconstruction 与 permutation control 完全相同地重放。 【作者需决定：在任何比较前，明确 evaluation manifest 及其 task、seed、horizon 和 success 字段】 每个 replanning point 保存完整候选排序（candidate ordering）、相对于 FP16 按 E2 得到的 candidate-order agreement、按 the tie rule fixed in the frozen-probe step 选出的 first action 和既有闭环任务成功结果（closed-loop task-success）；按 horizon 和 episode 汇总但不改变 planner。每条结果同时记录固定的 histogram、端到端 measured `PeakBytes`、native/emulation status 和 byte-matching stratum；最终 checkpoint 必须是 all-sites assignment，不能只复用 one-site probes。在同一个 all-sites evaluation 内，按照预先声明的规则选取代表性的高/低影响位置对（representative high/low site pairs），并构造完整的匹配变体（matched variants）：只固定高影响位置、只固定低影响位置，或同时固定两者，其余 sites 都按同一 completion rule 分配。所有 variants 使用相同的 candidate pool、histogram、measured `PeakBytes` target、planner schedule、receding-horizon convention 和 tie rule；single-site 与 joint rows 都采用 E2 定义的 candidate-order discordance。 【作者需决定：在观察结果前固定 pair-selection rule、horizon 和 bit stratum、tie-break 以及 full-assignment completion】 如果无法构造 exact histogram-and-byte match，就将该 pair diagnostic 标记为 `unavailable`，不比较这些 rows。这个 diagnostic 只作为 第4步 内的战术性证据（tactical evidence）：不定义新的估计目标（estimand）或汇总目标，不增加 step ID，不声称 joint optimality，也不改变既有 permutation-based 反证方向（falsification direction）；任何 discrepancy 只能限定 site-wise allocation signal 的解释。
   - _为什么：_ 需要保留的对象是 selected action 及其后续 consequences，因此 evaluation 留在 latent-dynamics planning 中，不切换到 joint video/action denoising。这个 pair diagnostic 在同一个 all-sites evaluation 内测试 interaction boundary，不改变 site-only estimand 或 falsification prediction。

