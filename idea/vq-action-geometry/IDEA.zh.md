# Goal-Anchored Off-Diagonal Trajectory VQ（条件式研究假设）

本候选研究 **weight vector/product VQ**，不把 VQ-VLA 这类 action tokenizer 当作模型参数压缩。目标是在固定实际 bytes、固定 grouping 和同一 packed lookup layout 下，检查量化模型是否保留 action-conditioned world model 的规划比较关系。当前没有实验或 novelty 认证；下文的“贡献”均是待证伪假设。

## 研究问题和边界

DINO-WM 的 planner 对候选 action sequence 调用 source 中冻结的 objective function，再由 CEM 选择 elite。local-MSE/VPTQ-like codebook 可以降低 weight 或 block output 的点误差，却未必保持两个相近候选 action 在量化模型中的相对排序。候选问题是：

> 在相同 packed bytes 下，给 additive/product VQ 加入量化模型自己产生的、带共同 goal anchor 的 off-diagonal trajectory relation，是否在未见过的 initial states、goals 和 action neighborhoods 上改善 planner first action 与 closed-loop return/success？

此问不预设答案。若 relation 项没有独立 downstream 增益，或 scalar/independent-codebook 达到相同结果，则保留为 negative result，不声称新的 VQ 原理。

## 精确定义和数据流

令 $e$ 表示一个 episode initial state/goal，$k$ 表示固定 action probe，$h=1,...,H$ 表示 imagined horizon，$P∈{F,Q}$ 分别表示 FP32 reference DINO-WM 与当前量化模型。相同的 initial observation、goal、action probe、planner random source 和环境/rollout seed 在 FP/Q 间配对：

$z^P_{e,k,h}=F_P(z_e,a_{e,k,0:h})$。

因此 fixed action probe 是 input；随 quantization 变化的是 Q model forward 产生的 $z^Q$，而不是预先计算固定 action 的 Gram。$z$ 是 model rollout 的 predicted latent/observation representation，不能把固定 action 序列自身当作量化后的 signal。

在具体 anchor 上，planner score 必须直接调用 reproduction/dino-wm-wall/source/planning/objectives.py 的 create_objective_fn(alpha, base, mode) 和 CEMPlanner 中同一个 objective_fn 调用；source 的 mode="last" 是默认路径，mode="all" 是另一个明确配置。候选不能把自定义 temporal-average latent distance 写成原 CEM score。定义

$J^P_{e,k}=objective_fn(rollout_P(o_e,a_{e,k}), encode_obs(o_{g_e}))$，

其中 rollout、target encoding、alpha、base、mode 和 shape broadcasting 完全沿用冻结 source/config。$J$ 是实际 planner score，CEM 的排序和 elite update 也保持 source 逻辑。第一版 local Wall anchor 记录 $H=5$、CEM SCREEN 的 $topk=30$、$num_samples=300$、$var_scale=1$、$opt_steps=5$；这些是 implementation anchor，不是本候选的实验结果。

另定义一个 auxiliary relation representation $r^P_{e,k}$：它从同一次 rollout_P 的 predicted visual/proprio outputs 取得，使用 CAL 前冻结的 pooling、flattening 和 per-channel normalization；$g^r_e$ 是同一坐标中由 goal observation 得到的 frozen target representation。它可以是 source mode="last" 对应的 terminal output，也可以是 mode-specific all-step concatenation，但必须和实际实现 manifest 一致。$r$ 只作为辅助 codebook-fitting object；它不被宣称等于 $J$，也不能替换 objective_fn。这样既保留原 planner score anchor，也让 relation 的 independent contribution 可单独测试。

令 $u^P_{e,k}=r^P_{e,k}-g^r_e$，$s^P_{e,k}=J^P_{e,k}$，并只对 $i != j$ 计算

$B^P_{e,ij}=u^P_{e,i}^T u^P_{e,j}$，

$G^P_{e,ij}=B^P_{e,ij}/(||u^P_{e,i}||_2 ||u^P_{e,j}||_2+epsilon)$。

$B$ 是未归一化、以共同 goal anchor 为中心的 Gram；$G$ 是其 cosine-normalized diagnostic，二者都只使用 off-diagonal。$G$ 的 diagonal 接近 1，并不等于 goal score；$B$ 的 diagonal 才是 $||u||^2$。共同 source output coordinate、$L_abs$ 中的 absolute anchor 和原 objective_fn 的 $J$ 防止将 rotation/translation ambiguity 当成规划不变性。

量化拟合目标为

$L_abs = mean_{e,k} ||r^Q_{e,k}-r^F_{e,k}||_2^2 + beta_J mean_{e,k}(J^Q_{e,k}-J^F_{e,k})^2$，

$L_off = mean_{e,i != j} |B^Q_{e,ij}-B^F_{e,ij}|^2$，

$L_TR=L_abs+lambda_rel L_off$。

$L_off$ 删除 diagonal，因为 $B$ 的 diagonal 是 $||u||^2$，而 CEM 的实际 score $J$ 已经单列为 anchor；$G$ 的 diagonal 又接近常数 1，二者都不能作为独立 relational evidence。$beta_J$、auxiliary pooling、normalization、epsilon 和 $lambda_rel$ 在 CAL 前写进 manifest；关系项默认先以 CAL 的 robust scale 归一化并使用一个固定正系数，DEV 只在一个很小的预注册候选集合中选择是否保留它。不得以 DEV/TEST 重新拟合 codebook。

## 固定 bytes 的 VQ 操作

对预注册 weight block $b$，把长度 $d$ 的 weight vector $w_{b,m}$ 表示为 additive/product codewords：

$\hat w_{b,m}=\sum_{r=1}^R C_{b,r}[q_{b,m,r}]$。

$d$、$R$、codebook size、vector grouping、block map、index bits、codebook dtype 和 scales 在开始前固定，沿用 VPTQ/AQLM 可表示的 packed lookup layout。先定义一个共同的 actual-byte ceiling $C_bytes$，再为每个 variant 单独记下实际 bytes 和 slack $C_bytes-bytes$；不能把不同 metadata/codebook overhead 的 variants 声称天然 same bytes。若某 variant 超过 ceiling 就淘汰或降低其 codebook storage，并重新登记，不能用 nominal bits 覆盖。候选删除 rate allocation、dynamic precision switching、execution-state schedule、centroid reuse 及 custom hardware，避免与 RSAVQ、MotionVQ 或旧 mixed-precision ranking 混成另一问题。

初始化是同 map 的 VPTQ-like/local-MSE codebook。首个 pilot 固定 indices，只优化 codebook entries，避免对每一个 weight vector/codeword candidate 重新跑完整 rollout：

1. 用当前 Q map 在 CAL paired action probes 上运行 forward，得到 predicted outputs、$J^Q$、$r^Q$、$B^Q$。
2. 固定所有 $q$，通过 lookup/gather 和 Q rollout 直接对 $L_TR$ 反向传播，只用 Adam 更新 codebook entries。固定 indices 时不需要 STE；非线性 trajectory loss 也不被声称具有 block least-squares 闭式解。FP teacher 冻结并缓存目标。
3. 每次拟合至多 200 optimizer steps，单 batch 至多 8 个 paired action probes、H=5，至少两个 fit seeds。先用真实单卡 forward/backward smoke 测 step time，再冻结能让全部必要对照落在总计 ≤16 allocated A100-hours 内的共同 step 数、数据量和拟合次数；所有拟合对照使用相同更新预算。200 是规划上限，不是收敛或完成时长保证。
4. 首轮不更新 indices，也不逐 vector/codeword candidate 做完整 trajectory search。index 优化另属未来工作，不能用于解释首轮结果。
5. CAL 用于拟合并记录 loss；DEV 只选择预注册候选，不更新 codebooks。全部 map 冻结后 TEST 只运行最终确认。若共同预算不足以完成必要对照或获得有效优化，标为 resource/optimization-inconclusive，不能用两次未收敛更新宣判研究假设失败。

这里的可检验 VQ 机制是 shared codebook entry 的一次更新同时改变所有引用它的 weight vectors，并通过 Q rollout 影响多个 imagined trajectories；首轮 indices 不变，不声称发生离散 assignment 搜索。必须用 same-target scalar quantization 和 independent-codebook VQ 检验共享更新是否有作用。若二者相同，VQ-specific claim 失败，但仍可保留一个 task-target ablation 结果。

## 相邻工作和差异假设

- **VPTQ / AQLM / QuIP#**：提供低比特 codebook/index、additive codebook、residual/lattice 或实际 packed inference 的基础；本候选改的是 CAL fitting object，并不重新声称这些存储表示新颖。
- **RSAVQ（arXiv:2510.01240）**：使用 language-loss Fisher/Riemannian natural-gradient error direction，并以 channel curvature 做 bit allocation。这里固定 byte map，不用 FIM projection、channel allocation，也不以“geometry/sensitivity”泛词冒充差异。
- **QuantWAMs（arXiv:2607.28405）和 QuantWM（arXiv:2602.02110）**：分别提供 joint video-action Fisher/reachable-state schedule 与 world-model/planner sensitivity audit；本候选不复用 Fisher/layer schedule，而测试 codeword assignment 对 candidate trajectory relation 的影响。
- **VQ-VLA（arXiv:2507.01016）**：是 executable action chunk 的 residual VQ-VAE/action tokenizer；它不压缩 backbone weights，本候选只把它列为 representation-only adjacent baseline，不能把 token vocabulary gain 算作 weight compression。
- **VQVLA/MotionVQ（arXiv:2607.24148）**：motion-aware execution-state precision、centroid reuse 和 custom accelerator；本候选无动态 precision、无 centroid-reuse accelerator，若没有 native A100 kernel 也不报告 accelerator speedup。
- **[RKD（CVPR 2019）](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html) / [QATMA，arXiv:2603.05964](https://arxiv.org/abs/2603.05964)**：RKD 的关系蒸馏与 QATMA 的 Text-anchored Pairwise Similarity Distillation 是直接碰撞；旧检索曾误标后者为 CR-QAT，以核实后的 QATMA 标题为准。目标 anchor 加 pairwise relation 再用于 quantization 的一般原则已存在。必须增加普通 RKD-style VQ（相同 trajectory/goal anchor、更新预算、bytes ceiling），并与 pointwise-only、goal-score-only、$L_abs+L_off$、scalar 和 independent-codebook 比较；若数学目标等价，只能称 WM adaptation，不能声称新目标。

因此唯一可辩护的 provisional difference 是：**固定 packed weight VQ 中，在首 pilot 固定 indices、只更新 shared codebook entries 的条件下，Q model imagined trajectories 的 goal-anchored off-diagonal relation 是否带来可重复的 planner benefit，并且该 benefit 是否依赖 shared VQ codeword coupling。** 这个差异仍可能被尚未完成的 relational/task-aware quantization 检索覆盖。

## 最小 matched-budget pilot

主 anchor 是 local DINO-WM Wall。reference 必须按 FP32 运行；已有 FP32 source/config 与先前 fullprecision evaluation 是 baseline evidence，不是 quantized evidence。pilot 先保持同一 model checkpoint、same planner config 和 same rollout implementation。

CAL、DEV、TEST 按 episode/initial state 分离；首轮是单一 Wall environment，不宣称 task-disjoint 或跨任务泛化：

- **CAL**：只用于 codebook/index fitting；含固定 action probes 和 paired FP/Q imagined trajectories。
- **DEV**：只选择 $lambda_rel$ 是否保留及做一次 implementation sanity check；不更新 $C,q$。
- **TEST**：所有 map frozen 后进行一次最终确认，报告 FP32 reference、first-action agreement、goal score ordering、held-out off-diagonal error、closed-loop return/success。

episode、task、initial state 和 quantizer-fit seed 是统计单位；frames、horizon steps、action probes 和 candidate pairs 不充独立样本数。使用 common random numbers、同一 task initial state 配对和至少两个不同 fit seeds 检查方向；小 pilot 只作方向/机制筛查，不宣称显著性或功效。先定义共同 actual-byte ceiling $C_bytes$，每个 variant 单独报告实际 bytes、codebook/index/metadata breakdown 和 slack $C_bytes-bytes$；不同 overhead 不能被称为天然 same bytes。推荐固定以下最小 map：

1. FP32 reference（非压缩）。
2. 同一 $C_bytes$ ceiling 下的 VPTQ-like/local-MSE additive/product VQ。
3. 同一 ceiling 下的 $L_abs$ only（$lambda_rel=0$）。
4. 同一 ceiling 下的 $L_abs+L_off$。
5. 同一 ceiling、同一 target 的 per-vector scalar quantization。
6. 同一 ceiling、同一 target 的 independent-codebook VQ。

资源允许时再接入 AQLM、QuIP#、RSAVQ adapter；没有可运行 adapter 时写 unavailable，不用名义 bit 数推断结果。relation-only 只作诊断，不作主要 deployment baseline，因为它缺少坐标/goal anchor。

## Falsification 和 stop rules

load-bearing variable 是 $lambda_rel$，包括 $0$ 的 removal control。最小可证伪预测是：如果 off-diagonal relation 是 planning-relevant 且 shared VQ coupling 有作用，则同 bytes 的 $L_abs+L_off$ 在 TEST held-out action-neighborhood/horizon 上相对 $L_abs$、VPTQ-like 和 scalar/independent-codebook controls 有一致的 first-action agreement 与 closed-loop return/success 方向，同时 off-diagonal error 改善；goal-score/diagonal-only 不足以解释该增益。

必须同时运行这些 controls：

- $L_abs$/goal-score-only：检验 off-diagonal 是否只是重复 CEM score。
- pair labels 在 episode、horizon、task/action-magnitude strata 内 shuffle：保持 marginal norms/energy，破坏 pair relation。
- $lambda_rel=0$：检验关系项的独立作用。
- scalar 和 independent-codebook：检验 shared codeword coupling 是否必要。
- 同 bytes random/index perturbation：检验收益是否只是随机 regularization 或 bytes 差异。

以下任一条件触发 no-go 或降级：$L_abs+L_off$ 不超过 $lambda_rel=0$ 与 local-MSE/VPTQ-like；只改善 diagonal/goal score 而 held-out pair/horizon 和 planner endpoint 无改善；stratified shuffle 与原关系项等效；scalar/independent-codebook 同样有效；关系项仅在 CAL 有效；或 codebook update 没有经过 Q model forward。若结果只改善 intermediate geometry 而不改善 action/return，只写作 representation diagnostic。

任何“5%”之类阈值都不作为显著性线。资源 stop bar 只作为主观规划：若最小 matched pilot 在固定的预注册 map、两个 fit seeds 和少量 paired TEST 后没有方向一致性，就停止扩展；不以更多 seeds、更多 rate allocation 或更大 model 追逐结果。

## 真实压缩、fake quant 和算力边界

fake/logical quant 可以验证 Q forward 与 planner behavior，但不能声称 native byte/latency。native deployment 只有在真实 packed loader/kernel 上，报告完整 checkpoint bytes（含 codebook/index/scales/metadata）、peak VRAM、synchronized throughput、control frequency，并与 FP32 及另行定义的 BF16/FP16 conversion baseline 对照时才成立。FP32 是本 anchor 的 numerical reference；codebook FP16 round-trip 需单列 conversion error，不能把它混写成 FP32 结果。没有 custom accelerator 时不报告 MotionVQ 类 speedup。

最多 $4×A100$ 表示两个候选合计的并发上限。VQ pilot 总 cap 规划为 ≤16 allocated A100-hours，包含 collection、拟合、必要对照、评估和失败成本，未测量。先做 A100-40GB 小 batch forward/backward smoke；80GB 只能另行配置，不能作为超时或超预算后的自动 fallback。若显存或最小公平 workload 超过 cap，报告预算不可行并停止。所有未来模型加载、heavy I/O、checkpoint/data transfer、编译和实验必须在真实 PBS allocation 中检查非空 $PBS_JOBID$、实际非-login hostname 与 GPU allocation；本轮没有执行 SSH/PBS，也没有提交集群。新增 paid connector/API budget 假设为 $0$。

## 条件式研究假设

若在完成 relational/task-aware collision review 后仍无直接覆盖，并且在固定 bytes、固定 group map、FP32 reference、严格 CAL/DEV/TEST 与 matched controls 下，$L_abs+L_off$ 能在至少不同 quantizer-fit seeds 对 held-out action neighborhoods/horizons 产生独立 planner endpoint 方向，同时该方向在 scalar/independent-codebook 和 stratified pair-shuffle controls 中消失，则 TR-PVQ 值得进入小规模方法 pilot。否则应把它标作 negative result 或 abandon；当前不声称已 novel、已有效或已 native deployable。
