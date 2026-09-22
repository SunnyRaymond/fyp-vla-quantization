# Idea 01｜编译式世界模型：把状态计算移出候选循环

**工作名：Compiled World Model / Context-Compiled Action Model**  
**适用 baseline：LeWM＋PushT；版本：v0.1；日期：2026-09-22**  
**状态：研究提案，未训练、未验证。** 本文的“编译”指将当前 context 转换成可重复查询的小型函数，不是编译器代码生成或 `torch.compile`。

> 大模型先把当前世界转换成一个小型动作响应函数。一次规划中的所有 candidates 只调用这个小函数，不再重复运行完整的状态—动作联合网络。

## 1. 核心假说与潜在贡献

普通动作条件预测为：

$$
\hat z_h=G_\theta(c,u_{1:h}),
$$

其中 $c$ 包含当前可观测历史及必要的历史动作，$u_{1:h}$ 是未来动作前缀。候选动作不同，但 $c$ 在一次 fixed-observation solve 内相同。

本方向尝试改成：

$$
\theta_c=H_\psi(c),\qquad \hat z_h=g_{\theta_c}(u_{1:h},h).
$$

**待验证假说不是“latent 本来很低维”，而是“复杂的 context 计算可以与廉价的 action 查询计算分离，并且跨 candidates 摊销”。**

与 Fast-LeWM 的区别：action-prefix 接口消除时间方向的递归依赖；本提案的候选贡献应体现在 **candidate 方向的 context 计算共享**。为避免混淆，必须与同样使用 prefix 接口、但没有这种结构分离的模型比较。[S3]

与 global PCA 的区别：不要求所有状态共享一个固定输出 basis。即使保留完整 192 维容量，只要每个候选的计算明显减少，方向仍可能成立。

## 2. 最小模型：先用一个可精确分析的分离形式

### 2.1 数学定义

取：

$$
\boxed{\hat z_h=b_h(c)+B_h(c)\phi_\eta(u_{1:h},h)}
$$

其中：

| 变量 | Shape | 谁生成／何时计算 |
|---|---|---|
| $c$ | 由本地 history 定义 | 固定观测的完整 context，不只假定一个 frame |
| $b_h(c)$ | $d$，当前 $d=192$ | context network，一次 solve 中计算一次 |
| $B_h(c)$ | $d\times r$ | context network，一次 solve 中计算一次 |
| $\phi_\eta(u_{1:h},h)$ | $r$ | 小型 action network，每个 candidate 查询 |
| $z_g$ | $d$ | 固定 goal 的 encoder 输出 |

第一版使用冻结的原 LeWM encoder；$H_\psi$ 可先用小型 MLP 处理完整历史 latent，输出 $b_h,B_h$。action network 可用 1–2 层 MLP 处理带 horizon mask 的 padded action sequence。固定 H=5 时先用这个最简单版本，不急于加 Transformer、mixture、router 或 fallback。

训练多个 horizons 时，$H_\psi$ 可一次输出所有 $h$ 的系数；若 planning 只使用 terminal cost，部署时仅准备 $h=H$。不能用只预测终点的计算成本去宣称对任意轨迹约束都适用。

### 2.2 结构约束必须真正成立

- $H_\psi$ 输入中不能包含 candidate-specific 未来动作，否则不能一次生成后供所有候选共享。
- 基础版本的 $\phi$ 不接收 $c$ 或 goal。若后来加入 context-conditioned action branch，它是新的变体，成本和解释均要单独报告。
- 预测 $h$ 时不能读取 $h$ 之后的动作。对 action padding 改动做不变性测试。
- $b,B$ 不接收 goal；同一状态和动作的预测不能随任务目标改变。goal 只在 scoring 编译中出现。
- 本地动作表示可能是归一化绝对坐标而非位移，不能未经核查直接使用 $u_t-u_{t-1}$。

### 2.3 不必把低 rank 设成先决条件

首先测试 $r\in\{32,96,192\}$ 的粗粒度版本；若有信号，再扩展 16/64 等中间容量。$r=192$ 不等于函数表达无限充分：虽然输出没有被限制在一个低于 192 维的子空间内，**共享 action feature map 的可分离结构仍是一个限制**。

$b,B$ 的输出头也可能很大。例如一个 hidden→$(192r)$ 的 dense head 参数并不小，只是在线只执行一次。报告总参数与每候选成本，不得只报 action network 的参数量。

## 3. 第二个收益：把目标代价也提前编译

若本地 criterion 为：

$$
J(u)=\|\hat z_H-z_g\|_2^2,
$$

令 $v=\phi(u)$，$b=b_H(c)$，$B=B_H(c)$，则：

$$
\begin{aligned}
J(u)&=\|b+Bv-z_g\|^2\\
&=v^\top Qv+2p^\top v+\kappa,\\
Q&=B^\top B,\quad p=B^\top(b-z_g),\quad \kappa=\|b-z_g\|^2.
\end{aligned}
$$

因此部署可只保留 $(Q,p,\kappa)$，每个 candidate 无需构造 192 维 future latent。

**代数重写对“这个已训练模型”的 quadratic cost 是精确的；它不表示这个模型等于原 LeWM，更不表示等于真实环境。**

对于固定正半定二次度量 $M$，可用 $Q=B^\top MB$、$p=B^\top M(b-z_g)$。若 criterion 有 prediction-dependent normalization、cosine、min-over-time、障碍约束或其他非二次项，应重新推导；不能直接套用上述式子。纯 MSE 的常数比例也应保留，尤其其他 cost 项存在时。

### 3.1 速度并非总能由 cost 编译获得

对 $N$ 个候选：显式构造 latent 约为 $O(Ndr)$；二次型求值约为 $O(Nr^2)$，另有一次 $O(dr^2)$ 的 $Q$ 构造成本。因此 $r<d$ 时可能有额外收益；$r=d$ 时未必，实际小矩阵 kernel 也可能不占优势。

这一点应做消融，不应先验假定“编译 cost”总是更快。核心跨候选摊销即使成立，第二项收益也可能不成立。

### 3.2 数值精度测试

用相同的随机 $b,B,v,z_g$ 比较显式 cost 与编译 cost：先 FP64，再 FP32，包含 near-goal、小 margin 和大 feature norm 的情况。记录 max absolute/relative error 及排序差异。

$\kappa$ 与其他项可能发生数值相消；数学等价不代表逐 bit 等价。不要直接把略为负的 quadratic cost 全部 clamp 成 0 后宣称排序不变。必要时提高 accumulation precision，但其成本必须计入。

## 4. 计算图与复杂度账本

```text
一次观测/目标刷新：
    c ← encode_context(observed_history)
    z_g ← encode_goal(goal)
    b, B ← context_compiler(c, horizon)
    Q, p, kappa ← compile_quadratic_cost(b, B, z_g)

每轮 CEM：
    U ← sample_from_current_proposal()
    V ← small_action_network(U)
    costs ← quadratic_score(V; Q, p, kappa)
    update_proposal(costs, U)
```

设 $I=30,N=300$，有 $IN=9000$ 次 sequence queries。旧 autoregressive baseline 在五步设定下还涉及多步 sample-transition evaluations，但这些是 batched computation，不是 45000 次独立 GPU 调用。

用实际测量定义：

$$
T_{\rm new}=T_{\rm encode}+T_{\rm prepare}+I\,[T_\phi(N)+T_{\rm score}(N)+T_{\rm CEM}],
$$

与同样经过无损缓存优化的 baseline 比较。大 context network 的成本不能省略；若 $T_{\rm prepare}$ 超过省下的 predictor 时间，方向没有在线收益。

## 5. 训练：先验证结构，不先堆 loss

### 5.1 第一阶段：固定表示、相同目标

冻结 encoder，所有 predictor 使用相同训练 contexts 和动作分布。真实标签与 teacher labels 分开报告：

$$
\mathcal L_{\rm pred}=\frac1H\sum_{h=1}^H
\frac1d\|b_h(c)+B_h(c)\phi(u_{1:h},h)-z_h^{\rm target}\|^2.
$$

先仅用该损失加常规训练稳定化。若原目标足够好，不急于加入 ranking loss；若后续加入，必须让匹配的 prefix baseline 使用相同训练目标。

每个训练 context 应对应多个不同 action sequences，否则无法有效约束 cross-candidate sharing。仅有一条 expert continuation 时，可先做 observed-data pilot，但不能据此认定共享的 action response 已被学到。需要 teacher branches 或仿真分支时按相同预算提供给所有对照。

### 5.2 第二阶段：仅在有正信号时联合训练 encoder

固定 encoder 版本用于判断 frozen LeWM 是否支持分离。联合训练版本用于判断是否能形成更适合分离的表示；必须保留原 JEPA 防坍塌机制，并与相同联合训练预算的 dense prefix 模型比较。

不直接将 joint-training 的提升归因于“当前 checkpoint 存在可提取结构”。这是两个不同命题。

## 6. 验证实验设计

### E0｜接口和代数单元测试：无需模型训练

**目标：** 确保所谓加速没有来自输入减少、目标变化或数值错误。

验证 action-prefix causal mask、batch/chunk 一致性、history 对齐、goal 改变只更新 $p,\kappa$、context 改变会更新 $b,B$，以及显式／编译成本一致性。将同一已训练分离模型的 context branch 故意重复执行与只执行一次进行比较：输出应相同，计时区别直接衡量摊销本身。

**停止条件：** 存在未来动作泄露、跨候选 state 混用、cost 定义不一致时，不进入模型效果评估。

### E1｜分离假说的容量测试

训练如下最小对照，使用相同冻结 encoder、同一 action-prefix interface、同样的数据：

| 编号 | 模型 | 要隔离的因素 |
|---|---|---|
| B0 | 原始 LeWM＋无损缓存 | 当前实际 baseline |
| B1 | 小型 dense prefix predictor $G(c,u)$ | “只换成小模型／直接预测”能做到多少 |
| B2 | 分离模型，$B$ 不随 context 改变 | 固定 basis 的限制 |
| B3 | 分离模型，$b(c),B(c)$，显式生成 latent | context-dependent 分离结构本身 |
| B4 | 与 B3 完全相同权重，编译 quadratic cost | cost 编译的纯计算收益 |

先用 B1/B3 的小规模参数量与在线延迟扫描，不以“总参数完全相等”掩盖 B3 的一次性大 head，也不以“便宜分支参数少”忽略它。报告两个轴：总参数／训练成本，以及每 solve 在线成本。

**主要结果：** 多步 latent loss、action response error，以及开发 contexts 上的 CEM update 差异。若只有 $r=192$ 才有合理质量，但在线仍明显更快，保留；不强迫 low-rank 叙事。

### E2｜跨候选共享能否泛化

训练候选覆盖 random、数据附近动作，以及训练 contexts 上 CEM 的不同阶段。验证集使用新 episodes；另设 proposal-shift 测试：不同标准差和靠近 elite 的候选。

做 context-swap negative control：故意把 $B(c_1)$ 用在 $c_2$，确认模型确实依赖 context，而不是 action branch 记忆平均后果。这个错误模型只是诊断，不能作为弱基线来证明优势。

**GO 信号：** context-dependent 版本优于固定 basis，并在相同在线预算下不被 dense prefix 支配。单看训练误差不足以通过。

### E3｜完整 adaptive CEM 与闭环任务

采用后附共用协议。必须保留完整 30 轮 adaptation；先不引入 fallback、候选精算、动态 rank 或减少 iterations，以免贡献无法归因。

特别比较 B3/B4：如果训练模型相同而 action 发生变化，先检查浮点 cancellation 和 elite tie，不把这种变化解释成模型能力差异。

### E4｜机制验证和成本扩展

固定模型，测 $N\in\{30,100,300,1000\}$，保持任务条件明确；主结果仍使用官方预算。记录准备成本与 query 成本交叉点。检查改变 goal 但不改变 context 时的重用成本，及真实新观测到来后的重新准备成本。

预期曲线只是理论假说：context 相关成本不随 N 线性重复；若实际 kernel 或 solver 开销遮蔽收益，应如实报告。

## 7. 关键消融与可证伪结论

| 观察 | 可得结论 | 不能声称 |
|---|---|---|
| B3 好于 B2 | state-dependent action response 比固定输出 basis 有帮助 | 已证明所有 context 可用极低 rank |
| B3 与 B1 一样好但明显更快 | 分离摊销可能有实际价值 | “低维表示”是收益来源 |
| B4 比 B3 更快且输出等价 | quadratic cost compilation 有独立收益 | 原 LeWM cost 被精确保留 |
| 只在 teacher labels 上好 | 可压缩 teacher 的行为 | 真实 PushT 任务一定不劣化 |
| 对 unseen contexts 或 adaptive CEM 失效 | 当前 factorization 泛化不足 | 应立即堆 mixture/fallback 才能算成功 |

**具体 NO-GO：** $r=192$ 仍被 matched dense prefix 明显支配；或需要近似同等昂贵的 action branch 才能保任务；或 prepare＋query 总成本不降。此时终止“当前分离形式”，而不是将其改成任何形式都算成功的泛化 hypernetwork。

## 8. 最小实施清单

```text
[ ] 锁定 local baseline 与精确缓存版本
[ ] 实现接口 prepare(context, goal, horizon) / score_many(actions, prepared)
[ ] 用合成 tensor 验证 compiled cost 等价和 dtype 敏感性
[ ] 冻结 encoder，训练 B1 与 B3，先 r=32/96/192
[ ] 增加 B2；从 B3 权重直接构造 B4，不重新训练
[ ] 跑固定候选诊断，再跑完整 CEM，最后 paired 环境执行
[ ] 完整统计 prepare/query/solver 的时间和显存
[ ] 仅在机制通过后考虑 encoder joint training
```

实现中的函数名仅是接口建议，不代表现有仓库已经有这些 API。

## 9. 创新边界和最小论文命题

Context-generated functions 与 HyperNetwork／operator learning 有历史联系；DeepONet 已有 branch–trunk 分离机制。[S5] 不应把“两个网络的输出相乘”作为首创。

可检验的贡献命题是：**在固定观测、多候选的视觉规划中，通过 context–action 分离和精确 cost contraction，将大部分状态相关计算从候选循环移出，并在完整自适应搜索与闭环任务上保持质量。**

需进一步查新的关键词：`context-conditioned dynamics`、`hypernetwork world model`、`separable dynamics`、`operator learning control`、`amortized multi-query planning`、`successor features goal planning`。本文只给出已核对的近邻，不代表穷尽查新。


## 研究起点：本文件使用的已知证据

**当前 baseline：LeWM＋PushT。** 不使用此前的 DINO-WM／Wall 设定。以下结果来自研究者在本次对话中提供的实验摘要，尚未重新检查原始日志、代码或 checkpoint。

| 已完成实验 | 已报告结果 | 对本方向的约束 |
|---|---|---|
| Per-trajectory oracle SVD | 每条候选五步 delta 为 $5\times192$；rank 3/4 的 median retained energy 为 0.9848/0.9967；rank 3/4 通过 fixed-candidate gate | 只是已知未来后拟合的短轨迹结构；矩阵秩天然不超过 5，不能当作可部署压缩器 |
| Reusable global PCA | rank≤64 不通过冻结的 held-out reconstruction gate | 不再以“一个固定小型线性子空间覆盖所有 transitions”为出发点 |
| Fixed candidate-bank ranking | model-PCA rank 64/96 通过；argmin agreement 分别为 87.50%/93.75% | 固定候选排序只作为诊断，不作为最终 GO 标准 |
| Official adaptive CEM | 300 candidates、30 iterations、top-30；rank 64/96 的 4 个 paired cases 均未通过 first-action fidelity gate；rank 192 零漂移 | 新方法必须评估真实自适应搜索；不能以很高 Spearman 替代 planner 验证 |

原有 first-action gate 继续原样记录：normalized L2≤0.15 且 absolute difference≤0.25，**归一化和 absolute difference 的具体计算沿用原实验代码，不能自行改成另一个范数。** 新增 task-quality gate 是另一项研究目标，不得用它改写旧实验的 FAIL。

本文件中的所有模型名称均为**工作名**；实验预算、超参数和 GO/NO-GO 阈值均为**建议预注册的设计值**，不是已有实验结果或公认标准。


## 共用验证协议：避免重复上一轮的误判

### P1. 先锁定本地实现，而不是凭论文默认值推断

记录 checkpoint SHA256、代码 commit、数据版本、encoder/projector/pred_proj、action normalization、history 长度、candidate tensor 的真实 shape、优化变量维度、criterion、执行动作前缀、仿真步长、precision、GPU 和依赖版本。

公开 LeWM `jepa.py` 使用 CLS 特征、递归 latent rollout，并以最后预测 latent 与 goal latent 的平方误差之和评分。公开 PushT 配置写有 `horizon: 5`、`receding_horizon: 5`、`action_block: 5`。[S1][S2] **这不是对本地实现的认证。** 特别核查 action block 如何转换成实际执行动作、历史动作是否被包含在 candidate tensor 中；不要直接把 horizon×2 当作优化维数，也不要默认只执行 first action。

完整 LeWM 的无损缓存版本应成为速度基线：固定观测／目标的确定性编码在一次 solve 内只做一次，所有方法使用相同预处理和缓存条件。先通过成本和输出一致性测试，再进行计时。状态或目标改变后相应缓存失效。

### P2. 数据划分和训练标签必须可追溯

按 **parent episode** 划分 train／validation／final test，再生成 contexts、candidate banks、counterfactual branches 和 goals。同一 parent episode 的不同片段、相同 context 的不同 candidates，以及复位后得到的所有分支不能跨 split。

已有 8 个 held-out contexts 和 4 个 planner cases 已参与过方向选择，今后作为开发／回归测试集使用，不再包装为未接触过的最终测试集。每个样本至少保存：

```text
parent_episode_id, context_id, goal_id, split, source_type,
context_history, action_sequence, horizon, action_normalization_id,
target_latent_or_cost, teacher_checkpoint_id, generation_seed
```

真实观察标签仅对应实际执行过的动作。未执行的候选必须来自正确复位后的仿真分支，或明确标注为 frozen-teacher prediction；不能把数据集中随后出现的真实帧当作任意候选动作的后果。

teacher 数据生成时间、仿真交互次数、额外训练数据量均需报告。测试阶段生成完整标签用于**离线诊断**可以，但不能混入部署路径或从速度账本中偷偷删除。

### P3. 三层评估，最后一层不可省略

**第一层：同一批候选的诊断。** 记录 latent/cost 误差、Spearman、top-30 overlap、argmin agreement，同时记录第 30/31 名代价间隔、elite 动作均值／标准差差异。低成本诊断可以用 32 个开发 contexts，每个包含初始、中期、后期 proposal 的候选；这些是建议规模，不是显著性保证。

**第二层：完整 adaptive CEM。** 保持原来的 300／30／30、初始化、action clipping、warm-start、方差下限、平滑、最终返回规则和执行前缀。预生成共享随机噪声，而不仅仅设置相同 seed。每一方法随后可以访问不同的 proposal 分布，这正是需要评估的现象。

记录 full-driven 和 method-driven 的双向 shadow traces。对相同候选计算两套 costs，但 shadow 结果不得改变 driver 的更新。核心日志包括：

```text
iteration, candidate_ids, elite_ids, elite_overlap,
mu_full, mu_method, sigma_full, sigma_method,
elite_boundary_gap, cost_error_quantiles, timing_breakdown
```

mean 差异可用当前 proposal 的标准差归一化：

$$
d_{\mu,j}=\left\|\frac{\mu^{\rm method}_{j+1}-\mu^{\rm full}_{j+1}}
{\max(\sigma_j,\sigma_{\rm floor})}\right\|_2.
$$

对同一候选集合，仅一个 elite 从 $u_p$ 换为 $u_q$，不带平滑的 mean update 就变化 $(u_q-u_p)/30$；若有平滑，应使用本地实现的精确更新。不能因为 cost 差很小就假定 proposal 差很小。

**第三层：paired 环境执行。** 从同一环境状态、goal 和预算启动，评价完整实际执行前缀及闭环任务。报告官方成功判据、任务分数／最终目标误差和失败类型。用完整 baseline 从可信观测前缀重新评价选出的完整动作序列：

$$
R_{\rm teacher}=J_T(u_{\rm method})-J_T(u_T).
$$

该量可以为负；teacher planner 不是真实最优解，teacher cost 也不是真实任务质量。

先用少量起点排查实现错误，再进行确认实验。一个可预算的起点是 **100 个新的 start-goal cases、3 个训练 seeds、每个 case 3 个 CEM noise seeds**；数据不足时报告实际规模。置信区间按 parent episode 聚类，训练 seed 的差异另外报告，不能把每轮 300 candidates 当作 300 个独立任务样本。100 cases 不自动保证足够的统计功效；必要时结论为 INCONCLUSIVE。

### P4. 预注册双轨判断

| 判断轨道 | 建议规则 | 解释 |
|---|---|---|
| Action fidelity | 继续报告已有 normalized L2／absolute difference gate | 不通过时，不能声称“行为等价替换” |
| Task-quality non-inferiority | 预先选主要任务指标；成功率差的配对 95% CI 下界大于 −5 个百分点 | −5pp 是可修改后冻结的研究容忍度，不是天然正确阈值 |
| 实际效率 | 对无损优化后的 baseline，完整 solve 的 median speedup≥1.5×，且 p95 不劣化 | ≥2× 可作为追求目标；不足时报告真实 Pareto 结果 |
| 机制成立 | 相同数据、encoder 条件和在线预算下，优于匹配的简单对照 | 仅胜过较慢的原始实现不足以证明想法有效 |

这些标准在 final test 前冻结。未满足样本精度时用 INCONCLUSIVE；明确的模型质量崩溃、没有速度收益或被更简单基线支配时用 NO-GO。允许报告“task-quality GO / fidelity FAIL”，但必须解释这不是原 planner 的精确替换。

### P5. 计时与复现

区分 `encode / prepare-or-compile / dynamics-or-surrogate / criterion / solver / transfers`。计时包含编译、Jacobian、缓存刷新、矩阵生成和所有部署所需校验；不只报便宜分支。报告单次完整 solve 的 p50/p95、峰值显存、模型与在线参数量、训练成本。

固定硬件、batch、precision 和 runtime 优化条件；先充分 warm-up，GPU 同步后计时；单独报告一次性 JIT/graph compile 与稳态时间。重建／loss 数值比较优先用 FP32 reference，代数等价单元测试先用 FP64，混合精度作为独立实验。

最终提交一张任务质量—实际延迟图，以及以下结果表；不预填预测结果：

| Method | Train seed | Task metric ± CI | Teacher-cost gap | First-action fidelity | Solve p50/p95 | Speedup | Peak memory | Verdict |
|---|---|---|---|---|---|---|---|---|
| Optimized LeWM | 待测 | 待测 | 0（reference） | reference | 待测 | 1.00 | 待测 | reference |
| Matched simple baseline | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Proposed mechanism | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 参考资料与事实边界

外部来源核对日期：2026-09-22。本文的模型定义、复杂度分析和实验方案是提案；下面来源用于 baseline 与既有思想的归属，不构成提案有效性的证据。

**[S1] LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.** 论文与官方实现。用于 baseline 的 CLS latent、预测接口和 terminal cost；实际实验以本地锁定 commit 为准。  
[论文](https://arxiv.org/abs/2603.19312) · [官方仓库](https://github.com/lucas-maes/le-wm) · [jepa.py](https://github.com/lucas-maes/le-wm/blob/main/jepa.py)

**[S2] LeWM 官方 PushT evaluation config.** 仅用于配置核对；`main` 是可变链接，应在实施时替换为 commit permalink。  
[config/eval/pusht.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/eval/pusht.yaml)

**[S3] Fast LeWorldModel.** Action-prefix prediction 是既有工作，不应把直接／并行预测未来 horizon 当作本提案的新贡献。  
[论文](https://arxiv.org/abs/2606.26217) · [项目页](https://fast-lewm.github.io/) · [官方仓库](https://github.com/Yuntian-Gao/Fast-LeWorldModel)

**[S4] LpWM: A Case for Sparse Representations in World Models.** 用于比较“通过表示几何降低 predictor 复杂度”的研究定位；不等同于另一篇同名缩写的 Latent Particle World Models。  
[论文](https://arxiv.org/abs/2608.22764)

**[S5] DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators.** 用于 branch–trunk／函数分离思想的既有工作边界，不等同于本提案在 PushT 上已经被验证。  
[论文](https://arxiv.org/abs/1910.03193)
