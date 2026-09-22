# Idea 02｜保留代价，而不保留全部状态

**工作名：Goal-Sufficient Cost Model / Projected State + Tail Cost**  
**适用 baseline：LeWM＋PushT；版本：v0.1；日期：2026-09-22**  
**状态：研究提案，未训练、未验证。** “Sufficient”仅指指定 goal-cost 查询的代数信息充分性，不宣称统计充分性、Markov 充分性或通用世界状态充分性。

> 被压掉的 latent 坐标不必逐维恢复，但它们对当前目标的贡献不能直接消失。保留一部分显式未来表示，再用一个标量预测剩余维度的目标代价。

## 1. 由已有负结果引出的新问题

原 global-PCA 路线要求：小型 basis 能够重建完整 dynamics，或至少在直接删去其余方向后维持 planner。现有结果不支持这种替换。

本方向不再问“tail 能不能被丢掉”，而问：

**对于当前 goal，tail 是否可以不恢复其每个坐标，只恢复它在评分中贡献的一个数？**

这是另一个可检验假说：从 context、actions 和 goal 预测这个标量，可能比预测所有 tail coordinates 更便宜；但标量目标不自动更容易，也不自动带来更小网络。

## 2. 核心等式：固定 goal 下的精确代价分解

### 2.1 基本定义

设 LeWM 未来 terminal latent 为 $z_H\in\mathbb R^{192}$，goal latent 为 $z_g$。选择一个列正交矩阵：

$$
P\in\mathbb R^{d\times r},\qquad P^\top P=I_r,
\qquad R=I-PP^\top.
$$

注意 $R$ 是正交投影算子，满足 $R^\top=R$、$R^2=R$，不需要真的构造 $192\times192$ 矩阵。

对于平方欧氏 goal cost：

$$
\boxed{
\|z_H-z_g\|^2
=
\underbrace{\|P^\top z_H-P^\top z_g\|^2}_{\text{显式保留部分}}
+
\underbrace{\|R(z_H-z_g)\|^2}_{\text{剩余部分，只贡献一个标量}}
}
$$

定义标签：

$$
y_H=P^\top z_H,\qquad
 e_H(c,u,z_g)=\|R(z_H-z_g)\|^2\ge0.
$$

新的小型 predictor 输出：

$$
\hat y_H=f_\theta(c,u_{1:H}),\qquad
\hat e_H=q_\theta(c,u_{1:H},z_g),
$$

最后评分：

$$
\boxed{\hat J(u)=\|\hat y_H-P^\top z_g\|^2+\hat e_H.}
$$

$\hat y$ 与 goal 无关；$\hat e$ 必须考虑 goal。可以共享基础 context/action feature extractor，再为 tail head 加入 goal 信息，但不得让 $\hat y$ 的定义随 goal 变动。

### 2.2 为什么不是恢复一个低维世界状态

$r=32$ 时输出 33 个数字，只能支持指定的目标评分查询。换一个 goal，$e_H$ 往往需要重新预测；换一个 reward、约束或后续动作，不保证这 33 个数字够用。

因此，这更准确地说是一个 **planning surrogate／goal-query model**，不是一个可递归替换原状态的通用 33 维 JEPA dynamics。

第一版应从完整可信 context＋action prefix 直接预测 $y_h,e_h$。不把 $(\hat y,\hat e)$ 接回原 LeWM 当作下一步 latent，也不声称其满足 Markov 性。若仅预测 terminal，训练多个 horizons 仍可作为监督，但部署接口明确只回答相应 horizon 的查询。[S3]

### 2.3 必须保留的交叉项

不能把 tail 定义成 $\|Rz_H\|^2$，因为：

$$
\|R(z_H-z_g)\|^2
=\|Rz_H\|^2-2\langle Rz_H,Rz_g\rangle+\|Rz_g\|^2.
$$

最简单的实现是直接监督完整的 $e_H$，而不是自行近似上述交叉项。

如果 $P$ 来自 mean-centered PCA，可以让当前／目标同时减去同一个训练均值；差分中的均值会消掉。**若 $P$ 是从 delta 拟合的，仍须用正确的完整 $z_H$ 与 $z_g$ 定义标签，不能把 delta 当成 terminal state。**

### 2.4 几个可直接单元测试的边界

| 设置 | 应得到的结果 |
|---|---|
| $r=0$ | 显式部分为空，退化为纯 scalar cost predictor |
| $r=d$，取 $P=I$ | tail 精确为 0，退化为完整 latent predictor |
| exact $y$＋exact $e$ | 完整平方欧氏 cost 的代数等价控制 |
| exact $y$＋任意 candidate-independent 常数 $k(c,g)$ | 仅整体平移 cost，不能改变同一 bank 的排序 |
| 非正交 $P$ 直接套公式 | 一般错误；先 QR 正交化或使用正确投影算子 |

所有关于精确分解的结论都以本地 criterion 是相应二次度量为前提。固定加权平方误差可以先变换到该度量的坐标再构造正交投影；非二次 goal cost 需要新推导。不要未经说明更改 criterion。

## 3. 为什么“一个标量”仍可能很难预测

CEM 关心的是候选之间很小的相对代价差，而不是 $e_H$ 的绝对数值是否总体预测得像。若 tail 的均值很大、候选间差异很小，一个只拟合均值的网络可能有不错的相对 MSE，却无法帮助 elite selection。

因此除普通误差外，必须检查同一 context-goal candidate bank 内的：

$$
e_i-\overline e,\qquad
\hat e_i-\overline{\hat e},
$$

以及 elite 边界附近的 tail error、cost gap 与 proposal update discrepancy。这里去均值是诊断，不意味着可以在部署中使用完整-label 均值。

代价误差满足以下直接展开：令 $\epsilon_y=\hat y-y$，$g_P=P^\top z_g$，则：

$$
\hat J-J
=2\langle y-g_P,\epsilon_y\rangle+\|\epsilon_y\|^2+(\hat e-e).
$$

所以：

$$
|\hat J-J|
\le2\|y-g_P\|\,\|\epsilon_y\|+\|\epsilon_y\|^2+|\hat e-e|.
$$

**有了精确 tail 也不能修复所有显式预测错误；准确预测 $y$ 也不能抵消错误 tail。** 若误差恰好相互抵消，不宜依赖这种未经约束的补偿维持泛化。

## 4. 最小模型与训练标签

### 4.1 第一版固定 encoder 和 projection

冻结现有 LeWM encoder。$P$ 仅使用训练 split 拟合；可以先重用训练范围内的 model-PCA basis，同时增加 orthogonal random basis 与 $P=I$ 控制。不需要先证明 $P$ 能重建全部 transitions，因为未保留部分由 $e$ 负责。

先测试 $r\in\{0,32,64,192\}$，有正信号后再细化。不要为每个测试 goal 或 held-out context 事后拟合 $P$。第一版不增加 learned projection、dynamic rank 或 mixture，以免同时改变多个假说。

网络可先用相同大小的 context/action trunk，输出 $\hat y$；tail head 读取 trunk features 与 goal embedding，输出非负 $\hat e$。例如用带训练集尺度 $s_e>0$ 的 softplus 保证非负。$r=d$ 时直接将 tail 设为 0，不靠 softplus 去逼近这个已知边界。

### 4.2 Teacher labels 与真实标签的区别

若训练目标是模仿现有 LeWM，对每个训练 $(c,u)$ 用完整 teacher 从可信 context rollout，得到 $z_H^T$，再据此计算 $y_H^T,e_H^T$。部署不再调用 teacher。

若训练目标是预测环境，$z_H$ 必须来自实际执行 $u$ 后的编码结果。两种标签可以作为独立实验；不能混在一起后不说明目标已经改变。

对于一个 terminal latent，可以配多个训练 goals 重算 tail 标签，而无需重新运行相同 trajectory 的 dynamics。但所有 goal 来源仍须遵守数据划分和任务协议。

### 4.3 Goal sampling 是一个容易产生伪结果的地方

不要让每个样本都以自身 terminal observation 为 goal，否则 $J=e=0$，问题会退化。训练 goals 应覆盖部署中的目标关系：例如在同一训练 episode 的不同合法 offsets 取目标，并提供非零距离的 goal-action 组合；跨 episode goal 可作为单独的泛化条件，不默认都可达。

建议为每个训练 $(c,u)$ 采样 4 个不同 goals 作为初始配置，再与单 goal 训练比较。Goal-disjoint 与 context-disjoint 的评估分别报告。测试 goal 不得用于 projection 拟合、loss 尺度估计或 tail calibration。

### 4.4 训练目标：先分别监督两部分

基础损失为：

$$
\mathcal L
=
\lambda_y\frac{\|\hat y-y\|^2}{\max(r,1)}
+
\lambda_e\frac{(\hat e-e)^2}{s_e^2+\epsilon},
$$

其中 $s_e$ 由训练集估计并冻结；$r=0$ 时删除第一项，$r=d$ 时删除第二项。初始化可用 $\lambda_y=\lambda_e=1$，但这只是起点，最终在 validation 冻结。

先不用 rank loss，也不让 total-cost loss 独自驱动所有 head。后续若加入 $\mathcal L_{\rm cost}$ 或 elite-focused loss，要让纯 scalar baseline 使用相同数据和相应目标，以免将监督优势归因于分解结构。

## 5. 验证实验设计

### E0｜代数恢复与标签审计：不训练新模型

在旧 candidate banks 和新开发 contexts 上，以完整 teacher terminal latents 计算 exact $y,e$。比较四种 cost：

1. 完整 $J_T$。
2. exact projected cost，tail=0。
3. exact projected cost＋exact tail。
4. exact projected cost＋每个 bank 的同一常数 tail。

第三种应恢复第一种，FP64 下仅有数值误差；第四种不应改变第二种排序。记录第三种在完整 CEM 中的数值控制效果，而不是只检查 MSE。

**这个实验全部依赖完整未来，只验证代数和遗漏来源，绝不是加速结果。** 它与旧 delta reconstruction 可能不是同一个近似插入位置，不能把两者的差别直接归因于尾部标量有效。

### E1｜先问 tail 标量能否被廉价预测

冻结 $P$ 和 encoder，训练一个小型 $q(c,u,g)$。在 held-out contexts 上做以下因子分离：

| projected 部分 | tail 部分 | 用途 |
|---|---|---|
| exact $y$ | 0 | 缺少 tail 的误差基准 |
| exact $y$ | learned $\hat e$ | 隔离标量预测的可行性 |
| learned $\hat y$ | exact $e$ | 隔离显式状态预测的误差 |
| learned $\hat y$ | learned $\hat e$ | 实际可部署候选 |
| exact $y$ | exact $e$ | 完整 teacher 的代数控制 |

任何带 exact 项的设置只用于 oracle diagnosis，计时不得当作部署速度。尤其要检查 learned tail 在 method-induced candidates 上是否仍然有效，而不只是原始固定 banks。

**早停信号：** 在 exact $y$ 的乐观条件下，廉价 tail head 仍无法改善 elite/proposal 更新；或为了达到足够的 tail 精度，网络成本接近完整 predictor。此时不急于构造更复杂联合模型。

### E2｜必须与纯 cost head、完整小模型正面对照

所有方法使用同一 frozen encoder、相同 prefix context 和相同训练数据。至少比较：

| 编号 | 方法 | 输出 | 公平比较的目的 |
|---|---|---|---|
| B0 | 优化后的 LeWM | 原 rollout | 实际参照 |
| B1 | 小型 full-latent prefix predictor | 192 维 | 仅缩小 predictor 是否已经足够 |
| B2 | 纯 cost predictor | 1 维，等价 $r=0$ | 分解是否优于直接预测任务量 |
| B3 | Projected-only | $r$ 维 | 证明 tail 是否有必要 |
| B4 | Projected＋tail | $r+1$ 维 | 本提案 |
| B5 | 相同 trunk＋完整输出＋相同辅助监督 | 192 维及训练期辅助 head | 区分“更好的训练监督”和“部署压缩” |

B5 可在部署只保留完整输出；B4/B5 的 training-only heads 和额外数据成本均需报告。

做小型 width/depth 扫描，按在线延迟而非只按输出维数对齐。若 B2 同样快且更好，应优先承认更简单的 cost-only 模型已经解释收益，而不是强行维护 $r+1$ 结构。

### E3｜完整 adaptive CEM

使用共用协议的 300／30／30，第一版不加候选精算或 fallback。特别记录 early/middle/late iterations 的 tail bias、elite margin、proposal 均值和方差变化。

分别进行 full-driven 与 method-driven shadow evaluation。若 $q$ 在 full-driven candidates 上表现好，却在自己诱导的 candidates 上严重低估代价，说明 adaptive planner 正在暴露／利用 surrogate 的误差；不能用整体 Spearman 隐藏该问题。

### E4｜新目标与真实环境

在 episode-disjoint contexts、未用于训练的 goal 组合和不同初始距离上评价任务质量。执行本地 planner 实际返回的整个动作前缀，并保持环境预算一致。

报告 failures 的分组：无接触、接触前后、物体旋转、临近目标等。若用 simulator state 分组，这些状态仅用于离线分析，不作为部署额外输入。具体分组阈值在 validation 确定，不能事后挑有利子集。

### E5｜是否真的减少了主要计算

将一个大网络的 192 维 output head 改成 33 维，通常只减少最后少量计算，不能预先宣称大幅加速。

真正的 GO 证据必须是：在任务质量相当时，B4 所需 trunk 足够小，从而完整 solve 明显更快；或者在相同速度下优于 B1/B2。记录 goal-conditioned head 的每候选额外成本和 prepare 成本。

## 6. 可证伪性与解释模板

| 结果 | 解释和下一步 |
|---|---|
| exact tail 能恢复 cost，learned tail 不行 | 恒等式正确，但标量难学；不能声称已得到可部署信息压缩 |
| B4 优于 B3，但不优于 B2 | tail 有价值，但显式 projected state 未证明必要 |
| B4 更准，但需要同样大的 trunk | 可能是训练／质量方法，不是 efficient-WM 的充分结果 |
| B4 保留 fixed ranking，但 adaptive CEM 失败 | 重复了已知缺口；不能通过 candidate gate 停止验证 |
| B4 在闭环不劣化并明显更快 | 支持 goal-query 压缩；仍不支持通用 Markov state 压缩 |
| 只对训练 goals 有效 | goal-conditioned 泛化不足，不是通用目标规划模型 |

明确 NO-GO 包括：被 matched scalar head 或 full-latent small model 在质量与速度上支配；尾部误差在自适应搜索中持续失控；或者主要成本没有下降。

## 7. 最小实施清单

```text
[ ] 固定本地 criterion，验证是否确为 terminal squared Euclidean cost
[ ] 仅用训练 split 得到 P，并测试 P^T P ≈ I
[ ] 实现 exact y / exact tail 标签，完成 r=0、r=d、constant-tail 控制
[ ] 确认 goal sampling 不退化成 terminal=goal
[ ] 先训练廉价 tail head，配 exact y 做乐观诊断
[ ] 训练 scalar-only、full-latent small、projected-only、projected+tail
[ ] 跑完整 adaptive CEM 与 method-induced distribution 诊断
[ ] 跑 paired 环境任务，记录执行前缀而非只看 first action
[ ] 报告 matched-latency Pareto，而不只报 output compression ratio
```

## 8. 创新边界和最小论文命题

“只学习规划需要的信息”已有 value equivalence 等明确思想来源。[S5] 本提案不等价于那些理论中的 Bellman value equivalence：这里是有限 horizon 的 goal-distance 查询及其特定分解。

可以验证的命题是：**将未来 latent 的显式几何部分与其余维度的 goal-dependent cost contribution 分开学习，能否使更小的预测器保留自适应规划质量。**

需进一步查新：`value-equivalent world model`、`goal-conditioned cost prediction`、`terminal cost distillation`、`task-sufficient representation`、`predictive state abstraction`、`residual value head`。不能仅凭“tail 变成一个标量”就宣称首创。


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

外部来源核对日期：2026-09-22。分解恒等式由本文直接推导；learned tail 是否可廉价预测、是否能保留 CEM 和任务质量，尚无本项目实验结果。

**[S1] LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.** 论文与官方实现。用于 baseline 的 CLS latent、预测接口和 terminal cost；实际实验以本地锁定 commit 为准。  
[论文](https://arxiv.org/abs/2603.19312) · [官方仓库](https://github.com/lucas-maes/le-wm) · [jepa.py](https://github.com/lucas-maes/le-wm/blob/main/jepa.py)

**[S2] LeWM 官方 PushT evaluation config.** 仅用于配置核对；`main` 是可变链接，应在实施时替换为 commit permalink。  
[config/eval/pusht.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/eval/pusht.yaml)

**[S3] Fast LeWorldModel.** Action-prefix prediction 是既有工作，不应把直接／并行预测未来 horizon 当作本提案的新贡献。  
[论文](https://arxiv.org/abs/2606.26217) · [项目页](https://fast-lewm.github.io/) · [官方仓库](https://github.com/Yuntian-Gao/Fast-LeWorldModel)

**[S4] LpWM: A Case for Sparse Representations in World Models.** 用于比较“通过表示几何降低 predictor 复杂度”的研究定位；不等同于另一篇同名缩写的 Latent Particle World Models。  
[论文](https://arxiv.org/abs/2608.22764)

**[S5] The Value Equivalence Principle for Model-Based Reinforcement Learning.** 用于“模型只需保留规划相关信息”的思想归属；其 Bellman-operator 定义不自动为本方案提供误差保证。  
[论文](https://arxiv.org/abs/2011.03506)
