# Idea 03｜拉直动作响应，而不是时间轨迹

**工作名：Action-Response Straightening / Counterfactual Action Curvature Regularization**  
**适用 baseline：LeWM＋PushT；版本：v0.1；日期：2026-09-22**  
**状态：研究提案，未训练、未验证。** 目标是训练出适合共享局部计算的 dynamics，而不是假定当前冻结模型已经线性。

> 固定当前世界，考察附近不同动作序列会产生的未来。若这些反事实未来在 latent 中呈近似仿射关系，一次参考预测及其动作 Jacobian 就可能代替一批独立 rollout。

## 1. 研究对象：沿候选动作变化，而不是沿时间变化

记完整 horizon predictor 为：

$$
z_H=G_\theta(c,u),
$$

其中 $G$ 可以是当前 LeWM 的多步 autoregressive rollout，也可以是单独训练的 prefix predictor。第一轮优先保留同一个 $G$ 架构，只改变训练约束，避免把 Fast-LeWM 的接口收益误算到本提案上。[S3]

时间 straightening 比较一条轨迹的 $z_t,z_{t+1},z_{t+2}$。本方向比较相同 context 和 horizon 下的：

$$
G(c,u-\delta),\quad G(c,u),\quad G(c,u+\delta).
$$

这是三个不同动作选择引出的未来，而不是同一轨迹的三个时间点。

核心假说为：**在实际搜索访问的动作范围内，降低“动作→未来 latent”映射的局部曲率，可以增加一次线性化可复用的范围，从而减少独立的大模型查询。**

这不要求所有 context 共享 basis；局部 Jacobian 可以随 context 和参考动作改变。也不要求整个 PushT dynamics 全局线性。

## 2. 最小机制：一个动作方向的二阶差分正则

### 2.1 先固定动作坐标系

使用训练集统计或已定义的 action bounds 将优化变量归一化为 $v$，例如 $u=\mu_a+s_a\odot v$。转换参数在训练后冻结；不允许通过扩大动作尺度让曲率指标虚假下降。

$D=\dim(v)$ 必须由本地 CEM 真正优化的 tensor 确定。历史固定动作不属于可自由扰动变量；action block、frameskip、动作编码展开也不能凭 horizon 猜测。

以下将归一化后的完整预测映射记为：

$$
F_\theta(c,v)=G_\theta(c,\mu_a+s_a\odot v).
$$

### 2.2 正则项

对实际训练／规划分布中的中心 $v$ 和方向 $\xi$，取 $\|\xi\|_2=1$、$\delta=\rho\xi$，定义：

$$
\boxed{
\mathcal L_{\rm action\text{-}curv}
=
\mathbb E_{c,v,\delta}
\frac{
\|F(c,v+\delta)-2F(c,v)+F(c,v-\delta)\|_2^2
}{d\,\|\delta\|_2^4}
}
$$

实际实现设置固定正的最小扰动半径，避免除以接近零的数；可加入明确的数值保护，但不能让保护项大到掩盖小半径结果。

训练目标从最简单形式开始：

$$
\mathcal L=\mathcal L_{\rm base\ JEPA}+\lambda_c\mathcal L_{\rm action\text{-}curv}.
$$

$\mathcal L_{\rm base\ JEPA}$ 保留 LeWM 原有的预测与防坍塌项；不是把它们删除后仅优化平滑性。[S1]

扰动必须在**归一化后、实际有效的动作空间**中对称。若 $v+\delta$ 或 $v-\delta$ 超出 action box，则重采样或调整对称半径；不能分别 clip 后继续把它当成相同间隔的中心二阶差分。

### 2.3 有限差分而非只依赖自动微分 Hessian

使用三次预测得到 finite differences，不要求训练时显式构造完整 Hessian。分段线性网络的自动微分 Hessian 可以在多数点为零，但跨激活／接触变化的有限步长行为仍然不线性，因此实际搜索尺度的有限差分更贴近问题。

初始化可测试归一化动作空间中的若干固定 L2 半径，例如 0.05/0.10/0.20；**这些只是候选设计值，不是已知适合 PushT 的物理尺度。** 首先测实际 CEM 相邻候选和 proposal 半径分布，再冻结训练扰动范围。

训练方向使用随机混合方向，不只沿单一坐标轴。动作分量之间的交互曲率可能沿各坐标轴观察不到。训练 centers 同时覆盖数据附近动作、随机合法动作以及训练 contexts 上 CEM 的 early/middle/late proposals。

## 3. 不能省略的反例：低二阶差分不是充分条件

考虑一维 $F(v)=v^3$，在中心 $v=0$：

$$
F(\delta)-2F(0)+F(-\delta)=0,
$$

但此处 Jacobian 为 0，线性模型预测 $F(\delta)\approx0$，真实值却是 $\delta^3$。

因此，一处、一个半径上的中心二阶差分很小，**并不证明邻域内一阶 Taylor 近似足够好**。方向、中心和半径都要变化，并直接评价 held-out 的 Taylor residual。

还有一个更严重的退化：让 $F$ 完全忽略 actions，也能把曲率降到 0。保留 latent 的边缘方差／SIGReg 不能单独排除 action-insensitive collapse；模型可能只保留与动作无关的信息。

所以本方向的证据不能是“curvature loss 降低了”，而必须是：

$$
\boxed{\text{曲率降低}\ +\ \text{动作后果仍准确}\ +\ \text{更大范围的线性化可用}\ +\ \text{任务／速度改善}.}
$$

## 4. 训练数据：反事实监督从哪里来

### 4.1 层级一：冻结 encoder，只训练 predictor

保留原 latent，训练带／不带正则的相同 predictor。可用 frozen teacher 生成动作分支标签，所有方法获得相同分支数据。

这一层检验“更平滑的 predictor 是否更适合局部计算”，**不证明 encoder 的表示几何被改善**。如果现有 representation 的真实动作响应高度弯曲，过强正则可能只是让预测失真。

### 4.2 层级二：真实分支监督＋联合训练

要检验表示学习版本，从相同可复位的 simulator state 出发，将归一化分支 $v-\delta,v,v+\delta$ 分别经 $u=\mu_a+s_a\odot v$ 映射回环境动作后执行，得到对应的真实未来观测。固定相同的历史 context、仿真配置和随机状态；必要的隐藏环境状态也要恢复，不能只重设可见物体位置。

对三个真实分支使用正确的预测监督，例如：

$$
\mathcal L_{\rm branch}
=\frac13\sum_{\eta\in\{-\delta,0,\delta\}}
\frac1d\|F_\theta(c,v+\eta)-E_\omega(o_H^{v+\eta})\|^2.
$$

encoder 的训练与 target-gradient 约定沿用锁定的 JEPA recipe，不在复现时擅自加／去 stop-gradient 或 EMA。加上 action-curvature loss 后，预测准确性把动作响应约束与真实 future encodings 联系起来，才可能形成更适合局部规划的表示。

**所有对照必须获得同样的额外真实分支**。否则提升可能只是多了 counterfactual data，而非 straightening。

### 4.3 Optional 变体：直接约束真实分支编码

可另做一个消融，将 curvature 施加在：

$$
E(o_H^{v+\delta})-2E(o_H^v)+E(o_H^{v-\delta}).
$$

它更直接改变表示，但并不自动保证 predictor 的导数准确。该变体单独命名和报告；不要在第一版同时叠加两个 curvature loss，让贡献无法解释。

若只有 teacher rollout、没有真实反事实观测，不得把 teacher 的 imagined latent 当作真实图像来训练 encoder，也不能宣称已验证真实分支表示被拉直。

## 5. 推理：一次参考预测＋一次动作 Jacobian

在当前 context 下选参考动作 $\bar v$，计算：

$$
z_{\rm ref}=F(c,\bar v),\qquad
A=\left.\frac{\partial F(c,v)}{\partial v}\right|_{\bar v}
\in\mathbb R^{d\times D}.
$$

对附近候选：

$$
\hat z_H(v)=z_{\rm ref}+A(v-\bar v).
$$

于是 goal cost 近似为：

$$
\boxed{\hat J(v)=\|z_{\rm ref}-z_g+A(v-\bar v)\|^2.}
$$

该式对于线性化 surrogate 是精确二次型；对于原非线性模型只是局部近似。目标改变时可重用同一 $(z_{\rm ref},A)$，但 context 或参考动作改变后需重新计算相应量。

### 5.1 不把 Jacobian 算成一次普通 forward

自动微分开销依赖输入维度 $D$、输出维度 $d$ 和实现。forward-mode 构造 Jacobian 通常涉及 D 个方向的 JVP，reverse-mode 可能涉及 d 个输出方向的 VJP；batching/vmap 能改变壁钟时间和显存，但不是免费。

若对原 autoregressive rollout 求导，必须通过完整 imagined trajectory 传播对动作的影响。中途 detach 未来 latent 会改变 Jacobian 的意义，不能仍称其为完整 $F$ 的动作导数。测试时 dropout 关闭；计算 Jacobian 的路径不能被错误包进阻断梯度的 inference context。

### 5.2 两种 solver 实验，不能混成一个贡献

**A. 保持 CEM 的候选机制不变。** 每轮以当前 proposal mean 作为参考，准备 $(z_{\rm ref},A)$，对本轮 300 candidates 用二次 surrogate 评分。仍做 30 轮、top-30，保持其他配置一致。

这最有利于隔离“更易线性化是否减少 predictor 查询”。但 early CEM 可能采样得很广，部分 candidates 超出有效局部范围。第一版记录这个问题，不偷偷缩小方差、丢弃远候选或加全量 fallback。

**B. 使用局部二次优化。** 在动作 box 和一个盒状 trust region 内求解：

$$
\min_{\Delta v}
\|z_{\rm ref}-z_g+A\Delta v\|^2+\lambda\|\Delta v\|^2
$$

$$
\text{s.t.}\quad
\ell\le\bar v+\Delta v\le u,\qquad
\|\Delta v\|_\infty\le\rho_{\rm trust}.
$$

这是带 box constraints 的凸二次规划。若把 trust region 改为 $\ell_2$ 球，则有二次约束，不能再称作同一个 box-QP。训练的 L2 扰动半径与推理的 L∞ trust radius 分别记录，不能当作同一数值尺度。

$\lambda$ 是数值阻尼／局部求解设置，不是原任务中自动存在的动作代价。需要明确它改变了局部子问题，最终仍按原 criterion 和真实任务评估。每次接受新参考、重新线性化，以及可选的真实模型校验都计入时间。

B 改变了 solver，应与**未经 action-straightening 的相同模型＋相同局部 solver**比较；不能把普通局部线性控制本身作为本提案的新贡献。[S6]

## 6. 理论上应期待什么，以及不能保证什么

若在参考点到候选的线段上，二阶导数满足合适的向量算子界 $M$，则 Taylor remainder 有：

$$
\|F(c,\bar v+\Delta v)-z_{\rm ref}-A\Delta v\|
\le\tfrac12M\|\Delta v\|^2.
$$

因此更小的曲率有可能增大给定误差预算下的可用半径。但有限方向、有限 centers 的训练正则不提供已认证的 $M$，也不排除接触 regime 变化；**这里只是有条件的数学动机，不是部署安全保证。**

对线性化预测 $z_{\rm lin}$ 与真实模型差 $r=F-z_{\rm lin}$，cost 差有：

$$
|\|F-z_g\|^2-\|z_{\rm lin}-z_g\|^2|
\le2\|z_{\rm lin}-z_g\|\|r\|+\|r\|^2.
$$

latent Taylor residual 小仍需结合 goal 距离和 elite margin 解读。

## 7. 验证实验设计

### E0｜数学／自动微分单元测试

使用不涉及 PushT 的合成函数：

| 函数 | 应验证的行为 |
|---|---|
| 仿射 $F(v)=Mv+b$ | curvature 与 Taylor residual 都接近 0，二次 score 等价 |
| 含二次项的函数 | 非零 directional curvature，Taylor residual 随半径变化 |
| $F(v)=v^3$ 在原点 | 中心二阶差分为 0，但有限步长 Taylor residual 非零 |
| 含交互项 $v_1v_2$ | 只测坐标方向可能遗漏，混合方向可以观察 |

对实际 $F$ 用小批量对比 autodiff JVP 与中心一阶差分，确认动作索引、normalization、history 和 rollout 梯度正确。有限差分误差需考虑步长和浮点精度，不把一个步长上的吻合当成完整证明。

### E1｜先测 frozen LeWM 的局部可用范围，不训练

用新的开发 contexts，在 early/middle/late CEM proposal 上选中心。对多个半径计算 exact $F$ 与 tangent predictions，并记录：

$$
\operatorname{RelTaylorErr}
=\frac{\|F(c,v)-F(c,\bar v)-A(v-\bar v)\|^2}
{\|F(c,v)-F(c,\bar v)\|^2+\epsilon}.
$$

当真实变化接近 0 时，该比值可能不稳定；同时报告绝对误差，并以预先定义的变化阈值分组。再记录 cost error、elite update 和不同半径下的有效候选覆盖率。

这一实验的价值是确定问题：现有模型是否只在极小范围内可用？是 late CEM 可用、early CEM 失败？还是同一接触模式内也不行？不要依据整体 median 推断存在稳定有效半径。

### E2｜用一个正则改善这个范围

先固定 architecture、encoder 和训练数据，仅训练 predictor，比较 $\lambda_c=0$ 与非零值。以原 loss 或梯度尺度为参照，在 validation 调整权重；先小范围扫描，不进行无上限试验后挑最好结果。

所有组使用同样的分支监督和训练更新数。记录：

- 标准未来预测误差与真实／teacher 分支 action-response error；
- Jacobian 非零响应、不同动作导致的预测差异与标签差异；
- 多半径 Taylor residual，而不只是训练 curvature loss；
- action 方向的模型输出方差和信息坍塌检查。

可用差分监督误差诊断 action blindness：

$$
\mathcal E_{\rm response}
=\|[F(c,v+\delta)-F(c,v)]-[z_H^{v+\delta}-z_H^v]\|^2.
$$

若 curvature 大幅下降而 response error 增大、task quality 下降，视为过平滑失败，不是成功。

### E3｜联合训练表示：分辨几何与数据收益

在真实 counterfactual branches 上比较：

| 编号 | 训练设置 | 要排除的混淆 |
|---|---|---|
| R0 | 相同数据＋相同 JEPA，无新增正则 | 额外分支数据本身的收益 |
| R1 | frozen encoder＋action-curvature | predictor 平滑本身的作用 |
| R2 | joint encoder/predictor＋action-curvature | 是否能学习更适合局部计算的表示 |
| R3 | 同预算 temporal-straightening 变体 | 是否只是一般的轨迹平滑收益 |

R3 需按对应方法定义实现，保留相同基础模型和数据预算；不能把时间相邻差分与动作反事实差分混写为同一 loss。[S5]

不同 encoder 的 raw latent loss／cost 不直接可比。比较每个模型内部的 exact-vs-linearized gap，辅以统一的真实任务指标；跨模型动作选择可由原始 frozen LeWM 重新评分，但不能把其 cost 当真实最优。

必要时用同预算的 physical/control probes 检查信息保留，这些标签只用于诊断，不进入部署输入。

### E4｜训练机制×solver 的因子实验

最小的因果比较是：

| 训练模型 | 原完整 CEM | 逐轮线性化 surrogate-CEM |
|---|---|---|
| 无 action-curvature，同数据同架构 | 基础任务质量 | 原模型能否直接被线性化加速 |
| 有 action-curvature，同数据同架构 | 是否损害原任务质量 | 正则是否让共享计算真正可用 |

另做局部 QP 变体时给两行都使用相同 solver、trust region、outer iterations 和停止条件。可以在 validation 比较 1/3/5 次 relinearization 等小预算点，但 final test 前冻结，所有完整模型校验计时。

**希望观察的交互：** 新正则对完整 CEM 的质量不造成明显损害，却显著缩小局部 solver 相对完整 CEM 的质量损失；同时端到端成本下降。这比“换 solver 后更快”更能支持机制。

### E5｜接触模式、边界和局部性失效

按可验证的 simulator events 或预注册的状态分组检查无接触、接触建立、接触解除、物体旋转与近目标场景。仅把状态标签用于离线分组，不为部署引入 oracle contact mask。

尤其检查某一 perturbation 跨过接触模式时是否被过度拉直。若所有收益只存在于容易的无接触样本，而核心 PushT 推动物体阶段失败，应认定任务范围不足。

### E6｜完整计时：Jacobian 和 solver 一起算

记录一次 $F$ forward、Jacobian construction、quadratic coefficient build、surrogate batch scoring、local solver、relinearization 和 exact verification。测候选数量与 action dimension 的扩展曲线。

对于 K 次线性化，账本至少是：

$$
T=T_{\rm encode}+K(T_F+T_A+T_{\rm build}+T_{\rm local\ solve}+T_{\rm verification}).
$$

若实际求 Jacobian 已包含 reference forward，不重复计数；如果没有，必须包含。不能把 $T_A$ 当成 0，也不能拿顺序单候选的慢实现与优化 batched baseline 比。

## 8. 失败条件与停止规则

| 观察 | 判定 |
|---|---|
| 训练 curvature 降低，held-out Taylor residual 不降 | 正则目标与实际近似不匹配 |
| Taylor residual 降低，但 action response／任务明显恶化 | 抹掉了关键 dynamics，而非有效简化 |
| 同样改善来自 $\lambda=0$ 的额外分支数据 | 不能归因于新几何约束 |
| 只在远小于 CEM 搜索范围的邻域有效 | 不足以支持直接共享候选计算 |
| 原模型＋同一局部 solver 已一样好 | solver 收益成立，但新训练机制未证明必要 |
| Jacobian/relinearization 成本吃掉收益 | 理论计算减少未转化为实际加速 |

第一版不通过时，不立即叠加 contact classifier、mixture、复杂 trust controller、cache、fallback。先给出窄而明确的 NO-GO，或者将结论限定为某类局部动作分布；避免简单机制悄悄变成一套复杂系统。

## 9. 最小实施清单

```text
[ ] 锁定真实 action optimization coordinates 和 bounds
[ ] 测试中心扰动是否对称，防止 clipping 破坏公式
[ ] 用 affine/quadratic/cubic/cross-term toy functions 验证数学和梯度
[ ] 在 frozen LeWM 上测多半径、多个 CEM 阶段的 Taylor residual
[ ] 训练相同数据下 lambda=0 / action-curvature 的 predictor 对照
[ ] 同时检查 action response，排除 action-insensitive collapse
[ ] 有真实分支条件后做 joint-encoder 版本和 temporal control
[ ] 跑训练机制×solver 因子实验，再做 paired 环境任务
[ ] 把 Jacobian、relinearization、local solver、verification 全部计时
```

## 10. 创新边界和最小论文命题

Temporal Straightening 已研究通过时间轨迹曲率改进 latent planning；E2C 等工作已经学习适于局部线性控制的表示。[S5][S6] 因此“让表示线性化”或“用 Jacobian 做 MPC”不是新的贡献。

本提案需要验证的特定区别是：**同一 context 下对反事实动作方向施加几何约束，使一次局部模型能可靠服务多个动作候选，并产生可测量的 planning quality–latency 改善。**

需进一步查新：`action-space curvature regularization`、`counterfactual latent geometry`、`control-affine representation`、`locally linear latent MPC`、`Jacobian world model planning`、`action-conditioned straightening`。本文只列已核对的近邻，不宣称穷尽查新或新颖性已被确认。

## 11. 本方向应用共用协议时的特别说明

下面的共用协议仍以原 LeWM＋PushT 为项目参照，但本方向有两个不同的 full comparator：

**模型内 comparator：** 每个新训练 checkpoint 自己的 exact rollout＋criterion。用于隔离 surrogate/solver 的近似误差。

**项目 comparator：** 原始固定 LeWM checkpoint 与其任务表现。用于评估整个新方法是否值得替换 baseline。

两个 comparator 的 teacher-cost gap、action fidelity、计时和成功率必须分别标注。不能把由训练导致的模型变化与由线性化造成的误差混为一谈。


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

外部来源核对日期：2026-09-22。本文关于有限差分、Taylor remainder 和二次 cost 的内容是数学推导；它们的前提是否在本地 PushT 访问分布上满足，必须通过实验检查。

**[S1] LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.** 论文与官方实现。用于 baseline 的 CLS latent、预测接口和 terminal cost；实际实验以本地锁定 commit 为准。  
[论文](https://arxiv.org/abs/2603.19312) · [官方仓库](https://github.com/lucas-maes/le-wm) · [jepa.py](https://github.com/lucas-maes/le-wm/blob/main/jepa.py)

**[S2] LeWM 官方 PushT evaluation config.** 仅用于配置核对；`main` 是可变链接，应在实施时替换为 commit permalink。  
[config/eval/pusht.yaml](https://github.com/lucas-maes/le-wm/blob/main/config/eval/pusht.yaml)

**[S3] Fast LeWorldModel.** Action-prefix prediction 是既有工作，不应把直接／并行预测未来 horizon 当作本提案的新贡献。  
[论文](https://arxiv.org/abs/2606.26217) · [项目页](https://fast-lewm.github.io/) · [官方仓库](https://github.com/Yuntian-Gao/Fast-LeWorldModel)

**[S4] LpWM: A Case for Sparse Representations in World Models.** 用于比较“通过表示几何降低 predictor 复杂度”的研究定位；不等同于另一篇同名缩写的 Latent Particle World Models。  
[论文](https://arxiv.org/abs/2608.22764)

**[S5] Temporal Straightening for Latent Planning.** 用于时间轨迹曲率／表示几何的既有工作边界；核对到的版本为 v3（2026-08-11）。不等同于本文提出的同 context 动作反事实方向正则。  
[论文](https://arxiv.org/abs/2603.12231)

**[S6] Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images.** 用于 locally linear latent dynamics 和控制的既有工作边界。  
[论文](https://arxiv.org/abs/1506.07365)
