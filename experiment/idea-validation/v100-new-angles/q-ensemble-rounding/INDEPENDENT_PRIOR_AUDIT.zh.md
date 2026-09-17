# Q-ensemble stratified SR：独立 prior / identifiability audit

**日期：2026-09-13**  
**审查状态：`conditional_prior_gap`；未运行实验，未解除现有 `resource_blocked` asset gate**

## 1. 审查问题与结论

候选只改变五个已训练 TD-MPC2 Q critics 的 rounding randomness。对每个 weight coordinate 采样一个 `U~Uniform(0,1)` 和一个独立的随机 permutation `pi`，令

`U_j = (U + pi(j)/5) mod 1`，`j=0,...,4`。

必须明确 `pi` 是该 coordinate 的五个 member offset 的 permutation；不能为每个 member 另采一个 `U`。在 `U` 真正 uniform、每个 member 使用同一 SR quantizer 定义、相同 per-output-channel granularity / clipping / bit-width，且保留各 member 自己的已冻结 scale 值的前提下，每个 `U_j` 的边际仍为 Uniform，故可与 independent SR 做边际匹配比较。联合分布被改变，但没有数学理由保证 covariance 的符号或 two-Q average 一定改善。

本次 bounded prior pass 的结论是：

| 层次 | 结论 |
|---|---|
| generic prior | low-precision ensemble 与 SR 误差建模已有明显先例；不能宣称“用随机 rounding 做 ensemble”本身新颖 |
| direct overlap | 未发现 primary source 已经在“五个已训练 critics + stratified/shared SR + 固定 two-Q avg”这一组合上给出直接 recipe 或结果；该判断仅限本次 bounded search，不是 exhaustive novelty certification |
| identifiability | 可做窄的 decision-interface diagnostic，但必须测 scalar Q-error covariance；weight rounding covariance 单独不能识别机制 |
| 本轮行动 | 若 source/input gate 解除，只保留小规模 falsification screen；不把它写成自动 variance reduction、uncertainty estimator 或 task-success 方法 |

## 2. 最近 primary prior 与重叠边界

1. [Ex Uno Pluria（NeurIPS 2024，arXiv:2411.14860）](https://arxiv.org/abs/2411.14860) 从一个 pretrained model 以 low-precision stochastic rounding training-free 地产生多个低精度成员，展示 rounding diversity 对 ensemble prediction 的用途。它直接覆盖“SR 产生 low-precision ensemble”这一大类想法，但没有覆盖本候选的五个**不同训练所得** Q critics、member 对应 coordinate 的 stratified coupling、TD-MPC2 的随机 two-Q subset，或固定 candidate pool 上的 planner `avg` error。
2. [Generation of Ensemble Perturbations Using Low-Precision Floating-Point Numbers（JMSJ 2025）](https://www.jstage.jst.go.jp/article/jmsj/103/4/103_2025-022/_html/-char/en) 把低精度 model arithmetic 当作 weather model ensemble perturbation，并比较 reduced-precision model ensemble 与 conventional ensemble。它支持“rounding error 可成为 ensemble perturbation”的先例，但不是 multi-critic Q aggregation，也没有本候选的 cross-member uniform coupling control。
3. [Deep Learning with Limited Numerical Precision（Gupta et al., 2015）](https://arxiv.org/abs/1502.02551) 与 [Stochastic Rounding: Implementation, Error Analysis, and Applications（Connolly, Higham, Mary, 2022）](https://doi.org/10.1098/rsos.211587) 说明 SR 的基本 unbiased / error-growth 语境；它们没有证明跨网络 member 的 shared randomness 会改善 scalar value aggregation。针对“decorrelation”检索到的 SR 讨论主要是训练或数值累加语境；本次没有找到把 `U_j=(U+pi(j)/5) mod 1` 用于已训练 Q ensemble 的 primary 直接覆盖。

因此，候选的可辩护差异只能是一个很窄的**审计问题**：在官方已存在的 five-member Q ensemble 和真实 two-Q aggregation path 中，保留各 member SR marginal，只改变跨-member joint rounding law，观察这种改变是否进入 Q-score 的 aggregate error。它不是新的 ensemble training、quantizer design 或 uncertainty method。

## 3. 为什么 weight correlation 不等于 Q-error correlation

对某个 member `j` 的量化坐标，写其归一化 fractional part 为 `f_j`、channel scale 为 `s_j`，SR residual 可写成

`delta_j = s_j * (1{U_j < f_j} - f_j)`。

即使五个 `U_j` 的 marginal 完全匹配，五个 critics 在同一 coordinate 的 `f_j`、scale、feature phase 和局部敏感方向也不同。它们是不同训练所得网络；同一 parameter index 不表示相同的 feature 或相同的 output direction。故 shared / stratified uniforms 改变的是 rounding indicators 的 joint law，而不是直接指定 `delta_i` 与 `delta_j` 的 scalar-Q covariance。随机 `pi` 还会把 coordinate-level effect 混合；covariance 可能为负、正或接近零，不能预先指定“更 decorrelated”或“更 stable”。

Q member 的 scalar value 还经过 two-hot symlog value decoding。量化后实际误差是

`e_j(x) = D(z_j^Q(x)) - D(z_j^FP(x))`，

其中 `D` 包含 logits 到 two-hot distribution/value 的非线性解码；局部近似还要乘以每个 critic 和每个 input 的 network Jacobian。于是即使 weight residual covariance 有清楚的符号，经过不同 hidden path、logits saturation 和 symlog inverse 后，`Cov(e_i,e_j)` 也可能换号或被抵消。未来必须从 live source 记录实际 decoder convention，不以“weight covariance 改变”替代 Q-output evidence。

这也是本候选最小的可辨识性要求：同时保存

- declared Q weight coordinates 的 residual / rounding-indicator covariance（只作 audit）；
- 同一 state/action cell 上五个 member 的 `Q_j^FP` 与 `Q_j^arm`，并计算 `e_j=Q_j^arm-Q_j^FP` 的 cross-member covariance / correlation；
- 固定 pair 上的 two-Q `avg` error 和 candidate ordering。

若只看到第一项变化，第二、三项没有稳定变化，则 coupling 没有被证明是 decision interface 的 load-bearing mechanism。

## 4. 最小负证据与 matched control

仅在既有 asset gate 解除后，使用协议中的 8 个 fresh DMC reset states、每 state 64 个 model-generated horizon-3 candidate actions、3 个 rounding seeds。3 seeds 是同一设计的 repeated measurements，不计作 24 个独立 state blocks。所有 arm 共享 state/action tensors、candidate order、FP32 reference、dtype 和一次生成并记录的 Q-pair schedule；quantized arms 使用该同一 pair schedule，不能让 `torch.randperm` 的 pair 变化伪装成 rounding effect。FP32 只作 reference，RTN 是 matched deterministic control，不另造 QFP pair/control。

实际 live `avg` 的 primary metric 固定为同一 pair `(i,c,j,c)` 上

`A_arm = (Q_i^arm + Q_j^arm)/2`，
`MSE_avg_arm = mean[(A_arm - A_FP)^2]`，

先在每个 state 内对 64 candidates 和 3 seeds 聚合，再做 state-level gate。不得把 `min`、post-hoc pair、task success 或 variance identity 改成主指标；`min` 只能作为官方 Q aggregation 的 secondary diagnostic，并清楚区分 live-Q 与 TD-target path。

最小的负证据规则建议预注册为：若 stratified arm 在至少 6/8 states 中确实改变了 weight-level covariance，但 scalar-Q error covariance 相对 independent SR 的变化落在小工程界限内（例如 state-level off-diagonal correlation difference `|Delta rho_Q| < 0.05`，且 Q-error 方差非退化），同时 `MSE_avg` 没有达到下述 gain gate，则报告 **mechanism no-go**。这不是统计显著性检验；它是防止把 weight-level coupling 当作 output-level causal evidence 的负筛选。若 Q covariance 改变而 aggregate error/order 没改善，则报告“可见但不具决策收益”，也不扩大实验挽救。

## 5. 预注册的窄 gate 与停止条件

候选只有同时满足下列条件才可称为该 fixed input 上的 preliminary mechanism signal：

1. aggregate `MSE_avg_strat <= 0.75 * MSE_avg_ind`（即相对 independent SR 至少 25% gain）；
2. 至少 6/8 state blocks 各自满足同一 25% gain；若 independent 分母接近零，该 block 记为 inconclusive，不改用人为 floor 制造 gain；
3. stratified 不差 RTN：aggregate `MSE_avg_strat` 不高于 aggregate `MSE_avg_RTN`；每个 state 的 RTN 差异仍须报告，但本 audit 不把未预注册的 state-level RTN 条件偷偷加入主 gate（任何比较容差必须在运行前固定）；
4. FP no-op、exact weight restore、same pair schedule、marginal SR audit 和 source/checkpoint/config identity 全部通过；Q-output covariance 的方向和 raw member values 可复核。

任一条件失败即停止在 `no-go` 或 `inconclusive`。即使全过，结论也只能是“该 TD-MPC2 checkpoint、该 fixed state/action pool、该 two-Q avg path 上，stratified rounding 改变了并可能降低 aggregate Q error”；不能外推为通用 decorrelation 定理、epistemic uncertainty improvement、closed-loop success 或 deployment benefit。

## 6. 执行边界与当前状态

本文件只做 primary-source reading 和数学可辨识性审查；没有下载、cluster 操作、模型加载或数值实验。现有 `PRIOR_GATE.zh.md` 的 `resource_blocked` 状态不因本 audit 改写。未来若 asset gate 解除，先验证实际 five-member scalar decoder 与 two-Q call path，再执行固定小 screen；不得用 synthetic input、post-hoc pair 或旧 DEV 输入补齐证据。
