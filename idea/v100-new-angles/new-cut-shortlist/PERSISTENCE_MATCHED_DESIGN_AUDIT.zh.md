# PERSISTENCE MATCHED DESIGN AUDIT：SmolVLA denoising-call rounding persistence

审查日期：2026-09-13。本文只审查一个已精化的 C02/C04 diagnostic design；不
运行 model、cluster 或数值，不修改已有 protocol/result，不消费数据 namespace。
结论是 **conditional-go（窄机制 screen）**，但 novelty 未认证，也不支持
deterministic native low-bit deployment claim。

## 1. 设计对象与可识别问题

先在执行前固定三套 expert-only W4 stochastic-rounding weight snapshots：

    W^(0), W^(1), W^(2),  seeds = 2101, 2102, 2103

三套 snapshot 必须来自同一 checkpoint、同一 scale/bit recipe；记录每个实际
weight code、scale、seed 和完整 restore hash。对每个 fresh state `s` 和两个
固定 initial noises `n`，固定 LeRobot `v0.4.4` 的 `K=10` Euler flow，
`dt=-0.1`，不改变 processor、time grid、action mapping 或 input。

每个 `(s,n)` 运行六条独立 trajectory：

* **frozen** trajectory `d` 使用 `W^(d)` 的 10 个 denoising calls；
* **cyclic** trajectory `d` 在 step `t` 使用
  `W^((d+t) mod 3)`，其中 `t=0,...,9`。

`cyclic` 不是平均三条 action，也不是增加一条 ensemble trajectory；它仍是三条
各自完整的 10-call trajectory，只切换每次 call 所用的既有 weight snapshot。
FP32 trajectory 与一条固定 W4-RTN trajectory 作为共同 baseline。

这个安排真正隔离的是 **weight perturbation 的跨-denoising-step coupling**：
它没有改变每个 step 可见的三套 draw multiset、初始 noise 或单 trajectory 的
call 数。它和原先只比较 frozen SR 与 redraw SR 的设计不同之处，是把 marginal
fairness 变成逐 step 的严格配对，而不是依赖有限次 independent draws 的经验
MSE matching。

## 2. Exact marginal matching 的数学检查

在同一 FP reference path 的第 `t` 步，定义向量 field error

    e[s,n,d,t] = v(W^(d), x_FP[s,n,t]) - v_FP(x_FP[s,n,t]) .

这里的 `x_FP[s,n,t]` 对所有 `d` 和两种 schedule 完全相同；不能把各自
free-running 的 `x_t` 代进这个匹配证明。对 frozen，step `t` 的 draw 集合为
`{e[s,n,0,t], e[s,n,1,t], e[s,n,2,t]}`；对 cyclic，因映射
`d -> (d+t) mod 3` 是 permutation，集合逐元素相同。因此在每个 `(s,n,t)`
上有

    (1/3) Σ_d ||e_frozen[d,t]||²
      = (1/3) Σ_d ||e_cyclic[d,t]||²                 (exact set identity),

只允许剩余的浮点归约误差；应保存三套 `e` 原值与 draw-index table，而不是只
保存均值。这个 equality 是设计的关键 gate，不能用总体平均 MSE 或 endpoint
MSE 代替。

定义每个 schedule 的 common-path forced-integral diagnostic：

    S_schedule[s,n] = (1/3) Σ_d || Σ_t dt · e[s,n,q_schedule(d,t),t] ||²,

其中 `q_frozen(d,t)=d`，`q_cyclic(d,t)=(d+t) mod 3`。对应的 diagonal energy

    D[s,n] = (1/3) Σ_d Σ_t ||dt · e[s,n,q_schedule(d,t),t]||²

在两种 schedule 间也应相同。展开后

    S = D + (2/3) Σ_d Σ_{t<u} dt·du
          <e[s,n,q(d,t),t], e[s,n,q(d,u),u>>.

所以 `S_cyclic < S_frozen` 只可解释为 cross-time cross-term 在该 FP path
上改变；不能称为每步误差变小。若三套 e 的 set identity 或 D equality 不
成立，设计不再是 matched persistence test，应直接 `identifiability_no_go`。

同时，`S` 是没有 Jacobian transport 的 additive/first-order proxy。真正的
自由 trajectory 会把每步状态送入不同的 nonlinear field；因此 endpoint result
必须独立保存，不能用 `S` 代替 endpoint，不能把 `S` 下降写成 task return
或 success 改善。

## 3. Closest primary prior 与 novelty 边界

[PTQD](https://arxiv.org/abs/2305.10657) 已将 diffusion denoising quantization
noise 分解为 correlated 与 residual uncorrelated parts，并讨论 mean/variance
deviation、后续 step 的 noise accumulation 和 step-specific correction。它使
“跨 denoising step 的相关误差可能积累”不新；PTQD 还包含 per-step mixed-
precision 方向。本设计不做 correction、variance schedule、bit allocation 或
训练，只固定三套 SR weights 并置换它们的时间 assignment。

[AccuQuant](https://arxiv.org/abs/2510.20348) 在 quantizer PTQ 中显式模拟多步
denoising，针对 independent per-step reconstruction 遗漏 accumulated error 的
问题。它与本设计共享 multi-step error concern，但本设计不优化 quantizer，且
通过三 draw 的 cyclic permutation 在同一 FP path 上保持逐步 marginal multiset；
这只能构成窄 diagnostic 差异，不能宣称新 PTQ method。

[Q-Diffusion](https://arxiv.org/abs/2302.04304) 关注多个 timestep 的 output
distribution 和 shortcut activation 对低比特 diffusion 的影响；本设计固定
10 steps、fixed scale/bit，不做 timestep grouping。campaign 内的 QuantWAMs、
Flow Geometry、Flow Step Refinement 也已覆盖 reachable state/denoising-step
读出或 solver-step 对照；本设计只保留同一 weight draw assignment 的
cross-step coupling。

与本地 Antithetic Rounding Pairs 的差异是：antithetic 项把两个 model/member
或 candidate-level rounding 放入 joint law，机制上 preliminary_go，实践上因
two-W4 成本劣于 one-W8（且 W8 更好）而停止扩验；individual-error/binding 结论属于
TDQ64799 的 inconclusive_binding。本设计没有 member aggregation、candidate ranking 或平均 action，
只在一个 SmolVLA action chunk 的 repeated denoising calls 中切换三套已冻结
weights。这个边界降低了 fairness 混杂，但仍不足以认证 novelty。若检索要求
“任何随机 rounding schedule 都必须新”，则应改标 `novelty_no_go`；当前审查
只允许 conditional-go diagnostic。

## 4. 控制、raw evidence 与工程 gate

每个 `(s,n)` 必须保存：

* FP32 的 `x_FP[t]`、`v_FP[t]`；
* 三条 frozen 和三条 cyclic 的每-step `x_t,v_t`、最终 action chunk 和
  相对 FP endpoint error；
* common FP path 的 `e[s,n,d,t]`，shape `3×10×chunk×action_dim`，以及
  `q_frozen/q_cyclic` 的整数 schedule table；
* `D_frozen,D_cyclic,S_frozen,S_cyclic` 的 raw values、两 noise 的 state-level
  aggregation、FP/RTN/SR snapshot identity；
* source/checkpoint/processor/time-grid/restore hashes 和实际用到的 three
  weight snapshots。

至少有两个工程负对照：

1. **FP no-op**：FP path 的 repeated wrapper 与 direct FP inference 必须在
   `1e-6` 内逐元素一致；
2. **RTN schedule-insensitivity**：对同一 deterministic RTN snapshot 复用或按
   frozen/cyclic switch，输出必须逐元素相同（`1e-6` 内），否则 scheduler、
   weight transaction 或 PRNG 污染了科学比较。

此外，三套 SR code 的 multiset hash 在每个 `t` 必须相同；每条 trajectory 的
   noise、time、input fingerprint 必须相同；所有 tensors finite；restore 后
   named parameter hash 必须回到 pre-arm 值。任何一项失败都停止，不把工程失败
   写成 temporal-correlation effect。

## 5. 最小 screen 与预注册 gate

若 root 冻结该项，使用 6 个 fresh states、每 state 两个 fixed initial noises，
仅做 10-step flow：FP32、RTN、3 frozen、3 cyclic。估计 workload 是
`6 × 2 × (6 × 10)` denoising calls 加 FP/RTN 控制，目标为单张 V100 <=10 min；
不做 environment rollout。三套 W4 snapshot 在每次 call 切换的 overhead 只能
作为 engineering/runtime 记录，不得声称 native deployment cost 或 speedup。

先要求所有工程和 exact-marginal gates 通过。对每个 state 先平均两个 noise，
定义自由 trajectory endpoint MSE：

    E_F[s] = mean_n mean_d ||A_frozen[s,n,d] - A_FP[s,n]||²,
    E_C[s] = mean_n mean_d ||A_cyclic[s,n,d] - A_FP[s,n]||².

定义同样的 state-level `S_F[s]`、`S_C[s]`。科学 gate 固定为：至少 **5/6
states** 同时满足

    S_C[s] <= (1 - 0.25) S_F[s]
    E_C[s] <= (1 - 0.10) E_F[s]

且每个分母都大于 `1e-12`。任一 state 的分母退化、exact set matching
失败、common path 与 free-running data 错绑，或 5/6 joint gate 不成立，就
停止为 `mechanism_no_go` / `inconclusive`；不得只凭 `S` 或只凭 endpoint
选择性通过。`S` 和 endpoint 的正向方向必须预先固定，不能事后选 cyclic
有利的 noise 或 draw label。

即使 5/6 通过，结论也仅是：在这个 SmolVLA checkpoint、固定 input/noise 和
10-step Euler path 上，matched draw assignment 与 endpoint drift 具有一致的
窄机制信号。它不能说明真实 robot return、policy success、generalization、
native W4 kernel 或 universal SR schedule benefit。

## 6. 与现有 campaign 的不可重复边界

该 design 不改 scorer 或 planner ranking；不做 RankCal global ranking、OTC
`C_r`、CEM-Update、FRT residual transport、PRR paired recovery 或 TR-PVQ。
它不计算 action gradient，不改 semantic visual/proprio/action scale，不做
flow-geometry rank correlation 或 K=5/10/20 step refinement，不做 padding
feedback 或 conditional action marginal，也不做 TD-MPC2 five-critic coupling、
value-head centering/decoder-tail rounding。固定 `K=10`，唯一 intervention 是
**同一个 three-snapshot W4 draw set 在时间轴上的 assignment**。

## 审查结论

* **机制可识别性：有条件支持。** Exact per-step draw multiset 和 FP-path
  `D` matching 能消除上一个设计的 finite-draw fairness 混杂；`S` 的变化可
  归因于 additive cross-time cross-term，但 endpoint 仍受 nonlinear Jacobian
  影响，不能把两者合并成因果证明。
* **实验阻塞条件：明确且可早停。** 若无法在 hook 中同时取得 FP path 的三套
  `e`、保持 weight transaction/reload exact，或无法确认 cyclic 每步只切换一
  套已冻结 snapshot，则不应运行；这是 identifiability/engineering blocker，
  不是继续扩展 seed 或 full rollout 的理由。
* **Novelty：未认证。** PTQD/AccuQuant 已覆盖 denoising-step correlated/
  accumulated quantization error；本设计的 exact three-draw cyclic matching
  可能是一个有用的 falsification control，但目前只能作为 checkpoint-specific
  diagnostic，不能称新方法。
