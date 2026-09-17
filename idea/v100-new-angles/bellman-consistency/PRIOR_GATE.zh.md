# Bellman consistency / cross-module residual：prior 与可辨识性 gate

**日期：2026-09-13**  
**范围：**只审查候选机制与最小 screen；未加载 checkpoint、未连接集群、未运行数值实验。

## 判定

对“用 cross-module shared rounding 让 Bellman residual 更接近 FP，并据此证明量化 value / return 更可靠”的方法主张，判定为 **`novelty_no_go`**。原因不是没有可测的差异，而是当前主指标是 learned-model 内部 proxy，且“误差抵消”本身是代数分解；它不足以识别 value fidelity 或环境 return。可以保留一个**条件性的 one-step consistency diagnostic**，但不值得单独启动 V100，也不能把结果写成通用 PTQ 方法或 closed-loop 改善。

## 官方 forward contract

固定 TD-MPC2 source commit [e9f5932](https://github.com/nicklashansen/tdmpc2/tree/e9f59321933cbc8e11a002b842adc7d4ffae8ff1) 后，`world_model.py` 的接口是：encoder 得到 `z`，`next(z,a)` 得到预测 latent，`reward(z,a)` 得到 two-hot reward logits，`Q(z,a,return_type='all'|'avg'|'min')` 再经官方 `two_hot_inv` 解码。`SimNorm` 只是 latent groups 上的 softmax。

这条路径没有一个可直接复用、且语义唯一的“Bellman residual”。`TDMPC2._estimate_value` 使用 live reward、live dynamics，并只在末端调用 live `Q(..., return_type='avg')`；训练 `_td_target` 则使用 target Q 的 `min` 路径，另有 stochastic policy prior。因而下式只有在 action、`Q` reduction、live/target 选择和 discount 都预先冻结后才有意义：

`δ_arm = r_arm(z,a) + γ Q_arm(z',a') − Q_arm(z,a)`。

若候选真正测的是该固定路径，则

`Δδ = δ_Q − δ_FP = Δr + γΔQ' − ΔQ`。

这是恒等式，不是 shared rounding 有效的证据；把 terminal-Q planner、target-Q training target、policy action 混用会直接改变被测对象。

## Prior 与重叠边界

- [QuantWM / An Empirical Study of World Model Quantization](https://arxiv.org/abs/2602.02110) 已直接研究 world-model PTQ 的 module sensitivity、joint weight/activation quantization、latent rollout 与 planning objective 的偏离。它没有给出本候选的 cross-module Bellman residual recipe，但“用 objective/rollout 观察量化影响”已有直接邻近先例。
- [QuaRL](https://arxiv.org/abs/1910.01055) 已覆盖 RL 中的 weight/activation PTQ 评估；因此“RL 场景使用非重建指标”本身不能构成方法新颖性。本次 bounded primary search 未找到一个精确复现“reward+dynamics+value shared rounding”的 primary recipe；这只是有限检索结果，不是 novelty certification。
- [The Bellman Error is a Poor Replacement for Value Error](https://arxiv.org/abs/2201.12417) 证明 current/next value error 可以互相抵消，低 Bellman error 不代表低 value error，甚至不代表较低 return 风险。这直接阻断“残差更小 ⇒ value 更准”的解释。

所以它与现有 **Q-ensemble SR** 的技术对象不同：后者只改变五个已有 Q critics 的 joint rounding law 和官方 two-Q aggregation，未量化 reward/dynamics，也未用 TD residual；与旧 **CEM-Update/FRT** 也分别不同于 planner update fidelity 与 latent residual transport。但这种对象差异不足以支持独立方法，因为 cross-module error cancellation 仍属于 objective-aware PTQ 的窄 proxy 变体。

## 关键 identifiability blockers

1. FP residual 不是 oracle。TD-MPC2 的 reward 与 value 都是 learned predictions；在 imagined transition 上没有真实环境 reward 或 true value 可供校准。FP residual 非零可能只是模型本身的 Bellman inconsistency。
2. live、target、policy 不可混合。改变 policy action、target-Q reduction 或 random Q subset，会把 policy/target drift 当成 rounding effect。必须固定 action，并单独声明只测 live `Q`；若量化 target-Q，须另设 arm，不能合并主结果。
3. “同 bit + 各模块 MSE 小”不推出 residual cancellation。reward、dynamics、Q 的 tensor shape、scale、feature basis 均不同；跨 tensor 共享同一 `U` 或 seed 没有参数坐标的语义对应，也没有保证输出误差 covariance 的符号。若先看 8 个 state 的输出误差再调 coupling，则是 calibration leakage。
4. residual 变小可能正是错误抵消。没有逐项记录 `Δr`、`γΔQ'`、`−ΔQ` 及其 covariance，无法判断 gain 是否来自宣称的 cross-module interaction；即使记录了，也只能支持该输入池上的 diagnostic，不支持 return。

## 可保留的最小 bounded screen（若坚持）

只用 8 个 fresh reset states，固定同一 `z,a,z'`、dtype、discount、action 与 source/checkpoint。四种语义不必扩展为完整闭环：FP32；matched W4 independent RTN；W4 structured shared-rounding（仅对预先列明的 reward、dynamics、live-Q tensors）；以及必要的 exact-restore/no-op control。Q-member 的 coupling、target-Q、policy 和环境 stepping 均保持不变，否则无法把差异归因到 cross-module rounding。任何 coupling 规则必须在看到 held-out states 前冻结；若需要调参，另留 calibration inputs。

主指标只能命名为 `MSE(δ_arm−δ_FP)` 或其 state-level absolute error；同时报告 FP `|δ_FP|`、三项 `Δr/γΔQ'/−ΔQ`、各模块 output MSE 与 latent one-step error。若 independent 分母近零，报告 absolute value 并标 `inconclusive`，不加事后 floor。一个可执行的探索 gate 是：structured arm 相对 independent RTN 的 residual-drift 至少减少 25%，且至少 6/8 states 同向；误差协方差分解必须显示 gain 由跨项 covariance 而非单模块 MSE 偶然下降解释。该 gate 只能叫 `conditional_preliminary_mechanism_signal`；失败则为 `mechanism_no_go` 或 `inconclusive`，不声称量化 value 或 return 变好。

## 停止条件

若不能冻结精确 live/target/policy path、不能提供不含 held-out tuning 的 shared-rounding rule、不能保存三项 residual decomposition，或把 learned FP residual 当成 ground truth，则候选停止。即使 bounded diagnostic 通过，也只能报告特定 TD-MPC2 checkpoint 和 8-state one-step 输入上的内部 consistency 变化；不得声称 Bellman-aware PTQ、通用 cancellation 定理、真实环境成功率或 deployment benefit。
