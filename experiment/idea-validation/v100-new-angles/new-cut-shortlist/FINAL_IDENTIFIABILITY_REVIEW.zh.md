# Physical readout / goal coordinate：最终 identifiability review

范围：只读两个 prior gate、campaign、pinned DINO-WM source 小文件；未读取数组、加载模型、连接 cluster，也未重开任何已完成 screen。

## 总判定

`physical readout` 的强主张仍是 `identifiability_no_go`：现有结构不能把 visual error 的某个 probe 投影解释成环境的 task-unobservable direction。`goal coordinate` 需要纠正原稿的过强论证：2×2 arms 可以识别一个精确定义的 downstream interaction；这不等于已经识别 common-mode 机制或 task benefit。一个新的、明确限定为 cross-observation stochastic-map coupling 的 3×3 diagnostic 值得保留为 `conditional_go / GPU=0` 候选；它的 novelty 仍未认证，也不解冻旧 protocol。

## Physical-state readout

这是结构性不可识别，不只是缺一个 baseline。`wall_dset.py:28–30` 将 states 复制成 proprios，`visual_world_model.py:96–109` 把 visual、proprio 和 action 拼入每个 patch，`encode_obs`/`separate_emb` 又提供独立 proprio slice。故一个 probe 看到的 2D state-predictive component 已经是模型的直接输入通路；probe residual/nullspace 由训练 split、正则化、尺度和线性坐标定义，不等于环境的 task-unobservable subspace。

一个 `visual-Q × readout` factorial 至多估计“Q visual error 中有多少被这个固定 readout 预测”，而 `readout × future/action` counterfactual 至多改变测量目标。没有独立 task intervention/outcome，不能从该投影识别控制相关不可观测性。现有 teacher-bias 只否定 recorded-future visual feature 的 H5 RTN 改善，不能补成 physical-state 因果证据；这不是通过换 metric 或重新拟合 probe 能修复的。

## Goal/reference coordinate

这里的 2×2 current/goal visual-encoder arms 确实能统计识别一个 interaction contrast：在同一 pair、FP predictor、同一 objective 下比较 `F/F、Q/F、F/Q、Q/Q`。

设固定 FP predictor 和 action 为 `y_F=P_F(E_F(o_c),a)`、`y_Q=P_F(E_Q(o_c),a)`，goal 为 `g_F=E_F(o_g)`、`g_Q=E_Q(o_g)`，并设 `Δy=y_Q−y_F`、`Δg=g_Q−g_F`。逐 pair 的平方 MSE 交互项满足：

`I=L_QQ−L_QF−L_FQ+L_FF = −2⟨Δy,Δg⟩`。

这是 MSE 展开恒等式；估计该 descriptive interaction 不需要 predictor equivariance，也不需要把 FP encoder objective 当作 truth。原稿把“没有 equivariance”写成 interaction 不可识别，应该撤回。

但该恒等式本身不是机制发现。它只能说明 current-side propagated error 与 goal-side encoder error 的坐标相互作用；`I` 的符号不能单独证明 shared common-mode cancellation、planning benefit 或 action improvement。现有 source 中 current/goal 是不同 observation 的两次 `encode_obs`，因此输入误差、FP predictor response 和普通 coordinate interaction 都是可行解释。

把问题改名为 `current/goal encoder-PTQ interaction` 可以做，但这是较弱的 descriptive diagnostic，且直接邻近 QuantWM/DINO-WM 的 representation/planning fidelity。

## 值得保留的新 bounded cut

新假设应改写为：在相同 current/goal pair、相同 FP predictor 和相同 per-call W4 marginal 下，把**同一个冻结 stochastic encoder map** 用于两侧，是否比给两侧使用不同 map 产生更小的 downstream pair error。它测试 cross-observation error coupling，区别于已做的 predictor-member antithetic coupling 和 same-locus activation spatial coupling；仍有 generic correlated-rounding prior 风险，不能宣称 novelty 已成立。

最小设计：沿用 6 个未锁定 locus，固定 action block 和 FP predictor；生成三个 marginal-matched frozen maps `q0,q1,q2`，计算全部 3×3 ordered pairs `L_ij = MSE(P_F(E_qi(o_c),a), E_qj(o_g))`。主量是每个 locus 的 diagonal mean (`i=j`) 对 off-diagonal mean (`i≠j`)；3×3 使每个 map 在两侧出现次数相同，避免把 map 难度当作 coupling。需保存每个 `L_ij`、两侧相对 FP 的 delta、输入/恢复 receipt 及 map identity。目标是 downstream pair alignment，不是 FP truth 或 task success。

预注册 bounded gate：所有 9 项 finite、三张 map 的 marginal/output error 在预设容差内可比、`D_off>1e−12`；至少 4/6 locus 满足 `D_same ≤ 0.9 D_off` 才记 `scope_limited_preliminary_go`，否则 `mechanism_no_go`；若 map identity、边际平衡或 FP restore 不成立则 `inconclusive_binding`。阈值只支持小 screen 的效应筛查，不作显著性或泛化证明。

## 最小修订与停止条件

原四臂 protocol 不需要解冻，也不应把 2×2 interaction 直接升级为 common-mode cancellation。若另立 diagnostic，标题和 estimand 应固定为 `cross-observation stochastic-map coupling`；使用同一 pair、same FP predictor、3×3 exact marginal pairing，并保存两侧 encoder deltas、每个 pairing cell 与 predictor input shift。三张 map 的误差质量相近可以报告，但不应成为额外 gate，因为 3×3 已平衡两侧边际。

最终状态：**physical readout：identifiability_no_go；原 goal common-mode 主张：未识别；same-map/different-map branch：conditional_go / GPU=0，novelty unresolved。** 不追加旧候选 sample、seed、module、metric 或闭环验证。
