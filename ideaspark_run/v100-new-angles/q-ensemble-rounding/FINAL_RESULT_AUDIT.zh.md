# TD-MPC2 Q-ensemble rounding：最终结果静态审查

## 范围与结论

本审查只读取冻结的 `PROTOCOL.zh.md`、`IMPLEMENTATION.zh.md`、`verify_tdq.py`，以及 GPU job `64799`、CPU verifier `64801` 的小结果文件；没有重新加载 raw model output，没有运行模型、数值实验或集群任务，也没有修改结果文件。旧的 `64797` 与 alias probe `64798` 只作为失败/修复 provenance，不纳入科学样本。

结论：**`binding_count=0` 导致 `inconclusive_binding` 是冻结协议下的正确判定；`practical_gate=true` 不能绕过 mechanism gate。** Raw 数值方向对 stratified SR 不利，但由于其 member-level fairness 已失败，不能把它写成已识别的 coupling 机制 no-go，也不能据此追加 noise seeds 挽救。

## binding 判定

`verification.json` 中 8 个 state 的 `A_ind` 与 `M_ind` 都大于 `1e-8`，所以不是 `inconclusive_degenerate`。但 `member_MSE_ratio_strat_ind` 为：

`1.4082, 1.4257, 1.4299, 1.3998, 1.3786, 1.4571, 1.3960, 1.3972`。

冻结协议要求每个 state 的 `M_strat/M_ind∈[0.9,1.1]`，且至少 6/8 个 state 通过；实际 8 个全部失败。因此 `binding` 八行均为 false，`binding_count=0`，`numeric_decision=inconclusive_binding`。这不是 verifier 漏算，也不是把 `M` 与 `A` 轴混淆；CPU verifier 先在 FP64 中按 state、seed、candidate、member/pair 聚合，并通过 `A=(M+C)/2` 恒等式检查。

该 fairness 失败还意味着 stratified 与 independent 的 scalar-Q 误差幅度没有满足预注册的 matched-marginal 证据边界。即使 stratified 的 coupling 改变了某些 cross term，也不能把总差异归因于 joint rounding law。

## practical gate 不能单独 go

结果中的 `mean_A_by_arm` 是：

`FP32=0`、`W4_RTN=151.6405`、`W4_independent_SR=71.4973`、`W4_stratified_SR=92.4174`。

`A_strat≤A_RTN` 在 8/8 state 成立，且 aggregate mean 也更低，所以 `practical_gate=true` 正确表示“相对 RTN 的 absolute MSE 对照通过”。冻结协议先要求 mechanism gate；verifier 的判定顺序也是先在不足 6 个 fair state 时返回 `inconclusive_binding`，不会因为 practical 为 true 而返回 `preliminary_go`。这里没有 RTN 分母或零 baseline 伪问题。

## raw negative direction 的正确表述

在不把它升级为机制结论的前提下，raw output 给出清楚的 adverse direction：8/8 state 都有 `A_strat>A_ind`；mean 从 `71.4973` 增至 `92.4174`，约增加 29.3%，`median_gain_all_states=-0.29017`。同时 `M_strat/M_ind` 已增加到约 1.38–1.46，说明 stratified arm 的 member error energy 本身明显更大。`C_ind-C_strat` 的记录为正并不足以抵消这一点；由于 `A=(M+C)/2`，不能只挑 cross-term reduction 宣称 coupling 有益。`cross_attribution_fraction` 在所有 state 为 null 也符合“没有正的 `A_ind-A_strat` 可供归因”的实现。

因此本次应报告为：

- 正式 decision：`inconclusive_binding`；
- 描述性 raw 方向：stratified SR 比 independent SR 更差，并且相对 RTN 的 absolute MSE 对照仍较低；
- 证据边界：fairness 未通过，不能认证 stratified coupling 的独立因果效果，也不能称为 mechanism_no_go 或通用负结果。

## 实现与结果完整性

`64801/verification.json` 的 raw checks 和 producer checks 全部为 true：包括 FP/RTN seed replication、相同 CUDA pair schedule、official `Q(avg)` decode gate、all-pair decomposition、strict load、pinned source/checkpoint、真实 V100、FP cache hash、actual TensorDict live/detach alias、target/bypass digest、完整 quantizer transaction 与 CPU candidate replay。`official_avg_max_abs_error=0`，说明实际 API 对照没有引入可见的 pair reduction 错误。

`64797` 的 live/detach alias 错误发生在任何 Q output 前；`64798` 只证明修复后的 actual TensorDict alias，未产生 inference。故不能把旧失败混入 64799 的 raw negative direction，也没有发现会改变当前 `inconclusive_binding` 的具体实现 bug。工程检查验证了记录和路径，不等于独立证明有限样本的 Uniform 定理或机制因果性。

## 停止规则

当前候选应停在上述 bounded screen。不得因 raw 方向不利而增加 rounding seeds、挑选 subset、放宽 `[0.9,1.1]`、删除不公平 state、改用 `M` 以外的新分母，或重写 mechanism gate。三个 rounding seeds 是同一设计的 repeated measurements，不是可在结果后追加的救援样本；任何改变都必须是新 protocol、新输入和新审查，不能修补本次结果。
