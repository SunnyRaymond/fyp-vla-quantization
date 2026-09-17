# B2 结果解释独立审查

日期：2026-09-13。范围仅为冻结 PROTOCOL.zh.md、BATCH2_REPAIR_GATE.zh.md、BATCH2_IMPLEMENTATION_RESOLUTION.zh.md、B2 runner/verifier、64825 小型 summary.json 与 64829/64831 verification.json；未读取 raw archive、未连接 cluster、未重算数组。

## 判定

结论可写为 scope_limited_preliminary_go，但只限于冻结的 6-state、2-noise、3-SR-draw、10-step、FP32 fake-quantized Euler screen。64831 的 checks 共 90 项且全部为 true；scientific replay 为 nondegenerate_count=6、joint_pass_count=6、numeric_decision=scope_limited_preliminary_go。64825 summary 显示 completed_count=12；64831 CPU replay 显示 mean_E_frozen=0.0048368999、mean_E_cyclic=0.0016748381；这些支持该有限 gate，不支持更广泛性能结论。

## 为什么修复后的结果可解释

64815 的 11/12 condition 超时被明确保留为 inconclusive_budget，没有与 B2 的 12/12 条件拼接。B2 仅把同一个 state 的两个固定 noise 组成 batch，再按原始 state,noise,arm,time,position,coordinate 轴还原；没有跨 state 合批、平均 action 或增加样本。BATCH2_REPAIR_GATE 允许的变化与 BATCH2_IMPLEMENTATION_RESOLUTION 的两个修正一致：timestep 为 [2]，endpoint 使用全部 50 个位置后截取 physical 前 7 坐标。

endpoint 与 common-path 指标没有混用。scientific_replay() 的 E_frozen/E_cyclic 来自各 arm 自己自由运行的最后 x_t，分别对应 F0–F2 与 C0–C2，并在每个 state 内平均 2 个 noise 和 3 个 draw。S_frozen/S_cyclic 则在同一 FP x_FP[t] 上，用 common_q_v 的三 draw field 做 forcing；D 是相同逐步平方项，S-D 仅为 off-diagonal cross-term。64831 同时通过 Euler recurrence、common-x hash、stepwise marginal 与 diagonal-energy equality，因此没有把 common-path 诊断冒称 endpoint 结果。

64829 中 actual_times_max_abs_error 的 0.0 被错误放入 boolean checks，所以显示为唯一 false；64831 只把 schedule/time receipt 中的 boolean gate 纳入 checks，同时保留数值误差字段。该修复改变记录逻辑，不改变 raw 数值或科学 gate；64831 的 actual_times_match_frozen_grid 仍为 true。

## 必须保留的边界

所有 provenance receipts、input/sample/source/package/checkpoint/protocol/helper hash、parent extension chain、V100 allocation、snapshot/112 expert Linear 与 restore receipts 在 64831 均通过。CPU verifier 独立复核 raw arithmetic、FP-path hash、schedule/axis 还原和文件 receipts，但没有重新执行 GPU model calls；它不能超出已保存 common-x binding 证明每个 velocity 的完整运行来源。

F0–F2/C0–C2 是 FP32 中写入 dequantized W4 weights 的 fake quantization；Cyclic arm 在每个 denoise step 切换 snapshot。这不是 native low-bit kernel、部署吞吐、内存/cache 节省或真实控制执行实验。结果只说明在该固定模型、输入、noise、draw multiset 与 10-step schedule 下，时间 assignment 与 endpoint/common-path 指标的关系；不声称 task success、closed-loop return、泛化或 native deployment benefit。按照 protocol，本结果后停止，不追加 seed、control、step 或完整 validation。
