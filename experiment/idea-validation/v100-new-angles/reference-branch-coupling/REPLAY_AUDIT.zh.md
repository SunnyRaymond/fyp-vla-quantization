# Reference-branch replay：独立静态审计

当前root复核：独立审查指出的arm-order/schema缺口已在模型运行前补齐，最终replay LF SHA256 `258c41d8af9e991130db3df0f09935d560019b104e2cbb0d8ea2b9809bc2ee9c`，AST通过。下面保留独立审查当时的意见；其“仍blocking”是修复前状态。尚未做数值运行，最终以真实CPU verifier为准。

范围：只读 `reference_replay.py`、`PROTOCOL.zh.md`、`RAW_CONTRACT.json`；未读取 raw、未运行 NumPy、未连接 cluster。

## Blocking 合同问题

1. **当前 raw key/type 已对齐。** 复核早期曾观察到 `valid_local_indices`/字符串 trajectory IDs 与 replay 不一致；root 随后已将 `RAW_CONTRACT.json` 固定为 `valid_indices`、整数 `trajectory_ids`，与 replay 的 `[124..129]` 和 `IDS` 比较一致。这两项不再是当前阻断，但应保留该修订记录，避免 producer 回退旧合同。
2. **未验证 arm order（仍为 blocking）。** replay 用 `pred_visual[...,0]` 当 FP、`[...,1]` 当 RTN、`[...,2:]` 当 SR0–2，却没有要求 raw 中的 `arm_names` 或 schema。必须把两者纳入 required/identity checks，并严格匹配 `RAW_CONTRACT` 的五项顺序；否则错误重排会产生看似有效的科学结果。

## 公式与 gate

除上述合同问题外，静态公式与冻结 gate 一致：`y[2:]`/`g[2:]` 代表三个 SR draw；diagonal 三格与 off-diagonal 六格的 current/goal marginal 加权完全相同。`A_de=mean((r_de-r_fp)^2)`、`B_de=(L_de-L_fp)^2` 及 `A_D/B_D` binding、10% joint gate 均与 protocol 相符；没有把 raw loss 变小误当 primary。float32 raw 转 float64 后重算 MSE，分解恒等式检查也与协议一致。

`predictor_input[..., :384] == current_visual` 与 384 后 tail 跨 arm 相同，正确检查实际 predictor input 和 FP non-visual channels；FP-copy/after-restore no-op 的 shape `[6,2,196,384]` 检查逻辑正确。`completed` 不完整返回 `inconclusive_budget`，缺字段/shape、非 finite 或身份失败保持 `implementation_inconclusive`，partial 分类没有把不完整数据升级为科学 no-go。

## 结论

当前 replay 的数组 key/type 已可按现有 `RAW_CONTRACT` 复算；但在补上 arm-order/schema hard check 前仍有一个科学正确性 blocking。修复该检查后，3×3 计算和 partial/finiteness 语义无需扩展测试或改动阈值。
