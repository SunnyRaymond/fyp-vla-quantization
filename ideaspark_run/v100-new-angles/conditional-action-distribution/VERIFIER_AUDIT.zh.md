# `verify_marginal.py` 科学指标与 gate 审查

日期：2026-09-13。对照已冻结的 `PROTOCOL.zh.md` 进行静态 source audit；不检查尚待 producer 补齐的 provenance 字段，不运行脚本、数值或集群任务。

## 结论

除一个会造成合法 raw 结果被误判为 implementation failure 的比较外，verifier 的 array axis、距离定义和跨 condition gate 与冻结 protocol 一致。该问题不改变科学 metric，但在修正前应视为 execution blocker。

## 逐项核对

- `actions` 的 schema 是 `(condition, arm, draw, horizon, dim)=(8,2,64,50,7)`；第 87 行取 `horizon=0` 后得到 `(64,7)`，第 88 行按 A/B 各 32 切分，paired `FP/Q` 与 cross-block `Q_A–FP_B、Q_B–FP_A` 均正确。总量为 1024 samples。
- 第 64–65 行在 16 个固定方向上逐 projection 排序后取 absolute difference mean，等价于冻结的 SWD1；第 67–74 行使用包含 self-pairs 的 nonnegative biased energy V-statistic，解析 zero/translation checks 也与定义一致。所有输入在第 41–44 行转为 FP64。
- 第 92–96 行的 `Dpair` 是 64 paired draws 的 7-coordinate mean MSE；`Dnoise` 是 FP A/B 全部 32×32 cross-pair 的 7-coordinate mean MSE；translation 为 `sqrt(7*.25*Dnoise)e1`，与 protocol 的 effect-scale 推导一致。
- 第 103–110 行按 16 projection 计算 IQR，`FP IQR>1e-6` 才算 valid；`valid≥13`、bounded 的 13 个 spread-good projection，以及 `≥8` 个 ratio<0.5 且 `W_QQ/W_FF<0.5` 才标 collapse，均匹配冻结规则。Q-Q 小但 IQR 不满足时不会标 collapse。
- 第 111–116 行把 reference、translation sensitivity、mapping binding、bounded/shift 组合为 per-condition joint labels。第 144–148 行使用 `positive_count≥6` 且无 collapse，或 `negative_count≥6` 的跨 condition gate；没有把独立的六个分项计数拼成 positive。translation 失败的 condition 不进入 negative count，因此不应单独触发 scientific no-go。

## 必须修正的 verifier bug

第 57 行使用 `np.array_equal(data['batch_check_batched'], data['actions'][0, :, :4])`。冻结 protocol 只要求 batch4 与逐条结果的 `max_abs≤1e-5`；producer 若把逐条结果作为 canonical `actions` 保存，即使 batch gate 通过，浮点差异也可能使 exact equality 失败，最终第 162 行把合法科学结果降为 `implementation_inconclusive`。应改为同一 `1e-5` 容差比较，或明确 `actions` 必须逐 bit 复制 batched probe；前者与 protocol 兼容且避免无意义 false failure。此修正不改变 metric/gate。

## 条件性注意事项（不属于当前数学 bug）

第 54 行 exact 比较两份 CPU `torch.randn` 输出，只有在 verifier 与 producer 的 torch CPU RNG/version 合同相同才可通过；该 version 应由既有 runtime identity 绑定。第 87–93 行按 raw draw 顺序推断 same-noise pairing，最终 producer provenance 必须证明 action 顺序与 frozen noise 的 A 后 B 顺序一致；静态 verifier 本身不能从数值结果反推该绑定。两点属于 provenance contract，不应由 verifier 猜测或放宽。

修正第 57 行后，且 producer 完成上述顺序/版本绑定，我认为该 verifier 的科学 gate 可冻结；无需增加 full statistics。若仅 translation sensitivity 失败，结果应保持 `sensitivity_inconclusive`，不能改写为 scientific `mechanism_no_go`。
