# Broadcast coupling 结果解释审查

审查对象：冻结 `PROTOCOL.zh.md` 与 `job-64839/verification.json`；仅读小 JSON，未读取 raw 数组、未运行数值或集群任务。

## 结论

`scope_limited_preliminary_go` 由冻结 gate 正确支持，不应扩写为通用方法或完整任务结论。外层 verification 和 scientific replay 的 checks 全部为 true，6/6 state binding，5/6 state 的 gain 达到预注册 `>=0.10`，正好满足 `>=5/6`。

1. 五个通过 state 的 gain 分别为 0.3701、0.9057、0.7067、0.2140、0.6701；state 3 为 0.0771，未被事后改阈值或删样本。
2. Shared/Spatial 的 input MSE 在六个 state 均完全相同（`abs_diff=0`），每个 patch 的三-draw multiset、codes、dequant 和实际 predictor hook 输入均通过复核。因此 gain 可归因于本协议固定的 spatial assignment，而非输入误差预算变化。
3. FP-copy allclose、RTN-before/after exact、60 次 hook/predictor calls、state digest、source/checkpoint、V100 allocation 与 budget 均通过；没有 implementation 或 partial-output 理由削弱该 screen。
4. MSE 是单步 predictor visual 384D output 相对同一 FP `model.predict(z)` target 的 bounded fidelity 指标。它不等于 recorded future fidelity、action quality、planner value 或 task success。
5. 结果只涉及 20 个 broadcast tail coordinates、196 patches、三个固定 A4 draws 和六个复用的 frozen observations；协议已明确这些不是 fresh samples，也没有推断一般 independent SR。
6. Spatial 是固定 balanced offset 的空间重排，仍共享同一三-draw multiset；不能把本结果称为 compressed-storage、latency、native low-bit kernel 或通用 stochastic-rounding 优势。
7. `scope_limited_preliminary_go` 只支持该 checkpoint 的单步 activation-coupling diagnostic。协议要求 STOP；不应追加 draw、scale、任务、horizon 或 closed-loop 验证。
8. 先前关于 freeze SHA 的观察已在 GPU audit 中更正：本地 CRLF 与 controller 上传 LF 的字节差异不影响 64839 pin checks。

无 blocking issue；保留当前 bounded-go 与 STOP 解释。
