# Meeting Card｜Compiled World Model Diagnostics

- **历史状态**：job `25309513` predictor-level NO-GO 不变。
- **问题**：B3 是 common future 错、action contrast 错，还是 compiler 生成 coefficients 失败？
- **Common oracle**：B3 `0.1307/0.1667 → 0.0453/0.1333`，不恢复。
- **Action oracle**：B3 `→ 0.9951/0.9333`，恢复。
- **Anchored residual**：`→ 0.0583/0.1000`，仅改善 `3/8` episodes；FAIL。
- **Frozen phi local fit**：cross-bank Spearman/top-30/relMSE median=`0.8846/0.6500/0.003897`；gate PASS，但使用 teacher-paid bank。
- **Positive control**：旧 anchor-aligned student raw=`0.8904/0.7167`，证明 evaluator 能识别强排序信号。
- **Timing**：anchored 30-query p50=`30.273 ms` vs teacher=`575.978 ms`；质量失败，不能算有效 speedup。
- **决策**：**`COMPILER_BOTTLENECK_SIGNAL`**。若继续，只改变 compiler/coefficient supervision；不扫 rank、不推进 CEM。
