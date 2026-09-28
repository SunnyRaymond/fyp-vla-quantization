# Compiled World Model：无重训诊断

本目录执行 job `25309513` 后冻结的机制诊断，不覆盖原 predictor-level NO-GO。

- `DIAGNOSTICS_FREEZE.json`：数据、oracle、local-phi、anchored residual 和 routing gates。
- `PROTOCOL.zh.md`：claim boundary 与执行顺序。
- `run_compiled_diagnostics.py`：只加载现有 checkpoints，不创建 optimizer 或调用 backward。
- `run_compiled_diagnostics.pbs`：compute-node guard、bounded timeout 与 5 秒 GPU telemetry。

当前状态：PBS job `25327163.pbs101` 已完成，routing 为 **`COMPILER_BOTTLENECK_SIGNAL`**。frozen `phi(u)` 的 teacher-paid local coefficients 可以跨 bank 泛化，但 simple anchored residual quality 失败；详见 [`RESULT.zh.md`](RESULT.zh.md) 与 [`MEETING_CARD.zh.md`](MEETING_CARD.zh.md)。

official CEM 与 closed-loop 均为 `NOT_RUN_BY_SCOPE`。
