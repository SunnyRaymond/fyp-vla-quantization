# Repair 02：第三次最小 run 已获准

ASPIRE2A job `22701154.pbs101` 在首个 episode 的 treatments 已落盘、但写入任何 per-state metric 前 fail-closed：`KeyError: 2`。原因是请求 horizons 已冻结为 `[1,5]`，内部 rollout 仍逐步经过 2/3/4，却错误地把每个 intermediate step 都写入只含 1/5 的输出字典。

本地修正仅在 `step+1` 属于 requested horizons 时保存 prediction，并在 arm 结束后验证每个 requested horizon 的 draw 数完整。未改变数据、arms、quantizer、seeds、metric、H=5 threshold 或停止规则。

用户随后明确要求继续完成 Experiment 1--3，并在信号不好时停止扩展。独立只读审计（`gpt-5.6-luna / xhigh`）结论为 GO；本地 6 项 tests、Python 编译和 PBS shell 语法均通过。允许原子上传本修正并提交第三个最小 A100 run；science protocol、arms、seeds、threshold 和停止规则保持不变。
