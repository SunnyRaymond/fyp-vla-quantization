# LeWM PushT / Reacher 延迟与异步结果

> 当前文件是空白报告模板。填入真实 runner 输出和分析 JSON 后再下结论；不得把 synthetic self-check 当成实验结果。

## 实验身份与有效性

| 项目 | PushT | Reacher |
|---|---|---|
| Checkpoint / pinned source |  |  |
| 50 个冻结 task ID 来源 |  |  |
| 条件 / N / K / 调度模式 |  |  |
| 原始 episode 记录与分析 JSON |  |  |
| 同步 K0 对齐 gate | 通过 / 未通过；证据： | 通过 / 未通过；证据： |
| 每个条件 episode 数 |  |  |
| RNG 配对规则 |  |  |

**比较边界：** batch-50 官方 K0 仅用于原 evaluator 的 gate / parity 参考。true-async 的配对基线是相同 task ID 的 N=1 synchronous K0。报告不得把 batch-50 成绩当作 N=1 true-async 的配对基线。

## 成功率与配对结果

每个任务逐条件报告 `successes / n` 和成功率。对 N=1 synchronous K0 以外的配对条件，填写同 task ID 上的：

| 条件 | 成功数 / n | 成功率 | 配对 gain | 配对 loss | 净配对变化（百分点） | McNemar exact 双侧 p |
|---|---:|---:|---:|---:|---:|---:|
| batch-50 official K0（仅 gate） |  |  | — | — | — | — |
| N=1 sync K0（paired baseline） |  |  | — | — | — | — |
| 其他条件 |  |  |  |  |  |  |

`gain` 是 baseline 失败而目标条件成功的 task 数；`loss` 是 baseline 成功而目标条件失败的 task 数。净配对变化为 `(gain - loss) / matched_n`。列出 matched task ID，或引用分析 JSON 中保存的 ID 清单；缺失或额外 ID 时不计算配对检验。

## 延迟与实时性

报告分析 JSON 的 episode-level `latency_ms`、`observation_age_ticks` 与 `rtf` 的 p50 / p90 / p95 / p99，并说明这些字段在 runner 中的聚合定义、样本数和缺失值数。

| 条件 | 指标 | n | p50 | p90 | p95 | p99 |
|---|---|---:|---:|---:|---:|---:|
|  | solve latency (ms) |  |  |  |  |  |
|  | observation age (control ticks) |  |  |  |  |  |
|  | RTF |  |  |  |  |  |

固定 K 只定义 observation-to-application 的仿真步延迟，不等同 wall-clock realtime。只有 true-async 在推理期间持续推进仿真；是否达到目标频率需结合 RTF、控制周期 overrun 和实际 observation age 判断。若 RTF < 1 或 pacing 超期，保留结果并明确标注未达到目标实时频率。

## 结论

- **Gate：** batch-50 官方 K0 是否通过；若未通过，停止解释该任务的延迟 / async 比较。
- **配对成功率：** N=1 sync K0 与每个 N=1 arm 的 gain/loss、净变化及精确检验结果。
- **时序：** 延迟、observation age、RTF 分位数及目标 pacing 是否满足。
- **范围：** 只陈述本 checkpoint、冻结 task list、runner、planner 配置和设备下观察到的结果。

## 输入格式（analysis v1）

JSONL 每行一个 episode；也接受相同对象的 JSON 数组，或 `{"episodes": [...]}`。必需字段为 `task`、`condition`、`task_id`、`success`（JSON boolean）；可选数值字段为 `latency_ms`、`observation_age_ticks`、`rtf`。默认条件名为 `batch50_official_k0` 和 `n1_sync_k0`，可通过命令行参数更改。每个 task-condition 内 task ID 必须唯一；配对条件的 ID 集必须与 N=1 sync K0 完全一致。
