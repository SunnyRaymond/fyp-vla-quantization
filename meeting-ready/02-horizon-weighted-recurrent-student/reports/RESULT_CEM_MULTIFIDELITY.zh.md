# PushT teacher-verified CEM mechanism gate

## 结论

`M=60` 与 `M=120` 的 sparse teacher verification 都是 **NO-GO**。它们能显著减少被 student 错选的 elites，并把 selected teacher-cost regret 压低，但仍无法阻止 CEM proposal 在 30 轮 update 中逐步偏离 full-teacher reference；两个 intervention 都未通过 frozen first-action drift gates。

因此停止继续扩大 verification budget，也不进入 closed-loop。下一优先级是使用 teacher-labelled on-policy CEM contexts 训练一个新的 CEM-DAgger student，再回到相同的 predictor / fixed-observation mechanism gate。

这个结论只适用于 paired pilot 中预先冻结的 6 个 teacher-success / student-failure cases，以及首个固定 observation；不是 population-level PushT success 结论。

## Frozen design

- job：`24910110.pbs101`，exit code `0`，walltime `00:21:49`
- cases：`[0, 1, 2, 4, 5, 7]`
- 每个 case：30 CEM iterations、300 candidates、top-30、`H=5`
- arms：full teacher、student only、student prefilter + teacher verify `M=60`、student prefilter + teacher verify `M=120`
- 各 arm 共享同一 case/iteration 的 CPU-generated standard-normal innovations；各自递推 `mu/sigma`
- candidate 0 固定为 pre-update mean，并强制包含在 exact-cardinality prefilter superset 中
- checkpoints：iterations `1 / 5 / 10 / 30`
- full-teacher shadow 只在 checkpoints 用于 diagnosis，不计入 intervention timing 或 query budget

独立实验单位是 PushT case；CEM iterations 和 candidates 是 case 内 nested measurements，未作为独立样本。

## Frozen gate 结果

| Arm | Iter-30 first-action RMS median | Iter-30 coordinate abs max | Selected teacher-cost regret median | Cases no worse than student | Decision |
|---|---:|---:|---:|---:|---|
| full teacher | `0.000000` | `0.000000` | `0.000000` | `6/6` | positive-control GO |
| student only | `0.731702` | `2.593516` | `0.096018` | `6/6` | NO-GO |
| prefilter + teacher `M=60` | `0.448167` | `2.365925` | `0.004825` | `6/6` | **NO-GO** |
| prefilter + teacher `M=120` | `0.368307` | `1.364404` | `0.000533` | `6/6` | **NO-GO** |

Primary thresholds：

- first-action RMS median `<= 0.15`
- first-action coordinate absolute maximum `<= 0.25`

Supporting thresholds：

- selected teacher-cost regret median 不高于 student-only median
- 至少 `4/6` cases 不差于 student-only

两个 sparse arms 都通过 supporting thresholds，但都同时失败于两个 primary action gates。`M=120` 相比 student-only 将 median first-action drift 降低约 `49.7%`，仍是 gate 上限的约 `2.46x`；它不是接近通过的结果。

## 误差如何继续累积

| Arm | Iteration | Verified teacher top-30 containment median | Selected teacher-cost regret median | First-action RMS median |
|---|---:|---:|---:|---:|
| student only | 1 | — | `0.042395` | `0.135645` |
| student only | 5 | — | `0.182082` | `0.507735` |
| student only | 10 | — | `0.153710` | `0.647466` |
| student only | 30 | — | `0.096018` | `0.731702` |
| `M=60` | 1 | `0.983333` | `0.000018` | `0.020833` |
| `M=60` | 5 | `0.616667` | `0.057136` | `0.290655` |
| `M=60` | 10 | `0.616667` | `0.014346` | `0.326349` |
| `M=60` | 30 | `0.450000` | `0.004825` | `0.448167` |
| `M=120` | 1 | `1.000000` | `0.000000` | `0.000000` |
| `M=120` | 5 | `0.850000` | `0.004846` | `0.132526` |
| `M=120` | 10 | `0.950000` | `0.000671` | `0.193100` |
| `M=120` | 30 | `0.816667` | `0.000533` | `0.368307` |

`M=120` 在 iteration 1 完全包含 full-teacher top-30，但后续 proposal 已是不同路径；即使 iteration 30 的 median containment 仍有 `0.8167`、selected-cost regret 很小，累计的 `mu/sigma` 差异仍产生明显 first-action drift。这说明只在每轮修正当前候选池的 elite selection，不能回收此前 update 已经造成的 proposal-state 偏移。

## Cost boundary

Iteration-30 median native scoring time：

| Arm | Scoring time | Relative to full teacher | Teacher candidates over all cases/rounds |
|---|---:|---:|---:|
| full teacher | `3512.89 ms` | baseline | `54000` |
| student only | `11.32 ms` | `99.68%` reduction | `0` |
| `M=60` | `721.06 ms` | `79.47%` reduction | `10800` |
| `M=120` | `1419.64 ms` | `59.59%` reduction | `21600` |

这些是 fixed-observation predictor-boundary timings，不包含环境 step，也不是 closed-loop wall-clock speedup。checkpoint 的 full-teacher oracle diagnosis 已从 intervention timing 与 query count 排除。

## Validity

- all six cases complete：`true`
- all outputs finite：`true`
- silent fallback：`false`
- exact superset cardinality：`true`
- full-teacher positive control：`GO`
- GPU telemetry：A100-SXM4-40GB，主要 samples 为 `99–100%` utilization，约 `23655 MiB / 40960 MiB`

## 决策

1. 不继续测试 `M=180` 或其它 post-hoc verification sizes。
2. 不运行 closed-loop。
3. 下一实验只训练一个新 student：teacher labels 来自 on-policy CEM contexts，直接覆盖 student 会访问的 proposal distributions。
4. 新 student 先通过 frozen held-out predictor / fixed-observation gate；失败即保留 NO-GO，不增加 closed-loop budget。

## Artifacts

- aggregate summary：`artifacts/24910110.pbs101/cem_multifidelity_summary.json`
- per-case traces：`artifacts/24910110.pbs101/case_00.json` 等 6 个 JSON
- execution record：`artifacts/24910110.pbs101/job.log`
- GPU telemetry：`artifacts/24910110.pbs101/gpu_usage.csv`

