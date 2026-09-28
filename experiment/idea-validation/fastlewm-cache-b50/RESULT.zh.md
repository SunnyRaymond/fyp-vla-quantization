# FastLeWM 50 active environments exact-cache 结果

**结论：整体 `NO_GO`。** 固定 observation 的 50-environment solve 通过 exact fidelity、性能和显存门；完整 closed-loop 未通过冻结的逐 solve actions/costs exact gate。该失败已保留，未调参或重跑。

实验固定 Fast-LeWM source `de3e9dac539f5bbe6ff1656a2fb00938d62a3c7d`、checkpoint revision `f95379fe193c8bfc6a59c9d8437d5052bd72ff71`、`stable-worldmodel==0.0.6`；50 个冻结 task rows，CEM `300 / top-30 / 30 iterations`，seeds 42/43/44。GPU entry 的 cache identity、逐字节比较、environment eviction、signed-zero/device guard 和官方 solver deepcopy closure identity self-check 均通过。

## Fixed-observation：50 active environments

这里的50指同一次solve中的active environments数量。沿用native solver的`batch_size=1`，encoder实际逐environment处理B=1；本实验未测试vectorized encoder batch B=50。

| Seed | Native median | Cache median | Paired latency reduction | Peak allocated ratio | Full trace |
|---|---:|---:|---:|---:|---|
| 42 | 24.043 s | 10.475 s | 56.42% | 1.01193 | bitwise exact |
| 43 | 24.263 s | 10.589 s | 56.31% | 1.01193 | bitwise exact |
| 44 | 25.027 s | 10.871 s | 56.58% | 1.01193 | bitwise exact |

每个 seed 的 native/cache trace 都覆盖 1,500 次 `get_cost`、450,000 个候选评分；所有候选 actions、costs 和 returned actions/costs bitwise 相等。缓存每次 solve 计算 100 个 embeddings、命中 2,900 次，最多常驻 2 项，payload peak 为 1,205,760 bytes。三 seed 的 latency 与 memory frozen gates 全部通过。

## Closed-loop 50-task evaluation

| Seed | Native / cache successes | Solve 0 | Solve 1 的差异 | Raw E2E wall（native / cache） |
|---|---:|---|---|---:|
| 42 | 49 / 49 | actions、costs exact | 550/2,500 action values 不同；11/50 costs 不同；最大绝对 action 差 1.4826 | 152.07 / 71.61 s |
| 43 | 49 / 49 | actions、costs exact | 1,000/2,500 action values 不同；20/50 costs 不同；最大绝对 action 差 3.6580 | 154.67 / 71.82 s |
| 44 | 48 / 48 | actions、costs exact | 500/2,500 action values 不同；10/50 costs 不同；最大绝对 action 差 1.7741 | 153.24 / 71.52 s |

三组的 50-task ordered outcomes 均相同，native outcomes 也均匹配既有 control；active environment、cost-call 和 candidate-score 计数匹配。但每组第二次 replan 的 final actions 与 costs 都不 exact，因此 closed-loop fidelity gate 失败。以上 raw E2E wall 包含 runner 的 fixed-benchmark warmup/timed overhead；实测 overhead 分别为 native/cache：seed42 `96.46/42.97 s`、seed43 `98.02/43.08 s`、seed44 `97.47/42.89 s`。由于 replan outputs 未保持 exact，这些 raw timing 不构成可部署 closed-loop speedup 证据。

本次结果没有在每次 replan 保存输入 observation/goal，也没有保存当时的 RNG states。两臂设置了相同的 `cfg.seed` 与 `solver.seed`，且第一 solve 输出 exact，但现有记录不能判断第二 solve 的输入或 RNG state 是否相同，也不能把差异归因于 cache、状态演化或 RNG。固定 observation 的 exact 结果仍独立成立；要解释 closed-loop 分歧需单独设计能记录并配对 replan context/state 的诊断，当前结果本身不支持因果结论。

PBS job `25577481.pbs101` 在 `x1000c1s5b0n0` 结束，walltime `00:33:37`，PBS `Exit_status=4`、wrapper `EXIT_STATUS=4`、`Stageout_status=1`。Exit status 4 对应实验脚本因 closed-loop scientific exact gate 未通过而返回 `NO_GO`；全部冻结配对均已完成。GPU telemetry 保存在该 job 的 `job.log`/`gpu_info.csv`。

精简汇总见 `artifacts/25577481.pbs101/compact_summary.json`；原始逐 solve 结果、完整 `run_summary.json`、`entry_self_check.json`、PBS wrapper 状态和 telemetry 均保留在同名 artifact 目录中。
