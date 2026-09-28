# LeWM PushT CEM budget sweep

这是一个独立实验，用于在固定官方 LeWM teacher、PushT dataset protocol 与相同 50 个任务上比较 CEM 预算。结果见 [RESULT_25542796.zh.md](RESULT_25542796.zh.md)。未修改 upstream 或既有实验。

## 冻结问题

每个 solver 都运行 30 iterations。预算按 `num_samples/topk` 记：

| Arm | Candidates | Elites | 用途 |
|---|---:|---:|---|
| `cem_300_30` | 300 | 30 | 同 allocation 的 control；也是 Stage 1 gate |
| `cem_150_30` | 150 | 30 | 只减 candidates |
| `cem_300_15` | 300 | 15 | 只减 elites |
| `cem_150_15` | 150 | 15 | 目标 10% elite-ratio 对照 |
| `cem_100_10` | 100 | 10 | 更大幅度的 10% elite-ratio 对照 |

所有 arm 复用 [`25534994.pbs101` 的 50 个 task rows](../../../meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/official-lewm-dataset-teacher-baseline/artifacts/25534994.pbs101/selected_tasks.json)、官方 checkpoint、dataset scalers、image transforms、state/goal callables、evaluation budget 与 seed。Stage 1 首先重新运行 300/30；只有当其 50 项 success vector 与该 reference baseline 完全一致、结果有效且至少 5/50 成功时，runner 才运行其余四个 arm。此 5/50 是实验继续门槛。

`FREEZE.json` 保存 arm、task 来源、source commit references、评估 protocol、资源要求和停止规则。上游 pinned 文件本身由远端 staging 提供；PBS runner 会检查其导入 root 与配置语义，不声称从无 Git metadata 的 staging tree 验证 commit hash。

## 计时与结果

runner 会在每次 `solver.solve()` 前后同步 CUDA，保留每次 solve 时长、planner solve 总时、solve 调用数、根据 pinned CEM batch / iteration 生命周期计算的 candidate cost evaluation 数、evaluator walltime，以及逐 task success rows。报告包含完整 planner solve 总时（包括第一个 cold call）、每个 solve 的时长、`second_solve_seconds`，以及剔除第一个 solve 后的中位数和 `post_first_solve_sample_count`。若每次 evaluation 只有两个 `solver.solve()` 调用，剔除首个调用后的中位数就只有一个样本，且与第二次调用相同；这只能作为单次描述性计时，不作为稳健速度估计。GPU utilization 和 memory 由 PBS wrapper 每 5 秒写入 `job.log`。

结果只支持这 50 个固定 dataset tasks 上的 paired 描述性比较。可报告每 arm 的 success 差、control-only / treatment-only success 配对计数和第二次 solve 的同步计时比。`FREEZE.json` 中的预算 screen（最多少 5 个成功且第二次 solve 至少快 20%）是单样本描述性筛选，不是稳健速度估计或 formal noninferiority claim。不要把固定 dataset evaluation 写成 random-reset simulator success。

## PBS 执行

作业入口为 [`run_cem_budget_sweep.pbs`](run_cem_budget_sweep.pbs)，已提交并运行一次：`25542796.pbs101`。五组结果完整写出，但 PBS 因最终完成提示的 `TypeError` 以非零状态结束；原执行脚本与错误原因见结果报告。脚本按项目既有路径约定查找 baseline artifacts、LeWM staging、stable-worldmodel staging 与 virtual environment；如集群路径变化，只通过相应环境变量改路径，不在 login node 上执行数据/模型读取、推理、下载或安装。输出放在 `artifacts/$PBS_JOBID/`。

预期小型输出：

- `selected_tasks.json`、`run_order.json`
- `stage1_gate.json`，以及每个完成 arm 的 `arms/<arm>.json`
- `arms/episode_results.jsonl` 与 `arms/solve_times.jsonl`
- `run_summary.json`、`execution_identity.txt`、`job.log`、`job_status`

PBS 脚本在 runner 前后都检查实际 allocation。Python runner 在检查 `PBS_JOBID`、`PBS_NODEFILE`、hostname 前不导入模型或 HDF5 依赖。
