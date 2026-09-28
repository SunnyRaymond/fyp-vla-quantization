# LeWM PushT CEM 迭代深度实验

本实验比较 official LeWM teacher 的 CEM 在 **5、10、20、30 轮**停止时的 dataset-task success 和同步 planner latency。每个 arm 都在同一批 50 个固定 source tasks 上运行完整的 `World.evaluate` rollout；CEM 候选数固定为 300，elite 数固定为 30，模型、预处理、seed 42、task rows 与官方 `PlanConfig.warm_start=true` 沿用有效基线。

## 冻结设计

先运行 `cem_300_30_iter_30`，并要求它在同一批 50 个任务上逐项复现 `25534994.pbs101` 的 success vector（49/50）；不匹配、无效或低于 5/50 时停止，不运行其余 arm。通过后按冻结顺序执行 5、10、20 轮。

每个 depth 使用自己的 seed-42 solver，从所选任务列表开始完成真实 replanning 和环境 rollout。逐轮数不同会消耗不同数量的随机数，因此任务之间的候选 innovations 不构成精确配对；配对单位是相同 source episode/task，后续 action、状态和轨迹可以分叉。保持官方 `warm_start=true`，但当前 `horizon=receding_horizon=5` 会执行完整计划，没有剩余动作尾段传给下一次 solve。

主要质量结果是 50 项 success vector、相对 30 轮的成功数差，以及双方成功/仅 30 轮成功/仅 treatment 成功/双方失败的配对计数。计时记录每次 `solver.solve()` 前后的 CUDA 同步时间、所有 solve 时长、调用数、总 solve time、首个调用之后的 median 及样本数，并估算 CEM candidate-cost evaluations。固定 seed、单次运行和固定 arm 顺序下的 latency 仅作描述性比较；不是 random-reset simulator success，也不是正式 noninferiority 结论。

## 运行与产物

- 冻结协议：[FREEZE.json](FREEZE.json)
- PBS 入口：[run_cem_depth_sweep.pbs](run_cem_depth_sweep.pbs)
- Runner：[run_cem_depth_sweep.py](run_cem_depth_sweep.py)
- PBS 产物目标：`experiment/idea-validation/lewm-pusht-cem-depth/artifacts/$PBS_JOBID/`
- PBS wrapper 每 5 秒将 GPU utilization 和 memory 追加到 `job.log`，并在模型、数据 I/O 和推理前验证 PBS allocation 与 compute-node hostname。

PBS 作业 `25546847.pbs101` 已于 2026-09-25 正常结束；四组结果与解释见 [RESULTS_25546847.zh.md](RESULTS_25546847.zh.md)。不得因结果不理想而换 task、重跑、调参、追加 depth 或追加 seed。
