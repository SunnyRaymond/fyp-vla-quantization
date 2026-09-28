# Student-induced shadow pilot：episode-selection preflight

## 当前范围

本目录只准备 episode-selection preflight。CPU PBS job 在获批的 compute allocation 内读取 PushT HDF5 的 `ep_len`、`ep_offset` 小向量，并仅对最终 32 个所选 row 读取 `episode_idx[row_index]` 与 `step_idx[row_index]` 标量作核验；按冻结规则生成 32 个独立 source episode，每个 episode 只选一个 `row_index` 和 `start_step`。不读取 pixels、actions、checkpoint 或模型；不运行 simulator，不训练，不提交作业。

`SELECTION_FREEZE.json` 在作业前固定了数据路径、baseline `25534994.pbs101` 的 50 个精确 episode IDs、旧 valid-prefix 排除规则、候选选择 seed `20260924` 和 16/16 分组。脚本只额外读取 baseline 的小型 `selected_tasks.json`，逐项核对 50 个 task rows 与冻结 IDs；不读取其他旧 result/freeze 文件。它从 HDF5 的 `ep_len` 元数据按 recurrent-student 源代码重建 `ep_len>=26` 的 episode 列表，用 `random.Random(20300903)` shuffle 后排除前 600 个 episode IDs。若 baseline IDs 不匹配、valid episode list 不足 600 个、必要元数据缺失或候选不足 32 个 episode，脚本立即失败，不会放宽规则。

## 选择与隔离规则

1. 在 compute allocation 中读取 HDF5 的 `ep_len` 与 `ep_offset` 小向量。只为最终 32 个选择读取 `episode_idx[row_index]` 和 `step_idx[row_index]` 两个标量作一致性校验；不读取 pixels、actions 或完整 row columns。候选起点限定为 `0 <= start_step <= ep_len - 26`，`row_index = ep_offset + start_step`。
2. 从 baseline 的小型 `selected_tasks.json` 读取 50 项 `episode_idx`，核对任务数、唯一数及与 freeze 中 50 个 IDs 的集合完全相同。之后按 source episode 排除这些 IDs。
3. 重建 recurrent-student 旧 valid episode list：`[i for i, ep_len in enumerate(ep_len_column) if ep_len >= 26]`，用 `random.Random(20300903)` shuffle，并按 episode ID 排除前 600 项。此处不是全局 row 的 `valid_rows[0:600]`。
4. 对排序后的剩余 episode IDs 用 `random.Random(20260924)` shuffle 并取前 32；每个 episode 用同一个 RNG 在合法 start-step 范围内抽一个起点。前 16 个标为 `collection`，后 16 个标为 `reserved_holdout`。GPU diagnostic 不会把 holdout task rows 传给 `World.evaluate`、打分或用于决策；为保留 official evaluator 语义，dataset-wide preprocessing 可能读取 `action/proprio/state` 列中对应的数据值，因此不声称 holdout 的底层列数据完全未读。

## 后续 shadow-only gate（不属于本次 CPU preflight）

GPU diagnostic 只把 `collection` 16 个 episode 作为 evaluator task；holdout task 不评估、不打分、不用于阈值或决策。每个 collection episode 单独新建 `World` 和 CEM solver，调用 pinned `World.evaluate` dataset mode，并保留 dataset start row、goal offset 25、50-step budget 与 student-only action commits。t25 是从 t0 起实际提交 25 次 `env.step` 后发生的第二次 native solve；逐项记录 task identity、每个真实 batch 的 ordinal、提交动作、simulator state 和 pre-solve RNG。若 episode 在 25 次 transition 前终止，记录 `terminal_before_t25` 和真实 active step count，不补样或强行继续。每 episode solver seed 为 42，t0 到 t25 native generator 连续前进、不 reseed。

前 4 个 collection task 先做 shadow-off/shadow-on 配对 non-interference gate，使用相同初始 dataset state 和 solver seed 42；精确核对每次 solve 的 pre/post native RNG、候选/成本/mean/var、solver action、真实提交动作、每步 state trace 和 success。通过后复用 shadow-on 轨迹作为这 4 个 collection 结果。Teacher score 在 `Route.get_cost` 内对 student 调用前 clone 的同一 `info` 和 candidates 旁路计算；CEM 只接收原样 student cost。单 episode `World.evaluate` 令每个 solve 只含一个 active task 和一个真实 CEM batch，round 计数在每次真实 solve/batch 重置。`t25 - t0` 的 regret 变化反映 on-policy simulator state、action history 和 CEM candidate distribution 的联合变化，不解释成单独的 state effect。

主指标为 round30 standardized elite regret 的 episode-paired `t25 - t0`；regret 是 `(mean teacher cost on student Top30 - mean teacher cost on teacher Top30) / max(population std of all 300 teacher costs, 1e-6)`。Secondary 指标为 rounds10/20 的 regret delta 和 rounds10/20/30 `recall@120 = teacher Top30 recovered in student Top120 / 30`。预先门槛为：至少 12/16 个 collection episodes 同时有 t0 和 t25；primary episode-level delta 中位数至少 `+0.05`；且至少 `ceil(0.75 × matched episodes)` 个 episode 的差值严格为正。候选、CEM rounds 与两个 snapshots 均是 episode 内重复测量。未过门即停止，不训练；holdout task 不用于本次诊断，留待未来独立确认。

所有 HDF5/model/simulator/shadow computation 均须在带真实 `PBS_JOBID` 且 hostname 符合 allocation 的 PBS compute job 中执行。登录节点只作轻量 qsub/qstat 与小文件控制。每 5 秒把 GPU utilization 与显存 telemetry 记入本作业 `job.log`；本次只准备 runner/PBS，不提交 GPU job。

## 作业资源与边界

脚本及 PBS wrapper 参考项目已有 CPU preflight 配置：`normal` queue、1 node、4 CPUs、16 GB RAM、20 分钟，无 GPU。Wrapper 在 Python 启动前检查 `PBS_JOBID`、非空 `PBS_NODEFILE`、实际 hostname 不含 login/head/submit，且 hostname 出现在 nodefile。PBS compute node 是唯一允许读取 HDF5 元数据的环境；login node 不访问 HDF5。Preflight 完成后只需回收新的小型 `selection_manifest.json` 与 job log/status。

无下载、hash、大文件复制、模型载入或依赖安装。Job 只在唯一 job artifact 目录写 `selection_manifest.json` 和简短 `job.log/status`。本目录没有提交作业，也没有改写旧 freeze/results。
