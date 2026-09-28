# Job 状态

## 原始启动失败：25531848.pbs101

- `qstat -x -f`：`F`，`Exit_status=1`，walltime `00:02:36`，host `x1000c0s6b0n1`。
- import-root guard 错误要求 `stable_pretraining` 位于 `le-wm` 源码树；它实际安装在同 staging venv 的 site-packages。失败在配置加载、任务选择、HDF5/teacher 加载和评测之前。
- `summary.json` 与 `selected_tasks.json` 均不存在；success 数和 `>=5/50` 工程门槛无法评估。不要把它记为 0/50。
- 小型原始证据保存在 `artifacts/25531848.pbs101/`；revision-1 freeze 快照为 `FREEZE.revision1-attempt-25531848.json`。

## Revision-2 唯一重跑：25534994.pbs101

- 通过授权 `normal` route 接受并路由至 `gdev`；资源请求为 1 GPU、16 CPU、110 GB RAM、2 小时。
- 权威 `qstat -x -f` 终态：`F` / `Exit_status=0`，host `x1000c3s7b0n1`，walltime `00:06:36`；PBS report 的 GPU peak memory 为 628 MB、SM peak utilization 75%。
- runner summary：`validity=PASS`，严格 50 行且 50 个不同 source episode，success `49/50`（98%），所有数值有限；超过冻结的 `>=5/50` 工程 floor。HDF5、teacher checkpoint、dataset selection seed 42、upstream CEM 参数和 callables 与 freeze 相符。该门槛只决定是否进入独立 paired 实验，不是统计检验。
- GPU 遥测在 `job.log` 中有 79 条，间隔 5 秒，A100 显存记录最高 639 MiB；runner 自报 evaluation 74.15 秒、总时长 239.57 秒。
- 四臂条件设计已满足启动条件。Stage 2 必须使用本作业 `selected_tasks.json` 的原始 50 项；teacher-only arm 的逐项 success vector 若不与本 baseline 完全一致，则结果 fail-closed，报告差异，不重选任务或重跑。
- 小型原始证据保存在 `artifacts/25534994.pbs101/`；完整判定见 `BASELINE_RESULT_25534994.zh.md`。

## Stage 2 首次启动：25535684.pbs101

- 权威 `qstat -x -f`：`F` / `Exit_status=1`，运行 10 秒后退出；执行节点 `x1000c0s1b0n0`。仅 `job.log` 中 1 条 GPU telemetry，显存 1 MiB。
- 原因是启动前的 Python guard 将洗牌后的 `list` 与预期 `tuple` 直接比较，尽管实际 arm order 正确却恒不相等。失败发生在写 run-order 之后、加载 HDF5/dataset/teacher/student 或运行 CEM 之前；无实验 outcome。
- 保留的小型原始失败证据在 `artifacts/25535684.pbs101/`。本地修复只将比较改为 `tuple(arm_order)`；之后的 PBS attempt 使用独立 job ID。该失败不改变科学设计、task、seed 或 gate。

## 修复后的唯一 Stage 2 attempt：25535692.pbs101

- 只修正 list/tuple arm-order assertion；修复后 AST/CLI/PBS shell 与 frozen-order preflight 通过。未改变任务、处理、arm、solver、统计或 gate。
- 经 `normal` route 提交并路由到 `gdev`；资源为 1 GPU、16 CPU、110 GB RAM、2 小时。首次权威 `qstat -x -f` 为 `R`，compute host `x1000c0s1b0n0`；启动 17 秒，仍在运行。
- 权威 `qstat -x -f` 终态：`F` / `Exit_status=1`，walltime `00:03:24`，host `x1000c0s1b0n0`。
- 本作业通过 arm-order guard，并已打开 HDF5、缓存 action/proprio/state、检查 50 个 row↔episode_idx/start_step；但在 teacher/student checkpoint 加载前因调用错模块失败：`load_modules` 属于 `run_official_pusht_cem.py`，不是 `run_adaptive_teacher_schedule.py`。没有 summary 或 episode results，记为 `NO_RESULT`，不是实验指标。
- 保留小型日志和 task/order sidecars 于 `artifacts/25535692.pbs101/`。随后新 attempt `25535711.pbs101` 进入 Stage 2，但因 callback 接口不完整，在首次 CEM 输出组装时失败；详见下一段。

## Stage 2 callback 接口修复 attempt：25535711.pbs101

- 权威 `qstat -x -f`：`F` / `Exit_status=1`，walltime `00:05:02`，host `x1000c0s6b0n1`。
- 本作业通过 baseline/task/arm-order guard、加载 teacher/student、进入 pinned `CEMSolver.solve`。首个 solve 在 output callbacks 收尾时触发 AttributeError，因为新写的轻量 callback 缺少 CEM 所需 `.history`（之前已确认它还会访问 `reset/start_batch/__call__/end_solve/output_key`）。没有 episode outcome 或 summary，仍为 `NO_RESULT`。
- 小型状态和 job.log 保存在 `artifacts/25535711.pbs101/`。修复为重用已验证的 `old_router.TraceCallback(retain_tensors=False)`，它提供 solver 所有 callback 属性和轻量候选 shape/finite 记录，不改变 CEM/任务/门槛；静态核验后按持续授权唯一重提。

## Stage 2 批次边界 attempt：25535774.pbs101

- PBS 终态 `F` / `Exit_status=0`，walltime `00:03:16`；runner summary 为 `FAIL_CLOSED`。Pinned `CEMSolver` 每个 environment chunk 调用一次 callback `start_batch()`，再执行 30 个 CEM steps。原 route counter 只在整次 `solver.solve` 清零，因此 late7/uniform7 各只调用 teacher 14 次，未按冻结策略逐 batch 路由。
- 该 job 有 200 条结果，teacher-only 也复现了 49/50，但 schedule 失效；不得用它的学生/late 成功率做策略结论。证据保留在 `artifacts/25535774.pbs101/`。

## 修正后的 Stage 2：25535873.pbs101

- Stage2 专用 callback 在 pinned `start_batch()` 边界重置 route round counter，并按每 30 个 cost evaluation 分组检查冻结 schedule。任务、四臂、seed、预处理、solver、统计方法和 gates 均未改变；本地 AST、CLI 与纯 schedule-helper 检查通过后唯一提交。
- 权威 PBS 终态 `F` / `Exit_status=0`，walltime `00:08:09`，host `x1000c1s1b0n1`。有效性全过：50 tasks/200 outcomes、无缺失或重复、50 个唯一 source episodes、teacher-only 成功向量精确重现 baseline（49/50），obs/RNG 配对、routes 与 finite 均通过。
- 四臂成功：student-only 11/50、late7 26/50、teacher-only 49/50、uniform7 18/50。Late7 比 student 多 15，exact cluster sign-flip `p=0.0007286`，收益门通过；但 teacher gap 为 23（阈值 ≤5），planner latency ratio 为 0.7583（阈值 ≤0.70），整体为有效结果中的 `FAIL`，不是 `FAIL_CLOSED`。详见 `STAGE2_RESULT_25535873.zh.md`；没有提交新作业。
