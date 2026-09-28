# Teacher baseline 启动尝试：25531848.pbs101

权威 PBS `qstat -x -f` 记录：作业 `lewm_teacher_base` 在 `gdev` 结束，状态 `F`、`Exit_status=1`，compute host `x1000c0s6b0n1`，walltime `00:02:36`。runner 与 PBS 状态文件均为 exit 1。

该尝试没有产生 baseline outcome。PBS wrapper 与 runner 前置路径检查确认 HDF5 和 checkpoint 存在、job-private cache symlink 指向预期 staging 文件，metadata 大小分别为 46,300,921,856 与 72,345,781 bytes。runner 随后在 `stable_pretraining` imported-root guard 失败：它实际位于同一 staging Python venv 的 `lib/python3.11/site-packages`，而 revision 1 错误要求该依赖属于 `le-wm` 源码目录。错误发生在 Hydra 配置组合、HDF5Dataset 创建/读取、`torch.load` 和 World.evaluate 之前。没有 `selected_tasks.json` 或 `summary.json`，因此 success count 与 `>=5/50` 工程门槛均未测量；不得报告为 0/50，也不进入四臂实验。

`job.log` 有 31 条五秒 GPU telemetry 样本，均为 0% utilization、1 MiB used。原始 status、execution identity、GPU 信息与小型 job log 保存在 `artifacts/25531848.pbs101/`。Revision 2 只把依赖 guard 改为：`Path(sys.prefix).resolve()` 必须精确等于 staged `$STAGE_ROOT/venv`，并要求 `stable_pretraining` 在该 venv 内；`stable_worldmodel` 仍要求位于 staged source root。revision-1 freeze 作为 `FREEZE.revision1-attempt-25531848.json` 保留。其他实验参数、模型、数据、样本选择、预处理、gate 与资源没有改变。

结果分类：`STARTUP_FAILURE / NO OUTCOME`，不是策略质量 NO-GO。此后唯一允许的动作是 revision 2 静态检查通过后重跑一次相同 teacher-only baseline；后续四臂仍需先获得有效 `>=5/50` teacher baseline。
