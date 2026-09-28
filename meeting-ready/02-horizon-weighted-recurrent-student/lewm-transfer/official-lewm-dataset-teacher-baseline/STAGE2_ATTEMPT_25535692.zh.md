# Stage 2 startup attempt：25535692.pbs101

PBS 历史状态为 `F` / `Exit_status=1`，在 `x1000c0s1b0n0` 上运行 `00:03:24`。该作业已越过 arm-order guard，进入真实 compute allocation；GPU telemetry 每 5 秒写入 `job.log`。原始小型文件位于 `artifacts/25535692.pbs101/`。

runner 先核验了 baseline job、49/50 gate、冻结的 50 task rows 和 arm order；随后按正常 evaluator 加载 HDF5 的 action/proprio/state 列、拟合全数据 scaler，并确认每个 `row_index` 对应的 `episode_idx/start_step`。此后在模型权重载入前退出：实现误把 `load_modules` 从 `run_adaptive_teacher_schedule` 模块调用；实际该函数定义在 `run_official_pusht_cem.py`。

本 job 没有加载 teacher/student checkpoint，没有创建四臂 planner，也没有生成 episode outcome 或 summary。状态应为 `NO_RESULT`，不可解释为 0/50。小型 `job.log` 记录了明确 traceback；没有结果行可分析。修复只把调用改为 `old_router.load_modules(...)`，得到 upstream schedule 与 reference；冻结的数据任务、process、solver、arm、seed、order 与 gate 均不变。
