# Stage 2 callback-interface attempt：25535711.pbs101

权威 PBS 状态为 `F` / `Exit_status=1`，walltime `00:05:02`，host `x1000c0s6b0n1`。这是一次 `NO_RESULT`：作业读取了相同 baseline tasks、完成数据身份检查、加载模型并进入 pinned CEM solver，但没有 action execution、episode outcomes 或 summary。

traceback 显示 pinned `CEMSolver.solve` 在首个 CEM call 的结果整理阶段访问 `cb.history` 时失败。进一步只读核对 pinned `cem.py` 的全部 callback 访问点：`reset()`、`start_batch()`、`__call__()`、`end_solve()`、`output_key` 和 `history`。自定义 RoundCallback 漏掉了 `history`，因此在 solver 返回前异常退出。作业 `job.log` 留有 traceback 和 GPU telemetry；原始小型文件位于 `artifacts/25535711.pbs101/`。

修复是移除重复实现，改用 Stage 1 已验证且满足完整 callback 接口的 `old_router.TraceCallback(retain_tensors=False)`。它只留小型每轮 finite flags 与 candidate shape，不保留大型 trace tensor；CEM 参数、cost route、task/preprocess、seed、统计与 gate 不变。当前无有效臂结果。
