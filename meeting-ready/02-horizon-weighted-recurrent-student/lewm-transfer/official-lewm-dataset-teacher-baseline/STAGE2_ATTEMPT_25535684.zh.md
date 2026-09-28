# Stage 2 startup attempt：25535684.pbs101

权威 `qstat -x -f` 返回终态 `F`、`Exit_status=1`，作业在 compute node `x1000c0s1b0n0` 运行约 10 秒后退出。PBS 通过 `normal` route 路由到 `gdev`，资源申请为 1 GPU、16 CPU、110 GB RAM、2 小时。

小型日志显示失败来自 frozen arm-order guard：程序洗牌后生成了正确的预期顺序，但将 Python `list` 与 `tuple` 直接比较，所以判定为不相等并提前退出。错误在 baseline JSON 检查后、写入 run order 前触发；没有打开 HDF5 dataset、加载 checkpoint、创建 planner 或执行 episode。GPU telemetry 仅有一条采样（A100 显存 1 MiB）；该作业没有 Stage 2 结果，必须标记 `NO_RESULT`，不能当作任何一臂表现。

原始 `job_status`、`final_exit_status.txt`、`execution_identity.txt`、`gpu_info.csv` 和 `job.log` 保存在 `artifacts/25535684.pbs101/`。修正仅把顺序比较的左侧转成 tuple。冻结的 task rows/order、四臂、预处理、seed 42 native CEM stream、统计方案与 success/latency 门槛均不变。修复后静态核验通过，唯一新 attempt 为 `25535692.pbs101`，首次权威状态为 `R`；本次失败作业不包含实验结果。
