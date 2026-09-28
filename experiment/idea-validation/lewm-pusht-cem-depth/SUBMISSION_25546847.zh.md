# LeWM PushT CEM 轮数实验提交记录

- PBS job：`25546847.pbs101`，2026-09-25 16:50:36（Asia/Singapore）进入队列；经 `normal` 路由至 `gdev`。
- 同一次提交前，`FREEZE.json`、`run_cem_depth_sweep.py` 和 `run_cem_depth_sweep.pbs` 三个小控制文件传至 `/scratch/users/ntu/yguo017/dino-wm-wall/experiment/idea-validation/lewm-pusht-cem-depth/`；连接保留 JumpHost 与 ASPIRE2A 的 pinned host-key 校验。没有从 login node 读取模型或数据内容。
- 首次 `qstat -x -f`：`job_state=Q`，comment 指向 GPU 资源不足；这是排队快照，没有 `Exit_status` 或实验指标。
- 作业请求 1 GPU、16 CPU、110 GB、2 小时。PBS wrapper 在 compute node allocation 确认后才读取模型/数据并运行，期间每 5 秒把 GPU utilization 与显存写入 job log。
- 原有 iCEM 作业 `25546602.pbs101` 没有被取消、修改或重新提交。不得因查询超时或仍在排队而自动重提本作业。
- 终态须查看 `qstat -x -f` 的 `Exit_status`，以及该 job 的 `job_status`、`final_exit_status.txt`、`stage1_gate.json`、`run_summary.json` 和 GPU log，才可解释 5/10/20/30 轮结果。
