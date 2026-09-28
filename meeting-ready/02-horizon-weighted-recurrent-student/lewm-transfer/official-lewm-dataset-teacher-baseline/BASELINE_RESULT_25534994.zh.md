# Teacher-only dataset baseline：25534994.pbs101

权威 PBS 历史查询确认作业终态 `F`、`Exit_status=0`；作业在 `x1000c3s7b0n1` 上运行，实际 walltime `00:06:36`，资源为 1 GPU、16 CPU、110 GB RAM、2 小时。调度器记录的 GPU 最大显存为 628 MB、SM 利用率峰值 75%。

runner summary 为 `COMPLETED`、`validity=PASS`：固定选择 50 个 dataset rows，50 个 source `episode_idx` 均不同；每个任务有成功结果，49/50 成功（98%），指标有限。`selected_tasks.json` 的 selection seed 是 42。该作业使用 pinned upstream PushT config 的 dataset evaluator、`goal_offset_steps=25`、`eval_budget=50`、`world.max_episode_steps=100`（有效 rollout budget 仍为 50）、官方 state/goal callables、全数据 `StandardScaler` 和 ImageNet/224 transform。它是 protocol-aligned teacher baseline，并非逐字运行 upstream `eval.py`：teacher 权重由已验证的项目 loader 反序列化，且视频输出为 `None`。

数据与权重身份由 runner 记录：dataset 为 `/scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home/pusht_expert_train.h5`（46,300,921,856 bytes）；official teacher 为 `/scratch/users/ntu/yguo017/lewm-pusht-iteration/stablewm_home/pusht/lewm_object.ckpt`（72,345,781 bytes）。CEM 使用 pinned `CEMSolver`，seed 42、batch 1、300 candidates、30 rounds、topk 30、`var_scale=1`、CUDA。

baseline 超过预先冻结的 5/50 工程 floor，因此允许开展已冻结的四臂对照。这个 floor 仅是是否继续的工程门槛，不是显著性结论。四臂只能重用本作业的 50 个 `selected_tasks.json` rows，不能重选、扩样或过滤；Stage 2 teacher-only arm 的 50 项 success vector 必须逐项等于本 baseline，否则完整报告 mismatch 并 fail-closed。

作业日志中有 79 条每 5 秒 GPU telemetry，A100 显存峰值记录为 639 MiB；runner 报告 evaluation `74.1479 s`、总 runner `239.5714 s`。原始小型文件保存在 `artifacts/25534994.pbs101/`：`summary.json`、`selected_tasks.json`、`job_status`、`final_exit_status.txt`、`execution_identity.txt`、`job.log` 和 `gpu_info.csv`。

**判定：teacher baseline 有效，四臂 follow-up 达到冻结的工程启动门槛；尚无四臂结果或 student/late-teacher 结论。**
