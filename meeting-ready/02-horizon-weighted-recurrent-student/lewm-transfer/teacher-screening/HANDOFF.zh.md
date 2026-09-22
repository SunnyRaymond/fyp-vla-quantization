# 当前执行交接

用户已授权开始实验，遵循根 AGENTS.md 与此前门控规则。额外条件：主 Codex 余额
从非100%突然回到100%时立即停止 agent 推进，由用户更换模型。

- 实验配置：`LEWM_TEACHER_SCREENING_FREEZE.json`、`PROTOCOL.zh.md`，结果产生前已冻结。
- Stage A：balanced_base，K=60/120，fresh valid[544:552]，48 blocks；不通过即停止。
- Stage B：只有 Stage A 通过后实施；其样本、端点与 gate 已预定。无 closed-loop。
- 原 Phase5/6 job 未保存 checkpoint/balanced rows，已用远端目录元数据确认。
  本轮需按原 recipe 在 PBS compute allocation 重建一次并保存，标记 reconstructed。
- Phase2 prepared512 位于 cluster 的 `state-coverage/artifacts/24926383.pbs101/prepared_512/prepared_rows.pt`，已确认存在。
- 远端 experiment root：`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/teacher-screening`。
- 连接复用：`nscc-access/.venv/Scripts/python.exe experiment/reproduction/dino-wm-wall/remote.py`。
  该 helper 使用本地凭据和 pinned host-key verification；不输出凭据。普通 OpenSSH 本轮超时，helper 已成功。
- 当前阶段：Stage A runner/PBS 已通过必要 preflight 并上传，Luna xhigh agent `/root/balanced_evidence` 已完成唯一作业监控。
- PBS job：`25213164.pbs101`，终态 `F`，`Exit_status=0`，walltime `00:06:18`；结果见 `RESULT.zh.md` 与本地 `artifacts/25213164.pbs101/teacher_screening_stage_a_summary.json`。不可重复提交。
- Stage A gate=`FAIL`：K=60 与 K=120 均未通过；Stage B/adaptive CEM=`NOT_RUN_BY_GATE`，closed-loop=`NOT_RUN_BY_SCOPE`。不得扩 K、调阈值、重训或重提作业。
- 余额起点：主 codex used70（remaining30），随后 used72（remaining28）；Luna reserve 原本100%，不是触发条件。
- 短时余额监控：Luna xhigh agent `/root/usage_watch`，最多15分钟；主线程恢复后需重新确认其真实状态。

所有计算、训练、模型/数据读取、benchmark 和重 I/O 只在有非空PBS_JOBID的非login
compute allocation执行。Login仅轻量控制文件/提交/状态。每5秒GPU telemetry写job.log。
不要重启旧 workload、放宽 gate、改 seed、做无关清理或修改其他研究线。
