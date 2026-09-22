# CEM elite-boundary ranking handoff

状态：**设计、runner、PBS wrapper 已完成；轻量 preflight PASS；已通过 review、上传并提交唯一 bounded PBS。job 已正常结束，已取回 small evidence；冻结 gate 为 FAIL，停止后续实验。**

已冻结文件：

- `FREEZE.json`：同 anchor-aligned step3000 起点；historical control=`25239551.pbs101` treatment checkpoint；同 mixed rows、1000 updates、AdamW/schedule；新增 pairwise positive ranks20:30、negative30:40、100 pairs/context、lambda=`0.05`、temperature=`1.0`、degenerate teacher-std abort。
- `PROTOCOL.zh.md`：fresh `valid[568:576]`、selection seed `20300903`、新 action-prefix seeds `20301001/20301002`、shared frozen-start CEM banks、primary/forgetting gates 和 evidence boundary。
- `run_cem_boundary_ranking.py`：复用已验证 CEM runner 的 fresh bank/CEM/metric/gate helper；historical control 明示非 concurrent；treatment 保留 latent+score loss 并加 boundary term；compute-only run guard。
- `run_cem_boundary_ranking.pbs`：normal，1 GPU/16 CPU/110GB/30min，PBS/non-login guard，5 秒 direct-PID GPU telemetry，cleanup/final status。

轻量 preflight：

- Python `py_compile`：PASS。
- `FREEZE.json` JSON parse：PASS。
- PBS `bash -n`：PASS。
- runner `--mode preflight`：PASS；`model_work_started=false`，interface semantics equal，512×64、1000 updates、pair 20:30/30:40、lambda .05/temperature 1.0、fresh 568:576/48 blocks、official CEM/closed-loop NOT_RUN_BY_SCOPE。
- blocker 修复：`validate_freeze` 锁定 `25239551.pbs101`、treatment arm、exact checkpoint/rows/summary filenames 与 canonical artifact directory；compute run 逐项比较 args 与 historical summary 的 resolved exact paths，并验证 trace updates=1000、mixed tail counts/provenance、anchor start provenance。

提交记录：

- usage_watch 最终限定 review：`PASS`，无剩余 blocker；未加载模型/数据、未运行训练。
- 远端小文件已串行上传并核对：`FREEZE.json`、`PROTOCOL.zh.md`、`HANDOFF.zh.md`、runner、PBS wrapper；未上传模型、rows 或 HDF5。
- 唯一 PBS：`25263388.pbs101`，`gdev`，1 GPU/16 CPU/110GB/30min；提交时间 `2026-09-22 19:06:19`。
- 首次权威状态：`job_state=Q`；comment=`Not Running: would exceed overall limit on resource ngpus in queue gdev`。未 qalter、qdel 或重复提交。

依赖（只在 PBS compute 读取）：

- rows：`.../cem-distribution-distill/artifacts/25239551.pbs101/cem_distribution_train_rows.pt`
- historical control checkpoint：`.../cem-distribution-distill/artifacts/25239551.pbs101/treatment_step1000.pt`
- historical summary：`.../cem-distribution-distill/artifacts/25239551.pbs101/cem_distribution_distill_summary.json`
- anchor step3000：`.../anchor-aligned-bank/artifacts/25223859.pbs101/anchor_aligned_bank_step3000.pt`

不要在本地加载上述模型/rows；review 前不要上传或提交。

终态与结果：

- PBS `25263388.pbs101`：`job_state=F`，`Exit_status=0`；`stime=2026-09-22 19:09:28`，`obittime=2026-09-22 19:13:27`，walltime `00:03:48`。
- runner `EXIT_STATUS=0`，summary `status=COMPLETE`；训练 `1000/1000` updates，fresh CEM `48/48` blocks。
- gate `FAIL`：primary median delta `+0.08918560296`，strictly improved episodes `3/8`，round10/20/30 median deltas `+0.066989/+0.065361/+0.065455`；forgetting median `-0.013334`、max episode `+0.125912`，两项 guard 通过。
- 小证据已取回至 `artifacts/25263388.pbs101/`：`cem_boundary_ranking_summary.json`、`job.log`、`job_status`、`final_exit_status.txt`、`qstat_final.txt`、`gpu_info.csv`。单次权威 `qstat -xf` 确认 `F/Exit_status=0`、实际 walltime `00:03:48`；未取回 checkpoint、rows 或 167 MB fresh trajectory bank。
- `gpu_info.csv` 仅为静态启动摘要；5 秒 GPU utilization/memory telemetry 由 `job.log` 持续记录，满足 wrapper 约束。
- 详细结果见同目录 `RESULT.zh.md`。本实验只支持 fixed-observation candidate-ranking，按协议不进入 official CEM/closed-loop。
