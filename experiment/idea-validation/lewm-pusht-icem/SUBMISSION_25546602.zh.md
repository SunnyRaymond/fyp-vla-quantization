# iCEM population-decay PBS submission

- 作业：`25546602.pbs101`；提交日期：2026-09-25（Asia/Singapore）。
- 冻结控制文件：`FREEZE.json`、`run_icem_decay.py`、`decay_cem.py`、`run_icem_decay.pbs` 已以小文件传至 `/scratch/users/ntu/yguo017/dino-wm-wall/experiment/idea-validation/lewm-pusht-icem/`，随后通过该 PBS 入口提交一次。
- 资源：`normal` route → `gdev`，1 GPU、16 CPU、110 GB、2 小时；GPU utilization/显存每 5 秒写入 job log。
- 首次 `qstat -f`：`job_state=Q`，`comment=Not Running: Insufficient amount of resource: ngpus (R: 1 A: 0 T: 256)`。这仅是排队快照，非终态。
- 必须在终态读取 `qstat -x -f` 的 `Exit_status`、作业 `job_status`、`final_exit_status.txt`、`stage1_gate.json`、`run_summary.json` 和 GPU telemetry 后才能判断实验有效。不得因查询超时自动重提或调整已有作业。
- 定时监控：Codex heartbeat `lewm-icem-pbs-25546602`，每 15 分钟唤醒独立任务 `01a0d7c8-cedd-7681-866a-c5ae1765c052`（首轮已核实 `gpt-6-luna` / `xhigh`）。监控只读；终态或异常时使用 `send_message_to_thread` 唤醒原实验任务 `01a0d673-eb15-7662-bd6c-37011d4e37da`，随后停用 heartbeat。首次监控读到 `Q`，没有向原任务发送无变化消息。
