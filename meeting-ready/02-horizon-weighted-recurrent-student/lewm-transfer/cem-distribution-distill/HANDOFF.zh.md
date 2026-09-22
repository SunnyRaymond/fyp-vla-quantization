# CEM distribution distillation handoff

状态：**已通过 `/root/usage_watch` 最终有限 review；唯一 bounded PBS 已完成，Exit_status=0，但预设机制 gate=FAIL；不推进后续 CEM/closed-loop**。

## 已冻结设计

- 起点：`anchor_aligned_bank_step3000.pt`，来源 `25223859.pbs101`，两臂同一 state dict。
- control/treatment：各自重置 AdamW，沿用原 score-distill loss、batch 8、额外 1000 updates；control 使用原 64-candidate anchor bank，treatment 使用原 32 + 固定 CEM round10/20/30 的 11/11/10 teacher-labelled tail candidates。
- CEM：LeWM pinned semantics，30 rounds、300 candidates、top30、candidate-zero=`mu`、unbiased `std`、无 clipping；teacher 只作 shadow label，不参与采样或 proposal update。
- Fresh：沿用 selection seed `20300903` 的 `valid[560:568]`，48 standard blocks；primary trajectory bank 合并 round10/20/30 后按 episode 的 18 blocks 汇总。
- primary gate：8-episode median standardized regret delta `<=-0.05`，至少 5/8 episode delta `<0`，各 round episode median delta `<=0`。
- forgetting guard：独立 fresh current-anchor 300 bank，episode median delta `<=+0.05`，且无 episode delta `>+0.20`。
- scope：fixed-observation candidate-ranking only；official CEM deployment 和 closed-loop 均 `NOT_RUN_BY_SCOPE`。

## 本地验证

- `run_cem_distribution_distill.py`：`py_compile PASS`。
- `run_cem_distribution_distill.pbs`：`bash -n PASS`。
- interface/schema preflight：`PASS`，`model_work_started=false`。
- preflight 输出：`local-status/preflight_status.json`。

## review、上传与提交

1. `/root/usage_watch` 最终有限静态 review：`PASS`，未发现 correctness blocker；未加载模型/数据、未运行训练。
2. 已通过单一 `remote.py --script` control 上传新目录的 `FREEZE.json`、`PROTOCOL.zh.md`、`run_cem_distribution_distill.py`、`run_cem_distribution_distill.pbs`；远端四文件已 `ls` 核对，未传模型、rows 或 HDF5。
3. 唯一 bounded PBS：`25239551.pbs101`，`normal/gdev`，1 GPU、16 CPU、110 GB、30 分钟；提交时间 `2026-09-22 17:21:43`（PBS ctime）。
4. 首次权威状态：`qstat -f 25239551.pbs101` 为 `job_state=Q`；PBS comment=`Not Running: would exceed overall limit on resource ngpus in queue gdev`。未 qalter、qdel 或重复 qsub。
5. 运行期间只做 45–60 秒轻量 `qstat`/`tail`，同时检查主 Codex quota；usedPercent 回到 0 立即停止推进并通知 parent。
6. 终态只回传 summary、job_status、final_exit_status、job.log、GPU info/qstat；checkpoint、rows、CEM metadata 留在 cluster。

## 终态结果

- `25239551.pbs101`：PBS `F`、`Exit_status=0`、`resources_used.walltime=00:05:14`；权威小证据已保存于 `local-status/25239551.pbs101/`。
- primary 8-episode median standardized-regret delta=`-0.031312`，预设阈值 `<=-0.05` 未通过；严格改善 `4/8`，预设至少 `5/8` 未通过。
- round10/20/30 episode-median delta 分别 `-0.332304/+0.212484/+0.204261`，round20/30 未通过逐 round `<=0`。
- forgetting median delta=`-0.068779` 通过，但最差 episode delta=`+0.351707`，超过 `+0.20` 上限；最终 gate=`FAIL`。
- 详细数字与结论见 `RESULT.zh.md`。official CEM、planner deployment、closed-loop 均 `NOT_RUN_BY_SCOPE`；不得因该结果扩轮次、改阈值或重提作业。

不得因 gate 失败扩轮次、扩候选、改阈值或推进 closed-loop。
