# Stage 2 配对结果：25535873.pbs101

## 判定

本次四臂结果数据有效，冻结的主要收益门通过，但完整推进 gate 未通过：late7 相比 student-only 多成功 15/50，配对精确检验通过；它仍比 teacher-only 少成功 23/50，且规划耗时只快 24.2%，没有达到预设的 teacher gap ≤5 和 latency ratio ≤0.70。最终 `overall=FAIL`，这是有效实验中的性能未达标，不是 `FAIL_CLOSED`。

## 作业与协议

- 权威 PBS 状态：`F`，`Exit_status=0`，walltime `00:08:09`，compute host `x1000c1s1b0n1`，GPU `gdev`。
- 遥测日志有 94 个 GPU 样本；利用率范围 0–75%，显存 1–643 MiB。样本间隔中位约 5 秒，最大间隔 24.8 秒。
- 使用条件冻结中的 baseline `25534994.pbs101` 所选 50 项，四臂顺序为 student-only、teacher-only、late7、uniform7。HDF5、teacher checkpoint、student checkpoint、StandardScaler/ImageNet 处理、dataset callables、50 步评估预算、seed 42、CEM 30 rounds × 300 candidates / top30 均未改变。具体来源见 summary 和本作业的 `execution_identity.txt`。
- 该实验是与 upstream PushT dataset evaluation 对齐的配对规划比较，不是逐字运行 upstream `eval.py`，也不是 random-reset simulator benchmark。

## 有效性门

全部通过：共 50 个 task、200 条 arm outcome，每臂 50 条；arm/task key 无重复或缺失；四臂逐项使用相同的 row、source episode 和 start step。50 个 source episode 均不同。六个 prepared observation keys 精确配对；两个 solver 调用中各臂 native RNG state 与 candidate shape 的 common prefix 相同。Teacher-only 的逐项成功向量与 baseline 完全一致（49/50，0 个不匹配）。所有 route schedule 与数值有限性检查通过，progress capture 无错误。

已按 pinned staging CEM source 的实际生命周期修正批次计数：`CEMSolver.solve` 每个 environment chunk 调用一次 callback `start_batch()`，然后执行 30 个 CEM iterations。Stage2 callback 现在在此边界重置 route round counter，并按每 30 个 cost evaluation 分组核对 schedule。旧 job `25535774.pbs101` 保留为 `FAIL_CLOSED`：其 counter 只在整次 `solver.solve` 重置，late7 和 uniform7 各只调用 teacher 14 次，未执行冻结的逐 batch 路由，因此旧 job 的性能不可用于判断策略。

## 四臂结果

| Arm | 成功 | Teacher cost calls | Planner solve time |
|---|---:|---:|---:|
| student-only | 11/50 (22%) | 0 | 46.59 s |
| late teacher7 | 26/50 (52%) | 595 | 41.59 s |
| teacher-only | 49/50 (98%) | 1,800 | 54.84 s |
| uniform teacher7 | 18/50 (36%) | 637 | 44.73 s |

late7 与 student-only 的配对表为 17 个 late-only success、2 个 student-only success、9 个双方成功、22 个双方失败。50 个独立 source-episode clusters 上的 exact two-sided sign-flip `p=0.0007286`，成功差 `+15/50`，因此预设的 `+5` 且 `p<0.05` 收益门通过。Late7 距 teacher-only 还差 23 个 success；late7/teacher-only planner solve time ratio 为 `0.7583`，两项分别未达到 gap ≤5 与 ratio ≤0.70 的冻结门槛。Uniform7 为 18/50，作为次要同预算对照低于 late7。

## 下一步方向

结果支持 late-stage teacher scoring 能改善 student-only 的 episode success，但目前保留了明显的 teacher quality gap，节时幅度也不足以通过预设标准。下一轮可在相同 50 个 task、solver seed 和处理流程上做 teacher-call 数相同的早/晚时间对照：student-only、teacher rounds 1–7、teacher rounds 24–30、teacher-only。若早期 1–7 明显优于晚期 24–30，说明早期 teacher updates 对 CEM 分布锚定更关键；若晚期 schedule 更好，则更支持 late candidate re-scoring 的作用。可用 callback 仅记录每个 batch 的 CEM mean/variance 标量轨迹作为次要诊断，不保留大候选张量或增加 teacher calls。先冻结这项问题与门槛，再决定是否提交；本报告没有提交后续作业。

## 证据文件

本地完整小型结果位于 `artifacts/25535873.pbs101/`：`stage2_summary.json`、`stage2_episodes.jsonl`、`rng_pairing.json`、`selected_tasks.json`、`job_status`、`final_exit_status.txt`、`execution_identity.txt`、`job.log` 与 GPU telemetry。`25535774.pbs101` 的失败实现、summary 和 episode 记录继续独立保留。
