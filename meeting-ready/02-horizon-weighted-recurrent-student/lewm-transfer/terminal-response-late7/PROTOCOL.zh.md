# Terminal-response student + late teacher7 冻结协议

## 问题与范围

在 `25549480.pbs101` 训练出的 `terminal_response_step3000.pt` 上，按每个 CEM batch 的最后七轮调用 LeWM teacher，能否相对该 student-only 臂改善 PushT task success，同时保留旧 late7 的质量差距与速度门槛？

本实验只比较三个臂：

| 臂 | 每个 CEM batch 的 scorer |
|---|---|
| `student_only` | 30 轮均为冻结的 terminal-response student |
| `late_teacher7` | rounds 24–30 为 official teacher，其余 23 轮为同一 student |
| `teacher_only` | 30 轮均为 official teacher |

Student checkpoint 固定为 `/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/terminal-response-loss/artifacts/25549480.pbs101/terminal_response_step3000.pt`。它来自 `terminal_response` arm、3000 updates；本实验不训练、不选择 snapshots、不更换 checkpoint。该 student 的 predictor-level absolute ranking gate 原为 NO-GO，因此本实验若通过，也只说明这个 checkpoint 与 late7 组合在本冻结评估上的结果，不会回写为 predictor replacement 通过。

评估使用 pinned `World.evaluate(dataset=...)` 的 dataset-mode 路径。它与 upstream PushT dataset evaluation 对齐，但不是逐字运行 upstream `eval.py`，也不是 random-reset benchmark。Episode success 取 evaluator 的 `episode_successes`；截断不算成功。

## 固定任务、模型和 CEM

复用 teacher-only baseline `25534994.pbs101` 的 50 个 task keys，完整有序清单和预期 teacher-only success vector 都写在同目录 `FREEZE.json` 中。每个 task key 是 `(row_index, episode_idx, start_step)`，50 个 source episode 各出现一次。不得重采样、过滤、重排或替换。

沿用 baseline 的 dataset、pinned callables、full-dataset `StandardScaler`、NaN-row exclusions、goal scaler、ImageNet/Resize(224)、goal offset 25、`eval_budget=50`、`world.max_episode_steps=100` 和 `video=None`。Full-dataset scaler 可能读取所选任务所在行；不声称 row-level preprocessing isolation。

三臂均实例化 pinned `stable_worldmodel.solver.CEMSolver`，配置为 batch size 1、300 candidates、30 rounds、top-30、`var_scale=1`、CUDA、solver seed 42。每个臂创建 fresh solver，并沿用 native generator progression；不在 solve calls 之间重新播种。只在 cost scorer 处路由 teacher/student，不改 official sampling、candidate zero、`torch.topk`、elite gather、unbiased standard deviation update 或 final-mean return。

`CEMSolver.solve` 对每个 vectorized environment chunk 调用 callback 的 `start_batch()`，再执行 30 次 cost evaluation。每次 `start_batch()` 都将局部 1-indexed route counter 归零，并独立检查 30 个 decisions：student-only 为 0 次 teacher call，late7 恰在 rounds 24–30 调用 7 次，teacher-only 调用 30 次。不能把整个 `solve` 的多个 batch 合并后只检查一次 schedule。

## 配对、指标和门槛

主要比较是 `late_teacher7 - student_only` 的 50 个 paired episode success。50 个 task 各自来自不同 source episode，因此使用 exact two-sided paired McNemar test；这里与每个 source episode 一个 binary outcome 的 exact cluster sign-flip test 等价。报告每臂 success count/rate、paired success table、success 差和 exact p-value。

同时报告 late7 相对 teacher-only 的 success-count 差、每臂和总 CUDA-synchronized `CEMSolver.solve` 时间、late7/teacher-only solve-time ratio、teacher calls、环境 steps/progress、peak CUDA memory、prepared observation pairing 和逐 batch schedule 检查。

沿用之前 late7 的全部性能门槛，不按本次结果更改：

1. **相对质量：** late7 比新 student-only 至少多 5/50 个成功，且 exact paired two-sided `p < 0.05`。
2. **Teacher gap：** late7 的成功数最多比 teacher-only 少 5/50。这是 practical margin，不是 formal non-inferiority 结论。
3. **规划耗时：** `late_teacher7 / teacher_only` 的 CUDA-synchronized CEM solve 总时间比值 `<= 0.70`，即至少降低 30%。
4. **Validity：** 任务身份和顺序一致、三臂各有 50 个完整且唯一的结果；teacher-only 的逐任务 success vector 必须与 baseline 完全一致（49/50）；所有 prepared observation keys 配对；每 batch schedule/call 数准确；RNG common-prefix/candidate-shape、有限输出、计时和 allocation/source-root checks 均通过。

总判定只有在 validity 和上述三项性能门槛全部通过时才为 `PASS`。Validity 失败时 `FAIL_CLOSED`，不作性能结论；有效运行未过任一性能门槛则如实记 `FAIL` 或 `INCONCLUSIVE`。不扩充任务、不重跑、不换 checkpoint 或 schedule，也不放宽门槛。

## 旧 student 对照与 anti-leakage

旧 `treatment_step1000.pt` 不作为本次并发臂，以保持这次三臂比较聚焦于“新 student 上 late7 是否优于该 student-only”。旧结果只作历史描述：同一任务集上 student-only 为 11/50，late7 为 26/50；另一有效 timing run 的 late7 为 25/50，teacher-only 为 49/50。旧结果不与新结果合并，不参与本次 p-value，也不能据此作新旧 checkpoint 的 paired causal claim。若新 late7 与旧 late7 的数值不同，只能描述为同一 cohort 上后续 checkpoint/schedule 组合比较，不能称为独立复现。

这 50 个任务已经用于旧 student-only/late7 评估；复用它们是为了沿用同一 task set 做配对后续比较，因此本实验**不是独立验证或 fresh generalization test**。新 checkpoint 在运行前固定；不得根据这 50 项的结果选择训练 snapshot、checkpoint、schedule、阈值或重试任务。`FREEZE.json` 内联了来源于 baseline 的 task keys 和 teacher-only vector，运行时还须逐项与源 artifact 对照。

## 执行边界

HDF5/model 读取、CEM、inference 和环境评估必须在带真实非空 `PBS_JOBID` 的获批 compute allocation 内运行；并检查 `PBS_NODEFILE` 有效且执行 hostname 属于 allocation，拒绝 login/head/submit host。Login node 仅用于连接、作业提交、状态查询和小型控制文件操作，不运行计算、不加载模型、不读写/传输大数据、不做 benchmark 或 heavy I/O。

按每 5 秒采样 GPU utilization 和显存，并写入该 job 的 `job.log`；作业结束时终止并等待 telemetry 进程。不得查询或使用 banked reset credits。该协议只冻结评估设计；本文件本身没有提交或启动作业。
