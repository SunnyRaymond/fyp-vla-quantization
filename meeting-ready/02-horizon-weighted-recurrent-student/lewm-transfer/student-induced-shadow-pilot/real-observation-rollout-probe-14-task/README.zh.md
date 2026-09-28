# Real-observation latent probe：14-task 扩展

本扩展只运行 `SEEDED_PILOT_FREEZE.json` 中 `selection_order` 2–15 的14个 collection tasks。selection order 0、1 已由 PBS job `25542215.pbs101` 完成，作为既有证据保留；本 runner 不会重跑它们，也不会读取其结果。16个 `reserved_holdout` task 不参与 episode、打分或决策。

每个任务沿用已通过的 seed 42 controlled protocol、student-only native CEM、真实 env actions 与 forecast token 的 `1e-5` 对齐检查，以及相同的 official H=1 observation encoder 和 relative MSE。每个 task 先分别计算 t0 与 t25 solve 的 horizon-5 student/teacher relative MSE 差值，再按 episode 汇总中位数和计数；缺少 t25 horizon 5 时保留缺失标记，不补齐。

本 job 只报告这14个任务。后续合并时，必须把其输出与 job `25542215.pbs101` 的前两任务结果按 collection identity 合并，不得重跑前两项。冻结判据为：合并16个任务后，t0 和 t25 各自至少12个任务的 `student_relative_mse - teacher_relative_mse >= 0.05` 才支持进一步研究真实 observation target，否则停止该方向。这里的判断是描述性机制诊断，不是训练、CEM ranking 或 closed-loop success 结论。

PBS wrapper 请求1张 GPU、walltime 90分钟，并每5秒把 GPU 利用率和显存写入 `job.log`。模型/HDF5 读取和计算仅在核实后的 PBS compute allocation 内发生。本地只做静态检查，不提交作业。
