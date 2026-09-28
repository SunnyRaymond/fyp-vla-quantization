# Full-bank oracle-shortlist post-hoc 诊断

此诊断只读取已完成的 `25538135.pbs101` full-bank artifacts，在 CPU PBS compute allocation 上离线计算；它不改动 bank 文件，也不训练模型。脚本固定评估 `K={30,60,120,300}`，训练 split 与 validation split 分开报告。validation 已用于 frozen residual-ranker gate，因此相关数字只能作为机制描述，不能当作新的独立 gate、阈值调节或模型选择依据。

对每个 300-candidate bank，令 `S_K` 为按 student objective ascending 排序的前 `K` 个候选，`G` 为全 bank teacher objective 最低的 30 个候选。令 `O_K` 为 `S_K` 中 teacher objective 最低的 30 个候选，`B` 为 student objective 最低的 30 个候选。对任意候选集合 `A`，标准化 teacher-elite regret 定义为：

```text
R(A) = (mean(teacher_cost[A]) - mean(teacher_cost[G]))
       / population_std(teacher_cost[all 300 candidates])
```

报告 `R(O_K)`、student top30 的 `R(B)`，以及 `delta_vs_student_top30 = R(O_K) - R(B)`；负 delta 表示 shortlist 内 oracle 重排低于 student top30 的 regret。`teacher_elite_recall_at_k = |S_K ∩ G| / 30`。这里的 oracle 可以使用 teacher 标签挑选 shortlist 内的最佳 30 个，代表 shortlist 的后验上限，不是可部署的打分器。

聚合时以 source episode 为统计单位：先对每个 episode 的可用 replan step（0、25）和 CEM rounds（10、20、30）取 median，再在 episode 间取 median。另按 `replan_step × cem_round` 分层，在每个组合内先保留 episode 级观测，再报告跨 episode median 和覆盖数。不会把同一 episode 的多个 banks 当作独立样本，也不计算新的 GO/NO-GO gate 或显著性结论。

PBS wrapper 请求 2 个 CPU cores、8 GB memory、15 分钟 walltime，并在脚本和 shell 层检查 `PBS_JOBID`、nodefile、compute hostname 和 GPU allocation。源 bank 位于 ranker artifacts 的 `25538135.pbs101/`；摘要写入新的 `artifacts/oracle-shortlist-posthoc/<PBS_JOBID>/` 目录，不覆盖源结果。wrapper 已准备好，但本次未提交任何作业。
