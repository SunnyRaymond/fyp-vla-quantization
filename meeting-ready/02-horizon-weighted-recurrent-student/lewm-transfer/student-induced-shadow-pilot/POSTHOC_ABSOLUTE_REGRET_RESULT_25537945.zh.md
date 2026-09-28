# Seeded pilot 绝对 elite regret：描述性后验分析

CPU-only PBS 作业 `25537945.pbs101` 正常结束（`Exit_status=0`）。它仅读取有效 seeded pilot `25537667` 已保存的 `candidate_scores.jsonl`、summary 和 freeze，不运行模型、teacher 推理或 simulator。15 个有 t0/t25 配对的 source episode 是统计单位；第 25 步终止的 episode `15946` 不进入成对统计。该分析**不改变原 pilot 的 NO-GO**。

| CEM round | t0 regret 中位数 [Q1,Q3] | t25 regret 中位数 [Q1,Q3] | t0/t25 中 regret>0.5 的 episode | t0→t25 recall@120 中位数 |
|---:|---:|---:|---:|---:|
| 10 | 1.823 [1.593,2.327] | 1.646 [1.376,1.959] | 15/15；15/15 | 0.233→0.233 |
| 20 | 1.628 [1.462,2.101] | 1.642 [1.494,1.878] | 15/15；15/15 | 0.267→0.233 |
| 30 | 1.669 [1.501,1.956] | 1.705 [1.594,1.816] | 15/15；15/15 | 0.233→0.267 |

这里的 regret 是 student top30 的 teacher 平均 cost 与 teacher top30 的差，除以该 300-candidate bank 的 teacher cost population standard deviation；越低越好。所有 90 个被统计的 episode×state×round 单元 regret 都为正且超过 0.5，显示广泛的候选排序缺口已经存在于 t0，不能将低成功率主要解释为单纯的多步状态累积误差。Round30 的 t25−t0 中位增量虽然为正，但只有 9/15 episode 为正，原预设“普遍退化”假设仍是 NO-GO。

这仍是受控 seeded 协议下、student-generated candidate banks 上的 teacher shadow 诊断。t0/t25 候选分布和 planner context 不同；absolute regret 也不等同于完整 teacher CEM 或闭环成功率。先前固定 observation candidate-tail 蒸馏和 teacher K-screening 均已失败，不能把这些较高 regret 直接当作重跑同配方的依据。下一步应单独设计能在真实 planner context 上改善 elite 排序、且在独立 candidate bank 上验证的训练/纠偏方法。

证据文件：[posthoc_absolute_regret_summary.json](results/25537945.pbs101/posthoc_absolute_regret_summary.json)、[paired_episode_rows.jsonl](results/25537945.pbs101/paired_episode_rows.jsonl)。
