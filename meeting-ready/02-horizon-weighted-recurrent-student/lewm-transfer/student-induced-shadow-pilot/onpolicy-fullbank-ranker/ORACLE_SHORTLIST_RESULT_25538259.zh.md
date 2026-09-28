# On-policy full-bank oracle shortlist 后验诊断：25538259.pbs101

PBS 历史 `Exit_status=0`；只读 CPU 脚本完成，对 `25538135.pbs101` 已保存的完整 CEM candidate banks 计算了固定的 K=30/60/120/300 曲线。输入为 student-driven 的 93 个 train bank（16 source episodes）与 48 个 validation bank（8 source episodes）。Validation 已用于前一轮 residual-ranker gate，**本诊断不是新的独立验证或 GO gate**。

对每个 bank，先取 student objective 最低的 K 个候选，再用已知的 teacher objective 在其中选最佳 30 个。表中 regret 为相对全 300 候选 teacher 最佳 30 个的标准化成本差，越低越好；每个 source episode 先在其 banks 中取中位数，再跨 episode 取中位数。

| Split | K | Student Top30 regret | Shortlist oracle regret | Δ vs student | Teacher elite recall@K |
|---|---:|---:|---:|---:|---:|
| Train，16 episodes | 30 | 1.6611 | 1.6611 | 0.0000 | 3.33% |
| Train | 60 | 1.6611 | 1.1525 | −0.5412 | 9.17% |
| Train | 120 | 1.6611 | 0.7153 | −0.9718 | 23.33% |
| Train | 300 | 1.6611 | 0 | −1.6611 | 100% |
| Used validation，8 episodes | 30 | 1.6536 | 1.6536 | 0.0000 | 5.00% |
| Used validation | 60 | 1.6536 | 1.0477 | −0.5846 | 11.67% |
| Used validation | 120 | 1.6536 | 0.6409 | −1.0487 | 25.83% |
| Used validation | 300 | 1.6536 | 0 | −1.6536 | 100% |

这些数字说明 teacher 在 student shortlist 内精选有**后验质量上限**，但 student Top120 仍漏掉约四分之三的全 bank teacher elite。K=120 的 oracle regret 仍约 0.64–0.72，并非达到全 teacher 排序。这里的 oracle 使用了现成的 teacher 标签，未测 student+teacher-K 的原生耗时，也没有把重排分数反馈到 CEM；不能据此声称加速、规划或闭环成功。

历史 `balanced_base` 固定 observation K-screening 在 K=60/120 的 native hybrid timing 已比 teacher300 慢约 10–11%，且最差 block recall 失败。其 checkpoint 与状态分布不同，不能直接替代本轮 on-policy timing；但它提示“student300+teacher-K”可能没有速度余量。若考虑实际 teacher-in-loop，应先为**当前** student 与候选分布测 native latency，并对照当前 residual oracle regret；没有速度与排序双门槛证据前不推进 CEM。

原始小型结果：[oracle_shortlist_summary.json](results/25538259.pbs101/oracle_shortlist_summary.json)、[job log tail](results/25538259.pbs101/job_log_tail.txt)。
