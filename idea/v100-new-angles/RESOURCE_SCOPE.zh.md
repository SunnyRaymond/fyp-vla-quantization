# V100执行规模与资源证据

2026-09-13。下表摘录已保存的SLURM scheduler receipts；Elapsed是整个producer作业的经过时间，包含加载、校验与写出，不是kernel benchmark或计费金额。每项均为一个V100 allocation。CPU准备、verifier、未列出的诊断与排队不在表内。

| Screen / producer | Job | Elapsed | Scheduler |
|---|---:|---:|---|
| Antithetic | 64733 | 3m48s | COMPLETED |
| Semantic scales | 64738 | 35s | COMPLETED |
| Flow geometry | 64763 | 1m29s | FAILED，完整raw后最终summary失败 |
| Action gradient | 64767 | 22s | COMPLETED |
| Padded feedback | 64772 | 5m06s | COMPLETED |
| Conditional distribution | 64778 | 3m12s | COMPLETED |
| Q ensemble | 64799 | 18s | COMPLETED |
| Value gauge | 64807 | 23s | COMPLETED |
| Persistence B2 | 64825 | 5m47s | COMPLETED |
| Recorded future | 64828 | 26s | COMPLETED |
| Policy prior | 64833 | 21s | COMPLETED |
| Euler Jacobian | 64835 | 1m40s | COMPLETED |
| Broadcast coupling | 64838 | 15s | COMPLETED |
| Reference-branch coupling | 64840 | 26s | COMPLETED |

这些14个主要producer的Elapsed合计24m08s，**不是整个campaign总GPU成本**。另外已保存的失败GPU lineage包括Persistence原64815的9m03s，以及64797的9s、64766的14s、64760/64761/64762的2s/19s/40s；准备、其他probe及CPU成本应按各自receipt另查，不能遗漏后冒充完整总账。

所有主要producer均已退出。原Persistence失败样本没有与B2拼接；Flow的FAILED仍保留在正式inconclusive_provenance状态，不能因raw齐全而把scheduler状态改成成功。源证据在`scheduler/<job>/slurm_status.txt`和各候选RESULT。

这说明本轮采用的是短screen规模，不能推出完整LIBERO/Wall控制评估耗时。低比特运算使用FP32 dequantized operators；没有测native INT4吞吐、真实量化显存节省或部署成本。V100型号/compute capability、peak allocation（有保存的项目）和source身份见各producer runtime/engineering receipts。

Reference-branch GPU64840已完成26s，CPU64841复核8s，mechanism_no_go，STOP；已计入14项。冻结预算为GPU内部180s/SLURM5min，未扩验。
