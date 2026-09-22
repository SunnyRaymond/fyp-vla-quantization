# Stage A 结果：balanced_base teacher screening

## 作业与来源

- PBS job：`25213164.pbs101`，`gdev`，A100-SXM4-40GB；终态 `F`，`Exit_status=0`。
- PBS 资源记录：walltime `00:06:18`，cput `00:08:12`，峰值内存约 `1,563,508 kb`。
- 远端 summary：`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/teacher-screening/artifacts/25213164.pbs101/teacher_screening_stage_a_summary.json`。
- 本地回传：`artifacts/25213164.pbs101/teacher_screening_stage_a_summary.json`、`job.log`、`job_status`、`gpu_info.csv`。
- Phase 5/6 未提供可加载 checkpoint；本次从 Phase2 prepared512 与原 balanced recipe 在 PBS 内重建一次。checkpoint provenance=`reconstructed`，远端路径：`/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/teacher-screening/artifacts/25213164.pbs101/balanced_base_step3000_reconstructed.pt`。
- balanced rows 来源=`reconstructed_on_compute_node_from_phase2_prepared_rows`，留在远端；没有回传模型或 rows。

## 冻结评估

使用 fresh `valid[544:552]`，episode IDs=`17653,18455,14201,17971,5795,9065,2827,3867`；8 episodes × early/middle/late × seeds `20300947/20300948` = 48 blocks，每 block 300 candidates。接口、finite 检查均通过。

| K | overall median recall | minimum block recall | full-elite containment | median elite mean-cost regret | hybrid latency reduction | gate |
|---:|---:|---:|---:|---:|---:|:---|
| 60 | 0.866667 | 0.000000 | 0.270833 (13/48) | 0.382996 | -0.109316 | FAIL |
| 120 | 1.000000 | 0.000000 | 0.562500 (27/48) | 0.000000 | -0.105788 | FAIL |

K=60 的 stratum recall median：early `0.983333`、middle `0.866667`、late `0.700000`；K=120：early `1.000000`、middle `1.000000`、late `0.983333`。但 K=120 仍有 minimum block recall `0`，且 hybrid latency reduction 为 `-10.5788%`，未满足 gate。

K=120 的 15 个 recall<0.8 blocks 按 episode/anchor 聚合如下，seed 为两个 nested action-prefix replicate：

- episode 17653：early anchor 0，recall `0.133333/0.300000`；late anchor 120，`0.133333/0.000000`。
- episode 14201：middle anchor 78，`0.133333/0.233333`。
- episode 17971：middle anchor 55，`0.366667/0.366667`。
- episode 2827：middle anchor 36，`0.533333/0.433333`。
- episode 5795：late anchor 77，`0.766667/0.700000`。
- episode 9065：late anchor 34，seed `20300948` 为 `0.700000`。
- episode 3867：early anchor 0，`0.766667/0.733333`。

最差 episode 为 17653：K=60 episode median recall `0.166667`，K=120 为 `0.216667`。这些 blocks 是 nested episode 内的诊断，不能当作 15 个独立 replicate。

计时覆盖全部 48 blocks、warmup=3、repeats=10、交错顺序；teacher baseline 含 300 候选的 rollout/cost/top30 selection，hybrid 含 student300、sort/gather、teacher-K 与 elite selection。block-median 中位数：teacher `20.4976 ms`，hybrid-K60 `22.7391 ms`，hybrid-K120 `22.6744 ms`。负 latency reduction 是实测结果；本报告不把其原因归因于 launch overhead。

## 门控结论

Stage A gate=`FAIL`，没有可选 K；Stage B=`NOT_RUN_BY_GATE`，adaptive CEM 与 closed-loop 均未运行。不得扩 K、调阈值、重训或据此宣称 closed-loop 成功。
