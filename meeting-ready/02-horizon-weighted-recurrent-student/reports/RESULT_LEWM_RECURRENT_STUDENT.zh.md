# LeWM PushT：Horizon-Weighted Recurrent Student Predictor-Level 结果

## 结论先行

PBS job `24564619.pbs101` 在官方 LeWM PushT backend 上完成了冻结的 predictor-level protocol，但 `predictor_feasibility` 为 **NO-GO**。训练、finite/integrity、causality、relative latent MSE 和 predictor-only latency 均通过；失败点是 16 个 held-out blocks 上的 terminal objective ranking 不足以达到冻结阈值。

因此，本次结果只支持：在这个固定 LeWM PushT predictor cell 中，h256 horizon-weighted recurrent student 能够以较低 predictor-only latency 产生有限的 latent fidelity，但不能作为 planner-facing predictor replacement。Stage B official CEM 按用户要求 **未运行**，也没有 closed-loop 或 environment success 结论。

## 1. 实验边界与版本

- backend：official LeWorldModel PushT；LeWM commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`；stable-worldmodel CEM commit `10c26dbd5677083fa31dba69eb738b973845e9a4`。
- checkpoint：官方 `lewm_object.ckpt`；dataset：官方 `pusht_expert_train.h5`。本地报告不复制这些远端大文件。
- student：`LeWMCompactRecurrentTransitionStudent`，latent `192-D`，latest-3 latent history，official policy initial history `H=1`，packed action `10-D`，hidden `256`，参数量 `775,872`。
- student 无 goal input、无 `encode_obs` call、无 teacher/source parameters、无 teacher forcing；使用自己的 predicted latent feedback。
- training：1500 updates；snapshots `500/1000/1500`；batch contexts `8`；effective query rows/update `32`；horizon weights `[1/3, 2/3, 1, 4/3, 5/3]`。
- held-out：8 contexts × 2 fresh action-prefix seeds = **16 blocks**；每 block `300` candidates；top-k=`30`。block 是主要分析单位，candidate 不是独立 replicate。

## 2. 训练收敛

冻结 convergence gate 使用 weighted dense latent training loss，要求 `last10 / first <= 0.8`。

| 指标 | 数值 |
|---|---:|
| step 1 weighted MSE | `4.202648` |
| last-10 weighted MSE median | `0.012788` |
| last-10 / first ratio | `0.003043` |
| step 1500 weighted MSE | `0.013434` |
| convergence/integrity | **PASS** |

step 1500 最后一次 update 的 per-horizon training MSE（不是 held-out ranking）为：
`[0.003926, 0.009272, 0.011446, 0.011413, 0.019808]`。

## 3. Held-out ranking 与 latent fidelity

以下使用每个 snapshot 的 16-block median/minimum；正式 gate 只使用冻结的 step `1500`。

| snapshot | Spearman median / min | top-30 median / min | relative latent MSE median |
|---|---:|---:|---:|
| step 500 | `0.036010 / -0.285519` | `0.100000 / 0.033333` | `0.017934` |
| step 1000 | `0.318916 / -0.307697` | `0.233333 / 0.000000` | `0.011637` |
| step 1500 | `0.415374 / -0.404356` | `0.266667 / 0.000000` | `0.012116` |

step 1500 的正向 block 数为 Spearman `12/16`、top-30 `15/16`；两者达到冻结的 positive-block 数量要求，但 ranking effect size 不足：

- Spearman 要求 median `>=0.95`、minimum `>=0.80`；实测 `0.415374 / -0.404356`；
- top-30 要求 median `>=0.75`、minimum `>=0.50`；实测 `0.266667 / 0.000000`；
- relative latent MSE 要求 median `<=0.25`；实测 `0.012116`，通过。

作为 per-horizon held-out 描述统计，step 1500 在 16 blocks 上的中位数为：

| future horizon | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| latent MSE | `0.004032` | `0.008407` | `0.011723` | `0.012982` | `0.020116` |
| cosine | `0.998058` | `0.995885` | `0.993980` | `0.992990` | `0.989661` |

误差随 free-running horizon 增长，但 latent cosine 仍保持较高；这不能替代 terminal objective ranking gate。

## 4. Causality 与 latency

Causality 对 unchanged prefix length `1/2/3/4` 的 maximum absolute difference 均为 `0.0`，冻结阈值为 `1e-6`，因此 causality gate **PASS**。

Predictor-only latency boundary 是 cached `H=1` latent + normalized packed action prefix → five-step predictor；不包括 encoder、CEM、environment 或 closed-loop control。batch=`300`，warmup=`3`，technical repeats=`10`，含 CUDA synchronization。

| arm | median latency |
|---|---:|
| official teacher | `23.529471 ms` |
| recurrent student | `1.852928 ms` |
| reduction | `92.1251%`（约 `12.70x`）|

冻结 latency gate 要求 reduction `>=20%`，因此通过；这个数不能解释为端到端或 `encode_obs` speedup。

## 5. 决策与 claim boundary

| decision | result |
|---|---|
| integrity/convergence | **PASS** |
| predictor fidelity/ranking | **FAIL** |
| causality | **PASS** |
| predictor-only latency | **PASS** |
| predictor feasibility | **NO-GO** |
| strict predictor replacement | `NOT_COMPUTED` |
| full CEM viability | `NOT_RUN_BY_SCOPE` |

`NO-GO` 是 frozen cell 的 predictor-level ranking 结论，不是训练崩溃，也不是允许 retune width、weights、steps、seeds 或 candidate bank 的信号。当前证据不支持 LeWM closed-loop PushT success、CEM planner viability、Fast-LeWM comparison、跨模型泛化或 `encode_obs` speedup claim。

## 6. 运行与 artifacts

- job status：`EXIT_STATUS=0`；runner 输出状态为 `PREDICTOR_LEVEL_COMPLETE`。
- GPU：A100 40 GB；job log 保留 30 秒采样，共 7 samples，最大 observed utilization `11%`，最大 observed memory `637 MiB`。这些是 job-log 采样值，不作额外吞吐推断。
- summary：[lewm_recurrent_student_summary.json](../artifacts/24564619.pbs101/lewm_recurrent_student_summary.json)
- context manifest：[context_manifest.json](../artifacts/24564619.pbs101/context_manifest.json)
- job log/status：[job.log](../artifacts/24564619.pbs101/job.log)、[job_status](../artifacts/24564619.pbs101/job_status)

本地没有取回 `prepared_rows.pt` 或任何 student checkpoint。Stage B official CEM 没有提交或执行。
