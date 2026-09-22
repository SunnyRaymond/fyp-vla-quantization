# Compiled World Model：predictor-level 结果

## 结论

本轮预注册 verdict 为 **NO-GO**。`b_h(c)+B_h(c)phi(u_{1:h},h)` 的 context-dependent 结构能够训练，并且 B4 quadratic contraction 对该已训练模型数值等价；但 selected `r=192` 在独立 final test 上的 ranking/latent-quality gate 明显失败，context dependence 相对 fixed basis 的改善不稳定，也未在冻结的 30-query timing 中优于更简单的 dense-prefix B1。按协议停止，不进入 official CEM、closed-loop 或 joint encoder training。

## 作业与证据

- PBS job：`25309513.pbs101`，A100-SXM4-40GB，runner/PBS `Exit_status=0`。
- walltime `00:04:38`，cput `00:09:43`，峰值 PBS memory `1,392,260 kb`。
- PBS 最终状态为 `F`，但 `Stageout_status=1`；scratch 内 runner summary、`job_status=EXIT_STATUS=0`、`final_exit_status=RUNNER_EXIT_STATUS=0` 均已完整生成并回传，因此不重跑。
- GPU utilization/memory 以 5 秒间隔写入 `job.log`；训练期间 utilization 最高采样到 91%，显存最高采样约 2,193 MiB。
- 大 checkpoint 留在 compute scratch；本地只回传 summary、log、status 和 telemetry。

## E0：结构与代数

| 检查 | 结果 |
|---|---:|
| causal prefix leakage max abs | `0` |
| batch/chunk max abs | `1.79e-7` |
| context-swap mean effect | `0.3259` |
| synthetic FP64 compiled max abs | `1.82e-12` |
| synthetic FP32 compiled max abs | `0.001953` |
| trained B3 official-vs-terminal affine residual | `1.53e-5` |
| trained B3 compiled-vs-official max abs | `5.34e-5` |
| compiled/explicit argmin | exact match |

因此，B4 对**这一个已训练分离模型**的 terminal quadratic cost contraction 成立；它不等于原 LeWM predictor，也不代表模型质量通过。

## Development rank selection

rank 只在 `valid[552:560]` 上选择，final test 没有参与选择。

| B3 rank | Spearman median | top-30 median | relative latent MSE median |
|---:|---:|---:|---:|
| 32 | `0.1031` | `0.1333` | `0.1087` |
| 96 | `0.0936` | `0.1333` | `0.1088` |
| 192 | `0.1307` | `0.1667` | `0.1111` |

按冻结的 lexicographic rule 选择 `r=192`。三个 rank 的绝对表现都弱；选择只决定 final test 的唯一 rank，不构成正信号。

## 独立 final test

final test 固定为 selection seed `20300903` 的 `valid[592:600]`，8 episodes × early/middle/late × seeds `20301201/20301202`，共 48 blocks。

| Arm | Parameters | Spearman median / min | top-30 median / min | relative latent MSE median |
|---|---:|---:|---:|---:|
| B1 dense-prefix | `179,118` | `0.1760 / -0.4775` | `0.1667 / 0` | `0.1916` |
| B2 fixed basis, r=192 | `544,622` | `0.0045 / -0.2872` | `0.1000 / 0` | `0.1816` |
| B3 context basis, r=192 | `47,730,542` | `0.0534 / -0.4996` | `0.0833 / 0` | `0.1389` |

B3 的 latent MSE 比 B1/B2 低，但 ranking 更差，说明 reconstruction improvement 没有转化成 planner-facing candidate ranking。B3 相对 B2 的 episode-median Spearman delta 为 `+0.0596`，但 top-30 delta 为 `-0.00833`，joint strict improvement 只有 `4/8` episodes（要求 `>=5/8`）。

absolute gate 也全部失败：B3 Spearman 要求 `>=0.9`、实际 `0.0534`；top-30 要求 `>=0.7`、实际 `0.0833`；relative latent MSE 要求 `<=0.1`、实际 `0.1389`。finite check 通过。

## 30-query timing

边界为 cached H=1 latent/goal、30 次 × 300 candidates，只包含 predictor/criterion；不包含 encoder、CEM update、transfers 或 environment。

| Path | p50 | p95 |
|---|---:|---:|
| official LeWM teacher | `603.735 ms` | `611.492 ms` |
| B1 dense-prefix | `14.349 ms` | `14.972 ms` |
| B3 explicit latent | `14.536 ms` | `15.361 ms` |
| B4 compiled quadratic | `16.821 ms` | `17.614 ms` |

- B3/B1 p50 speedup=`0.987×`：B3 略慢，未证明 candidate-direction context sharing 的实际收益。
- B4/B3 p50 speedup=`0.864×`：quadratic contraction 在本 GPU/kernel/rank 下反而更慢。
- B4/teacher p50 speedup=`35.89×`，但模型质量 gate 失败，不能将该数字写成有效 replacement speedup。
- breakdown p50：B3 prepare `0.275 ms`、B3 query300 `0.422 ms`、B4 compile `0.209 ms`、B4 score300 `0.408 ms`。

## 决策

| Gate | Result |
|---|---|
| E0 synthetic and trained-model compilation | **PASS** |
| absolute predictor quality | **FAIL** |
| B3 context dependence vs B2 | **FAIL** |
| B3 not dominated by B1 | **FAIL** |
| frozen efficiency gate | **FAIL** |
| overall | **NO-GO** |

本结果否定的是当前 frozen encoder、1500-step、MLP factorization recipe。它不否定所有 hypernetwork/operator-style world models；但按既有规则，不应通过加 steps、换 thresholds、扩大 seeds、增加 mixture/router/fallback 或直接进入 official CEM 来延长同一失败 recipe。若以后恢复，必须提出新的机制假说并重新冻结协议。

## Claim boundary

这是 predictor-level、fixed-observation、multi-query evidence。`official CEM`、full planner、closed-loop PushT、encoder speedup 和 task success 均为 `NOT_RUN_BY_SCOPE`。
