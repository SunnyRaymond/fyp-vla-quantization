# LeWM EMA temporal confirmatory predictor：正式结果

## 结论

本轮 confirmatory experiment 的 primary gate 为 **NO-GO**。在同一条 training trajectory
上维护的 `EMA(decay=0.999)` 没有在真正 fresh、episode-disjoint 且包含 early/middle/late
anchors 的 LeWM predictor evaluation 上通过 absolute gate，也没有优于 concurrent online
control。证据支持“EMA 训练过程可收敛且保留 predictor latency advantage”，但不支持
“EMA 能修复 LeWM recurrent student 的 temporal ranking failure”。

本轮严格停在 predictor level：official CEM、planner viability、closed-loop 均为
**NOT_RUN_BY_SCOPE**。

## 作业与 scope 证据

| 项目 | 已核实结果 |
|---|---|
| PBS job | `25142797.pbs101` |
| `job_state` / `Exit_status` | `F` / `0`（即 `Exit_status=0`） |
| walltime / exec host | `00:03:22` / `x1000c1s3b0n0/0*16` |
| queue / allocation | `gdev` / 1 GPU, 16 CPU, 110 GB |
| formal start / finish | `10:39:17` / `10:42:45` |
| output status | `PREDICTOR_LEVEL_COMPLETE`, primary `NO-GO` |
| architecture | `LeWMCompactRecurrentTransitionStudent`, h256, shared recurrent, no attention/conditioner |
| training | 512 rows, 3000 updates, AdamW, unchanged horizon-weighted latent MSE + score SmoothL1 |
| arms | same-trajectory `online` control + `EMA(decay=0.999)` primary |
| training bank | Phase 2 formal `512` bank reused verbatim；未重新生成或返回 prepared rows |
| fresh selection | `valid[520:528]`，selection seed `20300903`，8 episodes，result-independent |
| fresh evaluation | 24 contexts，48 blocks，300 candidates/block；early/middle/late anchors |
| interface | H=1，5-step future，raw action dim 2，student/official semantics equal |

Fresh episode IDs 为 `14388, 12892, 13619, 8983, 16705, 16273, 837, 2681`。每个 episode
使用 `early=0`、`middle=floor(late/2)`、`late=episode_length-25-1`，并使用 action-prefix
seeds `20300917`、`20300918`。

### Integrity and scope checks

summary 中以下 checks 均为 `PASS`：formal 512 training bank reuse、fresh episode selection、
fresh rows compute-only generation、same-trajectory online/EMA、EMA isolation、nested
episode aggregation、no result-dependent selection。candidate 是 block 内重复观测单位，
episode 才是 paired mechanism comparison 的 replicate；不能把 48 blocks 当作 48 个独立
episodes。

## Terminal step 3000 metrics

阈值沿用冻结的 inherited absolute predictor gate：Spearman median/minimum ≥ `0.95/0.80`，
top-30 median/minimum ≥ `0.75/0.50`，relative latent MSE median ≤ `0.25`，positive
Spearman/top-30 blocks ≥ `36/48`；另要求 integrity、convergence、causality 与 latency。

| arm | Spearman med / min | top-30 med / min | relative MSE med | positive Spearman | positive top-30 | absolute |
|---|---:|---:|---:|---:|---:|---|
| online control | `0.828863 / -0.321581` | `0.550000 / 0` | `0.143349` | `44/48` | `47/48` | **NO-GO** |
| EMA primary | `0.815925 / -0.344404` | `0.466667 / 0` | `0.147960` | `46/48` | `47/48` | **NO-GO** |

两臂都通过 finite、convergence、causality 与 latency 子检查，但 fidelity/ranking 的
median 与 minimum thresholds 失败。relative MSE 和 positive-block counts 单独通过，不能
替代 absolute ranking gate。

## Temporal strata

stratum protection 要求每个 16-block stratum 的 Spearman median ≥ `0.95`、top-30 median
≥ `0.75`，以及正向 blocks 各 ≥ `12/16`；不重复 overall minimum thresholds。

| arm / stratum | Spearman med / min | top-30 med / min | relative MSE med | positive S / top-30 | 解释 |
|---|---:|---:|---:|---:|---|
| EMA early | `0.962553 / 0.811669` | `0.800000 / 0.400000` | `0.014085` | `16/16`, `16/16` | median 条件通过 |
| EMA middle | `0.486430 / 0.071584` | `0.283333 / 0.066667` | `0.258155` | `16/16`, `16/16` | median 条件失败 |
| EMA late | `0.280386 / -0.344404` | `0.266667 / 0` | `0.284923` | `14/16`, `15/16` | median 与 positive-S 均不达标 |
| online early | `0.971247 / 0.899797` | `0.816667 / 0.600000` | `0.013897` | `16/16`, `16/16` | concurrent descriptive control |
| online middle | `0.589153 / -0.039629` | `0.300000 / 0.033333` | `0.218835` | `14/16`, `16/16` | concurrent descriptive control |
| online late | `0.596889 / -0.321581` | `0.416667 / 0` | `0.220336` | `14/16`, `15/16` | concurrent descriptive control |

因此 EMA 的 early anchor 表现不能掩盖 middle/late 的系统性失败，temporal stratum gate 为
**NO-GO**。

## Snapshots 与 convergence

snapshot 只作 descriptive view，terminal 固定为 `step_3000`，未按 fresh result 选择
snapshot。整体结果如下：

| arm / step | Spearman median | top-30 median | relative MSE median | positive S / top-30 |
|---|---:|---:|---:|---:|
| online / 500 | `0.561564` | `0.266667` | `0.154530` | `36/48`, `41/48` |
| online / 1000 | `0.628674` | `0.283333` | `0.151018` | `42/48`, `43/48` |
| online / 1500 | `0.708651` | `0.433333` | `0.153539` | `43/48`, `46/48` |
| online / 3000 | `0.828863` | `0.550000` | `0.143349` | `44/48`, `47/48` |
| EMA / 500 | `0.223658` | `0.183333` | `1.613260` | `37/48`, `41/48` |
| EMA / 1000 | `0.375035` | `0.233333` | `0.527130` | `36/48`, `42/48` |
| EMA / 1500 | `0.566493` | `0.316667` | `0.266951` | `33/48`, `41/48` |
| EMA / 3000 | `0.815925` | `0.466667` | `0.147960` | `46/48`, `47/48` |

training 的 `last10_to_first_ratio=0.002646`，`early_stop=false`，online 与 EMA 都到达
`step_3000`；因此 convergence/integrity 子检查为 **PASS**。EMA early snapshots 明显滞后，
到 terminal 才接近 online，但 middle/late temporal ranking 仍未达到 gate。

## EMA mechanism comparison

paired unit 是“每个 episode 的六个 blocks 先取 median，再在 8 个 episodes 上比较”。
terminal EMA − online：

| 指标 | episode-median delta / gate |
|---|---:|
| Spearman | `-0.018833` / `NO-GO`（要求 ≥ 0） |
| top-30 | `-0.033333` / `NO-GO`（要求 ≥ 0） |
| joint non-worsening improvement | `2/8` / `NO-GO`（要求 ≥ 5/8） |

EMA 在 episode `13619`、`16705` 满足 joint rule；其余 6 个 episode 不满足。由于两个
episode-level median delta 都为负，secondary mechanism gate 为 **NO-GO**，且该 gate
不能替代 absolute gate。

## Latency、causality 与 GPU telemetry

计时边界是 cached official H=1 latent 加 normalized packed action prefix，执行 five-step
predictor；warmup `3`、repeats `10`，不含 encoder、CEM、environment。

| arm | student ms | teacher ms | reduction |
|---|---:|---:|---:|
| online | `1.851904` | `20.530176` | `90.9796%` |
| EMA | `1.841152` | `20.563457` | `91.0465%` |

两臂 causality prefixes `1–4` 的 `max_abs=0`，阈值 `1e-6`，全部 `PASS`。GPU telemetry
来自作业内 `nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total
-l 5`：A100-SXM4-40GB（40,960 MiB），日志样本约 `10:39:18–10:42:38`；初始化阶段
显存约 `1 MiB`，训练阶段约 `499–681 MiB`，GPU utilization 峰值 `40%`。PBS 资源摘要同时
报告 `cuda.3.maxGpuMemoryUsed=672 MiB`、`smUtilization_avg=19%`、`smUtilization_max=40%`、
`wallTime=80s`。这些 telemetry 只证明该 bounded PBS run 的资源使用，不支持 native
deployment 或端到端速度结论。

## Gate summary

| gate | status | 原因 |
|---|---|---|
| EMA absolute predictor | **NO-GO** | overall Spearman/top-30 median 与 minimum 不达标 |
| temporal stratum protection | **NO-GO** | middle/late median ranking 失败 |
| EMA mechanism secondary | **NO-GO** | paired median deltas 为负，joint `2/8` |
| primary | **NO-GO** | absolute 与 stratum 两项均失败 |
| integrity / convergence / causality | **PASS** | same trajectory、EMA isolation、收敛与 prefix invariance 均通过 |
| predictor latency | **PASS** | EMA reduction `91.0465%`，但计时边界仅 predictor |

## Interpretation and boundary

1. 这是 fresh temporal generalization 的负结果，而非旧 held-out leakage：fresh selection
   明确排除了旧 held-out 和旧 512 train episodes，且不依赖实验结果。
2. EMA 并未解决 middle/late anchor 的 ranking collapse；early-only 的通过不足以支持
   overall replacement 或 temporal robustness。
3. EMA 是同一 trajectory 的 concurrent smoothing arm，不能被解读为独立 seed 或独立
   statistical replicate。
4. `1.84 ms` predictor latency reduction 不是 LeWM planner、CEM、encoder 或 closed-loop
   success 的证据。
5. official CEM、planner viability、closed-loop：**NOT_RUN_BY_SCOPE**；本轮没有返回
   checkpoint、prepared rows、manifest 或 eval rows。

方法与 reporting 组织参考已核实的 [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, arXiv:2609.00065](https://arxiv.org/abs/2609.00065)。该引用只用于说明 reporting/method provenance，不是本实验的结果证据，也不改变本轮 gate。

## Artifact provenance

- [summary JSON](../lewm-transfer/ema-temporal-confirm/artifacts/25142797.pbs101/lewm_ema_temporal_confirm_summary.json)
- [job.log 与 5-second GPU telemetry](../lewm-transfer/ema-temporal-confirm/artifacts/25142797.pbs101/job.log)
- [job_status](../lewm-transfer/ema-temporal-confirm/artifacts/25142797.pbs101/job_status)
- [冻结 protocol](../lewm-transfer/ema-temporal-confirm/PROTOCOL_LEWM_EMA_TEMPORAL_CONFIRM.zh.md)
- [冻结 contract](../lewm-transfer/ema-temporal-confirm/LEWM_EMA_TEMPORAL_CONFIRM_FREEZE.json)
