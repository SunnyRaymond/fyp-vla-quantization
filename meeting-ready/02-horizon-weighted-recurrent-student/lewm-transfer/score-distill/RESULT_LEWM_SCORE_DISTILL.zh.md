# LeWM PushT：Bounded Score Distillation 结果

## 结论先行

正式作业 `24916520.pbs101` 在 official LeWM PushT backend 上完成了三臂
predictor-level training/evaluation（`Exit_status=0`，PBS walltime `00:03:44`）。
`score_distill` 用全 64 个 candidates 的 context-normalized teacher score
SmoothL1，取代无界的 elite-boundary pairwise margin。相对 concurrent
`pairwise_rank`，step-1500 的 Spearman median、top-30 median 和 relative latent
MSE 都改善：分别为 `+0.268554`、`+0.300000` 和 `-0.007516`。

但冻结的 score-distill screening gate 仍为 **NO-GO**：`positive_top30_blocks` 为
`15/16` 而要求恰为 `16/16`，minimum top-30 overlap 仍为 `0`。因此 bounded score
distillation 是更强的 predictor-level 训练信号，但尚不能宣称通过 screening，也不能
支持替换 LeWM teacher。三臂 absolute predictor feasibility 均为 **NO-GO**。

## 1. 冻结实验与公平性

- backend：official LeWM PushT；LeWM commit
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`；stable-worldmodel CEM commit
  `10c26dbd5677083fa31dba69eb738b973845e9a4`。
- 三臂均为原始 `LeWMCompactRecurrentTransitionStudent`：latent `192-D`、packed
  action `10-D`、shared recurrent `h256`，无 attention、conditioner、goal input 或
  teacher forcing；参数量均为 `775,872`。
- 三臂复用 dense-rank formal job `24908446.pbs101` 的 `prepared_rows.pt`，并共享
  initial state、context schedule、64-candidate train bank、held-out bank 和
  `1500` updates；没有重新生成或下载 prepared rows。
- held-out 为 `8 contexts × 2 fresh seeds = 16 blocks`，每 block `300` candidates，
  top-k=`30`；block 是统计单位，candidate 不是独立 replicate。
- `pairwise_rank`：已验证的 teacher top-6 对 ranks 7–12 pairwise logistic reference，
  temperature `0.5`、loss weight `0.1`。
- `score_distill`：对全 64 candidates 做 context normalization，
  `t_norm=(t-mean(t))/clamp(std(t))`，student score 使用同一 teacher std，
  SmoothL1 `beta=1`；teacher ranks 1–12 权重 `2`，其余权重 `1`，loss weight `0.1`。
- `shuffled_score`：固定 context 内 shuffled teacher scores 的 negative control。

## 2. Held-out snapshot 结果

以下为每个 snapshot 的 16-block median/minimum；正式 screening 使用 step 1500。

### `pairwise_rank` concurrent reference

| snapshot | Spearman median / min | top-30 median / min | relative latent MSE median |
|---|---:|---:|---:|
| step 500 | `0.555102 / -0.503822` | `0.366667 / 0` | `0.022030` |
| step 1000 | `0.599983 / -0.283017` | `0.350000 / 0` | `0.017620` |
| step 1500 | `0.701261 / -0.368217` | `0.433333 / 0` | `0.025179` |

### `score_distill`

| snapshot | Spearman median / min | top-30 median / min | relative latent MSE median |
|---|---:|---:|---:|
| step 500 | `0.736084 / -0.790968` | `0.533333 / 0` | `0.020829` |
| step 1000 | `0.952318 / -0.231695` | `0.733333 / 0` | `0.015060` |
| step 1500 | `0.974011 / -0.010504` | `0.816667 / 0` | `0.013728` |

step 1500 的 `score_distill` 有 `15/16` 个 positive Spearman blocks 和 `15/16` 个
positive top-30 blocks；最后一个 block 的 top-30 overlap 仍为 `0`，所以最差 block
没有被消除。

### `shuffled_score` negative control

| snapshot | Spearman median / min | top-30 median / min | relative latent MSE median |
|---|---:|---:|---:|
| step 500 | `-0.058420 / -0.260947` | `0.066667 / 0` | `0.017640` |
| step 1000 | `0.185050 / -0.102595` | `0.166667 / 0` | `0.010950` |
| step 1500 | `0.233274 / -0.168441` | `0.183333 / 0` | `0.011658` |

shuffled control 没有复现 `score_distill` 的 ranking 改善；相对 `pairwise_rank` 的
step-1500 top-30 delta 为 `-0.300000`（`1/15/0` improve/worse/tie），Spearman
delta 为 `-0.463372`（`2/14/0`）。它的 latent MSE 较低不能单独构成 ranking 成功
证据。

## 3. Paired comparison 与 screening gate

### `score_distill - pairwise_rank`

| 指标 | improve / worse / tie | median delta | 冻结要求 | 判定 |
|---|---:|---:|---:|---|
| Spearman | `16 / 0 / 0` | `+0.268554` | median `>=0.701261` | PASS |
| top-30 overlap | `15 / 1 / 0` | `+0.300000` | median `>=0.433333` | PASS |
| relative latent MSE | `16 / 0 / 0` | `-0.007516` | median `<=0.0175` | PASS |
| positive top-30 blocks | `15/16` | — | 恰 `16/16` | **FAIL** |
| minimum top-30 | `0` | — | strictly positive | **FAIL** |

shuffled-score 不能匹配 score-distill 的 ranking 收益这一条件为 **PASS**，但上述
两项 worst-case 条件失败，summary 的 screening status 因此为 **NO-GO**。这支持
“bounded objective 减少 latent drift、显著改善整体排序”的局部结论，不支持
“所有 held-out blocks 均安全”或“已通过下一阶段 gate”。

## 4. Absolute predictor gates、causality 与 latency

三臂 integrity/convergence、causality 和 predictor-only latency 均 PASS；三臂的
absolute fidelity/ranking 均 FAIL。

| arm | student / teacher latency | reduction | Spearman median / min | top-30 median / min | relative MSE | feasibility |
|---|---:|---:|---:|---:|---:|---|
| pairwise_rank | `1.839104 / 20.381696 ms` | `90.9767%` | `0.701261 / -0.368217` | `0.433333 / 0` | `0.025179` | **NO-GO** |
| score_distill | `1.852416 / 22.257153 ms` | `91.6772%` | `0.974011 / -0.010504` | `0.816667 / 0` | `0.013728` | **NO-GO** |
| shuffled_score | `1.844224 / 21.670913 ms` | `91.4899%` | `0.233274 / -0.168441` | `0.183333 / 0` | `0.011658` | **NO-GO** |

`score_distill` 的 absolute predictor gate 仍因 minimum Spearman、minimum top-30 等
worst-case replacement thresholds 失败；它不是一个已经可以 planner-facing 使用的
student。Causality 对 unchanged prefix length `1/2/3/4` 的 maximum absolute
difference 均为 `0.0`（阈值 `1e-6`）。Latency boundary 是 cached `H=1` latent +
normalized packed action prefix → five-step predictor；不包含 encoder、CEM、environment
或 closed-loop。

## 5. 运行证据与 scope

- PBS：`24916520.pbs101`，`job_state=F`，`Exit_status=0`，compute host
  `x1000c0s0b0n0`，walltime `00:03:44`。
- GPU：A100-SXM4-40GB；job log 每 5 秒采样。训练期间有多条非零 utilization 样本，
  observed utilization 最高 `40%`，observed memory 最高 `697 MiB`；不是空跑或只做
  启动检查。
- 三臂都达到 `PREDICTOR_LEVEL_COMPLETE`；last10/first training-loss ratio 分别为
  `pairwise_rank=0.005975`、`score_distill=0.003505`、`shuffled_score=0.003341`。
- `summary.manifest` 复用正式 LeWM student 的
  `24564619.pbs101/context_manifest.json`；没有为本 job 伪造 job-specific manifest。
- 只从远端拉取了 `lewm_score_distill_summary.json`、`job.log` 和 `job_status`；没有拉取
  checkpoint 或 `prepared_rows.pt`。

## 6. 决策与 claim boundary

| gate | result |
|---|---|
| score-distill paired screening | **NO-GO** |
| integrity/convergence | **PASS** |
| causality | **PASS** |
| predictor-only latency | **PASS** |
| absolute predictor fidelity/ranking | **FAIL** |
| predictor feasibility | **NO-GO** |
| official CEM viability | `NOT_RUN_BY_SCOPE` |
| planner viability | `NOT_RUN_BY_SCOPE` |
| closed-loop PushT | `NOT_RUN_BY_SCOPE` |

本轮支持的结论是：在完全相同的 h256 recurrent student、candidate bank、held-out
protocol 和 training budget 下，bounded context-normalized score distillation 相对
pairwise rank 同时提升 ranking median 并降低 relative latent MSE，且 shuffled labels
不能复现该收益；然而仍有一个 held-out block 的 top-30 overlap 为零，因此 frozen
screening 和 absolute replacement gates 都没有通过。按 scope 没有运行 official CEM、
planner viability 或 closed-loop，也不能声称 end-to-end control improvement、
encode_obs speedup 或 Fast-LeWM comparison。

## 7. Artifacts

- [formal summary](artifacts/24916520.pbs101/lewm_score_distill_summary.json)
- [job log with 5-second GPU telemetry](artifacts/24916520.pbs101/job.log)
- [job status](artifacts/24916520.pbs101/job_status)
- [context manifest reused from baseline](../artifacts/24564619.pbs101/context_manifest.json)
- [freeze](LEWM_SCORE_DISTILL_FREEZE.json)
- [frozen protocol](PROTOCOL_LEWM_SCORE_DISTILL.zh.md)
