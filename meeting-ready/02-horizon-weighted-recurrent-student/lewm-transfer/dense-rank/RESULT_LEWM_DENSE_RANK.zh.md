# LeWM PushT：Dense Query + Elite-Boundary Rank Distillation 结果

## 结论先行

正式作业 `24908446.pbs101` 在 official LeWM PushT backend 上正常完成三臂
predictor-level training/evaluation（`Exit_status=0`，PBS walltime `00:05:32`）。
增加 train query coverage 并加入 teacher top-6 对 ranks 7–12 的 pairwise rank loss
后，`dense_rank` 相对同预算 `dense_latent` 的 held-out top-30 median 提高
`+0.166667`，16 个 paired blocks 中 `14` 个改善；Spearman median 提高
`+0.122187`，`14/16` 个 blocks 改善。固定的 screening gate 为 **GO**。

但 `dense_rank` 的 absolute predictor fidelity/ranking gate 仍为 **FAIL**（Spearman
median `0.701261`、top-30 median `0.433333`，仍低于 replacement 阈值），所以
`predictor_feasibility` 仍为 **NO-GO**。这说明 rank supervision 是目前第一个明显、且
由 shuffled control 支持的正向训练信号，但还不足以把 student 变成 planner-facing
predictor replacement。

## 1. 冻结实验与公平性

- backend：official LeWM PushT；LeWM commit
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`；stable-worldmodel CEM commit
  `10c26dbd5677083fa31dba69eb738b973845e9a4`。
- 三臂均为原始 `LeWMCompactRecurrentTransitionStudent`：latent `192-D`、packed
  action `10-D`、shared recurrent `h256`，无 attention、conditioner、goal input 或
  teacher forcing；参数量均为 `775,872`。
- 每个 train context 使用同一 frozen `64-candidate` teacher bank；三臂共享
  `256` 个 train contexts、同一 initial state、context schedule、candidate bank、
  training budget（`1500` updates）和 held-out bank。
- held-out 为 `8 contexts × 2 fresh seeds = 16 blocks`，每 block `300` candidates，
  top-k=`30`；block 是统计单位，candidate 不是独立 replicate。
- `dense_latent`：原 horizon-weighted free-running latent MSE；
  `dense_rank`：同一 latent loss 加 context-normalized pairwise logistic rank loss；
  `shuffled_rank`：同样 pair 数但在 context 内固定打乱 teacher labels 的 negative control。
- rank loss 固定为 teacher top `6` 对 ranks `7–12`，temperature `0.5`，loss weight
  `0.1`；goal 只进入 external official criterion，不输入 student。

## 2. Held-out snapshot 结果

以下为每个 snapshot 的 16-block median/minimum；正式 screening 只使用 step 1500。

### `dense_latent`

| snapshot | Spearman median / min | top-30 median / min | relative latent MSE median |
|---|---:|---:|---:|
| step 500 | `0.015869 / -0.262144` | `0.083333 / 0.000000` | `0.017545` |
| step 1000 | `0.328187 / -0.226948` | `0.216667 / 0.000000` | `0.010885` |
| step 1500 | `0.514727 / -0.301539` | `0.266667 / 0.000000` | `0.009985` |

### `dense_rank`

| snapshot | Spearman median / min | top-30 median / min | relative latent MSE median |
|---|---:|---:|---:|
| step 500 | `0.555102 / -0.503822` | `0.366667 / 0.000000` | `0.022030` |
| step 1000 | `0.599983 / -0.283017` | `0.350000 / 0.000000` | `0.017620` |
| step 1500 | `0.701261 / -0.368217` | `0.433333 / 0.000000` | `0.025179` |

step 1500 的 positive top-30 blocks 为 `dense_latent=15/16`、`dense_rank=15/16`。
Rank loss 提升了整体 ranking median，但没有改善最差 block，且 relative latent MSE
高于 dense latent；这符合“针对局部排序优化可能牺牲全局 latent fidelity”的预期。

### `shuffled_rank` negative control

| snapshot | Spearman median / min | top-30 median / min | relative latent MSE median |
|---|---:|---:|---:|
| step 500 | `0.088460 / -0.570930` | `0.133333 / 0.000000` | `0.021312` |
| step 1000 | `0.008240 / -0.558662` | `0.083333 / 0.000000` | `0.015491` |
| step 1500 | `0.086698 / -0.335159` | `0.100000 / 0.033333` | `0.024909` |

Shuffled control 没有复现 dense-rank 收益，反而相对 dense latent 的 top-30 median
delta 为 `-0.150000`（`2` improve / `12` worse / `2` tie），支持收益来自正确的
teacher ordering，而不是额外 loss 或候选覆盖本身。

## 3. Paired screening gate

### `dense_rank - dense_latent`

| 指标 | improve / worse / tie | median delta | 冻结要求 | 判定 |
|---|---:|---:|---:|---|
| top-30 overlap | `14 / 1 / 1` | `+0.166667` | median `>=+0.10`，改善 `>=10/16` | PASS |
| Spearman | `14 / 2 / 0` | `+0.122187` | median 不下降 | PASS |
| minimum top-30 | `0 → 0` | 不下降 | 不下降 | PASS |
| positive top-30 blocks | `15 → 15` | 不下降 | 不下降 | PASS |

六项 screening conditions（包括 shuffled control 不能复现收益）全部满足，summary
记录的 screening status 为 **GO**。这只是“是否值得继续研究 rank-aware training”的
筛选 gate，不等于 official predictor replacement gate。

## 4. Absolute predictor gates、causality 与 latency

三臂的 integrity/convergence、causality 和 predictor-only latency 均 PASS；三臂的
fidelity/ranking 均 FAIL，因此三臂 `predictor_feasibility` 均为 **NO-GO**。

| arm | student / teacher latency | reduction | Spearman median / min | top-30 median / min | absolute feasibility |
|---|---:|---:|---:|---:|---|
| dense_latent | `1.585152 / 20.670977 ms` | `92.3315%` | `0.514727 / -0.301539` | `0.266667 / 0` | **NO-GO** |
| dense_rank | `1.645056 / 19.951616 ms` | `91.7548%` | `0.701261 / -0.368217` | `0.433333 / 0` | **NO-GO** |
| shuffled_rank | `1.571840 / 20.896255 ms` | `92.4779%` | `0.086698 / -0.335159` | `0.100000 / 0.033333` | **NO-GO** |

Causality 对 unchanged prefix length `1/2/3/4` 的 maximum absolute difference 均为
`0.0`（阈值 `1e-6`）。Latency boundary 是 cached `H=1` latent + normalized packed
action prefix → five-step predictor；不包含 encoder、CEM、environment 或 closed-loop。

## 5. 运行证据与 scope

- PBS：`24908446.pbs101`，`job_state=F`，`Exit_status=0`，compute host
  `x1000c2s1b0n1`。
- GPU：A100-SXM4-40GB；job log 每 5 秒采样。训练期间有多条非零利用率样本，
  observed utilization 最高 `80%`，observed memory 最高 `697 MiB`；不是空跑或只做
  启动检查。
- 三臂都达到 `PREDICTOR_LEVEL_COMPLETE`；last10/first training-loss ratio 分别为
  `dense_latent=0.003098`、`dense_rank=0.005975`、`shuffled_rank=0.005894`。
- `summary.manifest` 复用原正式 LeWM student 的
  `24564619.pbs101/context_manifest.json`；`24908446` 输出目录本身没有额外的
  job-specific `context_manifest.json`，因此没有伪造副本。该 manifest 的本地副本仍在
  [baseline context manifest](../artifacts/24564619.pbs101/context_manifest.json)。
- `24884626.pbs101` 是 RNG parity 修复前仍排队的旧 job，运行前已 `qdel`；
  `24897213.pbs101` 在进入三臂前因 `HORIZON` `NameError` 失败；两者均不是结果。
  `24908446.pbs101` 是唯一正式结果。

## 6. 决策与 claim boundary

| gate | result |
|---|---|
| dense-rank paired screening | **GO** |
| integrity/convergence | **PASS** |
| causality | **PASS** |
| predictor-only latency | **PASS** |
| absolute predictor fidelity/ranking | **FAIL** |
| predictor feasibility | **NO-GO** |
| official CEM viability | `NOT_RUN_BY_SCOPE` |
| planner viability | `NOT_RUN_BY_SCOPE` |
| closed-loop PushT | `NOT_RUN_BY_SCOPE` |

本轮支持的结论是：在不改变 h256 shared recurrent student structure 的前提下，
增加 64-query coverage 并加入 frozen teacher elite-boundary rank distillation，能在
相同 held-out protocol 上产生明显的 ranking 改善，而且 shuffled labels 不会产生同样
收益；但 absolute ranking 仍不足以支持替换 LeWM teacher。按冻结边界，没有运行 official
CEM、planner viability 或 closed-loop，也不能声称 end-to-end control improvement、
encode_obs speedup 或 Fast-LeWM comparison。

## 7. Artifacts

- [formal summary](artifacts/24908446.pbs101/lewm_dense_rank_summary.json)
- [job log with 5-second GPU telemetry](artifacts/24908446.pbs101/job.log)
- [job status](artifacts/24908446.pbs101/job_status)
- [context manifest reused from baseline](../artifacts/24564619.pbs101/context_manifest.json)
- [freeze protocol](LEWM_DENSE_RANK_FREEZE.json)
- [frozen experiment protocol](PROTOCOL_LEWM_DENSE_RANK.zh.md)
