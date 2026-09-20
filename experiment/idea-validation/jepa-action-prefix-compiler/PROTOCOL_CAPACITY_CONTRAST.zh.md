# DINO-WM PushT：WIDE+QUERY Capacity-Only Contrast 冻结协议

## 一句话目标

本轮只回答一个问题：在完全相同的 DINO-WM PushT `WIDE+QUERY` training bank、1500 updates、optimizer、seeds 和 held-out candidate blocks 下，把 `NativeDinoPrefixStudent` 的 `hidden_dim` 从 `128` 增加到 `256`，是否会 materially improve planner-candidate ranking？

这是一个单 cell 的 capacity-only contrast。唯一新增的 treatment 是 `hidden_dim=256`；不新增 query mixture、不增加 context pool、不改变 loss、不改变 held-out protocol，也不重跑 `hidden_dim=128`。

## 现有 baseline 与证据边界

权威 baseline 是已完成的 `24440788.pbs101` optimization-length run 的 `WIDE-QUERY` `step1500`，机器可读结果为：

`artifacts/24440788.pbs101/optimization_length_summary.json`

该 baseline 的 `hidden_dim=128`、连续 `1500` steps、同一 frozen WIDE+QUERY contract，且包含 16 个 held-out paired blocks 的逐 block ranking 结果。因此本轮只读取并引用其 `step1500` 结果；不重跑 h128，不加载或变形 h128 checkpoint，不修改 baseline summary。

baseline 参考值：

| 指标 | h128 step1500 |
|---|---:|
| Spearman median | `0.8824331462` |
| Spearman minimum | `0.7763445973` |
| top-30 median | `0.6000000000` |
| top-30 minimum | `0.4666666667` |
| logged teacher-relative MSE | `0.0666916159` |
| predictor-only student median | `9.8180 ms` |

本协议仍然只支持 predictor-level evidence；不支持 closed-loop、完整 CEM/control loop、environment success、LeWM transfer 或 observation encoder claim。

## Frozen treatment

| 项目 | 冻结值 |
|---|---|
| Student | `NativeDinoPrefixStudent` |
| Arm | `WIDE-QUERY` |
| 唯一架构变化 | `hidden_dim=256` |
| 输入 visual | `384` channels，context shape `[B,1,196,384]` |
| 输入 proprio | `10` channels，context shape `[B,1,10]` |
| action | `[B,5,10]`，每 token packing `5×2` primitive actions |
| 输出 | visual `[B,5,196,384]`、proprio `[B,5,10]` |
| causal Transformer | depth `2`、`nhead=4`、FFN `4×hidden=1024` |
| dropout / activation | `0.0` / GELU |
| future-action leakage | 禁止 |
| goal input | 禁止 |
| target loss | dense native-latent MSE only |

`hidden_dim=256` 必须使用与 baseline 相同的构造与 RNG procedure，但不能声称与 h128 使用相同 `state_dict`。两种宽度的参数形状不同；treatment 必须 fresh initialize，不得复制、reshape 或 interpolate h128 权重。

## Frozen WIDE+QUERY bank

training context 固定为 query-coverage manifest 中全部 `128` 个 WIDE contexts，来自 `32` 个 train episodes，每个 episode `4` 个 contexts；held-out contexts 完全排除。

每个 batch 固定 `32` 行：

- `16` logged action prefixes；
- `8` independent Gaussian planner-init prefixes；
- `8` one-step CEM elite-resample prefixes。

one-step CEM bank 固定为：`M=64`、`K=8`、per-coordinate variance floor `0.05`，使用同一 context 的 frozen teacher terminal objective。bank 在 step 1 前由 CPU generator 预计算一次；student 结果不能影响 bank。不得改变 mixture、M、K、variance floor、objective、context order、call order 或 seeds。

由于旧 baseline 的 WIDE schedule 保留了旧 query runner 的 RNG compatibility，新的 h256 runner 必须先执行等价的 `500`-step NARROW burn-in（`16000` context draws），再生成连续 `1500` steps 的 WIDE schedule。这样才能与 h128 baseline 的 WIDE context-index schedule 对齐。

## Training

| 项目 | 冻结值 |
|---|---|
| updates | `1500`，不得 early stop |
| batch | `32` |
| optimizer | AdamW |
| learning rate | `3e-4` |
| scheduler | none |
| teacher target | frozen、detached dense rollout |
| student encode_obs | false |
| student goal call | false |
| optimizer reset | false；单个 h256 treatment 从 step 1 连续到 step 1500 |

Seeds 固定为：`training_seed=20260925`、`context_schedule_seed=20260925`、`action_query_seed=20260926`、`role_schedule_seed=20260927`；held-out action seeds 为 `[20264925, 20264926]`，timing seed 为 `20270925`。

不得根据 step 中间结果换 seed、改 mixture、重算 CEM bank、调 learning rate、改 steps 或放宽 gates。

## Held-out ranking 与 paired comparison

使用全部 `8` 个 held-out contexts、`2` 个 fresh action-prefix seeds；每个 block `300` candidates，`top-k=30`，共 `16` blocks。h256 与 h128 baseline 必须按完全相同的 `(seed, context_index)` key 对齐。

对每个 block 使用同一 candidate action tensor、context、goal 与 frozen teacher cost，计算：

```text
ΔSpearman_i = Spearman_h256_i − Spearman_h128_i
Δtop30_i    = top30_h256_i − top30_h128_i
```

primary effect 使用 `16` 个逐 block delta 的 median，而不是两个 arm-level median 相减；正向 block count 是严格 `delta > 0` 的数量。candidate 不是独立 replicate，统计单位是 block。

同时报告：

- h256 absolute Spearman median / minimum；
- h256 absolute top-30 median / minimum；
- 每 block teacher-relative latent MSE 与 per-horizon 汇总；
- h256 与 h128 logged-action teacher-relative MSE ratio；
- 每 block `(seed, context_index)` 对齐记录。

## Frozen gates

### 1. Capacity and integrity

必须全部通过：所有 outputs/training finite；last-10/first training MSE ratio `≤0.8`；shape/dtype/device match；future-action leakage `≤1e-6`；没有 hidden `encode_obs`、source encoder reference、OOM、NaN 或 silent fallback。

### 2. Capacity effect（primary attribution）

`h256 step1500 − authoritative h128 step1500` 必须同时满足：

| Gate | 冻结门槛 |
|---|---:|
| median ΔSpearman | `≥ +0.05` |
| median Δtop-30 | `≥ +0.10` |
| positive Spearman blocks | `≥ 12/16` |
| positive top-30 blocks | `≥ 12/16` |

四项全部通过才标记 `capacity_effect_established`；否则标记 `capacity_effect_not_established`。该 attribution label 与 replacement readiness 分开报告。

### 3. Absolute fidelity（unchanged）

| Gate | 冻结门槛 |
|---|---:|
| h256 median Spearman | `≥0.99` |
| h256 minimum Spearman | `≥0.95` |
| h256 median top-30 | `≥0.95` |
| h256 minimum top-30 | `≥0.80` |

### 4. Logged-action non-inferiority（unchanged ratio）

在相同的 `8` 个 held-out logged prefixes 上，

```text
MSE_ratio = mean_relative_MSE(h256) / mean_relative_MSE(h128)
```

必须 `≤1.25`。ranking 或 training loss 不能替代该 gate。

### 5. Causality 与 latency

- 对每个 prefix horizon 改变 future action tokens，最大 output delta 必须 `≤1e-6`。
- latency boundary 固定为 cached native observation latent + normalized action prefix；不包括 encoder、CEM orchestration、environment 或 closed-loop。
- batch `300`，warmup `3`，technical repeats `10`，前后 CUDA synchronize。
- h256 对 frozen teacher 的 predictor-only reduction 必须 `≥20%`。

parameter count `h256/h128` 与 predictor latency `h256/h128` 只作 diagnostic，不能作为 success gate；也不能将 latency ratio 写成完整 planner 或 control-loop speedup。

## Decision table

| 当前结果 | 下一步 |
|---|---|
| cell invalid（OOM/NaN/leakage/fallback 等） | 不得据此进入 density contrast；不得事后改成 h192、换 seed 或调参 |
| valid，`capacity_effect=PASS`，replacement `NO-GO` | 仍视为 capacity-active；本 freeze 不授权 data/query density contrast |
| valid，`capacity_effect=PASS`，replacement `GO` | predictor-level replacement 阶段结束 |
| valid，`capacity_effect=FAIL`，replacement `NO-GO` | 才进入单独的 data/query coverage-density freeze；固定 h256、1500 steps、loss、seeds、held-out blocks 与 latency boundary，只改变一个 coverage/density factor |

无论哪种结果，都不得把本轮解释为 closed-loop success、multi-step CEM integration、full-plan latency、LeWM transfer 或 population-level inference。

## Execution boundary and artifacts

所有模型加载、teacher rollout、training、evaluation、GPU telemetry 和重 I/O 必须在有 `PBS_JOBID` 的 compute-node allocation 内完成；login node 只允许轻量控制和状态查询。

本轮 primary artifacts 为 summary JSON、job log/status、GPU info/usage/telemetry。checkpoint 若存在也只是 audit artifact；不要求取回作为主要结果。

本协议与 [CAPACITY_CONTRAST_FREEZE.json](./CAPACITY_CONTRAST_FREEZE.json) 一起冻结；创建 freeze 后不得在观察结果后修改 protocol 或 gates。
