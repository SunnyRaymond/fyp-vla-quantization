# DINO-WM PushT：WIDE+QUERY Optimization-Length 结果

## 结论

本轮 frozen optimization-length diagnostic 正常完成，但结论是：

- `optimization_materially_helps = FAIL`：从连续 `500` steps 延长到 `1500` steps，没有在同一组 held-out paired blocks 上建立 material ranking improvement。
- `full_replacement = NO-GO`：当前 `hidden_dim=128` student 的 absolute ranking fidelity 仍不足以替代 frozen teacher predictor。
- 因此，在本 frozen DINO-WM PushT predictor-level setup 中，训练长度不是主要限制；下一步应优先判别 student capacity 与 data/query coverage，而不是只增加 optimizer steps。

这里的 `NO-GO` 是 replacement contract 的结果，不是 implementation failure，也不否定 WIDE+QUERY action-query coverage 方向。实验只涉及 DINO-WM PushT 的 predictor-level evaluation；不支持 closed-loop、完整 CEM/control-loop、LeWM transfer 或 observation encoder claim。

## 实验身份与证据边界

- PBS job：`24440788.pbs101`
- 状态：`F`，`Exit_status=0`，`final_exit_code=0`；compute host `x1000c0s0b0n1`
- walltime：`00:06:47`
- Student：`NativeDinoPrefixStudent`，`hidden_dim=128`，只训练一个 `WIDE-QUERY` arm
- Training：连续 `1500` updates，batch `32`，AdamW，learning rate `3e-4`，optimizer 未重置
- Query mixture：`50% logged + 25% Gaussian planner-init + 25% one-step CEM elite-resample`
- Held-out：8 contexts × 2 fresh seeds = 16 paired blocks；每 block 300 candidates，top-k=30
- Snapshots：`step500`、`step1000`、`step1500`；训练完成后统一进行 held-out evaluation
- GPU：NVIDIA A100-SXM4-40GB；`gpu_usage.csv` 有 13 个 30-second samples，`gpu_telemetry.jsonl` 有 14 条（含 complete）；峰值显存 `26,400 MiB`

冻结的 student、teacher targets、action bank、seeds、held-out candidates、loss、optimizer、thresholds 和 gates 均未改变；没有取回 checkpoint 或 `plan_targets.pkl`。

## Training convergence

| 指标 | 数值 |
|---|---:|
| first latent MSE | 0.392576 |
| step 500 loss | 0.153307 |
| step 1000 loss | 0.115430 |
| step 1500 loss | 0.094236 |
| last-10 latent MSE median | 0.105927 |
| last-10 / first ratio | 0.269824（门槛 ≤ 0.8） |

Training finite、shape/dtype/device match、student 与 teacher 不相同，且没有 hidden `encode_obs`、source encoder reference、OOM/NaN 或 silent fallback。`capacity_and_integrity = PASS`。这说明本轮不是因训练未运行、数值异常或误调用 teacher 而失败。

## Step 500/1000/1500 held-out snapshot

| Snapshot | Spearman mean | Spearman median | Spearman min | top-30 mean | top-30 median | top-30 min |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.872667 | 0.881070 | 0.800395 | 0.604167 | 0.616667 | 0.466667 |
| 1000 | 0.869889 | 0.881668 | 0.765865 | 0.627083 | 0.616667 | 0.400000 |
| 1500 | 0.869994 | 0.882433 | 0.776345 | 0.633333 | 0.600000 | 0.466667 |

Step 1000 和 step 1500 的 ranking 中位数基本停滞；top-30 median 在 step 1500 反而低于 step 500/1000。

## Logged-action teacher-relative MSE

这是 8 个 held-out logged action prefixes 上相对 frozen teacher 的 native-latent MSE；primary ratio 是 `step1500 / step500`。

| Snapshot | Mean relative MSE | H=1 | H=2 | H=3 | H=4 | H=5 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.067501 | 0.026444 | 0.067639 | 0.069678 | 0.077156 | 0.096587 |
| 1000 | 0.060128 | 0.022503 | 0.057794 | 0.061657 | 0.074517 | 0.084167 |
| 1500 | 0.066692 | 0.021053 | 0.062199 | 0.069044 | 0.087425 | 0.093737 |

`step1500 / step500 = 0.988011 ≤ 1.25`，因此 `teacher_relative_mse_final = PASS`。这表明延长训练没有损坏 logged-action fidelity，但也没有转化为 ranking 的明显收益。

## Paired optimization effect（Level 1）

Primary effect 严格使用同一 16 个 paired blocks 的逐 block delta，再取 median；不是 delta of medians。

| Comparison | median ΔSpearman | median Δtop-30 | Spearman positive | top-30 positive |
|---|---:|---:|---:|---:|
| 1000 − 500（diagnostic） | −0.004252 | +0.033333 | 6/16 | 9/16 |
| 1500 − 500（primary） | −0.000368 | +0.016667 | 8/16 | 8/16 |
| 1500 − 1000（trajectory） | −0.000021 | 0.000000 | 8/16 | 6/16 |

冻结 Level-1 门槛为：median ΔSpearman `≥ +0.05`、median Δtop-30 `≥ +0.10`、两类正向 blocks 各至少 `12/16`。Primary 四项均未通过，因此 `optimization_materially_helps = FAIL`。这直接支持“在当前 frozen WIDE+QUERY bank 上继续延长训练不是主要杠杆”。

## Level 2：frozen full-replacement gates

| Gate | 实际 | 冻结门槛 | 判定 |
|---|---:|---:|---|
| Capacity/integrity | finite；ratio 0.269824；无 fallback/leakage | 全部通过；ratio ≤ 0.8 | PASS |
| Final median Spearman | 0.882433 | ≥ 0.99 | FAIL |
| Final minimum Spearman | 0.776345 | ≥ 0.95 | FAIL |
| Final median top-30 | 0.600000 | ≥ 0.95 | FAIL |
| Final minimum top-30 | 0.466667 | ≥ 0.80 | FAIL |
| Logged teacher-relative MSE ratio | 0.988011 | ≤ 1.25 | PASS |
| Future-action leakage | max abs delta 0.0 | ≤ 1e-6 | PASS |
| Predictor-only latency reduction | 99.7198% | ≥ 20% | PASS |

Latency boundary 是 cached native observation latent + normalized action prefix 到五步 predictor output，batch=300、warmup=3、repeats=10，并包含 CUDA synchronization；teacher median `3504.480 ms`，student median `9.818 ms`，median speedup `356.94×`。它是 predictor-only result，不能写成完整 planner 或 closed-loop control speedup。

Level 2 的 absolute ranking gates 失败，所以 `full_replacement = NO-GO`，即使 MSE、causality、latency 和 implementation integrity 均通过。

## Capacity 与 data/query coverage：下一步如何判

当前证据把“只增加 optimization length”降为 no-go：loss 已收敛，step 500→1500 的 paired ranking delta 近零，而 final absolute ranking 仍约为 Spearman median `0.882`、top-30 median `0.600`。因此不应仅把本轮当作“还需要更多 steps”。

下一步应保持现有 WIDE+QUERY bank、held-out blocks、seeds、loss、latency boundary 和 gates，先做最小的 capacity-only contrast（例如只改变 student width/parameterization），以检验 absolute ranking ceiling 是否随 capacity 移动；不能同时重调 query mixture 或放宽 gate。

data/query coverage 要单独判断：若要检验 data coverage，固定 student capacity 与 optimization length，只改变预先定义并可审计的 context/action-query support，继续使用相同 held-out block 评估。已有本地 WIDE+QUERY 结果支持 planner-query coverage 是有效方向，但不允许据此自动 retune 本轮冻结配方。若 capacity-only contrast 不提升 absolute ranking，再做独立的 coverage contrast，才能区分“表示容量不足”和“训练分布覆盖不足”。

## 可以与不可以声称什么

可以声称：在 frozen DINO-WM PushT predictor-level contract 内，连续 1500 steps 相对 500 steps 没有建立预注册的 material ranking improvement；当前 hidden=128 student 未达到 near-lossless teacher replacement，但 predictor-only latency、causality、logged-action MSE ratio 与 implementation integrity 通过。

不可以声称：closed-loop task success、完整 CEM/环境控制加速、LeWM transfer、observation encoder acceleration、all-JEPA universality、population-level inference，或“增加训练长度已证明模型容量足够/不足”的普遍结论。本结论只针对 frozen WIDE+QUERY、DINO-WM PushT、hidden_dim=128 与本次 16 个 held-out paired blocks。

## 产物

- Summary：`artifacts/24440788.pbs101/optimization_length_summary.json`
- Job status/log：`artifacts/24440788.pbs101/job_status.txt`、`artifacts/24440788.pbs101/job.log`
- GPU telemetry：`artifacts/24440788.pbs101/gpu_info.csv`、`gpu_usage.csv`、`gpu_telemetry.jsonl`
- Exit status：`artifacts/24440788.pbs101/final_exit_status.txt`、`runner_exit_status.txt`
- 未取回：`wide_query_step*.pt` checkpoints、`plan_targets.pkl`
