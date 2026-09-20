# DINO-WM PushT：Capacity-only `hidden=256` 结果

## 结论

本轮 frozen capacity-only contrast 已在 compute node 正常完成。结论分为两个层次：

- `capacity_and_integrity = PASS`：训练、配对、数值、causality 与实现边界均有效。
- `capacity_effect = FAIL`：在固定的 WIDE+QUERY、1500-step contract 下，`hidden=256` 没有相对 authoritative `hidden=128` 建立预注册的 ranking improvement。
- `full_replacement = NO-GO`：h256 的 absolute ranking fidelity 仍未达到 replacement gates；logged-action MSE、causality 和 predictor-only latency 虽通过，不能抵消 absolute fidelity 失败。

由于本 cell 有效、`capacity_effect=FAIL` 且 `full_replacement=NO-GO`，下一步按 freeze 进入独立的 **context-support density contrast**：固定 h256、1500 steps、loss、learning rate、seeds、held-out blocks 与 latency boundary，只改变一个预先定义的 coverage/density factor。本轮不授权 retune，也不支持 closed-loop 或完整 planner 结论。

## 实验身份与边界

- PBS job：`24456882.pbs101`
- 状态：`F`，PBS `Exit_status=0`；`job_status.txt` 为 `completed`，runner/final exit 均为 `0`
- Compute host：`x1000c2s1b0n1`；walltime：`00:06:51`
- Treatment：`NativeDinoPrefixStudent`，`hidden_dim=256`
- Read-only baseline：同一 frozen contract 的 `hidden_dim=128`、step 1500 summary；不重跑、不加载或变形 h128 checkpoint
- Training：连续 `1500` updates，batch `32`，AdamW，learning rate `3e-4`，optimizer 未重置
- Query mixture：`50% logged + 25% Gaussian planner-init + 25% one-step CEM elite-resample`
- Held-out：8 contexts × 2 fresh action-prefix seeds = 16 paired blocks；每 block 300 candidates，top-k=30
- Snapshots：step 500、1000、1500；训练后统一进行 held-out evaluation

这份结果只覆盖 DINO-WM PushT 的 predictor-level boundary：cached native observation latent + normalized action prefix → five-step predictor output。它不包括 observation encoder、CEM orchestration、environment interaction 或 closed-loop control。

## 训练与完整性

| 指标 | 数值 | 判定 |
|---|---:|---|
| first latent MSE | 0.404592 |  |
| last latent MSE | 0.075085 |  |
| last-10 latent MSE median | 0.082817 |  |
| last-10 / first ratio | 0.204692 | ≤ 0.8，PASS |
| future-action leakage max abs | 0.0 | ≤ 1e-6，PASS |
| shape/dtype/device match | true | PASS |
| outputs/training finite | true | PASS |
| hidden `encode_obs` / source encoder reference | none | PASS |
| OOM/NaN/silent fallback | none | PASS |

Pairing integrity 为 `PASS`：16 个 `(seed, context_index)` blocks 完整对齐，candidate contract 为 300 candidates、top-k=30、CPU `torch.Generator`、seed outer/context inner order。

GPU 运行也有记录：NVIDIA A100-SXM4-40GB（40,960 MiB），`gpu_usage.csv` 有 14 个 30-second samples，`gpu_telemetry.jsonl` 有 14 条（含 `complete`），峰值显存 26,393 MiB。没有在 login node 执行计算、模型加载或重 I/O。

## Held-out absolute ranking

| Snapshot | Spearman mean | Spearman median | Spearman min | top-30 mean | top-30 median | top-30 min |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.870973 | 0.878038 | 0.789250 | 0.631250 | 0.600000 | 0.533333 |
| 1000 | 0.863568 | 0.863580 | 0.791963 | 0.625000 | 0.600000 | 0.533333 |
| 1500 | 0.859979 | 0.853065 | 0.797216 | 0.627083 | 0.600000 | 0.466667 |

Frozen h256 absolute gates 为 median Spearman `≥0.99`、minimum Spearman `≥0.95`、median top-30 `≥0.95`、minimum top-30 `≥0.80`。step 1500 的四项分别为 `0.853065`、`0.797216`、`0.600000`、`0.466667`，因此全部未通过。

## Capacity effect：h256 − h128

Primary comparison 使用同一 16 个 paired blocks 的逐 block delta，再取 median，而不是两个 arm-level median 相减。

| Gate | 实际值 | frozen threshold | 判定 |
|---|---:|---:|---|
| median ΔSpearman | −0.012570 | ≥ +0.05 | FAIL |
| median Δtop-30 | −0.016667 | ≥ +0.10 | FAIL |
| positive Spearman blocks | 5/16 | ≥ 12/16 | FAIL |
| positive top-30 blocks | 7/16 | ≥ 12/16 | FAIL |

所以 `capacity_effect = FAIL`（`capacity_effect_not_established`）。step 1500 的 h256 并未表现出 width-only 的正向 ranking effect；这只否定当前 frozen training/support contract 下的该 attribution，不等同于对所有更大 capacity 方案的普遍 no-go。

## Logged-action fidelity、causality 与 latency

### Logged-action teacher-relative MSE

| Snapshot | h256 mean relative MSE |
|---:|---:|
| 500 | 0.066760 |
| 1000 | 0.067060 |
| 1500 | 0.076642 |

step 1500 的 h256/h128 ratio 为 `0.076642 / 0.066692 = 1.149199 ≤ 1.25`，所以 `teacher_relative_mse_final = PASS`。

### Causality

改变 future action tokens 的最大 output delta 为 `0.0`（四个 unchanged-prefix cases 均通过），`causality_final = PASS`。

### Predictor-only latency

- frozen teacher median：`3503.321 ms`
- h256 student median：`16.490 ms`
- h256 对 teacher reduction：`99.5293%`（gate `≥20%`，PASS）
- h256 / h128 student latency ratio：`1.679569`（h256 更慢；diagnostic only）

这个 reduction 只说明 predictor-level cached boundary 的测量结果，不能写成完整 planner 或 control-loop speedup。

## Parameter diagnostic

- h256 trainable parameters：`2,413,450`
- h128 trainable parameters：`682,634`
- h256/h128 parameter ratio：`3.535496`

parameter ratio 与 student latency ratio 都是 diagnostic，不是 success gate。summary 中 `hidden_128_same_class.hidden_dim` 字段因 runner helper 使用 treatment 常量而序列化为 `256`；实际 h128 diagnostic model 是按 `BASELINE_WIDTH=128` 构造的，参数计数和 ratio 使用的是该实际 h128 model。这个字段标签问题不改变本轮 gates 或实验决策。

## 决策与下一步

本轮不是 implementation failure：训练完成、summary 完整、pairing 有效、无 leakage/fallback，且 PBS exit 为 0。失败发生在科学 gates：h256 没有提高 absolute ranking，absolute replacement gates 仍失败。

因此下一 cell 应是独立的 context-support density contrast，保持以下内容不变：

- h256 student、1500 updates、batch/loss/AdamW/lr；
- initialization/training/evaluation seeds 与 held-out 16 blocks；
- 同一 teacher target、action bank、candidate contract 与 causality gate；
- 同一 predictor-only latency boundary 和 thresholds。

只允许改变一个预先冻结、可审计的 context/action-query support density factor。不得根据本轮结果增加 steps、修改 seeds、放宽 gates、同时调整 capacity/query mixture，或直接跳到 CEM/closed-loop。

## 产物

- Summary：`artifacts/24456882.pbs101/capacity_contrast_summary.json`
- Job status/log：`artifacts/24456882.pbs101/job_status.txt`、`artifacts/24456882.pbs101/job.log`
- GPU telemetry：`artifacts/24456882.pbs101/gpu_info.csv`、`gpu_usage.csv`、`gpu_telemetry.jsonl`
- Exit status：`artifacts/24456882.pbs101/final_exit_status.txt`、`runner_exit_status.txt`
- 未取回：`capacity_h256_step*.pt` checkpoints、`plan_targets.pkl`

