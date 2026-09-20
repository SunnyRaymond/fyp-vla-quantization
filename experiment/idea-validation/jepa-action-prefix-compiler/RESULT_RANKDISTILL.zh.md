# DINO-WM PushT Planner-Aware Rank Distillation 实验结果

## 结论

**按照冻结方案判定为 NO-GO。** Planner-aware listwise distillation 相比 latent-MSE control 稳定改善了 candidate ranking，但未达到冻结的绝对 planner-fidelity 阈值，而且 native-latent fidelity 的退化超过 non-inferiority 上限。因此不得进入 CEM integration，也不得针对本次结果进行事后调参。

## 运行信息

- PBS job: `24064105.pbs101`
- 计算资源：单个 compute node，NVIDIA A100-SXM4-40GB
- 终止状态：runner exit `0`、final exit `0`，summary 存在
- 实际 walltime：`00:05:57`
- 冻结协议：`RANKDISTILL_FREEZE.json`
- 结果来源：`artifacts/24064105.pbs101/rank_distill_summary.json`

作业正常完成。`overall=FAIL` 表示科学判据未通过，不是运行故障。

## 冻结判据结果

| 判据 | 冻结要求 | 实测结果 | 判定 |
|---|---:|---:|---|
| Capacity：control latent last-10/first | `<= 0.8` | `0.2896` | PASS |
| Capacity：treatment latent last-10/first | `<= 0.8` | `0.3254` | PASS |
| Future-action leakage | max-abs `<= 1e-6` | 两个 arms 均为 `0.0`，`k=1..4` | PASS |
| Treatment 的 median Spearman | `>= 0.99` | `0.9276` | **FAIL** |
| Treatment 的最弱 block Spearman | `>= 0.95` | `0.8172` | **FAIL** |
| Treatment 的 median top-30 overlap | `>= 0.95` | `0.6167` | **FAIL** |
| Treatment 的最弱 block top-30 | `>= 0.80` | `0.4333` | **FAIL** |
| Paired Spearman 差值的 median | `>= +0.05` | `+0.0621` | PASS |
| Paired top-30 差值的 median | `>= +0.10` | `+0.1500` | PASS |
| 正向 paired blocks | 两项指标均至少 `3/4` | `4/4`、`4/4` | PASS |
| Latent non-inferiority | treatment/control mean relative-MSE `<= 1.25` | `1.4228` | **FAIL** |
| Treatment predictor 延迟降幅 | `>= 20%` | `99.7179%` | PASS |

冻结的 overall gate：**FAIL**。

## Paired held-out blocks 结果

比较单位是一个 `anchor × held-out action seed` block。每个 block 内的 300 个 candidates 是嵌套测量，不是 300 个独立重复。

| Seed | Anchor | Control Spearman | RankDistill Spearman | 差值 | Control top-30 | RankDistill top-30 | 差值 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20262920 | 0 | 0.7723 | 0.8730 | +0.1007 | 0.3333 | 0.5000 | +0.1667 |
| 20262920 | 1 | 0.9588 | 0.9822 | +0.0234 | 0.6667 | 0.7333 | +0.0667 |
| 20262921 | 0 | 0.6873 | 0.8172 | +0.1299 | 0.2000 | 0.4333 | +0.2333 |
| 20262921 | 1 | 0.9703 | 0.9846 | +0.0143 | 0.7333 | 0.8667 | +0.1333 |

Treatment 在四个 blocks 上都同时改善了两项 ranking 指标。改善主要来自 anchor 0；anchor 1 的 control 原本已经较强。这支持一个机制层结论：planner-aware supervision 确实把 student 推向了更正确的排序方向。但最弱 treatment block 仍远低于 planner-safety 下限。

## Latent fidelity 结果

五个 horizons 上的 mean relative MSE：

- control: `0.01936`
- RankDistill: `0.02754`
- treatment/control ratio: `1.4228`

这说明 RankDistill 通过牺牲 latent fidelity 换取 ranking 改善，并超过了冻结的 `1.25` non-inferiority 上限。Treatment 的 per-horizon relative MSE 为 `[0.03141, 0.01674, 0.01875, 0.02900, 0.04182]`，terminal cosine 为 `0.97872`。

## Predictor latency 与 GPU 执行情况

在 cached native observation latents、batch `300`、horizon `5` 的边界下：

- frozen teacher median: `3501.63 ms`
- RankDistill student median: `9.879 ms`
- predictor-only speedup: `354.45x`
- predictor-only reduction: `99.7179%`

该计时不包含 `encode_obs`、CEM、environment interaction 或 closed-loop execution，不能解释为 full-plan speedup。

allocation 全程记录了 GPU telemetry。训练阶段采样到的 GPU utilization 通常为 `97–100%`，显存约 `3335 MiB`；evaluation/timing 阶段显存最高约 `25111 MiB`。未记录到 OOM、NaN 或 fallback。

## 结果说明

本次冻结实验否定了该 idea 的强版本：仅加入一个 listwise KL 项，不足以把 post-hoc action-prefix compiler 变成 planner-safe replacement。但它建立了一个较窄、却有用的结果：planner-aware term 能在 held-out candidates 上稳定改善排序，而单独使用 latent MSE 会留下 decision-level mismatch。

因此，这是一条有价值的方向性证据，而不是已经验证成功的 recipe。任何后续 formulation 都应预注册为新实验，不能包装成本次冻结实验的继续调参。

## 可宣称范围

可以支持：

> 在两个固定的 official DINO-WM PushT anchors 上，相比 paired latent-MSE control，planner-aware listwise distillation 稳定改善了 held-out action-slate ranking；但它没有通过冻结的 absolute-fidelity 与 latent-non-inferiority gates。

不能支持：

- planner-safe substitution;
- CEM or closed-loop success;
- full planner speedup;
- unseen-goal or unseen-observation generalization;
- LeWM transfer;
- observation-prefix composition;
- an all-JEPA acceleration framework.

## 证据文件

- `artifacts/24064105.pbs101/rank_distill_summary.json`
- `artifacts/24064105.pbs101/job_status.txt`
- `artifacts/24064105.pbs101/job.log`
- `artifacts/24064105.pbs101/gpu_info.csv`
- `artifacts/24064105.pbs101/gpu_usage.csv`
- `artifacts/24064105.pbs101/gpu_telemetry.jsonl`
