# DINO-WM PushT：Context-support density 对比结果

## 一句话结论

这次重跑是一个有效的 frozen cell，结论是：把同一批 32 条训练 episode 的 context support 从 4 个/episode 扩展到 8 个/episode，确实让 student 更容易拟合 teacher latent，但没有把 CEM candidate ranking 提升到预设门槛。因此本 cell 的三个独立决策都是：

| 决策层 | 结果 |
|---|---|
| density effect | **FAIL** |
| full replacement | **NO-GO** |
| density-supported replacement | **NO-GO** |

这里的 `FAIL` 是“改善幅度没有达到预先冻结的 effect gate”，不是实现失败，也不是说增加 context 完全没有任何收益。

## 实验身份与边界

- PBS job：`24477396.pbs101`
- PBS 状态：`completed`，job final exit code `0`
- runner exit status：`0`；`NO-GO` 是 summary 内的科学决策，不是进程错误码
- student：`NativeDinoPrefixStudent hidden_dim=256`
- 对照：authoritative `h256-Dlow`；实验：`h256-Dhigh`
- 唯一变化：同一批 32 个 train episodes 内，context support 从 128 条扩展到 256 条
- 训练：1500 updates，batch 32，AdamW，learning rate `3e-4`
- query mixture：logged / Gaussian planner init / one-step CEM resample = `50/25/25`
- held-out：2 个 evaluation seeds × 8 contexts = **16 paired blocks**，每个 block 300 candidates，top-k=30

本 cell 只覆盖 predictor-level 的 `cached observation latent + normalized action prefix → predicted latent`。不覆盖 observation encoder、query density、CEM execution、environment interaction、closed-loop control、LeWM 或 Fast-LeWM。

## Integrity、收敛与运行证据

Manifest/integrity gate **PASS**：

- legacy 128 条 context 保持原顺序并逐字段保留；新 128 条只追加到 ordinal `128..255`
- 32 个训练 episode、validation split 和 held-out split 均未改变
- 新 context 不复用旧 start，新增 trajectory 数为 `0`
- 每个 batch 固定保留 16 个 Dlow rows，再从新 support rows 中采样 16 个；schedule 的 low prefix 保持不变
- paired candidate contract 完整：16 blocks、300 candidates、top-k=30、CPU `torch.Generator`、固定 held-out seeds
- student 不是 teacher，未进入 `encode_obs`，没有 source encoder / teacher 参数共享或 fallback
- 所有 snapshot outputs/state dicts finite，shape/dtype/device match，future-action leakage 最大绝对值为 `0`

训练收敛 gate 也 **PASS**：

- latent MSE：`0.416612 → 0.091970`
- last-10 / first MSE ratio：`0.2211`，冻结上限为 `0.8`

GPU telemetry 正常：单张 NVIDIA A100-SXM4-40GB；训练约 5.99 GiB，evaluation 峰值约 25.8 GiB；无 OOM、NaN 或 silent fallback。小型运行产物已取回，未取回 checkpoints 和 `plan_targets.pkl`。

## Density effect：16 个 paired blocks

下表均为 `h256-Dhigh − authoritative h256-Dlow`。冻结 effect gate 要求最终 snapshot 同时达到 Spearman median `≥0.05`、top-30 median `≥0.10`，并且两项各自至少 12/16 个 block 为正。

| snapshot | Spearman Δ median | Spearman 正向 | top-30 Δ median | top-30 正向 | mean relative-MSE Δ median |
|---|---:|---:|---:|---:|---:|
| step 500 | +0.0337 | 16/16 | +0.0333 | 10/16 | -0.0040 |
| step 1000 | +0.0571 | 16/16 | +0.0667 | 14/16 | -0.0039 |
| step 1500（冻结决策） | **+0.0499** | **16/16** | **+0.0500** | **11/16** | **-0.0062** |

最终 `density_effect=FAIL` 的原因很具体：

1. Spearman median `0.04991`，略低于冻结的 `0.05` 门槛；
2. top-30 median `0.0500`，低于 `0.10`；
3. top-30 正向 block 为 `11/16`，低于 `12/16`。

也就是说，Dhigh 在 16/16 个 block 上都提高了 rank correlation，但提高得太小；对 planner 真正关心的 top-30 candidate set，改善不稳定且不足。

## Absolute fidelity 与 latent fit

最终 Dhigh 的 absolute ranking 仍未达到 replacement 要求：

| 指标 | Dhigh step 1500 | 冻结门槛 | 状态 |
|---|---:|---:|---|
| Spearman median | 0.9215 | ≥0.99 | FAIL |
| Spearman minimum | 0.8308 | ≥0.95 | FAIL |
| top-30 overlap median | 0.6833 | ≥0.95 | FAIL |
| top-30 overlap minimum | 0.5000 | ≥0.80 | FAIL |

另一方面，logged-action teacher-relative MSE 明显改善：

- Dhigh：`0.03833`
- Dlow：`0.07664`
- Dhigh/Dlow ratio：`0.5001`，冻结上限 `1.25`，因此 **PASS**

这组结果把问题定位得更清楚：更多 context support 帮助 student 降低平均 latent regression error，但这种平均误差改善没有可靠地转化成 candidate ranking/top-k fidelity。当前瓶颈更像是 action-conditioned predictor 对 action-conditioned ranking 几何的表达/训练目标，而不是“同一个 observation 需要更多 context 行”本身。

## Causality 与 latency

- prefix causality：**PASS**；所有 unchanged-prefix probes 的最大绝对变化为 `0.0`，阈值 `1e-6`
- predictor-only teacher median：`3501.03 ms`
- predictor-only student median：`16.57 ms`
- teacher→student median reduction：`99.5267%`，冻结门槛 `≥20%`，**PASS**
- Dhigh/Dlow student latency ratio：`1.0048`；context support 加倍几乎没有改变 predictor-only latency

latency 的计时边界是 teacher 和 student 的 predictor forward，包含 CUDA synchronization，不包含 observation encoder、CEM loop 或 environment stepping。因此这里支持的是 predictor-level speed claim，不是端到端 control-frequency claim。

## 研究解释与 claim boundary

本 cell 支持的最强表述是：

> 在 DINO-WM PushT、h256、固定 32 个 episode、固定训练预算和固定 action-query mixture 下，把每个 episode 的 context support 从 4 增加到 8，会稳定改善 latent regression（Dhigh/Dlow logged-MSE ratio≈0.50），并在 16 个 held-out blocks 上逐块提高 Spearman；但没有达到预注册的 ranking-effect gate，也没有达到 absolute replacement fidelity gate。

不能据此声称：

- context density 在所有 JEPA-style WM 上无效；
- observation encoder 或 query density 已被排除；
- 更长训练、更大模型或不同数据量一定无效；
- CEM closed-loop success、端到端 wall-clock 或 LeWM/Fast-LeWM 已得到验证。

在当前已完成的对照中，optimization length 和 hidden width 也没有通过相应 gate；结合本 cell，下一步更值得优先验证的是 action-conditioned predictor 的 query-support / ranking-aware 学习信号，而不是继续单纯增加 same-episode context rows。

## 产物

- 中文结果文档：`RESULT_CONTEXT_DENSITY.zh.md`
- 小型本地状态目录：`artifacts/24477396.pbs101/`
- `context_density_summary.json`
- `job.log`
- `job_status.txt`、`runner_exit_status.txt`、`final_exit_status.txt`
- `gpu_info.csv`、`gpu_usage.csv`、`gpu_telemetry.jsonl`
- 未取回：`context_density_h256_step0500.pt`、`step1000.pt`、`step1500.pt`、`plan_targets.pkl`
