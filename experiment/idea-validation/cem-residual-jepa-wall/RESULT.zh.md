# CEM-Residual JEPA — Wall Stage A 结果

## 决策

PBS job `24464560.pbs101` 正常完成（`Exit_status=0`），但 frozen Stage A 判定为 **NO-GO**。

这个 recipe 的计算机制成立：一次 exact mean rollout 加 residual batch 的 predictor-boundary latency 为 `31.85 ms`，相对 300-candidate frozen teacher 的 `1232.60 ms` 减少 `97.42%`；causality、finite outputs 与 shuffled-epsilon negative control 也正常。可是 planner-relevant fidelity 远低于门槛，因此不能进入 chained CEM 或 closed loop。

## Frozen held-out 结果

Train 使用完整 episodes `development:000..004` 的 18 个真实 CEM pools；held-out 使用 `development:005..007` 的 8 个 pools。每个 pool 有 300 candidates，但统计单位是 pool，不是 candidate。

| Metric | Result | Gate | Decision |
|---|---:|---:|---|
| Residual median Spearman | `0.2802` | `>=0.95` | FAIL |
| Residual median top-30 overlap | `0.2167` | `>=0.80` | FAIL |
| Median ΔSpearman vs mean-only | `+0.2533` | `>=+0.05` | PASS |
| Median ΔSpearman vs shuffled | `+0.2835` | `>=+0.05` | PASS |
| Median Δtop-30 vs mean-only | `+0.1000` | `>=+0.10` | boundary / float-below |
| Median Δtop-30 vs shuffled | `+0.1500` | `>=+0.10` | PASS |
| Future-action leakage max abs | `0` | `<=1e-6` | PASS |
| Predictor-boundary latency reduction | `97.42%` | `>=20%` | PASS |

Absolute ranking gates clearly 失败，所以 `mechanism_pass=false`；top-30 的 floating-point boundary 不影响总判定。

## 解释

Residual conditioning 不是完全无效：在多数 held-out pools，它相对 mean-only 与 shuffled-epsilon 都提高了 Spearman，说明 student 确实读到了 candidate-specific perturbation，而不只是记住 base trajectory 或 population statistics。

但改善没有达到 planner-safe 程度，而且在 `development:005/mpc_00/cem_05` 上几乎失效（Spearman `0.0327`、top-30 `0.0667`）。这说明“围绕 CEM mean 的 latent residual 更容易预测”这个假设，在当前 one-layer、hidden-64、4-pass recipe 下不足以保持 elite ranking；late/refined pool 并没有自然变成一个可由该轻量 residual model 可靠拟合的局部线性邻域。

因此当前证据支持的窄结论是：

> CEM-mean factorization 可以显著降低 predictor compute，并提供真实的 candidate-specific signal，但当前 residual predictor 的 ranking fidelity 不足，不能安全替代 frozen JEPA teacher 参与 CEM update。

本结果不支持 full-planner speedup、closed-loop task success、LeWM transfer 或普遍否定所有 CEM-residual architecture。按照 frozen sequential rule，本 recipe 不追加 steps、不扩大 hidden size、不放宽门槛；若继续，必须提出新的、可区分的 architecture hypothesis，例如显式 action–patch bilinear/Jacobian basis，而不是继续训练同一模型。

## 运行证据

- Summary：`artifacts/24464560.pbs101/summary.json`
- GPU telemetry：`artifacts/24464560.pbs101/gpu_telemetry.csv`
- Job log：`artifacts/24464560.pbs101/job.log`
- Compute host：`x1000c0s3b0n0`
- Walltime：`00:02:31`
- GPU telemetry：5 个 30-second samples，median/max utilization `100%/100%`，sampled peak VRAM `1901 MiB`
- Frozen teacher score replay max absolute error across pools remained below `6e-4`。
