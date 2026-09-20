# CEM-Residual JEPA：DINO-WM PushT Stage A 预注册协议

## 问题与边界

本实验只检验一个 predictor-level 机制问题：在同一个 CEM proposal block 内，先用 frozen DINO-WM teacher 对 distribution mean `mu` 做一次 rollout，再用轻量 causal student 预测每个 candidate 相对该 base trajectory 的 latent residual，能否比直接复用 mean trajectory 更准确地保持 candidate ranking，并在计入一次 base rollout 后减少 predictor wall time。

本阶段不运行完整 multi-round CEM、environment interaction 或 closed loop；不支持 LeWM transfer、task success 或 full-planner speedup claim。

## 实验单位与阻断

- 独立实验单位：一个 held-out context × 一个 fresh candidate seed，即一个 block。
- 固定 8 个 held-out contexts × 2 个 seeds，共 16 个 paired blocks。
- 每个 block 固定 300 candidates、`H=5`、`topk=30`。block 内 candidates 是重复测量，不计作独立 replicate。
- train/held-out 按完整 trajectory/episode 分离，禁止 held-out context、action、seed 或 target 进入训练。

## 模型与处理

对每个 block，candidate action 为

`a_i = mu + sigma * epsilon_i`。

Frozen teacher 产生 base trajectory `T(z_t, mu)` 与 candidate targets `T(z_t, a_i)`。Residual student 输入 base future native latents、causal `epsilon_i` prefix 与 `sigma`，输出 dense residual，并形成

`z_hat_i = T(z_t, mu) + R(T(z_t, mu), epsilon_i, sigma)`。

训练 query 预先固定为两类各 50%：

1. initial CEM：`mu=0, sigma=1`；
2. one-step refined CEM：从 64 个标准正态 proposals 由 frozen teacher objective 选择 8 elites，按其逐坐标 mean/std（std floor 0.05）得到 `mu/sigma`。

Student 不接收 goal。Goal 只用于生成 one-step refined distribution 和 held-out teacher objective。

## 对照

- `mean_only_base_repeat`：把 exact teacher base trajectory broadcast 给全部 candidates，等价于 residual 恒为零。
- `trained_residual`：训练后的 causal residual student。
- `shuffled_epsilon_negative_control`：在 block 内固定置换 residual student 的 epsilon-to-candidate pairing；用于排除仅凭 base 或 population statistics 获益。

所有处理使用同一 context、goal、`mu/sigma` 和 candidate tensor。

## 指标

Primary（逐 block 后跨 16 blocks 汇总）：

- teacher objective Spearman；
- teacher/student top-30 set overlap；
- teacher-scored selected-best regret：surrogate top-1 的 teacher cost 减去 teacher top-1 cost，并以该 block teacher cost std 归一化。

Secondary：逐 horizon relative latent MSE/cosine、future-action leakage、训练收敛。

Latency boundary：cached native context latent 开始，比较 `300 × teacher rollout` 与 `1 × teacher base rollout + residual batch(300)`；包含 base、student 和 CUDA synchronization，不包含 `encode_obs`、objective、CEM orchestration、environment 或 closed loop。

## 冻结 gates

Mechanism gate（与 `FREEZE.json` 一致）需要全部成立：

- finite outputs/training，future-action leakage max abs `<=1e-6`；
- `trained_residual - mean_only_base_repeat` 的 median ΔSpearman `>=0.05`、median Δtop-30 `>=0.10`；
- `trained_residual - shuffled_epsilon_negative_control` 的 median ΔSpearman `>=0.05`、median Δtop-30 `>=0.10`；
- finite 与 causality 必须通过。Latency 只作 boundary diagnostic，不进入本 gate。

Near-lossless replacement 是更严格、独立的标签：strict median/minimum Spearman `>0.99/0.95` 且 strict median/minimum top-30 `>0.95/0.80`。Mechanism PASS 不自动等于 replacement GO。

## 顺序停止

只运行本 frozen Stage A 一次。若 capacity/integrity 失败，结果无效并仅修 implementation bug；若实验有效但 mechanism gate 失败，则记录 scientific NO-GO，不增 steps、不换 seeds、不放宽 thresholds，也不进入 chained CEM 或 closed loop。只有 mechanism PASS 才另行冻结 Stage B。
