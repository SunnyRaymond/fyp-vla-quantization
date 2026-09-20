# CEM-Residual JEPA — Wall Stage A

本实验检验：对一个真实 CEM candidate pool，先用 frozen DINO-WM teacher 在该轮 `mu` 上做一次 base rollout，再由轻量 causal student 从 `base latent + epsilon + sigma` 预测 300 个 candidate 的 dense latent residual，能否保持 teacher objective ranking，并在计入 base rollout 后更快。

## Frozen data split

- 资产：既有 FP32 Wall workload，8 个完整 episodes、26 个真实 CEM pools；每 pool 为 `300 × H5 × action_dim10`，并使用对应 `pools.npz` 的 exact `mu_before/sigma_before`。
- Train：`development:000..004`；held-out：`development:005..007`。以完整 episode 分割，pool 与 candidate 都不会跨 split。
- 实验单位是 held-out pool；300 candidates 是同一 CEM round 内的重复测量，不是 300 个独立样本。

## Arms

- `mean_only`：exact base trajectory broadcast 到全部 candidates。
- `residual`：base 加 trained causal residual。
- `shuffled`：在 pool 内打乱 epsilon 与 candidate 的对应关系，作为 negative control。

三臂共享同一个 context、goal、candidate tensor、mu、sigma 与 FP32 teacher。

## Gates

一次 frozen Stage A 的 GO 必须同时满足：finite、future-action leakage `<=1e-6`、held-out median Spearman `>=0.95`、median top-30 overlap `>=0.80`、residual 相对 mean-only/shuffled 的 median Spearman 增量各 `>=0.05`、top-30 增量各 `>=0.10`，并且 `one base teacher + residual batch` 相对 `300-candidate teacher` 的 synchronized predictor-boundary latency reduction `>=20%`。

若实验有效但 gate 失败，则记录 scientific NO-GO，不追加 steps、不换 split、不放宽门槛。只有 Stage A GO 才另行冻结 chained multi-round CEM；本阶段不运行 environment、MPC 或 closed loop，也不支持 task-success、full-planner speedup、native low-bit deployment 或 LeWM transfer claim。
