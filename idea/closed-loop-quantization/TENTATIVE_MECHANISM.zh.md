# Historical tentative non-VQ mechanism（STRC；已 rejected_attempt，供追溯）

## Signed Temporal Residual Calibration（STRC）

暂定载体为可执行的 action-conditioned WM/WAM（首选已有 DINO-WM Wall 的 closed-loop planner；VLA/WAM 作为后续 transfer），目标是固定低比特部署的 joint calibration。核心量不是单 site importance 或单步 consequence，而是量化残差在闭环中的**有符号时间累积**：对同一初始状态配对 FP/Q rollout，记录 action residual `r_t` 与可观测 state/transition residual `e_{t+ell}`，对每个完整 joint bit map 计算去均值/固定尺度后的 signed accumulation，而不把相关量冒充因果 kernel。

`D_H(q)=E[||sum_(ell=0)^(H-1) gamma^ell W e_(t+ell)^q||_2^2]`，其中 `e` 由 same-state simulator replay 的 paired FP/Q action residual 驱动，`W` 是预先冻结的 state-coordinate whitening matrix，`gamma`、`H` 与 state representation 在 CAL 前冻结。其 energy-matched phase-shuffle control 打乱 `e` 的时间顺序并保持每个 lag 的边际能量；signed sum 因而保留 persistence/cancellation 信息，而非把 `sum ||e||^2` 当作时序机制。该 estimand 仍标为 controlled downstream association，不宣称由 `E[e_(t+ell)r_t^T]` 直接识别因果 transfer。它不同于 DA-PTQ 的 virtual-Jacobian/motion magnitude、QuantWAMs 的 reachable-state denoising schedule，或 OTC 的一步 `C_r`。

首版只保留一个可归因 endpoint：从共同 all-W4 map 出发，每次对一个**joint block/group**做明确的 W4→W8 conditional swap，重新测 `D_H(q)`，在固定 bytes/latency 预算下以当前完整 map 的 conditional reduction 选下一 swap；不是把独立 single-site scores 相加。必须与 LocalMSE、DA-PTQ-style motion score、QuantWAMs-style state schedule、static on-policy recalibration、energy-matched phase-shuffle 和 raw-action/no-memory controls 在 matched bytes/latency 下比较。删除额外 delta-action accumulator，因为它会复现通用 observer/error-feedback controller，无法与 Feedback World Model 分开归因。

## 已知边界

- `DA-PTQ`（arxiv:2604.11572）已用 virtual Jacobian、trajectory motion error、gradient layer ranking；TRFC 不能只说“trajectory drift-aware”。
- `QuantWAMs`（arxiv:2607.28405）已用 FP/Q reachable-state replay 的 distribution profile 与 fixed-intervention denoising schedule repair；TRFC 必须依赖 lagged residual-transfer/phase test，而不是重命名 state audit。
- `Feedback World Model`（arxiv:2605.15705）已用真实 transition residual 更新 latent observer、做 action-aware guidance；TRFC 若保留 B，必须证明它是量化残差的固定 calibration-derived delta compensator，并将通用 observer 作为强 baseline，不能声称首次闭环 feedback。
- `Calibrate Where You Deploy` 的作者仓库已报告 on-policy recalibration 负结果，但其公开 audit 仍有 task/seed/control 分布限制；TRFC 不能把 on-policy state collection 作为唯一贡献，也不能据此宣称普遍不可能。
- `MARR`（connector record `semanticscholar:c45e8f717cdf07690560c5eb96dbac4305236755`）提供 module-adaptive residual reconstruction + PID feedback，但对象是重建/Hessian bias，不是 embodied closed-loop transition residual；需在 collision 中显式比较。

## 可证伪核心预测

在相同低比特、相同 bytes/latency 和相同 calibration budget 下，若 signed temporal accumulation 是 load-bearing，STRC 的 held-out closed-loop accumulated state drift / task success 应优于局部重建、DA-PTQ motion score 与 phase-shuffled score；若 phase shuffle 保留同等收益、conditional swap 的方向跨 fit seed 不稳定，或 raw-action/no-memory control 同样有效，则 temporal mechanism 判为 no-go。主结果不以 `D_H` 自身变小作为充分证据，而以 downstream drift/success 的方向和 phase intervention 的作用为判据。

本文件仅为预审提示，不是 pipeline 的 canonical output，也没有运行模型、实验或集群任务。
