# action→gradient geometry：prior gate

**结论：conditional go；最小机制 screen。**

## H / 源码

固定输入、history、H、normalized action 时，predictor-only W4 RTN 可能保留 rollout output MSE/score rank，却改变 GD planner 的 `g=∇_a L(rollout(a),z_goal)`。只有 output/rank 低误差、gradient 方向差，且 Q-gradient 一步更新的 FP32 objective 下降更少，才支持独立机制。

`planning/gd.py:71–106` 对 action 反传 objective，goal latent detached；`visual_world_model.py:273–308` 保留 action graph；`objectives.py:17–55` 是 visual MSE + `alpha`·proprio MSE。RankCal/CEM-Update/FRT/PRR 测 score ordering、elite 更新、residual finite difference、free-H2 recovery，均未测 `∇_a L`，不可包装为换 loss。

## 先例边界

QuantWM (2602.02110) 已在 DINO-WM 覆盖 weight-only PTQ 与 objective-success alignment，却无 fixed-action gradient cosine/sign。2402.05290 研究 architecture/horizon policy-gradient path，非 PTQ；DA-PTQ (2604.11572) 用 physical Jacobian、drift、`∂L/∂W`，非 WM input-action gradient。QuantWAMs 已覆盖 WAM joint video-action Fisher；generic gradient-aware PTQ 不新，本候选限于 DINO-WM GD planner 的 input-action geometry。

## 最小 screen / gate

只比较 FP32 与 predictor-only W4 RTN（dequantized FP32）。冻结 action、无 optimizer/noise，用 `autograd.grad(loss.sum(),actions)` 得 exact `g`；同 pool 记录 latent MSE、score rank、gradient cosine/sign/norm，再比较 FP/Q gradient 一步后的 FP32 objective 下降，阈值事前冻结。

graph 非 finite、W4 output/rank 崩溃、只剩 scalar、加入 gradient-aware quantizer/training，或要求 full closed-loop，均 `inconclusive/no-go`。仅当 fresh probes 同时显示低 output/rank 误差、低 gradient fidelity 且 Q-gradient 一步更差，才记 `conditional_signal`；否则 `novelty_no_go`。