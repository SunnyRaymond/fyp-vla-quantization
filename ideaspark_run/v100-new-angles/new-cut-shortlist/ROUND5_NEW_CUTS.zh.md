# ROUND5 新切面短名单

日期：2026-09-13。本文只做 C02/C04 prior/source 筛选；已先读 `INDEX.zh.md`，不重复已有 30 cuts，不加载模型、不读取数据、不运行数值、不连接或提交 cluster。novelty 只表示本次 targeted review 的边界。

## 1. TD-MPC2 policy-prior support under PTQ

**状态：`conditional_prior_go`；novelty 未认证。** Hypothesis：official planner 把 24 条 `_pi` policy trajectories 放进 512 条候选的前缀；只量化 `_pi` 可能通过改变这些候选被 top-64 选中的 support，放大到 MPPI 的 `mean/std` 更新，即使 encoder、dynamics、reward、Q 都保持 FP。反事实 observable 是固定初始 latent、policy noise 与 488 条 random candidates，比较 actor-FP/actor-W4，以及把 24 个 policy slots 替换为固定 random trajectories 的 matched pool；保存 support inclusion、elite overlap、更新后的 mean/std 和 first action。最强 prior 是官方 [TD-MPC2 planner source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py) 与 [WorldModel policy source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py)；QuantWM 的 PTQ sensitivity 仍未覆盖这种 support mediation。若 `num_pi_trajs=0`、support removal 与 full pool 无差异，或差异只等于 actor action drift，则直接否决该机制。

## 2. DINO-WM action-token reinjection boundary

**状态：`conditional_prior_go`；novelty 未认证。** Hypothesis：`VWorldModel.rollout` 每次 predictor 后都用 `replace_actions_from_z` 覆盖 action slice（`visual_world_model.py:273–301`）；因此 action-encoder W4 的误差在 action slot 上每步被重置，而 predictor W4 的 latent error 可递归传播。反事实 observable 是固定 action sequence 下 FP predictor + Q action encoder，并加入“每步以 FP action embedding 重插入”的 matched cell；比较 action-token delta、逐步 latent drift 与现有 objective。最强 primary prior 是 [DINO-WM paper](https://arxiv.org/abs/2411.04983) 与 [official source](https://github.com/world-model-2026/dino-wm)，它们证明 action-conditioned latent rollout，但没有 PTQ reset-boundary claim。若实际 `concat_dim/action_dim` 不暴露独立 action slice，或 clean-reinsert 与 Q-reinsert 的 drift 不分离，则 `identifiability_no_go`，不得退化成普通 layer sensitivity。

## 3. DINO-WM online success-mask feedback

**状态：`identifiability_no_go`；不申请 GPU。** Hypothesis：Wall 的 `MPCPlanner` 在每轮真实 env feedback 后以 `successes` 更新 `is_success`，对已成功轨迹执行 `_apply_success_mask`，并停止这些轨迹的后续 replanning（`planning/mpc.py:60–113`）；predictor PTQ 可能改变分支时刻，形成反馈放大。反事实 observable 可记录 FP/Q 各自 branch trace，并用 FP success mask 重放相同 action plan，比较 mask 时刻、被置零/重归一化的 action 与后续差异。最强 prior 是 [DINO-WM planning source](https://github.com/world-model-2026/dino-wm/blob/main/planning/mpc.py)；generic MPC 已有 replanning，但未见该 PTQ branch mediation 的直接 primary claim。若 Wall 配置不启用 `mpc_cem`、无法保存同一 env trace，或 branch-replay 差异完全由 action drift 解释，则机制不可识别，不能把普通 closed-loop success evaluation 改名为新 quantization 方法。

## 决定

保留前两项作为窄的 prior candidates，均需 execution 前冻结 module locus、matched counterfactual 与 FP/no-op/restore evidence；不把 actor support 或 action reinjection 的 positive 预期写成方法收益。第三项因 success branch 与真实环境 outcome、action drift 同时变化，当前 identifiability 不足，明确 `GPU=0`。本轮没有新实验 protocol，也没有声称 exhaustive novelty coverage。

