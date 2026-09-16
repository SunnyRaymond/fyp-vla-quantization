# Action-gradient geometry 独立审查

日期：2026-09-13。范围为只读核对冻结 PROTOCOL、旧 smoke_runner/screen_runner 和本地 official planning/gd.py、objectives.py；未修改 protocol/实现，未连接集群、未运行实验。

## 判定

结论：conditional go（机制问题可由最小 screen 区分），但当前旧 runner 不能直接执行该 protocol；需先完成下列执行合同修订。6 episodes × 64 candidates × 4 fixed anchors 足以在每个 episode 内分别测 global objective fidelity 与 local gradient mismatch，不能支持显著性、真实 success 或 novelty 结论。

## objective/action graph

official planning/gd.py L71–106 明确存在 action → wm.rollout → objective_fn → backward 的可微图。它同时包含 optimizer update、action_noise 和 evaluator；因此只应复用图，不应直接调用 GDPlanner.plan 做本 screen。

official planning/objectives.py L6–32 的 mode=last objective 返回每个 candidate 的 terminal visual MSE 加 alpha × terminal proprio MSE。旧 smoke_runner.py L436–439 使用 alpha=1、base=2、mode=last，L442–467 复用了相同 graph。protocol 的“visual terminal objective”若表示 visual-only，与实际 official objective 不一致；最小修订是明确写成 alpha=1 的 visual+proprio terminal objective（并记录配置），或明确将 alpha 改为 0，不能两者混用。

旧 smoke_runner.py L102–105 全局调用 torch.set_grad_enabled(False)，L464 还在 torch.no_grad() 中评分。这条路径只能用于 global scores，不能用于 gradient。gradient scorer 必须在 torch.enable_grad() 中运行，冻结所有 model parameters，每个 anchor 从 detached action 建立新的 leaf 并 requires_grad_(True)，对 per-candidate loss 使用 sum（或 official GD 的 mean × batch-size），再用 autograd.grad；不能对 latent 或 batch 做平均后再求一个共享梯度。

## 6 × 64 × 4 是否可识别

同一 episode 的同一 64 个 action pool 在 FP32/W4 两臂下计算 scores，可给出逐 episode Spearman（average rank、exact ties）和 centered-score NRMSE，代表 global binding。anchors 固定为 pool index 0–3 且在看分数前确定；用 20 维 flatten 后的 unit gradient、同一 0.10 negative-gradient step，并由 FP32 objective 重评分，可给出 local cosine 与 action-step improvement。两组量的来源和 selection 互不污染，4/6 的联合 gate 能区分“全局保持但局部破坏”。4 anchors 不是四个独立 episode，64 也不是统计独立样本；结果只能按 frozen screening gate 解读。

若旧代码的 candidates[0] = 0（smoke_runner.py L677–681）被复用，则 anchor 0 不是 standard-normal pool，破坏 protocol 的 64 standard-normal 及 anchor 公平性；必须删除该覆盖或在 protocol 中显式冻结为特殊候选。旧代码还固定 HORIZON=5、FRAMESKIP=5（L38、L42），而 protocol 是 H=2、actiondim=10；必须从 runtime dset.action_dim × frameskip 派生并 assert shape [64,2,10]，不能沿用 H=5 常量。

## 执行前必须修正的阻断项

1. 旧 smoke_runner.py 的 _make_explicit_targets L208–284 会调用 25-step env.rollout（L239–246），而 protocol 明确不跑 environment rollout；不得直接复用该 target-preparation helper。若必须生成 obs_0/obs_g，应只保留协议允许的 target preparation，并记录其来源。

2. predictor-only W4 要求 encoder 保持 FP32。旧量化组/全量 helper 若把 encoder 与 predictor 一起量化，会改变解释；新 scorer 必须只选择 predictor groups，并在 arm 前后检查 encoder hash/权重未变，FP/Q 之间 restore 同一 snapshot。

3. FD gate 需冻结符号：令 g 为该 anchor 的 objective gradient，u=g/||g||；中心差分应为 [L_FP(a+eps·u)−L_FP(a−eps·u)]/(2 eps)，比较 g·u=||g||，eps=0.005。negative step 应为 a−0.10u，improvement 定义为 L_FP(a)−L_FP(a−0.10u)，正值才表示 objective 下降。FD 与 step 若使用相反符号，会错误地把实现失败解释成坏 local geometry。

4. global metrics 必须逐 episode 在 64 scores 上计算，不能跨 episode pool；NRMSE 分母为该 episode FP score 的 population std（ddof=0，std ≤ 1e−8 为 no_binding）。明确 average-tie rank 和阈值比较的数值 dtype；FD 的 1e−4 项不应被“仅 1e−12”覆盖。

5. 118..123 应命名为 validation dataset_index，不要称为 state index。旧 screen_runner.py _new_target L298–322 只从 dset[index] 取 layout/env_info，初始/目标 state 实际由 env_seed 产生；因此必须保存 dset revision/source trajectory ID 或 fingerprint、split、length（至少 124），并独立记录 env_seed 与 candidate_seed。当前旧 manifest 不能证明 118..123 与 0..117 的底层 episode 不重叠；若 runtime mapping/registry 无法证明，engineering/independence gate 应 fail closed，结论只能 inconclusive_binding，且不能读取 reserved 84..95 来补证。

6. runtime import 的 planning/objectives.py 来自 source，而 preflight gd.py/objectives.py 是本地 snapshot；计算前必须记录并 hash 实际 import 的 source、checkpoint、helper，确认 objective 配置确为冻结版本，避免仅凭 snapshot 声称 official graph。

## 最小验收语义

每 episode 保存同一 candidate matrix、FP/Q 的 64 原始 scores、4 anchors 的 FP/Q gradients 与 norms/cosines、FP/Q step scores、第一 anchor 的 FP/Q FD 原值，以及 finite、weight-restore、source/hash、allocation 证据。global pass 且 local bad 才能计入 conditional_signal；global 或 local 不可识别计 inconclusive_binding；可识别但联合不足 4/6 才是 mechanism_no_go。负结果不应标 novelty_no_go，也不应扩成环境 rollout 或完整 planner 验证。
