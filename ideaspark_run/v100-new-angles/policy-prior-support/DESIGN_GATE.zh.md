# TD-MPC2 policy-prior support：最小 design gate

日期：2026-09-13。结论为 conditional_prior_go，不是 novelty 认证；若实际 config 不匹配或 scorer/actor RNG 无法配对，则 identifiability_no_go、GPU=0。

## 选择的主 observable

选择 official softmax-weighted update value，不选 max-value：24 个 policy slots 与 488 个 random slots 的样本数不对称，max 很容易产生极值假象。

每个 fresh reset state 固定同一 latent、候选随机数和 score RNG，建立三池：FP-policy24+common-random488、W4-policy24+同一488、random24+同一488。random24 必须是预先固定的同分布替代 slots。三池均由 FP dynamics/reward/Q 评分；terminal policy 与 two-Q random pair 每池从同一 score RNG 状态开始。取 official top-64 和 source-labelled softmax weights，计算一次内部 update mean μ。定义

G_A = J_FP(μ_A) - J_FP(μ_random)，其中 J_FP 用相同三-step FP scorer 复评 μ；主量为 Δ_support = G_W4 - G_FP，按 8 个 state 配对报告。Δ_support<0 才表示量化后 policy-slot 对 planner update 的相对贡献下降。另保存 policy-slot elite count 与 softmax weight mass，作为 support mediation receipt；如果 J 下降而 slot inclusion/weight mass 没有变化，只能标为 actor-value drift/inconclusive，不能称 support 损害。

这与旧 score/rank fidelity 或同候选 CEM update 不同：干预只改变 24 个 policy-generated candidates，488 个 random candidates、FP scorer 和 update 规则保持相同，并以 random24 替代作同 slot 对照。若既有结果已经测过同一 source-labelled μ contribution，则本案应 identifiability_no_go；当前短名单不足以认证不重叠。

## 必须先满足的条件

- pinned checkpoint 的实际 num_pi_trajs=24、num_samples=512、num_elites=64、horizon=3、iterations=6；runner 明确只执行第 1 次 update 后停止。这是对官方 six-iteration config 的有界截断，不得把 iterations 改写成 1，也不能把结果外推到完整六轮 planner。
- FP 与 W4 的 24 条 policy proposal 使用同一 policy sampling noise；TD-MPC2 WorldModel.pi() 内部会重新采样 Gaussian noise。不能只复用 488 条 random noise。
- candidate slot order、top-k tie 处理、clamp/squash、terminal model.pi noise 和 Q-member pair 全部固定并留 receipt；三池每次 score 后 actor 立即恢复 FP。复评每个 μ 时使用专用 μ-only scorer 的 shallow-copy config，将该 proxy 的 num_samples 设为 1，断言 scorer 输出为 [1,1]；原 model.cfg 与 512 候选 pool 保持不变。不能以 episodic=false 为 batch=1 的充分条件；三臂从同一 CUDA RNG state 开始，不能把随机 bootstrap 波动当作 pool 差异。
- 只在 proposal generation 期间量化 _pi 的全部 Linear；dynamics、reward、Q、encoder、terminal scoring 始终 FP。一次 update 后停止，不运行闭环。

预注册 binding guard：对每个 state，要求 G_FP>1e-4 且 FP policy-slot weighted mass > random24 mass+1e-4；两者同时满足才算有效 state。少于 4/8 个有效 state 时记为 inconclusive_binding，不解释 W4。对有效 state，若至少 4 个同时满足 G_W4<=0.9*G_FP 且 W4 policy mass<=0.9*FP mass，则给 bounded scope-limited preliminary damage；否则为 mechanism_no_go。1e-4 是固定的非零/数值稳定 guard，不是显著性阈值，也不构成因果 mediation 证明。

## 主要 false positives 与边界

普通 policy action drift、tanh/clamp saturation、top-k ties、terminal scorer 随机性或 24-vs-488 的极值差异都可能伪造 support effect；因此必须使用同 slot random24、source-labelled mass、paired RNG 和 update-value 复评。FP learned Q/reward 仍只是 model score，J_FP(μ) 不等于真实 return。官方 source 中 policy trajectories 先进入 candidate pool，再经 top-k 与 exponential weighting 更新 mean/std；最终 action 还可能经过 gumbel selection，因此本 screen 的 μ value 是 planner 的内部一步 update diagnostic，不是执行动作或 deployment result。参考 [TD-MPC2 planner source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/tdmpc2.py) 与 [WorldModel policy source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/e9f59321933cbc8e11a002b842adc7d4ffae8ff1/tdmpc2/common/world_model.py)。
