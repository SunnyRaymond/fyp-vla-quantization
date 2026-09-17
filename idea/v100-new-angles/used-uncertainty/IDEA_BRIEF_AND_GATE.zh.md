# MOPO：量化与 dynamics uncertainty

**判定：`resource_blocked`（机制通过，资产未证实）；非 empirical no-go。** 不下载、不运行。

MOPO 论文明确用 dynamics uncertainty 惩罚 reward，在 uncertainty-penalized MDP 上优化 policy。[paper](https://arxiv.org/abs/2005.13239) 源码设 `num_networks=7、num_elites=5`；`FakeEnv.step` 计算各 member 的 state mean，以 `u=max_j ||mu_j-mean_k(mu_k)||_2` 得 penalty，返回 `reward-penalty_coeff*u`，并送入 synthetic replay 及 actor/critic update。[source](https://raw.githubusercontent.com/tianheyu927/mopo/master/mopo/models/fake_env.py) 这是 rollout/planning penalty，不是 online MPC 的 action-time chooser；若后者是硬要求，判 `structural_no_go`。

## 先例边界

QuaRL 覆盖 RL 的 PTQ/QAT，SPEQ 覆盖 stochastic precision ensemble/self-distillation；均未把 PTQ 对 dynamics ensemble disagreement 的改变接入 rollout penalty。[QuaRL](https://research.google/pubs/quantized-reinforcement-learning-quarl/) [SPEQ](https://ojs.aaai.org/index.php/AAAI/article/view/16839) 未见直接重叠；不能声称新 estimator/method。

## 资源与最小 gate

官方 [repo](https://github.com/tianheyu927/mopo) 列出代码/config 目录；README 要求 D4RL、旧版 MuJoCo，写明 local-only，未发现单一 checkpoint。`model_load_dir` 只是 loader。因此 `<=500MB`、V100 复现和最小输入资产未证实，当前阻断。

若取得可核验 `<=500MB` 单 task checkpoint：冻结 7 members、scaler、state/action batch、seed、penalty coefficient；仅 dynamics members 逐成员 W4-RTN，actor/Q/环境 FP32。保存 FP32/W4 的 member means、`u`、penalized reward、ranking，以 held-out transition error 参照，不重训 policy。资产、源码和真实 compute allocation 全满足才 `go-to-minimal-protocol`；否则 `resource_blocked`。把 MOPO penalty/audit 包装成新算法则 `novelty_no_go`。