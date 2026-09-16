# TD-MPC2 ensemble disagreement under PTQ：可行性初筛

**原 framing：`structural_no_go`；改成离线 Q-spread audit 后：`go-to-minimal-protocol`。** 未实验，不能下 empirical no-go。

## 结构与先例

官方 TD-MPC2 有 5 个 Q-functions：target 对随机两个 EMA Q 取 `min`，planner terminal value 对随机两个当前 Q 取 `avg`，没有使用 variance/disagreement。[paper](https://arxiv.org/html/2310.16828v2) [source](https://raw.githubusercontent.com/nicklashansen/tdmpc2/main/tdmpc2/tdmpc2.py) 因此不能声称 planner 使用 disagreement，或把 two-Q `min` 当 epistemic uncertainty。可救回的问题是：W4 是否污染附加的 Q-spread diagnostic，以及 Q-only 与 latent-model-only 扰动是否不同；不推断 joint PTQ。

`Q(return_type='all')` 可暴露成员输出。[world_model.py](https://raw.githubusercontent.com/nicklashansen/tdmpc2/main/tdmpc2/common/world_model.py) 分析须分开记录 target pessimism 与 planner scoring，并冻结 Q-pair。SPEQ 是随机 activation precision 的 self-distillation，不是 RL critic ensemble。[SPEQ](https://ojs.aaai.org/index.php/AAAI/article/view/16839)；QuaRL 研究 RL 的 PTQ/QAT，也没有 Q-spread 诊断。[QuaRL](https://research.google/pubs/quantized-reinforcement-learning-quarl/) 有限检索未找到直接覆盖本 audit 的 primary 先例，支持 conditional novelty，不支持新 ensemble method。

## 资源与最小协议

官方 single-task checkpoint 为 5M parameters；DMC `cartpole-balance-1.pt` 约 31.3MB，单任务推荐 8GB GPU，V100 32GB 可用。[files](https://huggingface.co/nicklashansen/tdmpc2/tree/main/dmcontrol) 只取单文件，不能拉取 14.9GB 全仓库。未来在真实 SLURM/PBS allocation 准备一个 state-based DMC task、冻结 observations/action pool。

三 arm：FP32、`W4-Q-only`（live `_Qs` 五成员）、`W4-latent-only`（encoder/dynamics/reward，Q FP32）；不做 joint。冻结 source/config/hash、eval、scaling、state、action pool、Q-pair；约 8 state×64 sequences，保存 member Q、`spread=std_j(Q_j)`、two-Q min/avg、order、Q error 与 spread change；按 state 等权。

## Gate

- **`go-to-minimal-protocol`**：Q=5、min/avg source path、Q-pair、三 arm allowlist 与小 checkpoint 可复现。
- **`structural_no_go`**：仍声称 planner 使用 disagreement，或无法冻结 Q-pair/聚合路径。
- **`novelty_no_go`**：把 Q-spread、PTQ audit 或 generic critic ensemble 宣称为新算法；本窄诊断暂不判此项。
- **`resource_blocked`**：小 checkpoint、DMC state 依赖或 V100 不可用。
