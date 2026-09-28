# Fast-LeWM + PushT 源码接口 Contract

本说明固定在 Fast-LeWorldModel 官方仓库当前 HEAD `de3e9dac539f5bbe6ff1656a2fb00938d62a3c7d`；本地 `upstream/` 保留该 commit 的相关源码。指定 historical commit `492752d96b2a11ec802c55322c46bdc87885da09` 与当前版本在本次检查文件中的差异仅为 `eval.py` 改用 Hydra runtime output directory，以及 `requirements.txt` 增加 MuJoCo/环境依赖。以下为静态源码结论，尚未证明 checkpoint 可反序列化、依赖可导入或评估可运行。

## Checkpoint 与模块路径

- README 的评估示例指向 `Fast-lewm_pusht_object.ckpt`（`upstream/README.md:93-101`）。训练入口以顶层模块名导入 `jepa.JEPA`、`module.*` 和 `utils.*`（`upstream/train.py:17-19`）；`ModelObjectCallBack` 调用 `torch.save(model, path)` 保存完整 Python model object，而非单纯 state dict（`upstream/utils.py:28-57`）。因此 unpickle 时需让该 commit 的 source root 能以相同顶层名称导入，并用允许完整对象的加载模式；仅能读权重张量不等价于加载此 checkpoint。
- Fast-LeWM `eval.py` 把 `eval.ckpt_path` 的文件名去掉 `.ckpt` / `_object.ckpt`，再调用 `swm.policy.AutoCostModel(policy_name, cache_dir=...)`（`upstream/eval.py:108-128, 298-308`）。仓库 `requirements.txt` 固定 `stable-worldmodel==0.0.6` 与 `stable-pretraining==0.1.6`，并固定 Torch/torchvision CUDA 12.8 wheel（`upstream/requirements.txt:1-8, 36-45`）。Fast-LeWM 源码 API 与该包版本的兼容性不能从 requirements 名称推定。项目现有 historical stable-worldmodel source backend 的补查观察到 `swm.policy.AutoCostModel` 不存在；该证据不能归因于 PyPI `stable-worldmodel==0.0.6`，其隔离包尚未完成 CPU 检查。入口需按实际 pinned package API 选择 loader，并记录 adapter 和加载结果。
- 项目现有 historical backend 的补查发现 `WorldModelPolicy` 配置字段为 `cfg`，不应假设有 `policy.config`；`policy.solver` 则是 Fast-LeWM evaluator 自己访问的属性（`upstream/eval.py:321-327, 377-396`）。此为现有 backend 的观察，PyPI 0.0.6 仍待验证。取规划字段可捕获构造时传入的 `PlanConfig`；CEM 类型使用 Hydra config 完整 target `stable_worldmodel.solver.CEMSolver`（`upstream/config/eval/solver/cem.yaml:1`），直接访问 `swm.solver` 前应显式导入该子模块。

## PushT action horizon 与 episode budget

`config/eval/pusht.yaml` 固定 `horizon=1`、`receding_horizon=1`、`action_num_blocks=5`、`action_block_size=5`、`eval_budget=50`、`num_eval=50`、`seed=42`、`goal_offset_steps=25`（`upstream/config/eval/pusht.yaml:20-39`）。`eval.py` 计算 `action_block=5*5=25`，将 `horizon=1`、`action_block=25` 和 `receding_horizon=1` 交给 `swm.PlanConfig`，并检查 `horizon*action_block <= eval_budget`（`upstream/eval.py:262-266, 315-327`）。这表示每个 CEM 候选包含 25 个 primitive actions；一个 receding block 同样包含 25 个 primitive actions。不是 25 个 CEM horizon iterations。`action_num_blocks_per_step=[2,3]` 配合的 consistency loss 权重为 0，因此不参与默认 cost（配置 `:27-29`；`jepa.py:292-305, 332-342`）。

评估脚本把 `world.max_episode_steps` 设成 `2*eval_budget=100`（`upstream/eval.py:268-270`）；这满足配置注释“至少 eval_budget”，而 evaluator 仍显式收到 50 步 `eval_budget`（`upstream/config/eval/pusht.yaml:9,35`；`upstream/eval.py:399-407`）。

## `JEPA.get_cost` 输入与返回

- 方法签名为 `get_cost(info_dict, action_candidates)`；`info_dict` 至少要有 `goal`，路径还会取 `action`、张量化的 initial observation (`pixels`) 和可能的 `goal_*` 列（`upstream/jepa.py:307-331`）。候选张量前两维是 `(B,S)`：活动环境数、候选数；随后是 rollout 维和打包动作维。实现以 `action_candidates.size(1)` 判断并裁切重复候选维输入，然后把 tensor 移到模型 device；把首个候选的 `goal` 编成目标 latent，再对候选动作 rollout（同文件 `:311-342`）。
- `rollout` 要求 candidate 的 packed action dimension 能被 `action_encoder.input_dim` 整除，并据此拆成 action tokens；每次 rollout 接收 `(B,S,T,packed_action_dim)`（`upstream/jepa.py:142-173`）。`ActionPrefixEmbedder` 接受 `(B,F,input_dim)` token 序列（`upstream/module.py:335-359,457-463`）。默认 `get_cost` 比较最后一个预测 latent 与 goal latent 的平方差，对 embedding 维求和，返回 `(B,S)` 候选成本（`upstream/jepa.py:268-305`）。
- wrapper 应围绕原始 `get_cost` 单次调用记录动作输入 shape、返回 cost shape、有限性和耗时；不要为取 trace 重复调用 teacher。候选环境数可从 solve 的 `info_dict['pixels'].shape[0]` 记录，每次 solve 单独保存，避免不同活动环境数的数组被错误拼在一起。

## 官方 task selection 与结果

`eval.py` 按 `episode_idx`（缺省时 `ep_idx`）求每条 trajectory 长度，允许的最大起点为 `length - goal_offset_steps - 1`，再从 `step_idx <= max_start_idx` 的有效数据行中使用 `default_rng(seed)` 不放回抽取 50 行，并按 row index 排序（`upstream/eval.py:334-364`）。这是随机采样 dataset start rows，不是按 episode 分层抽任务；同一 source episode 可以多次入选。上游表达式 `choice(len(valid_indices)-1, ...)` 使最后一条 valid row 不会入选，复现时应原样保留以保证协议一致。

官方脚本调用 `World.evaluate_from_dataset`，随后打印并向 `pusht_results.txt` 追加其 `metrics` 与配置（`upstream/eval.py:399-407,423-438`）；这里没有验证 metrics 是否包含完整 50 项的 episode success vector。它另外将 CEM 每次 `outputs['costs']` 收集起来，以 `np.stack` 聚合成跨 replan 的 per-task 均值（`upstream/eval.py:160-190,377-420`）。若成功环境退出活动集导致每次 cost 数组长度不同，`np.stack` 不适用；应保留 per-solve cost 数组，不要伪造跨 solve 的同列对应关系。

最小结果文件建议保存：解析后的官方 config、官方 commit、checkpoint 来源/加载方式、50 条 `(row_index, episode_idx, start_step)`、原始 evaluator metrics、逐任务 success vector、逐 solve 的 active environment 数/candidate 输入输出 shape/同步计时，以及 finite 与数量完整性门。官方 wrapper 已有 `solver.solve` 与 `model.get_cost` 可挂接的位置；现有本地 teacher helper 已示范保存 row index 和逐任务 outcome（`meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/official-lewm-dataset-teacher-baseline/run_dataset_teacher_baseline.py:268-312,316-370`）。若 wrapper 只能从 evaluator 参数看到 episode/start step，可在该 dataset 上回查对应源 row index。

## 与本地 LeWM teacher baseline 的边界

本地 helper 与 Fast-LeWM config 共用 seed 42、50 个起点、goal offset 25、budget 50、100-step world cap、全数据 scaler 和 224 ImageNet pixel transform；其 protocol 记载了完全相同的官方随机 row 选择（`.../official-lewm-dataset-teacher-baseline/PROTOCOL.zh.md:7-17`）。但它使用 LeWM teacher checkpoint loader 及 `World.evaluate(..., video=None)`，并非 Fast-LeWM checkpoint 或逐字调用 `evaluate_from_dataset`，不能替代 Fast-LeWM 结果。它可复用的是任务身份记录、完整 success vector/finite gate 和结果 JSON 模式。

项目现有 historical backend 的补查还发现 `_action_buffer` 是每环境 deque 列表，而 Fast-LeWM `eval.py` 的 `_install_fast_buffered_action_path` 假定单个 deque 并直接 `.popleft()`（`upstream/eval.py:21-59`）；该观察同样不能归因于 PyPI 0.0.6。若 pinned package 保持 list 结构，应禁用此优化并保留 native `WorldModelPolicy.get_action`。当前 `eval.py` 也依赖 `HydraConfig.get().runtime.output_dir`（`:332-333`），直接调用 `run.__wrapped__(cfg)` 时需先显式初始化 HydraConfig/runtime output dir。上述是入口兼容项，须以隔离 pinned package 检查确认。

**验证边界：**Fast-LeWM 源码结论为静态分析；historical backend 观察与 PyPI `stable-worldmodel==0.0.6` 不是同一证据。checkpoint 反序列化、完整依赖导入、官方 evaluator metrics 键/数量、活动 batch 的实际变化和 25-action buffer 行为，仍需后续验证；本次未训练、未运行评估或提交作业。
