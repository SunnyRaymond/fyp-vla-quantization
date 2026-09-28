# LeWM official CEM → PushT closed-loop protocol

## 研究问题与边界

本实验检验已通过 fixed-observation adaptive-CEM gate 的 `late_teacher7` 能否接入 pinned official LeWM CEM，并在真实 PushT 环境闭环里保留 planner 效果。固定 checkpoint 为训练 job `25239551.pbs101` 的 `treatment_step1000.pt`；实验期间不再训练或挑 checkpoint。

历史 `adaptive-teacher-schedule` 使用 hand-written CEM 更新环。它证明的是该自定义 loop 下的 30-round adaptive-CEM 结果，不是 official `CEMSolver` 部署结果。本实验直接实例化 pinned `stable_worldmodel.solver.CEMSolver`，不改 `solve` 内部 sampling、candidate-0、`torch.topk`、elite gather、mean/variance update 或 final-mean 返回逻辑。只有 `get_cost` 按冻结轮次在官方 LeWM teacher 和现有 student 之间路由。由于原 hand-written loop 与 official solver 的 RNG、tie-breaking 和更新实现可能不同，不要求两者互相 bitwise 重现；Stage 1 的 fidelity 比较只在同一 pinned official solver 内进行。

所有设置、seed、gate 和停止条件已写入同目录 `FREEZE.json`。不读取 HDF5 `valid[600:]` 生成 simulator tasks。LeWM 原始 evaluator 支持 dataset-driven evaluation；本 Stage 2 明确使用 `World.evaluate(dataset=None)` episodic 路径，因此由 simulator 的 reset/step 产生任务状态。

## pinned source 的执行语义

- `stable_worldmodel/envs/__init__.py` 注册 `swm/PushT-v1` 到 `stable_worldmodel.envs.pusht.env:PushT`。
- `stable_worldmodel.world.World` 构造 Gymnasium env pool；`World.evaluate(..., dataset=None)` 走 episodic `_evaluate → _run_iter`。`_run_iter` 以给定 seed reset，循环调用 `policy.get_action(infos)` 和 `envs.step(actions)`，直到终止或截断。`reset_mode="wait"` 保留 terminal 状态用于记录。
- `World.envs` 是 `EnvPool`，其 `envs` 保存具体单环境 wrapper。源码复核更正了上一版对 reset action 的描述：`EverythingToInfoWrapper.reset` 先令 `info['action'] = action_space.sample()`，但随后对非-dict action 执行 `info['action'] = np.full_like(info['action'], np.nan)`。因此 instrumentation 中看到的 finite sample 是占位生成过程，`World.reset` 实际收到并由 `EnvPool._stack_fresh` 堆叠进 `world.infos['action']` 的是全 NaN 数组；`World.reset` 将 EnvPool 返回的 stacked info 原样保存。Base `Policy._prepare_info` 通过 `torch.from_numpy` 把数字 numpy action 转为 tensor，保留 NaN mask。源码位置与引用版本：[`EverythingToInfoWrapper.reset`](https://github.com/galilai-group/stable-worldmodel/blob/10c26dbd5677083fa31dba69eb738b973845e9a4/stable_worldmodel/wrapper/default.py#L201-L259)、[`EnvPool._stack_fresh`](https://github.com/galilai-group/stable-worldmodel/blob/10c26dbd5677083fa31dba69eb738b973845e9a4/stable_worldmodel/world/env_pool.py#L148-L167)、[`World.reset`](https://github.com/galilai-group/stable-worldmodel/blob/10c26dbd5677083fa31dba69eb738b973845e9a4/stable_worldmodel/world/world.py#L168-L174)、[`BasePolicy._prepare_info`](https://github.com/galilai-group/stable-worldmodel/blob/10c26dbd5677083fa31dba69eb738b973845e9a4/stable_worldmodel/policy.py#L77-L134)。该 staging provenance 与本轮 runtime import API guard 的边界不变。
- 配对仍覆盖**全部** prepared keys，且不移除 action。除 action 外每个 key 必须 shape/dtype 完全一致并通过 `torch.equal`。对 reset action 要求 shape/dtype 一致、NaN/+Inf/-Inf mask 分别完全一致，且 `torch.equal` 比较所有 finite entries；因此只把源码定义的 NaN placeholder 按语义比较，没有容忍任何有限 action 差异或异号 infinity。每次 Stage 1 reset 前及 Stage 2 `World.evaluate` 前继续 seed `world.envs.envs[0].action_space`，然后以同一 task seed reset simulator；action-space RNG 与 PushT simulator RNG 分开。
- 该 reset action 是占位字段，不参与本实验首个 planner cost：Pinned `CEMSolver.prepare_init_action` 对未实现 `Actionable` 的 LeWM `JEPA` 使用 zero padding；`JEPA.get_cost` 将其从 goal encode 字典中移除，随后 `JEPA.rollout` 用当前 candidate 的 `act_0` 覆盖 `info["action"]`。`reward=np.nan` 不在 runner 的 prepared-observation 对比白名单中。
- `World.set_policy` 调用 `policy.set_env(self.envs)`；官方 `WorldModelPolicy` 随后以 frozen `PlanConfig` 调用 solver 的 `configure`。plan horizon 为 5 个 packed 10-D token，每 token 表示 5 个连续 2-D PushT actions；每次 MPC plan 实际执行 `receding_horizon × action_block = 25` 个环境 step。因此 50-step episode 最多规划两次。
- pinned `PushT.reset(seed)` 通过 seeded variation space 生成初始 agent/block state 与目标；同 seed 重置并要求四臂实际 `state` 与 `goal_state` 逐元素一致。
- pinned `PushT.step` 推进 PD controller 和 Pymunk physics。environment `terminated` 对应 `eval_state` 的成功条件：`state[:4]` 到 `goal_state[:4]` 的 L2 距离小于 20，且考虑 block 形状旋转对称后的角度误差小于 π/9。`World.evaluate` 把 `terminateds` 作为 episode success；50-step TimeLimit 的 `truncated` 不是成功。

`FREEZE.json` 中的两个 commit ID 是 staging provenance/reference。远端 LeWM 与 stable-worldmodel 目录是无 Git metadata 的 source snapshots，因此本轮 runtime 会验证实际 import root 和所用 API/lifecycle，但不能独立证明 snapshot 与 commit ID 完全一致。旧 DINO-WM environment-eval 使用过 seeds `[1,100,199,298,397,496,595,694]`；新 Stage 1/2 simulator seed 范围与其分开。旧 HDF5 episode ids 与这批新 simulator seeds 没有对应关系。

## Stage 1：official solver 语义、planner 质量与完整耗时

Stage 1 用 8 个新 simulator reset seeds 生成真实 PushT reset observations，但不执行环境动作。四个策略 arm 的执行顺序用冻结 Python seed 随机化并记录；另以独立冻结 seed 随机化 native/routed teacher fidelity pair。每个 fresh `World` 在 `World.reset(seed=s)` 前将具体环境的 `action_space` seed 为同一 `s`。所有路径的 `pixels`、`goal`、`state`、`goal_state`、`proprio`、`action` 等 policy-prepared key 必须逐 key 配对：非-action key 要求 shape/dtype 完全一致且 `torch.equal`；reset action 要求 shape/dtype 一致、NaN 与正负 infinity masks 各自一致，并对所有 finite 值执行 `torch.equal`。summary 保存 per-key flags。每条配对路径使用相同 solver-owned `torch_gen` seed，官方 solver 会据此生成相同顺序的标准正态 innovations；CEM distribution 和 model scores 可随策略分叉。

先做 native teacher fidelity 检查：对每个 reset seed，比较 unmodified CEMSolver + 原始 teacher `get_cost` 的 `native_teacher_reference` 轨迹，与 unmodified CEMSolver + 透明 `get_cost` 委托给原始 teacher 的 `routed_teacher_reference` 路径。native 路径外层只有只读 capture proxy：保存首次真实 cost 输入的 sample-0 view，然后以相同参数调用原始 `JEPA.get_cost` 并原样返回，不改输入或返回值。要求每个 iteration 的 candidate tensor、cost tensor、top-k indices、elite tensor、post-update mean/variance 以及最终 plan/first action 均 bitwise 相同。runner 仅在这对 solves 中暂存 GPU tensor 引用，在计时结束后逐字段 `torch.equal`，summary 只保存每轮布尔相等标志，不做大张量搬运或哈希。四个策略 arm 使用相同的 lightweight round/finite callback，因此其 full-solve latency 可直接比较。

再比较 `student_only`、`late_teacher7`、`teacher_only` 和同预算 secondary arm `uniform_teacher7`。每个 solve 记录每轮编号、cost 与 mean/variance update 的轻量 finite 标志、final packed action、teacher-call rounds、同步 planner walltime、峰值 CUDA memory 和输出 finite 标志；不为这些臂搬运候选张量或计算哈希。late schedule 固定为 rounds 24–30，不根据运行情况平移或改策略；uniform7 固定为 4、8、12、16、20、24、28。

质量审计重算 final plan 的 teacher cost 时，复用每个策略臂与 native/routed fidelity 路径首次真实 CEM `get_cost` 调用的 expanded `info_dict`，从候选 sample 轴选 sample 0，再把 final plan 作为唯一候选 `[B,1,T,10]` 送入原始 teacher。这样沿用 CEM 实际的 prepared observation 和 sample/time 维度；不再次对 raw `world.infos` 调用 `_prepare_info`。报告中保存 policy-prepared info、实际 CEM candidates/costs、expanded cost info、单候选 scorer 输入和 scorer 输出 shape，便于核对官方 image encoder 的 batch/channel 维度。

Stage 1 质量差值定义为 `(late_teacher7 final-plan official teacher cost − student_only final-plan official teacher cost) / native teacher round-1 candidate-cost population SD`；负值有利于 late7。Stage 1 只有同时满足以下条件才继续 Stage 2：

1. 8/8 paired seeds 的 native teacher 与 routed teacher-only traces 全部逐字段 bitwise 相等。
2. late7 每个 solve 恰好 7 次 teacher call，且恰为 rounds 24–30；其他 arm 的调用次数和 uniform rounds 也与 freeze 一致。
3. late7 相对 student-only 的 official-teacher final-plan cost，按 native teacher round-1 candidate population standard deviation 标准化后，中位数 ≤−0.1，且 8 个 reset seed 中至少 5 个严格改善。
4. late7 mean full `CEMSolver.solve` time 至少比 teacher-only 低 30%。计时包含官方候选生成、scoring、top-k 和分布更新，并在 GPU 前后同步；固定 observation 下不包含 environment stepping。
5. reset pairing、完整性、finite 检查全部通过。

如果任一门失败，Stage 1 写出完整摘要并以 fail-closed 方式停止；Stage 2 不提交。

## Stage 2：50 个 fresh paired simulator episodes

只有 Stage 1 summary 的 `overall=PASS` 才允许 Stage 2 runner 开始。每个 block 是一个 task seed，四臂分别创建 fresh `World`，调用 `World.evaluate(episodes=1, seed=task_seed, reset_mode="wait", dataset=None)`，episode 在 official success 或 50-step TimeLimit 时结束。N=50 取自 pinned official LeWM PushT evaluator 默认 `num_eval=50`，不是临时 pilot；没有可靠的 paired LeWM closed-loop 先验效应可支持正式 power claim，因此不声称该样本量对某个 effect 已有特定 power，也不在查看结果后追加 seeds。

主要三臂是 `student_only`、`late_teacher7` 和原始 official `teacher_only`。`uniform_teacher7` 是同 7 次 teacher budget 的 secondary 对照。每个 block 的四臂顺序由预冻结 Python RNG 逐 seed 随机化，并写入 summary。每臂在 `World.evaluate` 前以同一 task seed 设置 concrete `action_space` RNG，再以该 task seed 执行 simulator reset；实际起始 state、goal state 和所有 policy-prepared key 都必须按上一节的语义精确规则一致，否则实验无效，per-key equality 会写入 episode JSONL/summary。CEM solver 在每个 task seed 与 plan-call index 用相同冻结 seed 重置 pinned `torch.Generator`，使对应 call 的 candidate innovations 顺序完全配对。若两臂因成功/失败步数不同执行了不同数量的 plan call，报告实际调用数；不虚构不存在的配对调用。

每步记录 official state distance、position error、rotation error、reward、terminated/truncated；每个 episode 记录成功、环境步数、总墙钟时间、CUDA 同步的每次/合计 planner walltime、teacher calls 以及 peak allocated/reserved GPU memory。Pinned `CEMSolver` 实际持有 `self.torch_gen = torch.Generator(device=device).manual_seed(seed)`，且候选采样明确传入 `generator=self.torch_gen`；runner 在每个配对 planner call 前重设这个 solver-owned generator 并记录 seed，不改全局 CUDA RNG state。每完成一个 arm episode 就 append 并 flush 一行 JSONL；若 job 因资源或节点事件中断，已完成结果保留，不自动 resubmit。

主要 outcome 是每 seed 的二元 simulator success。报告各臂 50 个 episode 的 success count/rate、三臂 paired success table、`late7 − student_only` 及 `late7 − teacher_only` 差异，并对 `late7` 对 `student_only` 做 two-sided exact McNemar test。冻结的 practical gate 要求 late7 比 student-only 至少多成功 5/50 个 paired seeds 且 McNemar `p < 0.05`；相对 teacher 的 practical noninferiority 要求 late7 success count 不低于 teacher 5 例以上。两个阈值都必须满足并通过完整性 gate 才记 `PASS`。否则标记 `FAIL` 或 `INCONCLUSIVE`，保留结果，不扩样、不改预算、不调 schedule。

## PBS 与可复现要求

Stage 1 使用单 GPU、16 CPU、110 GB、30 分钟上限；Stage 2 使用同资源与 2 小时上限。PBS 脚本和 runner 在重操作前同时要求真实非空 `PBS_JOBID`、有效 `PBS_NODEFILE`、运行 hostname 属于 nodefile 且不是 login/head/submit。所有 checkpoint/model load、encoder、planner、simulator、trajectory 处理都只在 PBS compute allocation 内执行。登录节点只用于小型控制文件上传、`qsub` 和低频 `qstat`。GPU 利用率及显存每 5 秒由独立 `nvidia-smi` 进程写入 `job.log`，退出 trap 直接 kill/wait 该 PID。SSH 默认 host-key verification 保持开启。

本协议借鉴实验设计中的分层、配对与顺序随机化，以及在无可靠效应量时避免虚构 power claim 的做法：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills*, arXiv:2609.00065, https://doi.org/10.48550/arXiv.2609.00065。该引文只说明程序性方法背景，不作为 LeWM/PushT 结果证据。
