# LeWM 延迟与 true-async runner 架构审查

只读核对基于 pinned 上游版本：LeWM `lucas-maes/le-wm@8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`，stable-worldmodel `galilai-group/stable-worldmodel@10c26dbd5677083fa31dba69eb738b973845e9a4`。项目冻结记录见 [`official-pusht-cem/FREEZE.json`](../../../meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/official-pusht-cem/FREEZE.json)；其说明 commit 是 staging provenance，实际执行目录没有 Git metadata。PushT/Reacher 官方配置都设 `num_envs=50`、`horizon=5`、`receding_horizon=5`、`action_block=5`，见 pinned LeWM `config/eval/pusht.yaml:5-8,20-28` 与 `reacher.yaml:5-9,19-29`。

## 最小接入点

保留 `World.evaluate` 的任务初始化、成功统计与 callback 流程，只替换其内部 `_run` 调度器，或由等价的小 runner 调度。Pinned stable-worldmodel 源码位置：

- `stable_worldmodel/world/world.py::World.evaluate`（175-238）、`_evaluate_from_dataset`（456-523）、`_run`（327-348）；原同步循环 `_run_iter`（349-417）。`_evaluate_from_dataset` 已负责 reset、应用 `set_state`/`_set_state` 和目标 callables、逐步成功记录与视频帧 callback。
- `stable_worldmodel/policy.py::WorldModelPolicy.set_env`（302-318）和 `get_action`（320-409）。`get_action` 的计划路径是准备输入、找出空 buffer 环境、调用 solver、更新 warm-start tail、向每环境 deque 写入计划、pop 当前动作并 inverse-transform（331-409）。
- `stable_worldmodel/solver/cem.py::CEMSolver.solve`（109-259）；`stable_worldmodel/world/env_pool.py::EnvPool.step`（107-129）。

true-async runner 由 control thread 独占环境步进、policy deque、`_next_init` 和计划安装。每次请求时在 control thread 用 pinned `_prepare_info` 准备并切片输入，对 `World.infos` 的数组/tensor 及 `init_action` 做快照；worker 只调用 `solver.solve(snapshot, init_action)`，完成后把结果交回 control thread，再按 `WorldModelPolicy.get_action` 的 372-389 行规则安装动作和 warm-start tail。不要从 worker 调用 `policy.get_action`：它同时会读写 deque、`_next_init`、flush 状态并 pop 动作，和控制线程并发会竞争。`EnvPool` 的 stacked infos 每步原位更新（`env_pool.py:7-10,169-183`），因此不能把 live info 引用交给 worker。CEM solver 共享可变 `torch_gen`（`cem.py:47`），并在 solve 中调用 callback 的 reset/start/update/end 生命周期（137-139,172-238）；一个 solver 至多允许一个 solve worker in-flight。

## 启动动作、buffer 与计时

true-async 从 reset 后立即启动首个 plan 请求，仿真从第一个 tick 起继续推进。首个 plan 尚未返回时执行经环境语义确认的 neutral **环境动作**，并单独报告 startup 延迟；之后 buffer 用尽时重复上一个实际执行的环境命令。模型动作经过 `process['action'].inverse_transform`（policy.py:406-408），因此 neutral 应直接以 environment action 表示，不能把 policy 坐标中的零填进 deque。PushT 的 action space 是 `Box[-1,1]^2`；默认 relative action 会将 `a=0` 映射为当前位置目标（`envs/pusht/env.py:41-45,73-75,303-320`），但 PD 和物理过程仍继续。Reacher 由 `swm/ReacherDMControl-v0` 注册（`envs/__init__.py:69-72`），使用 qpos-match task；DMControl wrapper 把动作缩放并每个 world tick 重复两次（`envs/dmcontrol/reacher.py:28-47,135-156`；`envs/dmcontrol/dmcontrol.py:28-33,101-118`）。Reacher neutral 应在 pinned action bounds 下确认并直接用环境动作零值；报告 world-tick pacing，若换算秒数则同时记录实际 DMC control timestep。不要将 sleep 后冻结环境称作 true-async。

默认计划每次写入 `5*5=25` 个 world-tick 动作（policy.py:298-315,372-389）。由于 `horizon == receding_horizon == 5`，`rest = actions[:,5:]` 为空；默认 `warm_start=True` 不会产生非空 `_next_init` tail。若配置改变，仍应保持 pinned 的 tail 更新规则。计划返回后若 episode 已终止则丢弃；仍存活时按协议立即从返回计划的第一个动作执行，同时记录 observation age/staleness，别悄悄跳过前缀或重采样。World 每次 step 会产生像素（`wrapper/default.py::AddPixelsWrapper.step`, 437-448），render、resize 和视频 callback 都会计入墙钟开销；保留模型必需的像素路径，并明确视频是否开启。

fixed-K 是独立的同步延迟条件：在观测点同步求出计划，之后精确推进 K 个 world tick（既有 buffer、上个命令或 startup neutral），再开始应用新计划；K=0 使用 pinned 同步 `get_action` 路径。K>0 的 solve 期间仿真冻结，这个条件测的是固定 simulation-age，不是 realtime async。true-async 才是在 solve 期间持续推进环境。两种模式均按真实 world tick 记动作数、观测年龄、solve wall time、step/render wall time、deadline overrun 与 RTF。

## 50-env K0 与 single-env RNG 边界

原版 LeWM evaluator 的 `num_envs=50` 应保留为 batch-50 K0 gate/reference；`EnvPool.step` 是 Python 顺序遍历各环境，而非 50 个并行仿真线程（env_pool.py:121-129）。单环境 realtime deadline 应以 `num_envs=1` true-async 测量。此 N=1 runner 需额外跑同 task list 的 N=1 synchronous K0 作为 paired baseline；已有 batch-50 的 49/50 仅作原 evaluator 的独立 parity/reference gate，不能宣称与 N=1 逐动作等价。

RNG caveat：CEM `torch_gen` 按传入 batch 形状顺序消耗随机数，且 `batch_size=1` 会在一个 solve 中逐环境循环（cem.py:47,140-184）。batch-50 每次 solve 先为所有环境分配 draw；串行 N=1 则先耗尽一个 task 的全部 replans，环境间候选创新不同。每个 task 都重置成同一个 seed 还会重复首轮 innovations。报告中应明确这项 RNG 流差异；配对的 N=1 K0/delay/async 若要求相同候选创新，可冻结 task-id × planner-call seed，并在各 arm 同点重置 solver generator，同时将其说明为 runner 配对策略，而非原 batch-50 RNG 等价。

