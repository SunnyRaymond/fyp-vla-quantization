# Rolling Ball Interception / LeWM：第一轮协议草案

范围只包含 ReflexBench Rolling Ball Interception。这里记录待执行协议；不是已训练 checkpoint 或闭环结果。

## 当前前置条件

- 原生视觉任务依赖 Isaac Lab / Isaac Sim 的 RGB CameraCfg。已使用的 ASPIRE2A normal allocation 是 A100；NVIDIA 明确不支持没有 RT Cores 的 A100 / H100。用户确认目前只有 ASPIRE2A；NSCC 官方旧培训资料另列 A40 visualization nodes，当前访问路径和运行许可尚待核实。`headless` 不移除相机渲染需求。
- 数据与正式 100 epochs 训练已经完成；三阶段原生视觉闭环需要受支持的仿真资源。用户授权届时租 AutoDL RTX 4090，经 SSH 访问；先准备迁移和评估代码，到只剩 RTX runtime 时暂停 goal 等用户开机。
- 不将 state-only 输入替代视觉 baseline：任务 policy vector 含 ball pose/velocity、phase 和 predicted intercept 等内部状态。

## 固定来源与原生任务

- ReflexBench commit：`8bb931485093c6d98f8729774ad01bf824964e16`。
- 数据 commit：`cyx337/ReflexBench_dataset@9295b6e9878609a992047f0b8b65421a493299e7`；合并 LeRobot 数据集，Rolling Ball 是 task index 3，对应 200 个 episode。需要按 episode metadata 提取，不能将其他任务混入。
- `RollingBallInterception-Franka-DataCollection-v0`；fixed / wrist RGB 224×224；joint-position control。
- Physics dt=10 ms，control dt=40 ms，episode limit=3 s。所有条件使用相同的 control-step 和 task-event cadence。
- 当前 reset 直接进入 rolling phase；README 中随机 gate wait 描述与源码不同。球横向位置随机 ±0.30 m，机器人关节初值有 ±0.05 的扰动。
- 原生成功判据为 ball 在 catcher mouth 附近连续三次 containment checks 后进入 phase 4。部分物理步的手工推进不能让其中一个条件增加检查次数。

## Baseline 与训练

使用 vanilla LeWM encoder / predictor、prediction loss + SIGReg，以及 CEM / MPC；在该任务上从头训练一个 checkpoint。原始实现固定为 `lucas-maes/le-wm@8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`，另行准备在任务目录中；没有将本地 Fast-LeWorldModel 的训练脚本当作 vanilla baseline。CEM 固定为 StableWM `10c26dbd5677083fa31dba69eb738b973845e9a4`。

原始训练 recipe：224×224 单路 RGB、192-D latent、history 3、num_preds 1、tiny ViT patch 14、六层 AR predictor；loss 为 prediction MSE + 0.09 × SIGReg。AdamW lr=5e-5 / weight decay=1e-3，batch=128，100 epochs，bf16。官方 PushT frameskip=5；本任务已在 [TRAIN_FREEZE.json](TRAIN_FREEZE.json) 固定为 frameskip=1，预测步长与原生 40 ms control 一致。这是任务配置适配，必须随结果报告。第一轮只用 fixed camera；多视角不是默认范围。

数据 observation.state 与 action 都是 8D，但 action 存的是 absolute joint target；环境 action term 是 scale=0.1 的 relative joint command。七个关节需用 `(target-current_joint)/0.1` 转换，gripper intent 使用原生二值映射。内部 task state 不进入 model / planner。

只使用 Rolling Ball 训练集。该 task 的 200 个 episode 各有 26 帧、合计 5,200 帧；LeRobot episode metadata 未提供逐条 success 标签。Collector 默认只保存成功演示，因此它们很可能为成功演示，但这个判断不是由发布 metadata 逐条验证。按完整 episode 划分 train / validation，避免重叠时间窗口跨 split；图像预处理、action normalization 只由训练集确定。狭窄示范分布不保证 CEM 候选的 action coverage，不能仅凭示范数宣称模型具备规划能力。

Goal / task cost 在离线检查前固定，仅由训练集构造。不能读取测试 episode 的未来图像或内部 ball state。使用每个训练 episode 的最后可用 fixed-camera 图像，它由 collector 在 env.step 前记录，不保证已经处于接球成功后的 phase 4。Goal bank 是否能成为有效规划目标仍未验证；这项适配不能隐去或称为原封不动的官方任务 baseline。

离线接入使用 [PLANNER_FREEZE.json](PLANNER_FREEZE.json)：固定训练集 terminal-image goal bank；CEM 300 samples / top-30 / 30 rounds / horizon 5，MPC 每次执行一步。CEM initial info 只用最新一个真实 RGB frame，保持官方 WorldModelPolicy 的时间语义；模型训练 history=3 不变。示范 action envelope 的相同投影用于预测成本和最终命令，不能将该 envelope 称作机器人硬件安全认证。该 adapter 的实际输出与 timing 仍须 GPU smoke 验证。

## 三阶段比较

1. 同步、零计算延迟惩罚：planner 计算时 simulation 暂停。记录 planner wall time，但不将其换成环境推进。首先检查 baseline 能否完成任务。
2. 同一 checkpoint、同一初始条件快照和 planner RNG seeds，固定模拟延迟后再执行新动作。拟定延迟网格为 0 / 40 / 80 / 160 / 320 / 640 ms，均为 control dt 的整数倍。等待期间继续推进物理并保持最近 joint target；无旧动作时保持 reset joint target。这个网格仍待 runtime smoke test 后正式冻结。
3. 推理与 simulation 真正并行。simulation 连续使用旧 target，完成的最新计划在下一 control boundary 生效。加入 wall-clock pacing，并报告实际 RTF；若不能跟上实时速度，不能用目标 RTF 代替实测值。

三组共用 checkpoint、数据 split、goal/cost、相机、动作空间、CEM budget、history、action chunk 和成功判据。初始状态必须实际保存/恢复；仅重复 seed 不足以保证各异步运行的 reset RNG consumption 相同。比较使用 episode 作为独立单位，成对报告 success 和失败原因；同一条件不把多个 CEM solves 当成多个独立样本。

## 指标与最小诊断

- Native task success / episode。
- 完整 CEM / MPC wall time，以及 observation capture 到动作首次应用的 wall age 和 simulation age；不能用 predictor forward 时间代替。
- Simulation elapsed、wall elapsed、RTF，以及控制 deadline misses。
- Goal / score、action clipping、episode 终止原因。
- 固定 catcher pose 的诊断 baseline 使用相同初始状态，判断静态守候是否已经覆盖部分 lane。这是诊断，不替代 LeWM 结果。

原生 `async + use_real_latency + backend=server` 才进入并行路径；plain async 仍先阻塞推理再注入模拟时间。当前 evaluator 还缺 observation-age 记录与严格 pacing。最小适配还需保证各模式 task-event cadence 相同。

## 来源

- [ReflexBench 固定版本](https://github.com/LxRoboticsLab/ReflexBench/tree/8bb931485093c6d98f8729774ad01bf824964e16)
- [数据固定版本](https://huggingface.co/datasets/cyx337/ReflexBench_dataset/tree/9295b6e9878609a992047f0b8b65421a493299e7)
- [Vanilla LeWM](https://github.com/lucas-maes/le-wm)
- [Isaac Sim GPU 要求](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
