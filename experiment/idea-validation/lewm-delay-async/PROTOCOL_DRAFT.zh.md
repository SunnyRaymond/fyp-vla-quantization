# LeWM PushT / Reacher 延迟与异步评估草案

目标是用原版 LeWM checkpoint、任务和 CEM，对 PushT 与 Reacher 分别测量同步、固定仿真步延迟和推理期间继续演进的 true-async 闭环成功率。模型不重训；先验证评估接入，再比较调度模式。

## 共同定义

- 每个任务按 pinned 官方 evaluator 从有效 dataset start rows 中选择 50 个不重复的 row；同一 episode 可以贡献多个 start step。manifest 固定 `(row_index, episode_idx, start_step, goal_step)`，所有条件重复使用相同起点与目标。PushT 固定使用作业 `25534994.pbs101` 的 50 个任务；Reacher 在其首个同步作业中按官方 seed 42 选择并冻结 50 个 rows。
- `K=0,1,2,4,8,16`，单位是该环境的 control tick。`K=0` 是同一新 runner 的同步 control，必须先对齐原始 evaluator；PushT 还必须逐任务对齐已有 49/50 向量。若对齐失败，停止该任务的延迟/异步运行。
- 固定延迟：在观测时发起 CEM；推理计算期间仿真冻结，随后环境先推进恰好 `K` tick，执行既有动作 buffer；buffer 用尽时保持上一个已执行命令。首次无 buffer 时以已定义的中性环境动作推进这 `K` tick。然后从返回计划的第一个动作开始执行。episode 的总仿真步预算保持官方值。这个条件模拟固定 observation-to-application *simulation age*，不声称 wall-clock realtime。
- true-async：观测与发起 CEM 后，环境以该环境的目标 control 周期继续推进；期间执行既有 buffer，耗尽则保持上一个命令。推理完成后从返回计划的第一个动作开始执行；episode 若已终止则丢弃该结果。记录实际 wall/simulation observation age、每次 CEM 耗时、控制周期 overrun、RTF、终止前丢弃数。不得用 `sleep` 但停止仿真来冒充 async。
- 首次请求无既有动作 buffer；strict true-async 从 reset 就发起 worker solve，control thread 以环境动作空间中的中性命令逐 tick 推进，不能等待首个 solve。PushT 的中性命令为 env-space 零位移；Reacher 须在 pinned wrapper 上确认中性命令含义。固定步延迟的首轮等待定义单独记录。结果必须区分启动成本与后续重规划；不能从 batch-50 solve 时间直接推断单环境 deadline。
- 官方同步 K=0 是 50-env batch；true-async 为每个任务独立 wall-clock controller，需有对应 single-env 同步对照。CEM RNG 消费随 batch shape 改变，不能把 batch-50 49/50 直接当成 single-env true-async 的严格配对成功率基线。
- 原版图像预处理、action/state scaler、CEM `300/30/30`、plan horizon/receding horizon/action block、成功判据和 episode budget 均不更改。若需改变这些设置以让 true-async 跑通，作为新协议单独报告，不能与原同步成绩直接拼接。

## 有效性门槛

1. compute-node wrapper 先验证 `PBS_JOBID`、`PBS_NODEFILE` 与非 login 主机；login 只做轻量连接、提交和状态/小日志读取。模型与数据 I/O、环境安装、推理和 benchmark 均在获批 PBS allocation 内。
2. batch-50 `K=0` 与 frozen 同步任务逐项一致。Reacher 若尚无本地同步基线，先完成 pinned 同步 50-task control；single-env 同步对照则必须与 true-async 使用相同的单任务 planner 配置、随机种子规则与任务身份。
3. 每个条件保存完整 50 项 task success、起点身份、planner request/apply/drop trace、实际动作数及 timing；GPU 作业期间定期将利用率和显存采样追加到 job log。
4. 报告 paired gain/loss，不把 50 个相同起点在多条件下的运行当作独立任务；若 true-async RTF 未达到 1 或 simulation pacing 超期，保留原始结果，但不称为达到目标频率的真实实时评估。

这个文件在首个新结果前按实际 runner 和 pinned Reacher 资源确认后冻结。不要据此草案声称任何新实验已完成。
