# 第一轮：块内 action-conditioned dynamics 与低维通信

状态：本轮已完成，主 PBS job `25563194.pbs101` 的 45 runs 与预设 global16 PBS job `25563721.pbs101` 的三个附加 runs 均正常退出并核验。结果见 [RESULTS.zh.md](RESULTS.zh.md)。权威配置为 [FREEZE.json](FREEZE.json)，附加配置单独保留于 [GLOBAL16_FREEZE.json](GLOBAL16_FREEZE.json)。实际源代码均在各 compute run 目录保存 snapshot。

## 范围与结论边界

本轮验证 64D 受控动力系统中的表示和计算结构。完整执行独立、低秩耦合、稠密耦合三种条件，以及 dense、random-block、learned-block、learned-global4、learned-local4 五个 primary arms，每个三个 training seeds。保留失败与 inconclusive；不以某一个 passing subset 代替本轮全部结果。

模型保留全部 64 个状态维度，使用完整正交坐标变换，不进行 PCA 截断。四块各 16D；已知 8D action 条件在所有 arms 中保持相同。局部模型用两个 hidden-64 layers，预测 residual update。公共 message 为 4D 线性全局摘要；local4 仅看本块，message 投影总参数、局部 MLP 输入宽度与 global4 相同。

可学习正交 Q、局部 MLP、message projection 同时训练。部署时固定 Q 并缓存其数值；rollout 保持在新坐标中。必须另报每步重构时的延迟，不能以不兼容最终接口的计时证明加速。Dense 的 hidden width 通过参数计数预先确定，目标是 matched global4 total trainable parameter count，误差不超过 5%；报告所有 arms 的实际参数和计算范围。

## 受控系统

真实状态 x 由四个局部 16D 子状态组成，观测 z=Mx 使用同一个固定稠密正交 M。三个条件使用相同局部动力学、action 输入和尺度。independent 无跨块项；lowrank_coupled 增加真实 rank-4 线性跨块反馈；dense_coupled 增加 action-dependent 稠密非线性反馈。生成器必须确定性、状态数值稳定，并提供 known-coordinate control 和真实 local/coupling component。它们是方法的正/负对照，不证明真实视觉世界具有相同结构。

执行前具体生成器冻结为 x_next = 0.8x + 0.15 tanh(A_local x + B_local a) + coupling。低秩项为四条邻块 rank-one 线性边，系数 0.08；稠密项为 0.035 tanh(Dx) * (1 + 0.5 tanh(Ca))，D 为正交矩阵。两个系数没有做 test 调整；稠密条件的作用范围广，但不将该系数直接解释为更大的误差或严格不可分解。

生成器与模型的必要测试在 compute allocation 中运行：正交可逆性/距离保持、独立条件的跨块敏感性为零、lowrank Jacobian correction rank bound、global/local message 参数匹配、坐标正确重构、future-action causality、deterministic paired batches。稠密耦合是经验负对照，不声称数学上不可能分解。

## 数据、训练与分析

每个条件 train/dev/test 为 512/128/128 个不同 seed 生成的 episodes，每条 40 步。三个训练 seeds 为 1101/1102/1103，各 arm 使用完全相同的 minibatch draws 和 action sequences。固定 1500 steps，batch 128，lr 1e-3；one-step loss 加 0.5 倍五步 free-running 平均 loss；训练集 delta energy 归一化。固定最后 checkpoint，不以 dev 或 test 最佳结果挑 checkpoint。

评估 horizons 1/5/10/20，先按 episode 汇总，再按 training seed 配对比较；不把同一 episode 的帧当独立重复。报告每个条件/arm/seed 的 rollout error、action-response error、可解释结构诊断、参数量、训练时间和 GPU memory。Action-response 从同一 heldout 状态出发，base actions clamp 到 [-0.9,0.9]，加减四组固定 Rademacher ±0.1 扰动；真实系统重新 rollout，报告预测差分误差，另以真实差分 energy 归一化作为诊断。主比较是 learned-global4 对 learned-local4；表示比较是 learned-block 对 random-block。报告每个 seed、episode paired median/mean difference 与 episode-bootstrap interval；三个 seeds 的区间只作为 pilot 描述，不写总体因果定理。

初步 quality gate：相对参考 dense 的 primary h10 error 不超过 10%，并以 0.02 的 absolute normalized tolerance 处理 near-zero reference。Speed gate：包含变换/通信/重构的 batch-1 与 batch-300 latency 分别报告，不选择有利 batch 冒充统一结论。20% 是预设工程门槛，不是统计显著性标准。

global16 只在 primary global4 对 local4 和 learned-block 的低秩条件 dev contrast 出现一致改善时附加，并独立标记，不修改 primary 冻结配置，也不复用 test 调参。任何 visual / LeWM / CEM 扩展均不属于本轮。

## 运行与监控

所有数据生成、测试、训练、benchmark、环境计算在获批 PBS compute allocation 内；login node 只连接、提交、查状态和操作小控制文件。脚本检查 PBS_JOBID、PBS_NODEFILE 中本机 membership、非 login hostname。GPU job 每 30 秒向 job log 写利用率和显存。外部 job 监控使用 gpt-6-luna，不重复提交 queued/running 作业。未建立指定模型并能唤醒本 goal 的定时监控闭环时，goal 保持 active。

## 方法来源

本模型为本轮研究设计，不声称 prior-art novelty。局部机制/通信结构参考 RIMs；显式动力学坐标思想参考 EDMD。实验分组、配对与重复层级使用 experimental-design skill。该 skill 的来源引用：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, arXiv:2609.00065（当前 v2）。https://doi.org/10.48550/arXiv.2609.00065
