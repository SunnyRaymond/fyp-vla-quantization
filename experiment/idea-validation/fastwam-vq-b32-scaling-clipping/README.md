# B32 grouped scaling + clipping

目标：在已有 C80 clip_only80 的收益上，检查 activation 分辨率的 group 间分配能否继续降低 first10 motor RMSE 和 future latent NMSE。late22、full32 和 gripper 仅记录为诊断，不进入 loss 或推进门槛。

从原 C8 rho1.5 baseline 初始化，执行固定80次更新；不从已训练的 clip_only80 继续训练。C80 contexts、顺序、seeds、共享 teacher、原 loss 和 ratio optimizer 设置均沿用已完成对照；不使用新的 development set，不以 held-out 选择 checkpoint。baseline 和 C80 clip_only80 的320条端点记录直接复用，只新增本方法的 calibration80 和原 held-out80 评估。

对固定 Smooth0.5/H128 后的 activation Z，采用 `Q_A4(Z / D; ratio) @ (Wq * D).T + bias`。每个物理 Linear 的输入每32个 channels 共享一个 D，跨 token 和 denoising step 共享，`D = 2**k`，`k ∈ {-1,0,1}`，初始化 k=0、ratio=1。冻结 codebooks、indices、row gains、Smooth/H128、bias 及其他模型参数；补偿仅确定性地将原 B32 exponent 加 k，不引入新的 weight 拟合。

每次 forward 都使用实际整数 k。训练 FP32 shadow log2，Adam 学习率0.02；ratio 学习率保持0.01，80次更新、首次更新后清 Adam state、gradient norm clip=1 均保持原设置。shadow 的 backward 是 LSQ/STE surrogate：将 dynamic absmax 视为常数、按 group 大小归一化并去除层内共同 log-scale 梯度；不声称这是硬量化函数的精确梯度。整数 exponent 始终受原编码的合法范围约束。

训练时使用 factorized activation compensation，节省重新 decode 的成本。正式端点评估前将 k 折入实际 B32 exponents 并 decode，要求与 factorized 路径的完整 action/future query 一致，所有报告指标来自 folded weight 路径。GPU allocation 内先执行 k=0 输出/梯度、非零 k 权重折叠、零输入和 tail/bounds 的小型 selfcheck。

推进条件：原40 contexts ×2 seeds 的80个 held-out queries 完整覆盖，first10 motor 和 future latent 两项均优于原 baseline 及 C80 clip_only80。即使失败，也须先检查实际变动的 groups 数量和 calibration/held-out 差距；B32 离散 scaling 的失败不能直接否定连续 per-channel scaling。

每个 GPU job walltime 为01:59:00，软截止6840秒；每15秒向 job log 记录 GPU utilization/VRAM。仅在确认 PBS terminal/exit75 和 checkpointed 状态后才允许续跑。全部 Torch 和模型工作只在真实 PBS compute allocation 进行。监控采用 gpt-6-luna / xhigh，每10分钟检查，Q/R 无实质变化时静默。

结果属于 fixed-input、decoded-BF16 Microscaling VQ + fake A4 数值证据；不代表 closed-loop success、native INT4 性能或未见任务泛化。原 held-out 已被多次分析，因此属于固定 cohort 的工程比较。

## 本轮结果

已完成。GPU `25855238.pbs101` 用时01:27:00，CPU aggregate `25856260.pbs101` 用时00:00:21；两者均通过 terminal/exit0、artifact exit0、PHASE_COMPLETE 和 matching complete summary。完成80 updates、新方法 calibration80 + 原 held-out80 queries，复用320条对照记录；新增 teacher 为0。详见 [结果表](results/RESULTS.zh.md)、[解释与下一步](results/ANALYSIS.zh.md)、`completion_evidence.json`。

相对 C80 clip_only80，held-out first10 motor RMSE +0.3154%，future latent NMSE −1.2700%；两项同时改善门槛未通过。本轮只改变13/56,466个 groups（0.0230%），涉及3/614个 Linear。下一步建议先检验连续 B32 scaling，暂不直接放开全量 per-channel；未自动提交下一轮。
