# Continuous B32 scaling + clipping80

本轮检验：将上一轮每个输入 B32 group 的离散 D∈{0.5,1,2} 改为连续 D=2^log2D，是否能在 C80 clip_only80 之上同时降低 first10 motor RMSE 和 future latent NMSE。group size=32，log2D 范围[-1,1]，不开放 per-channel、shifting 或 rotation。

从原 C8 rho1.5 baseline 的 D=1、ratio=1 开始固定80次更新，冻结 codebooks、indices、row gains、原 B32 exponents、Smooth0.5/H128、bias 和其他参数。沿用原 C80/update order、seeds、共享 teacher、loss、ratio Adam LR=.01；连续 log2D Adam LR=.02，沿用上一轮 shadow 的学习率。仍在首次更新后清 Adam state、gradient norm clip=1。late22/full32/gripper 仅诊断。

训练与评估的真实 forward 都是 `F.linear(Q_A4(Z/D; ratio), BF16(Wq * D), bias)`。连续 D 不取整；单独保存 FP32 log2 scales，不能声称精确写入现有整数 exponents。BF16 weight compensation 在 backward 重算，避免为每个 denoising step 保存一整套 dense weights。训练后的 cached BF16 weights 与训练所用 uncached 路径须通过完整 action/future query parity。

Backward 延续 stored-weight LSQ surrogate：`g = G @ BF16(Wq*D)`；input gradient 通过 clipping mask、D^-1、inverse H128/Smooth；ratio 与 group 梯度由 operand-precision `(q-u)`／clipped q scale derivative 聚合，dynamic absmax 视为常数。group 梯度按 sqrt(rows×groupwidth×7) 归一化并减层内均值。这是约定的 surrogate，不是硬量化或 BF16 rounding 的精确导数。小型 GPU selfcheck 检验 forward、D=1 与已有 clipping 的兼容、该 surrogate 的公式、zero/tail 等边界。

最终记录补偿 BF16 weight 在还原到原坐标后的 NMSE，避免把额外 weight rounding error 当作未改变权重。原 VQ 内容保持冻结，新增尺度 metadata 成本单独统计，不做 native kernel 性能声明。

新方法完整评估 C80的80个 seed0 queries，以及原40 held-out contexts×2 seeds=80 queries。复用 baseline、clip_only80、离散 B32+clip80 的480条端点记录，合计640条；不新增 teacher、development set 或更换 cohort。推进主门槛为两项均优于原 baseline 与 clip_only80，同时报告与离散 B32 的比较，不按 held-out 选 checkpoint。

GPU walltime01:59:00、软截止6840秒；每15秒记录 utilization/VRAM 到 job log；gpt-6-luna/xhigh 每10分钟检查，Q/R 无实质变化静默。只在真实 PBS compute allocation 做 Torch/模型计算，login 仅小型控制文件和状态查询。结果出来后判断是否扩展 per-channel，不自动扩展。

固定 cohort 已多次用于工程比较；这里是 decoded-BF16 VQ + fake A4 的固定输入数值证据，无 closed-loop 或未见任务泛化结论。

## 已完成

GPU `25856825.pbs101` 用时01:09:53；CPU aggregate `25857520.pbs101` 用时00:00:25。两者均通过 PBS terminal/exit0、artifact exit0、PHASE_COMPLETE 及 matching complete summary。完成80 updates、160条新方法端点，复用480条对照；原 held-out80 完整覆盖，新增 teacher=0。gpt-6-luna 监控已结束。

Held-out first10 motor RMSE=0.044588993，future latent NMSE=0.060690613。相对 clip_only80 分别变差1.6889%、10.6521%；四 suites 两项均变差，推进门槛未通过。当前配方不直接扩展 per-channel，先检验 scale 优化的梯度与耦合处理。详见 [结果](results/RESULTS.zh.md)、[分析](results/ANALYSIS.zh.md) 和 `completion_evidence.json`。
