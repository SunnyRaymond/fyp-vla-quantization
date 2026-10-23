# Per-channel scaling + clipping

直接将已完成 coupled B32 的 scale 粒度扩展为每个物理 input channel 一个 FP32 log2D。所有 tokens/denoising steps 共享同一 Linear 的 D；ratio 仍每 Linear 一个。

从原 C8 rho1.5 baseline 的 D=1、ratio=1 开始，新增80 updates；不从已训练 B32 继续训练。固定 C80/update order、原 loss、seeds、共享 teacher、Smooth0.5/H128、codebooks、indices、row gains、原 B32 exponents、bias 及其他模型参数。log2D∈[-1,1]、D LR=.02、ratio LR=.01、Adam 首步后清 state、gradient norm clip=1 均沿用。

Forward 为 `F.linear(Q_A4(Z/D;ratio), BF16(Wq*D), bias)`。保留完整 dynamic-absmax/input coupling、amax ties 等分、stored-weight LSQ/BF16 coefficient surrogate；逐 channel 梯度只沿 token rows 求和，按 sqrt(rows×1×7) 归一化，保留层内公共尺度梯度中心化。训练与评估使用同一 BF16 compensation；D 单独保存，不能精确折入原整数 B32 exponents。

GPU allocation 内先做独立 live-amax autograd STE 检查，覆盖 D=1/非1、unique max/ties、zero、非32整除宽度及 H128；检查 replicated B32 D 与原 B32 forward 完全一致、cache parity。通过才允许训练。

新增一组最终 update80 的 calibration80、原 held-out40×2 seeds=80 queries。复用 baseline、clip_only80、离散 B32、未耦合连续 B32、coupled B32 的800条端点，总共960条；新增 teacher=0。不新增 development set，不根据 held-out 选择 checkpoint。

主判据：held-out first10 motor RMSE 和 future latent NMSE 均优于 coupled B32，并保留相对 baseline/clip_only 的改善。逐 suite 和 seed-averaged case 胜负必须报告，特别检查 Spatial 的取舍；late22/full32/gripper 仅诊断。

GPU walltime01:59:00，软截止6840秒；GPU总预算不超过2h，不自动续提。每15秒向 job log 写 GPU utilization/VRAM，gpt-6-luna/xhigh subagent 每10分钟监控，无实质变化时静默。计算/重 I/O 仅在真实 PBS compute allocation 内进行，login 仅轻量控制。

本实验是固定输入 decoded-BF16 VQ + fake A4 数值评估。原 held-out 已多次用于工程比较；没有 native kernel、closed-loop、统计显著性或未见任务泛化声明。

## 已完成

GPU `25862598.pbs101`：F/Exit0，01:17:42；CPU aggregate `25863316.pbs101`：F/Exit0，00:00:30。完成80 updates、新方法 calibration80+原 held-out80，复用五组对照800条端点，共960条；新增teacher=0。两阶段均通过 terminal/exit0、artifact exit0、PHASE_COMPLETE及匹配summary。

Held-out motor=0.031641466、latent=0.055377875；相对 coupled B32分别-27.0847%、+2.4241%，预设双指标门槛未通过。Motor收益在39/40 cases保留；latent问题主要在Goal和LIBERO10。详见[分析](results/ANALYSIS.zh.md)、[完整比较](results/comparison.json)、[结果表](results/RESULTS.zh.md)。监控已结束，没有启动下一轮实验。
