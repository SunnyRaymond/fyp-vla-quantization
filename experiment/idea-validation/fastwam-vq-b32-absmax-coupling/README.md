# Continuous B32 + clipping：dynamic absmax gradient coupling

目标：检验上一轮连续 B32 scaling 的 backward 缺少 dynamic absmax 链式项是否限制了优化。复用 baseline、clip_only80、离散 B32+clip80、未耦合连续 B32+clip80；只新增本轮一组训练与完整评估。

从原 C8 rho1.5 baseline 的 D=1、ratio=1 开始，固定80 updates。原 C80/update order、loss、seeds、共享 teacher、group size32、log2D∈[-1,1]、ratio LR=.01、log2D LR=.02、Adam 首次更新后清 state 和 gradient norm clip=1 均保持不变。冻结 codebooks、indices、row gains、原 B32 exponents、Smooth0.5/H128、bias 和其他模型参数。late22/full32/gripper 不进入 loss 或推进门槛。

Forward 完全保持 `F.linear(Q_A4(Z/D;ratio), BF16(Wq*D), bias)`；仍单独保存 FP32 log2D，记录实际 BF16 weight compensation rounding，并验证 uncached/cached 完整 action/future query parity。

## 本轮唯一干预：补齐 absmax 的链式项

设 `v=Z/D`，`m=amax(abs(v))`，`s=m*r/7`，`u=v/s`。令 `b=inside?(q-u):q`，保留原 BF16 operand-precision 系数近似。有效输入 adjoint 为 `g=G @ BF16(Wq*D)`。

`h=sign(v)*1(abs(v)=m)/ties` 是 m 对 v 的导数；并列最大按原 `amax` 等分，零输入时 h=0。

- ratio 梯度仍为 `sum(g*b*m/7)/sqrt(numel*7)`。
- 新增 `dm=sum_channels(g*b)*r/7`，输入梯度改为 `(g*inside+dm*h)/D`，之后走 inverse H128/Smooth。该项让上游 D 也能收到后续层 absmax 的梯度。
- 原 log2D 的逐 channel 项为 `a=g*b*(m/7)*r*ln2`。新增修正 `a -= sum_channels(a)*abs(h)`：将全 token scale 的联动归给实际决定最大值的 groups，再沿用原 group 聚合、sqrt(rows×actual_groupwidth×7) 归一化及层内中心化。

旧的层内减均值没有表达这个逐 token 的 max-group 归属；本轮保留归一化和中心化，但不再用它替代 max 链式项。仍是 stored-weight LSQ/STE surrogate，不声称硬 round 或 BF16 rounding 的精确导数。

## 独立验证与评估

GPU allocation 内，在加载模型前执行新 selfcheck：使用 live `amax` 的独立 autograd STE reference 核对参数及输入链，覆盖 unique max、跨 groups 的正负 ties、zero、B32 tail 与 H128。构造缺项旧公式不能通过的 fixture，避免只检查“实现等于自身公式”。同时要求新旧 forward 和 cached forward 一致。该测试通过才允许80次训练。

新方法评估原 C80的80个 seed0 queries，以及原40 held-out contexts×2 seeds=80 queries。复用四组对照的640条端点，合计800条，不增加 teacher、development set 或更换 cohort。报告相对未耦合版本的两项改善，推进门槛仍要求两项同时优于 baseline 与 clip_only80，不根据 held-out 选择 checkpoint。

GPU walltime01:59:00、软截止6840秒；每15秒向 job log 写 utilization/VRAM。gpt-6-luna/xhigh 每10分钟监控，Q/R 无实质变化时静默。所有 Torch/模型工作只在真实 PBS compute allocation，login 只做轻量控制。结果出来再判断下一步，不自动扩展 per-channel。

固定 cohort 已多次用于工程比较。本轮证据范围为 decoded-BF16 VQ + fake A4 的固定输入数值结果，不代表统计显著性、native kernel、closed-loop 或未见任务泛化。

## 已完成

GPU `25861303.pbs101` 用时01:14:26；CPU aggregate `25861881.pbs101` 用时00:00:27。两者通过 scheduler terminal/exit0、artifact exit0、PHASE_COMPLETE、matching complete summary。完成80 updates、新方法 calibration80+原 held-out80、复用640条对照，合计800条端点；新增 teacher=0。gpt-6-luna 监控已结束。

Held-out motor=0.043394818、latent=0.054067248，相对 clip_only 分别改善1.0345%、1.4237%，相对未耦合连续版本改善2.6782%、10.9133%。总体主门槛通过，但 suite 间有取舍。建议先做 D=1 的 coupled-input clipping-only 归因对照，再考虑 per-channel；尚未自动启动。

详见 [结果表](results/RESULTS.zh.md)、[分析与下一步](results/ANALYSIS.zh.md)、`completion_evidence.json`。
