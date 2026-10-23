# Dynamic absmax coupling：结果与下一步

本轮完整补齐 continuous B32 scaling 与 dynamic absmax 的链式项，并完成固定80 updates、原 calibration80 和 held-out80 queries。原 baseline、C80、loss、seeds、共享 teacher、optimizer/LR、bounds 及原 VQ 内容均沿用；没有新的 development set。

## 总体结果

原 held-out 为40 contexts×2 seeds，先在 case 内平均 seeds，再对四 suites 等权。

| 配置 | First10 motor RMSE ↓ | Future latent NMSE ↓ |
|---|---:|---:|
| baseline | 0.044296774 | 0.060877371 |
| clip_only80 | 0.043848437 | 0.054848124 |
| 离散 B32 + clip80 | 0.043986743 | 0.054151555 |
| 连续 B32，缺 absmax coupling | 0.044588993 | 0.060690613 |
| 连续 B32，补齐 coupling | **0.043394818** | **0.054067248** |

- vs baseline：motor −2.0362%，latent −11.1866%。
- vs clip_only：motor −1.0345%，latent −1.4237%。
- vs 离散 B32：motor −1.3457%，latent −0.1557%。
- vs 缺 coupling 的连续版本：motor −2.6782%，latent −10.9133%。

预设的“两项均优于 baseline 和 clip_only”门槛通过；补齐后相对未耦合连续版本两项也均改善。在本次固定 cohort 的总体均值上，这是五组中两项同时最低的候选。

## 收益并不均匀

| Held-out suite：vs clip_only | Motor 变化 | Latent 变化 |
|---|---:|---:|
| libero_spatial | +4.2959% | +19.4858% |
| libero_object | +0.5087% | −17.6788% |
| libero_goal | −1.1622% | −2.2418% |
| libero_10 | −7.7449% | +1.1686% |

总体 motor 收益较多来自 libero_10，latent 收益较多来自 Object。只有 Goal 两项都优于 clip_only；Spatial 两项均退步。因此不能把总体门槛通过表述成四 suites 全面改善。

相对 clip_only，case 内平均两 seeds 后，motor 22/40 cases 改善、18/40变差；latent 19/40改善、21/40变差。相对未耦合连续版本，两项各23/40 cases 改善。80 queries 不当作80个独立 contexts。

Held-out gripper RMSE 相对 clip_only 高9.3313%，完整指标保留在 comparison.json。gripper、late22、full32 仅诊断，未加入 loss 或改变本轮主门槛。

## Calibration 并没有同步变好

| 配置 | Calibration motor RMSE ↓ | Calibration latent NMSE ↓ |
|---|---:|---:|
| baseline | 0.044146961 | 0.054735372 |
| clip_only80 | 0.041858108 | 0.047223291 |
| 未耦合连续 B32 | 0.043243457 | 0.048459245 |
| 已耦合连续 B32 | 0.042250882 | 0.054565659 |

相对未耦合版本，calibration motor 改善2.2953%，latent 变差12.6011%。相对 clip_only，calibration motor 高0.9383%、latent 高15.5482%，mixed loss 高9.1410%；held-out mixed loss 则低1.3636%。

所以不能把本轮解释成“更好地拟合训练集，因而 held-out 更好”，也不能说已证明解决过拟合。修正改变了梯度流与误差分配，当前不同 suites、不同 cohorts 响应不同。单次训练及反复使用的固定 cohort 不足以确定这些响应在其他训练或数据上是否稳定。

## 已验证修正的含义

Forward 不变；只补 backward 中 max 的链式项。对于 `v=Z/D`，`m=amax(abs(v))`，`s=m*r/7`，m 的变化会联动该 token 全部 groups 的量化尺度。

- D 的逐 channel 原始项现在减去 token 总 scale 项在实际 max channels 上的贡献。
- 输入梯度也加上同一 max 项，随后走 D^-1、inverse H128/Smooth，让更上游的 D 得到完整链。
- 正负并列最大按原 amax 的规则均分；零输入的 signed max support 为0。
- 原 LSQ parameter normalization、group mean projection、BF16 derivative coefficient/stored-weight approximation 均保留。

GPU selfcheck 用 live-amax 的独立 autograd STE graph 验证 unique max、跨 groups 的 ties 和 B32 tail。输入梯度最大误差为0；normalized log2-D 梯度最大误差1.192093e-7；同一 fixture 明确拒绝旧缺项 input/log2-D 公式。H128 完整 backward 对照的 input/ratio/log2-D 误差均为0。新旧 forward 及 cached weight forward 均逐位一致。

这些证据说明：在约定的 STE 下，确实修正了数学缺项；本轮 held-out 的改善支持该缺项影响了优化。但这不是硬量化/BF16 rounding 的精确导数证明，也不能把全部性能变化归因于 D 的直接梯度，因为输入梯度同时被修正。

原 VQ codebooks、indices、row gains、integer exponents 均冻结。D=1、ratio=1 的原 baseline 完整 action parity=0；最终 uncached/cached 完整 action/future query max_abs 均为0。BF16 compensation 还原至原坐标后，相对原 decoded BF16 baseline 的新增 weight rounding NMSE=2.747841e-6，与未耦合版本2.745702e-6相近；该量并非完整精度 weight 的总 VQ 误差，不能单凭相近就排除端到端影响。

最终56,466个 groups 中43,951个偏离D=1超过1%，21,776个超过5%；D范围0.574489至1.403952。FP32 log2D 数值 metadata 为225,864 bytes（约220.57 KiB，不含序列化/键名开销）。Clipping ratio 范围[0.5,1]，均值0.954693；至少一个 ratio 达到原下界，没有据此改 bounds。

## 下一步：先归因，再增粒度

建议保留已耦合 B32 作为当前总体最好的候选，同时保留 clip_only 作对照。暂不直接增加约32倍的 per-channel scaling 参数。

更有信息量的一组对照是：**D固定为1，只训练614个 clipping ratios，但输入 STE 同样包含完整 absmax 链**。仍从原 baseline、ratio=1开始，用原 C80、80 updates、原 loss/seeds/teacher，完整评估原80个 held-out queries；复用现有对照。

原因：本轮既修正 D 的直接项，也修正传播到上游的 input 项。后一项即使没有 learnable D，也可能改善 clipping 的训练。当前不能判断额外225,864 bytes的 continuous scales 是否必要。

- 如果 coupled-input clip-only 达到相近或更好的两项结果，可以保留更简单的 ratio-only 方案。
- 如果 coupled B32 仍提供额外收益，再考虑受约束的 per-channel residual，并优先核对 Spatial 的回退。

这个归因对照可以避免把“量化器梯度改好了”的收益误判为“需要更多 channel 参数”。本轮未启动该对照或 per-channel 训练。

## 完成记录

GPU `25861303.pbs101`：01:14:26；CPU aggregate `25861881.pbs101`：00:00:27。两者均通过 scheduler F/exit0、artifact exit0、PHASE_COMPLETE、matching complete summary。

完成80 updates、160条新方法端点；复用四组640条端点，共800条。原 held-out40×2完整覆盖，新增 teacher=0。3个 parity 用途 queries 中1个复用为 calibration endpoint，因此新增独立 parity queries 为2；endpoint/parity合计162次新 forward，另有80次 training forward/backward（含 checkpoint 内部重算）。GPU utilization/VRAM 每15秒记入 job log，gpt-6-luna 每10分钟监控，现已结束。

这是单次训练、固定输入、decoded-BF16 VQ + fake A4 的工程比较；不代表 native kernel 性能、closed-loop success、统计显著性或未见任务泛化。
