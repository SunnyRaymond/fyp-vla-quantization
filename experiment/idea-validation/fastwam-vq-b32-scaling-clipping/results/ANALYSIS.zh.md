# B32 scaling + clipping80：解释与下一步

本轮从原 baseline 开始训练80 updates。保持原 C80、update order、loss、seeds 与共享 teacher；仅新增 B32 整数 exponent shadows，原 codebooks、indices、row gains 冻结。没有新增 development set 或 teacher，也没有更换或扩大 held-out cohort。

## 原 held-out：40 contexts ×2 seeds

| 配置 | First10 motor RMSE ↓ | Future latent NMSE ↓ |
|---|---:|---:|
| 原 baseline | 0.044296774 | 0.060877371 |
| C80 clip_only80 | 0.043848437 | 0.054848124 |
| B32 scale + clip80 | 0.043986743 | 0.054151555 |

- B32 vs baseline：action −0.6999%，latent −11.0481%。
- B32 vs clip_only：action +0.3154%，latent −1.2700%。
- 相对 clip_only，按 case 内平均两 seeds 后，action 23/40 cases 改善、17/40变差；latent 22/40改善、18/40变差。不是所有 case 一致获益，也没有统计显著性结论。
- 因 action 未改善，预设的“两项同时优于 baseline 和 clip_only”门槛未通过。late22/full32/gripper 不参与此判断。

## Calibration 也没有超过 clipping-only

| 配置 | First10 motor RMSE ↓ | Future latent NMSE ↓ |
|---|---:|---:|
| 原 baseline | 0.044146961 | 0.054735372 |
| C80 clip_only80 | 0.041858108 | 0.047223291 |
| B32 scale + clip80 | 0.043173617 | 0.048167793 |

B32 vs clip_only 的 calibration action +3.1428%、latent +2.0001%。因此，本轮相对 clip_only 的不足并不符合“训练拟合更好而 held-out 更差”的简单过拟合模式；优化或参数化限制是更值得先检查的解释。不能凭一次训练给各原因确定因果占比，也不能说更大的 calibration 一定无效。

## 干预确实很稀疏

614个 Linear 共56,466个输入 groups。最终13个 groups 的 k 非零（6负、7正），涉及3个 Linear，仅占0.0230%；其余仍是 D=1。所有614个模块的 scaling 梯度连接且有限，612个有非零信号；单一 group 的均匀缩放对 dynamic per-token A4 冗余，层内中心化会使其梯度为0。

本轮 forward 的 D 仅取0.5、1、2。它不能表达1.05或0.95之类的小幅调整，shadow 必须跨过取整阈值才改变实际 forward；跨阈值时则直接产生2倍的尺度变化。这些事实提示离散步长可能限制了探索，但尚未证明它是唯一原因。shadow backward 是将 absmax 视为常数的 LSQ/STE surrogate；保留这一局限，不把本轮视为连续 channel scaling 的充分测试。

工程路径已验证：D=1/r=1 的原 baseline action parity=0；最终 factorized 与实际 folded B32 decode 的完整 action/future query max_abs 均为0。codebooks、indices、row gains 未改变。因此，没有发现 exponent 折叠造成训练/端点评估不一致的证据。

## 建议的下一步

暂不直接增加到全量 per-channel。先做连续 B32 + clipping，先改变尺度分辨率、保持 group granularity，检验“比2倍更细的调整是否有用”。从 D=1/ratio=1 的原 baseline 重新开始，并保持同一 C80、80 updates、loss、seeds、teacher 及完整原 held-out80；复用已有对照，不加入 late22 loss，不按 held-out 选 checkpoint。先不同时增加 shifting、rotation、可训练 codebooks 或新的 loss 项。

连续 D 不能直接精确折入现有整数 B32 exponents。若允许该数值诊断，应显式保存 continuous group scales，并把它们的存储与执行成本计入方案；若要求编码格式完全不扩展，就需要另一种离散搜索方案，不能把连续训练后取整当作相同方法。任意连续 D 还可能改变 BF16 rounding，本轮 power-of-two factorization 的精确等价不能自动沿用；下一轮必须让训练和正式评估使用一致的实际补偿路径。

只有连续 B32 已在两项主指标上显示额外收益、且多数 group 能参与有效调整时，才进一步测试更细 granularity 或受约束的 per-channel。这个顺序先检验步长，再检验粒度，便于解释失败原因。本轮未启动这些后续实验。

## 完成与证据范围

GPU `25855238.pbs101`：01:27:00；CPU aggregate `25856260.pbs101`：00:00:21。80 updates、160条新方法端点、320条复用对照，合计480条端点记录。新 teacher=0；parity 用途的3个 queries 中，folded query 复用为一个 calibration endpoint，因此单独增加2个 parity queries，实际新 endpoint/parity forwards 合计162，另有80次 training forward/backward。GPU utilization/VRAM 已写入 job log，gpt-6-luna 每10分钟监控并已结束。

这是固定输入、decoded-BF16 VQ + fake A4 的单次训练工程比较。没有运行连续 scaling、native kernel 或 closed-loop，也未重新验证未见任务泛化；0.315%的 action 差异不应被表述为统计显著的退步。
