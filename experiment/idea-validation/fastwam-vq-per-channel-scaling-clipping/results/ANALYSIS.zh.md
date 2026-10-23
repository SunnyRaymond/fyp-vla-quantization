# Per-channel scaling + clipping：完成结果与判断

本轮 motor 获得明显改善，但没有通过预设的双指标推进门槛。相对 coupled B32，原 held-out 的 first10 motor RMSE 降低27.0847%，future latent NMSE 上升2.4241%。因此保留为有价值的 motor 改善候选，暂不替换 coupled B32 作为同时优化 action 与 latent 的方案。

**执行与比较条件**

GPU `25862598.pbs101`：F/Exit0，实际01:17:42，小于2h；CPU aggregate `25863316.pbs101`：F/Exit0，00:00:30。两者均有 artifact exit0、PHASE_COMPLETE、匹配 job ID 的 complete summary。

从原 C8 rho1.5 baseline 初始化 D=1、ratio=1，新增80 updates。原 C80及顺序、loss、seeds、共享 teacher不变；没有从 B32 checkpoint 继续训练，没有 development set、held-out checkpoint selection 或 early stopping。只将完整 coupled scaling 的粒度从 B32 改为 per-channel，其余训练配方沿用。

新方法完成 calibration80、原 held-out40 contexts×2 seeds=80。复用五组对照的800条端点，总计960条；新增 teacher queries=0。先在每个 case 内平均两个 seeds，再按 suite 平均，最后四个 suites 等权。两次 sampler seeds 不是两次独立训练重复。

**原 held-out 完整比较，误差越低越好**

| 配置 | First10 motor RMSE | Future latent NMSE |
|---|---:|---:|
| baseline | 0.044296774 | 0.060877371 |
| clip_only80 | 0.043848437 | 0.054848124 |
| discrete B32 + clipping | 0.043986743 | 0.054151555 |
| uncoupled continuous B32 + clipping | 0.044588993 | 0.060690613 |
| coupled continuous B32 + clipping | 0.043394818 | 0.054067248 |
| **per-channel + clipping** | **0.031641466** | **0.055377875** |

per-channel 相对 baseline：motor -28.5694%、latent -9.0337%；相对 clip_only：motor -27.8390%、latent +0.9659%；相对 coupled B32：motor -27.0847%、latent +2.4241%。

相对 coupled B32，seed-averaged held-out motor 在39/40个 cases改善、1个退化；latent 在18/40个改善、22个退化。Motor 收益分布广，latent 没有显示同样的配对改善。

**Calibration 与 held-out 的关系**

| 指标 | Coupled B32 calibration | Per-channel calibration | Calibration变化 | Held-out变化 |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 0.042250882 | 0.029800742 | -29.47% | -27.08% |
| Future latent NMSE | 0.054565659 | 0.044684950 | -18.11% | +2.42% |
| 原 mixed loss | 0.004143552 | 0.002850323 | -31.21% | -19.22% |

不能把这轮结果概括为“更强拟合全部变成过拟合”。Motor 的 calibration 与 held-out 改善幅度接近，四个 suites 均改善，说明这部分收益在原 held-out 上保留。Latent 则在 calibration 明显改善、held-out 略差于 coupled B32，支持 latent 泛化不足或 calibration 覆盖不够的可能性；单轮固定 cohort 不能区分这些原因，也不能证明是唯一原因。

原目标是 first10 motor MSE + rho×normalization_c×future latent NMSE，仍为固定权重相加。Held-out motor MSE 降低44.67%，latent NMSE 上升2.42%，合成 mixed loss 仍降低19.22%。因此优化加权总和并不保证两项分别改善；本轮既有 latent 的 calibration/held-out 落差，也有实际的目标取舍。仅增加训练步数或调整 rho，均没有现成证据保证恢复 latent。

**各 suite 相对 coupled B32 的变化**

| Suite | Motor变化 | Latent变化 |
|---|---:|---:|
| Spatial | -29.94% | -9.38% |
| Object | -36.93% | -8.96% |
| Goal | -19.24% | +21.87% |
| LIBERO10 | -22.38% | +7.38% |

上一轮重点担心的 Spatial 相对 coupled B32 两项都改善。不过 Spatial latent 相对 clip_only 仍高8.27%，尚未完全收回此前损失。新的 latent 问题主要在 Goal 和 LIBERO10，尤其 Goal；不能只看四个 suites 的总平均。

Gripper 仅为诊断：first10 gripper RMSE=0.078307935，相对 coupled B32 上升20.75%，相对 baseline 上升37.25%。当前 motor loss 只监督前6个 action 维度，gripper 没有直接监督；这一观察与未受约束的输出退化相容，但尚未定位具体模块。它不改变本轮预先定义的两个主指标门槛。Full32/late22 motor 相对 coupled B32 分别改善23.37%/22.75%，同样仅作为诊断。

**实现、表示和成本证据**

- 614个 Linear 的 ratio 与 scale 梯度在全部80次更新均连通且有限。独立 live-amax STE 检查覆盖 D=1/非1、unique max/ties、zero、非32整除宽度及 H128；缺失 absmax 链的旧 surrogate 被区分性 fixture 拒绝。
- 初始 baseline action parity max-abs=0；最终 uncached/cached 完整 action 和 future parity均为0。训练与评估均使用实际 BF16 compensated weights。
- Codebooks、indices、row gains、原 B32 exponents、Smooth/H128及其他模型参数冻结。D 是额外 FP32 metadata，不能精确折进原整数 B32 exponent 表。
- Scale 数从56,466增至1,806,863，约32倍；raw metadata 从225,864 bytes增至7,227,452 bytes，约6.89 MiB。614个 modules 均发生 scale 更新；613,053个 channels 的 |D-1|>5%。D范围约[0.5025,1.7372]；ratio范围[0.7315,1]、均值0.95729。
- BF16 compensation 还原到原坐标、相对原 decoded BF16 Wq 的增量 weight NMSE=2.738762e-6；coupled B32为2.747841e-6。没有观察到整体 compensation rounding error 随 per-channel 显著增大；该数值不是相对原 FP weights 的总 VQ误差，也不能排除局部误差传播。
- 本轮未额外捕获固定输入的逐层 activation/Linear output error。因此可以确认端到端 motor 改善，尚不能把全部收益归因于某一种 activation error 的下降。
- GPU job log 每15秒记录 utilization/VRAM，共310个可解析 samples，无法解析的相关日志行=0。按最近一次 phase 标签粗分，training mean utilization约45.52%，evaluation约33.49%，peak VRAM31,387 MiB。包含模型加载开销的作业用时不是 native inference latency。
- 完整训练 checkpoint与deployment metadata保存在 remote results/train_eval，分别23,246,693和7,728,459 bytes；见 checkpoint_artifacts.json。没有为检查它们而下载大文件。

**下一步建议，尚未执行**

优先保留 per-channel 的 motor 收益，再定位 latent 为什么未泛化。最小的下一组实验是按已存在的 expert 边界限制粒度：`action_expert.*` 使用 per-channel，`video_expert.*` 与其他模块沿用 coupled B32；ratio仍每 Linear一个，完整跨层 absmax/input chain保留。从原 baseline 初始化、80 updates、原 loss/C80/held-out/seeds/teacher不变，与本轮 full per-channel及coupled B32比较。

这能检验“video侧新增自由度是否是 latent 退化的重要来源”，同时观察 motor 收益是否依赖 video侧 per-channel。Experts 存在耦合，因此不能预设 action侧改动只影响 action。若该实验保留 motor 收益并恢复 latent，可继续采用分支粒度；若仍退化，再单独比较扩大 calibration覆盖或对 log2D 加约束，避免同时修改多个因素。

不建议现在把这轮结果简单归为训练过拟合后统一削弱全部 scales：motor 已显示明显的原 held-out 收益，而 latent 问题具有 suite差异。原双指标门槛本轮未通过，固定 cohort 的工程筛选实验则已完整完成；没有自动启动下一轮。

证据范围：fixed-input decoded-BF16 VQ + fake A4，无环境动作执行、native kernel或closed-loop结论。原 held-out 已多次参与方案比较，本轮也只有一条训练轨迹；不据此声称统计显著性或未见任务泛化。
