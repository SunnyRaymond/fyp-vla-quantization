# AQLM 两臂 fixed-input 消融

官方 input-aware AQLM 初始化共用；仅 loss 不同，indices 与底模冻结。
Control 为官方 Linear-output reconstruction objective 的 Fast-WAM adaptation，不是 block fine-tuning。
含共同 preprocessing 的 target-weight effective bpw：4.031080。

共享初始化来源：{'heavy_v1_retained': 22, 'light_v2': 12, 'minimal_v3': 580}；既有各预算 banks 原样保留，其余采用用户指定 minimal_v3 预算。

| Held-out 指标 | Reconstruction | Ours | 相对变化 | Context 胜/负/平 |
|---|---:|---:|---:|---|
| first10motor_rmse | 0.051755376 | 0.11687385 | +125.82% | 0/40/0 |
| full32motor_rmse | 0.054076112 | 0.11570391 | +113.96% | 0/40/0 |
| late22motor_rmse | 0.05465307 | 0.1129103 | +106.59% | 0/40/0 |
| first10gripper_rmse | 0.035122972 | 0.12769436 | +263.56% | 0/40/0 |
| future_latent_nmse | 0.069989021 | 0.23959217 | +242.33% | 0/40/0 |
| mixed_loss | 0.0057627445 | 0.025813018 | +347.93% | 0/40/0 |

Held-out：40 contexts × 2 seeds；先在 context 内平均 seeds，再按四个 suites 等权汇总。
表中 mixed_loss = action MSE + lambda × future NMSE，不含 reader KL；两臂按同一指标比较。
每臂另外完整报告 C80 calibration。固定 final update8，无 held-out checkpoint selection。
闭环 episodes = 0；停止在 280-episode 闭环之前。结果不代表 native-kernel latency 或 closed-loop success。
