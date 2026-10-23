# Soft-bin / relational KL：280-episode 配对闭环结果

PBS `25886452.pbs101` 正常完成，exit 0，实际 walltime 08:29:40；四个 worker 各 140/140，两组共 560 个新 episodes，13 组 frozen references 共 3,640 条。gpt-6-luna / xhigh subagent 按 30 分钟间隔监控，现已停止。

两组直接复用源对话的 final80 checkpoint，没有新增训练。四个 worker 的源完整 action/future 重放、uncached/cached parity、正式 infer 路径检查全部通过，最大绝对差均为 0；每次覆盖 614 个目标 Linears。

| 配置 | 成功 / episodes | 成功率 |
|---|---:|---:|
| Soft-bin KL | 199/280 | 71.07% |
| Soft-bin + relational KL | 198/280 | 70.71% |
| Per-channel MSE（原对照） | 203/280 | 72.50% |
| 原 C8 rho1.5 | 206/280 | 73.57% |
| C80 clip-only | 207/280 | 73.93% |
| SpinQuant adapted W4A4 | 203/280 | 72.50% |
| Microscaling VQ A4 | 204/280 | 72.86% |
| BF16 | 216/280 | 77.14% |

Relational 相对 soft-bin 为 9 wins、10 losses、261 ties，净少 1 个成功；相对 per-channel MSE 为 10 wins、15 losses、255 ties，净少 5 个成功。Soft-bin 相对 per-channel MSE 为 10 wins、14 losses、256 ties，净少 4 个成功。这些是本组固定 episodes 的观察差异，不作统计显著性结论。

| Suite | Soft-bin | Relational | Per-channel MSE | BF16 |
|---|---:|---:|---:|---:|
| libero_10 | 47/70 | 46/70 | 47/70 | 54/70 |
| libero_goal | 48/70 | 47/70 | 46/70 | 51/70 |
| libero_object | 52/70 | 54/70 | 56/70 | 55/70 |
| libero_spatial | 52/70 | 51/70 | 54/70 | 56/70 |

| 扰动维度 | Soft-bin | Relational | Per-channel MSE | BF16 |
|---|---:|---:|---:|---:|
| background_textures | 26/40 | 23/40 | 24/40 | 26/40 |
| camera_viewpoints | 14/40 | 16/40 | 15/40 | 21/40 |
| language_instructions | 37/40 | 37/40 | 38/40 | 38/40 |
| light_conditions | 36/40 | 37/40 | 37/40 | 35/40 |
| objects_layout | 34/40 | 34/40 | 34/40 | 35/40 |
| robot_initial_states | 28/40 | 30/40 | 29/40 | 34/40 |
| sensor_noise | 24/40 | 21/40 | 26/40 | 27/40 |

全部 suite × dimension、逐配置 paired wins/losses/ties 和分实现 timing 见 `summary.json`。完成证据见 `FINAL_COMPLETION_EVIDENCE.json`。

执行边界：decoded BF16 VQ weights + dynamic A4 + BF16 KV，compensated weights 在每个 worker 缓存一次；结果不代表 native W4A4 kernel 的速度或显存。源 fixed-input RMSE/KL 与本次 closed-loop success rate 保持独立口径。
