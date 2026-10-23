# C80 per-channel + clipping：LIBERO-Plus 闭环评测

状态：complete；新增 episode：280/280；复用 frozen references：2800/2800。

本轮直接复用源目录中完成的 C80 update 80 deployment，未重新训练。评测采用 decoded BF16 VQ、per-token A4 activations、BF16 KV cache。

Setup gate 检查源 endpoint 的 7 项 scalar metrics，并验证 uncached/cached source action/future parity 和实际 infer action parity。源结果没有保存最终 action vector，因此这不是 saved-action replay。

| Arm | Successes / episodes | Success rate |
|---|---:|---:|
| bf16 | 216/280 | 77.14% |
| quarot_adapted_w4a4 | 195/280 | 69.64% |
| spinquant_adapted_w4a4 | 203/280 | 72.50% |
| smooth05_hadamard_scalar_w4a4 | 200/280 | 71.43% |
| row_scale_vq_a4 | 202/280 | 72.14% |
| micro_scale_vq_a4 | 204/280 | 72.86% |
| rho0.1 | 201/280 | 71.79% |
| rho1.5 | 206/280 | 73.57% |
| clip_only | 207/280 | 73.93% |
| joint | 202/280 | 72.14% |
| per_channel_clip | 203/280 | 72.50% |

Paired win/loss/tie 及 suite、dimension、suite×dimension 统计见 summary.json。
Timing 按 batched_h128_v2 单独报告；它是 decoded-BF16 实现证据，不代表 native low-bit kernel latency。

## 结果解读

新增 per-channel + clipping 为 **203/280（72.50%）**，未超过 C80 clip-only（207/280）或 C8 rho1.5（206/280）。与 SpinQuant 总成功数相同，配对案例则为 15 wins / 15 losses / 250 ties；不能视为相同成功集合。相对 clip-only 为 11 wins / 15 losses / 254 ties，净少 4 个成功。

| Suite | Per-channel | C80 clip-only | C8 rho1.5 | BF16 |
|---|---:|---:|---:|---:|
| libero_10 | 47/70 | 52/70 | 53/70 | 54/70 |
| libero_goal | 46/70 | 50/70 | 48/70 | 51/70 |
| libero_object | 56/70 | 54/70 | 54/70 | 55/70 |
| libero_spatial | 54/70 | 51/70 | 51/70 | 56/70 |

相对 clip-only，libero_10 少 5 个、libero_goal 少 4 个；libero_object 多 2 个、libero_spatial 多 3 个。

| Perturbation | Per-channel /40 | Clip-only /40 | Paired wins / losses |
|---|---:|---:|---:|
| background_textures | 24 | 26 | 1 / 3 |
| camera_viewpoints | 15 | 17 | 1 / 3 |
| language_instructions | 38 | 37 | 1 / 0 |
| light_conditions | 37 | 37 | 2 / 2 |
| objects_layout | 34 | 34 | 0 / 0 |
| robot_initial_states | 29 | 31 | 3 / 5 |
| sensor_noise | 26 | 25 | 3 / 2 |

实际 PBS walltime **04:22:06**；仅加载 4 次模型，不重新训练。启动时额外 12 次不计入 episode timing 的检查 query 全部通过，source endpoint 指标复现和 action/future/infer parity 最大绝对误差均为 0。

排除每进程首次正式 query 后，共 7099 次 action queries：full query wall mean 4905.08 ms、median 4868.93 ms。峰值 allocated 28.4540 GB；reserved 29.4042 GB（decimal GB）。

原 fixed-input RMSE 的改进在这套 frozen cohort 上没有转化为闭环成功数提升。这里是配对描述性结果，未运行重复 rollout 或显著性检验；不能据此把小幅排名差异解释为稳定优势。

完成证据：FINAL_COMPLETION_EVIDENCE.json；原始已汇总统计：summary.json；实际 PBS worker records 保留在远端 results/evaluations/per_channel_clip。监控已在终端成功后停止。
