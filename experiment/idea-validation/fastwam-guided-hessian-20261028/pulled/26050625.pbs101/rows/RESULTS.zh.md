# H̃ + 0 bit 输出行 scale：QTIP W2

作业 `26050625.pbs101`。H̃ 与第一阶段相同；行权重 d 取第一阶段 A + λZ + βK 探针的逐 token 行权重，s² = √d/‖W_c‖，几何均值 1，限制在 [1/16, 16]，并进 QTIP 的 fp32 SV，不加 bit、不训练。

bpw 2.0159；614/614 层有行权重；按参数加权的预估 guided loss 比 0.730。

## proxy（参数加权）

| 权重 \ Hessian | H_U | H̃ | H̃ + 行权重 |
|---|---:|---:|---:|
| H_U（baseline QTIP_W2） | 0.00583 | 0.00770 | 0.00721 |
| H̃（第一阶段） | 0.00793 | 0.00378 | 0.00359 |
| H̃ + 行 scale（本阶段） | 0.01380 | 0.00641 | 0.00303 |

## held-out 40×2

| 臂 | first10 motor | future NMSE | late22 motor | gripper RMSE | 换回 BF16 future 后 motor | mixed loss |
|---|---:|---:|---:|---:|---:|---:|
| H_U（baseline QTIP_W2） | 0.0324 | 0.0532 | 0.0360 | 0.0860 | 0.0246 | 0.00336 |
| H̃（第一阶段） | 0.0332 | 0.0623 | 0.0400 | 0.0876 | 0.0239 | 0.00382 |
| H̃ + 行 scale（本阶段） | 0.0324 | 0.0569 | 0.0376 | 0.0740 | 0.0249 | 0.00352 |

## LIBERO-Plus 280 闭环（BF16 216）

| 臂 | 成功 | background_textures | camera_viewpoints | language_instructions | light_conditions | objects_layout | robot_initial_states | sensor_noise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H_U（baseline QTIP_W2） | 204/280 | 25 | 18 | 37 | 36 | 35 | 29 | 24 |
| H̃（第一阶段） | 208/280 | 29 | 14 | 38 | 36 | 34 | 33 | 24 |
| H̃ + 行 scale（本阶段） | 204/280 | 25 | 15 | 37 | 37 | 35 | 31 | 24 |

本臂对 H_U（baseline QTIP_W2）：8 胜 8 负，McNemar p = 1.00。

本臂对 H̃（第一阶段）：8 胜 12 负，McNemar p = 0.50。
