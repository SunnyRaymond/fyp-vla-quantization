# LIBERO-Plus 280：`QUIPSHARP_LRJ_W2`（quipsharp W2，A16，2.052 bpw）

完成 280/280，成功 **202**。权重误差代理（加权）0.00767。policy call p50 0.742 s，episode 平均 88.8 s。

| 臂 | 成功/280 |
|---|---:|
| **QUIPSHARP_LRJ_W2** | **202** |
| bf16 | 216 |
| reader_kl | 208 |
| spinquant_adapted_w4a4 | 203 |
| quarot_adapted_w4a4 | 195 |
| viditq_w4a4 | 203 |

| 扰动（每组 40） | QUIPSHARP_LRJ_W2 | bf16 | reader_kl | spinquant_adapted_w4a4 | quarot_adapted_w4a4 | viditq_w4a4 |
|---|---:|---:|---:|---:|---:|---:|
| background_textures | 25 | 26 | 26 | 23 | 24 | 26 |
| camera_viewpoints | 15 | 21 | 19 | 16 | 12 | 14 |
| language_instructions | 37 | 38 | 39 | 36 | 36 | 37 |
| light_conditions | 37 | 35 | 35 | 39 | 36 | 37 |
| objects_layout | 34 | 35 | 34 | 33 | 36 | 34 |
| robot_initial_states | 31 | 34 | 31 | 31 | 30 | 32 |
| sensor_noise | 23 | 27 | 24 | 25 | 21 | 23 |

| 配对对象 | 胜 | 负 | 平 |
|---|---:|---:|---:|
| BF16 | 7 | 21 | 252 |
| Q0_A4 | 13 | 19 | 248 |
