# LIBERO-Plus 280：`QTIP_CA_W2`（qtip W2，A16，2.016 bpw）

完成 280/280，成功 **204**。权重误差代理（加权）0.01330。policy call p50 0.529 s，episode 平均 79.8 s。

| 臂 | 成功/280 |
|---|---:|
| **QTIP_CA_W2** | **204** |
| bf16 | 216 |
| reader_kl | 208 |
| spinquant_adapted_w4a4 | 203 |
| quarot_adapted_w4a4 | 195 |
| viditq_w4a4 | 203 |

| 扰动（每组 40） | QTIP_CA_W2 | bf16 | reader_kl | spinquant_adapted_w4a4 | quarot_adapted_w4a4 | viditq_w4a4 |
|---|---:|---:|---:|---:|---:|---:|
| background_textures | 26 | 26 | 26 | 23 | 24 | 26 |
| camera_viewpoints | 14 | 21 | 19 | 16 | 12 | 14 |
| language_instructions | 39 | 38 | 39 | 36 | 36 | 37 |
| light_conditions | 36 | 35 | 35 | 39 | 36 | 37 |
| objects_layout | 35 | 35 | 34 | 33 | 36 | 34 |
| robot_initial_states | 30 | 34 | 31 | 31 | 30 | 32 |
| sensor_noise | 24 | 27 | 24 | 25 | 21 | 23 |

| 配对对象 | 胜 | 负 | 平 |
|---|---:|---:|---:|
| BF16 | 8 | 20 | 252 |
| Q0_A4 | 10 | 14 | 256 |
