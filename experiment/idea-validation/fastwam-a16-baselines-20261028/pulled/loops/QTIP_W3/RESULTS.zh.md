# LIBERO-Plus 280：`QTIP_W3`（qtip W3，A16，3.016 bpw）

完成 280/280，成功 **202**。权重误差代理（加权）0.00143。policy call p50 0.505 s，episode 平均 78.0 s。

| 臂 | 成功/280 |
|---|---:|
| **QTIP_W3** | **202** |
| bf16 | 216 |
| reader_kl | 208 |
| spinquant_adapted_w4a4 | 203 |
| quarot_adapted_w4a4 | 195 |
| viditq_w4a4 | 203 |

| 扰动（每组 40） | QTIP_W3 | bf16 | reader_kl | spinquant_adapted_w4a4 | quarot_adapted_w4a4 | viditq_w4a4 |
|---|---:|---:|---:|---:|---:|---:|
| background_textures | 26 | 26 | 26 | 23 | 24 | 26 |
| camera_viewpoints | 15 | 21 | 19 | 16 | 12 | 14 |
| language_instructions | 37 | 38 | 39 | 36 | 36 | 37 |
| light_conditions | 37 | 35 | 35 | 39 | 36 | 37 |
| objects_layout | 34 | 35 | 34 | 33 | 36 | 34 |
| robot_initial_states | 28 | 34 | 31 | 31 | 30 | 32 |
| sensor_noise | 25 | 27 | 24 | 25 | 21 | 23 |

| 配对对象 | 胜 | 负 | 平 |
|---|---:|---:|---:|
| BF16 | 4 | 18 | 258 |
| Q0_A4 | 14 | 20 | 246 |
