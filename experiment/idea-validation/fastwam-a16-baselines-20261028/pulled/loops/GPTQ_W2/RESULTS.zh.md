# LIBERO-Plus 280：`GPTQ_W2`（gptq W2，A16，2.006 bpw）

完成 280/280，成功 **15**。权重误差代理（加权）0.04217。policy call p50 0.508 s，episode 平均 142.9 s。

| 臂 | 成功/280 |
|---|---:|
| **GPTQ_W2** | **15** |
| bf16 | 216 |
| reader_kl | 208 |
| spinquant_adapted_w4a4 | 203 |
| quarot_adapted_w4a4 | 195 |
| viditq_w4a4 | 203 |

| 扰动（每组 40） | GPTQ_W2 | bf16 | reader_kl | spinquant_adapted_w4a4 | quarot_adapted_w4a4 | viditq_w4a4 |
|---|---:|---:|---:|---:|---:|---:|
| background_textures | 0 | 26 | 26 | 23 | 24 | 26 |
| camera_viewpoints | 0 | 21 | 19 | 16 | 12 | 14 |
| language_instructions | 4 | 38 | 39 | 36 | 36 | 37 |
| light_conditions | 1 | 35 | 35 | 39 | 36 | 37 |
| objects_layout | 5 | 35 | 34 | 33 | 36 | 34 |
| robot_initial_states | 5 | 34 | 31 | 31 | 30 | 32 |
| sensor_noise | 0 | 27 | 24 | 25 | 21 | 23 |

| 配对对象 | 胜 | 负 | 平 |
|---|---:|---:|---:|
| BF16 | 0 | 201 | 79 |
| Q0_A4 | 0 | 193 | 87 |
