# LIBERO-Plus 280（engine）：`H1B_A8_KL`
完成 280/280，成功 **210**。policy call p50 0.836 s（6845 次），episode 平均 88.9 s。
| 臂 | 成功/280 |
|---|---:|
| **H1B_A8_KL** | **210** |
| bf16 | 216 |
| reader_kl | 208 |
| az | 204 |
| H_W16A4 | 204 |
| C80_R128_W16A4 | 203 |
| spinquant_adapted_w4a4 | 203 |
| quarot_adapted_w4a4 | 195 |
| smooth05_hadamard_scalar_w4a4 | 200 |
| viditq_w4a4 | 203 |

| Suite（每组 70） | H1B_A8_KL | bf16 | reader_kl | az | H_W16A4 | C80_R128_W16A4 | spinquant_adapted_w4a4 | quarot_adapted_w4a4 | smooth05_hadamard_scalar_w4a4 | viditq_w4a4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| libero_10 | 50 | 54 | 51 | 52 | 50 | 48 | 49 | 47 | 48 | 52 |
| libero_goal | 48 | 51 | 47 | 44 | 51 | 46 | 48 | 45 | 47 | 48 |
| libero_object | 57 | 55 | 55 | 55 | 50 | 54 | 51 | 52 | 52 | 49 |
| libero_spatial | 55 | 56 | 55 | 53 | 53 | 55 | 55 | 51 | 53 | 54 |

| 扰动（每组 40） | H1B_A8_KL | bf16 | reader_kl | az | H_W16A4 | C80_R128_W16A4 | spinquant_adapted_w4a4 | quarot_adapted_w4a4 | smooth05_hadamard_scalar_w4a4 | viditq_w4a4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| background_textures | 25 | 26 | 26 | 24 | 25 | 24 | 23 | 24 | 26 | 26 |
| camera_viewpoints | 18 | 21 | 19 | 14 | 14 | 19 | 16 | 12 | 14 | 14 |
| language_instructions | 35 | 38 | 39 | 38 | 38 | 35 | 36 | 36 | 37 | 37 |
| light_conditions | 38 | 35 | 35 | 36 | 37 | 35 | 39 | 36 | 35 | 37 |
| objects_layout | 35 | 35 | 34 | 35 | 35 | 36 | 33 | 36 | 34 | 34 |
| robot_initial_states | 33 | 34 | 31 | 33 | 30 | 31 | 31 | 30 | 30 | 32 |
| sensor_noise | 26 | 27 | 24 | 24 | 25 | 23 | 25 | 21 | 24 | 23 |

| 配对对象 | 胜 | 负 | 平 | 共同 slot |
|---|---:|---:|---:|---:|
| BF16 | 11 | 17 | 252 | 280 |
| Q0_A4 | 18 | 16 | 246 | 280 |
| AZ_A4 | 18 | 12 | 250 | 280 |
