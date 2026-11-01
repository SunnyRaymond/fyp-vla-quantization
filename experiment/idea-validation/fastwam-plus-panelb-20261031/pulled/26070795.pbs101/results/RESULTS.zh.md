# LIBERO-Plus 面板 B + 面板 A：每臂 560 个 episode

面板 B 有效 280/280。配对检验为精确 McNemar（双侧）。

| 臂 | bpw | 面板 A /280 | 面板 B /280 | 合计 /560 | 对 BF16 胜/负（合计） | p |
|---|---:|---:|---:|---:|---:|---:|
| QUIPSHARP_W2 | 2.010 | 198 | 198 | **396** | 19 / 45 | 0.00156 |
| GPTVQ_W2 | 2.063 | 201 | 191 | **392** | 24 / 54 | 0.000901 |
| QTIP_W2 | 2.016 | 204 | 198 | **402** | 25 / 45 | 0.0225 |
| BF16 | 16.000 | 216 | 206 | **422** | — | — |

| 配对 | 面板 A 胜/负 | 面板 B 胜/负 | 合计胜/负 | 合计 p |
|---|---:|---:|---:|---:|
| QUIPSHARP_W2 对 GPTVQ_W2 | 9 / 12 | 22 / 15 | 31 / 27 | 0.694 |
| QUIPSHARP_W2 对 QTIP_W2 | 10 / 16 | 17 / 17 | 27 / 33 | 0.519 |
| GPTVQ_W2 对 QTIP_W2 | 13 / 16 | 17 / 24 | 30 / 40 | 0.282 |

| 扰动（A+B） | QUIPSHARP_W2 | GPTVQ_W2 | QTIP_W2 | BF16 |
|---|---:|---:|---:|---:|
| objects_layout | 67 | 64 | 67 | 71 |
| camera_viewpoints | 32 | 37 | 37 | 46 |
| robot_initial_states | 55 | 53 | 53 | 56 |
| language_instructions | 76 | 74 | 74 | 73 |
| light_conditions | 71 | 72 | 74 | 70 |
| background_textures | 49 | 43 | 51 | 53 |
| sensor_noise | 46 | 49 | 46 | 53 |

| suite（A+B） | QUIPSHARP_W2 | GPTVQ_W2 | QTIP_W2 | BF16 |
|---|---:|---:|---:|---:|
| libero_10 | 96 | 96 | 97 | 106 |
| libero_goal | 82 | 78 | 88 | 94 |
| libero_object | 108 | 110 | 107 | 113 |
| libero_spatial | 110 | 108 | 110 | 109 |

| 臂（面板 B） | policy call p50 | episode 平均 | worker 次数 | 加载耗时 |
|---|---:|---:|---:|---|
| QUIPSHARP_W2 | 0.501 s | 78.6 s | 3 | 134, 822, 870 s |
| GPTVQ_W2 | 0.505 s | 81.1 s | 3 | 101, 846, 817 s |
| QTIP_W2 | 0.502 s | 76.9 s | 5 | 185, 424, 221, 807, 870 s |
| BF16 | 0.502 s | 77.2 s | 5 | 341, 114, 713, 673, 378 s |
