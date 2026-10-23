# C8 rho1.5 activation clipping 四组 held-out 结果

每组40 contexts x 2 seeds = 80 paired queries；四suite等权。无closed-loop episodes。

| Arm | 额外updates | First10 motor RMSE | Full32 motor RMSE | Late22 motor RMSE | Future latent NMSE |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | 0.044296774 | 0.047500980 | 0.048528504 | 0.060877371 |
| weight_only | 8 | 0.045364196 | 0.054219896 | 0.056802480 | 0.062812625 |
| clip_only | 8 | 0.045434657 | 0.056937468 | 0.059886000 | 0.068720982 |
| joint | 8 | 0.044525844 | 0.048785945 | 0.050145723 | 0.061005909 |

相对baseline变化（负值为改善）：

| Arm | First10 motor RMSE变化 | Future latent NMSE变化 |
|---|---:|---:|
| weight_only | +2.410% | +3.179% |
| clip_only | +2.569% | +12.884% |
| joint | +0.517% | +0.211% |

Clipping参数：

| Arm | Min | Mean | Max | Ratios < 1 |
|---|---:|---:|---:|---:|
| baseline | 1.000000 | 1.000000 | 1.000000 | 0/614 |
| weight_only | 1.000000 | 1.000000 | 1.000000 | 0/614 |
| clip_only | 0.945762 | 0.992315 | 1.000000 | 401/614 |
| joint | 0.943342 | 0.993148 | 1.000000 | 435/614 |

预设channel scaling推进gate：False。
Gate仅为固定held-out cohort的工程筛选；没有统计显著性、closed-loop成功率或native INT4 latency结论。
从同一C8 rho1.5 update8 checkpoint初始化；后三组固定额外8updates，未按held-out挑选checkpoint。
