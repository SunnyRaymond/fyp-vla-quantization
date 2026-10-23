# Activation clipping checkpoint 组合与 C8 端点评估

无重新训练；复用原 shared teacher 和配对 seeds。

| Held-out arm | First10 motor RMSE | Future latent NMSE | Late22 motor RMSE |
|---|---:|---:|---:|
| baseline | 0.044296774 | 0.060877371 | 0.048528504 |
| weight_only | 0.045364196 | 0.062812625 | 0.056802480 |
| clip_only | 0.045434657 | 0.068720982 | 0.059886000 |
| joint | 0.044525844 | 0.061005909 | 0.050145723 |
| joint_w_ratio1 | 0.045520372 | 0.063400208 | 0.052702255 |
| baseline_w_joint_ratios | 0.044939815 | 0.058589487 | 0.049666121 |

每组 held-out 均为40 contexts × 2 seeds=80 queries，四suite等权。

| C8 final checkpoint | Action MSE | Future latent NMSE | Mixed loss |
|---|---:|---:|---:|
| baseline | 0.001883638 | 0.054480982 | 0.004196602 |
| weight_only | 0.001638515 | 0.039012120 | 0.003294757 |
| clip_only | 0.001512912 | 0.036119622 | 0.003046354 |
| joint | 0.001618540 | 0.036117150 | 0.003151877 |

C8每组完整8 contexts、原seed_index=0，每suite两例。Mixed loss沿用rho1.5与冻结normalization_c。
两组新held-out组合160 queries，加四组C8端点32 queries，共192；启动parity query单独记录。
仅fixed-input decoded-BF16 VQ+A4证据；不构成统计显著性、closed-loop或native INT4结论。
