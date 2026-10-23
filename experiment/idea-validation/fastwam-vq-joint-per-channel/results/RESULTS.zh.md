# Joint per-channel C80 evaluation

新 arm 从 C8 rho1.5 update8 初始化，在原 C80、原 loss、teacher 与 seed 上联合优化 VQ books、log row gains、614 个 Linear 的 per-channel D 向量与 ratios。
没有沿用已训练 per-channel checkpoint；held-out 为 40 contexts × 2 paired seeds。4 组旧端点共 640 records，新 arm 160 records，总计 800。

| Held-out arm | First10 motor RMSE | Future latent NMSE |
|---|---:|---:|
| baseline | 0.044296774 | 0.060877371 |
| clip_only | 0.043848437 | 0.054848124 |
| joint_c80 | 0.041846837 | 0.048973336 |
| per_channel_clip | 0.031641466 | 0.055377875 |
| joint_per_channel | 0.032668161 | 0.048842217 |

joint_per_channel 相对 per_channel_clip 与 joint_c80 的变化（负值表示数值降低）：

- vs per_channel_clip: first10motor_rmse +3.245%, full32motor_rmse -4.370%, late22motor_rmse -6.245%, first10gripper_rmse +2.781%, future_latent_nmse -11.802%, first10motor_mse +5.091%, mixed_loss -6.484%
  - first10motor_rmse: 15 improved / 25 worsened / 0 tied across 40 cases after seed averaging.
  - future_latent_nmse: 23 improved / 17 worsened / 0 tied across 40 cases after seed averaging.
- vs joint_c80: first10motor_rmse -21.934%, full32motor_rmse -27.808%, late22motor_rmse -29.324%, first10gripper_rmse +81.890%, future_latent_nmse -0.268%, first10motor_mse -36.758%, mixed_loss -17.173%
  - first10motor_rmse: 38 improved / 2 worsened / 0 tied across 40 cases after seed averaging.
  - future_latent_nmse: 17 improved / 23 worsened / 0 tied across 40 cases after seed averaging.

Held-out 四个 suite 的 primary metric 差异：
- vs per_channel_clip, libero_spatial: first10motor_rmse +5.515%，future_latent_nmse -27.609%。
- vs per_channel_clip, libero_object: first10motor_rmse +14.522%，future_latent_nmse +21.282%。
- vs per_channel_clip, libero_goal: first10motor_rmse -12.839%，future_latent_nmse -31.518%。
- vs per_channel_clip, libero_10: first10motor_rmse +9.848%，future_latent_nmse -1.620%。
- vs joint_c80, libero_spatial: first10motor_rmse -18.955%，future_latent_nmse +20.552%。
- vs joint_c80, libero_object: first10motor_rmse -29.752%，future_latent_nmse -5.261%。
- vs joint_c80, libero_goal: first10motor_rmse -23.474%，future_latent_nmse -22.784%。
- vs joint_c80, libero_10: first10motor_rmse -15.402%，future_latent_nmse +25.774%。

每个 suite 的全指标、Calibration 差异与指标绝对值一并写入 comparison.json。Late22/full32/gripper 为诊断项。
结果范围是 fixed-input decoded-BF16 VQ + fake A4 数值诊断，不代表 closed-loop 或 native-kernel 性能。
