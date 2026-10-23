# Rollout C80 joint held-out comparison

rollout_joint 使用 anchor 原 teacher 与 visited-state teacher 训练80次；评估仅为原40 held-out contexts × 2 seeds。
所有指标为 fixed-input decoded-BF16 VQ + A4；不表示 closed-loop 或 native-kernel 性能。

| Arm | first10motor_mse | first10motor_rmse | full32motor_rmse | late22motor_rmse | first10gripper_rmse | future_latent_nmse | mixed_loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.002014132 | 0.044296774 | 0.047500980 | 0.048528504 | 0.057054031 | 0.060877371 | 0.004598652 |
| clip_only | 0.001977701 | 0.043848437 | 0.050111666 | 0.051734803 | 0.059316845 | 0.054848124 | 0.004306253 |
| joint_c80 | 0.001794784 | 0.041846837 | 0.048101473 | 0.049807107 | 0.044249685 | 0.048973336 | 0.003873924 |
| rollout_joint | 0.001906974 | 0.042925590 | 0.047025115 | 0.048305113 | 0.035083380 | 0.053785088 | 0.004190395 |

rollout_joint 相对原 C80 arms 的百分比变化（负数表示误差降低）：

- joint_c80: first10motor_mse +6.251%, first10motor_rmse +2.578%, full32motor_rmse -2.238%, late22motor_rmse -3.016%, first10gripper_rmse -20.715%, future_latent_nmse +9.825%, mixed_loss +8.169%。
- clip_only: first10motor_mse -3.576%, first10motor_rmse -2.105%, full32motor_rmse -6.159%, late22motor_rmse -6.629%, first10gripper_rmse -40.854%, future_latent_nmse -1.938%, mixed_loss -2.690%。
- baseline: first10motor_mse -5.320%, first10motor_rmse -3.095%, full32motor_rmse -1.002%, late22motor_rmse -0.460%, first10gripper_rmse -38.508%, future_latent_nmse -11.650%, mixed_loss -8.878%。

逐 case 先平均两个 seed 后的配对计数：

- vs joint_c80, first10motor_rmse: 17 improved / 23 worsened / 0 tied (40 cases)。
- vs joint_c80, future_latent_nmse: 14 improved / 26 worsened / 0 tied (40 cases)。
- vs clip_only, first10motor_rmse: 23 improved / 17 worsened / 0 tied (40 cases)。
- vs clip_only, future_latent_nmse: 22 improved / 18 worsened / 0 tied (40 cases)。
- vs baseline, first10motor_rmse: 27 improved / 13 worsened / 0 tied (40 cases)。
- vs baseline, future_latent_nmse: 27 improved / 13 worsened / 0 tied (40 cases)。
