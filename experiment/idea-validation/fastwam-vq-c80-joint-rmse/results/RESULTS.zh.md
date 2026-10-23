# C80 joint checkpoint RMSE evaluation

直接评估已完成的 joint C80 update80 checkpoint，没有在本任务重训。
Calibration 为原80 contexts × 1 seed；held-out 为原40 contexts × 2 seeds；复用原 teacher 与 baseline/clip_only 共320条端点。
所有指标均为 fixed-input decoded-BF16 VQ + A4 证据，不代表闭环或 native kernel 性能。

| Split / arm | first10motor_rmse | full32motor_rmse | late22motor_rmse | first10gripper_rmse | future_latent_nmse | first10motor_mse | mixed_loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| calibration / baseline | 0.044146961 | 0.046424454 | 0.046974373 | 0.041486297 | 0.054735372 | 0.002028089 | 0.004351853 |
| calibration / clip_only | 0.041858108 | 0.044334697 | 0.045167795 | 0.031405831 | 0.047223291 | 0.001791671 | 0.003796513 |
| calibration / joint_c80 | 0.040851498 | 0.042404108 | 0.042874793 | 0.046257211 | 0.043192356 | 0.001716501 | 0.003550212 |
| heldout / baseline | 0.044296774 | 0.047500980 | 0.048528504 | 0.057054031 | 0.060877371 | 0.002014132 | 0.004598652 |
| heldout / clip_only | 0.043848437 | 0.050111666 | 0.051734803 | 0.059316845 | 0.054848124 | 0.001977701 | 0.004306253 |
| heldout / joint_c80 | 0.041846837 | 0.048101473 | 0.049807107 | 0.044249685 | 0.048973336 | 0.001794784 | 0.003873924 |

joint 相对对照变化，负值表示该指标数值降低：

- calibration vs baseline: first10motor_rmse -7.465%, full32motor_rmse -8.660%, late22motor_rmse -8.727%, first10gripper_rmse +11.500%, future_latent_nmse -21.089%, first10motor_mse -15.364%, mixed_loss -18.421%。
- calibration vs clip_only: first10motor_rmse -2.405%, full32motor_rmse -4.355%, late22motor_rmse -5.077%, first10gripper_rmse +47.289%, future_latent_nmse -8.536%, first10motor_mse -4.195%, mixed_loss -6.488%。
- heldout vs baseline: first10motor_rmse -5.531%, full32motor_rmse +1.264%, late22motor_rmse +2.635%, first10gripper_rmse -22.442%, future_latent_nmse -19.554%, first10motor_mse -10.890%, mixed_loss -15.760%。
- heldout vs clip_only: first10motor_rmse -4.565%, full32motor_rmse -4.011%, late22motor_rmse -3.726%, first10gripper_rmse -25.401%, future_latent_nmse -10.711%, first10motor_mse -9.249%, mixed_loss -10.040%。

Held-out 按每个 case 先平均两个 seed 后计数（只对 First10 motor RMSE 与 future latent NMSE 计数）：

- vs baseline, first10motor_rmse: 25 improved / 15 worsened / 0 tied (40 cases)。
- vs baseline, future_latent_nmse: 31 improved / 9 worsened / 0 tied (40 cases)。
- vs clip_only, first10motor_rmse: 27 improved / 13 worsened / 0 tied (40 cases)。
- vs clip_only, future_latent_nmse: 26 improved / 14 worsened / 0 tied (40 cases)。

Late22、Full32 与 gripper 仅作诊断，不作为 gate。
本任务训练更新数为 0；读取的 source joint checkpoint 来自 25858027.pbs101。
