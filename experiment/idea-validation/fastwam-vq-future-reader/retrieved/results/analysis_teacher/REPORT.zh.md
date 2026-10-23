# Fixed-reader future intervention

完整60 contexts；test22 cases/44 contexts。误差均相对同输入同seed的BF16 action。

| Split | Reader | Future donor | first10 | full32 | late22 | gripper |
|---|---|---|---:|---:|---:|---:|
| calibration | bf16 | action_latent | 0.023199588 | 0.038835464 | 0.043090758 | 0.002304730 |
| calibration | bf16 | action_only | 0.028313573 | 0.045738716 | 0.050643721 | 0.002300384 |
| calibration | bf16 | bf16 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| calibration | bf16 | parent | 0.024272881 | 0.045496573 | 0.051224286 | 0.002263137 |
| selection | bf16 | action_latent | 0.017535995 | 0.018961243 | 0.019448984 | 0.000835998 |
| selection | bf16 | action_only | 0.016231808 | 0.017339556 | 0.017589001 | 0.000990406 |
| selection | bf16 | bf16 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| selection | bf16 | parent | 0.015238110 | 0.018246112 | 0.019274020 | 0.001012991 |
| test | bf16 | action_latent | 0.035083670 | 0.043542794 | 0.045684393 | 0.001868341 |
| test | bf16 | action_only | 0.041243219 | 0.052907799 | 0.056283144 | 0.001630140 |
| test | bf16 | bf16 | 0.000000000 | 0.000000000 | 0.000000000 | 0.000000000 |
| test | bf16 | parent | 0.036144691 | 0.048867316 | 0.052088175 | 0.001903076 |

配对差值为前者减后者；负值表示改善。

| Contrast | First10 mean delta | Improved | Worsened | Median delta |
|---|---:|---:|---:|---:|
| test/bf16/action_latent-minus-action_only | -0.006159549 | 13/22 | 9/22 | -0.003881685 |
| test/bf16/action_only-minus-parent | 0.005098528 | 11/22 | 11/22 | 0.000563184 |
| test/bf16/action_latent-minus-parent | -0.001061021 | 9/22 | 13/22 | 0.002830838 |

Cross progression gate: True. On all22 test cases, common BF16 reader action_latent has lower mean first10 RMSE than action_only and improves at least12/22 cases. Report action_only versus parent separately; do not require or assume that contrast is positive. This is an engineering progression rule, not statistical significance.

Replay checks=60, maximum absolute action difference=0.

Previously seen frozen-cohort internal intervention; BF16 teacher fidelity; decoded BF16 VQ/fake A4 student; zero training, zero executed episodes; no native kernel, new generalization or true-future claim
