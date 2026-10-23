# Fixed-reader future intervention

完整60 contexts；test22 cases/44 contexts。误差均相对同输入同seed的BF16 action。

| Split | Reader | Future donor | first10 | full32 | late22 | gripper |
|---|---|---|---:|---:|---:|---:|
| calibration | action_latent | action_latent | 0.040721533 | 0.057446952 | 0.062643209 | 0.018327174 |
| calibration | action_latent | action_only | 0.044096676 | 0.062184914 | 0.067630764 | 0.019245725 |
| calibration | action_only | action_latent | 0.042358917 | 0.057799928 | 0.062575594 | 0.020252034 |
| calibration | action_only | action_only | 0.044329114 | 0.061158544 | 0.066103472 | 0.018668849 |
| selection | action_latent | action_latent | 0.042003208 | 0.044451281 | 0.045474348 | 0.017086087 |
| selection | action_latent | action_only | 0.041586236 | 0.043664900 | 0.044549412 | 0.017963235 |
| selection | action_only | action_latent | 0.043487889 | 0.043387196 | 0.043281511 | 0.016576725 |
| selection | action_only | action_only | 0.041507430 | 0.043314914 | 0.043980499 | 0.016138743 |
| test | action_latent | action_latent | 0.052951009 | 0.061331653 | 0.063866255 | 0.018825540 |
| test | action_latent | action_only | 0.058676505 | 0.069477049 | 0.072971262 | 0.018386650 |
| test | action_only | action_latent | 0.053481176 | 0.060817783 | 0.062988386 | 0.017981764 |
| test | action_only | action_only | 0.058110442 | 0.069571863 | 0.073341530 | 0.018109958 |

配对差值为前者减后者；负值表示改善。

| Contrast | First10 mean delta | Improved | Worsened | Median delta |
|---|---:|---:|---:|---:|
| test/action_latent/action_latent-minus-action_only | -0.005725496 | 12/22 | 10/22 | -0.000413754 |
| test/action_only/action_latent-minus-action_only | -0.004629266 | 10/22 | 12/22 | 0.000549668 |

Replay checks=120, maximum absolute action difference=0.

Previously seen frozen-cohort internal intervention; BF16 teacher fidelity; decoded BF16 VQ/fake A4 student; zero training, zero executed episodes; no native kernel, new generalization or true-future claim
