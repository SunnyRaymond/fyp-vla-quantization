# Equal-future-NMSE control

Previously seen60-context fixed-input BF16-reader intervention. C is a teacher-assisted diagnostic preserving A error direction up to BF16 rounding and matching B NMSE;no new future generation,training,episodes,generalization,native kernel or VQ superiority claim.

| Split / donor | first10 motor | full32 motor | late22 motor | gripper | future NMSE |
|---|---:|---:|---:|---:|---:|
| calibration/action_only | 0.0283135731 | 0.0457387164 | 0.0506437209 | 0.00230038406 | 0.0820088281 |
| calibration/action_latent | 0.0231995881 | 0.0388354644 | 0.0430907584 | 0.00230472985 | 0.0593886649 |
| calibration/matched | 0.0239345223 | 0.0415038287 | 0.0464182959 | 0.00225565321 | 0.0593871491 |
| selection/action_only | 0.0162318083 | 0.0173395558 | 0.0175890005 | 0.000990406465 | 0.0345952883 |
| selection/action_latent | 0.0175359953 | 0.0189612434 | 0.0194489838 | 0.000835998377 | 0.0338947588 |
| selection/matched | 0.0162457948 | 0.0173194685 | 0.0175467185 | 0.0010543644 | 0.0338943572 |
| test/action_only | 0.0412432187 | 0.0529077989 | 0.0562831438 | 0.00163013973 | 0.0930109987 |
| test/action_latent | 0.0350836697 | 0.0435427942 | 0.0456843929 | 0.00186834122 | 0.0717869447 |
| test/matched | 0.0360741596 | 0.0397012407 | 0.0404886108 | 0.00185887132 | 0.0717862951 |

Frozen progression gate: {"mean_delta": -0.0009904898787763987, "median_delta": 4.216935485601425e-05, "improved": 11, "worsened": 11, "tied": 0, "leave_one_case_out_mean_min": -0.0022095084811250367, "leave_one_case_out_mean_max": 0.0005942173745660554, "passed": false}

Matching and replay: {"max_relative_nmse_gap": 9.923824452951918e-05, "max_absolute_nmse_gap": 1.7315187127359133e-05, "min_direction_cosine": 0.9999185609750559, "replay_count": 8, "replay_max_abs": 0.0}
