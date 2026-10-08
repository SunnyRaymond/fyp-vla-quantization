# Completed-shard paired analysis

Frozen snapshot: 2026-10-06T00:53:18+00:00
Primary: 60 shards, 232 paired variants, 928 episodes.
Supplementary: 64 shards, 984 episodes.

| Arm | Success | Rate |
|---|---:|---:|
| bf16 | 182/232 | 78.45% |
| w4a8 | 189/232 | 81.47% |
| w4a4 | 0/232 | 0.00% |
| w4a4kv4 | 0/232 | 0.00% |

| Suite | Dimension | n | BF16 | W4A8 | W4A4 | W4A4KV4 |
|---|---|---:|---:|---:|---:|---:|
| libero_spatial | camera_viewpoints | 50 | 26 | 26 | 0 | 0 |
| libero_spatial | language_instructions | 50 | 46 | 46 | 0 | 0 |
| libero_spatial | light_conditions | 32 | 30 | 30 | 0 | 0 |
| libero_spatial | objects_layout | 50 | 47 | 47 | 0 | 0 |
| libero_spatial | robot_initial_states | 50 | 33 | 40 | 0 | 0 |

| First 10 predicted motor actions | Median normalized RMSE |
|---|---:|
| w4a8_vs_bf16 | 0.022809 |
| w4a4_vs_bf16 | 0.326077 |
| w4a4kv4_vs_bf16 | 0.326858 |
| w4a4_vs_w4a8 | 0.322308 |
| w4a4kv4_vs_w4a4 | 0.038482 |

## Interpretation limits

- Descriptive paired analysis of completed submitted tasks, not the complete 1400-variant/5600-slot benchmark.
- Variants share base tasks; they are not independent replications across all LIBERO-Plus suites.
- First action is paired at reset observation and sampler seed; later chunks follow divergent closed-loop observations and are excluded from fixed-observation error comparisons.
- Policy control_step_timeout is a valid failure; successful job exit and real-quant evidence certify execution, not preserved model accuracy.
- Final fixed-context Lat./Spd./Peak/physical Storage remain pending the separate performance job.
