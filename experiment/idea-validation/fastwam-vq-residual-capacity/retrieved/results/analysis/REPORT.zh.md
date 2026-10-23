# Residual VQ capacity diagnostic

All17 matched-budget arms, full60 contexts; test22 cases/44 contexts. Paired seeds averaged within each case.

| Allocation | Fixed-reader first10 | Native first10 | Future NMSE | First10 versus uniform | Improved cases | Median delta |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 0.068937641 | 0.092264524 | 0.181304080 | 0.000000000 | 0/22 | 0.000000000 |
| random0 | 0.071791876 | 0.090961376 | 0.182378278 | 0.002854235 | 9/22 | 0.003410444 |
| random1 | 0.069524446 | 0.087311626 | 0.199133969 | 0.000586805 | 10/22 | 0.002104847 |
| random2 | 0.072693468 | 0.094015571 | 0.192912459 | 0.003755827 | 10/22 | 0.001587722 |
| magnitude | 0.088761733 | 0.108119461 | 0.215550248 | 0.019824092 | 3/22 | 0.022759703 |
| reconstruction | 0.086329724 | 0.106438711 | 0.216542912 | 0.017392083 | 3/22 | 0.020337426 |
| local_output | 0.090313925 | 0.109869187 | 0.215023124 | 0.021376284 | 4/22 | 0.015301008 |
| activation_outlier | 0.113045278 | 0.143133350 | 0.251065234 | 0.044107637 | 1/22 | 0.034911342 |
| action_fisher | 0.040995453 | 0.056459964 | 0.099004093 | -0.027942188 | 19/22 | -0.019920460 |
| action_damage | 0.160171257 | 0.211750884 | 0.288660271 | 0.091233616 | 1/22 | 0.099168455 |
| joint_fisher | 0.045683374 | 0.061760153 | 0.103195445 | -0.023254267 | 20/22 | -0.017251895 |
| latent_damage | 0.190910312 | 0.172288760 | 0.410076409 | 0.121972670 | 0/22 | 0.125405047 |
| reader_damage | 0.179039738 | 0.160330349 | 0.312420898 | 0.110102097 | 0/22 | 0.088679567 |
| future_reader_damage | 0.174578645 | 0.189465928 | 0.367431617 | 0.105641004 | 0/22 | 0.102943297 |
| matched_module_random | 0.058538228 | 0.106324504 | 0.150747718 | -0.010399413 | 14/22 | -0.008351965 |
| matched_module_reconstruction | 0.049497594 | 0.104205548 | 0.131812852 | -0.019440047 | 21/22 | -0.013058202 |
| matched_module_joint_fisher | 0.046392964 | 0.100515934 | 0.113717407 | -0.022544677 | 19/22 | -0.015341911 |

Matched bytes: {'base_index_bytes': 1504079616, 'residual_index_bytes': 752041984, 'allocation_tag_bytes': 91802, 'codebook_bytes': 2514944, 'scale_bytes': 4147598, 'transform_tensor_bytes': 9034315, 'layout_metadata_bytes': 110083, 'total_bytes': 2272020342}

Frozen-cohort internal capacity-removal diagnostic from action-only epoch2. No codeword/indices refit, no new training or episodes; sparse representation bytes are accounting, execution decoded BF16/fake A4; no native speed or novel/generalization proof.

Ranking overlap and calibration leave-one-context-out stability: allocation_summary.json. Full horizons/gripper and proposed-versus-all-baseline paired contrasts: primary_summary.json; detailed per-case secondary metrics: analysis.json.
Joint Fisher here uses rollout action+future distillation losses at the frozen student, not QuantWAMs pretraining/co-training objective. It is an information-matched Fisher control, not a reproduction claim.
