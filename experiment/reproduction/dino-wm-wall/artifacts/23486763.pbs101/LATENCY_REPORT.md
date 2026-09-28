# DINO-WM Wall latency profile

- Complete cases: 2; successes: 2
- Pipeline: 65.949 s; observed process: 66.129 s
- JSONL write overhead: 39.290 ms
- Boundary: Fine-grained GPU sections synchronize CUDA for attributable wall time; totals include profiling perturbation and are not production-throughput claims.

## Largest exclusive contributors

| label | count | exclusive s | pipeline % | inclusive mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| setup.workspace_and_targets | 1 | 30.272 | 45.90 | 30271.708 | 30271.708 |
| wm_rollout.predict_autoregressive | 154 | 18.857 | 28.59 | 122.451 | 211.344 |
| setup.load_model_and_checkpoint | 1 | 4.918 | 7.46 | 4917.583 | 4917.583 |
| wm_rollout.predict_terminal | 36 | 4.695 | 7.12 | 130.417 | 211.315 |
| wm_rollout.encode_initial | 36 | 4.037 | 6.12 | 112.128 | 178.800 |
| evaluator.environment_rollout | 14 | 1.161 | 1.76 | 82.909 | 137.170 |
| setup.create_environments | 1 | 0.996 | 1.51 | 995.951 | 995.951 |
| setup.load_dataset | 1 | 0.269 | 0.41 | 268.684 | 268.684 |
| final_export.videos | 1 | 0.251 | 0.38 | 250.915 | 250.915 |
| cem.encode_goal | 2 | 0.101 | 0.15 | 50.687 | 92.189 |
| evaluator.encode_achieved | 14 | 0.072 | 0.11 | 5.176 | 5.314 |
| wm_rollout.append_state | 154 | 0.050 | 0.08 | 0.323 | 0.764 |
| wm_rollout.replace_action | 154 | 0.035 | 0.05 | 0.230 | 0.242 |
| cem.optimization_step.total | 11 | 0.031 | 0.05 | 2602.793 | 2691.296 |
| cem.topk_and_distribution_update | 22 | 0.024 | 0.04 | 1.110 | 0.254 |
| setup.read_model_config | 1 | 0.022 | 0.03 | 21.529 | 21.529 |
| wm_rollout.append_terminal | 36 | 0.021 | 0.03 | 0.578 | 0.937 |
| cem.objective | 22 | 0.021 | 0.03 | 0.941 | 0.307 |
| evaluator.task_metrics | 14 | 0.011 | 0.02 | 0.814 | 0.862 |
| evaluator.preprocess_achieved | 14 | 0.011 | 0.02 | 0.755 | 0.792 |
| planner.cem.total | 2 | 0.010 | 0.02 | 14372.808 | 24982.013 |
| evaluator.embedding_divergence | 14 | 0.009 | 0.01 | 0.677 | 2.696 |
| evaluator.preprocess_initial | 14 | 0.009 | 0.01 | 0.623 | 0.784 |
| cem.candidate_world_model_rollout | 22 | 0.007 | 0.01 | 1242.203 | 1239.851 |
| evaluator.preprocess_goal | 14 | 0.006 | 0.01 | 0.446 | 0.501 |

## MPC rounds

| round | wall s | events |
|---:|---:|---:|
| 1 | 26.261 | 802 |
| 2 | 2.769 | 133 |

## CEM optimization steps

| step | count | total s | mean ms | p95 ms |
|---:|---:|---:|---:|---:|
| 1 | 2 | 5.370 | 2684.969 | 2780.752 |
| 2 | 1 | 2.591 | 2591.198 | 2591.198 |
| 3 | 1 | 2.585 | 2584.971 | 2584.971 |
| 4 | 1 | 2.583 | 2583.217 | 2583.217 |
| 5 | 1 | 2.579 | 2579.469 | 2579.469 |
| 6 | 1 | 2.582 | 2582.331 | 2582.331 |
| 7 | 1 | 2.584 | 2584.297 | 2584.297 |
| 8 | 1 | 2.583 | 2583.427 | 2583.427 |
| 9 | 1 | 2.584 | 2583.806 | 2583.806 |
| 10 | 1 | 2.588 | 2588.067 | 2588.067 |

## Evaluator calls

| kind | count | total s | mean ms |
|---|---:|---:|---:|
| cem_diagnostic | 11 | 1.219 | 110.825 |
| final | 1 | 0.181 | 181.370 |
| mpc_execution | 2 | 0.279 | 139.644 |
