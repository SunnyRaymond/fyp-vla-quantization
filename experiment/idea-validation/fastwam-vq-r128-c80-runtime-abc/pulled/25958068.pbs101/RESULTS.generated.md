# Fast-WAM R128 C80 Runtime ABC profile

Frozen full panel: contexts 3000-3039 x seeds 0/1 (80 queries per mode).
All candidate action outputs are compared against the cached layer_decode_fp32 panel.
Exact equality and max absolute error <= 1e-6 are reported; latency and memory regressions are not parity failures.
The configured diagnostic profiler uses one extra query after the measured 80 and does not enter primary latency.
No baseline reruns, new training updates, or closed-loop episodes.

| Mode | Status | Mean ms | p50 ms | p95 ms | Query peak GB | Load peak GB | Max abs vs primary | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---|
| bf16 | cached | 1451.283845 | 1450.438946 | 1505.931088 | 25.130502 | nan | 2.015625 | False |
| eager | cached | 6547.440812 | 6527.626395 | 6695.141585 | 25.139422 | nan | 0 | True |
| input_fused | cached | 3520.412587 | 3503.949576 | 3724.215564 | 25.139422 | nan | 0 | True |
| layer_decode_fp32 | cached | 3593.032253 | 3532.909280 | 3936.028164 | 16.550766 | 16.251951 | 0 | True |
| runtime_a | reused_candidate | 3676.167921 | 3693.648779 | 3770.497241 | 16.592381 | 16.294098 | 0 | True |
| runtime_b | reused_candidate | 3054.289947 | 3047.031710 | 3243.571450 | 16.688547 | 16.294098 | 0 | True |
| runtime_c | reused_candidate | 956.361245 | 956.914356 | 959.222475 | 16.949808 | 16.294098 | 0 | True |
| runtime_compile | failed | nan | nan | nan | nan | nan | nan | False |

Deliverable complete: False
Attempted modes: runtime_a, runtime_b, runtime_c, runtime_compile
Measured modes: 
Per-mode validation flags and CUDA graph/compile diagnostics are in summary.json.

runtime_c execution evidence: {"kind": "cuda_graph", "regions": {"video_step": {"captures": 1, "replays": 800, "capture_target_calls": 244, "captured_modules": [], "setup_seconds": 2.7948796921409667}, "video_prefill": {"captures": 1, "replays": 80, "capture_target_calls": 240, "captured_modules": [], "setup_seconds": 0.6771186641417444}, "action_step": {"captures": 1, "replays": 800, "capture_target_calls": 245, "captured_modules": [], "setup_seconds": 1.005631088744849}}, "unique_compiled_graphs": null, "dynamo_counters": null, "gate_passed": true}
runtime_compile execution evidence: {"kind": null, "regions": {}, "unique_compiled_graphs": null, "dynamo_counters": null, "gate_passed": false}
