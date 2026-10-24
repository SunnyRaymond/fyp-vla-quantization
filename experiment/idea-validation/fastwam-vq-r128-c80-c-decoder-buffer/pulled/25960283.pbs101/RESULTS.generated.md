# Fast-WAM R128 C80 C decoder/buffer profile

Frozen full panel: contexts 3000-3039 x seeds 0/1 (80 queries per mode).
The BF16/eager/input_fused/layer_decode_fp32 and runtime_c panels are reused; runtime_c is not rerun.
Both new modes are compared to the cached layer_decode_fp32 panel with exact full80 action parity and exact anchor action/future parity.
Decoder operand checks cover all 614 modules. The c_buffer stage runs only after c_decoder passes its parity gate.
Latency and memory regressions are reported as outcomes and do not determine parity success.
Storage inventory runs before profiling without a model forward or compilation. No training or closed-loop episodes.

| Mode | Status | Mean ms | p50 ms | p95 ms | Query peak GB | Load peak GB | Max abs vs primary | Parity |
|---|---|---:|---:|---:|---:|---:|---:|---|
| bf16 | cached | 1451.283845 | 1450.438946 | 1505.931088 | 25.130502 | nan | 2.015625 | False |
| eager | cached | 6547.440812 | 6527.626395 | 6695.141585 | 25.139422 | nan | 0 | True |
| input_fused | cached | 3520.412587 | 3503.949576 | 3724.215564 | 25.139422 | nan | 0 | True |
| layer_decode_fp32 | cached | 3593.032253 | 3532.909280 | 3936.028164 | 16.550766 | 16.251951 | 0 | True |
| runtime_c | reused_candidate | 956.361245 | 956.914356 | 959.222475 | 16.949808 | 16.294098 | 0 | True |
| c_decoder | reused_candidate | 890.745392 | 890.160186 | 898.469794 | 16.949808 | 16.294098 | 0 | True |
| c_buffer | measured | 886.964442 | 887.096445 | 894.262185 | 16.914255 | 16.260380 | 0 | True |

Deliverable complete: False
Measured candidate modes: c_buffer
Storage inventory passed: True
Per-mode decoder tuning, Graph, workspace, validation, and performance reports are in summary.json and tuning.json.

