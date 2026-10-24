# C80 R128 A+Z+KL：逐层解码与 BF16 对照

固定40 contexts x2 seeds，各80 queries；已有对照复用25943391结果，新候选在同一张A100完成。零训练更新，零新闭环。
GPU memory使用decimal GB，weight storage使用GiB。主计时无profiler。

| Mode | Mean s | p50 s | p95 s | Speedup vs eager | Speedup vs BF16 | Query peak GB | Load peak GB | First10 motor RMSE vs first round | Gripper disagreement /800 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 1.451284 | 1.450439 | 1.505931 | 4.5004 | 1.0000 | 25.130502 | 24.831867 | 0.0462718196 | 5 |
| eager | 6.547441 | 6.527626 | 6.695142 | 1.0000 | 0.2222 | 25.139422 | 24.901912 | 0 | 0 |
| input_fused | 3.520413 | 3.503950 | 3.724216 | 1.8629 | 0.4139 | 25.139422 | 24.901912 | 0 | 0 |
| layer_decode_fp32 | 3.593032 | 3.532909 | 3.936028 | 1.8477 | 0.4106 | 16.550766 | 16.251951 | 0 | 0 |
| layer_decode_bf16_u | 4.136921 | 4.166183 | 4.359765 | 1.5668 | 0.3481 | 16.550831 | 16.252016 | 0.0456158407 | 3 |
| layer_decode_bf16_all | 4.023239 | 3.982432 | 4.258963 | 1.6391 | 0.3642 | 16.554446 | 16.255631 | 0.045823589 | 4 |

First-round gate: True
Failed checks: []
BF16 intermediate arithmetic can change outputs; fixed-input errors do not establish closed-loop equivalence.
Load durations depend on cache/order and are not a fair cold-start speedup comparison.
Intrusive operators.json records are localization evidence and are not an additive E2E decomposition.
