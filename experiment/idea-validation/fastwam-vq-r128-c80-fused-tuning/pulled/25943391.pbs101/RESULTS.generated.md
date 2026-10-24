# Frozen C80 R128 A+Z+KL fused VQ kernel results

All modes: same GPU, full40 contexts x2 seeds, fresh process, no training. GPU GB=1e9, storage GiB=2^30.

| Mode | n | Mean ms | p50 ms | p95 ms | Speedup vs BF16 | Warm allocated GB | Query peak allocated GB | Query peak reserved GB | Load peak allocated GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 80 | 1451.284 | 1450.439 | 1505.931 | 1.0000 | 24.865422 | 25.130502 | 25.419579 | 24.831867 |
| eager | 80 | 6547.441 | 6527.626 | 6695.142 | 0.2222 | 24.874590 | 25.139422 | 25.685918 | 24.901912 |
| input_fused | 80 | 3520.413 | 3503.950 | 3724.216 | 0.4139 | 24.874590 | 25.139422 | 25.685918 | 24.901912 |
| packed_fused | 80 | 7097.266 | 7092.558 | 7161.170 | 0.2045 | 16.066148 | 16.329718 | 16.426992 | 24.831867 |
| packed_direct | 80 | 7099.053 | 7094.879 | 7151.960 | 0.2044 | 16.172259 | 16.437519 | 16.538141 | 16.138769 |

## Full80 error relative to the frozen eager VQ checkpoint

| Mode | Action max abs | Full32 action RMSE | First10 motor RMSE | First10 gripper threshold disagreement |
|---|---:|---:|---:|---:|
| bf16 | 2.015625 | 0.0912011713 | 0.0462718196 | 0.00625000009 |
| eager | 0 | 0 | 0 | 0 |
| input_fused | 0 | 0 | 0 | 0 |
| packed_fused | 2.0078125 | 0.0786525011 | 0.0449622646 | 0.00750000011 |
| packed_direct | 2.0078125 | 0.0786525011 | 0.0449622646 | 0.00750000011 |

BF16 is the unquantized reference, so its error relative to eager VQ is expected.
Packed weights are decoded by tile inside GEMM. BF16 operands, FP32 transform/decoder/accumulator. No native INT4/INT8 arithmetic.
Decoded weights are bitwise equal; GEMM accumulation order may change. Full-output errors above do not establish closed-loop equivalence.
Raw operators.json is an intrusive diagnostic recorded after primary timing.
Storage is unchanged from the frozen inference bundle; external cached nonweight assets are excluded.
