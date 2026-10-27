# Fast-WAM C80 reader-KL + inference engine：标准 LIBERO

作业 `26020749.pbs101`；40 tasks × 10 episodes，共 400。未重新校准；四个常驻模型动态领取任务。

| Suite | 成功数 | 成功率 |
|---|---:|---:|
| libero_spatial | 98/100 | 98.00% |
| libero_object | 99/100 | 99.00% |
| libero_goal | 97/100 | 97.00% |
| libero_10 | 97/100 | 97.00% |
| Average | 391/400 | 97.75% |

结果为每 task 前 10 个官方 initial states 的单轮评测，不代表多次重复或显著性结论。
Engine 使用 packed MXVQ → 在线 BF16 decode → BF16 cuBLAS GEMM，并启用 condition cache/CUDA Graph。
