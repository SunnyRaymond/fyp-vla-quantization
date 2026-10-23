# C80 clip-only 与 joint：LIBERO-Plus 闭环评测

状态：`complete`；新增episode：280/280；复用baseline：2240/2240。

两个arm均从同一 C8 rho1.5 update8 bank开始，使用冻结C80的80个context完成80次更新；clip_only只训练614个clipping ratio，joint训练FP32 codebooks、log row gains与clipping ratios，不包含channel scaling。评测执行decoded BF16 weights、dynamic A4 activations、BF16 KV cache。

每个worker在正式episode前重放case 2000 / seed index 0；保存的 reference action 和当前 runtime 都必须覆盖614个target Linear，且最大绝对误差不超过1e-6。

| 配置 | 成功 / episodes | Success rate |
|---|---:|---:|
| bf16 | 216/280 | 77.14% |
| quarot_adapted_w4a4 | 195/280 | 69.64% |
| spinquant_adapted_w4a4 | 203/280 | 72.50% |
| smooth05_hadamard_scalar_w4a4 | 200/280 | 71.43% |
| row_scale_vq_a4 | 202/280 | 72.14% |
| micro_scale_vq_a4 | 204/280 | 72.86% |
| rho0.1 | 201/280 | 71.79% |
| rho1.5 | 206/280 | 73.57% |
| clip_only | 207/280 | 73.93% |

配对 win/loss/tie，以及suite、dimension和suite×dimension统计见 `summary.json`。
Timing按 `batched_h128_v2` implementation报告，并区分每个worker process的首个正式query与后续query；该实验没有native low-bit kernel调用。
