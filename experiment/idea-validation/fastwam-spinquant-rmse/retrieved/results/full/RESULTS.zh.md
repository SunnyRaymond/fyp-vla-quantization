# SpinQuant 与 action-aware VQ：同输入 action RMSE

22 个 test observations × 2 sampler seeds；先在 case 内平均，再对 case 等权平均。

| 配置 | Selection motor RMSE | Test motor RMSE | Test gripper RMSE | Test 32-step motor RMSE |
|---|---:|---:|---:|---:|
| bf16 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| bf16w_a4 | 0.039102 | 0.048559 | 0.015465 | 0.059609 |
| scalar_w4a4 | 0.044648 | 0.054994 | 0.012511 | 0.059341 |
| original_vq_a4 | 0.063518 | 0.071311 | 0.040502 | 0.074634 |
| row_scale_vq_a4 | 0.047404 | 0.057052 | 0.031798 | 0.059709 |
| row_output_vq_a4 | 0.042646 | 0.057256 | 0.018110 | 0.066828 |
| row_action_vq_a4 | 0.041168 | 0.061030 | 0.021261 | 0.066600 |
| spinquant_native_w4a4 | 0.047347 | 0.057373 | 0.024286 | 0.068293 |

SpinQuant 直接使用原 closed-loop 的 native INT4 实现；本轮没有 decoded-BF16 SpinQuant 对照。
同一个200-step learned bank、原packed codes/scales/rotation，均未重新拟合；KV BF16。
这些是已有 observation cohort 上的描述性复用比较，训练数据和量化方法不同；不表示成功率或 native speedup。
