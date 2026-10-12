# 原始 action projection 覆盖诊断

来源 campaign：`/scratch/users/ntu/yguo017/fastwam-vq-action-aware-20261009`（full job `25743693.pbs101`）。
固定范围：8 个 calibration cases × 4 个 arms，共 32 条 action 记录；原 campaign 的 614 个 target Linears 范围不变。

| Arm | 实际 motor RMSE | 日志 proxy RMSE | 原 4 个 probe 的实际投影 RMS | 新 16 个投影 RMS | 原 probe span 能量占比 |
|---|---:|---:|---:|---:|---:|
| `row_scale_vq_a4` | 0.0521499 | 0.272319 | 0.0527346 | 0.0524765 | 0.0825 |
| `row_output_vq_a4` | 0.0467055 | 0.235102 | 0.0465199 | 0.0491409 | 0.0750 |
| `row_action_vq_a4_lambda_0.1` | 0.0485129 | 0.00192225 | 0.0496321 | 0.0459223 | 0.0844 |
| `row_action_vq_a4_lambda_1` | 0.0514457 | 0.000964684 | 0.0395827 | 0.0493491 | 0.0480 |

配对差值使用实际前 10 步、6 个 motor 的 RMSE；负值表示前一个 arm 更低。

- `row_output_vq_a4_minus_row_scale_vq_a4`: mean paired difference -0.00544437; improved in 6/8 cases, worsened in 2/8.
- `row_action_vq_a4_lambda_0.1_minus_row_output_vq_a4`: mean paired difference 0.00180742; improved in 3/8 cases, worsened in 5/8.
- `row_action_vq_a4_lambda_1_minus_row_output_vq_a4`: mean paired difference 0.00474022; improved in 2/8 cases, worsened in 6/8.
- `row_action_vq_a4_lambda_0.1_minus_row_scale_vq_a4`: mean paired difference -0.00363694; improved in 6/8 cases, worsened in 2/8.
- `row_action_vq_a4_lambda_1_minus_row_scale_vq_a4`: mean paired difference -0.000704147; improved in 4/8 cases, worsened in 4/8.

没有保存 sampled-proxy 向量，因此这里只比较 scalar RMS，不重构 signed proxy projection。该诊断仅描述固定输入 calibration observations，不用于 test-set 选择，也不支持 closed-loop 结论。
