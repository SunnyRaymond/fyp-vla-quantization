# 1b：H 坐标 + 整数 codebook VQ + A8，A+Z+K

作业 `26030496.pbs101`。held-out 40 contexts × 2 seeds，聚合 seed→context→suite→四 suite 等权。

| 配置 | first10motor_rmse | future_latent_nmse | full32motor_rmse | late22motor_rmse | first10gripper_rmse | first10gripper_disagreement |
|---|---:|---:|---:|---:|---:|---:|
| Q0（R0+MXVQ+A4，参照） | 0.045606 | 0.061738 | 0.051059 | 0.052555 | 0.057319 | 0.006250 |
| VQ0′（整数 VQ 拟合，未校准） | 0.020923 | 0.029136 | 0.024204 | 0.025036 | 0.039579 | 0.003750 |
| 1b（A+Z+K 8 步） | 0.018290 | 0.030373 | 0.021907 | 0.022638 | 0.029493 | 0.003750 |

- β = 0.00622863；初始 C80 均值：{"latent_nmse": 0.028344140527769922, "reader_kl": 0.01931946104286908, "reader_mse": 0.00625912457544473}
- 整数 codebook 训练中改变的条目：365638/1257472
- 拟合：加权 weight NMSE 0.01468，格式 bpw 4.0089（含 Smooth/signs 4.0209），最大使用的码字和 127
- 导出：`/scratch/users/ntu/yguo017/fastwam-1b-h-int8vq-a8-20261027/results/export/inference_bundle.pt`；engine anchor：`/scratch/users/ntu/yguo017/fastwam-1b-h-int8vq-a8-20261027/results/arms/reader_kl/anchor.pt`
