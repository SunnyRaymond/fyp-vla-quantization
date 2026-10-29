# WAM-guided Hessian：S1 灵敏度 + QTIP W2 宿主（held-out 40×2）

作业 `26043905.pbs101`。BF16 模型，A16；L = A + λZ + βK 的 Gauss-Newton 探针，C80 × 1 探针。

## 灵敏度分布（各向同性权重噪声下的 loss 质量，C80 平均）

| video 去噪步 | prefill | action 步 | 其他（proprio 等） |
|---:|---:|---:|---:|
| 0.529 | 0.045 | 0.426 | 0.000 |

video 去噪步逐步占比：0.240, 0.111, 0.035, 0.013, 0.007, 0.006, 0.007, 0.010, 0.019, 0.081

action 步逐步占比：0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.002, 0.424

分头诊断（8 个 context，各头单独反传）：A 直接 0.365、A 经 future 0.024、βK 0.012、λZ 0.599；A 头里经 future 的比例 0.069。

BF16 query/reader 对 teacher 最大差 0；reader 调用签名最大相对差 0。

## QTIP W2 宿主：H_U（baseline 的 QTIP_W2 权重）vs H̃

| 臂 | bpw | proxy(H_U) | proxy(H̃) | first10 motor | future NMSE | gripper RMSE | 换回 BF16 future 后 motor |
|---|---:|---:|---:|---:|---:|---:|---:|
| QTIP_U_W2 | 2.016 | 0.00583 | 0.00770 | 0.0324 | 0.0532 | 0.0860 | 0.0246 |
| QTIP_G_W2 | 2.016 | 0.00793 | 0.00378 | 0.0332 | 0.0623 | 0.0876 | 0.0239 |

QTIP_U_W2 的 proxy(H_U) 在本作业重算；baseline 自己记录的是 0.00582。

换回 BF16 future：同一量化模型的 action 通路读 BF16 teacher 的 future（C1 在 weight-only 下的重测）。

闭环（LIBERO-Plus 280）见 `loops/QTIP_G_W2/RESULTS.zh.md`；QTIP_W2 的闭环在 A16 baseline 目录。
