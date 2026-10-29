# 阶段 1a：Q0 的 A16 / A8 / 只用 H + INT8 行格点（固定输入）

作业 `26029171.pbs101`。held-out 40 contexts × 2 seeds，每臂 80 query；先 seed、再 context、再 suite、四 suite 等权。Q0_A4 复用 Q-Clip 作业的 80 条 Q0 endpoint，未重跑。decoded BF16 权重 + fake-quant activation。

## 主表

| 臂 | first10 motor RMSE | future NMSE | full32 motor RMSE | late22 motor RMSE | first10 gripper RMSE | gripper 判定不一致率 |
|---|---:|---:|---:|---:|---:|---:|
| Q0_A4 | 0.045606 | 0.061738 | 0.051059 | 0.052555 | 0.057319 | 0.006250 |
| Q0_A16 | 0.021785 | 0.020388 | 0.022270 | 0.022129 | 0.027583 | 0.002500 |
| Q0_A8 | 0.021604 | 0.021201 | 0.022268 | 0.022192 | 0.027094 | 0.002500 |
| Q0_H_INT8_A8 | 0.021726 | 0.020828 | 0.022416 | 0.022328 | 0.019766 | 0.001250 |

## 相对 Q0_A4 的变化

| 臂 | first10 motor RMSE | future NMSE | full32 motor RMSE | late22 motor RMSE | first10 gripper RMSE | gripper 判定不一致率 |
|---|---:|---:|---:|---:|---:|---:|
| Q0_A16 | -52.23% | -66.98% | -56.38% | -57.89% | -51.88% | -60.00% |
| Q0_A8 | -52.63% | -65.66% | -56.39% | -57.77% | -52.73% | -60.00% |
| Q0_H_INT8_A8 | -52.36% | -66.26% | -56.10% | -57.52% | -65.51% | -80.00% |

## 分解

| 指标 | A4→A8 回收比例 (A4−A8)/(A4−A16) | A8 相对 A16 | 只用 H + INT8 行格点相对 Q0_A8 |
|---|---:|---:|---:|
| first10 motor RMSE | 100.8% | -0.83% | +0.56% |
| first10 motor MSE | 100.4% | -1.35% | +1.20% |
| future NMSE | 98.0% | +3.99% | -1.76% |
| full32 motor RMSE | 100.0% | -0.01% | +0.66% |

## 权重级诊断（614 层，按原权重能量归一化）

- VQ 本身误差 NMSE：1.418e-02
- INT8 行取整误差 NMSE：R0 坐标 7.872e-05，H 坐标 7.885e-05
- 逐层 INT8(H)/VQ 误差比：中位 5.016e-03，最大 1.320e-02

## 逐 context 配对（40 contexts，先平均两 seeds；better = 左边误差更低）

| 对比 | motor better/worse/tied | future better/worse/tied |
|---|---:|---:|
| Q0_A8_vs_Q0_A4 | 40/0/0 | 40/0/0 |
| Q0_A8_vs_Q0_A16 | 26/14/0 | 16/24/0 |
| Q0_H_INT8_A8_vs_Q0_A8 | 11/29/0 | 24/16/0 |

## 逐 suite（first10 motor RMSE / future NMSE）

| 臂 | libero_spatial | libero_object | libero_goal | libero_10 |
|---|---:|---:|---:|---:|
| Q0_A4 | 0.04544 / 0.05868 | 0.04475 / 0.07111 | 0.04642 / 0.07000 | 0.04582 / 0.04715 |
| Q0_A16 | 0.02331 / 0.01956 | 0.02257 / 0.02152 | 0.02158 / 0.02586 | 0.01969 / 0.01461 |
| Q0_A8 | 0.02292 / 0.01953 | 0.02248 / 0.02385 | 0.02149 / 0.02666 | 0.01952 / 0.01477 |
| Q0_H_INT8_A8 | 0.02349 / 0.01921 | 0.02240 / 0.02218 | 0.02170 / 0.02741 | 0.01932 / 0.01450 |

## 来源绑定

- 启动时 Q0 case 2000 anchor 最大差：0。
- 复用的 Q0_A4 聚合与 Q-Clip 记录值最大差：0。
- 坐标改写检查（随机输入单层，相对误差）：bf16_pairing_vs_original=4.08e-03, noU_vs_R0_q0=4.49e-03, q0_vs_original=1.30e-01。
- 每臂的 case 2000 eager anchor 存于 `results/anchors/<arm>.pt`，供 engine 评测器启动检查。
