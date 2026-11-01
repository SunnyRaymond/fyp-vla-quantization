# C2 连续阶段初测：每个宿主 W2 上用 A + λZ + βK 调两侧对角 scale

作业 `26057132.pbs101`。起点是 A16 baseline 的 W2 权重（H_U），码字冻结；每个 Linear 调 FP32 log-scale s（行）、t（列），W' = bf16(diag(e^s) W diag(e^t))；QTIP、QuIP# 并进 SU/SV 不加 bit，GPTVQ 每层多存两个 FP16 向量。配方照 Q0：C80 一遍，8 次 Adam、每次 10 个 context、clip 1、第一步后清动量；K 用冻结 BF16 reader 读 student future。

## 步长检查（第 1 批，s = t = 0）

| 宿主 | LR | 一阶预测下降 | 实际下降 | 比例 |
|---|---:|---:|---:|---:|
| QTIP | 0.0005 | 0.00285 | 0 | 0.00 |
| QTIP | 0.002 | 0.0114 | -0.0023 | -0.20 |
| QTIP | 0.008 | 0.0456 | -0.017 | -0.37 |
| QTIP | 选定 | 无（不调） | | |
| GPTVQ | 0.0005 | 0.00237 | 0 | 0.00 |
| GPTVQ | 0.002 | 0.00947 | 0.000433 | 0.05 |
| GPTVQ | 0.008 | 0.0379 | -0.0099 | -0.26 |
| GPTVQ | 选定 | 无（不调） | | |
| QUIPSHARP | 0.0005 | 0.00253 | 0 | 0.00 |
| QUIPSHARP | 0.002 | 0.0101 | -0.00116 | -0.11 |
| QUIPSHARP | 0.008 | 0.0404 | -0.00984 | -0.24 |
| QUIPSHARP | 选定 | 无（不调） | | |

## 训练与 C80 拟合

| 宿主 | 各批更新前 loss | C80 mixed loss 前 → 后 | bpw | proxy(H_U) 前 → 后 | 最大 |s| / |t| |
|---|---|---|---:|---|---|

## held-out 40×2

| 宿主 | 臂 | mixed loss | first10 motor | future NMSE | future 在 motor MSE 里的占比 | gripper RMSE | 换回 BF16 future 后 motor |
|---|---|---:|---:|---:|---:|---:|---:|

## 判定（预先登记）

规则：至少 2 个宿主上 C80 拟合下降、held-out mixed loss ≤ 0.97 × baseline、held-out future NMSE 低于 baseline，则跑闭环。

| 宿主 | 已调 | C80 下降 | held-out mixed 比 | future 下降 | 满足 |
|---|---|---|---:|---|---|
| QTIP | 否 | — | — | — | 否 |
| GPTVQ | 否 | — | — | — | 否 |
| QUIPSHARP | 否 | — | — | — | 否 |

0/3 个宿主满足，结论：不跑闭环。
