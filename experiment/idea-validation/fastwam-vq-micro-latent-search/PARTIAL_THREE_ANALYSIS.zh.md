# Microscaling + latent loss：3 个候选的阶段分析

仅 selection 4 cases × 2 seeds，先在 case 内平均 seeds，再等权平均 cases。
所有误差均相对同条件 BF16；百分比相对未训练 Micro parent，负值表示误差下降。

| ρ / epoch | first10 RMSE | 变化 | full32 | late22 | gripper | future NMSE | 变化 | temporal NMSE | guards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.1 / 0 | 0.047929 | +0.00% | 0.048523 | 0.048719 | 0.020474 | 0.050511 | +0.00% | 0.089023 | pass |
| 0.1 / 1 | 0.044994 | -6.12% | 0.044515 | 0.044245 | 0.016201 | 0.032543 | -35.57% | 0.052796 | pass |
| 0.1 / 2 | 0.041802 | -12.78% | 0.044273 | 0.045242 | 0.020333 | 0.029060 | -42.47% | 0.046638 | pass |
| 0.5 / 1 | 0.044357 | -7.45% | 0.045075 | 0.045272 | 0.016673 | 0.027324 | -45.91% | 0.043556 | pass |
| 0.5 / 2 | 0.042850 | -10.60% | 0.045170 | 0.046025 | 0.019828 | 0.027881 | -44.80% | 0.045997 | pass |
| 1.0 / 1 | 0.044680 | -6.78% | 0.046889 | 0.047704 | 0.018353 | 0.034793 | -31.12% | 0.057284 | pass |
| 1.0 / 2 | 0.042257 | -11.83% | 0.044926 | 0.045969 | 0.022898 | 0.030852 | -38.92% | 0.051265 | pass |

## epoch2 逐 case

| case | parent action | ρ=0.1 action | ρ=0.5 action | ρ=1 action | parent future | ρ=0.1 future | ρ=0.5 future | ρ=1 future |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.047442 | 0.048116 | 0.050529 | 0.046831 | 0.024892 | 0.024125 | 0.020553 | 0.048184 |
| 1001 | 0.048735 | 0.039681 | 0.041310 | 0.044009 | 0.058474 | 0.032859 | 0.030680 | 0.031123 |
| 1002 | 0.042495 | 0.041251 | 0.039673 | 0.038388 | 0.044314 | 0.029449 | 0.022457 | 0.018979 |
| 1003 | 0.053045 | 0.038158 | 0.039889 | 0.039799 | 0.074366 | 0.029810 | 0.037837 | 0.025124 |

目前可用候选中的暂时最优：ρ=0.1，epoch=2，λ=0.003033027。
这不是全局 optimal λ；其余 2 个 R1、整个 R2 和最终 test 尚未纳入。
本次新增模型 forward=0，未读取任何 .pt。ρ0 action-only 对照尚未完成，不能单独归因 latent loss。

## rho=1.0 新增结果的解释

rho=1.0（lambda=0.03033027132999191）的最终 job 25805536.pbs101 已通过完整完成检查：PBS F/Exit0、artifact exit0、PIPELINE_COMPLETE、匹配 job/run/protocol、614 targets、2 epochs/16 updates，以及完整 epoch0/1/2 selection history。CPU 三组汇总 25810606.pbs101 通过同样的 terminal/artifact 检查，并匹配三个来源 jobs、9 个 selection points、每 epoch 三对候选比较。

1. 当前三组中仍按原定 action 主指标选择 rho0.1/epoch2。rho1/epoch2 的 first10 RMSE 比它高 1.09%，略超出预定 1% near-tie 区间；future NMSE 高 6.17%，temporal delta NMSE 高 9.92%，gripper 高 12.62%。rho1 的 first10 P90 略低 0.16%，因此不能说所有指标都由 rho0.1 占优。

2. 与未训练 Micro parent 相比，rho1/epoch2 的 first10 RMSE 下降 11.83%、full32 下降 7.41%、late22 下降 5.64%、future NMSE 下降 38.92%。first10 action 在四个 cases 都改善，是当前三组中唯一达到 4/4 的 epoch2 候选。rho1 与 rho0.1 的 action 逐 case 是 2 胜/2 负：rho1 在1000/1002更好，rho0.1在1001/1003更好。它的均值稍差，不能解释成每个 case 都更差。

3. rho1 的 gripper 平均误差比 parent 高 11.84%，仍在允许的 20% guards 内。rho1 从 epoch1 到 epoch2，motor/future 均值继续改善，但 gripper 上升 24.77%。这些 guards 以 parent 为参考；与上一 epoch 的变化不能直接当成 guard failure。训练直接监督 first10 六个 motor coordinates，gripper 和 late22 没有对应的直接 action loss；该监督范围可以帮助理解不同输出的取舍，但不是机制因果证明。

4. case1000 给出了新的反方向例子：rho1 的 future NMSE 从 parent 的0.024892变为0.048184，接近翻倍；同时 first10 action RMSE 从0.047442变为0.046831，反而改善。此前 rho0.1/rho0.5 是 future 改善但 action 退化。合在一起，说明当前全局 clean-future latent fidelity 与 action fidelity 并不是同一个排序目标。

5. 更强 latent 权重没有使 selection future 均值单调改善。但不能简单说 rho1 在所有 cases 上的 latent 都更差：rho1/epoch2 相比 rho0.1，在1001/1002/1003的 future NMSE 均更好；平均值变差是由1000的较大退化拉动。这更具体地提示了 scene-dependent trade-off 与全局均值掩盖局部退化的问题。现有结果没有区域/通道误差或梯度夹角，不能断言背景 dominance、梯度冲突或 latent 因果作用。

6. 当前只完成16次训练 updates，且训练与 selection cohort 不同；增加 lambda 改变的是共享参数上的联合优化路径，不保证 held-out selection future 单调改善。rho 按共同初始 calibration loss 归一化，不是每步梯度强度比例。此阶段保持原定 search/guards，无需依据三组结果临时改目标或筛选标准。

当前阶段结论：三个候选共同支持进一步搜索，但暂时最优仍是 rho0.1/epoch2（lambda约0.003033027）。rho0 action-only 的同配置训练对照尚未完成，不能把相对未训练 parent 的改善单独归因于 latent loss。须等 rho0.7/rho1.5 完成第一轮，再按五点网格生成第二轮三个候选，最终才冻结 optimal lambda 和评测既有22 cases x2 seeds。

这份分析仅对应历史 selection 四个 cases x两个 paired seeds；先在 case 内平均两个 seeds，再等权平均 cases。它是 decoded-BF16/fake-A4 的固定输入 BF16 fidelity 证据，不能当成独立泛化、closed-loop success、真实未来预测正确性或 native-kernel 加速结果。本次新增模型 forward=0，test forward=0，没有读取或下载.pt；数值汇总在真实 PBS CPU allocation 中完成。

rho0.7 job25808524.pbs101 与 rho1.5 job25808525.pbs101 在本轮实时检查中均为 R，尚未纳入本报告；没有发布 round2 choice，也没有恢复定时监控。
