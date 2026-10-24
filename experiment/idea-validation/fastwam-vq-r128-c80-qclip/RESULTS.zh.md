# C80 R128 Q-Clip 对 Q0：固定输入结果

Q-Clip 在原 held-out 40 contexts ×2 seeds 上，相对同一最终 Q0 的 first10 motor RMSE 下降3.56%，future latent NMSE下降16.29%；full32、late22和gripper RMSE的整体均值也下降。该结果只支持固定输入数值改善，尚不能称为新的闭环最佳模型。

长作业25937633.pbs101先运行，并在job log中记录取消短作业25937632.pbs101。长作业已PBS F/Exit0、artifact exit0、PHASE_COMPLETE，controller verify通过。实际GPU为NVIDIA A100-SXM4-40GB。每15秒GPU利用率/显存日志持续记录；本轮未做性能benchmark。

## 同一 checkpoint 的对照

Q0为25921795.pbs101的reader_kl/update8最终C80 R128 Microscaling VQ A+Z+KL模型。Q-Clip只追加学习614个activation ratios；weights、VQ codebooks/row gains/indices/B32 exponents、Smooth/H/learned-R均冻结。固定C80一遍80个contexts，80次单context Adam更新，LR0.01，ratio从1开始并限制[0.5,1]。A+Z+KL的系数保持原值，固定最终update80，无heldout选择checkpoint。

Loss = first10 motor MSE + 0.04245452683169643 × future latent NMSE + 0.006083668576461056 × reader KL。端点只测A/Z和动作指标；完整最终reader KL端点未另外追加，训练日志有各context更新前的KL，不能视为同一endpoint的KL对比。

## Held-out：原40 contexts ×2 seeds

每个配置80条查询；先平均同context两seeds，再平均context、四suite等权。下表变化为(Qclip/Q0−1)，负值表示误差下降。独立context数为40，不将80条查询视为80个独立context。

| 指标 | Q0 | Q-Clip | 相对变化 |
|---|---:|---:|---:|
| First10 motor RMSE（normalized） | 0.045606257 | 0.043983056 | -3.56% |
| First10 motor MSE | 0.002141081 | 0.001989554 | -7.08% |
| Future latent NMSE | 0.061737563 | 0.051682642 | -16.29% |
| A+Z mixed loss（不含 KL） | 0.004762120 | 0.004183716 | -12.15% |
| Full32 motor RMSE | 0.051059191 | 0.047379521 | -7.21% |
| Late22 motor RMSE | 0.052555470 | 0.048407856 | -7.89% |
| First10 gripper RMSE | 0.057318556 | 0.050237561 | -12.35% |

Gripper经过正式controller的denormalization、符号翻转和离散化后，与BF16 teacher的first10动作判定不一致为5/800→4/800（0.625%→0.500%）。这些是固定查询中的动作位置，不是闭环失败次数。

## Calibration：原80 contexts ×1 seed

| 指标 | Q0 | Q-Clip | 相对变化 |
|---|---:|---:|---:|
| First10 motor RMSE（normalized） | 0.042323641 | 0.042622016 | +0.70% |
| First10 motor MSE | 0.001820558 | 0.001853652 | +1.82% |
| Future latent NMSE | 0.043845070 | 0.041965319 | -4.29% |
| A+Z mixed loss（不含 KL） | 0.003681980 | 0.003635270 | -1.27% |
| Full32 motor RMSE | 0.045097518 | 0.044126291 | -2.15% |
| Late22 motor RMSE | 0.045991409 | 0.044588808 | -3.05% |
| First10 gripper RMSE | 0.038132879 | 0.052948204 | +38.85% |

Calibration first10 motor RMSE上升0.70%，gripper RMSE上升38.85%，离散gripper不一致为2/800→5/800。A+Z mixed loss仍下降1.27%，体现多目标之间的折中；不能只展示held-out正向均值。

## Held-out suite分解

| Suite | Motor RMSE Q0→Qclip | Motor变化 | Latent NMSE Q0→Qclip | Latent变化 |
|---|---:|---:|---:|---:|
| libero_spatial | 0.045441 → 0.043308 | -4.69% | 0.058683 → 0.038421 | -34.53% |
| libero_object | 0.044750 → 0.043848 | -2.02% | 0.071113 → 0.058335 | -17.97% |
| libero_goal | 0.046419 → 0.044458 | -4.22% | 0.070003 → 0.060075 | -14.18% |
| libero_10 | 0.045815 → 0.044318 | -3.27% | 0.047152 → 0.049899 | +5.83% |

四个suite的first10 motor均改善，但libero_10的latent NMSE上升5.83%，full32 motor RMSE上升0.95%；其余三个suite的这两项均改善。因此不能称所有输出或所有任务都改善。

## 学到的 clipping 与复用

614个ratios：mean=0.960495，median=0.985709，min=0.703582；485/614小于1，84/614小于0.9，没有触及0.5下界。平均ratio是层阈值比例，不能解释为被删除activation或weight的比例。

复用160份BF16 teacher packets、原reader statistics和beta、同一Q0源checkpoint及case2000的完整tensor anchor。Q0与Qclip各覆盖calibration80+heldout80，共320个逻辑端点。新teacher生成0次；student query481、formal inference1、reader query80、parameter reverse pass80。Reader仍有实际BF16计算，不能把teacher targets复用理解为零reader成本。

所有80次更新均到达614个ratio梯度，重放误差全为0。r=1原Q0 action/future最大差异0，最终正式infer对sampler最大差异0。冻结weights/R检查通过。训练日志最大gradient norm约0.002970，未达到gradient clip1。

## 证据与限制

- [完整汇总](pulled/25937633.pbs101/summary.json)
- [80次训练记录](pulled/25937633.pbs101/training.json)
- [完成证据](completion_evidence.json)
- [实验协议](protocol.json)

原held-out已经参与过多次工程诊断。本轮是单个冻结cohort、单次校准recipe的比较，不作独立测试集、统计显著性或跨训练seeds稳定性声明。当前执行仍为decoded-BF16 weight + fake A4、BF16 KV；无闭环、native kernel、speedup或memory reduction结论。按用户要求，本轮完成固定输入比较后停止，不自动扩展实验。

集群部署产物：`/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-qclip-20261023/results/deployment.pt`，保存ratios及冻结Q0引用；完整optimizer/RNG断点为同目录checkpoint.pt。
