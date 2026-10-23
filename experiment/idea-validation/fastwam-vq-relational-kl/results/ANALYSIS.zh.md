# 第一轮时空关系 KL 对照

两组使用相同 C80、共享 teacher、原 held-out40cases×2seeds、初始化、80updates 和可训练参数。soft-bin 复用原 dual_kl final80；关系组仅增加 temporal/spatial KL。

温度和关系项系数仅用完整 C80 teacher 及预设 standardized noise=0.1、相对权重0.25 冻结。未使用新的 development set 或 held-out 选参数。

| Split | Arm | Action KL | Latent KL | Temporal KL | Spatiotemporal KL | Soft-bin objective | Composite objective | Motor RMSE | Latent NMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| calibration | softbin | 0.071938882 | 0.14394428 | 0.04180668 | 0.0050667305 | 0.002797668 | 0.0049441032 | 0.030608191 | 0.045182512 |
| calibration | relational | 0.07147981 | 0.14685875 | 0.034701079 | 0.0047080519 | 0.0028307459 | 0.0047936246 | 0.030427245 | 0.046244474 |
| heldout | softbin | 0.083403036 | 0.17095114 | 0.040418109 | 0.0063970953 | 0.0032975557 | 0.0059133057 | 0.03232986 | 0.053033005 |
| heldout | relational | 0.075975663 | 0.17523083 | 0.034484781 | 0.006147169 | 0.0032630464 | 0.0057434071 | 0.031438487 | 0.054295895 |

## Held-out 相对变化

| Metric | Relational vs soft-bin | Improved cases / 40 | Median paired delta |
|---|---:|---:|---:|
| action_kl | -8.905% | 21/40 | -0.00342606 |
| latent_kl | +2.503% | 21/40 | -0.00447781 |
| action_temporal_kl | -14.680% | 22/40 | -0.00221891 |
| latent_spatiotemporal_kl | -3.907% | 30/40 | -0.000572656 |
| softbin_objective | -1.047% | 19/40 | +1.95003e-05 |
| relational_objective | -2.873% | 25/40 | -0.000320677 |
| first10motor_rmse | -2.757% | 23/40 | -0.000921725 |
| future_latent_nmse | +2.381% | 21/40 | -0.00149968 |

## 统一 composite 各项占比

| Split | Arm | Action anchor | Latent anchor | Action relation | Latent relation |
|---|---|---:|---:|---:|---:|
| calibration | softbin | 17.90% | 38.69% | 6.45% | 36.97% |
| calibration | relational | 18.34% | 40.71% | 5.52% | 35.43% |
| heldout | softbin | 17.35% | 38.41% | 5.21% | 39.02% |
| heldout | relational | 16.27% | 40.54% | 4.58% | 38.61% |

## 分布定义验证与边界

受控错误使用现有完整 C80 teacher；不产生新的 teacher/student forward。时间错序、空间错位与匹配 MSE 的整体偏置分别报告，不能把所有等幅误差视为相同。

关系距离存在平移、整体反号和部分时间方向对称性；独立自检明确包含线性匀速动作反转、uniform temporal latent shift 的反例。soft-bin 绝对值约束保留以覆盖这些盲区。

此轮衡量固定位置邻接的关系分布，未估计完整 action–latent 生成联合密度，也未执行环境闭环。结构 loss 下降和固定输入 RMSE/NMSE 变化均不能替代闭环成功率。

每组仅一条训练轨迹；40个held-out case先平均各自两个seed，不能将80queries视为80个独立case。

## 结果解读

这轮提供了正向的结构指标证据：统一 composite 在 C80 降低约3.04%，在原 held-out 降低2.87%；held-out action temporal KL降低14.68%、latent spatiotemporal KL降低3.91%。对应配对改善分别为22/40、30/40cases，composite为25/40cases。收益同时出现在calibration和held-out，当前主要现象更符合目标之间的tradeoff，而不是仅训练集改善的模式；单条训练轨迹仍不足以排除过拟合或证明一般化收益。

Latent关系更接近teacher时，latent逐点KL仍上升2.50%、NMSE上升2.38%。距离关系与绝对数值保真衡量不同属性，weighted sum不保证每个分项同时改善。逐点latent KL虽然21/40cases改善，其均值仍变差；统一soft-bin objective虽然均值降低1.05%，却只有19/40cases改善且median delta为正，因此不能描述为全面提升。

异质性也存在于suite：held-out action KL与temporal KL在四个suite均改善；motor RMSE在libero_spatial变差；latent spatiotemporal KL和统一composite在libero_goal变差。总体均值应与这些退化一起判断，不应据此调参挑选本次held-out的有利子集。

实际held-out composite中，latent anchor约40.54%、latent relation约38.61%、action anchor约16.27%、action relation约4.58%。预设“0.25”是在teacher white-noise参照下配平，不意味着实际模型误差中结构项恒占25%，也不能据该占比直接推断梯度或闭环因果贡献。

## 原 MSE final80 的历史同协议参照

以下原MSE/NMSE per-channel组为之前已完成、已重放核对的同一原held-out80结果，本轮未重训/重跑它，来源为`fastwam-vq-dual-kl/results/heldout/mse_nmse/summary.json`。没有其完整student latent缓存，因此不提供MSE组的新增关系KL或新composite比较。

| Arm | First10 motor RMSE | Future latent NMSE | Action soft-bin KL | Latent soft-bin KL |
|---|---:|---:|---:|---:|
| 原 MSE/NMSE final80（历史） | 0.031641466 | 0.055377875 | 0.074794248 | 0.179423136 |
| Soft-bin final80 | 0.032329860 | 0.053033005 | 0.083403036 | 0.170951142 |
| Soft-bin + relational final80 | 0.031438487 | 0.054295895 | 0.075975663 | 0.175230830 |

新增组相对历史MSE组的motor RMSE和latent NMSE均略低，但action逐点KL仍略高。这再次表明不同固定输入指标的排序可能不一致，不能从其中任一项推断闭环。

## 下一步判断

保留本次固定final80作为闭环候选。若继续验证实际价值，优先在既定配对闭环协议下比较冻结的MSE、soft-bin与relational checkpoint，避免继续只依据proxy改复杂loss。若需要归因，再做temporal-only与spatiotemporal-only消融；当前两组实验无法分离两个新增项各自的因果贡献。本轮没有启动这些后续实验。

## 作业与资源

CPU定义验证25883706.pbs101、soft-bin评估25884637.pbs101、relational训练评估25883750.pbs101和CPU汇总25884638.pbs101均已F/Exit_status=0，artifact exit、完成marker和完整覆盖门槛通过。两组各C80 80条、held-out 80条，共320条端点；teacher全复用，无新development set。新训练80updates的顺序、梯度连接/有限性、冻结权重及deployment/infer parity均通过。

成功GPU作业分别41m53s和1h05m58s，合计1h47m51s。对照首个尝试因tensor保存父目录缺失退出（16m33s），已修复；计入该失败尝试，本轮累计GPU allocation为2h04m24s。每个GPU作业均小于2h；未将累计时间表述为小于2h。关系组训练未重启。GPU利用率/显存每15秒写入job.log，gpt-6-luna/xhigh完成周期监控并已结束。
