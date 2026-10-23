# Fast-WAM action + future latent：冻结队列结果汇总

- 状态：`complete`；analysis PBS job `25799647.pbs101`；实验 job `25798439.pbs101`。
- Sampling unit：22 个 test case；每个 case 先平均两个 seed，再对 case 等权平均。
- Cohort：calibration 8 cases / 8 contexts；test 22 cases / 44 contexts。
- 范围：此前已见 cohort 上的 paired fixed-input recovery；0 new model forwards、0 episodes、非 native low-bit kernel；结果不代表 closed-loop success、独立泛化或 native W4A4 latency。
- 正 delta 表示误差变大；负 delta 表示误差改善。只报告描述性均值、中位 paired delta、case 计数和尾部 case，不做 p-value。

## 训练与选择

- 训练完成 16 updates / 2 epochs；chosen epoch=2，last-trained epoch=2。chosen checkpoint 按四个 selection case 的 first-10 motor RMSE 选择；epoch 0 是未更新的初始化点。
- 冻结 latent weight `w=0.002991882294627757`；calibration 初始 action MSE=0.002226252，future NMSE=0.074409744，fraction=0.1。
- fraction=0.1 表示初始化 calibration 均值上 latent 项的 loss 数值为 action 项的 10%，不表示梯度贡献的 10%。
- Gradient coverage：16/16 updates 的 connected 与 finite 均为 614/614；nonzero 最低为 614；Adam 在 update 1 后清空一次，之后 resume 保留 optimizer。

- 首次 checkpointed parity：action max abs=0，future max abs=0；loss recipe 同时包含 action 与 future latent supervision。
- 每个 update 的 connected/finite/nonzero、10+10 denoising steps、scaled latent loss 与 reset 记录保存在 analysis.json。

### Selection history

| Epoch | First-10 motor RMSE | Future latent RMSE | Future latent NMSE | 备注 |
|---:|---:|---:|---:|---|
| 0 | 0.042645646 | 0.15800685 | 0.039690209 | cached parent actions and diagnostic futures |
| 1 | 0.055259443 | 0.14195886 | 0.032728153 |  |
| 2 | 0.042003208 | 0.14430961 | 0.033894759 | chosen |

## Action fidelity

下表覆盖 source full summary 声明的全部 7 个完整 test arms、旧 final-action distilled 和本次 chosen action+latent checkpoint。selection-only lambda candidates 不属于完整 test arms。所有指标按同一 seed 的 BF16 action 复算；source 已有的 first-10 motor/gripper 与 full-32 指标均逐行复核。

| Method | First-10 motor RMSE | Full-32 motor RMSE | Late-22 motor RMSE | First-10 gripper RMSE |
|---|---:|---:|---:|---:|
| source `bf16` | 0 | 0 | 0 | 0 |
| source `bf16w_a4` | 0.048559209 | 0.059608922 | 0.062832462 | 0.015464677 |
| source `scalar_w4a4` | 0.054994168 | 0.059340984 | 0.060688242 | 0.012510659 |
| source `original_vq_a4` | 0.071310809 | 0.074633729 | 0.074858261 | 0.040501796 |
| source `row_scale_vq_a4` | 0.057052058 | 0.059709202 | 0.060271143 | 0.031797958 |
| source `row_output_vq_a4` | 0.057255758 | 0.066828375 | 0.069300306 | 0.018110268 |
| source `row_action_vq_a4` | 0.061029794 | 0.066599787 | 0.067578688 | 0.021260778 |
| final-action distilled (epoch 2) | 0.058110442 | 0.069571863 | 0.07334153 | 0.018109958 |
| chosen action+latent (epoch 2) | 0.05295101 | 0.061331654 | 0.063866255 | 0.01882554 |

### Calibration action fidelity

使用冻结的 8 个同源 calibration case、seed 0；此表展示最终选定 checkpoint 的 action 指标。

| Arm | First-10 motor RMSE | Full-32 motor RMSE | Late-22 motor RMSE | First-10 gripper RMSE |
|---|---:|---:|---:|---:|
| row_output_vq_a4 | 0.046705511 | 0.06193001 | 0.066680216 | 0.018832755 |
| old_action_only | 0.044329114 | 0.061158546 | 0.066103474 | 0.018668849 |
| action_latent_chosen | 0.040721534 | 0.057446954 | 0.062643209 | 0.018327174 |

### Chosen action+latent vs parent and old action-only

每行的 delta 先按 case 配对，再在 22 个 case 上汇总；`improved` 指 target RMSE 低于 baseline。

| Metric | Baseline | Baseline mean | Chosen mean | Mean paired delta | Median paired delta | Improved / 22 | Worst 5 case IDs |
|---|---|---:|---:|---:|---:|---:|---|
| first10motor_rmse | row_output_vq_a4 | 0.057255758 | 0.05295101 | -0.0043047488 | -0.0039822432 | 15 | 1022, 1014, 1020, 1019, 1005 |
| full32motor_rmse | row_output_vq_a4 | 0.066828375 | 0.061331654 | -0.0054967205 | -0.0023117763 | 14 | 1017, 1022, 1021, 1019, 1014 |
| late22motor_rmse | row_output_vq_a4 | 0.069300306 | 0.063866255 | -0.0054340512 | -0.0013681468 | 13 | 1017, 1021, 1022, 1019, 1013 |
| first10gripper_rmse | row_output_vq_a4 | 0.018110268 | 0.01882554 | 0.00071527249 | -0.00019148076 | 12 | 1007, 1006, 1015, 1013, 1005 |
| first10motor_rmse | old_action_only | 0.058110442 | 0.05295101 | -0.0051594321 | -0.0014326991 | 12 | 1022, 1017, 1008, 1009, 1014 |
| full32motor_rmse | old_action_only | 0.069571863 | 0.061331654 | -0.0082402092 | -0.00060407571 | 13 | 1017, 1022, 1021, 1019, 1014 |
| late22motor_rmse | old_action_only | 0.07334153 | 0.063866255 | -0.0094752752 | 0.0001249958 | 10 | 1017, 1021, 1022, 1019, 1014 |
| first10gripper_rmse | old_action_only | 0.018109958 | 0.01882554 | 0.00071558204 | 0.0009606314 | 9 | 1007, 1012, 1013, 1011, 1010 |

## Future latent fidelity

Parent 与 old action-only 的 future 指标复用既有 diagnostic JSON receipts 的 `parent_self` / `distilled_self` rows；没有读取诊断 `.pt`，也没有新 forward。表中均值对每个 case 的两个 seed 先平均。Future 指标比较的是 BF16 模型的最终 clean latent，排除被 observation clamp 的 frame 0，包含两个 future latent frames；它衡量量化后对 BF16 future 的保真度，不是对真实未来视频的准确度。

| Arm | Future latent RMSE | Future latent NMSE | Temporal delta RMSE | Temporal delta NMSE |
|---|---:|---:|---:|---:|
| parent_self | 0.22294884 | 0.084435442 | 0.24961907 | 0.15307188 |
| old_action_only | 0.22809272 | 0.093010997 | 0.2537686 | 0.16885551 |
| action_latent_chosen | 0.20320697 | 0.071786944 | 0.23066442 | 0.13384145 |

### Calibration future fidelity

使用同一组 8 个 calibration contexts、seed 0，观察 future loss 的拟合与 test 表现是否一致。

| Arm | Future latent RMSE | Future latent NMSE | Temporal delta RMSE | Temporal delta NMSE |
|---|---:|---:|---:|---:|
| parent_self | 0.21864795 | 0.074409744 | 0.24146865 | 0.12413387 |
| old_action_only | 0.22450256 | 0.082008824 | 0.25164974 | 0.14270438 |
| action_latent_chosen | 0.1864057 | 0.059388664 | 0.21081545 | 0.10181257 |

### Per-frame latent profile means

以下列出两帧的 RMSE/NMSE profiles；case 与 seed 按同一 paired protocol 聚合。

| Arm | Profile | Frame 1 | Frame 2 |
|---|---|---:|---:|
| parent_self | future_rmse | 0.18144917 | 0.25464661 |
| parent_self | future_nmse | 0.061943715 | 0.10376743 |
| parent_self | temporal_delta_rmse | 0.18144917 | 0.30123524 |
| parent_self | temporal_delta_nmse | 0.082127281 | 0.22977629 |
| old_action_only | future_rmse | 0.1867217 | 0.26045021 |
| old_action_only | future_nmse | 0.069757794 | 0.11254191 |
| old_action_only | temporal_delta_rmse | 0.1867217 | 0.30529805 |
| old_action_only | temporal_delta_nmse | 0.093409508 | 0.24941626 |
| action_latent_chosen | future_rmse | 0.17930495 | 0.22115607 |
| action_latent_chosen | future_nmse | 0.061868664 | 0.08034145 |
| action_latent_chosen | temporal_delta_rmse | 0.17930495 | 0.27119848 |
| action_latent_chosen | temporal_delta_nmse | 0.082152037 | 0.19017262 |

### Future paired comparisons

| Metric | Baseline | Baseline mean | Chosen mean | Mean paired delta | Median paired delta | Improved / 22 | Worst 5 case IDs |
|---|---|---:|---:|---:|---:|---:|---|
| future_latent_rmse | parent_self | 0.22294884 | 0.20320697 | -0.019741868 | -0.01900976 | 17 | 1019, 1021, 1022, 1017, 1014 |
| future_latent_nmse | parent_self | 0.084435442 | 0.071786944 | -0.012648498 | -0.011022151 | 17 | 1019, 1022, 1021, 1017, 1014 |
| temporal_delta_global_rmse | parent_self | 0.24961907 | 0.23066442 | -0.01895465 | -0.024281323 | 16 | 1019, 1022, 1021, 1017, 1014 |
| temporal_delta_global_nmse | parent_self | 0.15307188 | 0.13384145 | -0.019230439 | -0.020452362 | 15 | 1019, 1022, 1017, 1021, 1014 |
| future_latent_rmse | old_action_only | 0.22809272 | 0.20320697 | -0.024885747 | -0.019848809 | 16 | 1022, 1019, 1021, 1017, 1009 |
| future_latent_nmse | old_action_only | 0.093010997 | 0.071786944 | -0.021224053 | -0.012478895 | 16 | 1022, 1019, 1021, 1017, 1009 |
| temporal_delta_global_rmse | old_action_only | 0.2537686 | 0.23066442 | -0.023104175 | -0.016020872 | 15 | 1022, 1019, 1021, 1017, 1009 |
| temporal_delta_global_nmse | old_action_only | 0.16885551 | 0.13384145 | -0.03501406 | -0.012338361 | 15 | 1022, 1019, 1017, 1021, 1014 |

## Original / Plus 探索性分组

按 10 个 Original 与 12 个 Plus test cases 分组，仅作描述性查看，不作独立验证结论。

| Contrast | Domain | Metric | Cases | Mean paired delta | Improved | Worsened |
|---|---|---|---:|---:|---:|---:|
| parent_self | original | first10motor_rmse | 10 | -0.0050356676 | 8 | 2 |
| parent_self | original | future_latent_nmse | 10 | -0.015890163 | 10 | 0 |
| parent_self | plus | first10motor_rmse | 12 | -0.0036956497 | 7 | 5 |
| parent_self | plus | future_latent_nmse | 12 | -0.0099471101 | 7 | 5 |
| old_action_only | original | first10motor_rmse | 10 | -0.0021476066 | 7 | 3 |
| old_action_only | original | future_latent_nmse | 10 | -0.016380705 | 9 | 1 |
| old_action_only | plus | first10motor_rmse | 12 | -0.0076692866 | 5 | 7 |
| old_action_only | plus | future_latent_nmse | 12 | -0.025260176 | 7 | 5 |

## 重点 case

以下列出 1016 与 1021（如在冻结 cohort 中）；这里只展示实际计算结果，不预设方向。

| Case | Domain / dimension | New minus parent first10 motor | New minus old action-only first10 motor | New minus parent future latent NMSE |
|---:|---|---:|---:|---:|
| 1016 | plus / robot | -0.0099571029 | -0.080871705 | -0.062436815 |
| 1021 | plus / language | -0.0051320168 | 0.0036380164 | 0.039647745 |

## Evidence boundary

这是一组此前 cohort 上的 paired recovery measurement，action/future 对照都使用固定输入和缓存参考。它不能证明 closed-loop success、独立泛化、native kernel 性能或 native W4A4 latency。新模型的 test 是 freeze 后 chosen checkpoint 的评估；`chosen epoch` 可能早于 `last-trained epoch=2`，因此应按实际 selection 结果解释，不能把 epoch 0 直接当成训练无效的证据。
