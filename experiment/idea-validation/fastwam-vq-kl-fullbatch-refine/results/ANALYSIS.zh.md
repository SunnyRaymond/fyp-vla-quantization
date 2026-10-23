# Fast-WAM KL full-C80 refinement aggregate

分析单位是固定 checkpoint 下的输入 cohort：calibration 为 C80 的 80 个 case；held-out 为 40 个 case、每个 2 个 seed。先在 case 内对 seed 求均值，再在四个 suite 内对 case 求均值，最后四个 suite 等权。两条训练轨迹各只有一个 checkpoint；case bootstrap 仅描述固定 seed 的 case 波动，不代表训练重复，也不作显著性结论。

证据范围为 fixed-input decoded-BF16 student evaluation、fake-A4 execution 与 BF16 KV 缓存配置。没有 rollout、closed-loop 或 native-kernel speed 证据。Gripper 不进入主门槛。

## Held-out 描述统计

| 条件 | first10motor_mse | first10motor_rmse | full32motor_rmse | late22motor_rmse | future_latent_nmse | action_kl | latent_kl |
|---|---|---|---|---|---|---|---|
| mse_nmse original | 0.0010800749 | 0.031641466 | 0.03631233 | 0.03754631 | 0.055377875 | 0.074794248 | 0.17942314 |
| mse_nmse refined | 0.0011154796 | 0.032115747 | 0.040339974 | 0.042546452 | 0.061798732 | 0.076617651 | 0.19831295 |
| dual_kl original | 0.0011386869 | 0.03232986 | 0.036817921 | 0.038043025 | 0.053033005 | 0.083403036 | 0.17095114 |
| dual_kl refined | 0.0011094497 | 0.031900538 | 0.041515414 | 0.044069665 | 0.060566725 | 0.077745084 | 0.19433998 |

## 主配对：dual_kl refined 减 mse_nmse refined

负值表示 dual_kl refined 指标较低。改善/变差/相同按 40 个 case 计数；seed 已先在 case 内平均。LOCO 每次从所属 suite 移除一个 case，仍对四个 suite 等权。Bootstrap 区间使用固定 seed 对 40 个 case 重抽样，仅作描述。

| 指标 | 等 suite mean Δ | case median Δ | 改善/变差/相同 | suite mean Δ | 等 suite LOCO range |
|---|---|---|---|---|---|
| first10motor_mse | -6.0298607e-06 | -2.7057831e-07 | 20/20/0 | libero_spatial: -5.0110665e-05, libero_object: 5.6033627e-05, libero_goal: -8.604555e-05, libero_10: 5.6003146e-05 | [-3.1329365e-05, 2.6234919e-05] |
| first10motor_rmse | -0.00021520862 | -0.00010487903 | 20/20/0 | libero_spatial: -0.00076819174, libero_object: 0.00089925146, libero_goal: -0.0011618753, libero_10: 0.00016998108 | [-0.00053499885, 0.00018735303] |
| full32motor_rmse | 0.0011754406 | 0.00099867908 | 11/29/0 | libero_spatial: 0.0005250236, libero_object: -0.0012888468, libero_goal: 0.0046007031, libero_10: 0.00086488249 | [0.00063797887, 0.0014861356] |
| late22motor_rmse | 0.0015232128 | 0.0009560585 | 13/27/0 | libero_spatial: 0.0008244629, libero_object: -0.0021096373, libero_goal: 0.006200083, libero_10: 0.0011779426 | [0.00086171341, 0.0018869276] |
| future_latent_nmse | -0.001232007 | -0.0045671905 | 23/17/0 | libero_spatial: -0.0026238411, libero_object: -0.0038692673, libero_goal: 0.0036200077, libero_10: -0.0020549275 | [-0.0021500411, -0.00029506136] |
| action_kl | 0.0011274331 | 0.0011243839 | 20/20/0 | libero_spatial: -0.0090283104, libero_object: 0.0083475741, libero_goal: 0.0022830662, libero_10: 0.0029074026 | [-0.00099856776, 0.0048032948] |
| latent_kl | -0.0039729631 | -0.013569033 | 23/17/0 | libero_spatial: -0.0085108723, libero_object: -0.012133738, libero_goal: 0.011975881, libero_10: -0.0072231233 | [-0.0069526952, -0.00059320967] |

同一公式下的 objective loss 配对变化如下；不同 loss 公式的绝对值不可横向比较。

| 固定 loss 公式 | 等 suite mean Δ | case median Δ |
|---|---|---|
| mse_nmse | -5.8334137e-05 | -0.00014654978 |
| action_kl | -3.8434053e-05 | -8.2114224e-05 |
| latent_kl | -5.8820105e-05 | -0.00017118175 |
| dual_kl | -3.8920021e-05 | -0.00010861535 |

## 相对原始 final-80 checkpoint 的 held-out 配对上下文

| 对比 | first10 motor RMSE Δ | future latent NMSE Δ | action KL Δ | latent KL Δ |
|---|---|---|---|---|
| mse_refined_minus_mse_original | 0.00047428119 | 0.0064208567 | 0.0018234033 | 0.01888981 |
| dual_refined_minus_dual_original | -0.00042932113 | 0.0075337202 | -0.0056579521 | 0.023388842 |
| dual_refined_minus_original_mse | 0.00025907257 | 0.0051888497 | 0.0029508365 | 0.014916847 |

## C80 校准选优

两组都在 C80 自身目标上，从 baseline/no_step 与六个 Adam direction/fraction 候选中选均值最低者；selection 未使用 held-out。`no_step` 表示没有 refinement improvement。

各 arm 的 C80 objective 使用各自损失公式；这里按 arm 标注，不横向比较公式不同的 objective 数值。

| Arm | 选择 | refinement updates | C80 baseline objective | 公式 | C80 selected objective |
|---|---|---|---|---|---|
| mse_nmse | fresh_adam_f0p25 | 1 | 0.0028503231 | mse_nmse formula | 0.002797217 |
| dual_kl | saved_adam_f0p25 | 1 | 0.002797668 | dual_kl formula | 0.0026390678 |

## 预设 joint screen

joint gate: False（dual refined held-out first-10 motor RMSE 和 future latent NMSE 都低于 matched mse_nmse refined；motor case median Δ < 0；至少 24/40 case motor RMSE 改善）。实际 refinement 条件：True。若选中 `no_step`，按无 refinement improvement 解释。

Action KL 相对 matched MSE refined 变化：0.0011274331；latent KL 变化：-0.0039729631。只有两个原始 KL 项均下降时，才称两项 information loss 同时改善。

单次训练与固定输入结果不能支持 closed-loop、成功率、native latency 或独立训练复现结论。

## 主任务解读与后续检验

本轮没有通过采用门槛。相对原 MSE final80，两组新候选的 held-out first10 motor RMSE 与 future latent NMSE 均值都变差。双 KL 精修相对原 MSE 为约 +0.82% / +9.37%；MSE 精修约为 +1.50% / +11.59%。双 KL 略胜同样退化的 MSE 精修，不应被解读为超过原来的已知好解。主配对 motor 只有20/40 cases改善，LOCO范围跨0；两个原始KL项也没有同时改善。

两组完整 C80 自身目标分别下降约1.86%和5.67%，held-out却退化，因此目前已有直接的训练目标下降与泛化指标失配证据。这既发生在MSE也发生在KL，不能归因于KL独有缺陷，也不能把完整C80选步当作泛化保证。它补充而不替代前一轮16/24 surrogate预测下降、真实C80上升的有限步优化诊断。

接下来仅检验原mse_dual诊断中已经按C80选出的KL候选：从原MSE final80共同起点出发，复用其完整80梯度与六条件C80结果，物化最佳saved_adam_f1p0候选，再评估原C80与held-out80。该检验不重新计算梯度、不增加teacher、不改变loss、cases、seeds或候选网格；其matched MSE对照就是本轮已经完成的MSE精修。研究方向受到本轮held-out结果启发，但下一候选的选择仍固定来自此前完成的C80诊断，并非用held-out挑六个候选。
