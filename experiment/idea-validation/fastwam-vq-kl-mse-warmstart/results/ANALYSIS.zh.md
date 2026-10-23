# Fast-WAM MSE-start dual-KL warm-start aggregate

Held-out 描述以 40 个 case 为单位：先在每个 case 内平均两个 seed，再对四个 suite 等权。四个条件共用冻结 cohort 与 teacher；每个 arm 只有一条训练轨迹，因此 bootstrap 仅描述固定输入 case 波动，不代表训练重复，不作显著性结论。

证据范围是 decoded BF16 compensated VQ、dynamic fakeA4 与 BF16 KV 的 fixed-input 结果；没有 rollout、closed-loop 或 native-kernel speed 证据。Gripper 不作主判断。

## 四个 checkpoint 的 held-out 描述统计

| 条件 | first10motor_rmse | full32motor_rmse | late22motor_rmse | future_latent_nmse | action_kl | latent_kl |
|---|---|---|---|---|---|---|
| original_mse | 0.031641466 | 0.03631233 | 0.03754631 | 0.055377875 | 0.074794248 | 0.17942314 |
| matched_mse_refined | 0.032115747 | 0.040339974 | 0.042546452 | 0.061798732 | 0.076617651 | 0.19831295 |
| previous_dual_refined | 0.031900538 | 0.041515414 | 0.044069665 | 0.060566725 | 0.077745084 | 0.19433998 |
| warmstart_dual | 0.032662221 | 0.037767563 | 0.039082224 | 0.061515833 | 0.08063624 | 0.19965451 |

## 配对对比

Δ 定义为对比名称左侧条件减右侧条件；负值表示左侧指标较低。改善/变差/相同按 40 个 case 计数，LOCO 仍对四个 suite 等权；bootstrap 为固定 seed 的描述性 case 重抽样。主对照是 warm-start dual-KL 对 matched MSE-refined；同时单独报告相对 original MSE。

| 对比 | 指标 | 等 suite mean Δ | case median Δ | 改善/变差/相同 | suite mean Δ | 等 suite LOCO range |
|---|---|---|---|---|---|---|
| warmstart_dual_minus_matched_mse | first10motor_rmse | 0.00054647399 | 0.00028231507 | 18/22/0 | libero_spatial: -1.6015302e-05, libero_object: -0.00020191334, libero_goal: 0.00045835311, libero_10: 0.0019454715 | [0.00011542919, 0.0010349733] |
| warmstart_dual_minus_matched_mse | future_latent_nmse | -0.00028289855 | -0.0027003274 | 22/18/0 | libero_spatial: -0.0028710992, libero_object: -0.013435919, libero_goal: 0.011477881, libero_10: 0.0036975428 | [-0.0022835641, 0.0030419089] |
| warmstart_dual_minus_matched_mse | action_kl | 0.0040185887 | 0.0024086824 | 19/21/0 | libero_spatial: -0.00084944274, libero_object: 0.0050726208, libero_goal: 0.0030571984, libero_10: 0.0087939783 | [0.0020140895, 0.0055210199] |
| warmstart_dual_minus_matched_mse | latent_kl | 0.0013415664 | -0.0088061746 | 22/18/0 | libero_spatial: -0.0093826756, libero_object: -0.040904515, libero_goal: 0.041412608, libero_10: 0.014240848 | [-0.0046075654, 0.011106798] |
| warmstart_dual_minus_original_mse | first10motor_rmse | 0.0010207552 | 0.00068387017 | 16/24/0 | libero_spatial: 0.00084458245, libero_object: 0.0029342637, libero_goal: -0.0021999586, libero_10: 0.0025041332 | [0.00048729082, 0.00124246] |
| warmstart_dual_minus_original_mse | future_latent_nmse | 0.0061379582 | -0.0019010096 | 22/18/0 | libero_spatial: 0.0001884738, libero_object: 0.011077631, libero_goal: 0.0092817928, libero_10: 0.0040039356 | [0.0037822451, 0.0071390027] |
| warmstart_dual_minus_original_mse | action_kl | 0.005841992 | 0.0038837995 | 17/23/0 | libero_spatial: 0.0081992113, libero_object: 0.015043178, libero_goal: -0.010862776, libero_10: 0.010988354 | [0.0033874721, 0.0072977737] |
| warmstart_dual_minus_original_mse | latent_kl | 0.020231377 | -0.0056051258 | 22/18/0 | libero_spatial: -8.0303103e-05, libero_object: 0.034626039, libero_goal: 0.031765282, libero_10: 0.014614489 | [0.012085133, 0.023740731] |
| warmstart_dual_minus_previous_dual | first10motor_rmse | 0.00076168261 | -0.0011878503 | 23/17/0 | libero_spatial: 0.00075217644, libero_object: -0.0011011648, libero_goal: 0.0016202284, libero_10: 0.0017754904 | [0.00026236113, 0.0011877712] |
| warmstart_dual_minus_previous_dual | future_latent_nmse | 0.0009491085 | 0.0012460919 | 19/21/0 | libero_spatial: -0.00024725809, libero_object: -0.0095666513, libero_goal: 0.007857873, libero_10: 0.0057524703 | [-0.00060183938, 0.0042609746] |
| warmstart_dual_minus_previous_dual | action_kl | 0.0028911556 | -0.00052462192 | 22/18/0 | libero_spatial: 0.0081788677, libero_object: -0.0032749533, libero_goal: 0.00077413218, libero_10: 0.0058865757 | [0.00027234019, 0.0046155505] |
| warmstart_dual_minus_previous_dual | latent_kl | 0.0053145295 | 0.0035193656 | 18/22/0 | libero_spatial: -0.00087180324, libero_object: -0.028770778, libero_goal: 0.029436727, libero_10: 0.021463972 | [0.0006801153, 0.015083691] |
| previous_dual_minus_matched_mse | first10motor_rmse | -0.00021520862 | -0.00010487903 | 20/20/0 | libero_spatial: -0.00076819174, libero_object: 0.00089925146, libero_goal: -0.0011618753, libero_10: 0.00016998108 | [-0.00053499885, 0.00018735303] |
| previous_dual_minus_matched_mse | future_latent_nmse | -0.001232007 | -0.0045671905 | 23/17/0 | libero_spatial: -0.0026238411, libero_object: -0.0038692673, libero_goal: 0.0036200077, libero_10: -0.0020549275 | [-0.0021500411, -0.00029506136] |
| previous_dual_minus_matched_mse | action_kl | 0.0011274331 | 0.0011243839 | 20/20/0 | libero_spatial: -0.0090283104, libero_object: 0.0083475741, libero_goal: 0.0022830662, libero_10: 0.0029074026 | [-0.00099856776, 0.0048032948] |
| previous_dual_minus_matched_mse | latent_kl | -0.0039729631 | -0.013569033 | 23/17/0 | libero_spatial: -0.0085108723, libero_object: -0.012133738, libero_goal: 0.011975881, libero_10: -0.0072231233 | [-0.0069526952, -0.00059320967] |
| matched_mse_minus_original_mse | first10motor_rmse | 0.00047428119 | -0.00064761098 | 23/17/0 | libero_spatial: 0.00086059775, libero_object: 0.0031361771, libero_goal: -0.0026583117, libero_10: 0.00055866167 | [-3.3842444e-05, 0.00078842616] |
| matched_mse_minus_original_mse | future_latent_nmse | 0.0064208567 | 0.0048985356 | 16/24/0 | libero_spatial: 0.003059573, libero_object: 0.024513549, libero_goal: -0.002196088, libero_10: 0.00030639274 | [0.0029321341, 0.0081191916] |
| matched_mse_minus_original_mse | action_kl | 0.0018234033 | 0.0011471342 | 18/22/0 | libero_spatial: 0.009048654, libero_object: 0.0099705576, libero_goal: -0.013919974, libero_10: 0.0021943758 | [-0.00025878139, 0.0043391739] |
| matched_mse_minus_original_mse | latent_kl | 0.01888981 | 0.014743799 | 16/24/0 | libero_spatial: 0.0093023725, libero_object: 0.075530555, libero_goal: -0.0096473258, libero_10: 0.0003736401 | [0.0087358369, 0.024896493] |

四种固定 loss 公式下的 objective 配对变化（每列均使用同一公式计算，不能横向比较不同公式的绝对值）：

| 公式 | warm-start dual − matched MSE mean Δ | case median Δ |
|---|---|---|
| mse_nmse | 4.0234908e-05 | -0.00010591336 |
| action_kl | 3.742828e-05 | -0.00010933761 |
| latent_kl | 7.0071125e-05 | -9.2655685e-05 |
| dual_kl | 6.7264497e-05 | -6.8319903e-05 |

## C80 选优与预设 screen

选择只使用 C80 的 dual-KL objective；基线加六个原 diagnostic 候选共七个条件。选中 `saved_adam_f1p0`，refinement updates=1。

| 条件 | 方向 | fraction | C80 dual-KL objective mean |
|---|---|---|---|
| no_step | None | 0.0 | 0.0027439961937488987 |
| saved_adam_f1p0 | saved_adam | 1.0 | 0.0026860426762141286 |
| saved_adam_f0p5 | saved_adam | 0.5 | 0.002994466522795847 |
| saved_adam_f0p25 | saved_adam | 0.25 | 0.0030329825371154585 |
| fresh_adam_f1p0 | fresh_adam | 1.0 | 0.0029117559344740583 |
| fresh_adam_f0p5 | fresh_adam | 0.5 | 0.0028911979410622735 |
| fresh_adam_f0p25 | fresh_adam | 0.25 | 0.0028227624963619745 |

主 screen（warm-start dual-KL vs matched MSE-refined）：False；是否超过 original MSE（motor RMSE 与 latent NMSE 的等-suite mean 均较低）：False。原筛选阈值仍是 motor RMSE case median Δ<0 且至少 24/40 case 改善。

Raw action KL Δ vs matched MSE=0.0040185887; raw latent KL Δ=0.0013415664. 仅在两项都下降时标记为 both-information-loss improvement：False。

该结果是固定 checkpoint 的 fixed-input engineering screen；不支持成功率、closed-loop、native latency 或独立训练复现结论。

## 主任务解读

共同从原MSE final80出发，不能挽救当前双KL候选的泛化：相对原MSE，held-out first10 motor RMSE约+3.23%，future latent NMSE约+11.08%，两个原始KL均值也更差。Motor只有16/40 cases改善，median delta为正，等suite逐case移除后的mean delta范围仍全为正。因而不能把前一轮失败主要解释为双KL原训练轨迹起点较差，也不能把warm start当作已验证的修复。

目前必须区分两个问题。第一，有限步优化未稳定降低真实量化C80目标：原诊断24个surrogate预测下降的方向中16个实际上升，而真实C80选步可以把双KL目标压到原MSE点以下。第二，C80下降仍未泛化：MSE精修、双KL自身精修以及从MSE出发的KL精修都在原held-out80上出现退化。第二个问题不局限于KL；现有证据支持优先检验calibration泛化与参数自由度，而非继续增加同一种密集C80更新。

下一项待检验因素是更新子空间。保留原MSE的per-channel D数值并固定它，仅开放614个clipping ratios，成对比较原MSE/NMSE和双KL；两组均从同一原MSE80出发，分别复用已有对应完整C80梯度的ratio部分，保持同样六候选加不更新的C80选步规则。与本轮已有的共同起点full-D+ratio两组形成loss×更新子空间的2×2对照。该假设尚未被证实，不能先称D是失败原因；下一轮仍不改变teacher、case/seed、lambda或loss定义，也不增加新的development set。
