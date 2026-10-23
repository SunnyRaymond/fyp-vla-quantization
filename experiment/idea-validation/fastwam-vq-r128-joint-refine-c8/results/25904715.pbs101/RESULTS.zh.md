# Fast-WAM R128 Joint C8 refinement 固定输入聚合结果

Protocol：`fastwam-vq-r128-joint-refine-c8-v1`；target Linears：614；heldout 每臂40 contexts × 2 seeds = 80 queries。
heldout 按 seed → case → suite 聚合，四个 suite 等权；相对变化为 candidate / baseline - 1，负值表示误差下降。
`Joint8_R1e4` 是第一轮 rotation learning rate=1e-4 的结果；本轮 `Joint8` 使用 rotation learning rate=1e-5，并与 VQ learning rate=5e-4 联合更新。
`H_currentpath` 固定 R=I，只做8次 VQ 更新，使用当前 JointBanks forward path。原 `H_VQ8` incumbent 保持冻结。

## Heldout 各臂总体结果

每格为四 suite 等权的 mean；每臂均为完整40 contexts × 2 seeds。

| Arm | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| H_W16A4 | 0.042604556 | 0.038380617 | 0.003474206 | 0.045009860 | 0.045766080 | 0.030154183 | 0.001844775 |
| R8_W16A4 | 0.044585121 | 0.040343879 | 0.003740738 | 0.046935371 | 0.047702240 | 0.034528374 | 0.002027958 |
| H_VQ8 | 0.044296774 | 0.060877371 | 0.004598652 | 0.047500980 | 0.048528504 | 0.057054031 | 0.002014132 |
| R8_VQ8 | 0.046408990 | 0.065187050 | 0.004980800 | 0.050613965 | 0.051931984 | 0.060208080 | 0.002213315 |
| Joint8_R1e4 | 0.045524040 | 0.060773535 | 0.004718190 | 0.048873038 | 0.049926803 | 0.042294998 | 0.002138078 |
| H_currentpath | 0.045305496 | 0.065714797 | 0.004894043 | 0.049968888 | 0.051394088 | 0.051747971 | 0.002104152 |
| Joint8 | 0.045226322 | 0.065914399 | 0.004893894 | 0.053483246 | 0.055737509 | 0.058941074 | 0.002095530 |

## Heldout comparisons

### R8_W16A4 vs H_W16A4

| Metric | Baseline | Candidate | Relative change |
|---|---:|---:|---:|
| First10 motor RMSE | 0.042604556 | 0.044585121 | +4.649% |
| Future latent NMSE | 0.038380617 | 0.040343879 | +5.115% |
| Mixed loss | 0.003474206 | 0.003740738 | +7.672% |
| Full32 motor RMSE | 0.045009860 | 0.046935371 | +4.278% |
| Late22 motor RMSE | 0.045766080 | 0.047702240 | +4.231% |
| First10 gripper RMSE | 0.030154183 | 0.034528374 | +14.506% |
| First10 motor MSE | 0.001844775 | 0.002027958 | +9.930% |

First10 motor RMSE 的 case 配对计数：improved 12，worsened 28，tied 0（共 40 cases）。

### Joint8 vs H_VQ8

| Metric | Baseline | Candidate | Relative change |
|---|---:|---:|---:|
| First10 motor RMSE | 0.044296774 | 0.045226322 | +2.098% |
| Future latent NMSE | 0.060877371 | 0.065914399 | +8.274% |
| Mixed loss | 0.004598652 | 0.004893894 | +6.420% |
| Full32 motor RMSE | 0.047500980 | 0.053483246 | +12.594% |
| Late22 motor RMSE | 0.048528504 | 0.055737509 | +14.855% |
| First10 gripper RMSE | 0.057054031 | 0.058941074 | +3.307% |
| First10 motor MSE | 0.002014132 | 0.002095530 | +4.041% |

First10 motor RMSE 的 case 配对计数：improved 14，worsened 26，tied 0（共 40 cases）。

### Joint8 vs R8_VQ8

| Metric | Baseline | Candidate | Relative change |
|---|---:|---:|---:|
| First10 motor RMSE | 0.046408990 | 0.045226322 | -2.548% |
| Future latent NMSE | 0.065187050 | 0.065914399 | +1.116% |
| Mixed loss | 0.004980800 | 0.004893894 | -1.745% |
| Full32 motor RMSE | 0.050613965 | 0.053483246 | +5.669% |
| Late22 motor RMSE | 0.051931984 | 0.055737509 | +7.328% |
| First10 gripper RMSE | 0.060208080 | 0.058941074 | -2.104% |
| First10 motor MSE | 0.002213315 | 0.002095530 | -5.322% |

First10 motor RMSE 的 case 配对计数：improved 23，worsened 17，tied 0（共 40 cases）。

### Joint8 vs H_currentpath

| Metric | Baseline | Candidate | Relative change |
|---|---:|---:|---:|
| First10 motor RMSE | 0.045305496 | 0.045226322 | -0.175% |
| Future latent NMSE | 0.065714797 | 0.065914399 | +0.304% |
| Mixed loss | 0.004894043 | 0.004893894 | -0.003% |
| Full32 motor RMSE | 0.049968888 | 0.053483246 | +7.033% |
| Late22 motor RMSE | 0.051394088 | 0.055737509 | +8.451% |
| First10 gripper RMSE | 0.051747971 | 0.058941074 | +13.900% |
| First10 motor MSE | 0.002104152 | 0.002095530 | -0.410% |

First10 motor RMSE 的 case 配对计数：improved 20，worsened 20，tied 0（共 40 cases）。

### Joint8 vs Joint8_R1e4

| Metric | Baseline | Candidate | Relative change |
|---|---:|---:|---:|
| First10 motor RMSE | 0.045524040 | 0.045226322 | -0.654% |
| Future latent NMSE | 0.060773535 | 0.065914399 | +8.459% |
| Mixed loss | 0.004718190 | 0.004893894 | +3.724% |
| Full32 motor RMSE | 0.048873038 | 0.053483246 | +9.433% |
| Late22 motor RMSE | 0.049926803 | 0.055737509 | +11.638% |
| First10 gripper RMSE | 0.042294998 | 0.058941074 | +39.357% |
| First10 motor MSE | 0.002138078 | 0.002095530 | -1.990% |

First10 motor RMSE 的 case 配对计数：improved 25，worsened 15，tied 0（共 40 cases）。

### H_currentpath vs H_VQ8

| Metric | Baseline | Candidate | Relative change |
|---|---:|---:|---:|
| First10 motor RMSE | 0.044296774 | 0.045305496 | +2.277% |
| Future latent NMSE | 0.060877371 | 0.065714797 | +7.946% |
| Mixed loss | 0.004598652 | 0.004894043 | +6.423% |
| Full32 motor RMSE | 0.047500980 | 0.049968888 | +5.195% |
| Late22 motor RMSE | 0.048528504 | 0.051394088 | +5.905% |
| First10 gripper RMSE | 0.057054031 | 0.051747971 | -9.300% |
| First10 motor MSE | 0.002014132 | 0.002104152 | +4.469% |

First10 motor RMSE 的 case 配对计数：improved 15，worsened 25，tied 0（共 40 cases）。

## C8 calibration 阶段表

C8 含8 cases（四 suite 各2）与 seed 0；每格为先按 case 聚合后四 suite 等权的 mean。

| Arm | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| H_W16A4 | 0.044464021 | 0.039335942 | 0.003835097 | 0.044510534 | 0.044311035 | 0.016933079 | 0.002165108 |
| R8_W16A4 | 0.040792220 | 0.034579013 | 0.003145532 | 0.042966808 | 0.043835163 | 0.009841247 | 0.001677496 |
| H_VQ0 | 0.043527374 | 0.068408117 | 0.004840390 | 0.051583920 | 0.054148697 | 0.026231369 | 0.001936156 |
| H_VQ8 | 0.042133243 | 0.054480979 | 0.004196602 | 0.044388045 | 0.044610900 | 0.020049706 | 0.001883638 |
| R8_VQ0 | 0.042909495 | 0.049997734 | 0.004001595 | 0.045753177 | 0.046833398 | 0.026354666 | 0.001878965 |
| R8_VQ8 | 0.038231716 | 0.047025546 | 0.003483861 | 0.041612425 | 0.042874698 | 0.025068609 | 0.001487414 |
| Joint8_R1e4 | 0.040900033 | 0.066418314 | 0.004515739 | 0.045310565 | 0.046645491 | 0.015295556 | 0.001695981 |
| H_currentpath | 0.039669442 | 0.048377061 | 0.003642991 | 0.041893134 | 0.042734562 | 0.021562261 | 0.001589166 |
| Joint8 | 0.041454705 | 0.040662807 | 0.003458731 | 0.040969091 | 0.040609463 | 0.018186816 | 0.001732411 |

## Refinement 判据、路径差异与来源

触发规则在读取本轮数据前（13:10:49 UTC）写定：若第一轮 Joint8 的 C8 mixed loss 不低于冻结 incumbent H_VQ8，则只进行 C8 refinement。本轮触发值为 Joint8 0.0045157389、H_VQ8 0.0041966021。此规则只说明启动本轮的条件，不构成 heldout 选择。
同一 H0 初始化下，旧/当前 forward path 的初始 VQ gradient norm 分别为 0.0293344 与 0.0292923。H_currentpath 用于量化此数值实现差异；仅凭该差值不推断其根因。
H_W16A4、R8_W16A4、H_VQ8、R8_VQ8 及第一轮 Joint8_R1e4 的 heldout/C8 records 来自 source job 25903221；复用来源、新旧训练记录、训练 cost/计时与触发判据只按 worker 传入的 provenance 报告。

| Provenance 项 | worker 传入值 |
|---|---|
| timing | {"control_updates_seconds": 350.517324404791, "joint_updates_seconds": 350.8524468950927, "wall_seconds_this_job": 2417.4071353222243} |
| source_job | "25903221.pbs101" |
| reused_diagnostic_protocol | "fastwam-vq-r128-joint-c8-v1" |
| refinement_gate | {"additional_arm": {"D": 1, "calibration_case_ids": [2000, 2019, 2021, 2025, 2051, 2057, 2066, 2070], "checkpoint": "fixed final update8", "clipping_ratio": 1, "heldout": "same full 40 contexts x 2 seeds; reference arms and all initial diagnostics reused", "initialization": "same original H+VQ0, U=I", "joint_updates": 8, "rho": 1.5, "rotation_learning_rate": 1e-05, "vq_learning_rate": 0.0005}, "additional_control_reason": "First H+VQ0 VQ gradient norm is .02929229848 in the current path versus .02933440171 in the legacy H8 run; quantify implementation numerics with a frozen-U matched control.", "candidate": "Joint8", "condition": "If Joint8 complete-C8 mixed_loss is greater than or equal to H_VQ8, run one additional Joint8 arm with rotation LR 1e-5; otherwise report the completed comparison without an LR sweep.", "decision_basis": "complete frozen C8 metrics only; no heldout-based LR or checkpoint selection", "declared_before_reading_joint_results": true, "declared_utc": "2026-10-21T13:10:49Z", "incumbent": "H_VQ8", "metric": "mixed_loss", "observed_C8_H_VQ8_mixed_loss": 0.004196602109004743, "observed_C8_Joint8_R1e4_mixed_loss": 0.0045157388667576015, "triggered": true} |
| original_incumbent | "H_VQ8" |
| matched_implementation_control | "H_currentpath" |
| parities | {"H_currentpath": 0.0, "Joint8": 0.0} |
| orthogonality | {"H_currentpath": {"action": 0.0, "video": 0.0}, "Joint8": {"action": 4.76837158203125e-07, "video": 3.5762786865234375e-07}} |
| residual_frobenius_from_identity | {"H_currentpath": {"action": 0.0, "video": 0.0}, "Joint8": {"action": 0.010209036991000175, "video": 0.008602367714047432}} |
| training_history | [{"action_steps": 10, "after": {"first10gripper_rmse": 0.02634909749031067, "first10motor_mse": 0.0016435000579804182, "first10motor_rmse": 0.04054010286927223, "full32motor_rmse": 0.05055873841047287, "future_latent_nmse": 0.0222926065325737, "late22motor_rmse": 0.05450727790594101, "mixed_loss": 0.0025899221654981375}, "before": {"first10gripper_rmse": 0.0267799012362957, "first10motor_mse": 0.0020414532627910376, "first10motor_rmse": 0.04518244415521622, "full32motor_rmse": 0.04940728843212128, "future_latent_nmse": 0.0904831811785698, "late22motor_rmse": 0.05121258646249771, "mixed_loss": 0.005882874131202698}, "case_id": 2000, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.17425835132598877, "rotation_gradients": {"action": true, "video": true}, "seconds": 43.96231750911102, "update": 1, "video_steps": 10, "vq_gradient_norm": 0.029321327805519104}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.02267644554376602, "first10motor_mse": 0.0045516518875956535, "first10motor_rmse": 0.06746593117713928, "full32motor_rmse": 0.06309344619512558, "future_latent_nmse": 0.07557697594165802, "late22motor_rmse": 0.06100242957472801, "mixed_loss": 0.007760236505419016}, "before": {"first10gripper_rmse": 0.018028095364570618, "first10motor_mse": 0.0037165482062846422, "first10motor_rmse": 0.060963500291109085, "full32motor_rmse": 0.05621262267231941, "future_latent_nmse": 0.07211178541183472, "late22motor_rmse": 0.05391491577029228, "mixed_loss": 0.006778019946068525}, "case_id": 2019, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.5362287163734436, "rotation_gradients": {"action": true, "video": true}, "seconds": 43.816849110182375, "update": 2, "video_steps": 10, "vq_gradient_norm": 0.07916832715272903}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.03599265590310097, "first10motor_mse": 0.0011653989786282182, "first10motor_rmse": 0.034137941896915436, "full32motor_rmse": 0.038183700293302536, "future_latent_nmse": 0.02250252291560173, "late22motor_rmse": 0.03988725692033768, "mixed_loss": 0.002120732795447111}, "before": {"first10gripper_rmse": 0.026808375492691994, "first10motor_mse": 0.0013227842282503843, "first10motor_rmse": 0.03637010231614113, "full32motor_rmse": 0.038901034742593765, "future_latent_nmse": 0.029608646407723427, "late22motor_rmse": 0.039998553693294525, "mixed_loss": 0.002579805441200733}, "case_id": 2021, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.047902531921863556, "rotation_gradients": {"action": true, "video": true}, "seconds": 44.73795384820551, "update": 3, "video_steps": 10, "vq_gradient_norm": 0.007256684359163046}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.014245768077671528, "first10motor_mse": 0.0011541832936927676, "first10motor_rmse": 0.033973272889852524, "full32motor_rmse": 0.03715769201517105, "future_latent_nmse": 0.020585039630532265, "late22motor_rmse": 0.03851822391152382, "mixed_loss": 0.00202811136841774}, "before": {"first10gripper_rmse": 0.019841289147734642, "first10motor_mse": 0.0015853791264817119, "first10motor_rmse": 0.03981681913137436, "full32motor_rmse": 0.04164925962686539, "future_latent_nmse": 0.047075141221284866, "late22motor_rmse": 0.04245604947209358, "mixed_loss": 0.00358393182978034}, "case_id": 2025, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.09993130713701248, "rotation_gradients": {"action": true, "video": true}, "seconds": 43.673780429176986, "update": 4, "video_steps": 10, "vq_gradient_norm": 0.013909432105720043}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.021108213812112808, "first10motor_mse": 0.0015524388290941715, "first10motor_rmse": 0.03940099850296974, "full32motor_rmse": 0.048672646284103394, "future_latent_nmse": 0.07881220430135727, "late22motor_rmse": 0.05234694480895996, "mixed_loss": 0.004898373503237963}, "before": {"first10gripper_rmse": 0.025042688474059105, "first10motor_mse": 0.0027645020745694637, "first10motor_rmse": 0.052578531205654144, "full32motor_rmse": 0.04560282453894615, "future_latent_nmse": 0.08885683864355087, "late22motor_rmse": 0.04205123707652092, "mixed_loss": 0.006536876782774925}, "case_id": 2051, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.4127466380596161, "rotation_gradients": {"action": true, "video": true}, "seconds": 43.93092671316117, "update": 5, "video_steps": 10, "vq_gradient_norm": 0.05925299972295761}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.015819106251001358, "first10motor_mse": 0.002353051444515586, "first10motor_rmse": 0.048508260399103165, "full32motor_rmse": 0.054782502353191376, "future_latent_nmse": 0.05739922448992729, "late22motor_rmse": 0.05740819498896599, "mixed_loss": 0.004789908416569233}, "before": {"first10gripper_rmse": 0.018855497241020203, "first10motor_mse": 0.001999847125262022, "first10motor_rmse": 0.04471965134143829, "full32motor_rmse": 0.0599512942135334, "future_latent_nmse": 0.10789842158555984, "late22motor_rmse": 0.065717913210392, "mixed_loss": 0.006580623332411051}, "case_id": 2057, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.11853009462356567, "rotation_gradients": {"action": true, "video": true}, "seconds": 43.612492692656815, "update": 6, "video_steps": 10, "vq_gradient_norm": 0.018229437991976738}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.015819106251001358, "first10motor_mse": 0.0023699935991317034, "first10motor_rmse": 0.04868258163332939, "full32motor_rmse": 0.04007771238684654, "future_latent_nmse": 0.03493712842464447, "late22motor_rmse": 0.03548315912485123, "mixed_loss": 0.0038532328326255083}, "before": {"first10gripper_rmse": 0.021993262693285942, "first10motor_mse": 0.002740088850259781, "first10motor_rmse": 0.05234585702419281, "full32motor_rmse": 0.04185359179973602, "future_latent_nmse": 0.034096065908670425, "late22motor_rmse": 0.0360896959900856, "mixed_loss": 0.004187621176242828}, "case_id": 2066, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.06364855915307999, "rotation_gradients": {"action": true, "video": true}, "seconds": 43.60844196239486, "update": 7, "video_steps": 10, "vq_gradient_norm": 0.009579366073012352}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.013699748553335667, "first10motor_mse": 0.0015038781566545367, "first10motor_rmse": 0.03877986967563629, "full32motor_rmse": 0.03862442821264267, "future_latent_nmse": 0.033850040286779404, "late22motor_rmse": 0.03855356574058533, "mixed_loss": 0.00294096558354795}, "before": {"first10gripper_rmse": 0.023632006719708443, "first10motor_mse": 0.0020079996902495623, "first10motor_rmse": 0.0448107086122036, "full32motor_rmse": 0.04162558540701866, "future_latent_nmse": 0.04865020141005516, "late22motor_rmse": 0.04009424149990082, "mixed_loss": 0.0040734210051596165}, "case_id": 2070, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.13145077228546143, "rotation_gradients": {"action": true, "video": true}, "seconds": 43.50968463020399, "update": 8, "video_steps": 10, "vq_gradient_norm": 0.018942441791296005}] |
| control_training_history | [{"action_steps": 10, "after": {"first10gripper_rmse": 0.032069336622953415, "first10motor_mse": 0.0020425906404852867, "first10motor_rmse": 0.04519502818584442, "full32motor_rmse": 0.05514226853847504, "future_latent_nmse": 0.07100166380405426, "late22motor_rmse": 0.05911296233534813, "mixed_loss": 0.00505693256855011}, "before": {"first10gripper_rmse": 0.0267799012362957, "first10motor_mse": 0.0020414532627910376, "first10motor_rmse": 0.04518244415521622, "full32motor_rmse": 0.04940728843212128, "future_latent_nmse": 0.0904831811785698, "late22motor_rmse": 0.05121258646249771, "mixed_loss": 0.005882874131202698}, "case_id": 2000, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.0, "rotation_gradients": {}, "seconds": 47.66643106425181, "update": 1, "video_steps": 10, "vq_gradient_norm": 0.029285509139299393}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.019334951415657997, "first10motor_mse": 0.002807718701660633, "first10motor_rmse": 0.05298791080713272, "full32motor_rmse": 0.048785801976919174, "future_latent_nmse": 0.046692799776792526, "late22motor_rmse": 0.04675106331706047, "mixed_loss": 0.004790039267390966}, "before": {"first10gripper_rmse": 0.02907467819750309, "first10motor_mse": 0.00634849863126874, "first10motor_rmse": 0.07967746257781982, "full32motor_rmse": 0.07096339762210846, "future_latent_nmse": 0.08539343625307083, "late22motor_rmse": 0.06662675738334656, "mixed_loss": 0.00997383613139391}, "case_id": 2019, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.0, "rotation_gradients": {}, "seconds": 43.26948879100382, "update": 2, "video_steps": 10, "vq_gradient_norm": 0.2558300495147705}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.026522258296608925, "first10motor_mse": 0.0010814644629135728, "first10motor_rmse": 0.03288562595844269, "full32motor_rmse": 0.03719610720872879, "future_latent_nmse": 0.017375225201249123, "late22motor_rmse": 0.038998235017061234, "mixed_loss": 0.0018191214185208082}, "before": {"first10gripper_rmse": 0.03358012065291405, "first10motor_mse": 0.0009160899790003896, "first10motor_rmse": 0.030266977846622467, "full32motor_rmse": 0.03895842283964157, "future_latent_nmse": 0.031970541924238205, "late22motor_rmse": 0.042323097586631775, "mixed_loss": 0.002273384016007185}, "case_id": 2021, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.0, "rotation_gradients": {}, "seconds": 43.12169240415096, "update": 3, "video_steps": 10, "vq_gradient_norm": 0.008530003018677235}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.018154609948396683, "first10motor_mse": 0.0017876422498375177, "first10motor_rmse": 0.042280517518520355, "full32motor_rmse": 0.04334964603185654, "future_latent_nmse": 0.02889888733625412, "late22motor_rmse": 0.04382699355483055, "mixed_loss": 0.0030145307537168264}, "before": {"first10gripper_rmse": 0.014511074870824814, "first10motor_mse": 0.001565110287629068, "first10motor_rmse": 0.039561476558446884, "full32motor_rmse": 0.03917620703577995, "future_latent_nmse": 0.02725219540297985, "late22motor_rmse": 0.038999829441308975, "mixed_loss": 0.0027220891788601875}, "case_id": 2025, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.0, "rotation_gradients": {}, "seconds": 43.585067364852875, "update": 4, "video_steps": 10, "vq_gradient_norm": 0.026354623958468437}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.02385692484676838, "first10motor_mse": 0.0013149023288860917, "first10motor_rmse": 0.036261580884456635, "full32motor_rmse": 0.048567116260528564, "future_latent_nmse": 0.034495025873184204, "late22motor_rmse": 0.053228262811899185, "mixed_loss": 0.0027793722692877054}, "before": {"first10gripper_rmse": 0.02324136719107628, "first10motor_mse": 0.0016967873089015484, "first10motor_rmse": 0.04119207710027695, "full32motor_rmse": 0.04993584752082825, "future_latent_nmse": 0.062208693474531174, "late22motor_rmse": 0.0534394197165966, "mixed_loss": 0.004337827675044537}, "case_id": 2051, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.0, "rotation_gradients": {}, "seconds": 42.86014885827899, "update": 5, "video_steps": 10, "vq_gradient_norm": 0.018684716895222664}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.018611140549182892, "first10motor_mse": 0.00223504938185215, "first10motor_rmse": 0.04727631062269211, "full32motor_rmse": 0.05732449144124985, "future_latent_nmse": 0.09491893649101257, "late22motor_rmse": 0.06135019287467003, "mixed_loss": 0.006264787632972002}, "before": {"first10gripper_rmse": 0.021323975175619125, "first10motor_mse": 0.0016393838450312614, "first10motor_rmse": 0.040489304810762405, "full32motor_rmse": 0.05707821622490883, "future_latent_nmse": 0.05416092649102211, "late22motor_rmse": 0.0631951093673706, "mixed_loss": 0.003938760608434677}, "case_id": 2057, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.0, "rotation_gradients": {}, "seconds": 43.47238074289635, "update": 6, "video_steps": 10, "vq_gradient_norm": 0.010311329737305641}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.015378921292722225, "first10motor_mse": 0.0022521200589835644, "first10motor_rmse": 0.04745650663971901, "full32motor_rmse": 0.04288046061992645, "future_latent_nmse": 0.04229264706373215, "late22motor_rmse": 0.040630415081977844, "mixed_loss": 0.004047634080052376}, "before": {"first10gripper_rmse": 0.02243969775736332, "first10motor_mse": 0.0017170363571494818, "first10motor_rmse": 0.04143713787198067, "full32motor_rmse": 0.0383421927690506, "future_latent_nmse": 0.03407934308052063, "late22motor_rmse": 0.03684956952929497, "mixed_loss": 0.0031638587825000286}, "case_id": 2066, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.0, "rotation_gradients": {}, "seconds": 43.82293615024537, "update": 7, "video_steps": 10, "vq_gradient_norm": 0.007783300708979368}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.01823846623301506, "first10motor_mse": 0.0013968194834887981, "first10motor_rmse": 0.03737404942512512, "full32motor_rmse": 0.043849676847457886, "future_latent_nmse": 0.03327194228768349, "late22motor_rmse": 0.0464959517121315, "mixed_loss": 0.0028093638829886913}, "before": {"first10gripper_rmse": 0.02495112456381321, "first10motor_mse": 0.0014978926628828049, "first10motor_rmse": 0.03870261833071709, "full32motor_rmse": 0.038506295531988144, "future_latent_nmse": 0.03294406458735466, "late22motor_rmse": 0.038416724652051926, "mixed_loss": 0.0028965172823518515}, "case_id": 2070, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.0, "rotation_gradients": {}, "seconds": 42.71917902911082, "update": 8, "video_steps": 10, "vq_gradient_norm": 0.012790639884769917}] |
| new_teacher_queries | 0 |
| lineage | ["25904715.pbs101"] |
| equal_walltime | false |
| fixed_final_checkpoint | true |

本轮 Joint8 与 H_currentpath 从相同 H_VQ0 初始化并使用相同 VQ 更新顺序、loss 与 learning rate；Joint8 另以 rotation learning rate=1e-5 更新 R，因此增加 R 参数与计算。Joint8 直接对 compressed student 做 coadaptation；训练中间的 R 不保证每一步都等价于原 FP 函数。

该结果只描述固定输入的 decoded-BF16/A4 配对指标。paired case 计数不是显著性检验；不构成 closed-loop、native-kernel、延迟或 equal-compute/walltime 结论。
