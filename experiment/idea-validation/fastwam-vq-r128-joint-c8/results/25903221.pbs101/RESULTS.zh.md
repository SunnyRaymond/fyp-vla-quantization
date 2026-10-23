# Fast-WAM R128 Joint8 固定输入聚合结果

Protocol：`fastwam-vq-r128-joint-c8-v1`；target Linears：614；heldout 每臂40 cases × 2 seeds = 80 queries。
heldout 按 seed → case → suite 聚合，四个 suite 等权；相对变化为 candidate / baseline - 1，负值表示误差下降。

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
| First10 motor RMSE | 0.044296774 | 0.045524040 | +2.771% |
| Future latent NMSE | 0.060877371 | 0.060773535 | -0.171% |
| Mixed loss | 0.004598652 | 0.004718190 | +2.599% |
| Full32 motor RMSE | 0.047500980 | 0.048873038 | +2.888% |
| Late22 motor RMSE | 0.048528504 | 0.049926803 | +2.881% |
| First10 gripper RMSE | 0.057054031 | 0.042294998 | -25.869% |
| First10 motor MSE | 0.002014132 | 0.002138078 | +6.154% |

First10 motor RMSE 的 case 配对计数：improved 18，worsened 22，tied 0（共 40 cases）。

### Joint8 vs R8_VQ8

| Metric | Baseline | Candidate | Relative change |
|---|---:|---:|---:|
| First10 motor RMSE | 0.046408990 | 0.045524040 | -1.907% |
| Future latent NMSE | 0.065187050 | 0.060773535 | -6.771% |
| Mixed loss | 0.004980800 | 0.004718190 | -5.272% |
| Full32 motor RMSE | 0.050613965 | 0.048873038 | -3.440% |
| Late22 motor RMSE | 0.051931984 | 0.049926803 | -3.861% |
| First10 gripper RMSE | 0.060208080 | 0.042294998 | -29.752% |
| First10 motor MSE | 0.002213315 | 0.002138078 | -3.399% |

First10 motor RMSE 的 case 配对计数：improved 19，worsened 21，tied 0（共 40 cases）。

## C8 calibration 阶段表

C8 含8 cases（四 suite 各2）与 seed 0；每格为先按 case 聚合后四 suite 等权的 mean。

| Arm | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| H_W16A4 | 0.044464021 | 0.039335942 | 0.003835097 | 0.044510534 | 0.044311035 | 0.016933079 | 0.002165108 |
| R8_W16A4 | 0.040792220 | 0.034579013 | 0.003145532 | 0.042966808 | 0.043835163 | 0.009841247 | 0.001677496 |
| H_VQ8 | 0.042133243 | 0.054480979 | 0.004196602 | 0.044388045 | 0.044610900 | 0.020049706 | 0.001883638 |
| R8_VQ8 | 0.038231716 | 0.047025546 | 0.003483861 | 0.041612425 | 0.042874698 | 0.025068609 | 0.001487414 |
| Joint8 | 0.040900033 | 0.066418314 | 0.004515739 | 0.045310565 | 0.046645491 | 0.015295556 | 0.001695981 |
| H_VQ0 | 0.043527374 | 0.068408117 | 0.004840390 | 0.051583920 | 0.054148697 | 0.026231369 | 0.001936156 |
| R8_VQ0 | 0.042909495 | 0.049997734 | 0.004001595 | 0.045753177 | 0.046833398 | 0.026354666 | 0.001878965 |

## 复用与训练成本来源

H_VQ8 与 R8_VQ8 的旧 VQ endpoint records 可复用；复用来源与训练成本逐项照录 worker 传入的 provenance。

| Provenance 项 | worker 传入值 |
|---|---|
| source_rotation_job | "25902163.pbs101" |
| baseline_validation | {"complete": true, "failure": "Runtime.close AttributeError after complete numerical results", "new_model_forwards": 0, "original_failure_artifacts_preserved": true, "pbs_jobid": "25848931.pbs101", "protocol": "fastwam-vq-activation-clipping-v1", "purpose": "validation_only", "queries": 80, "recomputed_from_query_records": true, "source_exit_code": 1, "source_job_id": "25848310.pbs101"} |
| parent_training_receipt | "/scratch/users/ntu/yguo017/fastwam-vq-c8-action-latent-plus-20261015/results/training/rho1.5/training_receipt.json" |
| orthogonality | {"action": 2.5033950805664062e-06, "video": 3.933906555175781e-06} |
| residual_frobenius_from_identity | {"action": 0.09206770360469818, "video": 0.08708412200212479} |
| initial_parity | {"first10gripper_rmse": 0.0, "first10motor_mse": 0.0, "first10motor_rmse": 0.0, "full32motor_rmse": 0.0, "future_latent_nmse": 0.0, "late22motor_rmse": 0.0, "mixed_loss": 0.0} |
| final_formal_infer_max_abs | 0.0 |
| new_teacher_queries | 0 |
| training_history | [{"action_steps": 10, "after": {"first10gripper_rmse": 0.03647693246603012, "first10motor_mse": 0.0011804501991719007, "first10motor_rmse": 0.03435768187046051, "full32motor_rmse": 0.04279467090964317, "future_latent_nmse": 0.038590457290410995, "late22motor_rmse": 0.04612226039171219, "mixed_loss": 0.0028187897987663746}, "before": {"first10gripper_rmse": 0.0267799012362957, "first10motor_mse": 0.0020414532627910376, "first10motor_rmse": 0.04518244415521622, "full32motor_rmse": 0.04940728843212128, "future_latent_nmse": 0.0904831811785698, "late22motor_rmse": 0.05121258646249771, "mixed_loss": 0.005882874131202698}, "case_id": 2000, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.1741757094860077, "rotation_gradients": {"action": true, "video": true}, "seconds": 47.72248768573627, "update": 1, "video_steps": 10, "vq_gradient_norm": 0.02929229848086834}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.019295452162623405, "first10motor_mse": 0.0017788901459425688, "first10motor_rmse": 0.04217689111828804, "full32motor_rmse": 0.04180970788002014, "future_latent_nmse": 0.03472016751766205, "late22motor_rmse": 0.041641734540462494, "mixed_loss": 0.003252918366342783}, "before": {"first10gripper_rmse": 0.02223476581275463, "first10motor_mse": 0.012330800294876099, "first10motor_rmse": 0.11104413866996765, "full32motor_rmse": 0.15677501261234283, "future_latent_nmse": 0.3338119387626648, "late22motor_rmse": 0.17362457513809204, "mixed_loss": 0.02650262787938118}, "case_id": 2019, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 3.630892753601074, "rotation_gradients": {"action": true, "video": true}, "seconds": 44.94536098698154, "update": 2, "video_steps": 10, "vq_gradient_norm": 0.497006893157959}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.03432169184088707, "first10motor_mse": 0.001427880022674799, "first10motor_rmse": 0.0377872996032238, "full32motor_rmse": 0.0397830456495285, "future_latent_nmse": 0.02846929430961609, "late22motor_rmse": 0.040657833218574524, "mixed_loss": 0.0026365304365754128}, "before": {"first10gripper_rmse": 0.021502124145627022, "first10motor_mse": 0.0012569705722853541, "first10motor_rmse": 0.035453781485557556, "full32motor_rmse": 0.037958309054374695, "future_latent_nmse": 0.02797439508140087, "late22motor_rmse": 0.03904365748167038, "mixed_loss": 0.002444610232487321}, "case_id": 2021, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.05790546163916588, "rotation_gradients": {"action": true, "video": true}, "seconds": 44.02540377480909, "update": 3, "video_steps": 10, "vq_gradient_norm": 0.008864186704158783}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.013131043873727322, "first10motor_mse": 0.001697362633422017, "first10motor_rmse": 0.04119906201958656, "full32motor_rmse": 0.044381801038980484, "future_latent_nmse": 0.051781944930553436, "late22motor_rmse": 0.04575537145137787, "mixed_loss": 0.0038957404904067516}, "before": {"first10gripper_rmse": 0.013304208405315876, "first10motor_mse": 0.0016596288187429309, "first10motor_rmse": 0.0407385416328907, "full32motor_rmse": 0.03863389045000076, "future_latent_nmse": 0.04001466557383537, "late22motor_rmse": 0.0376383513212204, "mixed_loss": 0.0033584325574338436}, "case_id": 2025, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.12118762731552124, "rotation_gradients": {"action": true, "video": true}, "seconds": 44.38795960508287, "update": 4, "video_steps": 10, "vq_gradient_norm": 0.01842857152223587}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.011653464287519455, "first10motor_mse": 0.001683473470620811, "first10motor_rmse": 0.041030153632164, "full32motor_rmse": 0.049185123294591904, "future_latent_nmse": 0.05086787790060043, "late22motor_rmse": 0.05247463285923004, "mixed_loss": 0.0038430453278124332}, "before": {"first10gripper_rmse": 0.014192110858857632, "first10motor_mse": 0.00200854055583477, "first10motor_rmse": 0.044816743582487106, "full32motor_rmse": 0.04982379078865051, "future_latent_nmse": 0.08122732490301132, "late22motor_rmse": 0.051940400153398514, "mixed_loss": 0.005457008257508278}, "case_id": 2051, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.12324850261211395, "rotation_gradients": {"action": true, "video": true}, "seconds": 44.342725766822696, "update": 5, "video_steps": 10, "vq_gradient_norm": 0.019211914390325546}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.013304208405315876, "first10motor_mse": 0.0015504587208852172, "first10motor_rmse": 0.03937586396932602, "full32motor_rmse": 0.04895197972655296, "future_latent_nmse": 0.08656471967697144, "late22motor_rmse": 0.052732985466718674, "mixed_loss": 0.00522552290931344}, "before": {"first10gripper_rmse": 0.016010861843824387, "first10motor_mse": 0.0015307764988392591, "first10motor_rmse": 0.03912514075636864, "full32motor_rmse": 0.04778548330068588, "future_latent_nmse": 0.0770522803068161, "late22motor_rmse": 0.05124039575457573, "mixed_loss": 0.004801994189620018}, "case_id": 2057, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.08597490191459656, "rotation_gradients": {"action": true, "video": true}, "seconds": 43.1776455482468, "update": 6, "video_steps": 10, "vq_gradient_norm": 0.013408642262220383}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.015329232439398766, "first10motor_mse": 0.0021549237426370382, "first10motor_rmse": 0.04642115533351898, "full32motor_rmse": 0.040839727967977524, "future_latent_nmse": 0.04101170599460602, "late22motor_rmse": 0.0380329005420208, "mixed_loss": 0.0038960562087595463}, "before": {"first10gripper_rmse": 0.020297469571232796, "first10motor_mse": 0.0018051458755508065, "first10motor_rmse": 0.0424870066344738, "full32motor_rmse": 0.03836095705628395, "future_latent_nmse": 0.044744160026311874, "late22motor_rmse": 0.03633090481162071, "mixed_loss": 0.0037047378718852997}, "case_id": 2066, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.04186922684311867, "rotation_gradients": {"action": true, "video": true}, "seconds": 44.22394779836759, "update": 7, "video_steps": 10, "vq_gradient_norm": 0.0071042426861822605}, {"action_steps": 10, "after": {"first10gripper_rmse": 0.01675594598054886, "first10motor_mse": 0.0017822758527472615, "first10motor_rmse": 0.042217008769512177, "full32motor_rmse": 0.0393071323633194, "future_latent_nmse": 0.05517907813191414, "late22motor_rmse": 0.03791069984436035, "mixed_loss": 0.004124877508729696}, "before": {"first10gripper_rmse": 0.0177293848246336, "first10motor_mse": 0.001170009607449174, "first10motor_rmse": 0.03420540317893028, "full32motor_rmse": 0.0382746085524559, "future_latent_nmse": 0.03831157460808754, "late22motor_rmse": 0.03998759388923645, "mixed_loss": 0.002796509303152561}, "case_id": 2070, "coverage": {"connected": 614, "disconnected_modules": [], "finite": 614, "nonzero": 614, "targets": 614, "zero_gradient_modules": []}, "rotation_gradient_norm": 0.058548688888549805, "rotation_gradients": {"action": true, "video": true}, "seconds": 43.21140003204346, "update": 8, "video_steps": 10, "vq_gradient_norm": 0.01265609823167324}] |
| lineage | ["25903221.pbs101"] |
| timing | {"joint_updates_seconds": 356.0369311980903, "wall_seconds_this_job": 4489.6574572171085} |
| same_vq_updates | true |
| equal_walltime | false |
| fixed_final_checkpoint | true |

Joint8 与 H_VQ8 从同一 H_VQ0 初始化，均按相同 VQ learning rate、顺序、loss 完成8次更新；Joint8 额外包含 R 参数及计算。Joint8 直接对 compressed student 做 coadaptation；训练中间的 R 不保证每一步都等价于原 FP 函数。

该结果只描述固定输入的 decoded-BF16/A4 配对指标。paired case 计数不是显著性检验；不构成 closed-loop、native-kernel、延迟或 equal-compute/walltime 结论。
