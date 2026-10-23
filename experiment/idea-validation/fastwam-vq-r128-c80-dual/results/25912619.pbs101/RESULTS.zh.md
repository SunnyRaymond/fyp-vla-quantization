# Fast-WAM R128 C80 dual-arm 固定输入结果

Protocol：`fastwam-vq-r128-c80-dual-v1`；目标 Linears：614。
校准集为 cases 2000–2079、seed 0；heldout 为 cases 3000–3039、seeds 0/1。每个 heldout arm 含 40 contexts × 2 seeds = 80 queries。
聚合顺序为 seed → context → suite，四个 suite 等权；表格报告七项固定输入指标。

Seq8 与 Joint8 均从共用 R0/VQ0 起点训练 8 次 VQ 更新、每次 10 contexts。Seq8 冻结 R；Joint8 另以 LR=1e-5 做 8 次 R 更新，因此增加 R 参数更新与计算。VQ LR=5e-4。
所有新臂使用固定 final update 8；不按 heldout 结果挑选 checkpoint 或超参数。

## C80 calibration：按 suite

| Arm / suite | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0_VQ0 / libero_spatial | 0.051893747 | 0.045905758 | 0.004890658 | 0.050888901 | 0.050045457 | 0.030374545 | 0.002941751 |
| R0_VQ0 / libero_object | 0.046374535 | 0.061464733 | 0.004833476 | 0.048514023 | 0.049091724 | 0.028744923 | 0.002224019 |
| R0_VQ0 / libero_goal | 0.051838198 | 0.067994869 | 0.005849739 | 0.053238130 | 0.053582214 | 0.030979583 | 0.002963049 |
| R0_VQ0 / libero_10 | 0.046779118 | 0.047911272 | 0.004281553 | 0.047796744 | 0.047986869 | 0.058647523 | 0.002247503 |
| Seq8 / libero_spatial | 0.045684844 | 0.036840962 | 0.003689356 | 0.044999804 | 0.044587050 | 0.022442832 | 0.002125291 |
| Seq8 / libero_object | 0.041443934 | 0.057514354 | 0.004190813 | 0.045822463 | 0.047383655 | 0.021723868 | 0.001749068 |
| Seq8 / libero_goal | 0.044644078 | 0.058348360 | 0.004520592 | 0.048626162 | 0.049885252 | 0.023577753 | 0.002043440 |
| Seq8 / libero_10 | 0.041932489 | 0.038416647 | 0.003415253 | 0.042210168 | 0.042200734 | 0.114379755 | 0.001784292 |
| Joint8 / libero_spatial | 0.044079869 | 0.040701554 | 0.003707987 | 0.044583590 | 0.044691047 | 0.016569957 | 0.001980022 |
| Joint8 / libero_object | 0.044507698 | 0.057953970 | 0.004526615 | 0.046879080 | 0.047651324 | 0.056173522 | 0.002066207 |
| Joint8 / libero_goal | 0.043102413 | 0.052596331 | 0.004105089 | 0.045994570 | 0.047023487 | 0.015629918 | 0.001872136 |
| Joint8 / libero_10 | 0.040785933 | 0.036353092 | 0.003217656 | 0.042834906 | 0.043562116 | 0.013393572 | 0.001674302 |

## Heldout：按 suite

| Arm / suite | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0_VQ0 / libero_spatial | 0.048226818 | 0.056051135 | 0.004781933 | 0.050519275 | 0.051272730 | 0.059004981 | 0.002402308 |
| R0_VQ0 / libero_object | 0.045104932 | 0.066084329 | 0.004865631 | 0.052326877 | 0.054665712 | 0.026360309 | 0.002060052 |
| R0_VQ0 / libero_goal | 0.048894958 | 0.060441985 | 0.004982226 | 0.056776115 | 0.059222918 | 0.063548259 | 0.002416190 |
| R0_VQ0 / libero_10 | 0.050284918 | 0.054287354 | 0.004891812 | 0.047748098 | 0.046267685 | 0.030253209 | 0.002587068 |
| Seq8 / libero_spatial | 0.045343076 | 0.049284682 | 0.004192306 | 0.045651768 | 0.045591163 | 0.021719785 | 0.002099948 |
| Seq8 / libero_object | 0.042808477 | 0.053358070 | 0.004135390 | 0.046802737 | 0.048276618 | 0.021658493 | 0.001870099 |
| Seq8 / libero_goal | 0.045663704 | 0.065205107 | 0.004895539 | 0.056891771 | 0.060266349 | 0.087467894 | 0.002127287 |
| Seq8 / libero_10 | 0.043015462 | 0.037755514 | 0.003477861 | 0.042462196 | 0.042078458 | 0.021269611 | 0.001874968 |
| Joint8 / libero_spatial | 0.047286104 | 0.062522764 | 0.005003836 | 0.050729395 | 0.051874745 | 0.012799064 | 0.002349462 |
| Joint8 / libero_object | 0.045441512 | 0.060559112 | 0.004691350 | 0.045597330 | 0.045449831 | 0.014540444 | 0.002120341 |
| Joint8 / libero_goal | 0.046098503 | 0.070422722 | 0.005159528 | 0.057356675 | 0.060900373 | 0.111604022 | 0.002169764 |
| Joint8 / libero_10 | 0.044173668 | 0.044375619 | 0.003874840 | 0.043832674 | 0.043536047 | 0.058310084 | 0.001990894 |
| H_VQ8 / libero_spatial | 0.044794729 | 0.053923931 | 0.004318821 | 0.046173595 | 0.046622019 | 0.046753075 | 0.002029506 |
| H_VQ8 / libero_object | 0.041680142 | 0.065074514 | 0.004552778 | 0.045767309 | 0.047370207 | 0.017308175 | 0.001790070 |
| H_VQ8 / libero_goal | 0.045092065 | 0.068296552 | 0.004991575 | 0.052540081 | 0.054876660 | 0.114934304 | 0.002092078 |
| H_VQ8 / libero_10 | 0.045620161 | 0.056214487 | 0.004531433 | 0.045522934 | 0.045245130 | 0.049220567 | 0.002144873 |
| R0_W16A4 / libero_spatial | 0.042614048 | 0.038612098 | 0.003475875 | 0.044634822 | 0.045396224 | 0.019361822 | 0.001836617 |
| R0_W16A4 / libero_object | 0.041334382 | 0.034790134 | 0.003222960 | 0.044258823 | 0.045316768 | 0.017360403 | 0.001745961 |
| R0_W16A4 / libero_goal | 0.043827714 | 0.042928921 | 0.003772301 | 0.049379558 | 0.051098314 | 0.063862870 | 0.001949774 |
| R0_W16A4 / libero_10 | 0.041999288 | 0.021065031 | 0.002680949 | 0.040840288 | 0.040097673 | 0.050279847 | 0.001786643 |

## Heldout overall：四 suite 等权

| Arm / suite | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0_VQ0 | 0.048127906 | 0.059216201 | 0.004880400 | 0.051842591 | 0.052857261 | 0.044791690 | 0.002366404 |
| Seq8 | 0.044207680 | 0.051400843 | 0.004175274 | 0.047952118 | 0.049053147 | 0.038028946 | 0.001993076 |
| Joint8 | 0.045749947 | 0.059470054 | 0.004682388 | 0.049379018 | 0.050440249 | 0.049313403 | 0.002157615 |
| H_VQ8 | 0.044296774 | 0.060877371 | 0.004598652 | 0.047500980 | 0.048528504 | 0.057054031 | 0.002014132 |
| R0_W16A4 | 0.042443858 | 0.034349046 | 0.003288021 | 0.044778372 | 0.045477245 | 0.037716236 | 0.001829749 |

## 配对比较

每项胜负按同一 40 个 context 配对；先对 seeds 0/1 求均值。误差越低记为 improved。

### Joint8 vs Seq8

| Metric | Baseline | Candidate | Relative change | Context improved / worsened / tied |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 0.044207680 | 0.045749947 | +3.489% | 19 / 21 / 0 of 40 |
| Future latent NMSE | 0.051400843 | 0.059470054 | +15.699% | 17 / 23 / 0 of 40 |
| Mixed loss | 0.004175274 | 0.004682388 | +12.146% | 15 / 25 / 0 of 40 |
| Full32 motor RMSE | 0.047952118 | 0.049379018 | +2.976% | 19 / 21 / 0 of 40 |
| Late22 motor RMSE | 0.049053147 | 0.050440249 | +2.828% | 20 / 20 / 0 of 40 |
| First10 gripper RMSE | 0.038028946 | 0.049313403 | +29.673% | 35 / 5 / 0 of 40 |
| First10 motor MSE | 0.001993076 | 0.002157615 | +8.256% | 19 / 21 / 0 of 40 |

按 suite 的相对变化（candidate / baseline − 1）：

| Suite | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_spatial | +4.285% | +26.860% | +19.358% | +11.123% | +13.782% | -41.072% | +11.882% |
| libero_object | +6.151% | +13.496% | +13.444% | -2.576% | -5.855% | -32.865% | +13.381% |
| libero_goal | +0.952% | +8.002% | +5.392% | +0.817% | +1.052% | +27.594% | +1.997% |
| libero_10 | +2.693% | +17.534% | +11.414% | +3.228% | +3.464% | +174.147% | +6.183% |

### Seq8 vs R0_VQ0

| Metric | Baseline | Candidate | Relative change | Context improved / worsened / tied |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 0.048127906 | 0.044207680 | -8.145% | 32 / 8 / 0 of 40 |
| Future latent NMSE | 0.059216201 | 0.051400843 | -13.198% | 28 / 12 / 0 of 40 |
| Mixed loss | 0.004880400 | 0.004175274 | -14.448% | 32 / 8 / 0 of 40 |
| Full32 motor RMSE | 0.051842591 | 0.047952118 | -7.504% | 31 / 9 / 0 of 40 |
| Late22 motor RMSE | 0.052857261 | 0.049053147 | -7.197% | 30 / 10 / 0 of 40 |
| First10 gripper RMSE | 0.044791690 | 0.038028946 | -15.098% | 36 / 4 / 0 of 40 |
| First10 motor MSE | 0.002366404 | 0.001993076 | -15.776% | 32 / 8 / 0 of 40 |

按 suite 的相对变化（candidate / baseline − 1）：

| Suite | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_spatial | -5.980% | -12.072% | -12.330% | -9.635% | -11.081% | -63.190% | -12.586% |
| libero_object | -5.091% | -19.258% | -15.008% | -10.557% | -11.688% | -17.837% | -9.221% |
| libero_goal | -6.609% | +7.880% | -1.740% | +0.204% | +1.762% | +37.640% | -11.957% |
| libero_10 | -14.457% | -30.452% | -28.904% | -11.070% | -9.054% | -29.695% | -27.525% |

### Joint8 vs R0_VQ0

| Metric | Baseline | Candidate | Relative change | Context improved / worsened / tied |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 0.048127906 | 0.045749947 | -4.941% | 25 / 15 / 0 of 40 |
| Future latent NMSE | 0.059216201 | 0.059470054 | +0.429% | 22 / 18 / 0 of 40 |
| Mixed loss | 0.004880400 | 0.004682388 | -4.057% | 28 / 12 / 0 of 40 |
| Full32 motor RMSE | 0.051842591 | 0.049379018 | -4.752% | 31 / 9 / 0 of 40 |
| Late22 motor RMSE | 0.052857261 | 0.050440249 | -4.573% | 27 / 13 / 0 of 40 |
| First10 gripper RMSE | 0.044791690 | 0.049313403 | +10.095% | 37 / 3 / 0 of 40 |
| First10 motor MSE | 0.002366404 | 0.002157615 | -8.823% | 25 / 15 / 0 of 40 |

按 suite 的相对变化（candidate / baseline − 1）：

| Suite | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_spatial | -1.951% | +11.546% | +4.640% | +0.416% | +1.174% | -78.309% | -2.200% |
| libero_object | +0.746% | -8.361% | -3.582% | -12.861% | -16.859% | -44.840% | +2.927% |
| libero_goal | -5.719% | +16.513% | +3.559% | +1.023% | +2.832% | +75.621% | -10.199% |
| libero_10 | -12.153% | -18.258% | -20.789% | -8.200% | -5.904% | +92.740% | -23.044% |

### Seq8 vs H_VQ8

| Metric | Baseline | Candidate | Relative change | Context improved / worsened / tied |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 0.044296774 | 0.044207680 | -0.201% | 19 / 21 / 0 of 40 |
| Future latent NMSE | 0.060877371 | 0.051400843 | -15.567% | 27 / 13 / 0 of 40 |
| Mixed loss | 0.004598652 | 0.004175274 | -9.207% | 24 / 16 / 0 of 40 |
| Full32 motor RMSE | 0.047500980 | 0.047952118 | +0.950% | 20 / 20 / 0 of 40 |
| Late22 motor RMSE | 0.048528504 | 0.049053147 | +1.081% | 21 / 19 / 0 of 40 |
| First10 gripper RMSE | 0.057054031 | 0.038028946 | -33.346% | 12 / 28 / 0 of 40 |
| First10 motor MSE | 0.002014132 | 0.001993076 | -1.045% | 20 / 20 / 0 of 40 |

按 suite 的相对变化（candidate / baseline − 1）：

| Suite | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_spatial | +1.224% | -8.603% | -2.929% | -1.130% | -2.211% | -53.544% | +3.471% |
| libero_object | +2.707% | -18.005% | -9.168% | +2.262% | +1.913% | +25.134% | +4.471% |
| libero_goal | +1.268% | -4.527% | -1.924% | +8.283% | +9.821% | -23.897% | +1.683% |
| libero_10 | -5.710% | -32.837% | -23.250% | -6.724% | -6.999% | -56.787% | -12.584% |

### Seq8 vs R0_W16A4

| Metric | Baseline | Candidate | Relative change | Context improved / worsened / tied |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 0.042443858 | 0.044207680 | +4.156% | 15 / 25 / 0 of 40 |
| Future latent NMSE | 0.034349046 | 0.051400843 | +49.643% | 6 / 34 / 0 of 40 |
| Mixed loss | 0.003288021 | 0.004175274 | +26.984% | 8 / 32 / 0 of 40 |
| Full32 motor RMSE | 0.044778372 | 0.047952118 | +7.088% | 14 / 26 / 0 of 40 |
| Late22 motor RMSE | 0.045477245 | 0.049053147 | +7.863% | 16 / 24 / 0 of 40 |
| First10 gripper RMSE | 0.037716236 | 0.038028946 | +0.829% | 10 / 30 / 0 of 40 |
| First10 motor MSE | 0.001829749 | 0.001993076 | +8.926% | 16 / 24 / 0 of 40 |

按 suite 的相对变化（candidate / baseline − 1）：

| Suite | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_spatial | +6.404% | +27.641% | +20.612% | +2.278% | +0.429% | +12.178% | +14.338% |
| libero_object | +3.566% | +53.371% | +28.310% | +5.748% | +6.531% | +24.758% | +7.110% |
| libero_goal | +4.189% | +51.891% | +29.776% | +15.213% | +17.942% | +36.962% | +9.104% |
| libero_10 | +2.420% | +79.233% | +29.725% | +3.971% | +4.940% | -57.698% | +4.944% |

### Joint8 vs H_VQ8

| Metric | Baseline | Candidate | Relative change | Context improved / worsened / tied |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 0.044296774 | 0.045749947 | +3.281% | 19 / 21 / 0 of 40 |
| Future latent NMSE | 0.060877371 | 0.059470054 | -2.312% | 22 / 18 / 0 of 40 |
| Mixed loss | 0.004598652 | 0.004682388 | +1.821% | 20 / 20 / 0 of 40 |
| Full32 motor RMSE | 0.047500980 | 0.049379018 | +3.954% | 19 / 21 / 0 of 40 |
| Late22 motor RMSE | 0.048528504 | 0.050440249 | +3.939% | 24 / 16 / 0 of 40 |
| First10 gripper RMSE | 0.057054031 | 0.049313403 | -13.567% | 34 / 6 / 0 of 40 |
| First10 motor MSE | 0.002014132 | 0.002157615 | +7.124% | 19 / 21 / 0 of 40 |

按 suite 的相对变化（candidate / baseline − 1）：

| Suite | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_spatial | +5.562% | +15.946% | +15.861% | +9.867% | +11.267% | -72.624% | +15.765% |
| libero_object | +9.024% | -6.939% | +3.044% | -0.371% | -4.054% | -15.991% | +18.450% |
| libero_goal | +2.232% | +3.113% | +3.365% | +9.167% | +10.977% | -2.898% | +3.713% |
| libero_10 | -3.171% | -21.060% | -14.490% | -3.713% | -3.777% | +18.467% | -7.179% |

### Joint8 vs R0_W16A4

| Metric | Baseline | Candidate | Relative change | Context improved / worsened / tied |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 0.042443858 | 0.045749947 | +7.789% | 11 / 29 / 0 of 40 |
| Future latent NMSE | 0.034349046 | 0.059470054 | +73.135% | 4 / 36 / 0 of 40 |
| Mixed loss | 0.003288021 | 0.004682388 | +42.407% | 4 / 36 / 0 of 40 |
| Full32 motor RMSE | 0.044778372 | 0.049379018 | +10.274% | 12 / 28 / 0 of 40 |
| Late22 motor RMSE | 0.045477245 | 0.050440249 | +10.913% | 13 / 27 / 0 of 40 |
| First10 gripper RMSE | 0.037716236 | 0.049313403 | +30.748% | 32 / 8 / 0 of 40 |
| First10 motor MSE | 0.001829749 | 0.002157615 | +17.919% | 11 / 29 / 0 of 40 |

按 suite 的相对变化（candidate / baseline − 1）：

| Suite | First10 motor RMSE | Future latent NMSE | Mixed loss | Full32 motor RMSE | Late22 motor RMSE | First10 gripper RMSE | First10 motor MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| libero_spatial | +10.964% | +61.925% | +43.959% | +13.654% | +14.271% | -33.895% | +27.923% |
| libero_object | +9.936% | +74.070% | +45.560% | +3.024% | +0.294% | -16.244% | +21.443% |
| libero_goal | +5.181% | +64.045% | +36.774% | +16.155% | +19.183% | +74.756% | +11.283% |
| libero_10 | +5.177% | +110.660% | +44.532% | +7.327% | +8.575% | +15.971% | +11.432% |

## 复用来源

| Arm | Source binding |
|---|---|
| H_VQ8 | `{"arm": "H_VQ8", "job": "25903221.pbs101", "protocol": "fastwam-vq-r128-joint-c8-v1", "root": "/scratch/users/ntu/yguo017/fastwam-vq-r128-joint-c8-20261021", "split": "heldout"}` |
| R0_W16A4 | `{"arm": "C80_once_lr1e4", "job": "25905203.pbs101", "lane": "coverage_c80", "protocol": "fastwam-r128-improvements-v1", "root": "/scratch/users/ntu/yguo017/fastwam-r128-improvements-20261021", "split": "heldout"}` |

其他 provenance：

```json
{
  "allocation_gpus": 2,
  "equal_walltime": false,
  "extra_joint_rotation_updates": 8,
  "fixed_final_checkpoint": true,
  "new_teacher_queries": 0,
  "pbs_jobid": "25912619.pbs101",
  "repro_receipt": {
    "complete": true,
    "cross_device": {
      "action_max_abs": 0.0,
      "delta_max_abs": 0.0,
      "delta_tensors": 1230,
      "future_max_abs": 0.0,
      "gradients_max_abs": 0.0,
      "gradients_tensors": 1230,
      "passed": true,
      "post_action_max_abs": 0.0,
      "post_future_max_abs": 0.0
    },
    "cublas_workspace_config": ":4096:8",
    "deterministic_algorithms": true,
    "passed": true,
    "protocol": "fastwam-vq-r128-c80-dual-v1",
    "within_process": [
      {
        "action_max_abs": 0.0,
        "delta_max_abs": 0.0,
        "delta_tensors": 1230,
        "future_max_abs": 0.0,
        "gradients_max_abs": 0.0,
        "gradients_tensors": 1230,
        "passed": true,
        "post_action_max_abs": 0.0,
        "post_future_max_abs": 0.0
      },
      {
        "action_max_abs": 0.0,
        "delta_max_abs": 0.0,
        "delta_tensors": 1230,
        "future_max_abs": 0.0,
        "gradients_max_abs": 0.0,
        "gradients_tensors": 1230,
        "passed": true,
        "post_action_max_abs": 0.0,
        "post_future_max_abs": 0.0
      }
    ]
  },
  "shared_initialization": true,
  "source_receipt": {
    "complete": true,
    "cross_device_tensor_max_abs": 0.0,
    "passed": true,
    "protocol": "fastwam-vq-r128-c80-dual-v1",
    "source_metric_max_error": 0.0,
    "source_rotation": {
      "arm": "C80_once_lr1e4",
      "deployment": "/scratch/users/ntu/yguo017/fastwam-r128-improvements-20261021/results/coverage_c80/arms/C80_once_lr1e4/deployment.pt",
      "job": "25905203.pbs101",
      "lane": "coverage_c80",
      "protocol": "fastwam-r128-improvements-v1",
      "root": "/scratch/users/ntu/yguo017/fastwam-r128-improvements-20261021",
      "training_exposures": 80,
      "updates": 8
    }
  },
  "source_rotation": {
    "arm": "C80_once_lr1e4",
    "deployment": "/scratch/users/ntu/yguo017/fastwam-r128-improvements-20261021/results/coverage_c80/arms/C80_once_lr1e4/deployment.pt",
    "job": "25905203.pbs101",
    "lane": "coverage_c80",
    "protocol": "fastwam-r128-improvements-v1",
    "root": "/scratch/users/ntu/yguo017/fastwam-r128-improvements-20261021",
    "training_exposures": 80,
    "updates": 8
  },
  "training": {
    "Joint8": {
      "arm": "Joint8",
      "exposures": 80,
      "history": [
        {
          "after": {
            "first10gripper_rmse": 0.15071649923920633,
            "first10motor_mse": 0.0024801219929940998,
            "first10motor_rmse": 0.049125340208411215,
            "full32motor_rmse": 0.05187284350395203,
            "future_latent_nmse": 0.06631787959486246,
            "late22motor_rmse": 0.05289178267121315,
            "mixed_loss": 0.005295616248622537
          },
          "before": {
            "first10gripper_rmse": 0.031524678133428095,
            "first10motor_mse": 0.0022750133415684105,
            "first10motor_rmse": 0.04708258472383022,
            "full32motor_rmse": 0.05202374085783958,
            "future_latent_nmse": 0.07581497319042682,
            "late22motor_rmse": 0.05395078994333744,
            "mixed_loss": 0.005493702064268291
          },
          "cases": [
            2033,
            2067,
            2010,
            2059,
            2016,
            2018,
            2031,
            2036,
            2058,
            2060
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.0466504767537117,
          "rotation_gradients": {
            "action": true,
            "video": true
          },
          "seconds": 1190.3354252260178,
          "update": 1,
          "vq_gradient_norm": 0.006821590941399336
        },
        {
          "after": {
            "first10gripper_rmse": 0.025967108458280562,
            "first10motor_mse": 0.0021626327536068855,
            "first10motor_rmse": 0.04596066921949386,
            "full32motor_rmse": 0.04818673692643642,
            "future_latent_nmse": 0.05755509492009878,
            "late22motor_rmse": 0.048997648432850836,
            "mixed_loss": 0.004606106993742287
          },
          "before": {
            "first10gripper_rmse": 0.02576045524328947,
            "first10motor_mse": 0.0023016151273623107,
            "first10motor_rmse": 0.047366729378700255,
            "full32motor_rmse": 0.05185611397027969,
            "future_latent_nmse": 0.06888538002967834,
            "late22motor_rmse": 0.05349798202514648,
            "mixed_loss": 0.005226111342199147
          },
          "cases": [
            2037,
            2020,
            2057,
            2072,
            2000,
            2047,
            2062,
            2044,
            2024,
            2014
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.050455089658498764,
          "rotation_gradients": {
            "action": true,
            "video": true
          },
          "seconds": 1192.7537917729933,
          "update": 2,
          "vq_gradient_norm": 0.012410465627908707
        },
        {
          "after": {
            "first10gripper_rmse": 0.08340366650372744,
            "first10motor_mse": 0.0024522805353626607,
            "first10motor_rmse": 0.04823170006275177,
            "full32motor_rmse": 0.04718737080693245,
            "future_latent_nmse": 0.04605717435479164,
            "late22motor_rmse": 0.04665384590625763,
            "mixed_loss": 0.004407616076059639
          },
          "before": {
            "first10gripper_rmse": 0.024326886609196664,
            "first10motor_mse": 0.0026060137315653265,
            "first10motor_rmse": 0.04871145151555538,
            "full32motor_rmse": 0.05004524439573288,
            "future_latent_nmse": 0.05399829763919115,
            "late22motor_rmse": 0.050447728112339975,
            "mixed_loss": 0.004898485820740461
          },
          "cases": [
            2017,
            2066,
            2034,
            2027,
            2046,
            2011,
            2077,
            2056,
            2048,
            2073
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.14800383150577545,
          "rotation_gradients": {
            "action": true,
            "video": true
          },
          "seconds": 1192.188345475006,
          "update": 3,
          "vq_gradient_norm": 0.021593976765871048
        },
        {
          "after": {
            "first10gripper_rmse": 0.02182494355365634,
            "first10motor_mse": 0.001773215807043016,
            "first10motor_rmse": 0.04200767651200295,
            "full32motor_rmse": 0.04560472890734672,
            "future_latent_nmse": 0.04179947599768639,
            "late22motor_rmse": 0.04686412177979946,
            "mixed_loss": 0.003547792718745768
          },
          "before": {
            "first10gripper_rmse": 0.021099142078310252,
            "first10motor_mse": 0.0019586142734624445,
            "first10motor_rmse": 0.044119260460138324,
            "full32motor_rmse": 0.04589755684137344,
            "future_latent_nmse": 0.04838029313832522,
            "late22motor_rmse": 0.046420855820178984,
            "mixed_loss": 0.004012576630339026
          },
          "cases": [
            2038,
            2079,
            2003,
            2054,
            2039,
            2009,
            2002,
            2049,
            2065,
            2078
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.03861261159181595,
          "rotation_gradients": {
            "action": true,
            "video": true
          },
          "seconds": 1266.4544952970464,
          "update": 4,
          "vq_gradient_norm": 0.005813203286379576
        },
        {
          "after": {
            "first10gripper_rmse": 0.07679460663348436,
            "first10motor_mse": 0.0018708376679569482,
            "first10motor_rmse": 0.043000882118940355,
            "full32motor_rmse": 0.04746186472475529,
            "future_latent_nmse": 0.05579754132777452,
            "late22motor_rmse": 0.04886825829744339,
            "mixed_loss": 0.0042396958218887445
          },
          "before": {
            "first10gripper_rmse": 0.1673262506723404,
            "first10motor_mse": 0.0027032915386371315,
            "first10motor_rmse": 0.048369685932993886,
            "full32motor_rmse": 0.0496919222176075,
            "future_latent_nmse": 0.06452009929344057,
            "late22motor_rmse": 0.05012372806668282,
            "mixed_loss": 0.005442461697384715
          },
          "cases": [
            2041,
            2026,
            2074,
            2007,
            2053,
            2004,
            2025,
            2005,
            2061,
            2032
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.19071847200393677,
          "rotation_gradients": {
            "action": true,
            "video": true
          },
          "seconds": 1199.0208688810235,
          "update": 5,
          "vq_gradient_norm": 0.03056790865957737
        },
        {
          "after": {
            "first10gripper_rmse": 0.013918452430516481,
            "first10motor_mse": 0.002069707226473838,
            "first10motor_rmse": 0.044748559966683386,
            "full32motor_rmse": 0.04478709734976292,
            "future_latent_nmse": 0.06008610688149929,
            "late22motor_rmse": 0.044654068723320964,
            "mixed_loss": 0.0046206344151869415
          },
          "before": {
            "first10gripper_rmse": 0.0178806908428669,
            "first10motor_mse": 0.0019611599971540274,
            "first10motor_rmse": 0.04380523301661014,
            "full32motor_rmse": 0.054216137155890465,
            "future_latent_nmse": 0.0901559803634882,
            "late22motor_rmse": 0.05742048174142837,
            "mixed_loss": 0.00578868945594877
          },
          "cases": [
            2035,
            2028,
            2021,
            2071,
            2064,
            2006,
            2043,
            2019,
            2051,
            2045
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.06663735210895538,
          "rotation_gradients": {
            "action": true,
            "video": true
          },
          "seconds": 1194.2418027790263,
          "update": 6,
          "vq_gradient_norm": 0.009920274838805199
        },
        {
          "after": {
            "first10gripper_rmse": 0.07618643697351217,
            "first10motor_mse": 0.002462319703772664,
            "first10motor_rmse": 0.04898584298789501,
            "full32motor_rmse": 0.04795632436871529,
            "future_latent_nmse": 0.046948434039950374,
            "late22motor_rmse": 0.04685226269066334,
            "mixed_loss": 0.004455493181012571
          },
          "before": {
            "first10gripper_rmse": 0.07562891119159758,
            "first10motor_mse": 0.0020981878391467036,
            "first10motor_rmse": 0.04551711678504944,
            "full32motor_rmse": 0.04850278310477733,
            "future_latent_nmse": 0.04849074073135853,
            "late22motor_rmse": 0.04925413504242897,
            "mixed_loss": 0.004156839242205024
          },
          "cases": [
            2030,
            2069,
            2055,
            2068,
            2063,
            2040,
            2050,
            2022,
            2008,
            2015
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.10062525421380997,
          "rotation_gradients": {
            "action": true,
            "video": true
          },
          "seconds": 1193.626054708031,
          "update": 7,
          "vq_gradient_norm": 0.014650191180408001
        },
        {
          "after": {
            "first10gripper_rmse": 0.014370504859834909,
            "first10motor_mse": 0.0017474648542702199,
            "first10motor_rmse": 0.04159352518618107,
            "full32motor_rmse": 0.04239670410752296,
            "future_latent_nmse": 0.0406465832144022,
            "late22motor_rmse": 0.042671366780996325,
            "mixed_loss": 0.003473096271045506
          },
          "before": {
            "first10gripper_rmse": 0.013341494603082538,
            "first10motor_mse": 0.0019142100471071899,
            "first10motor_rmse": 0.0435048408806324,
            "full32motor_rmse": 0.0441140279173851,
            "future_latent_nmse": 0.04344165343791247,
            "late22motor_rmse": 0.044251929968595505,
            "mixed_loss": 0.0037585048470646144
          },
          "cases": [
            2052,
            2012,
            2029,
            2023,
            2075,
            2042,
            2070,
            2076,
            2001,
            2013
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.04410557821393013,
          "rotation_gradients": {
            "action": true,
            "video": true
          },
          "seconds": 1186.030885469052,
          "update": 8,
          "vq_gradient_norm": 0.006288196891546249
        }
      ],
      "initial_parity_max_abs": 0.0,
      "lineage": [
        "25912619.pbs101"
      ],
      "protocol": "fastwam-vq-r128-c80-dual-v1",
      "updates": 8
    },
    "Seq8": {
      "arm": "Seq8",
      "exposures": 80,
      "history": [
        {
          "after": {
            "first10gripper_rmse": 0.030899590626358987,
            "first10motor_mse": 0.002153236547019333,
            "first10motor_rmse": 0.04575880393385887,
            "full32motor_rmse": 0.05248558409512043,
            "future_latent_nmse": 0.07871744222939014,
            "late22motor_rmse": 0.055072427913546565,
            "mixed_loss": 0.005495148268528283
          },
          "before": {
            "first10gripper_rmse": 0.031524678133428095,
            "first10motor_mse": 0.0022750133415684105,
            "first10motor_rmse": 0.04708258472383022,
            "full32motor_rmse": 0.05202374085783958,
            "future_latent_nmse": 0.07581497319042682,
            "late22motor_rmse": 0.05395078994333744,
            "mixed_loss": 0.005493702064268291
          },
          "cases": [
            2033,
            2067,
            2010,
            2059,
            2016,
            2018,
            2031,
            2036,
            2058,
            2060
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.0,
          "rotation_gradients": {},
          "seconds": 1189.706682822085,
          "update": 1,
          "vq_gradient_norm": 0.006821590941399336
        },
        {
          "after": {
            "first10gripper_rmse": 0.030511113069951534,
            "first10motor_mse": 0.0022404986200854184,
            "first10motor_rmse": 0.046941173076629636,
            "full32motor_rmse": 0.05087907910346985,
            "future_latent_nmse": 0.052608984708786014,
            "late22motor_rmse": 0.052203359082341194,
            "mixed_loss": 0.0044739879434928295
          },
          "before": {
            "first10gripper_rmse": 0.031971130147576333,
            "first10motor_mse": 0.0024613358778879046,
            "first10motor_rmse": 0.0492038793861866,
            "full32motor_rmse": 0.05154496021568775,
            "future_latent_nmse": 0.058975651301443575,
            "late22motor_rmse": 0.052275166660547254,
            "mixed_loss": 0.004965119226835668
          },
          "cases": [
            2037,
            2020,
            2057,
            2072,
            2000,
            2047,
            2062,
            2044,
            2024,
            2014
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.0,
          "rotation_gradients": {},
          "seconds": 1190.6098422209034,
          "update": 2,
          "vq_gradient_norm": 0.009513729251921177
        },
        {
          "after": {
            "first10gripper_rmse": 0.029551419988274576,
            "first10motor_mse": 0.002252641075756401,
            "first10motor_rmse": 0.04682678319513798,
            "full32motor_rmse": 0.04665281027555466,
            "future_latent_nmse": 0.04548478275537491,
            "late22motor_rmse": 0.04638024754822254,
            "mixed_loss": 0.0041836760006845
          },
          "before": {
            "first10gripper_rmse": 0.02670285003259778,
            "first10motor_mse": 0.002371974231209606,
            "first10motor_rmse": 0.04808790981769562,
            "full32motor_rmse": 0.04622499495744705,
            "future_latent_nmse": 0.04926491845399141,
            "late22motor_rmse": 0.04514107815921307,
            "mixed_loss": 0.004463492915965617
          },
          "cases": [
            2017,
            2066,
            2034,
            2027,
            2046,
            2011,
            2077,
            2056,
            2048,
            2073
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.0,
          "rotation_gradients": {},
          "seconds": 1191.301786193042,
          "update": 3,
          "vq_gradient_norm": 0.008275151252746582
        },
        {
          "after": {
            "first10gripper_rmse": 0.025638254545629024,
            "first10motor_mse": 0.0019588181981816887,
            "first10motor_rmse": 0.04415079057216644,
            "full32motor_rmse": 0.046502994000911714,
            "future_latent_nmse": 0.05121916923671961,
            "late22motor_rmse": 0.04737637490034104,
            "mixed_loss": 0.004133303742855787
          },
          "before": {
            "first10gripper_rmse": 0.027497264556586742,
            "first10motor_mse": 0.001937878190074116,
            "first10motor_rmse": 0.04379017725586891,
            "full32motor_rmse": 0.0462622731924057,
            "future_latent_nmse": 0.04143032506108284,
            "late22motor_rmse": 0.047143721207976344,
            "mixed_loss": 0.003696783073246479
          },
          "cases": [
            2038,
            2079,
            2003,
            2054,
            2039,
            2009,
            2002,
            2049,
            2065,
            2078
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.0,
          "rotation_gradients": {},
          "seconds": 1265.351864065975,
          "update": 4,
          "vq_gradient_norm": 0.009743575006723404
        },
        {
          "after": {
            "first10gripper_rmse": 0.08445841111242772,
            "first10motor_mse": 0.0024928059545345606,
            "first10motor_rmse": 0.04789500050246716,
            "full32motor_rmse": 0.049589911475777626,
            "future_latent_nmse": 0.06766189215704799,
            "late22motor_rmse": 0.05015491805970669,
            "mixed_loss": 0.005365359550341964
          },
          "before": {
            "first10gripper_rmse": 0.08705486450344324,
            "first10motor_mse": 0.0023884794558398426,
            "first10motor_rmse": 0.04744660593569279,
            "full32motor_rmse": 0.05012877434492111,
            "future_latent_nmse": 0.0582367654889822,
            "late22motor_rmse": 0.05107677653431893,
            "mixed_loss": 0.004860893683508038
          },
          "cases": [
            2041,
            2026,
            2074,
            2007,
            2053,
            2004,
            2025,
            2005,
            2061,
            2032
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.0,
          "rotation_gradients": {},
          "seconds": 1197.0778868099442,
          "update": 5,
          "vq_gradient_norm": 0.04216524586081505
        },
        {
          "after": {
            "first10gripper_rmse": 0.08266371935606003,
            "first10motor_mse": 0.0019487220793962478,
            "first10motor_rmse": 0.043304988741874696,
            "full32motor_rmse": 0.04365202561020851,
            "future_latent_nmse": 0.04039611341431737,
            "late22motor_rmse": 0.043729424849152564,
            "mixed_loss": 0.003663719887845218
          },
          "before": {
            "first10gripper_rmse": 0.10975885791704058,
            "first10motor_mse": 0.002247010904829949,
            "first10motor_rmse": 0.04624586142599583,
            "full32motor_rmse": 0.04570880904793739,
            "future_latent_nmse": 0.054095261730253696,
            "late22motor_rmse": 0.04530494958162308,
            "mixed_loss": 0.0045435996260493995
          },
          "cases": [
            2035,
            2028,
            2021,
            2071,
            2064,
            2006,
            2043,
            2019,
            2051,
            2045
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.0,
          "rotation_gradients": {},
          "seconds": 1192.5218401489547,
          "update": 6,
          "vq_gradient_norm": 0.0131831094622612
        },
        {
          "after": {
            "first10gripper_rmse": 0.08185054305940867,
            "first10motor_mse": 0.002062270534224808,
            "first10motor_rmse": 0.044947662949562074,
            "full32motor_rmse": 0.047881071269512174,
            "future_latent_nmse": 0.053139778040349486,
            "late22motor_rmse": 0.04870941936969757,
            "mixed_loss": 0.0043182946043089036
          },
          "before": {
            "first10gripper_rmse": 0.08655419126152993,
            "first10motor_mse": 0.0023065344314090906,
            "first10motor_rmse": 0.04774567000567913,
            "full32motor_rmse": 0.04960670843720436,
            "future_latent_nmse": 0.05038441093638539,
            "late22motor_rmse": 0.050026675686240196,
            "mixed_loss": 0.004445580672472716
          },
          "cases": [
            2030,
            2069,
            2055,
            2068,
            2063,
            2040,
            2050,
            2022,
            2008,
            2015
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.0,
          "rotation_gradients": {},
          "seconds": 1192.2480204199674,
          "update": 7,
          "vq_gradient_norm": 0.010239874012768269
        },
        {
          "after": {
            "first10gripper_rmse": 0.021578862611204386,
            "first10motor_mse": 0.0020135918515734375,
            "first10motor_rmse": 0.044584862515330316,
            "full32motor_rmse": 0.04372255653142929,
            "future_latent_nmse": 0.04189967084676027,
            "late22motor_rmse": 0.0431836374104023,
            "mixed_loss": 0.003792422451078892
          },
          "before": {
            "first10gripper_rmse": 0.021939299907535313,
            "first10motor_mse": 0.001794780814088881,
            "first10motor_rmse": 0.04206601046025753,
            "full32motor_rmse": 0.044348031654953955,
            "future_latent_nmse": 0.03730900324881077,
            "late22motor_rmse": 0.04530622325837612,
            "mixed_loss": 0.0033787168096750973
          },
          "cases": [
            2052,
            2012,
            2029,
            2023,
            2075,
            2042,
            2070,
            2076,
            2001,
            2013
          ],
          "coverage": {
            "connected": 614,
            "disconnected_modules": [],
            "finite": 614,
            "nonzero": 614,
            "targets": 614,
            "zero_gradient_modules": []
          },
          "rotation_gradient_norm": 0.0,
          "rotation_gradients": {},
          "seconds": 1184.8667106099892,
          "update": 8,
          "vq_gradient_norm": 0.005026470869779587
        }
      ],
      "initial_parity_max_abs": 0.0,
      "lineage": [
        "25912619.pbs101"
      ],
      "protocol": "fastwam-vq-r128-c80-dual-v1",
      "updates": 8
    }
  },
  "warm_start": true
}
```

结果是描述性的 paired fixed-input decoded-BF16/A4 evidence；context 胜负计数不是显著性检验。它不构成 closed-loop、native-kernel、equal-compute 或 equal-walltime 结论。
