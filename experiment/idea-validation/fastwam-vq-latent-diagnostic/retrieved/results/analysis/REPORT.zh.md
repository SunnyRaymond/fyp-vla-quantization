# Fast-WAM latent / teacher-future paired diagnostic

本报告汇总 frozen epoch 2 的固定输入比较。Calibration、selection、test 分开汇总；每个 case 先平均 seeds，再对 cases 等权平均。

分析范围：full frozen cohort，60/60 contexts，22/22 testcases，full_cohort_complete=true。已选 shard 均通过完成门槛：60 contexts；native queries 180；teacher-future queries 120；每个已选 shard 的 self-future parity 覆盖 3 个 arms。Test 覆盖 22 cases（Original 10，Plus 12）。

## 各 split 的 latent 与 action fidelity

Action RMSE 均相对同一 context 的 BF16 teacher action；较低表示更接近该 reference。Latent 指标仅对 self arms 计算。

| Split | Arm | Cases / contexts | Future latent RMSE / NMSE | Transition RMSE / NMSE | First10 / full32 / late22 motor RMSE |
|---|---|---:|---:|---:|---:|
| calibration | parent_self | 8 / 8 | 0.218648 / 0.0744097 | 0.241469 / 0.124134 | 0.0467055 / 0.06193 / 0.0666802 |
| calibration | distilled_self | 8 / 8 | 0.224503 / 0.0820088 | 0.25165 / 0.142704 | 0.0443291 / 0.0611585 / 0.0661035 |
| selection | parent_self | 4 / 8 | 0.158007 / 0.0396902 | 0.173183 / 0.0645006 | 0.0426456 / 0.0443471 / 0.0450141 |
| selection | distilled_self | 4 / 8 | 0.146567 / 0.0345953 | 0.159819 / 0.0555697 | 0.0415074 / 0.0433149 / 0.0439805 |
| test | parent_self | 22 / 44 | 0.222949 / 0.0844354 | 0.249619 / 0.153072 | 0.0572558 / 0.0668284 / 0.0693003 |
| test | distilled_self | 22 / 44 | 0.228093 / 0.093011 | 0.253769 / 0.168856 | 0.0581104 / 0.0695719 / 0.0733415 |

## VAE latent time-position profiles

向量按 seed 先在 case 内平均，再对 cases 等权平均；第 0 帧 clamp 排除。Future latent 的位置是 VAE latent frame 1..T−1；transition 的第一项表示观测帧 0→第 1 个 future latent。它们是 VAE latent 时间位置，不是 32 个 action steps。

| Split | Arm | Latent shape | Future frame: RMSE / NMSE | Transition: RMSE / NMSE |
|---|---|---|---|---|
| calibration | parent_self | 1×48×3×14×28 | 1:0.16479/0.046417; 2:0.25735/0.098929 | 0→1:0.16479/0.059657; 1→2:0.29728/0.19234 |
| calibration | distilled_self | 1×48×3×14×28 | 1:0.18173/0.059415; 2:0.2526/0.10186 | 0→1:0.18173/0.078551; 1→2:0.30333/0.21046 |
| selection | parent_self | 1×48×3×14×28 | 1:0.12456/0.026205; 2:0.18478/0.051549 | 0→1:0.12456/0.031687; 1→2:0.21046/0.10101 |
| selection | distilled_self | 1×48×3×14×28 | 1:0.11405/0.02193; 2:0.17236/0.045749 | 0→1:0.11405/0.026465; 1→2:0.1947/0.08796 |
| test | parent_self | 1×48×3×14×28 | 1:0.18145/0.061944; 2:0.25465/0.10377 | 0→1:0.18145/0.082127; 1→2:0.30124/0.22978 |
| test | distilled_self | 1×48×3×14×28 | 1:0.18672/0.069758; 2:0.26045/0.11254 | 0→1:0.18672/0.09341; 1→2:0.3053/0.24942 |

## Test domain breakdown

| Domain | Arm | Cases / contexts | Future latent RMSE / NMSE | First10 / full32 / late22 action RMSE |
|---|---|---:|---:|---:|
| original | parent_self | 10 / 20 | 0.188738 / 0.0562598 | 0.048122 / 0.0499084 / 0.0505385 |
| original | distilled_self | 10 / 20 | 0.189198 / 0.0567503 | 0.045234 / 0.0482109 / 0.0493908 |
| plus | parent_self | 12 / 24 | 0.251458 / 0.107915 | 0.0648672 / 0.0809284 / 0.0849352 |
| plus | distilled_self | 12 / 24 | 0.260505 / 0.123228 | 0.0688408 / 0.0873727 / 0.0933004 |

## Teacher-future paired rescue

Rescue 定义为同一 case/seed 上 `self action RMSE − teacher-future action RMSE`，再先平均 seeds、后等权平均 cases。正值表示替换未来 latent 后与 BF16 action 的误差下降。

| Split | Policy | Cases / contexts | First10 / full32 / late22 rescue |
|---|---|---:|---:|
| calibration | parent | 8 / 8 | 0.00842591 / 0.0233951 / 0.0280734 |
| calibration | distilled | 8 / 8 | 0.00571783 / 0.0229791 / 0.0281722 |
| selection | parent | 4 / 8 | 0.00376702 / 0.00560878 / 0.00638711 |
| selection | distilled | 4 / 8 | 0.00308918 / 0.00457646 / 0.00519078 |
| test | parent | 22 / 44 | 0.0194327 / 0.0274738 / 0.0293594 |
| test | distilled | 22 / 44 | 0.0198166 / 0.0306121 / 0.0341494 |

当前覆盖的 test cases 中，parent 与 distilled 的 first10 平均 rescue 均为正，方向上与局部未来 latent 对 action fidelity 有上游影响相符。
这个 teacher-future 操作只检验固定输入下的局部替换。观察到正向 rescue 时，它至多支持未来 latent 对后续 action 有上游贡献；不能换算成可加的因果百分比，也不代表 closed-loop success 或 native-kernel latency。

## Distilled minus parent paired change

差值定义为 `distilled_self − parent_self`；负值表示 distilled 的对应 error metric 较低。

| Split | Cases / contexts | Future latent RMSE / NMSE Δ | Transition RMSE / NMSE Δ | First10 / full32 / late22 action Δ |
|---|---:|---:|---:|---:|
| calibration | 8 / 8 | 0.00585462 / 0.00759908 | 0.0101811 / 0.0185705 | -0.0023764 / -0.000771465 / -0.000576742 |
| selection | 4 / 8 | -0.0114401 / -0.00509492 | -0.013364 / -0.00893091 | -0.00113822 / -0.00103217 / -0.00103357 |
| test | 22 / 44 | 0.00514388 / 0.00857556 | 0.00414952 / 0.0157836 | 0.000854684 / 0.00274349 / 0.00404122 |

## Exploratory per-case delta direction

每个 test case 先平均 seeds，再计算 distilled−parent；负值计为 error 改善，正值计为恶化。下表描述本范围内逐 case 的方向和中位数，不报告显著性或 p 值。

| Metric | Improved cases | Worsened cases | Unchanged | Median delta |
|---|---:|---:|---:|---:|
| future_latent_rmse | 11 | 11 | 0 | 0.00148885 |
| future_latent_nmse | 10 | 12 | 0 | 0.00149558 |
| temporal_delta_global_rmse | 10 | 12 | 0 | 0.00116519 |
| temporal_delta_global_nmse | 10 | 12 | 0 | 0.00182107 |
| first10motor_rmse | 13 | 9 | 0 | -0.0018509 |
| full32motor_rmse | 15 | 7 | 0 | -0.00273888 |
| late22motor_rmse | 14 | 8 | 0 | -0.00214121 |

## Exploratory per-case teacher-future rescue direction

每个 test case 先平均 seeds；rescue = self RMSE−teacher-future RMSE，正值表示 teacher-future 使 error 降低。计数和中位数描述本范围内全部 test cases，不报告显著性或 p 值。

| Policy | Horizon | Improved cases | Worsened cases | Unchanged | Median rescue |
|---|---|---:|---:|---:|---:|
| parent | first10motor_rmse | 22 | 0 | 0 | 0.011196 |
| parent | full32motor_rmse | 22 | 0 | 0 | 0.0152085 |
| parent | late22motor_rmse | 22 | 0 | 0 | 0.0147904 |
| distilled | first10motor_rmse | 21 | 1 | 0 | 0.00793969 |
| distilled | full32motor_rmse | 22 | 0 | 0 | 0.011686 |
| distilled | late22motor_rmse | 22 | 0 | 0 | 0.0130231 |

## Test case association and highlighted cases

以每个已覆盖 test case 的 seeds 均值为一个单位（n=22，不是 seed-context 数），比较 `Δ future latent RMSE` 与 `Δ first10 action RMSE`（均为 distilled−parent）。Spearman ρ=0.1124。这是 exploratory 描述，不报告显著性或 p 值。

逐 case 指标保留在 paired_cases.csv 与图中。

本范围实际覆盖的 highlight cases：1016, 1017。
- Case 1016 (plus/robot): parent latent RMSE/NMSE=0.318484/0.165484, transition RMSE/NMSE=0.368443/0.36968; distilled latent RMSE/NMSE=0.487272/0.387126, transition RMSE/NMSE=0.526569/0.754043; parent first10 self→teacher-future=0.0983879→0.0411259 (rescue 0.057262); distilled=0.169303→0.0432108 (rescue 0.126092).
- Case 1017 (plus/robot): parent latent RMSE/NMSE=0.337802/0.179164, transition RMSE/NMSE=0.380723/0.308893; distilled latent RMSE/NMSE=0.329796/0.171216, transition RMSE/NMSE=0.368724/0.290107; parent first10 self→teacher-future=0.115799→0.0396746 (rescue 0.0761248); distilled=0.0853222→0.0370409 (rescue 0.0482813).

## Evidence boundary

这是 decoded-BF16、fixed-input fidelity 与 teacher-future 内部 intervention 的诊断。没有训练、loss 改动、控制策略改动或 episode evaluation；不能据此声称闭环成功率提升、因果归因比例或 native-kernel 延迟变化。
