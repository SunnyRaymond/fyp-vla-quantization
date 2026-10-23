# Fast-WAM latent / teacher-future paired diagnostic

本报告汇总 frozen epoch 2 的固定输入比较。Calibration、selection、test 分开汇总；每个 case 先平均 seeds，再对 cases 等权平均。

分析范围：shard0 interim，30/60 contexts，11/22 testcases，full_cohort_complete=false。已选 shard 均通过完成门槛：30 contexts；native queries 90；teacher-future queries 60；每个已选 shard 的 self-future parity 覆盖 3 个 arms。Test 覆盖 11 cases（Original 5，Plus 6）。

## 各 split 的 latent 与 action fidelity

Action RMSE 均相对同一 context 的 BF16 teacher action；较低表示更接近该 reference。Latent 指标仅对 self arms 计算。

| Split | Arm | Cases / contexts | Future latent RMSE / NMSE | Transition RMSE / NMSE | First10 / full32 / late22 motor RMSE |
|---|---|---:|---:|---:|---:|
| calibration | parent_self | 4 / 4 | 0.211484 / 0.0662253 | 0.238225 / 0.117174 | 0.0454936 / 0.0497593 / 0.0515254 |
| calibration | distilled_self | 4 / 4 | 0.210059 / 0.0670956 | 0.239132 / 0.124617 | 0.04451 / 0.0480437 / 0.0494967 |
| selection | parent_self | 2 / 4 | 0.14769 / 0.0345039 | 0.164118 / 0.0572538 | 0.0422267 / 0.0414946 / 0.0411488 |
| selection | distilled_self | 2 / 4 | 0.134493 / 0.0286166 | 0.14765 / 0.0462617 | 0.0437303 / 0.0428874 / 0.0423706 |
| test | parent_self | 11 / 22 | 0.209502 / 0.0712029 | 0.234971 / 0.133157 | 0.0559409 / 0.0570262 / 0.0571818 |
| test | distilled_self | 11 / 22 | 0.221623 / 0.0892825 | 0.24449 / 0.163339 | 0.0587844 / 0.0755297 / 0.0810213 |

## Test domain breakdown

| Domain | Arm | Cases / contexts | Future latent RMSE / NMSE | First10 / full32 / late22 action RMSE |
|---|---|---:|---:|---:|
| original | parent_self | 5 / 10 | 0.194199 / 0.0587985 | 0.048574 / 0.0513428 / 0.0524109 |
| original | distilled_self | 5 / 10 | 0.181305 / 0.0499346 | 0.0429583 / 0.0475016 / 0.0493367 |
| plus | parent_self | 6 / 12 | 0.222254 / 0.0815399 | 0.0620799 / 0.0617623 / 0.0611575 |
| plus | distilled_self | 6 / 12 | 0.255221 / 0.122072 | 0.0719728 / 0.0988865 / 0.107425 |

## Teacher-future paired rescue

Rescue 定义为同一 case/seed 上 `self action RMSE − teacher-future action RMSE`，再先平均 seeds、后等权平均 cases。正值表示替换未来 latent 后与 BF16 action 的误差下降。

| Split | Policy | Cases / contexts | First10 / full32 / late22 rescue |
|---|---|---:|---:|
| calibration | parent | 4 / 4 | 0.00731998 / 0.0126896 / 0.0149809 |
| calibration | distilled | 4 / 4 | 0.00657818 / 0.0108255 / 0.0126393 |
| selection | parent | 2 / 4 | 0.00155663 / 0.00277715 / 0.00337263 |
| selection | distilled | 2 / 4 | 0.00469382 / 0.00437152 / 0.00418905 |
| test | parent | 11 / 22 | 0.0175378 / 0.0178437 / 0.0177163 |
| test | distilled | 11 / 22 | 0.0202245 / 0.0369842 / 0.0425364 |

当前覆盖的 test cases 中，parent 与 distilled 的 first10 平均 rescue 均为正，方向上与局部未来 latent 对 action fidelity 有上游影响相符。
这个 teacher-future 操作只检验固定输入下的局部替换。观察到正向 rescue 时，它至多支持未来 latent 对后续 action 有上游贡献；不能换算成可加的因果百分比，也不代表 closed-loop success 或 native-kernel latency。

## Distilled minus parent paired change

差值定义为 `distilled_self − parent_self`；负值表示 distilled 的对应 error metric 较低。

| Split | Cases / contexts | Future latent RMSE / NMSE Δ | Transition RMSE / NMSE Δ | First10 / full32 / late22 action Δ |
|---|---:|---:|---:|---:|
| calibration | 4 / 4 | -0.00142567 / 0.000870289 | 0.000907641 / 0.00744364 | -0.000983574 / -0.00171556 / -0.0020287 |
| selection | 2 / 4 | -0.013197 / -0.00588734 | -0.0164683 / -0.0109921 | 0.0015036 / 0.00139283 / 0.00122174 |
| test | 11 / 22 | 0.0121212 / 0.0180797 | 0.0095191 / 0.0301822 | 0.0028435 / 0.0185036 / 0.0238395 |

## Exploratory per-case delta direction

每个 test case 先平均 seeds，再计算 distilled−parent；负值计为 error 改善，正值计为恶化。下表描述本范围内逐 case 的方向和中位数，不报告显著性或 p 值。

| Metric | Improved cases | Worsened cases | Unchanged | Median delta |
|---|---:|---:|---:|---:|
| future_latent_rmse | 5 | 6 | 0 | 0.00644012 |
| future_latent_nmse | 4 | 7 | 0 | 0.00412186 |
| temporal_delta_global_rmse | 4 | 7 | 0 | 0.00818839 |
| temporal_delta_global_nmse | 4 | 7 | 0 | 0.00917932 |
| first10motor_rmse | 7 | 4 | 0 | -0.00204467 |
| full32motor_rmse | 7 | 4 | 0 | -0.00249998 |
| late22motor_rmse | 6 | 5 | 0 | -0.00130818 |

## Test case association and highlighted cases

以每个已覆盖 test case 的 seeds 均值为一个单位（n=11，不是 seed-context 数），比较 `Δ future latent RMSE` 与 `Δ first10 action RMSE`（均为 distilled−parent）。Spearman ρ=0.4273。这是 exploratory 描述，不报告显著性或 p 值。

逐 case 指标保留在 paired_cases.csv 与图中。

本范围未覆盖的 highlight cases：1017。
本范围实际覆盖的 highlight cases：1016。
- Case 1016 (plus/robot): parent latent RMSE/NMSE=0.318484/0.165484, transition RMSE/NMSE=0.368443/0.36968; distilled latent RMSE/NMSE=0.487272/0.387126, transition RMSE/NMSE=0.526569/0.754043; parent first10 self→teacher-future=0.0983879→0.0411259 (rescue 0.057262); distilled=0.169303→0.0432108 (rescue 0.126092).

## Evidence boundary

这是 decoded-BF16、fixed-input fidelity 与 teacher-future 内部 intervention 的诊断。没有训练、loss 改动、控制策略改动或 episode evaluation；不能据此声称闭环成功率提升、因果归因比例或 native-kernel 延迟变化。
