# C8 latent-anchor KL：固定输入配对结果

状态：complete；baseline cached queries：88/88；anchor-KL queries：88/88。
Baseline 新模型 forward：0；新 teacher query：0。

对照是原始 C8 A+Z rho=1.5、1 epoch/8 updates 的冻结结果；目录名包含 activation-clipping 仅表示缓存所在的旧实验目录。Baseline query 的额外 updates=0，父 checkpoint 已完成 C8 rho=1.5 的 8 updates；不把该缓存解释为新增 clipping 训练。Candidate 从同一原始 C8 bank 初始化，只训练 codebooks 与 log row gains，普通 dynamic A4，无 learned clipping/scaling。

每个 case 先平均 seed，再在 suite 内平均 case，最后四个 suite 等权。胜负按 case-level paired mean 计数；低值为胜。Gripper RMSE 仅作诊断。

## calibration（8 cases，8 queries/arm）

| Scope | Metric | Baseline | Anchor-KL | Δ (candidate−baseline) | W/L/T cases |
|---|---|---:|---:|---:|---:|
| Overall · four suites equally weighted | first10motor_rmse | 0.0421332 | 0.0415491 | -0.000584155 | 3/5/0 |
| Overall · four suites equally weighted | full32motor_rmse | 0.044388 | 0.0423086 | -0.00207946 | 6/2/0 |
| Overall · four suites equally weighted | late22motor_rmse | 0.0446109 | 0.0425163 | -0.00209456 | 5/3/0 |
| Overall · four suites equally weighted | first10gripper_rmse | 0.0200497 | 0.0186387 | -0.00141099 | 5/3/0 |
| Overall · four suites equally weighted | future_latent_nmse | 0.054481 | 0.0495689 | -0.00491209 | 7/1/0 |
| libero_10 | first10motor_rmse | 0.0408317 | 0.0422575 | 0.00142583 | 1/1/0 |
| libero_10 | full32motor_rmse | 0.0407933 | 0.0405606 | -0.000232756 | 1/1/0 |
| libero_10 | late22motor_rmse | 0.040712 | 0.0394236 | -0.00128837 | 1/1/0 |
| libero_10 | first10gripper_rmse | 0.0217866 | 0.017923 | -0.00386355 | 2/0/0 |
| libero_10 | future_latent_nmse | 0.0430182 | 0.033528 | -0.00949021 | 2/0/0 |
| libero_goal | first10motor_rmse | 0.0396628 | 0.0399575 | 0.000294644 | 1/1/0 |
| libero_goal | full32motor_rmse | 0.0524364 | 0.0444996 | -0.00793678 | 2/0/0 |
| libero_goal | late22motor_rmse | 0.0568506 | 0.0463983 | -0.0104522 | 2/0/0 |
| libero_goal | first10gripper_rmse | 0.0191438 | 0.0192029 | 5.91362e-05 | 1/1/0 |
| libero_goal | future_latent_nmse | 0.0787082 | 0.0720898 | -0.00661842 | 2/0/0 |
| libero_object | first10motor_rmse | 0.0342668 | 0.0366204 | 0.00235356 | 0/2/0 |
| libero_object | full32motor_rmse | 0.0386666 | 0.0371603 | -0.00150632 | 2/0/0 |
| libero_object | late22motor_rmse | 0.0405028 | 0.0374031 | -0.00309968 | 2/0/0 |
| libero_object | first10gripper_rmse | 0.0165727 | 0.0178733 | 0.00130055 | 1/1/0 |
| libero_object | future_latent_nmse | 0.0353471 | 0.0198968 | -0.0154503 | 2/0/0 |
| libero_spatial | first10motor_rmse | 0.0537717 | 0.047361 | -0.00641066 | 1/1/0 |
| libero_spatial | full32motor_rmse | 0.0456559 | 0.0470139 | 0.00135803 | 1/1/0 |
| libero_spatial | late22motor_rmse | 0.0403782 | 0.0468403 | 0.00646204 | 0/2/0 |
| libero_spatial | first10gripper_rmse | 0.0226957 | 0.0195556 | -0.0031401 | 1/1/0 |
| libero_spatial | future_latent_nmse | 0.0608504 | 0.072761 | 0.0119106 | 1/1/0 |

共同目标诊断（由记录中的 `first10motor_rmse² + rho × normalization_c × future_latent_nmse` 重建；RMSE 已舍入，因此 MSE 部分有舍入误差）：
- Baseline：0.0041966
- Anchor-KL：0.00385843
Baseline 的 `latent_anchor_kl` 与 `composite_objective` 标为 unavailable：缓存没有完整 future tensor，本轮不做 baseline 重放或补充 forward。

Candidate-only diagnostics（均值按相同 case/suite 层级汇总）：
- `action_mse`：0.00175401
- `latent_nmse`：0.0495689
- `latent_anchor_kl`：0.00622238
- `baseline_objective`：0.00385843
- `composite_objective`：0.0041781

## heldout（40 cases，80 queries/arm）

| Scope | Metric | Baseline | Anchor-KL | Δ (candidate−baseline) | W/L/T cases |
|---|---|---:|---:|---:|---:|
| Overall · four suites equally weighted | first10motor_rmse | 0.0442968 | 0.0458895 | 0.00159276 | 14/26/0 |
| Overall · four suites equally weighted | full32motor_rmse | 0.047501 | 0.0493055 | 0.00180457 | 17/23/0 |
| Overall · four suites equally weighted | late22motor_rmse | 0.0485285 | 0.0503193 | 0.0017908 | 15/25/0 |
| Overall · four suites equally weighted | first10gripper_rmse | 0.057054 | 0.0426101 | -0.014444 | 17/23/0 |
| Overall · four suites equally weighted | future_latent_nmse | 0.0608774 | 0.061471 | 0.000593599 | 18/22/0 |
| libero_10 | first10motor_rmse | 0.0456202 | 0.0466857 | 0.00106551 | 4/6/0 |
| libero_10 | full32motor_rmse | 0.0455229 | 0.0466707 | 0.00114779 | 5/5/0 |
| libero_10 | late22motor_rmse | 0.0452451 | 0.0465484 | 0.00130325 | 5/5/0 |
| libero_10 | first10gripper_rmse | 0.0492206 | 0.0201862 | -0.0290343 | 5/5/0 |
| libero_10 | future_latent_nmse | 0.0562145 | 0.0549033 | -0.00131118 | 5/5/0 |
| libero_goal | first10motor_rmse | 0.0450921 | 0.0479344 | 0.00284236 | 4/6/0 |
| libero_goal | full32motor_rmse | 0.0525401 | 0.0549723 | 0.00243222 | 4/6/0 |
| libero_goal | late22motor_rmse | 0.0548767 | 0.0570793 | 0.00220266 | 2/8/0 |
| libero_goal | first10gripper_rmse | 0.114934 | 0.115835 | 0.000900283 | 2/8/0 |
| libero_goal | future_latent_nmse | 0.0682966 | 0.0703342 | 0.00203769 | 5/5/0 |
| libero_object | first10motor_rmse | 0.0416801 | 0.0434633 | 0.00178315 | 2/8/0 |
| libero_object | full32motor_rmse | 0.0457673 | 0.0492938 | 0.00352646 | 2/8/0 |
| libero_object | late22motor_rmse | 0.0473702 | 0.0511275 | 0.00375728 | 3/7/0 |
| libero_object | first10gripper_rmse | 0.0173082 | 0.0163866 | -0.000921539 | 7/3/0 |
| libero_object | future_latent_nmse | 0.0650745 | 0.0648164 | -0.000258094 | 4/6/0 |
| libero_spatial | first10motor_rmse | 0.0447947 | 0.0454747 | 0.000680012 | 4/6/0 |
| libero_spatial | full32motor_rmse | 0.0461736 | 0.0462854 | 0.000111796 | 6/4/0 |
| libero_spatial | late22motor_rmse | 0.046622 | 0.046522 | -9.99924e-05 | 5/5/0 |
| libero_spatial | first10gripper_rmse | 0.0467531 | 0.0180328 | -0.0287203 | 3/7/0 |
| libero_spatial | future_latent_nmse | 0.0539239 | 0.0558299 | 0.00190598 | 4/6/0 |

共同目标诊断（由记录中的 `first10motor_rmse² + rho × normalization_c × future_latent_nmse` 重建；RMSE 已舍入，因此 MSE 部分有舍入误差）：
- Baseline：0.00459865
- Anchor-KL：0.0047495
Baseline 的 `latent_anchor_kl` 与 `composite_objective` 标为 unavailable：缓存没有完整 future tensor，本轮不做 baseline 重放或补充 forward。

Candidate-only diagnostics（均值按相同 case/suite 层级汇总）：
- `action_mse`：0.00213978
- `latent_nmse`：0.061471
- `latent_anchor_kl`：0.00659767
- `baseline_objective`：0.0047495
- `composite_objective`：0.00508845

Descriptive paired fixed-input results only. Seeds are averaged within each case, then cases within each suite, and the four suite means receive equal weight. Win/loss/tie counts use paired case means, not individual seeds. These repeated queries on the same fixed LIBERO cases are not independent tasks; no p-values or closed-loop inference are reported. Baseline records are reused cache entries; no baseline model or teacher was rerun.
