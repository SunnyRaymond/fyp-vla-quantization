# Fast-WAM 双 KL 2×2 固定输入分析

本分析比较 `mse_nmse`、`action_kl`、`latent_kl`、`dual_kl` 四组。每组为一次 80-update 训练；单次训练结果不是独立训练复现。全部结果来自固定输入、decoded-BF16 endpoint 记录，不包含 rollout 或 closed-loop 证据。

每个 case 先对 seed 求均值，再在 suite 内对 case 求均值，最后四个 suite 等权。Gripper RMSE 仅作诊断，未进入主指标或候选门槛。

## Held-out：原 40 cases × 2 seeds

| Arm | First10 motor RMSE | Full32 motor RMSE | Late22 motor RMSE | Future latent NMSE | Action KL | Latent KL |
|---|---|---|---|---|---|---|
| mse_nmse | 0.031641466 | 0.03631233 | 0.03754631 | 0.055377875 | 0.074794248 | 0.17942314 |
| action_kl | 0.031724741 | 0.03670592 | 0.038174443 | 0.059674341 | 0.079431221 | 0.1930077 |
| latent_kl | 0.032218684 | 0.036171718 | 0.037157766 | 0.053663947 | 0.081726648 | 0.17344146 |
| dual_kl | 0.03232986 | 0.036817921 | 0.038043025 | 0.053033005 | 0.083403036 | 0.17095114 |

## Held-out：suite 等权分项

| Suite | Arm | First10 motor RMSE | Full32 motor RMSE | Late22 motor RMSE | Future latent NMSE | Action KL | Latent KL |
|---|---|---|---|---|---|---|---|
| libero_spatial | mse_nmse | 0.032787263 | 0.035041091 | 0.035847643 | 0.05281825 | 0.079328912 | 0.16104981 |
| libero_spatial | action_kl | 0.033387898 | 0.035476025 | 0.035968617 | 0.05337029 | 0.092106421 | 0.16296828 |
| libero_spatial | latent_kl | 0.033174078 | 0.034508193 | 0.034916707 | 0.046227595 | 0.088985955 | 0.1416266 |
| libero_spatial | dual_kl | 0.032555013 | 0.034740822 | 0.035503545 | 0.051351888 | 0.086020571 | 0.15553682 |
| libero_object | mse_nmse | 0.026123236 | 0.035114356 | 0.037761066 | 0.050390146 | 0.048515268 | 0.15838574 |
| libero_object | action_kl | 0.031701938 | 0.035963748 | 0.037318023 | 0.069716484 | 0.070225701 | 0.22032883 |
| libero_object | latent_kl | 0.029745037 | 0.035175191 | 0.036999785 | 0.057777765 | 0.067998917 | 0.18086228 |
| libero_object | dual_kl | 0.029308274 | 0.033487414 | 0.034875969 | 0.051235075 | 0.066323604 | 0.16031429 |
| libero_goal | mse_nmse | 0.035958091 | 0.043435477 | 0.045360425 | 0.068123172 | 0.092439172 | 0.2268411 |
| libero_goal | action_kl | 0.031706404 | 0.042214971 | 0.045251476 | 0.06452491 | 0.082454575 | 0.21465051 |
| libero_goal | latent_kl | 0.035987131 | 0.044678652 | 0.046667043 | 0.069764096 | 0.099485247 | 0.23117737 |
| libero_goal | dual_kl | 0.035397254 | 0.046523862 | 0.049530492 | 0.060086478 | 0.095432969 | 0.19910318 |
| libero_10 | mse_nmse | 0.031697273 | 0.031658396 | 0.031216108 | 0.050179932 | 0.078893639 | 0.1714159 |
| libero_10 | action_kl | 0.030102725 | 0.033168938 | 0.034159656 | 0.051085681 | 0.072938187 | 0.1740832 |
| libero_10 | latent_kl | 0.02996849 | 0.030324837 | 0.030047528 | 0.040886333 | 0.070436472 | 0.14009957 |
| libero_10 | dual_kl | 0.032058896 | 0.032519587 | 0.032262094 | 0.049458578 | 0.085835002 | 0.16885029 |

## Held-out motor bias 诊断

先对每个 query 的前 10 步逐维计算 `abs(mean_t(student − teacher))`，再对同一 case 的 seeds 求均值，最后按 case 和 suite 等权汇总；action 与 command 分开计算。

| Arm | Action bias dims 0–5 | Mean abs bias | Command bias dims 0–5 | Mean abs bias |
|---|---|---|---|---|
| mse_nmse | 0.019014, 0.0170121, 0.0180894, 0.0119028, 0.0150057, 0.0119945 | 0.0155031 | 0.0178257, 0.0159489, 0.0169588, 0.00355808, 0.00562715, 0.00443367 | 0.0107254 |
| action_kl | 0.0189483, 0.0160194, 0.0161193, 0.0134235, 0.0160056, 0.0122266 | 0.0154571 | 0.0177641, 0.0150182, 0.0151119, 0.00401268, 0.00600211, 0.00451948 | 0.0104047 |
| latent_kl | 0.0170609, 0.0161532, 0.0168604, 0.0134216, 0.0156187, 0.0114125 | 0.0150879 | 0.0159945, 0.0151436, 0.0158066, 0.00401211, 0.00585701, 0.00421856 | 0.0101721 |
| dual_kl | 0.0195037, 0.015058, 0.0174731, 0.013358, 0.0175298, 0.0124658 | 0.0158981 | 0.0182847, 0.0141169, 0.016381, 0.00399309, 0.00657367, 0.00460788 | 0.0106595 |

## Calibration：原 C80 × 1 seed

| Arm | First10 motor RMSE | Full32 motor RMSE | Late22 motor RMSE | Future latent NMSE | Action KL | Latent KL |
|---|---|---|---|---|---|---|
| mse_nmse | 0.029800742 | 0.033264115 | 0.034289289 | 0.04468495 | 0.068850856 | 0.14276411 |
| action_kl | 0.029985735 | 0.034254344 | 0.035608806 | 0.044268872 | 0.069722996 | 0.14197876 |
| latent_kl | 0.030687704 | 0.033224459 | 0.033849132 | 0.045034906 | 0.075757416 | 0.14400908 |
| dual_kl | 0.030608191 | 0.033517214 | 0.034363042 | 0.045182512 | 0.071938882 | 0.14394428 |

## Calibration：suite 分项

| Suite | Arm | First10 motor RMSE | Full32 motor RMSE | Late22 motor RMSE | Future latent NMSE | Action KL | Latent KL |
|---|---|---|---|---|---|---|---|
| libero_spatial | mse_nmse | 0.029705407 | 0.030580117 | 0.030467453 | 0.033148289 | 0.06942959 | 0.09752898 |
| libero_spatial | action_kl | 0.030562014 | 0.033524803 | 0.034309542 | 0.033241229 | 0.066324641 | 0.099031728 |
| libero_spatial | latent_kl | 0.032842095 | 0.034146781 | 0.033984584 | 0.037655847 | 0.077220644 | 0.11141424 |
| libero_spatial | dual_kl | 0.0314643 | 0.034056192 | 0.034589734 | 0.045062985 | 0.076456142 | 0.13299193 |
| libero_object | mse_nmse | 0.030206488 | 0.033264419 | 0.034376132 | 0.053787242 | 0.075363461 | 0.17165353 |
| libero_object | action_kl | 0.029099025 | 0.034022908 | 0.035639102 | 0.048241385 | 0.074694962 | 0.15456095 |
| libero_object | latent_kl | 0.030354035 | 0.033441486 | 0.034116593 | 0.052272866 | 0.079154204 | 0.16865747 |
| libero_object | dual_kl | 0.030224947 | 0.032931279 | 0.033750379 | 0.047210228 | 0.064966831 | 0.15157464 |
| libero_goal | mse_nmse | 0.032154622 | 0.037889008 | 0.039734508 | 0.056032315 | 0.070767091 | 0.18410278 |
| libero_goal | action_kl | 0.033314231 | 0.039457622 | 0.041532964 | 0.063124779 | 0.080344265 | 0.20737159 |
| libero_goal | latent_kl | 0.031919088 | 0.03479843 | 0.035827304 | 0.046535588 | 0.080795388 | 0.15290574 |
| libero_goal | dual_kl | 0.032857743 | 0.036685735 | 0.037956366 | 0.052986546 | 0.079745992 | 0.17456519 |
| libero_10 | mse_nmse | 0.027136449 | 0.031322915 | 0.032579063 | 0.035771955 | 0.059843282 | 0.11777113 |
| libero_10 | action_kl | 0.026967672 | 0.030012043 | 0.030953616 | 0.032468094 | 0.057528116 | 0.10695077 |
| libero_10 | latent_kl | 0.027635599 | 0.030511141 | 0.031468047 | 0.043675324 | 0.065859428 | 0.14305888 |
| libero_10 | dual_kl | 0.027885773 | 0.030395649 | 0.031155686 | 0.03547029 | 0.066586563 | 0.11664537 |

## Loss 与 gripper 诊断

| Split | Arm | First10 motor MSE | Mixed loss | Objective loss | First10 gripper RMSE |
|---|---|---|---|---|---|
| calibration | mse_nmse | 0.00095324467 | 0.0028503231 | 0.0028503231 | 0.054311746 |
| calibration | action_kl | 0.0009610772 | 0.0028404912 | 0.0027371797 | 0.086660244 |
| calibration | latent_kl | 0.0010061084 | 0.0029180441 | 0.0029196108 | 0.062231324 |
| calibration | dual_kl | 0.00098322272 | 0.0029014249 | 0.002797668 | 0.078164925 |
| heldout | mse_nmse | 0.0010800749 | 0.0034311163 | 0.0034311163 | 0.078307935 |
| heldout | action_kl | 0.0010904803 | 0.0036239262 | 0.0035106469 | 0.069721285 |
| heldout | latent_kl | 0.0011342603 | 0.0034125378 | 0.0034388416 | 0.069593035 |
| heldout | dual_kl | 0.0011386869 | 0.003390178 | 0.0032975558 | 0.079954709 |

原 C80 用于既定的 80-update 训练，并用于拟合 KL 坐标和初始 loss scaling；held-out 用于冻结端点比较，未参与拟合或 checkpoint 选择。
Loss scaling 来源为 80 个 C80 calibration queries；`heldout_used=False`。Fit stats 标记 `heldout_used=False`。

## Held-out 配对变化：候选减 `mse_nmse`

负 delta 表示该指标下降。统计单位为 seed 平均后的 40 个 case；区间是固定 seed 的 case bootstrap 95% 区间，用于描述本 cohort 的波动，不作显著性结论。

| Arm | Metric | Mean Δ | Median Δ | Improved / worsened / tied | LOCO mean range | Bootstrap 95% CI |
|---|---|---|---|---|---|---|
| action_kl | first10motor_rmse | 8.327551e-05 | -0.0002462948 | 22 / 18 / 0 | [-0.0003753821, 0.0005923271] | [-0.002023327, 0.002109496] |
| action_kl | future_latent_nmse | 0.004296466 | -0.0008289085 | 22 / 18 / 0 | [0.002535293, 0.005950941] | [-0.003600164, 0.01257071] |
| latent_kl | first10motor_rmse | 0.0005772183 | -3.007241e-06 | 20 / 20 / 0 | [0.0001587385, 0.0008937549] | [-0.001077229, 0.002236049] |
| latent_kl | future_latent_nmse | -0.001713928 | -0.002567704 | 23 / 17 / 0 | [-0.003418593, -0.0004430329] | [-0.00760259, 0.004593556] |
| dual_kl | first10motor_rmse | 0.0006883937 | 3.528688e-05 | 19 / 21 / 0 | [0.0003335181, 0.00108251] | [-0.001077573, 0.002403057] |
| dual_kl | future_latent_nmse | -0.00234487 | -0.002183054 | 21 / 19 / 0 | [-0.003764759, -0.0005966371] | [-0.009836083, 0.004511963] |

## 2×2 interaction

Interaction 按 `dual_kl − action_kl − latent_kl + mse_nmse` 计算；正负号表示该指标上的交互方向，不单独证明机制。

| Metric | Equal-suite interaction | By-suite |
|---|---|---|
| first10motor_rmse | 2.789984e-05 | libero_spatial: -0.001219699; libero_object: -0.006015464; libero_goal: 0.003661809; libero_10: 0.003684954 |
| future_latent_nmse | -0.004927408 | libero_spatial: 0.004572253; libero_object: -0.02586903; libero_goal: -0.006079355; libero_10: 0.007666497 |

## 预设工程门槛与 Pareto 对比

Dual-KL 候选门槛：False。motor 低于 control：False；latent 低于 control：True；motor case 中位 delta 为 3.528688e-05；40 cases 中改善 19 个（至少 24 个）。

| 对比 single-KL | Dual-KL Pareto 优于 | Motor Δ | Latent Δ |
|---|---|---|---|
| action_kl | False | +0.0006051182 | -0.006641336 |
| latent_kl | False | +0.0001111754 | -0.0006309423 |

## 训练与 GPU 记录

三组新 loss 各有一次训练；case bootstrap 只反映固定训练 checkpoint 下的输入 cohort 变化，不能替代独立训练复现。Loss 数值下降不代替动作质量或闭环验证。

| GPU job | 阶段 | Samples | Mean util % | Peak util % | Mean memory MiB | Peak memory MiB |
|---|---|---|---|---|---|---|
| mse_nmse_control_replay | unknown | 17 | 0.0 | 0.0 | 189.9 | 565.0 |
| mse_nmse_control_replay | loading | 55 | 0.6 | 29.0 | 10212.1 | 28871.0 |
| mse_nmse_control_replay | initial_calibration | 37 | 52.8 | 79.0 | 28897.0 | 28925.0 |
| mse_nmse_control_replay | evaluation | 64 | 35.5 | 54.0 | 28932.5 | 28951.0 |
| mse_nmse_control_replay | complete | 5 | 0.0 | 0.0 | 28951.0 | 28951.0 |
| action_kl | unknown | 16 | 0.0 | 0.0 | 133.0 | 565.0 |
| action_kl | loading | 57 | 2.4 | 78.0 | 10219.7 | 31325.0 |
| action_kl | training | 142 | 43.9 | 79.0 | 31357.5 | 31387.0 |
| action_kl | deployment_parity | 2 | 72.5 | 95.0 | 31387.0 | 31387.0 |
| action_kl | evaluation | 67 | 34.7 | 54.0 | 31387.0 | 31387.0 |
| action_kl | complete | 3 | 0.0 | 0.0 | 31387.0 | 31387.0 |
| latent_kl | unknown | 17 | 0.6 | 10.0 | 120.6 | 565.0 |
| latent_kl | loading | 57 | 2.9 | 71.0 | 10208.1 | 31331.0 |
| latent_kl | training | 128 | 49.5 | 82.0 | 31353.8 | 31379.0 |
| latent_kl | deployment_parity | 2 | 37.5 | 74.0 | 31379.0 | 31379.0 |
| latent_kl | evaluation | 60 | 38.6 | 63.0 | 31379.0 | 31379.0 |
| latent_kl | complete | 1 | 0.0 | 0.0 | 31379.0 | 31379.0 |
| dual_kl | unknown | 15 | 0.0 | 0.0 | 179.1 | 565.0 |
| dual_kl | loading | 55 | 2.5 | 55.0 | 10687.0 | 31325.0 |
| dual_kl | training | 131 | 49.2 | 83.0 | 31356.9 | 31389.0 |
| dual_kl | deployment_parity | 2 | 44.0 | 52.0 | 31389.0 | 31389.0 |
| dual_kl | evaluation | 62 | 38.0 | 54.0 | 31389.0 | 31389.0 |
| dual_kl | complete | 2 | 0.0 | 0.0 | 31389.0 | 31389.0 |

本结果是固定输入 endpoint 证据；未运行 rollout，不作 closed-loop 或统计显著性结论。

## 主任务复核与结果解读

以下解读来自上述完整 aggregate 和原始 training receipts；未追加训练、teacher forward 或 rollout。完整复核摘要见同目录上一级的 `completion_evidence.json`。

比较基准是已训练的原 per-channel MSE+NMSE final80（motor 0.031641466、latent 0.055377875），不是初始化的量化 baseline（0.044296774、0.060877371）。三组新训练均从共同 D=1/r=1 初始化开始，原对照复用其已完成的 80-update checkpoint。

| 相对原 MSE+NMSE | Motor RMSE | Future NMSE | Action KL | Latent KL | Motor 改善 cases |
|---|---|---|---|---|---|
| action_kl | +0.26% | +7.76% | +6.20% | +7.57% | 22/40 |
| latent_kl | +1.82% | -3.09% | +9.27% | -3.33% | 20/40 |
| dual_kl | +2.18% | -4.23% | +11.51% | -4.72% | 19/40 |

双 KL 带来了 latent fidelity 的改善，motor 同时退化，且 action KL 本身也没有改善。原混合 MSE+lambda*NMSE 在 held-out 从 0.0034311163 降为 0.0033901780（约 -1.19%），所以结果存在明确的目标权衡；不能描述成所有指标都变差。它没有通过预设的联合改善门槛，也没有同时优于任一 single-KL 的两项主指标。

这轮也不能简单归因为训练过拟合。把**同一种目标公式**应用到不同 checkpoint 后，整个 C80 上的 latent-KL 混合目标为：原 MSE 对照 0.0028502046、latent_kl 0.0029196109；双 KL 混合目标为：原 MSE 对照 0.0027439962、dual_kl 0.0027976680。即两个含 latent KL 的新模型，连各自完整 calibration 目标也未优于旧 MSE checkpoint。Action-KL 混合目标在 C80 仅小幅下降（0.0027441146 到 0.0027371797），held-out 则上升（0.0032711962 到 0.0035106469）。这提示需区分有限 80 步的优化效果、loss 的梯度尺度及重加权与泛化问题；单凭本轮不能确定其中哪项是主因。不同 loss 公式的 objective 数值不能直接互相比较；交叉计算结果已保存于 `completion_evidence.json` 的 `objective_matrix`。

下一步建议暂时保留 action MSE。若继续研究 latent KL，先诊断冻结的 per-channel mean/std 带来的重加权，以及初始 loss 数值配平后梯度是否失衡；后续用匹配相同标准化的 weighted-MSE 对照，才能区分收益来自 KL 形状还是尺度变化。当前 soft-bin KL 是人为定义的逐位置数值分布，没有加入时间联合分布或任务语义，因此不能因为使用了 KL 就假定它更能预测闭环。这些后续实验本轮均未启动。

执行核对：6/6 PBS units 均正常退出。三组新训练分别用时 01:11:54、01:06:23、01:06:53；GPU 准备用时 00:44:39，每个 GPU job 均低于 2 小时，合计 GPU allocation 为 04:09:49。全部 GPU job 有定期利用率/显存记录，gpt-6-luna/xhigh 每 600 秒监控现已结束。四组各完成 calibration80+heldout80，共 640 条端点；复用全部 teacher，新增 teacher queries 为 0。三组的完整梯度覆盖、冻结参数、自检及 uncached/cached/infer parity 均通过。原对照仅验证旧标量指标的精确回放，旧 raw action/future 张量未保存。

## 完整 C80 目标为何落后：2026-10-19 离线诊断

本节只使用已完成的 C80 端点、训练 history、final80 optimizer state 和缓存 teacher。CPU PBS `25879960.pbs101` 正常退出，用时 00:03:14；新增训练更新、模型 forward、teacher forward 均为 0，未使用 held-out 拟合。明细见 `c80_diagnosis/summary.json`。这是事后机制诊断，不是独立训练复现。

### 同一种目标公式交叉评价

| 完整 C80 上采用的目标 | 原 MSE checkpoint | 对应 KL checkpoint | 相对变化 |
|---|---:|---:|---:|
| Action KL + lambda NMSE | 0.0027441146 | 0.0027371797 | -0.253% |
| MSE + lambda latent KL | 0.0028502046 | 0.0029196109 | +2.435% |
| Action KL + lambda latent KL | 0.0027439962 | 0.0027976680 | +1.956% |

真正落后的是 latent KL 与双 KL；action KL 的完整 C80 混合目标略好。三组各自相对共同初始点的目标仍下降 37.10%、32.91%、35.71%，所以不是完全没有学习。原 MSE checkpoint 与新训练位于同一可行参数空间；它已经给出了更低的新 KL 目标，因此后两组的最终差距是有限预算优化没有达到已有更好解的证据，不能仅由评价指标改变解释。latent KL 与双 KL 各有 37/80 cases 优于原 MSE checkpoint，case delta 中位数均为正。

### 更值得优先检查的是优化过程

80 updates 是一个 epoch，每次只处理一个 case。history 中的 loss 是第 t 个输入在 theta_(t-1) 上的更新前目标，不是整个 C80 在 theta_t 上的目标；用其首尾或分段平均不能证明完整训练目标收敛。按访问先后分组没有稳定的“越早访问越差”模式，当前记录不足以认定 catastrophic forgetting。

同一末尾 case 2079/seed0 的日志与最终评估如下：

| Arm | 第80步前的训练目标 | final80 评估目标 | 变化 |
|---|---:|---:|---:|
| MSE+NMSE | 0.00176328 | 0.00308808 | +75.13% |
| Action KL | 0.00187236 | 0.00250756 | +33.93% |
| Latent KL | 0.00320968 | 0.00283305 | -11.73% |
| 双 KL | 0.00189738 | 0.00360877 | +90.20% |

该比较跨越一次更新，但也跨越 checkpointed/grad-enabled training 与 no-grad evaluation 路径。已有 max_abs=0 的部署校验覆盖 uncached/cached/infer，未直接验证 final80 同一参数下 checkpointed training 对 no-grad evaluation 的 parity。因此这些跳变是优先排查信号，不能现在就全部归因于学习率过大；即使路径完全一致，Adam 为多样本目标积累的动量也可能提高某个单独 case 的 loss，不能由一例证明整个 C80 发散。

四组所有更新均未触发 norm clip=1；最大的记录梯度范数只有 0.002541。Channel scales 在边界上仅 0–1 个/1,806,863 个，ratio 下界没有饱和。梯度裁剪压坏了更新、channel 参数整体被 bounds 卡住都缺少支持。

final optimizer 的 bias-corrected RMS 梯度有大量坐标小于 Adam epsilon=1e-8：channel 分别为 MSE 46.50%、action KL 54.68%、latent KL 46.15%、双 KL 53.50%；ratio 分别为 52.12%、58.14%、49.84%、55.70%。因此本实现中 epsilon 不是普遍可忽略项，loss 数值配平不等价于实际步长配平。Adam 的梯度缩放抵消性质需要忽略 epsilon，见 [Adam 原论文 §2.1](https://arxiv.org/pdf/1412.6980)。这些是 final-state 证据，不能证明整个训练过程的所有步长；latent-only 的比例与 MSE 很接近，也不支持把它的退化只归因于统一缩放或 epsilon。

### KL 替换同时改变了隐含权重

此 soft-bin KL 在 teacher 附近的局部二阶形式为 `0.5 * I(teacher_standardized) * error^2 / channel_std^2`。直接对当前概率映射求导得到：标准化值 0、0.25、2.5 的 Fisher information 约为 6.194，中心区几乎恒定。因此多数位置接近按方差重加权的平方误差，而不是新增时间关联或任务语义约束。

Action 各轴 inverse-variance 权重最大相差 26.11 倍；latent channels 相差 3.49 倍。原 latent NMSE 还逐 query 除以 teacher energy，当前 KL 使用全 C80 的固定 channel std。C80 teacher energy 的最大/最小为 1.45 倍；初始 KL/NMSE 比值与 teacher energy 的 log Pearson correlation 为 0.991，确认该替换同时改变了 query 相对权重。全局 alpha 只能配平初始均值，无法消除这些通道和样本间差异。这证明优化几何改变，但不能单独证明它是 2% 差距的全部原因。

Tail saturation 不是当前最有证据的首要解释：teacher 标准化值越出 [-3.5,3.5] 的比例，action 为 0.542%，latent 为 0.0812%；latent 超出边缘再加 3sigma 的比例为 0.00372%。少数尾值可能重要，student tails 的 latent 全张量未保存在端点文件中，尚不能排除个别案例；但不能据此笼统归因为全局梯度消失。

量化 backward 仍然是 LSQ/STE surrogate。实现自检通过只验证了实现符合所定义的 surrogate，并不保证任意新 loss 下的实际离散前向一定沿该方向下降；[STE 理论工作](https://arxiv.org/abs/1903.05662)给出的下降性质也依赖具体假设，不能直接移植为本模型的保证。

### 下一步优先顺序（均未启动）

1. **先做已有 checkpoint 的方向诊断，暂不训练。** 在相同参数、C80 输入和 seed 下检查 checkpointed/no-grad action+future parity；通过后比较 action、latent 对 ratios 和 channel-D 的梯度范数及夹角，并用暂存后恢复参数的真实前向检查原步长、1/2、1/4 步长。报告 batch 与完整 C80 的变化，不把一个 case 的上升当作全局发散。这样可区分路径问题、步长问题、动量/目标冲突和 surrogate 方向问题。
2. **若恢复训练，先研究单独 latent KL，并从原 MSE final80 起步。** 配一组相同起点、相同步数的 MSE continuation 对照；先根据方向诊断选定一项优化改动。增加 step 0/20/40/60/80 的完整 C80 loss 记录，仍用固定 final80 和原 held-out80 比较。不要同时改 epsilon、batch、learning rate 和 loss 定义后将收益归因于 KL。
3. **再区分 KL 形状与重加权。** 使用相同 channel 标准化的 weighted-MSE 对照；若目标是只换 divergence 形状，则对 KL 补偿 channel_std^2，并保留 latent 的 query energy 归一化，使中心区的权重更接近原 NMSE。该候选改变的是明确的目标定义，不能当作修复数值 bug。小权重 KL 辅助项也不能替代这个机制区分。

若方向诊断支持较大的样本梯度噪声，再考虑每个 update 四个 suite 各一个 case 的 gradient accumulation。必须单独核算预算：80 updates × batch4 需要 320 次训练 query forward；若只保留 80 次 forward，就只有 20 updates，不能声称更新预算不变。扩大 calibration 暂不优先，因为当前首先缺少稳定降低既有 C80 目标的证据。
