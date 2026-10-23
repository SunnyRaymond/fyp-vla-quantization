# C8/C80 calibration-cohort 配对比较

共同 BF16 teacher 下，在同一 original-demo 测试输入与相同 sampler seed 上比较 decoded-BF16 Microscaling VQ。每个 cohort 覆盖 4 suites × 10 cases × 2 seeds，即每 cohort 80 个 query、614 个 target Linear。RMSE 越低越好；下表变化定义为 C80 − C8。

## Calibration 覆盖

| Cohort | 每 suite contexts | 总 contexts | 覆盖方式 |
| --- | ---: | ---: | --- |
| C8 | 2 | 8 | 每 suite 来自 2 个不同 tasks；嵌套于 C80 |
| C80 | 20 | 80 | 覆盖 40 tasks；每 task 2 个不同 demonstrations |

C80 每 suite 固定为 7 early、7 middle、6 late snapshots。

## 四套件 macro 指标

| 指标（主指标/次指标） | C8 | C80 | 绝对变化 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| 主指标：前10步 6维 motor RMSE | 0.046402156 | 0.046636498 | 0.00023434144 | 0.50502273% |
| 次指标：前10步 gripper RMSE | 0.045465892 | 0.045439792 | -2.6099884e-05 | -0.057405416% |
| 次指标：32步 6维 motor RMSE | 0.052329951 | 0.050788815 | -0.0015411364 | -2.9450369% |

相对变化栏以百分比表示；负的 RMSE 变化代表 C80 误差较低。

## 每 suite 主指标

| Suite | Cases | Queries | C8 motor RMSE | C80 motor RMSE | 绝对变化 | 相对变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| libero_spatial | 10 | 20 | 0.047288475 | 0.048748523 | 0.0014600487 | 3.0875361% |
| libero_object | 10 | 20 | 0.043049332 | 0.044158115 | 0.0011087827 | 2.5756096% |
| libero_goal | 10 | 20 | 0.047855667 | 0.048026787 | 0.00017111972 | 0.35757462% |
| libero_10 | 10 | 20 | 0.047415151 | 0.045612565 | -0.0018025855 | -3.8017077% |

## Paired per-case outcomes

胜负以每 case 的两个 seed 平均 RMSE 判定，tie tolerance 为绝对差 1e-12。

| 指标 | C80 误差较低 | Tie | C80 误差较高 |
| --- | ---: | ---: | ---: |
| 前10步 6维 motor RMSE | 21 | 0 | 19 |
| 前10步 gripper RMSE | 22 | 0 | 18 |
| 32步 6维 motor RMSE | 25 | 0 | 15 |

每个 case 的逐项 C8/C80 值与 delta 保存在 `comparison.json`。

## 预先固定 cases 的 gripper 配对诊断

以下 3027、3029、3031 三个 case 根据 C8 初步结果预先固定，只作诊断，不用于选择 quantization 方法。最大差异的 step 是归一化输出中的 0-based 索引，不代表已执行动作或 closed-loop 失败。

| Case | Seed | Sampler seed | C8 first10 RMSE | C8 最大绝对差 @ step | C80 first10 RMSE | C80 最大绝对差 @ step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3027 | 0 | 911006054 | 0.62299484 | 1.96875 @ 4 | 0.62544417 | 1.9765625 @ 4 |
| 3027 | 1 | 911006055 | 0.028863989 | 0.04296875 @ 4 | 0.026694296 | 0.046875 @ 3 |
| 3029 | 0 | 911006058 | 0.022097087 | 0.0546875 @ 3 | 0.63756895 | 2.015625 @ 3 |
| 3029 | 1 | 911006059 | 0.62543076 | 1.9765625 @ 2 | 0.625166 | 1.9765625 @ 2 |
| 3031 | 0 | 911006062 | 0.0234375 | 0.0390625 @ 8 | 0.024111405 | 0.0390625 @ 2 |
| 3031 | 1 | 911006063 | 0.63052732 | 1.9921875 @ 6 | 0.02751065 | 0.04296875 @ 4 |

| Case（两 seed 均值） | C8 | C80 | C80 − C8 |
| ---: | ---: | ---: | ---: |
| 3027 | 0.32592941 | 0.32606923 | 0.00013982039 |
| 3029 | 0.32376392 | 0.63136747 | 0.30760355 |
| 3031 | 0.32698241 | 0.025811028 | -0.30117138 |
| 三 case 等权均值 | 0.32555858 | 0.32774925 | 0.0021906632 |

## 解读范围

C8 与 C80 同时改变 calibration context 数量、task 覆盖与 phase 覆盖，因此结果比较的是两种预先固定的 calibration recipe，不能分离为单一数量因果效应。Smooth0.5/H128 transforms、VQ 设置与测试 cohort 均固定。这是 fixed-input decoded-BF16/A4 数值比较；不主张统计显著性、closed-loop 成功率、native low-bit kernel 或推理速度。

## 描述性 per-query wall time

下列时间包含输入加载/数据准备与 decoded-BF16 forward，仅作记录，不是 benchmark 或 latency claim。

| Cohort | Queries | Median seconds | Mean seconds |
| --- | ---: | ---: | ---: |
| teacher | 80 | 1.1113201 | 1.4348817 |
| c8 | 80 | 54.659294 | 55.314246 |
| c80 | 80 | 56.570876 | 57.201701 |
