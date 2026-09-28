# C-SWM 视觉分块动力学 pilot 结果

Protocol: `cswm-visual-block-local-pilot-v1`；官方来源 commit `e944b24bcaa42d9ee847f30163437a50f0237aa0`。

**结果判定：inconclusive: at least one frozen visual reference adequacy criterion failed; all planned fits remain reported。** 预定 fits：Stage A 3/3，Stage B 21/21。没有因 dev/test 结果增加训练步数或追加 fits。

## 先看结论

该 pilot 比较 C-SWM-style object-slot predictor、局部模块、低维摘要、六层 Transformer 与参数匹配 flat MLP。每个 seed 的 encoder 与 full GNN 联合训练后冻结，因此这组表示对 GNN 有训练偏置；若局部模型成功，只能证明已有 C-SWM object-slot interface 对该环境有利，不能说明 LeWM latent 已可同样分解或已实现 LeWM 加速。

global4 的 flat 参数匹配对照预算为 22,096 参数；flat MLP hidden=83，参数=22,075（差异 -0.095%）。

## Encoder 与参考模型是否可用

| seed | full GNN H@1 h1 | full GNN H@1 h10 | persistence H@1 h10 | 提升 | adequate | test latent std | near-zero dims | position probe test RMSE |
|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1101 | 1.000 | 0.758 | 0.719 | +0.039 | no | 0.291 | 0/80 | 0.022 |
| 1102 | 1.000 | 1.000 | 0.699 | +0.301 | yes | 0.187 | 0/80 | 0.114 |
| 1103 | 1.000 | 1.000 | 0.719 | +0.281 | yes | 0.296 | 0/80 | 0.020 |

Adequacy 的预设门槛为每 seed h1≥0.80、h10≥0.50 且 h10 比 persistence 至少高 0.05。test latent std 和 near-zero dimensions 是坍塌诊断；position probe 是 train-fitted/test-evaluated 的辅助表示诊断。dev 指标仅用于补充描述，未用于选择模型或 checkpoint。

## 各 seed 的多步预测与速度

### Seed 1101

| arm | H@1 h1 | H@1 h5 | H@1 h10 | H@1 h20 | latent MSE h10 | params | B1 median/p90 ms | B300 median/p90 ms | quality | B1 speed vs GNN/T | B300 speed vs GNN/T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| full_gnn | 1.000 | 0.980 | 0.758 | 0.359 | 0.02514 | 75,408 | 5.368/5.386 | 5.555/5.581 | pass | +0.0%/+61.1% | +0.0%/+61.9% |
| local | 1.000 | 0.996 | 0.988 | 0.957 | 0.00585 | 21,264 | 2.736/2.784 | 2.428/2.443 | pass | +49.0%/+80.2% | +56.3%/+83.4% |
| local4 | 1.000 | 0.996 | 0.988 | 0.953 | 0.00582 | 22,096 | 3.321/3.339 | 3.496/3.514 | pass | +38.1%/+75.9% | +37.1%/+76.0% |
| global4 | 1.000 | 0.996 | 0.992 | 0.980 | 0.00533 | 22,096 | 2.686/2.694 | 2.815/3.109 | pass | +50.0%/+80.5% | +49.3%/+80.7% |
| global16 | 1.000 | 0.996 | 0.992 | 0.973 | 0.00546 | 24,592 | 3.175/3.222 | 2.814/3.181 | pass | +40.9%/+77.0% | +49.3%/+80.7% |
| transformer6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00036 | 1,195,024 | 13.788/14.672 | 14.592/15.958 | pass | -156.9%/+0.0% | -162.7%/+0.0% |
| flat_mlp_matched | 1.000 | 0.996 | 0.988 | 0.941 | 0.00667 | 22,075 | 2.238/2.531 | 2.378/2.825 | pass | +58.3%/+83.8% | +57.2%/+83.7% |

| arm | one-step MSE free | one-step MSE object-blocked | one-step MSE boundary-blocked | rollout H@1 h10 free | rollout H@1 h10 object-blocked | rollout H@1 h10 boundary-blocked | counterfactual response MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_gnn | 0.00420 | 0.00007 | 0.00005 | 0.729 | 0.812 | 0.815 | 0.00922 |
| local | 0.00014 | 0.00335 | 0.00000 | 0.982 | 1.000 | 1.000 | 0.00106 |
| local4 | 0.00014 | 0.00336 | 0.00000 | 0.982 | 1.000 | 1.000 | 0.00106 |
| global4 | 0.00017 | 0.00281 | 0.00000 | 0.988 | 1.000 | 1.000 | 0.00091 |
| global16 | 0.00020 | 0.00269 | 0.00001 | 0.988 | 1.000 | 1.000 | 0.00092 |
| transformer6 | 0.00001 | 0.00001 | 0.00001 | 1.000 | 1.000 | 1.000 | 0.00001 |
| flat_mlp_matched | 0.00019 | 0.00313 | 0.00005 | 0.982 | 1.000 | 1.000 | 0.00105 |

### Seed 1102

| arm | H@1 h1 | H@1 h5 | H@1 h10 | H@1 h20 | latent MSE h10 | params | B1 median/p90 ms | B300 median/p90 ms | quality | B1 speed vs GNN/T | B300 speed vs GNN/T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| full_gnn | 1.000 | 1.000 | 1.000 | 0.953 | 0.00222 | 75,408 | 5.384/5.403 | 5.411/5.559 | pass | +0.0%/+62.5% | +0.0%/+65.9% |
| local | 1.000 | 0.996 | 0.992 | 0.953 | 0.00271 | 21,264 | 2.362/2.794 | 2.455/2.750 | pass | +56.1%/+83.5% | +54.6%/+84.5% |
| local4 | 1.000 | 0.996 | 0.992 | 0.949 | 0.00287 | 22,096 | 3.353/3.373 | 3.252/3.527 | pass | +37.7%/+76.6% | +39.9%/+79.5% |
| global4 | 1.000 | 0.996 | 0.992 | 0.965 | 0.00253 | 22,096 | 3.151/3.238 | 2.840/3.168 | pass | +41.5%/+78.1% | +47.5%/+82.1% |
| global16 | 1.000 | 0.996 | 0.992 | 0.965 | 0.00265 | 24,592 | 2.699/3.181 | 3.326/3.337 | pass | +49.9%/+81.2% | +38.5%/+79.0% |
| transformer6 | 1.000 | 1.000 | 0.996 | 1.000 | 0.00097 | 1,195,024 | 14.359/15.875 | 15.874/17.150 | pass | -166.7%/+0.0% | -193.4%/+0.0% |
| flat_mlp_matched | 1.000 | 1.000 | 0.996 | 0.914 | 0.00402 | 22,075 | 2.280/2.708 | 2.396/2.410 | pass | +57.6%/+84.1% | +55.7%/+84.9% |

| arm | one-step MSE free | one-step MSE object-blocked | one-step MSE boundary-blocked | rollout H@1 h10 free | rollout H@1 h10 object-blocked | rollout H@1 h10 boundary-blocked | counterfactual response MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_gnn | 0.00010 | 0.00041 | 0.00007 | 1.000 | 1.000 | 1.000 | 0.00029 |
| local | 0.00010 | 0.00132 | 0.00001 | 0.988 | 1.000 | 1.000 | 0.00048 |
| local4 | 0.00009 | 0.00134 | 0.00001 | 0.988 | 1.000 | 1.000 | 0.00047 |
| global4 | 0.00010 | 0.00117 | 0.00001 | 0.988 | 1.000 | 1.000 | 0.00045 |
| global16 | 0.00010 | 0.00117 | 0.00001 | 0.988 | 1.000 | 1.000 | 0.00044 |
| transformer6 | 0.00005 | 0.00007 | 0.00002 | 0.994 | 1.000 | 1.000 | 0.00007 |
| flat_mlp_matched | 0.00022 | 0.00107 | 0.00040 | 0.994 | 1.000 | 1.000 | 0.00086 |

### Seed 1103

| arm | H@1 h1 | H@1 h5 | H@1 h10 | H@1 h20 | latent MSE h10 | params | B1 median/p90 ms | B300 median/p90 ms | quality | B1 speed vs GNN/T | B300 speed vs GNN/T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| full_gnn | 1.000 | 1.000 | 1.000 | 1.000 | 0.00032 | 75,408 | 4.553/5.077 | 4.707/4.876 | pass | +0.0%/+70.3% | +0.0%/+72.7% |
| local | 1.000 | 0.996 | 0.996 | 0.961 | 0.00619 | 21,264 | 2.384/2.806 | 2.458/2.602 | pass | +47.6%/+84.4% | +47.8%/+85.7% |
| local4 | 1.000 | 0.996 | 0.992 | 0.965 | 0.00619 | 22,096 | 2.844/3.260 | 2.951/3.467 | pass | +37.6%/+81.4% | +37.3%/+82.9% |
| global4 | 1.000 | 0.996 | 0.992 | 0.977 | 0.00569 | 22,096 | 2.691/2.896 | 2.913/3.340 | pass | +40.9%/+82.4% | +38.1%/+83.1% |
| global16 | 1.000 | 1.000 | 0.992 | 0.977 | 0.00577 | 24,592 | 2.709/3.212 | 3.197/3.377 | pass | +40.5%/+82.3% | +32.1%/+81.4% |
| transformer6 | 1.000 | 1.000 | 1.000 | 1.000 | 0.00227 | 1,195,024 | 15.328/16.431 | 17.233/17.281 | pass | -236.6%/+0.0% | -266.1%/+0.0% |
| flat_mlp_matched | 1.000 | 1.000 | 0.992 | 0.953 | 0.00674 | 22,075 | 2.255/2.678 | 2.614/2.848 | pass | +50.5%/+85.3% | +44.5%/+84.8% |

| arm | one-step MSE free | one-step MSE object-blocked | one-step MSE boundary-blocked | rollout H@1 h10 free | rollout H@1 h10 object-blocked | rollout H@1 h10 boundary-blocked | counterfactual response MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_gnn | 0.00001 | 0.00002 | 0.00001 | 1.000 | 1.000 | 1.000 | 0.00001 |
| local | 0.00019 | 0.00340 | 0.00000 | 1.000 | 1.000 | 0.981 | 0.00114 |
| local4 | 0.00019 | 0.00341 | 0.00000 | 0.994 | 1.000 | 0.981 | 0.00114 |
| global4 | 0.00021 | 0.00285 | 0.00000 | 0.988 | 1.000 | 1.000 | 0.00098 |
| global16 | 0.00024 | 0.00281 | 0.00001 | 0.988 | 1.000 | 1.000 | 0.00099 |
| transformer6 | 0.00005 | 0.00005 | 0.00005 | 1.000 | 1.000 | 1.000 | 0.00001 |
| flat_mlp_matched | 0.00021 | 0.00330 | 0.00004 | 0.994 | 1.000 | 0.981 | 0.00109 |

Speed 列格式为“相对 full_gnn / 相对 transformer6 的延迟降低比例”；门槛按 60 repeats 的 median 计算，`+20%` 才达到预设速度门槛；p90 作为尾延迟一并显示。quality 表示 h10 H@1 不比 full GNN 低超过 3 个百分点。B=1、B=300 单独判定。encoder image-forward 的 median/p90 初始成本在 summary 中另列；predictor latency 包含完整 10 步 rollout 的 message/token 运算。

## 通信、动作响应与数据子集

`global4` vs `local4` 是主要通信对比：两者使用相同局部函数，新增投影参数均为 320，初始化配对。`flat_mlp_matched` 用于辨别结构收益是否只是参数减少。Counterfactual action-response 从相同图像/latent 对比存储动作与同一 object 的反向方向动作；MSE 越低表示预测的动作响应变化越接近 frozen encoder 编码的真实后继差分。

逐 arm 的 `free`、`other-object-blocked`、`boundary-blocked` 单步 latent MSE、counterfactual response MSE 和响应幅度保存在 `summary.json`。Test 状态类别由位置仅在评估时计算，训练过程不读取这些字段。每 seed 的 episode-level 误差、H@1、MRR 与 counterfactual response 数据保存在 `episode_metrics_*.npz`。

## 配对不确定性与运行环境

H@1 h10 的 paired episode bootstrap 95% CI；除 `global4-local4` 行外均相对 full GNN：

| seed/arm | H@1 差异 | paired 95% CI |
|---|---:|---:|
| 1101/full_gnn-full_gnn | +0.000 | [+0.000, +0.000] |
| 1101/local-full_gnn | +0.230 | [+0.180, +0.285] |
| 1101/local4-full_gnn | +0.230 | [+0.180, +0.285] |
| 1101/global4-full_gnn | +0.234 | [+0.184, +0.289] |
| 1101/global16-full_gnn | +0.234 | [+0.184, +0.289] |
| 1101/transformer6-full_gnn | +0.242 | [+0.191, +0.297] |
| 1101/flat_mlp_matched-full_gnn | +0.230 | [+0.180, +0.285] |
| 1101/global4-local4 | +0.004 | [+0.000, +0.012] |
| 1102/full_gnn-full_gnn | +0.000 | [+0.000, +0.000] |
| 1102/local-full_gnn | -0.008 | [-0.020, +0.000] |
| 1102/local4-full_gnn | -0.008 | [-0.020, +0.000] |
| 1102/global4-full_gnn | -0.008 | [-0.020, +0.000] |
| 1102/global16-full_gnn | -0.008 | [-0.020, +0.000] |
| 1102/transformer6-full_gnn | -0.004 | [-0.012, +0.000] |
| 1102/flat_mlp_matched-full_gnn | -0.004 | [-0.012, +0.000] |
| 1102/global4-local4 | +0.000 | [+0.000, +0.000] |
| 1103/full_gnn-full_gnn | +0.000 | [+0.000, +0.000] |
| 1103/local-full_gnn | -0.004 | [-0.012, +0.000] |
| 1103/local4-full_gnn | -0.008 | [-0.020, +0.000] |
| 1103/global4-full_gnn | -0.008 | [-0.020, +0.000] |
| 1103/global16-full_gnn | -0.008 | [-0.020, +0.000] |
| 1103/transformer6-full_gnn | +0.000 | [+0.000, +0.000] |
| 1103/flat_mlp_matched-full_gnn | -0.008 | [-0.020, +0.000] |
| 1103/global4-local4 | +0.000 | [-0.012, +0.012] |

实际环境：Torch `2.8.0+cu128`、CUDA `12.8`、GPU `NVIDIA A100-SXM4-40GB`、precision `float32, autocast off`、matmul TF32 `False`、cuDNN TF32 `False`。GPU 利用率和显存记录在 job log。

## 适用边界

三 seeds 分别报告，bootstrap 以 held-out episode 为单位；这些区间不代表训练 seed population。视觉 reference 未达到任一 seed 的预设 adequacy 时，全部 predictor 对比仍完整报告，但整体标为 inconclusive。该环境是离散 5×5 grid，不能代表连续 PushT 接触动力学。六层 Transformer 是共用 slots 接口的新架构对照，不是已训练的 LeWM checkpoint。结果不支持 CEM candidate ranking 或闭环控制结论。

## 设计来源

官方机制来源固定到 C-SWM commit；本实验做了明确记录的现代兼容改动，不是论文配置复现。冻结设计按 episode 分割、seed 内配对 minibatches，并以 episode 作为 bootstrap 单位。实验设计来源：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065。
