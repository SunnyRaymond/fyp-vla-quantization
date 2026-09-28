# C-SWM 视觉分块动力学 pilot：易读结果

## 先看结论

这次 pilot 按冻结协议完成了全部 **24 项训练**：3 个 Stage A encoder/GNN 联合训练，每个训练 5,000 updates；以及 3 个 seed × 7 个 predictor 的 21 项 Stage B，每项也训练 5,000 updates。没有根据 dev/test 结果改训练步数、挑 checkpoint 或追加实验。

**整体结论是 inconclusive（证据不足以判定分块方法有效）。** 预先规定的 reference adequacy 要求三个 seed 的 full GNN 都达到绝对检索指标，并且 h10 比 persistence 至少高 0.05。1102、1103 通过；1101 的 h1、h10 绝对门槛都通过，但相对 persistence 只高 0.0391，未到 0.05。因此全 seed reference 门槛失败。这个失败不能解释成 encoder 没学到视觉位置：三个 seed 的 latent 都没有近零维度，位置 probe 的 test RMSE 是 0.022、0.114、0.020 个格子。

本次最有用的机制信号是：`global4` 相对参数和初始化严格匹配的 `local4`，在三 seed 的单步 object-blocked MSE 都低约 **13%–16%**；但主指标 h10 H@1 的差别只有 0 或 1/256（仅一局的命中差），三组 paired bootstrap 95% CI 都包含 0。也就是说，低维跨块信息可能帮助了需要知道邻居占位的单步转移，但没有显示出可靠的 h10 检索收益。

小模型都比 full GNN 与六层 Transformer6 快，但**参数匹配的 flat MLP h10 几乎一样、而且每个 seed 的 B=1 和 B=300 median 都比 global4 快**。所以速度观察不能归因于“分块通信结构”本身。Transformer6 是本实验新训练的架构对照，不是 LeWM checkpoint。这个 pilot 没有测试 LeWM、CEM candidate ranking 或闭环控制。

## 实验具体做了什么

固定环境是 5 个物体、5×5 网格、50×50 RGB 图像，每局 40 步；动作编码为 `object_id × 4 + direction`。越界或目标格已有物体时动作不改变状态。训练、dev、test 分别为 1,000、128、256 个完整 episode；训练输入只有图像和动作，位置与阻挡标签只用于训练外诊断。渲染和随机数采用记录在冻结协议中的兼容实现，因此这是机制导向的 pilot，**不是逐项复现 C-SWM 论文训练配置**。官方代码来源固定为 commit `e944b24bcaa42d9ee847f30163437a50f0237aa0`。

Stage A 每个 seed 联合拟合图像 encoder 和 full-pair GNN，再冻结该 encoder。由此得到的表示天然经历过 GNN 训练，比较有利于 GNN 的表示偏置；所以本实验最多能说明这种 C-SWM object-slot 接口在这个小环境中的结果，不能外推到 LeWM 的 latent。

Stage B 比较七种 predictor：full GNN；无通信 local；每个 slot 只看自己的 4D 摘要的 local4；所有 slot 共用 4D 全局摘要的 global4；16D 全局摘要 global16；六层 Transformer6；以及参数匹配 flat MLP。每个模型都收到相同的 per-object action 输入。主指标是 h10 H@1：从 step 0 开始滚动预测 10 步，把预测 latent 与 256 个 held-out episode 的 h10 target 比距离，看正确 target 是否最近；同一个固定候选集合供所有模型和 persistence 使用。每个 horizon 每局只有一个检索查询。

## 冻结检查与完成状态

作业 `25564986.pbs101` 在 `x1000c1s7b0n0` 完成，退出状态 0；实际设备是 NVIDIA A100-SXM4-40GB，Torch 2.8.0+cu128、CUDA 12.8，float32、autocast/TF32 均关闭。7 项模型机制检查全部通过：action target routing、encoder `[B,5,16]` 输出、全部七种 predictor 的有限值和形状、flat MLP 参数预算、local4 无跨 slot 梯度路径、global4 有跨 slot 信息路径、local4/global4 参数与初始化匹配。PBS allocation guard 和 prepared-data 冻结配置一致性两项也通过。

Global4 与 local4 各有 22,096 个参数，其中通信投影均为 320 个参数；local predictor 函数和投影初始化配对。参数匹配 flat MLP 有 22,075 个参数，和 global4 相差 −0.095%。Full GNN 为 75,408 参数；global16 为 24,592（比 global4 多 2,496，约 +11.3%）；Transformer6 为 1,195,024。因 global16 多了容量，它不能单独用于判断更宽的信息通道是否更有效。

## Reference adequacy：为何整体是 inconclusive

| seed | full GNN h1 H@1 | full GNN h10 H@1 | persistence h10 H@1 | GNN 比 persistence 高 | adequacy |
|---:|---:|---:|---:|---:|:---:|
| 1101 | 1.0000 | 0.7578 | 0.7188 | 0.0391 | 未通过 |
| 1102 | 1.0000 | 1.0000 | 0.6992 | 0.3008 | 通过 |
| 1103 | 1.0000 | 1.0000 | 0.7188 | 0.2813 | 通过 |

门槛是 h1≥0.80、h10≥0.50、且 h10 比 persistence 至少高 0.05。1101 不是 h1 或 h10 绝对分数不够，而是第三项差 0.0109。由于冻结规则要求三个 seed 全通过，整体只能判 inconclusive。

其他诊断不支持“1101 的图像 encoder 塌陷”这个解释：三个 seed 的 test latent 平均标准差分别为 0.291、0.187、0.296；80 个 slot-dimension 中低于 `1e-3` 的数量都是 0；train-only 线性位置 probe 的 test RMSE 分别为 0.022、0.114、0.020 个格子。另一个线索是 1101 full GNN 在 free 动作反事实上的响应明显偏小：预测响应 RMS 0.0090，编码目标响应 RMS 0.1139，response MSE 0.01253。1102、1103 的对应 GNN 响应幅度分别是 0.0691/0.0742、0.1185/0.1182。它说明 1101 的 GNN reference 动作响应弱；本结果不足以确定弱响应的优化原因。

Persistence 的 h1 H@1 在三个 seeds 都是 1.0，h10 仍约 0.70，而多数 learned predictors 的 h10 约 0.99。测试随机动作中有一部分会因越界或碰撞成为 no-op；10 步后静态 latent 仍可能在随机场景的候选库里被正确检索。因此这个近饱和的 H@1 不能单独证明高精度动力学，也会让模型间的小差别难以分辨。

## 每个 seed、每个 predictor 的四个 horizon

以下为 held-out H@1；四列依次是 h1、h5、h10、h20。全部结果都保留，包括 reference 较弱的 1101，以及 persistence 对照。

| seed | predictor | h1 | h5 | h10 | h20 |
|---:|---|---:|---:|---:|---:|
| 1101 | persistence | 1.0000 | 0.9766 | 0.7188 | 0.4219 |
| 1101 | full GNN | 1.0000 | 0.9805 | 0.7578 | 0.3594 |
| 1101 | local | 1.0000 | 0.9961 | 0.9883 | 0.9570 |
| 1101 | local4 | 1.0000 | 0.9961 | 0.9883 | 0.9531 |
| 1101 | global4 | 1.0000 | 0.9961 | 0.9922 | 0.9805 |
| 1101 | global16 | 1.0000 | 0.9961 | 0.9922 | 0.9727 |
| 1101 | Transformer6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1101 | flat MLP matched | 1.0000 | 0.9961 | 0.9883 | 0.9414 |
| 1102 | persistence | 1.0000 | 0.9570 | 0.6992 | 0.3906 |
| 1102 | full GNN | 1.0000 | 1.0000 | 1.0000 | 0.9531 |
| 1102 | local | 1.0000 | 0.9961 | 0.9922 | 0.9531 |
| 1102 | local4 | 1.0000 | 0.9961 | 0.9922 | 0.9492 |
| 1102 | global4 | 1.0000 | 0.9961 | 0.9922 | 0.9648 |
| 1102 | global16 | 1.0000 | 0.9961 | 0.9922 | 0.9648 |
| 1102 | Transformer6 | 1.0000 | 1.0000 | 0.9961 | 1.0000 |
| 1102 | flat MLP matched | 1.0000 | 1.0000 | 0.9961 | 0.9141 |
| 1103 | persistence | 1.0000 | 0.9727 | 0.7188 | 0.4180 |
| 1103 | full GNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1103 | local | 1.0000 | 0.9961 | 0.9961 | 0.9609 |
| 1103 | local4 | 1.0000 | 0.9961 | 0.9922 | 0.9648 |
| 1103 | global4 | 1.0000 | 0.9961 | 0.9922 | 0.9766 |
| 1103 | global16 | 1.0000 | 1.0000 | 0.9922 | 0.9766 |
| 1103 | Transformer6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1103 | flat MLP matched | 1.0000 | 1.0000 | 0.9922 | 0.9531 |

### 主机制对比：global4 和 local4

两者的参数数目、local 函数初始化、投影参数数目都匹配。h10 H@1 的 paired episode 差值（global4 − local4）为：

| seed | H@1 差值 | episode bootstrap 95% CI | 对应 episode 数 |
|---:|---:|---:|---:|
| 1101 | +0.0039 | [0.0000, +0.0117] | +1/256 |
| 1102 | 0.0000 | [0.0000, 0.0000] | 0/256 |
| 1103 | 0.0000 | [−0.0117, +0.0117] | 0/256 |

三个区间都包含 0。这不是证实两者完全相同，而是这项冻结评估没有测到清晰的 h10 检索优势；接近天花板的 H@1 也限制了分辨率。global4 在 h20 H@1 高于 local4 约 0.012–0.027（按 seed）；这是较远 horizon 的描述性结果，不是冻结的主 gate，也没有单独的预设显著性结论。

`flat MLP matched` 也给出重要反例：它与 global4 参数只差 −0.095%，h10 H@1 是 0.9883、0.9961、0.9922；global4 是 0.9922、0.9922、0.9922。h10 质量非常接近，但 flat MLP 在六个 seed×batch 计时条件下都比 global4 快。global4 在 h20 的 H@1 则比 flat 高约 0.024–0.051。整体上，结果没有显示 h10 质量或延迟收益必须来自跨块通信结构；低参数量的平面预测器也能做得接近。

`global16` 的 h10 与 global4 相同到约 0.004 以内，但它多了约 11.3% 参数，延迟有时快、有时慢。这不是容量匹配的消息宽度对照，不能据此认定 16D 通信更好或更差。

## 单步阻挡动作、rollout 子集和 action response

对 held-out 的 10,240 个单步转移，1,478 个属于 other-object-blocked，1,956 个属于 boundary-blocked，其余 6,806 个是 free。下表只列主要 local4/global4 匹配对比的单步 latent MSE：

| seed | predictor | free | other-object-blocked | boundary-blocked |
|---:|---|---:|---:|---:|
| 1101 | local4 | 0.000139 | 0.003355 | 0.000004 |
| 1101 | global4 | 0.000171 | 0.002809 | 0.000004 |
| 1102 | local4 | 0.000090 | 0.001343 | 0.000014 |
| 1102 | global4 | 0.000102 | 0.001173 | 0.000014 |
| 1103 | local4 | 0.000187 | 0.003409 | 0.000005 |
| 1103 | global4 | 0.000214 | 0.002851 | 0.000005 |

object-blocked 动作确实明显比 free 更难做单步预测；global4 在三 seed 的 object-blocked MSE 比 local4 低约 13%–16%，但 free MSE 略高，boundary 的绝对误差近零。这个方向与“碰撞时需要邻居信息”相符，是机制线索；它没有转化成明显的主指标优势。

rollout 子集按 **step 0 动作的类别**分组，不是检查十步内每个动作都属于该类。global4/local4 在 object-blocked 起始组的 h10 H@1 大多是 1.0，不能读成“十步中所有阻挡都预测正确”。因为同一个预测还只需在其他随机场景的 latent 候选中检索到正确 episode，且 h10 persistence 本身约 0.70；因此该分类的 H@1 对细粒度碰撞误差不够敏感。单步的 1,478 个 object-blocked 转移更直接揭示局部模型少了邻居占位信息；即便如此，global4 的该项 MSE 仍比 Transformer6 大约 16–400 倍，不能说动力学精度等同 Transformer。

Counterfactual action-response 用相同起点 latent 比较记录动作与同一物体反方向动作，评估两种预测变化之差是否贴近 frozen encoder 的目标变化。free 起始动作的例子：1101 full GNN 预测响应 RMS 0.0090，而目标是 0.1139；global4 是 0.1039、response MSE 0.000672。1102 的 full GNN/global4 预测 RMS 为 0.0691/0.0659，目标 0.0742；1103 分别为 0.1185/0.1058，目标 0.1182。global4 对 local4 的 free response MSE 三 seed 为 0.000672/0.000384/0.000780 对 0.000676/0.000361/0.000809，差别很小且方向不完全一致。它支持 1101 GNN reference 的动作响应偏弱，但不支持 global4 在全部 seed 稳定改善反事实动作响应。

其他 arm 与三种 event 的逐项单步 MSE、按 step-0 类别的 rollout、response 幅度和误差均保留在下文链接的原始结果文件。

## 速度：测到了什么，不能声称什么

计时是同一 A100 allocation 上的 predictor-only、10 步 rollout；每种 batch 先 warm up 20 次，再同步测 60 次。表内每格为 median/p90 毫秒；batch=300 使用 held-out episode 起点循环取样。图像 encoder 的单次前向另计，不包含在 predictor latency 里。

### Seed 1101

| predictor | B=1 median/p90 ms | B=300 median/p90 ms |
|---|---:|---:|
| full GNN | 5.368 / 5.386 | 5.555 / 5.581 |
| local | 2.736 / 2.784 | 2.428 / 2.443 |
| local4 | 3.321 / 3.339 | 3.496 / 3.514 |
| global4 | 2.686 / 2.694 | 2.815 / 3.109 |
| global16 | 3.175 / 3.222 | 2.814 / 3.181 |
| Transformer6 | 13.788 / 14.672 | 14.592 / 15.958 |
| flat MLP matched | 2.238 / 2.531 | 2.378 / 2.825 |

### Seed 1102

| predictor | B=1 median/p90 ms | B=300 median/p90 ms |
|---|---:|---:|
| full GNN | 5.384 / 5.403 | 5.411 / 5.559 |
| local | 2.362 / 2.794 | 2.455 / 2.750 |
| local4 | 3.353 / 3.373 | 3.252 / 3.527 |
| global4 | 3.151 / 3.238 | 2.840 / 3.168 |
| global16 | 2.699 / 3.181 | 3.326 / 3.337 |
| Transformer6 | 14.359 / 15.875 | 15.874 / 17.150 |
| flat MLP matched | 2.280 / 2.708 | 2.396 / 2.410 |

### Seed 1103

| predictor | B=1 median/p90 ms | B=300 median/p90 ms |
|---|---:|---:|
| full GNN | 4.553 / 5.077 | 4.707 / 4.876 |
| local | 2.384 / 2.806 | 2.458 / 2.602 |
| local4 | 2.844 / 3.260 | 2.951 / 3.467 |
| global4 | 2.691 / 2.896 | 2.913 / 3.340 |
| global16 | 2.709 / 3.212 | 3.197 / 3.377 |
| Transformer6 | 15.328 / 16.431 | 17.233 / 17.281 |
| flat MLP matched | 2.255 / 2.678 | 2.614 / 2.848 |

在五个小 predictor（local、local4、global4、global16、flat MLP）中，相对 full GNN 的 median 延迟下降范围为 B=1 **37.6%–58.3%**、B=300 **32.1%–57.2%**；相对 Transformer6 为 B=1 **75.9%–85.3%**、B=300 **76.0%–84.9%**。global4 自己相对 GNN / Transformer6 的降幅分别是：

| seed | B=1：vs GNN / vs Transformer6 | B=300：vs GNN / vs Transformer6 |
|---:|---:|---:|
| 1101 | 50.0% / 80.5% | 49.3% / 80.7% |
| 1102 | 41.5% / 78.1% | 47.5% / 82.1% |
| 1103 | 40.9% / 82.4% | 38.1% / 83.1% |

这些是本次特定实现、A100、batch/horizon 和精度设置下的真实计时，不含 encoder，也不是 LeWM 的延迟。参数匹配 flat MLP 每个 seed、两个 batch 都比 global4 快；因此这里的速度证据更支持“较小 predictor 可以省时”，不支持“分块后必然更快”。Transformer6 参数约为 global4 的 54 倍，是新建的 slots-interface 对照；它不是 LeWM，也不能用来宣称对 LeWM 有相同速度优势。

## 应怎样解读这些结果

1. **关于“多数动力学可在块内完成，只有少量跨块信息”**：local/local4 在此离散任务上 h10 H@1 很高，且 global4 只需 4D 全局摘要；object-blocked 单步误差有 13%–16% 的小幅下降。这与局部主导、碰撞处需要少量邻居信息的想法相容，但 h10 的 global4−local4 区间包含 0，不能说已经验证。
2. **关于“分解能加速”**：所有小模型比 75k 参数 full GNN 快，说明降低 predictor 规模是可行的；但 22k 参数 flat MLP 同样有近似 h10，且比 global4 快，所以分块结构本身的加速优势没有被分离出来。
3. **对 LeWM/CEM 的边界**：这里输入是明确的 5 个 slot 视觉 latent，环境是随机动作下的 5×5 离散网格。没有输入 LeWM checkpoint、没有跑 CEM，也没有执行闭环策略。因此它不能回答 LeWM 高维 latent 如何拆块，也不能证明真实 planner 更快或任务成功率不变。

结论只适用于这个冻结 pilot：保留“低维跨块摘要对阻挡单步预测可能有帮助”作为后续假设；当前证据对主 h10 优势不确定，对专属于分块结构的速度收益不支持。此结果不触发加训练预算或自动转入 LeWM 实验。

## 可复核的原始输出

本文件是面向阅读的解释，原始数值和逐 arm 诊断保持不变：

- `results/25564986.pbs101/REPORT.zh.md`：逐 seed/arm 的完整 H@1、MSE、event 子集、延迟和 bootstrap 表。
- `results/25564986.pbs101/summary.json`：完整结构化 summary。
- `results/25564986.pbs101/DECISION.json`：冻结 gate 的逐 seed 决策及 inconclusive 结论。
- `results/25564986.pbs101/mechanism_tests.json`：7 项模型机制检查与 allocation/config 检查。
- `results/25564986.pbs101/episode_metrics_1101.npz`、`episode_metrics_1102.npz`、`episode_metrics_1103.npz`：逐 episode 指标。
- `FREEZE.json` 和 `PROTOCOL.zh.md`：预注册式冻结配置与方法说明。

