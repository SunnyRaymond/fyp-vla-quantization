# Fast-WAM：weight / activation 误差在哪里，以及这对 Smooth → VQ 意味着什么

2026-10-08 完成。GPU 采集 `25727294.pbs101` 和最终 CPU 绘图、汇总 `25728448.pbs101` 均为 **F、Exit_status=0**，完成文件检查通过；**10 张最终图已逐张查看**。本轮没有重新拟合 codebooks，也没有做 BF16 恢复实验。

这批图支持三个具体判断：**Smooth + Hadamard 明显缓解了 activation 的 outlier 问题；当前 VQ 的 weight 误差在部分 cross-attention output projections 中集中于少量输入列；继续加大 Smooth 并不能保证 Linear 输出更准确。** 因而，下一步的保护规则需要考虑误差经过真实输入和 weight 后的影响，不能仅按 weight 幅值或 activation 重建误差排序。

## 1. 本轮到底比较了什么

模型是原两阶段实验的 released Fast-WAM Optional-IDM clean checkpoint。全部模型 forward 保持 original BF16，hooks 只做旁路统计，不替换参数或输出。Scalar / VQ 因此面对同一次 BF16 forward 的相同输入。

| 项目 | 本轮范围 |
|---|---|
| 输入 | 已有 cases **1、13、21**，各使用 sampler seed index **0** |
| 新模型查询 | **3 次 BF16 queries**；没有新增量化模型 queries |
| 输入性质 | 已知的 exploratory inputs；case 1 为原始输入，13 为 camera 输入，21 为 background 输入 |
| Linear 覆盖 | 每次 **614/614**：Video 306、Action 307、root proprio 1 |
| Activation 配置 | Identity、Hadamard、Smooth α=0.5 + Hadamard、Smooth α=1 + Hadamard；均为 per-row absmax A4 |
| Weight 配置 | 两种 Smooth + Hadamard，复用原 Scalar W4 / additive VQ quantized banks，decode 后旁路比较 |
| Weight 记录 | **1,228**，即 614 层 × 2 transforms；不是 1,228 个独立模型 |
| Activation 记录 | **77,232**，即 19,308 次真实 Linear 调用 × 4 configs |
| 细粒度 activation 快照 | **180/180**；5 个有 scheduler 调用的 selected layers × 3 cases × steps 0/4/9 × 4 configs |
| Selected per-channel 文件 | **636**；其中 600 个 scheduler 文件用于浓度汇总，36 个 conditioning 文件单独保留 |
| 轨迹一致性 | 3 次 BF16 action 与原 full 实验保存结果的最大绝对差均为 **0** |
| 环境执行 | **0 episodes、0 个预测动作执行步骤** |

最终 action 热图复用 full 作业 `25722477.pbs101` 保存的 actions。三条可视化输入不能替代原先 10 个 held-out inputs、每个两个 seeds 的正式结果，也不能用来重选 frozen winner。原两阶段结论保留在 [RESULTS.zh.md](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/RESULTS.zh.md>)。

下面的 relative RMSE 定义为 `||误差||F / ||参考信号||F`。跨调用汇总先累加误差能量和参考能量，再开方；不是把每层、每次调用的 relative RMSE 直接平均。0.5 表示误差的 RMS 是参考信号 RMS 的一半，不表示成功率下降 50%。

## 2. Activation：最大的值未必是最大的误差点

先看一个直观例子：**case 13，Video block 29 的 `cross_attn.q`，video denoising step 9**。输入 shape 为 `[1,294,3072]`；以下数值来自全部输入元素。

| 变换 | 最大绝对值 | RMS | absmax / RMS | A4 relative RMSE | 重建值为零的比例 |
|---|---:|---:|---:|---:|---:|
| Identity | 53.750 | 0.970 | **55.42** | **0.5584** | **99.41%** |
| Hadamard | 5.616 | 0.970 | 5.79 | 0.1847 | 41.23% |
| Smooth α=0.5 + Hadamard | 3.350 | 0.579 | 5.79 | **0.1577** | **22.47%** |
| Smooth α=1 + Hadamard | 3.381 | 0.542 | 6.24 | 0.1561 | 21.15% |

![同一个 activation 输入在四种变换下的幅值、重建和误差](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/06_selected_activation_case13.png>)

图的四列依次是原始 `X`、变换后 `Z`、A4 重建 `QZ`、绝对误差。Identity 下少量大值控制整行的量化步长，多数较小特征被舍入为零。Hadamard 把能量分散到混合后的 channels：这个例子中 RMS 保持 0.970，但最大值从 53.75 降为 5.62。Smooth 进一步改变各原始 channel 的尺度，使 A4 更容易表示。

**最大 activation 自己可以被量化刻度保留得很好；受损的反而可能是同一行的其他值。** 例如 Identity 的最大绝对量化误差为 3.59375，精确位置是 `[token 70, channel 1613]`；Smooth α=0.5 下变为 0.24082，位置是 `[token 267, transformed channel 2923]`。定位应看误差图和原矩阵统计，而不是只找 activation 图里最亮的点。

零比例包含输入原本已有的零，不能读成“99.41% 神经元失效”。Hadamard 后的 channel 也是原 channels 的线性组合，跨配置的相同编号不再表示同一原始方向。

这不是单个局部示例独有的趋势。对 **case 13 的各 stream 全部调用，包括 conditioning**，能量汇总结果为：

| Stream | Identity | Hadamard | Smooth α=0.5 + H | Smooth α=1 + H |
|---|---:|---:|---:|---:|
| Video activation relative RMSE | 0.4567 | 0.1913 | 0.1732 | 0.1694 |
| Action activation relative RMSE | 0.4633 | 0.1799 | 0.1731 | 0.1627 |
| Proprio activation relative RMSE | 0.0684 | 0.0550 | 0.0527 | **0.0877** |

Video 和 Action 的汇总指标改善，Proprio 则在 α=1 时变差。上面的单层示例也显示 α=1 的 absmax/RMS 比 α=0.5 更高。因此，“更激进”不能直接等同于“所有地方都更平滑”。

**全层热点怎样看？** [activation layer-step 总览](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/02_activation_layer_step_heatmaps.png>) 按 case、stream 分行，四列按 Identity、Hadamard、Smooth 0.5、Smooth 1 排列，使用共同 relative RMSE 色标。横轴是真实 denoising step 0–9；纵轴是该 stream 内的层顺序，不是全局 module index。Conditioning 单独保留，不混进 scheduler 热图。

在三条输入的 scheduler 调用中，每种配置的最高 activation relative RMSE 是：

| 配置 | Case / 层 / denoising step | Relative RMSE | 最大绝对误差 | 精确误差位置 `[token,channel]` |
|---|---|---:|---:|---|
| Identity | 13 / Video 14 `cross_attn.q` / 0 | 0.7679 | 1.921875 | `[34,1843]` |
| Hadamard | 13 / Action 13 `self_attn.o` / 6 | 0.4618 | 0.378974 | `[21,2824]` |
| Smooth 0.5 + H | 13 / Action 17 `cross_attn.o` / 2 | 0.4575 | **0.001592** | `[29,1855]` |
| Smooth 1 + H | 1 / Action 2 `cross_attn.o` / 4 | 0.4328 | **0.001183** | `[14,2506]` |

后两行说明为什么相对误差不能单独作为保护优先级：它们的原信号本来很小，所以 relative RMSE 很大，绝对误差仍很小。还需要看这种误差经过 Linear 和后续网络之后是否被放大。

## 3. Weight：部分层的 VQ 误差集中在少量输入列

[全 614 层 weight relative RMSE 曲线](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/01_weight_rrmse_by_stream.png>) 显示各层差异；下面用 selected layers 的精确值解释。每个数都以该层、该 transform 的未量化 transformed weight 为参考。

| Selected 层（全局 index） | α=0.5 Scalar | α=0.5 VQ | α=1 Scalar | α=1 VQ |
|---|---:|---:|---:|---:|
| Video 29 `cross_attn.q`（299） | 0.1174 | 0.1120 | 0.1142 | 0.1154 |
| Video 29 `cross_attn.o`（302） | 0.1172 | 0.2516 | 0.1158 | 0.2667 |
| Action 16 `cross_attn.o`（479） | 0.1169 | **0.2728** | 0.1167 | **0.4048** |
| Action 17 `ffn.2`（491） | 0.1166 | 0.1375 | 0.1148 | 0.1497 |
| Action 23 `cross_attn.o`（549） | 0.1166 | 0.2695 | 0.1157 | 0.3172 |
| `proprio_encoder`（613） | 0.0712 | 0.1110 | 0.0669 | **0.0531** |

这 6 层按预先固定的 weight 统计规则选择，包括较差层、median representative 和 Video 较好层；**不是六个已证实的 action-sensitive layers**。

![Smooth alpha 1 下 selected weights 及 Scalar 与 VQ 的绝对误差](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/04_selected_weight_maps_smooth1_hadamard.png>)

图的三列是 transformed weight 幅值、Scalar 绝对误差、VQ 绝对误差。行是 output channels，列是 transformed input channels。**同一层的 Scalar / VQ 共用 error 色标，两种 α 也共用该层的色标；不同层各有自己的色标。** 因而可以比较同一层的量化方法和 α，不能把不同层的红色直接当作相同绝对误差。

最清楚的例子是 **Action block 16 `cross_attn.o`，shape `[1024,3072]`**。对每个输入列，把所有 output rows 的 weight 误差平方相加，再选最大的 **31 列（约 1%）**：

| 指标 | Smooth α=0.5 + H | Smooth α=1 + H |
|---|---:|---:|
| VQ / Scalar weight relative RMSE | **2.33 倍** | **3.47 倍** |
| VQ 最大 31 列占总误差能量 | **36.5%** | **47.4%** |
| Scalar 最大 31 列占总误差能量 | 7.0% | 10.8% |
| VQ 列误差能量排名前四的列 | 482、452、505、388 | 438、489、505、509 |
| VQ 最大绝对误差及 `[row,col]` | 0.41766，`[46,482]` | 1.52484，`[524,499]` |

**约 1% 的列承担近一半误差能量，说明当前 VQ 在这个变换后的坐标系中存在明显的方向集中性。** α=1 下 Action 23 `cross_attn.o` 也有 37.3% 的误差能量集中于最大 31 列；Video 29 `cross_attn.q` 则只有 3.6%。不能把 VQ 的问题概括成所有层都一样。

但这些列还不能直接称为“最值得保护的 action-sensitive 方向”：大 weight 误差未必会被当前输入激活，后续网络也可能削弱或放大它。Hadamard 后列编号还对应混合方向。

## 4. 关键补充：误差经过真实输入后，排名会改变

设原 Linear 为 `Y = XWᵀ`，Smooth 对角缩放为 `S`，signed block-Hadamard 为 `R`。本轮沿用原实验的 `S`：geometric-mean normalization、clamp `[1/16,16]`，Hadamard block 128。

```text
Z = X S⁻¹ R
B = W S R
Z Bᵀ = X Wᵀ

dZ = QZ − Z       activation 重建误差
dB = Bhat − B    weight 重建误差

QZ Bhatᵀ − Z Bᵀ = dZ Bᵀ + Z dBᵀ + dZ dBᵀ
                    eA        eW        interaction
```

`eA` 的直观含义是“activation 的误差经过未量化 weight 后还有多大”；`eW` 是“weight 的误差面对真实输入时产生多少输出偏差”。即使 `dZ` 更小，变换后的 `B` 也可能把它投影得更大。

三项是误差向量的和，**三个 RMSE 不能直接相加**，因为它们可能相互抵消或放大。

本轮每次调用抽取相同的 deterministic uniform、最多 **16 行 BF16 输入**，关闭 TF32，用 FP32 算旁路投影。以下结果汇总三条输入、全部调用，包括 conditioning；参考分母是相同的 original Linear 输出能量。这是 sampled local proxy。

![全层局部输出误差及 activation、weight、interaction 分量](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/03_local_output_error_components.png>)

两个不同问题浮现出来：

| 层 / 配置 | Scalar 总 relative RMSE | VQ 总 relative RMSE | Activation 分量 eA | Scalar weight 分量 eW | VQ weight 分量 eW |
|---|---:|---:|---:|---:|---:|
| Video 17 `cross_attn.o` / α=0.5 | 0.1246 | **0.7127** | 0.1209 | 0.0303 | **0.7009** |
| Video 17 `cross_attn.o` / α=1 | 0.0942 | **0.8195** | 0.0899 | 0.0304 | **0.8139** |
| Action 21 `cross_attn.o` / α=0.5 | 0.3084 | 0.3160 | **0.3005** | 0.0630 | 0.1148 |
| Action 21 `cross_attn.o` / α=1 | 0.6092 | 0.6122 | **0.6044** | 0.0835 | 0.1909 |

**Video block 17 的 VQ 问题主要体现在 weight 分量。** 它是 VQ 局部输出误差最大的层，却不在按 raw weight 统计选出的 selected6 中。这说明细图选层也应该结合真实输入投影，单靠参数空间排序会漏掉值得检查的层。

**Action block 21 则主要体现在 activation 分量。** α 从 0.5 增至 1 后，`eA` 从 0.3005 增至 0.6044。这里 `eA` 使用未量化的 transformed weight，说明只把 `Z` 的 A4 重建压得更好，还不足以控制输出误差。不能默认“所有风险都能移给 weight VQ 来消化”。

这不表示每层都出现上述极端情况。全层统计为：

| 配置 / Stream | VQ 总局部误差高于 Scalar 的层数 | 各层 VQ / Scalar 总局部 relative RMSE 的中位数 |
|---|---:|---:|
| α=0.5 / Video | 213/306 | 1.0213 |
| α=0.5 / Action | 226/307 | 1.0278 |
| α=1 / Video | 230/306 | 1.0356 |
| α=1 / Action | **286/307** | 1.0527 |

多数层的差异比较温和，但存在很大的尾部。这里衡量的是 local output，不能与旧报告中的 raw weight RMSE 层数混用，也不能把最大尾部直接认定为最终 action 偏差的原因。

部分旁路计算利用原坐标的等价表达，FP32 变换漂移可能进入分量。小型代数自检的 transform-equivalence 最大绝对漂移为约 `1.49e-6`；这个数字只约束该自检，不能当作每个生产调用的误差上界。

## 5. Codebook 的切分：幅值是一条线索，不能单独决定

[row RMS 与 row relative weight RMSE 散点](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/05_selected_row_amplitude_vs_error.png>) 合并了六个 selected layers。每个 panel 原始有 **13,312 rows**，图上 deterministic downsample 到 4,000；统计表使用全部 rows。

合并散点会混合不同层的分布，且各 panel 坐标轴自动缩放。因此更适合看同一层内部低振幅 Q1、高振幅 Q4 的比较。以下是 α=0.5 下 VQ 的例子：

| 层 | 低 row RMS 的 Q1：mean row relative RMSE | 高 row RMS 的 Q4：mean row relative RMSE |
|---|---:|---:|
| Action 16 `cross_attn.o` | 0.2354 | **0.2898** |
| Video 29 `cross_attn.o` | **0.2605** | 0.2447 |
| `proprio_encoder` | **0.1297** | 0.0907 |

Action 16 中大振幅 rows 更难，Video 29 output projection 和 Proprio 却呈现相反关系。共享 codebook 可能既遇到尾部表示不足，也遇到小振幅 rows 的分辨率不足。**这批数据支持按层检查尺度适配，不支持“一律优先保护最大 weight”或“一律只保护小 weight”。** Row normalization、不同尺度的分组 codebook 是可验证的假设，本轮尚未测试它们。

更贴近本轮问题的局部 VQ 目标是：

```text
Lweight = E ||z dBᵀ||² = tr(dB G dBᵀ)
G = E[zᵀz]       transformed inputs 的 second-moment matrix
```

它关心 weight 误差是否落在真实输入经常激活的组合方向。当前拟合使用量化输入的 diagonal second moment，忽略 cross-channel correlations；本轮 teacher-input 投影保留所抽样输入行中的相关性。上面的尾部提示有必要检查这种差异，但尚不能确认它就是尾部的成因。完整 `G`、低秩近似或 block 内相关性都只是后续候选方法，不是本轮已经实现的 codebook 改进。

## 6. “误差方向”与“action 敏感方向”还隔着一步

本轮计算的 top-channel / top-column 指标衡量**误差能量集中程度**。例如 case 13、Video 29 `cross_attn.q`，汇总 10 个 video steps 后，activation error 最大 31/3072 channels 的能量占比为：Identity **5.28%**、Hadamard **1.13%**、Smooth 0.5 **1.09%**、Smooth 1 **1.08%**。旋转后误差分布接近均匀，仍不能说明这些方向对 action 同样重要。

如果要形式化你之前提出的“方向 1 比方向 2 敏感 100 倍”，可以在明确的层、denoising step 和 hidden representation 上定义：

```text
J = ∂(first-10 motor actions) / ∂h
M = E[Jᵀ D J]
direction_score(v) = vᵀ M v，且 ||v|| = 1
```

`D` 表示 action 坐标的固定权重或尺度归一化；`J` 表示这个 hidden perturbation 经过剩余生成过程对最终 action 的一阶影响。同样大小的小扰动下，如果两个方向的 score 比为 100，预测的 action **平方误差**之比为 100，action 误差幅度之比约为 10。需要先说清“100 倍”比较的是能量还是幅度。

这是一个**待测的 sensitivity 定义**，本轮没有求 `J` 或 `M`，也不据此声称 novelty。建立保护优先级时，应把“方向有多敏感”与“当前 quantizer 在该方向实际造成多大误差”合起来看；一个敏感但几乎无量化误差的方向未必应占用最多额外 bits。

坐标变换也必须一致：若 `z = xT`，其中 `T = S⁻¹R`，按列向量误差表示，有 `δx = T⁻ᵀδz`，因此原坐标 metric `Mx` 应变为 `Mz = T⁻¹ Mx T⁻ᵀ`。不能把原始 channel 的保护名单直接套到 Hadamard 后相同的 channel 编号上。

## 7. 最终 action：看时间位置，也保留原正式指标

![原 full 实验保存的三个输入上的 action 绝对误差](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/07_action_step_coordinate_absolute_error.png>)

这张图的纵轴是预测 action chunk 的 **时间位置 0–31**，横轴是 motor 坐标 0–5 和 gripper。它与模型内部 denoising step 无关。横线划出主要指标使用的前 10 个 predicted actions，竖线分开 motor 和 gripper。

以下只是每个 case 的 **seed index 0**，first-10 × 6 motor coordinates 相对原 BF16 action 的 RMSE；不是原正式实验的两 seed case 均值。

| Case | α=0.5 Scalar | α=0.5 VQ | α=1 Scalar | α=1 VQ |
|---|---:|---:|---:|---:|
| 1 | 0.04285 | 0.07560 | 0.05270 | 0.06122 |
| 13 | 0.10779 | **0.09139** | 0.10145 | **0.24978** |
| 21 | 0.10093 | 0.13434 | 0.12676 | **0.12282** |

VQ 在部分单 seed 示例中更好，不能说每条可视化输入都更差；这也不能推翻原 held-out 配对结果。已知输入的局部改善不构成新的配置选择依据。

误差位置也影响如何读图：case 13、α=1 VQ 在主要前 10 步中的最大 motor 绝对误差为 **0.57031，位置 `[time 0,motor 2]`**；case 21、α=1 VQ 在完整 32 步中的最大值为 **1.20703，位置 `[time 28,motor 0]`**，但它在主要前 10 步的最大值是 **0.29663，位置 `[time 2,motor 2]`**。图尾部的大红块不能直接替代前 10 步主要指标。所有 action 数值均为 normalized coordinates，不是米、角度或成功率。

## 8. 读图路线和完整交付

建议先读 **06 case 13 → 04 α=1 → 03 → 07**：先理解 activation 怎样被压平，再看 VQ 在 weight 中哪里出错，然后看局部投影是否放大误差，最后对照原实验最终 action。

| 文件 | 用途 |
|---|---|
| [01 全层 weight 曲线](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/01_weight_rrmse_by_stream.png>) | 找 stream 和层级差异 |
| [02 activation layer-step 总览](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/02_activation_layer_step_heatmaps.png>) | 找真实 denoising steps 的局部热点 |
| [03 局部输出误差分解](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/03_local_output_error_components.png>) | 区分 eA、eW、interaction 和总误差 |
| [04 selected weights：α=0.5](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/04_selected_weight_maps_smooth05_hadamard.png>) | 看矩阵误差分布 |
| [04 selected weights：α=1](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/04_selected_weight_maps_smooth1_hadamard.png>) | 在同层色标下比较 Smooth 强度 |
| [05 row 振幅散点](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/05_selected_row_amplitude_vs_error.png>) | 检查尺度与相对误差的关系 |
| [06 activation：case 1](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/06_selected_activation_case1.png>) | Video 29 q，step 4 |
| [06 activation：case 13](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/06_selected_activation_case13.png>) | Video 29 q，step 9 |
| [06 activation：case 21](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/06_selected_activation_case21.png>) | Video 29 q，step 9 |
| [07 最终 action 绝对误差](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/07_action_step_coordinate_absolute_error.png>) | 看 predicted action 时间位置和坐标 |

大 weight / activation 矩阵通过 nonoverlapping block absmax 池化为最多 64×64：一个亮格表示其覆盖区域至少有一个大值，不代表区域内每个值都很差。精确 peak 坐标来自原矩阵，全部为 zero-based。所选 weight、activation 和 action 绝对误差图使用平方根色彩映射以显示小误差，数值大小应读 colorbar；不能按颜色深浅倍数推断误差倍数。统计不从 PNG 像素反推。

完整数字见 [report_metrics.json](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/report_metrics.json>)，细粒度热点表见 [hotspots.csv](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/hotspots.csv>)；采集覆盖见 [collector summary](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/visualize/summary.json>)，绘图覆盖、row / channel 统计见 [plot summary](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/results/vizplot/summary.json>)。

初始采集 `25726229.pbs101` 因 Python bound-method identity 检查误报退出；修正为方法 equality 检查并增加小型自检后完成上述成功采集。旧失败输出单独保留，未并入成功统计。最终 CPU 作业在已成功的首版绘图基础上补充色标、提高标签可读性并输出上述 measured tables。所有模型、数值汇总和绘图均在获批 PBS compute allocations 内执行，GPU 作业记录了利用率和显存。

本轮已完成误差定位和机制诊断；没有证明某层或方向导致最终 action 偏差，没有做恢复实验、codebook refitting、其他 baseline 数值实验或闭环评估。Local FP32 proxy 和 decoded quantization 结果也不构成 native low-bit kernel 加速证据。
