# Weight / activation 可视化：要找什么，怎样读图

本轮状态（2026-10-08）：GPU 采集 `25727294.pbs101` 与最终 CPU 绘图、汇总 `25728448.pbs101` 已成功完成，10 张最终图已查看；实测结果见 [VISUALIZATION.zh.md](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-smooth-vq-phases12/VISUALIZATION.zh.md>)。初始作业 `25726229.pbs101` 的方法身份检查误报已修正，旧失败输出单独保留。本文记录固定的诊断方法；复用原两阶段 transforms 和 quantized banks，没有重新拟合 codebooks 或做 BF16 恢复实验。

## 1. 最有用的问题

不是只找 weight 或 activation 中最大的值，而是依次问：

1. **哪里难以量化？** 找到误差集中在哪个 stream、Linear、denoising step、token 和 channel。
2. **为什么难？** 判断与 outliers、共享 codebook 的表示范围、低振幅区域的相对误差，或输入分布变化是否有关。
3. **局部误差有多大？** 使用该层实际的 BF16 输入，检查 weight / activation 误差经过 Linear 后的大小。
4. **最终 action 在哪里偏离？** 对照已保存的 action 序列，看哪个时间位置和 motor / gripper 坐标的偏差大。

前面三项能提供候选原因，不能单靠相关性断言某个 channel 导致最终 action 出错。最终 action sensitivity 和恢复实验需要独立验证，本轮不执行。

## 2. 公平比较：先固定输入

模型始终保持 original BF16。每个目标 Linear 的输入记为 `X`，weight 记为 `W`，原输出为 `Y = X Wᵀ`。旁路计算量化重建误差，不改变模型参数和模型输出。

这样比较 Scalar 与 VQ 时，两者看到的是同一次 BF16 forward 的同一个输入。否则量化模型产生的输入已经不同，图中的差异会混合本层误差和之前各层累积的误差。

对于 Smooth + Hadamard，记对角缩放矩阵为 `S`，正交的 signed block-Hadamard 为 `R`：

```
Z = X S⁻¹ R
B = W S R
Z Bᵀ = X Wᵀ
```

本轮 activation 对比 Identity、Hadamard、Smooth α=0.5 + Hadamard 和 Smooth α=1 + Hadamard。Weight 对比后两种变换下已经拟合好的 Scalar 与 VQ。量化后记为 `QZ` 和 `Bhat`。

Smooth 的 α 越大，越偏向降低 activation 的 channel 尺度差异；并不保证最终 action 误差单调下降。现有实现对 `S` 做 geometric-mean normalization 并限制到 `[1/16,16]`，Hadamard block 为 128。这里沿用原实验的实际 transforms，不从这三条可视化输入重新选择 α。

## 3. 先看全局，再看局部

| 图 | 横纵轴 / 内容 | 回答的问题 |
|---|---|---|
| 全层 weight error 曲线 | module index × relative RMSE；Video / Action 分开 | VQ 是普遍变差，还是只有少数层变差？ |
| activation layer-step 热图 | Linear × 实际 denoising step；四种变换 | 哪些层和 steps 难量化？变换是否把问题迁到别处？ |
| local Linear-output error | 每层总误差与 activation / weight / interaction 分量 | 参数重建误差经过真实输入后有多大？ |
| selected weight 热图 | transformed weight、Scalar error、VQ error | 误差集中在行、列还是局部区域？ |
| weight row 振幅与相对误差散点 | row RMS × row relative RMSE | 大振幅或小振幅行是否特别难表示？ |
| activation 四联图 | 原始 X、变换后 Z、Q4 重建 QZ、量化 error | Smooth / Hadamard 如何改变分布及误差位置？ |
| 已保存的 action error 热图 | action 时间位置 × action 坐标 | 最终 action 偏差集中在哪里？ |

扫描范围是全部 614 个 Linear：Video 306、Action 307、root proprio 1。保存细图的 6 层由已有 weight 结果确定：Action 的最大 VQ/Scalar ratio、最大 VQ relative error、median-ratio representative；Video 最大和最小 ratio；root proprio。若重复，按固定规则去重补齐。

选层用于解释现象，不代表这 6 层已经被证明最 action-sensitive。

## 4. 绝对误差与相对误差都要看

对任意矩阵或选定区域 `T`，重建为 `That`：

```
absolute error = |That − T|
relative RMSE = ||That − T||F / ||T||F
```

绝对误差说明偏差的实际尺度，相对误差说明损失占原信号的比例。若原信号能量很小，相对误差可以很高，但不一定有很强的最终影响，因此还要看 signal energy 和 local output error。

同一层、同一 transform 下的 Scalar / VQ error 图使用共同色标；跨配置的 activation 示例也使用共同色标。不能分别自动调整色标后，仅凭颜色比较方法好坏。

跨层/跨调用汇总时先累加能量，再开方归一化：

```
aggregate relative RMSE = sqrt(sum(error energy) / sum(signal energy))
```

不直接平均不同调用的 relative RMSE，避免小信号调用不成比例地影响结果。

## 5. Activation 要沿四个维度定位

`layer × step × token × channel` 是最基本的定位方式。

全层统计记录每次调用的 absmax、RMS、absmax/RMS、zero fraction、relative RMSE，并保留最大绝对误差的精确 `[flattened token row, transformed channel]` 坐标。对 selected6 另保存完整的 per-channel RMS、absmax、error energy 和 zero fraction。

Channel error energy 的排序可判断误差是散布在许多方向，还是集中在少量方向。报告 top 1% channels 和 top 16 channels 占总误差能量的比例。它衡量**误差集中程度**，尚不衡量**最终 action 敏感度**。

真实 scheduler 的 video/action steps 为 0–9；conditioning 单独记为 step -1。`call_index` 是调用顺序，不能当作 denoising step。细粒度 activation 热图保存实际出现的 steps 0、4、9 中每组的首次调用，并在图上写出 case / layer / stage / step。

旋转后的 channel 是原始 channels 的线性组合。若后续要保护原坐标方向，应把误差映回原坐标或让 sensitivity metric 随变换一起变换，而不是把 Hadamard 后的 channel 编号直接当作原方向编号。

## 6. Weight 要区分“数值大”与“codebook 没表示好”

每层绘制 transformed weight 幅值、Scalar error 和 VQ error。矩阵行对应 output channels，列对应 transformed input channels。

另外为每个 output row 计算 RMS 和 relative RMSE，绘制散点，并比较低振幅 Q1 与高振幅 Q4 的 mean relative error。可能观察到：

- 高振幅行误差大：codebook 对尾部或复杂分布的表示可能不足。
- 低振幅行相对误差高：共享 codebook 的尺度/分辨率可能不适配这批行。
- 一些列特别差：可能需要检查分组边界、输入 second moment 与相关性。

这些是待图验证的解释路径，不能预先认定某一种成立。对当前 additive VQ，不仅检查 weight 的 raw RMSE，也检查误差在实际输入上的投影。训练目标中的 diagonal second moment 会忽略 channel 之间的相关性，local output 诊断则保留选中输入行的相关性。

## 7. Local output error 的拆分

令 `dZ = QZ − Z`，`dB = Bhat − B`，则：

```
QZ Bhatᵀ − Z Bᵀ = dZ Bᵀ + Z dBᵀ + dZ dBᵀ
                    activation  weight   interaction
```

这是三项误差向量的和；三项 RMSE 不能直接相加。它们可以互相抵消或放大。图中同时报告总误差与三个分量，用来判断变换是不是把 activation 的问题转移成了更大的 weight/output 问题。

每次调用使用相同的最多 16 行 deterministic uniform BF16 输入，计算 FP32 Linear 投影，关闭 TF32。它是 sampled local proxy，不能代替完整 token 输出、实际 BF16 kernel 误差或最终 action sensitivity。实现为了节省显存，部分投影回到原坐标计算，极小的 FP32 等价变换漂移可能进入 weight / interaction 分量；小型自检检验代数和直接 transformed-weight 计算的一致性。

## 8. 图的空间分辨率与证据范围

大矩阵以 nonoverlapping block absmax 池化到最多 64×64。亮格表示这个区域至少存在一个大值，不表示整个区域都很差。保存 original shape 和每格对应的行列范围；精确 peak 坐标来自原矩阵统计，而非从缩略图猜测。

细粒度输入为 cases 1、13、21，各用 sampler seed index 0，一共 3 次 BF16 model queries。它们是已经看过的 exploratory inputs，不是新的 held-out evaluation。最终 action 图复用原 full 实验保存的 actions，不新增量化模型查询。

本轮能完成：定位误差、比较误差分布、提出有证据的机制假设。尚不能完成：因果确定最敏感方向、证明恢复某层能改善 action、闭环成功率验证、native low-bit kernel 性能测量。本轮不做 recovery experiments，也不重新拟合 codebooks。

可复用协议见 [visualization_protocol.json](visualization_protocol.json)；原两阶段结果见 [RESULTS.zh.md](RESULTS.zh.md)。
