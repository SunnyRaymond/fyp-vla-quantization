# 已知正确坐标的 toy oracle：结果怎么理解

> 这是给阅读者的解释版，依据 PBS job `25564754.pbs101` 已取回的结果整理。原始生成报告保存在 [REPORT.raw.zh.md](REPORT.raw.zh.md)，结构化数值仍以 `summary.json`、`reference_reproduction.json` 和 `DECISION.json` 为准。本文件只补充解释，没有修改任何实验数值，也没有重跑模型或 benchmark。

## 先说结论

给模型正确的坐标，确实大幅改善了三种系统里所有对应的 learned predictors。独立系统的局部模型因此超过 dense predictor；lowrank 系统则只有能读取全局信息的 `oracle_global4` 达到 dense 以上的预测表现；dense-coupled 系统中，全局摘要有帮助，但 4D 摘要仍没有追上 dense。

这把第一轮的两个问题分开了：**分块坐标没学好是普遍瓶颈；坐标学对以后，系统本身需要多少跨块信息仍取决于 dynamics。** 正确坐标没有自动带来更快运行，当前 oracle 分块实现的 30 个 runs 全部未通过 20% speed gate。

## 实验里“知道正确坐标”是什么意思

我们造了一个 64D 系统，真实 dynamics 在四个 16D 块中定义，再用正交矩阵 `M` 把它们混到观察向量 `z` 里。第一轮模型必须一边学坐标 `Q`，一边学 predictor `f`。这轮直接给模型生成器的 `M`，冻结坐标 `s=(z-mean_z)@M`，只训练 `f` 和 message。

每个 oracle predictor/message 与它对应的第一轮 learned arm 使用同 seed、逐张量相同的初始化。唯一的结构改动是把可学习坐标换成正确且固定的 `M`。所以结果回答的是：**如果分块已经分对，原 predictor 还会遇到什么困难？** 这是一种诊断用 oracle，不是实际 latent 中通常可直接取得的坐标。

`oracle_block` 每块只看自己的 16D 状态和 action；`oracle_local4` 还看由本块产生的 4D 摘要；`oracle_global4` 每块可看同一个、从整个 64D 状态产生的 4D 摘要。`global4` 与 `local4` 参数量同为 28,480，且初始化相同，因此两者是判断“读其他块的信息有没有帮助”的直接对照。`block` 与 `global4` 参数量不同，二者差异只能看作加入 message 后的整体结构收益。

## h10 结果

h10 是模型从真实起点开始、连续使用自己的预测 10 步后，最后一步的归一化误差；越低越好。表格给出三个训练 seeds 的均值。`learned → oracle` 左边是第一轮同结构模型，右边是本轮固定正确坐标的模型。

| System | Dense | Block: learned → oracle | Local4: learned → oracle | Global4: learned → oracle |
|---|---:|---:|---:|---:|
| independent | 0.12495 | 0.24762 → 0.07520 | 0.25745 → 0.07442 | 0.18745 → 0.07498 |
| lowrank_coupled | 0.12627 | 0.26551 → 0.14619 | 0.27918 → 0.14614 | 0.19695 → 0.07708 |
| dense_coupled | 0.13121 | 0.28932 → 0.17918 | 0.28590 → 0.17903 | 0.19995 → 0.14390 |

所有 27 个主实验中，oracle 与各自 matched learned arm 的 episode-paired h10 差值在三个 seeds 内方向都一致，且每个 seed 的 episode-bootstrap 95% CI 都低于 0。也就是说，正确坐标不是只帮某个模型或某个系统；它改善了每种对应结构。

## 按系统看，坐标修正分别解决了什么

**Independent：正确坐标基本解开了局部模型。** `oracle_block` 的 h10 从 learned-block 的 0.24762 降到 0.07520，低于 dense 的 0.12495；oracle 对 dense 的配对差值在三个 seeds 中都低于 0。由于生成器保证这个系统没有跨块依赖，结果符合预期：局部 predictor 有足够能力，第一轮主要卡在没有把混合后的 latent 分对。

正确坐标下，等参数的 `global4` 没有比 `local4` 更好。逐 seed 的 `global4 − local4` h10 均值差是 `+0.00055、+0.00036、+0.00079`，略微偏向 local4，幅度很小。这个系统本来没有跨块信号可传；这也说明第一轮 global4 的优势不能一概解释成“真实系统一定需要通信”，它可能在补偿未学好的坐标或 predictor 误差。

**Lowrank：坐标正确仍不够，跨块信息是关键。** `oracle_block` 和 `oracle_local4` 都降到约 0.146，但仍高于 dense 的 0.126；与 dense 的 episode-paired 差值在每个 seed 中都为正。它们知道块边界，却无法读取产生真实邻块影响所需的信息。相反，等参数的 `oracle_global4` 降到 0.07708，三个 seed 的 `global4 − local4` h10 差分别是 `−0.06995、−0.06639、−0.07081`，每个 seed 都明显有利于 global4。它也在每个 seed 上优于 dense。

这正是预想中的机制：rank-4 系统的跨块作用可以由少量全局信号概括；只把坐标分对、却不让块交换相关信息，仍然会漏掉真实 dynamics。这里最干净的证据是等参数、同初始化的 global4/local4，而不是参数不同的 global4/block。

`oracle_global16` 只在 lowrank 额外训练三个 seeds。其 h10 为 `0.07715、0.07644、0.07750`，与 global4 的 `0.07654、0.07726、0.07746` 几乎相同，逐 seed 差异方向也不一致。当前试验没有显示 16D message 比 4D 带来可见的 h10 收益；但 global16 比 global4 多 3,840 个可训练参数，且没有 local16 等参对照，不能据此声称两者严格等价或单独识别 message 维数的因果作用。比较稳妥的说法是：**这个低秩系统里，已知正确坐标后 4D message 已达到 global16 的观察水平；第一轮 global16 优于 learned global4 的差距，至少有一部分可能来自坐标学习难度或额外容量。**

**Dense-coupled：正确坐标和全局摘要都帮忙，但 4D 摘要没有追上 dense。** `oracle_global4` 的 h10 从 0.19995 降到 0.14390，明显好于 `oracle_block` 和 `oracle_local4` 的约 0.179；等参数 global4/local4 的逐 seed差值均约 `−0.034` 至 `−0.036`，方向一致。跨块信息确实有用。

不过 dense reference 的 h10 是 0.13121。global4 对 dense 的三个 seed 配对均值差分别为 `+0.01139、+0.00930、+0.01738`，三个 episode-bootstrap CI 都高于 0，说明它在每个 seed 的 held-out episode 上仍比 dense 差。报告里 global4 的 quality gate 是 3/3 pass，是因为门槛允许 `dense + max(0.1*dense, 0.02)`，而这里的 0.02 宽限大于 10% 相对误差界限；**通过这条宽松门槛不等于误差已与 dense 相同或优于 dense。**

这个系统的真实耦合更广，而且随 action 改变。四维全局摘要虽能补上部分交互线索，但它仍是一个窄的信息通道，局部 predictor 无法从中恢复所有复杂的 action-conditioned cross-block effect。结果与这种容量限制相符；固定 1500-step 预算下的优化因素仍未被完全排除，不能把它写成已证明的理论上限。

## 复现与速度证据

PBS job `25564754.pbs101` 正常结束，`EXIT_STATUS=0`，共完成 **27 个主 runs + 3 个 global16 runs**。四项机制检查全部通过：正确坐标可逆、`Q` 固定、12 组对应 predictor/message 初始化逐张量相同、独立系统与 block predictor 的跨块 Jacobian 为 0、global message 能读四块而 local message 只读本块。

同一 compute allocation 重新加载第一轮对应的 learned/dense/global16 checkpoints，并复算了全部 39 份 held-out horizon error arrays。每份都与原结果一致，最大绝对误差为 **0**。这确认 oracle 比较使用了同一批 frozen data 和同一归一化量。

质量提升没有转换成速度优势。沿用第一轮“完整 rollout 至少快 20%”门槛，B1 与 B300 的 speed gate 全部 **0/30**。按同一 allocation 的 dense timings，oracle 的 B1 reduction 为 −82% 至 −114%，B300 为 −66% 至 −123%；负值表示比 dense 慢。这里测到的是当前 PyTorch 实现的完整 rollout（含初始变换和最终还原）；没有 profiling kernel，也没有与六层 ViT 或 LeWM predictor 对比。因此它不能回答相对视觉 baseline 能否更快。

## 回到最初的猜想

第一轮想法是：找到合适表示后，大部分 action-conditioned dynamics 可在块内运行，只需少量跨块信息。oracle 结果把这个想法拆成了两段：

- **找对表示很重要。** 即便 predictor/message 初始化相同，把可学习 `Q` 换成已知正确 `M`，所有系统和结构都明显改善；独立系统尤其显示局部 predictor 本身可以很好地工作。
- **通信量需要匹配 dynamics。** 独立系统不需要全局摘要；rank-4 系统中 4D 全局摘要相对等参数 local4 有大幅优势，而且已达到 global16 的观察精度；更广的 action-conditioned 耦合中，4D 摘要能补救一部分误差，却仍落后 dense。
- **预测结构不等于加速结构。** 所有 oracle 变体都未过速度门槛。局部块在当前实现中没有变成更短的实际 rollout。

所以这轮支持继续问“怎样自动找到可预测的块，并识别需要交换的最小信息”，但它还没有给出一种适用于真实视觉 latent 的分块方法。并行的 C-SWM visual pilot 会单独检验视觉 object slots 下的机制；本 toy oracle 结果不构成 LeWM 替代、ViT 加速、CEM 排序或 closed-loop 控制证据。

### 数值与解释边界

- h10、h20 是以 train-only delta energy 归一化的末步 rollout MSE；表中均值跨三个训练 seeds。每个 seed 的 CI 是在该 seed 的 held-out episodes 上 bootstrap，不能视为训练随机性的总体置信区间。
- global4/local4 的直接 h10 点差来自同 seed、同 test episodes 的汇总均值相减；runner 没有为这组新对照保存单独的 paired-bootstrap CI，因此上面没有给它添加 CI 或显著性结论。
- global16 只在 lowrank 条件下训练；没有 independent 或 dense-coupled global16 结果。
- 本报告新增的是易读解释和机制归因边界。原始数值、门槛、checkpoint、episode arrays 均未修改。
