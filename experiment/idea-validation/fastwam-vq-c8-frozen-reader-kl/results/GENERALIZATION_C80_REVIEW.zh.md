# Frozen-reader KL：泛化、C80 与目标设计的已有证据

2026-10-22。仅复核已有标量结果、训练记录、teacher 统计和实现；新增训练、模型 forward、teacher query、PBS 作业、闭环均为 0。没有改变已失败的 gate，没有启动 reader MSE ablation 或参数搜索。

判断：本轮比“完全没有学到 calibration”更符合 calibration 收益未迁移的模式，但不足以把原因锁定为 C8 数量。C80 有一次匹配验证的价值，理由是状态覆盖和 loss 标准化统计的代表性；它不是已经验证的修复。支持 frozen-reader 思路的理由仍在，支持这套 soft-bin KL 形状本身的实验证据很弱。

以下新增分组和敏感性分析均为事后描述，不替代原固定主指标，不用于选择 checkpoint、训练 case 或参数。正式主表仍是 calibration 8 cases / seed0，held-out 40 cases × 2 seeds，case 内平均 seeds、suite 内平均 case、四 suite 等权。每个配方只有一次训练，两个 sampler seeds 不是两次训练复现；历史 held-out 是已多次查看的开发/回归 cohort。

## 本轮结果与泛化解释

| 指标 | Calibration：相对原 A+Z | Held-out：相对原 A+Z |
|---|---:|---:|
| First10 motor RMSE | −8.11%；6/8 改善 | +1.93%；12/40 改善 |
| Future latent NMSE | −16.66%；6/8 改善 | −5.78%；22/40 改善 |
| First10 motor 配对中位差 | −0.00139310 | +0.00127491 |

Held-out motor 两个 seeds 的均值均退化，三个 suite 均值退化、四个 suite 配对中位差均退化。因此不是单个 seed 或单个 suite 的偶然反向。这里的“泛化问题”指相对 baseline 的收益在另一批输入上未迁移，不等于已证明模型收敛后的经典过拟合。

Calibration 改善有集中性：case2019 从 0.06814616 降至 0.04762474，占八例 motor 净改善总量的 75.05%。只作敏感性诊断，其他七例等 case 平均从 0.03841711 降至 0.03744225（−2.54%，5/7 改善）。它说明收益并非完全由一个 case 造成，也说明 −8.11% 不能理解为所有 calibration 状态都得到同等明显改善。七例分析不重新按四 suite 等权，不替代完整 C8。

用原 plan 的 suite/task_id 连接逐 case 结果，可进一步区分 task 覆盖和 task 内迁移：

| Held-out 子集 | Cases | 原 A+Z motor RMSE | Reader KL | 相对变化 | 改善 |
|---|---:|---:|---:|---:|---:|
| C8 覆盖过的 task，换到 held-out demo | 8 | 0.04358004 | 0.04632382 | +6.30% | 2/8 |
| 未被 C8 calibration 覆盖的 task | 32 | 0.04447596 | 0.04485644 | +0.86% | 10/32 |

两个子集各 suite 的 case 数分别为 2 和 8，等 case 与等 suite 平均一致。“未覆盖”只相对于本次 C8 calibration，不表示 BF16 预训练从未见过这些任务。这是一个冻结小样本的事后分组，不是 task novelty 的因果估计。它反对“只因另外 32 个 task 没进训练”这个简单解释；同一 task 内的不同 demonstration、状态和噪声组合也没有稳定迁移。

C8 phase 分布是 early 2 / middle 5 / late 1；C80 是 early 28 / middle 28 / late 24。Held-out 的 phase 子集 motor 变化为 early +1.85%（4/12 改善）、middle −0.45%（6/16）、late +5.33%（2/12）。这与状态覆盖不足相容，但任务、phase 和误差难度混杂，不能据此认定 late 是独立原因或从测试结果反选 calibration。

Held-out 的 paired motor/NMSE 四象限：

| | Motor 改善 | Motor 退化 |
|---|---:|---:|
| NMSE 改善 | 8 | 14 |
| NMSE 退化 | 4 | 14 |

因此 latent fidelity 的一部分收益确实迁移了；22 个 NMSE 改善 case 中，14 个 motor 却退化。单纯“所有目标都只记住 C8”不足以解释此结果。此表也不证明 reader KL 在这些 case 上下降，因为最终 reader 指标未测。

## C8 同时影响训练覆盖与 loss 的度量

本轮 reader KL 先用 C8 teacher 的 first10×6 action mean/std 标准化。已有 C80 teacher 统计也使用同一 first10×6、seed0、population std、std floor0.02 和相同原 calibration pool，可直接核对；没有使用 held-out 拟合。

| Motor 维度（1-based） | C8 std | C80 std | C8 的 1/std² 相对 C80 |
|---|---:|---:|---:|
| 1 | 0.181096 | 0.319654 | 3.12× |
| 2 | 0.374551 | 0.381475 | 1.04× |
| 3 | 0.559367 | 0.570567 | 1.04× |
| 4 | 0.056384 | 0.111658 | 3.92× |
| 5 | 0.121802 | 0.157967 | 1.68× |
| 6 | 0.152814 | 0.170049 | 1.24× |

这不是 model activation 的 per-channel scaling；是辅助 loss 已采用的 action 坐标标准化。第一、第四维在 C8 中的变化范围明显小于 C80；第 2 维的 mean 也从 C8 0.41896 变成 C80 0.14618，说明坐标中心同样依赖样本覆盖。

在 soft-bin 的中央、小误差区域，KL 近似为标准化 action 平方误差。由 std 单独决定的最大/最小坐标权重比，C8 约 98.42，C80 约 26.11。该比值不是 VQ 参数梯度范数比，也不是物理控制代价比；一个统一 beta 不能消除轴之间的相对偏重。C80 统计能改变这部分偏重，但 C80 的经验方差仍不等于各维控制容忍度或真实预测不确定性。

这提供了比“8 太少”更具体的机制假设：小 calibration 不仅影响学到哪些状态，也影响用什么坐标权重定义“该修什么误差”。如果同时更换训练 pool 和统计，只能评价完整 C80 recipe，不能把改善全部归给训练样本数量。

## 还支持什么、反对什么

Frozen reader 的理论动机仍成立。令 e=z_Q−z_T，给定 observation 与配对噪声，辅助项在光滑局部近似下具有 eᵀJ_RᵀWJ_Re 的形式。J_R 描述 latent 误差如何变成 reader action 误差，W 是输出分布/标准化诱导的权重；它包含 NMSE 的各向同性距离没有表达的方向、跨 latent 坐标耦合和 observation 依赖。但当前 BF16/A4 backward 含 surrogate，局部几何不能当作实际有限步收益保证。

本轮 614 个 Linears 的梯度连接、数值有限、staged replay、初始化和推理 parity 已通过，首步确实改变部署输出。Calibration motor 改善和 held-out NMSE 改善说明配方有实际影响。它们支持继续保留 reader-aware 目标这个假设，但不证明新增 KL 单独贡献了哪一部分收益，也不证明最终 reader loss 已充分下降。

Soft-bin KL 的独立价值缺少支持，而且有具体反证线索：

- 已存的 8 个初始化 reader 记录中，KL / standardized-reader-MSE 为 3.0970–3.1056；8 个更新前记录中为 3.0971–3.1186（与初始化重复的首例为 3.0975）。这些是不同 case/checkpoint 的 loss 值，不是固定训练集收敛曲线。
- 已有同一 bins/sigma 的数值诊断，中央区域 Fisher information 约 6.194，其二阶系数为 3.097，与上述比例相符。远尾部的 Fisher 权重会下降；当前记录没有给出这轮 held-out 的尾部占比，不能声称 saturation 已造成失败。
- 因此在已观察区域，它主要表现为经过 reader 和 action 标准化的平方误差。loss 比例相近不能证明参数梯度、Adam trajectory 完全等价，更不代替未运行的 MSE ablation，但削弱了“特殊 KL 分布结构提供了额外收益”的说法。
- 这是逐位置标量 soft-bin 分布，不是实际随机 action policy 的 density，不建模联合轨迹概率、跨动作维的概率相关性或真实不确定性。跨 latent 的耦合主要来自 reader。

还存在 frozen BF16 reader 与实际量化 reader 的迁移缺口。在同一 observation/noise 条件下，精确分解为：

    R_Q(z_Q) − R_T(z_T)
      = [R_Q(z_Q) − R_T(z_Q)] + [R_T(z_Q) − R_T(z_T)]

Reader KL 直接约束右边第二项的分布距离，实际 motor 指标取决于两项相加，包括误差抵消或放大。即使第二项改善，也不能保证和更小；换 C80 不会自动消除这个问题。本轮没有 baseline/candidate 最终 held-out 的完整 frozen-reader 对照，因此不能区分“辅助目标本身没泛化”和“辅助目标改善但没有迁移到量化 reader”。原 baseline 未保存所需的完整 future 缓存，不能宣称仅靠现有标量已完成这项分解。

优化仍未排除：只有 batch1 的 8 次更新，各 case 一次，首次更新后 reset Adam。当前辅助项 loss 数值占原 objective 的 0.70%–17.66%，总梯度未触发 clip；这既不证明权重安全，也不证明权重过大。此前 affinity-KL 的有限步/BF16 部署死区诊断适用于提示这条量化训练路径的风险，不能移用其梯度比例来判定当前 reader KL。没有本轮的分量参数梯度和固定 cohort 训练曲线，不能认定充分优化、发生收敛后的过拟合，或指定一个更好的 beta/LR。

## 已有 C80 结果如何影响决定

| 历史证据 | 观察 | 能支持的解释与限制 |
|---|---|---|
| Clip-only C8→C80 | 相对同一 C8 A+Z baseline，motor 从 +2.57% 退化转为 −1.01% 改善；NMSE 从 +12.88% 转为 −9.90% | 同项目确有扩大 calibration recipe 后收益转移的先例；但 8→80 updates、更新顺序也改变，训练的是 clipping ratios，不是本轮 books/gains |
| 原始 VQ C8→C80 | first10 motor +0.505%，full32 −2.945%，first10 21/40 改善 | C80 不是主指标必然变好的按钮；该实验 action_lambda=0，与当前 reader 辅助不同 |
| C80 soft-bin dual-KL | 相对该分支 MSE+NMSE，held-out motor 约 +2.18%；latent-KL 单替换约 +1.82% | C80 本身不能确保 KL 收益；该分支训练 activation ratios/channel scales，且 KL 定义不同，不能直接否定本轮 frozen-reader 设计 |
| C80 joint 与 clip-only | joint fixed-input motor/NMSE 更低，既有 Plus 闭环为 202/280 对 207/280 | 即便修复 fixed-input 泛化，也不能直接推出闭环改善；它们不是本轮模型或闭环结果 |

这些历史分支仅作证据比较，本轮没有引入 activation clipping/per-channel 参数。尤其不能把历史 C80 clip/joint 的 endpoint 当成“同设置的 C80 A+Z baseline”。

## 是否值得换 C80

如果继续研究 reader-aware loss，我支持最多一次 C80 的匹配验证，优先级高于继续在 C8 调 beta、bins 或 sigma；理由是已有覆盖不足及统计坐标差异的具体证据，并且 80 份 teacher 已存在。它应该被定位为检验 C80 recipe 能否让 reader-aware 辅助收益迁移，不能预先称作修复。

有解释力的比较必须是 C80 A+Z 对 C80 A+Z+reader KL，共同初始化、同可训练参数、同 optimizer、同 update budget/order/seed、同最终 checkpoint 规则，仍然不引入 activation clip/per-channel。历史 C8 A+Z 继续作 incumbent；如果仅让 KL 在 C80/80 updates 训练、再对照旧 C8/8 updates，即使改善也分不清更大训练预算与辅助 loss 的贡献。若已有完全匹配的 C80 baseline 可以复用；已核对的上述 C80 分支不满足本轮匹配条件。

采用 C80 时应在训练前明确 action 坐标/统计与 beta 规则；C80-only teacher stats 可直接复用，不能按 held-out 表现挑标准化方式。若同时更改训练 contexts、步数和统计，结论限于整体 recipe，不能解释为单一数据量效应。未来至少应同时记录固定 endpoint 的 reader 误差与实际 student motor，使成功/失败更能定位机制；任何缺少缓存而需要的新 forward 均应计入后续预算。

当前选择仍是保留原 C8 A+Z、rho1.5 baseline，停止已失败的 C8 reader-KL 配方。C80 是后续新决策，不是原 gate 失败后的自动续跑。没有证据支持全面扩大投入、扫描参数或直接跑 280 闭环。

## 证据来源

- 本轮主结果：[aggregate/summary.json](aggregate/summary.json)；逐 case：[aggregate/paired_cases.json](aggregate/paired_cases.json)；训练与 C8 stats：[reader_kl/summary.json](reader_kl/summary.json)。
- Task/phase 映射：[原始 plan](../../fastwam-vq-calibration-suites/retrieved/data/plan.json)。
- C80 teacher action stats：[fit_stats/summary.json](../../fastwam-vq-dual-kl/results/fit_stats/summary.json)；同 bins/sigma 的 Fisher 诊断：[c80_diagnosis/summary.json](../../fastwam-vq-dual-kl/results/c80_diagnosis/summary.json)。
- C80 clip-only：[README](../../fastwam-vq-c80-activation-clipping/README.md)；原 VQ C8/C80：[配对报告](../../fastwam-vq-calibration-suites/retrieved/results/comparison/report.md)。
- C80 dual-KL：[分析](../../fastwam-vq-dual-kl/results/ANALYSIS.zh.md)与[参数范围](../../fastwam-vq-dual-kl/README.zh.md)；C80 joint：[既有分析](../../fastwam-vq-c80-joint-rmse/results/ANALYSIS.zh.md)。
- 旧 affinity-KL 优化证据：[机制结论](../../fastwam-vq-c8-latent-anchor-kl/results/MECHANISM_CONCLUSIONS.zh.md)，只用于识别共享训练路径的风险。
- KL 局部 Fisher 二次近似：[Strimmer, Local divergence](https://strimmerlab.github.io/publications/lecture-notes/MATH27720-stats2/05-entropy3.html)。将其通过 reader 拉回 latent 空间是对本实现的推导，不是该资料对本方法的性能背书。
- 闭环状态依赖的一般背景：[Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html)。它说明 imitation learning 的观测依赖历史动作；不证明本轮退化来自该机制。
