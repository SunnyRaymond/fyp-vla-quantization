# Comp8 更新稳定性：现有证据、R/VQ 分解与改进顺序

分析日期：2026-10-23（Asia/Singapore）。仅复查已有小型训练记录、聚合结果、源码和数值 receipts；新增数字是已保存标量的差值/比例。新增训练、模型 forward、teacher query、PBS 提交均为 0。没有修改训练实现或已有结果。

## 判断

Comp8 的 4/8 次同 batch mixed 下降，支持当前更新配方存在 hard-loss 稳定性问题，但不能单凭这个比例归因为 R，也不能推断补偿失败。Seq8 冻结 R 后同样只有 4/8；原 Joint8 为 7/8，却具有最差的最终 heldout mixed。下降次数不衡量下降幅度、跨 batch 保持性或泛化。

目前最有力的判断是：

1. **VQ-only 也存在不稳定，Comp8 另有 R 与 VQ 相互影响的嫌疑。** 后者需要同一起点的反事实端点拆分，现有第 2–8 步跨臂比较不能识别纯 R 作用。
2. **Comp8 不只是 heldout 泛化问题。** calibration mixed 比 Seq8 高 2.256%，latent 高 4.710%；同 batch 第 3、7 步两项目标都恶化。
3. **最终与 Seq8 的差距主要表现为 latent 适配不足。** heldout mixed 差值的 99.426% 来自加权 latent 项，这是 loss 的算术分解，不是参数/机制的因果归因。
4. **原 Joint8 的 RJ 有 activation 几何泛化问题的证据；Comp8 的 RC 尚未被独立运输测试。** 不应将前者结论直接当成后者的已验证机制。

## 一、逐步拆分真实 hard loss

实际目标为 `L = A + lambda Z`，其中 A 是 first10×6 motor MSE，Z 是排除 frame0 的 future latent NMSE，lambda=0.04245452683169643。以下全部为同一 batch、同一 teacher、同一 sampler seed 的 after−before，负数表示改善。不是 mean RMSE 的平方。

| Update | ΔA ×10⁴ | lambda ΔZ ×10⁴ | ΔL ×10⁴ | Comp8 ΔL / before | Seq8 ΔL / before | 原 Joint8 ΔL / before |
|---|---:|---:|---:|---:|---:|---:|
| 1 | +3.994 | −2.650 | +1.344 | +2.447% | +0.026% | −3.606% |
| 2 | +0.036 | −3.300 | −3.264 | −6.720% | −9.892% | −11.864% |
| 3 | +3.135 | +14.535 | +17.670 | +43.484% | −6.269% | −10.021% |
| 4 | −1.677 | +2.074 | +0.397 | +0.982% | +11.808% | −11.583% |
| 5 | −3.478 | −2.460 | −5.937 | −11.250% | +10.378% | −22.100% |
| 6 | −1.526 | −10.909 | −12.434 | −22.183% | −19.365% | −20.178% |
| 7 | +2.484 | +3.493 | +5.978 | +14.459% | −2.863% | +7.185% |
| 8 | −1.828 | +0.086 | −1.741 | −4.595% | +12.244% | −7.594% |

第 1、4 步属于实际输出上的任务取舍；第 3、7 步则是 A、Z 同时恶化，不能仅用“latent 权重过大而牺牲 action”解释。第 3 步 loss 增量约 82.26% 来自 weighted latent，但第 6 步的大幅收益也主要来自 weighted latent；这是响应波动，不足以证明 latent 梯度始终支配训练。

Seq8 的第 4、5、8 步也同时恶化 A、Z。Comp8 第 5、8 步反而改善，说明它不是逐步全面劣于冻结 R；仅数 4/8 隐藏了不同轨迹和不同失败 batch。Seq8 首步 +0.026% 几乎持平，也不应与 Comp8 第 3 步 +43.484% 赋予相同严重程度。

8 个 batch 不同且参数连续演化，不能把 before/after 跨 batch 串成固定目标的收敛曲线，也不能把 8 步当成独立二项试验。各步差值相加不等于全 C80 的训练前后收益。

训练总体仍有效：

| 端点 | Calibration mixed | Heldout mixed |
|---|---:|---:|
| R0+VQ0，共同初始模型 | 0.004963857 | 0.004880400 |
| Seq8 | 0.003954003 | 0.004175274 |
| 原 Joint8 | 0.003889337 | 0.004682388 |
| Comp8 | 0.004043222 | 0.004436499 |

Comp8 相对共同初始模型的 calibration/heldout mixed 分别降低 18.547%/9.096%。应称为“有收益但每步不稳定，且不如 Seq8”，不是“Comp8 没学到东西”。

## 二、现有数据能拆到什么程度

### 第一更新：最接近可用的配对分解

三臂共同起点、首 batch、VQ LR 和 fresh Adam 相同。在初始 R=R0 时，相对补偿是精确 identity，源码预期 VQ 梯度路径不变；已保存 case2000 preflight 的完整 1228 组 VQ gradient/Adam delta 差值为 0，首训练 batch 的 VQ norm 也相同。

但 case2000 preflight 不是首训练 batch 的 10-context gradient；没有保存该训练步完整 delta 的逐元素跨臂比较。以下将 Seq8 首步用作有强结构支持的 VQ-only 参照，而不把 norm 相等升级为所有训练梯度的独立实测证明。

| 首步端点 | A | Z | Mixed |
|---|---:|---:|---:|
| 共同更新前 | 0.002275013 | 0.075814973 | 0.005493702 |
| Seq8：只更新 VQ | 0.002153237 | 0.078717442 | 0.005495148 |
| Comp8：VQ+补偿 R | 0.002674412 | 0.069574004 | 0.005628144 |

Comp8 after−Seq8 after：A 增加 0.000521176，weighted Z 降低 0.000388180，mixed 增加 0.000132995。支持“加入补偿 R 更新后，在更新后的 VQ 上出现更强的 action/latent 取舍，mixed 额外恶化”。

这里仍不是纯 R-only 结果：`L(R1,theta1)−L(R0,theta1)` 包含 R 在已更新 VQ 上的条件效应；缺少 `L(R1,theta0)`，不能把它拆成独立 R 效应与交互项。

### 第 2–8 步：跨臂状态已经不同

Seq8 的 VQ-only 是沿 Seq8 自己轨迹的更新，不是“Comp8 当前状态冻结 R 后会怎样”。Comp8 与 Seq8 的 R、C/g、Adam moments 和输入到各层的 activation 分布都已分化。故第 3 步 Seq8 下降而 Comp8 上升只定位了可疑 batch/状态，不能相减得到 R 的净贡献。

源码只原子覆盖一个 `arms/Comp8/state.pt`，history 保存 scalar metrics/norms，未按 update 归档全部参数、delta 和 optimizer state。现有本地记录无法恢复每步 2×2 反事实表。最终 state/optimizer 也不能倒推出所有早期状态；若未来需要 update3，先检查是否另有明确备份，否则只能从共同起点有界重放前两步，不应声称无成本还原。

记录中的 `rotation_gradients.action/video=true` 是两个 expert stream 的 R 参数有梯度，**不是** action loss 和 latent loss 的分量梯度。没有 `g_A`、`g_Z`、cosine 或逐层贡献，不能据此判断任务梯度冲突/latent 梯度主导。

## 三、最值得怀疑的机制

### 1. Hard forward 与 surrogate backward 不一致：R、VQ 都受影响

实际 A4 为 `Q(z)=s(z) round(z/s(z))`，`s=max|z|/7`；backward 直接返回 identity STE。它没有跟随真实 round 阈值或动态 absmax 的局部导数。忽略最终 BF16、固定整数码 k 与 absmax 赢家时，真实局部 `dQ=k ds`，一般不等于 `dz`；跨阈值又出现跳变。

VQ 也并非完全连续前向：FP32 master codebooks/log gains 转为 BF16 存储后 decode；decoded weight 再转 BF16。其 backward 使用 rounded values 配合连续 surrogate。Comp8 还把 decoded BF16 weight 在 FP32 中运输后再 round 到 BF16。小 master 更新可能暂时不改变存储值，越过阈值又使共享 codeword 的多个使用位置一起变化。

因此 R 冻结时仍可能不稳定，符合 Seq8 的实际记录。固定 indices/exponents 是适配能力约束，但本次没有更新它们，不能把不稳定解释为训练中发生了 VQ index 重分配。

STE 文献仅在特定模型和 estimator 条件下给出 coarse gradient 的下降性质；不能推出这里每个 hard batch 都应下降。见 [Yin et al., ICLR 2019](https://arxiv.org/abs/1903.05662)。这提供机制背景，不是本实验的因果验证。

### 2. 补偿使 R 信号变小，Adam 不会因此自动减小步幅

固定 VQ、关闭量化的 FP32 路径，补偿后 R 两支导数抵消。理想正交切向和 identity STE 下，R 的局部方向与 activation 量化残差相关；旧 Joint 则还包含直接改变浮点函数的方向。补偿修正了优化对象，未保证更容易降低 calibration loss。等价旋转梯度的背景见 [SpinQuant Appendix B.1](https://arxiv.org/html/2405.16406v4#A2.SS1)。

首 batch R norm 从旧 Joint 的 0.0466505 变为 Comp8 的 0.00601738，但 fresh Adam 首步近似 `delta p = −lr*g/(|g|+eps)`，不能据此推断旋转小了约 7.75 倍。当前没保存实际 `||Unew−Uold||`、相对角度、A4 翻码率或 absmax 赢家变化。

第一步 fresh Adam 在非零 surrogate gradient 上的 `g·delta p` 应为负；第一步 Comp8 hard loss 却上升。这说明无需历史 momentum 或 resume 错误，也能出现 surrogate 与有限 hard 更新收益不一致。它仍不能区分 proxy 方向失真与有限步长过大，两者要靠缩步曲线进一步区分。

### 3. R 与 VQ 同步改变，交互与共享参数使适配目标移动

R 只有 action/video 两个共享矩阵，跨多层反复使用；一个旋转更新同时改变许多 A4 网格以及后续采样轨迹。C/g 同时更新，也会改变后续层 activation 分布。同步 Adam 使用旧状态计算的两组方向，最终一起落到新状态；某组单独有利，合起来未必有利。

该机制合理，但未被独立测量。不能仅凭 Comp8 第 3 步回退就断言联合训练存在负交互。

当前补偿允许 C/g 保留在 R0 anchor 坐标，修复了“VQ 必须自行追赶坐标”的问题；它并未改成随 R 重编码 `Q_VQ(W_s R)`。R-only 主要优化该压缩模型下的 activation 量化几何，并带有 BF16 transport 数值效应，不能期待它同时自动改善原生 weight VQ 的几何。

### 4. Optimizer reset 是可疑配方，尚非已证实主因

update1 后只清 VQ Adam state，R moments 保留。第 2 步因此使用不同历史长度的两组 optimizer；一共仅 8 步，短暂状态差异可能重要。可是 Seq8 也清 VQ state，第一步问题又先于 reset，所以 reset 不能解释全部现象。

应优先保存实际 group step、分量 gradient 与 `g·delta`，再单独比较 reset 规则。不要把“取消 reset、改 LR、加正则、换 loss”一次全部改变。

### 已有证据不支持的解释

- 不是 before/after 使用不同训练 batch：代码都取同一 10 contexts/seed0，模型设为 eval，teacher 绑定一致。
- 不是第 3 步由改单卡续训造成：resume 从已完成第 3 步开始，第 3 步 +43.484% 已发生在取消双卡之前；恢复了 Adam/RNG，未重置科学配置。
- 不支持 gradient clipping 触发导致本轮失败：Comp8 VQ/R 最大 pre-clip norm 为 0.026427/0.013722，阈值 1；Seq8 也未触发。
- 未见 NaN、整组断连或正交性崩坏证据；现有 parity/重复性通过。但这些 checks 不证明 surrogate 方向质量，也不证明所有 contexts 的 BF16 transport 残差可忽略。

## 四、适配与泛化分别看

Comp8 相对 Seq8：calibration mixed +2.256%、latent +4.710%；heldout mixed +6.256%、latent +11.902%。因此既有训练分布上的拟合/优化差距，也有更大的独立 contexts/seed 上的差距；不能只说 overfitting，也不能仅凭 8 步就下结论需要更多 epoch。

Heldout mixed 增量为 0.000261225400；其中 action MSE 增量 0.000001500428，weighted latent 增量 0.000259725007，占 99.426%。这定位了最终性能缺口，但不说明应提高或降低 lambda。第 3、7 步对任意正权重的两项求和都恶化；重新训练时权重会改变方向，因此也不能由这个事实推断调权永远无效。

Suite 层面 Comp8 vs Seq8：object 在 calibration/heldout mixed 均变差；libero_10 的 calibration mixed 接近 Seq8（+0.412%），heldout 却 +24.282%；spatial 的 calibration mixed 变差而 heldout 改善；goal 在 calibration 改善而 heldout 近乎持平。不是所有 suites 都符合统一的“训练更好、测试更差”模板。

之前的运输探针确实发现旧 RJ 在固定 decoded anchor 下 calibration 倾向更好、heldout A4 更差。但其对象是原 Joint8 的 RJ，尚无 Comp8 的 `RC→R0` 或 `Seq weight→RC` 端点，不能归因 Comp8 到同一个具体旋转机制。

Heldout 为 40 个独立 paired contexts、各 2 seeds，来自相同 tasks 的不同 demonstrations。当前训练阶段只用每个 context 的 seed0；缺少同一训练 context 上 seed1 的对照，不能分别量化 context 泛化与 seed 泛化，也不能称为未见 task 泛化。

R0 及 VQ0 已经使用 C80；此时再从 C80 切一部分充当 validation，不是对整个 pipeline 完全未见的数据。若未来调 LR/早停/频率，应使用未进入 R0 学习和 VQ fitting 的独立 calibration-validation，并保留既有 heldout 作为描述性结果；反复查看过的 heldout 不应再用于选超参。

## 五、最小而有区分度的后续诊断（仅方案，未运行）

### A. 先做最终端点 2×2 交换：无需训练

Seq8 与 Comp8 的 C/g 均定义在 R0 anchor；保持统一的补偿执行语义，定义 `F(R,theta)` 为同一 hard BF16/A4 路径的 loss。

|  | Seq8 的 thetaS | Comp8 的 thetaC |
|---|---|---|
| R0 | L00：已有 Seq8 | L01：缺失，Comp8 的 C/g 换回 R0 |
| RC | L10：缺失，Seq8 的 C/g 配对运输到 RC | L11：已有 Comp8 |

`Delta_R=L10−L00`，`Delta_V=L01−L00`，`I=L11−L10−L01+L00`，且 `L11−L00=Delta_R+Delta_V+I`。

- 若 L01≈L00，而 L10 明显变差：优先怀疑 RC 的几何效果。
- 若 L01 也明显差：Comp8 的 VQ 适配路径本身留下了缺口。
- 若 L10、L01 各自好，L11 差：支持负面交互。
- 若 calibration 好、heldout 差：再定位对应因子的泛化。

它拆分的是最终参数状态的条件作用，不是 8 步训练的唯一因果贡献。I 是依赖 anchor/参数取值的有限差分交互，不能把它包装为普适百分比。

完整复用原 calibration80 + heldout80 queries，两个缺失端点共 320 次 student query；已有两个角和 teacher 全复用。每个 query 是完整 sampler，不能称为 320 次单 Linear forward。若要将 R 作用进一步归为 A4 而非 BF16 transport，需另列配对 noA4 控制及成本；不把它悄悄算进现有 320。

### B. 再针对 update1/update3 做局部更新 2×2

从同一个 pre-update 模型、同一个 Adam state、同一 batch/seed，计算一次真实 proposed deltas，再分别评估：不更新、只应用 deltaR、只应用 deltaV、同时应用两者。R-only 始终配对补偿 weight；VQ-only 固定该步 R。对所有角分别保存 A、Z、mixed 和 per-context 数值。

使用同一 proposed update 可以分离“本步两个参数块”的有限作用。不要先训练一个 R-only 模型和一个 VQ-only 模型，再把不同轨迹相减。不要为每个角重置 Adam 后重算不同 update。每次切换角都刷新解码 weight 和 matrix/relative/dense cache。

优先 update1：共同起点可用，已保存 preflight tensor 可供检查，但它是 case2000 单-context probe，不能冒充首个 10-context update。要复查 update3 的历史问题，若无独立旧 checkpoint，需重放前两步并核对 scalar replay；不需重跑完整 Uncomp8/Joint8/Seq8。

再对有害 block 的实际 delta 做少量 `alpha=1,1/2,1/4,0` 缩步，alpha 是诊断网格，不是已验证最佳 LR。局部 R raw 参数插值后重新生成 Cayley matrix，保持正交；不直接线性插值两个正交矩阵。

- 缩步后变好：支持有限步长/阈值跨越过大。
- 多尺度均坏、surrogate 预测却改善：支持该状态下 proxy/量化敏感性问题；不能证明整个邻域不存在下降方向。
- 各单独更新有利、联合有害：交替并在另一组更新后重新求梯度。
- VQ-only 已明显有害：先处理 VQ LR/存储舍入/optimizer，而不是只减 R LR。

### C. 增补能判别机制的少量日志

保存两 block 的实际 update norm、R 矩阵相对变化、C/g BF16 存储改变比例、代表层 A4 翻码和 absmax 赢家变化；为 A、Z 分开记录 gradient norm/cosine，以及它们与实际 Adam delta 的内积。R 的 action/video stream 与 loss 的 action/latent 分量必须用不同字段命名。

这些指标只在指定诊断步骤采集，避免所有 614 层、所有 sampler steps 持续保存大 tensor。若出现大的 batch 均值波动，先看 per-context delta；现有均值无法说明是多数 contexts 恶化还是单个 outlier 主导。

## 六、改进优先级

1. 保留 Seq8 为 A+Z 主参考，继续使用相同 loss、rho1.5、D=I/r=1。先做上述最终端点分解和有界单步诊断，不追加完整 Uncomp8 重训。
2. 若 R-only 有害而 VQ-only 有益：优先限制实际 R 更新；比较更小 R 步/较低频率，以及“先固定 R 做 VQ，再解冻 R”的方案。R0 是较好的已学习起点，补偿后再次大幅移动不是必需条件。
3. 若联合交互有害：采用 block-coordinate 更新，另一 block 更新后重新计算梯度，并对 R 用 hard-loss candidate 检查。单纯把现有两个 optimizer.step 的调用顺序互换、却复用旧梯度，不算真正交替。额外 forward/backward、exposures 和 GPU 时间需单独记账。
4. 若 VQ-only 也有害：独立比较 VQ LR、reset 规则和实际 BF16 存储变化；小 LR 可能造成暂时零存储变化，因此不能只看 FP32 master delta。保留 master 累积，按 hard 端点评估。
5. 若两 block 在训练上稳定而独立 validation 仍退化：再处理 coverage、seed 多样性和对 R0 的 trust region/早停。每个候选规则有固定计算预算，拒绝更新时需要正确处理 optimizer state；训练 batch 上接受不等于泛化改善。
6. loss 权重、D/scaling、gripper/late22 保护和 native VQ 重编码分别作为后续轴，不与上述优化器改变一起引入。现有最终缺口落在 latent 上，但没有证据支持现在就把 rho 调大或调小。

当前实现仍是 decoded-BF16 transport reference；任意 R128 运输通常不封闭于原 d4/B32 编码。机制诊断有效并不自动建立同 bpw/native latency 的部署优势。

## 证据入口

- [Comp8 training](results/25921642.pbs101/training.json)、[Comp8 comparison](results/25921642.pbs101/comparison.json)
- [Seq8/Joint8 training](../fastwam-vq-r128-c80-dual/results/25912619.pbs101/training.json)、[初始与最终 comparison](../fastwam-vq-r128-c80-dual/results/25912619.pbs101/comparison.json)
- [Comp8 worker](worker.py)：train、persist、before/after、resume；[transport bank](transport_bank.py)：相对补偿与 BF16 cast
- [归档 rotation_bank](../fastwam-vq-r128-c80-dual/results/25912619.pbs101/evidence/executed_sources/rotation_bank.py)：A4 identity STE 与共享 Cayley R
- [归档 decoder](../fastwam-vq-r128-c80-dual/results/25912619.pbs101/evidence/executed_sources/c8_train_banks.py)：stored BF16 books/gains、decoder backward 与 refresh
- [unit receipt](results/25921642.pbs101/evidence/unit_receipt.json)、[resume receipt](results/25921642.pbs101/arms/Comp8/resume_receipt.json)

尚未新增数值探针，因此 R-only/VQ-only 的独立幅度、交互项、实际步长与泛化差距的具体机制仍待上述有界诊断；本文不把推测标成实测结论。
