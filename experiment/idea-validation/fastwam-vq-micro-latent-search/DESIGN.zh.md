# Microscaling + action/latent VQ：λ 搜索实验方案

状态：历史设计建议。用户2026-10-14已批准实际配置：旧4-case selection，第一轮rho0.1/0.5/1，第二轮再三点，guards20%，不做互补判断。当前执行配置见[README.zh.md](README.zh.md)与[protocol.json](protocol.json)，以下原建议中的新增selection、5%guards、八点与互补实验不作为本轮执行要求。

## 1. 要回答的三个问题

1. 在已拟合好的 B32 Microscaling VQ 上，增加 future latent loss 是否比不训练的 micro parent、micro action-only 更好？
2. 固定这套表示、calibration 与训练预算，哪个 latent loss 强度在 selection 上最合适？
3. 最终强度的收益是否受到 microscaling 的影响，还是在普通 row-scale VQ 上也能得到相似收益？

“最佳”指给定候选区间、16 updates、选择规则下的最佳已测试强度，不表示全局最优，也不表示 closed-loop 最佳。

## 2. 已有证据与 parent 的选择

同一 22-case/44-context cohort 的已有结果：

| 方法 | First10 motor RMSE | Full32 motor RMSE |
|---|---:|---:|
| 原 Row-scale VQ | 0.05705206 | 0.05970920 |
| 原 B32 Microscaling VQ | **0.05240395** | **0.05726171** |
| 本轮 Row-output parent + action/latent | 0.05295101 | 0.06133165 |

Microscaling 单独已经是较强起点。两种方法的收益不能相加，也不能直接用旧 row-output action-only 作为 micro 的 λ=0 对照。两者初始化表示和拟合 recipe 不同。

本轮建议复用已完成 `25755896.pbs101` 的 `micro_scale_vq_a4`，bank 来源为：

`/scratch/users/ntu/yguo017/fastwam-vq-microscale-20261011/results/full/banks/micro_scale_vq_a4`

保持原 action-aware 数据源 `/scratch/users/ntu/yguo017/fastwam-vq-action-aware-20261009`、source job `25743693.pbs101`。Bank 来源与 observation/teacher 数据源必须拆成不同字段。

另一个对话最新 C8/C80 的测试是四个 suites、40 cases×2 seeds；calibration recipe 也不同。本轮先复用同 22-case cohort 的 micro banks，不混入 C80 calibration 或四-suite 测试。若最终目标改为四-suite，需另行冻结对应 calibration/selection/test 与 references。

来源：[Microscaling 结果](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-microscale/RESULTS.zh.md>)、[action/latent 结果](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-action-latent/INTERPRETATION.zh.md>)、[四-suite C8/C80 比较](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-calibration-suites/retrieved/results/comparison/report.md>)。

## 3. 先固定表示，只扫一个强度

保留 B32、d4、两本 K256 BF16 books、Smooth0.5/H128、A4 和 BF16 KV。固定 indices 与 uint8 block exponents，训练 books 和原 row gains；全部 614 个目标 Linears。

这保持与既有 distillation 相同的可训练参数类型。Micro 额外存储仍约为 0.25 bits/weight，已有含 transforms 的有效编码约 4.2709 bpw，相对 row-scale 的 4.0209 bpw；不是同存储预算对比，也不是 native INT4 execution。

不能仅对旧 row-VQ 解码结果后乘 B32 scale：已有 micro books/indices 是在 micro 归一化坐标与 scale² 加权 metric 下重拟合的，应直接复用这些已拟合 banks。

本轮不同时搜索 block size、learning rate、训练 epochs、可训练 block exponents、temporal loss 或 late22 loss。先使 λ 曲线可解释。

## 4. 搜索归一化强度 ρ，而不是照搬旧原始 λ

固定 loss：

`L = L_action(first10 × 6 motor MSE) + λ × L_future(global clean future latent NMSE)`。

Future 始终排除 observation-clamped frame 0，包含两个 future latent frames。Student 使用自己生成的 future；不向输入注入 teacher future。

用同一 micro 初始化在原 8 个 calibration contexts 上计算一次：

`A0 = mean_8cal(initial action MSE)`

`Z0 = mean_8cal(initial future NMSE)`

冻结 `c = A0 / Z0`，每个候选的 `λ = ρ × c`。这与把两项分别除以初始均值后相加，仅相差一个共同正标量；训练实现仍保留原 action loss 尺度和原 learning rate。

ρ=0.1 表示 micro 初始化 calibration 均值上，latent 项的数值是 action 项的 10%。它不等于梯度贡献的 10%。旧 `λ=0.0029918823` 对应 row-output parent，不是 micro 的预设最优系数；micro 的原始 λ 数字须等 A0/Z0 实测后才能给出。

各 λ 独立从同一 micro masters、相同空 Adam 状态与相同 RNG/数据顺序开始，不能从上一个 λ 的训练结果接着训练。c 不按每步 loss 动态重算。

## 5. 数据划分：保持旧 calibration/test，改善用于选择 λ 的 dev

固定 calibration：`[0,2,4,6,10,12,14,16]`，seed index 0；固定 test：`1004..1025`，seed indices 0/1，22 cases/44 contexts。仍是先平均 case 内两个 seeds，再对 cases 等权平均。

旧 selection `1000..1003` 只有 4 个 Original cases。它可以延续历史可比性，但不足以可靠选择适合 Plus camera/robot/layout/language/light/background 的强度。尤其 micro 已出现旧 selection 略差而完整 test 更好的情况。

**推荐版：** 在训练和观察新模型结果前，冻结额外的 22 个 held-out selection cases：10 Original、12 Plus（六类扰动各两个），每个两个 sampler seeds，共 44 case-seed contexts/queries 每 epoch。用尚未用于 calibration/旧 test 的 demonstrations、variants 或 snapshots；不能仅换旧 test 的采样 seed。先按数据元信息确认输入不重叠，不按 BF16 或量化误差挑选。

这些具体 IDs 尚未生成；生成后以独立 manifest 保存。旧 4-case selection 作为历史诊断，不参与最终 λ 排名。原 test 完全保留。

**低预算版：** 直接用旧 4-case selection 搜索，保留原协议；结论只能是“小 selection 上的候选最佳 λ”。不能用已经看到的 Plus test 来弥补 selection 覆盖或重新挑 λ。

新 selection 依然可能与旧数据共享 base tasks，属于 held-out 输入上的开发集；不能称为 unseen-task generalization。所有评测 counts 与 paired seeds 都完整记录。

## 6. λ 的粗搜与细搜

第一轮六个归一化强度：

| ρ | 意义 |
|---:|---|
| 0 | 从 micro parent 训练的 action-only 对照 |
| 0.01 | 初始 1% loss 比例 |
| 0.03 | 初始 3% |
| 0.1 | 对应本轮已验证的相对强度 |
| 0.3 | 初始 30% |
| 1 | 初始两项 loss 均值相当 |

额外保留未更新 micro parent（epoch 0）。它与“训练后的 ρ=0”是两个不同对照。

每个候选均完成 2 epochs/16 updates，使用相同学习率、clip、calibration 顺序、sampler seeds 和 Adam schedule；包括原 baseline 的 update1 后一次 reset，以保持现有比较口径。以后若要改 Adam schedule，应另立实验。

在 epoch 0/1/2 评估 selection；epoch 0 只跑一次并跨 λ 复用。不要因为 epoch1 action 分数差就淘汰候选：本轮已出现 epoch1 明显较差、epoch2 恢复的轨迹。

最多再补两个强度，提前规定规则：在粗搜最佳正 ρ 的相邻区间按 log-space 加密；例如 0.1 最好时补约 0.055 与 0.173。若 0 最好，补 0.003/0.006；若 1 最好，用一个额度试 3，另一个试约 0.55。仍落在边界时，只报告预算内的边界最优，不宣称区间外没有更优值。

总共至多八个 micro 训练候选，不做大规模 Bayesian/多维搜索。

## 7. 如何定义“最佳”

主目标仍是 selection 的 case-mean first10 motor RMSE，保持历史评价重点。

为避免用远期 action 或 gripper 的退化换取很小的近期改善，建议在运行前冻结以下工程 guardrails，参考同 selection 的未训练 micro parent：

- Late22 mean RMSE 不高于 parent 的 1.05 倍。
- First10 gripper mean RMSE 不高于 parent 的 1.05 倍。
- First10 motor 的 per-case P90 不高于 parent 的 1.05 倍；统一使用预先指定的同一种分位数计算方式。

5% 是拟议的工程容忍值，不是显著性阈值；应在看到新候选结果前固定，不能因为某个 λ 被筛掉再修改。

先排除不满足 guardrails 的 candidate/epoch，找到其中最小 first10 motor 分数 Jmin；把分数不高于 `1.01 × Jmin` 的点作为近似 tie 集合，优先选较小 ρ，同 ρ 再优先较早 epoch。这是预先固定的工程近似 tie 规则，不声称统计相等。

Future NMSE、motion NMSE、full32、Original/Plus 分组、improved cases、中位 paired delta 与尾部 cases 都报告，但不以最小 future NMSE 替代 action 排名。

Epoch0 必须允许胜出。若没有训练候选可靠优于它，部署候选仍保留 micro parent，结论是本次预算没有找到增益，不给“未发生训练变化”的 epoch0 强行附上最佳 λ。

这与无 guardrail、严格按 first10 排名的旧规则不同：应同时报告严格 first10 最优点与最终 guardrail 选择，以清楚说明取舍。

## 8. 分离 Microscaling 与 λ 的收益

主搜索只在 micro 上进行。选出 ρ* 后，在同一 selection 上完成以下局部 2×2 对照：

| 表示 | 训练 ρ=0 | 训练 ρ=ρ* |
|---|---|---|
| 普通 Row-scale VQ | 新对照 | 新对照 |
| B32 Microscaling VQ | 粗搜已含 | 粗搜/细搜已含 |

普通 row-scale 与 micro 来自同 calibration/fitting recipe 的表示比较；旧 row-output action-only 仅作历史参考，不能替代上述新对照。为这两条非 micro 对照使用主搜索冻结的同一个 c，使相同 ρ 对应相同 raw λ，避免另行归一化又改变一个变量。所有 arms 训练预算相同；性能表可使用相同选择规则分别选 checkpoint，guardrails 各自参照该表示的未训练 parent。

用于解释局部 interaction 的2×2应固定共同 epoch：采用主搜索选定的 micro 正强度 epoch e*，四个 cells 都取 epoch e*（相同 e*×8 updates）；不要把四个分别选过 epoch 的均值混成“固定训练量”的 interaction。此共同 epoch 同样在 test 前冻结。若最终胜出的是epoch0，说明尚未找到有收益的训练候选，没有必要强行做λ训练收益的interaction结论。

对某个 metric 的局部 interaction 可描述为：

`(E_micro,λ* − E_micro,0) − (E_row,λ* − E_row,0)`。

负值表示在该点上 latent loss 的降低误差效果在 micro 表示下更大。这里只检查已选强度附近，不代表找到了非 micro 的最佳 λ，也不证明整个 λ 区间的 interaction。

普通 row-scale 未训练的 epoch0也保留。报告 microscaling增加的存储代价。

## 9. 值得做的小型梯度诊断

先选两个预先固定的 calibration contexts，在 optimizer 更新前分别测 action、latent 对共享 video 参数的 gradients；保持所有正式候选的参数初值不变。

记录 `||g_action||`、`||c g_latent||`、夹角 cosine，以及拟议各 ρ 对应的梯度 norm ratio。分别看 books 与 row gains；对 action-only 参数的 latent gradient 为零可能是模型结构的正常结果，不能当作失联。

这可辨别“loss 数值只占10%，但 latent 梯度已压过 action”的情况。诊断不代替 selection 搜索，也不自动引入逐步动态 λ。首先完成可解释的固定 λ 曲线；GradNorm 等自适应方案属于后续另一种训练算法。

## 10. 最小代码适配与 references

现有 `train_banks.py` 只支持普通 row-scale decoder，不能直接加载 micro bank。必要改动限定为：

- 根据 bank format 支持 micro checker/decoder，保存不可训练 block exponents。
- Forward 与 custom backward 都纳入 `row_gain × 2**exponent`；snapshot/restore 保留固定 indices/exponents。
- 分开 data root 与 bank root。
- 将原 row-output parent 绑定的 references/loss-contract 改为 micro 初始化对应的 contract；A0/Z0 不能复用旧 parent 数值。

既有 BF16 teacher actions/futures 在输入、seed相同的旧 cohort上复用。Micro 缓存目前有旧 selection/test actions，没有 future，也没有 micro calibration actions。因此需要在 GPU allocation 中补 micro calibration action+future，复用一次生成的初始化 references；新 dev 的 BF16与micro references也各生成一次。Micro test futures及新训练候选 test 统一等 λ/epoch 冻结后再生成，不为每个 λ重跑 teacher。

开训门限只保留有意义的验证：小 tensor decoder/解析梯度 self-check（含 e=0、e≠0、尾块）；首个 checkpointed training forward 对真实 micro 初始化 action/future parity；真实 combined backward 的 finite/coverage与 future autograd 连接。这些只在真正 PBS compute allocation 上执行，不添加多余 hashes 或大型审计。

每个 λ 独立保存 optimizer/RNG/cursor，续跑不改变该 λ 或 cohort；共享 teacher cache为只读，不允许跨 λ 的可变 master 状态串用。

## 11. 最终 test 与解释

冻结 micro 的候选 λ/epoch、非 micro 对照 checkpoints 后，再跑原完整 22-case/44-context test。至少报告：micro epoch0、micro action-only、micro最佳 action+latent；机制对照版另外报告 row-scale action-only、row-scale action+latent。同一 checkpoint与context已有可靠缓存时可以复用，不降低分母。

Test 汇总 first10/full32/late22/gripper、future/temporal-delta NMSE、case 胜负和尾部敏感性。保留旧 row-output action/latent 的0.052951结果作为历史参考。即使最终 test 上另一候选更好，也不据此换 λ 或重新打开搜索。

最好展示 strength curve 而非只报一个最小点。若最优点附近形成较宽的稳定区间，比一个尖锐、由单个case决定的最小值更有说服力。

当前训练在固定 inputs/seeds 下可能基本确定性。若没有真实随机环节，单纯换 global seed重跑不构成有效重复。可选的稳定性检查是：最优两个 λ在相同两套 calibration 顺序上分别重训，仍用相同8inputs与sampler seeds。它检查顺序敏感性，额外预算与结果单独报告，不替换正式主协议。

## 12. 资源估计与执行边界

上轮实际 update约6分钟、decoded-BF16 query约1分钟；micro梯度版尚未实测，下面均是估计，启动pilot后校正。

| 范围 | 估计 GPU-hours |
|---|---:|
| 六个粗搜候选，用旧4-case selection | 约12–16 |
| 六个粗搜候选，用推荐22-case selection | 约20–24 |
| 推荐selection + 最多两点细搜 + 两个非micro训练对照 + 最终test | 约30–40 |

估计含一定初始化、references与评测余量，不含 queue wait、失败返工或可选训练顺序重复。这里的“最多两点细搜”也承担边界扩展额度。

有用的省算来自：复用已有 fitted banks、teacher cache、同 parent 的epoch0、跨PBS断点与已经完成的evaluation contexts；不来自减小既定test或用epoch1排名过早淘汰。

本方案没有启动监控。若后续执行，训练/模型/references/数值/图表全部放在真实 guarded PBS CPU/GPU allocations；login node仅轻量连接、qsub/qstat与小控制文件。GPU利用率/显存定期写job log。保留host-key verification，不输出credentials，不使用banked reset。

证据范围为已见test cohort上的 paired fixed-input recovery、decoded-BF16 VQ/fake A4、BF16 KV；不声称独立泛化、真实未来准确度、closed-loop success或native kernel加速。

## 设计方法参考

- Gradient balancing 的方法背景：Chen, Z., Badrinarayanan, V., Lee, C.-Y., & Rabinovich, A. (2018). [GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks](https://proceedings.mlr.press/v80/chen18a.html). 本方案只参考梯度尺度诊断，未实施GradNorm。
- 资源分配的背景：Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://www.jmlr.org/papers/v18/16-558.html). 本方案因已观察到跨epoch排序变化，暂不采用早停淘汰。
- 本方案使用 experimental-design skill 协助明确实验单位、factor对照与冻结规则：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). 当前arXiv记录于2026-10-14核对。
