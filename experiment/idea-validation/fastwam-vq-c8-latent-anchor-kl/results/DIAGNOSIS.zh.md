# C8 latent anchor KL：泛化、优化参数与指标诊断

2026-10-21。主要依据为本轮完整 8 calibration cases、40 heldout cases × 2 sampler seeds，以及保存的 C8 teacher / 未训练初始化 / KL 分支最终输出。原 MSE+NMSE 对照全部复用。本次补充诊断只在 PBS CPU allocation 中处理已有输出，新增模型 forward、训练和闭环 episodes 均为 0。

后续 2026-10-22 已完成新的参数梯度与冻结 reader 诊断，见 [MECHANISM_CONCLUSIONS.zh.md](MECHANISM_CONCLUSIONS.zh.md)。它将本报告中的输出偏导推测进一步限定：KL/原目标的平均参数梯度比仅约 10–12%；优化与部署落差的证据较明确，单纯 beta 太大或 KL 方向整体错误均缺乏支持。后续阶段有 224 个新逻辑 queries，不能与本报告的 0-forward CPU 补查混淆。

**结论：有泛化问题，但不能只归结为“训练拟合很好、测试过拟合”。** Calibration 的 motor 平均收益主要来自少数 case，新增 KL 的 calibration 均值从未训练初始化到最终反而上升。Heldout 上，同一个 calibration task 换 demonstration 也退步。现有证据更支持同时审视有限样本下的优化、输入泛化和 KL 所强调的 latent 方向；尚不能确定哪个因素是主因，也不能据此证明 latent loss 比 action loss 更偏离闭环目标。

所有差值为 candidate − baseline，负值表示改善。主汇总先平均同 case 两个 seeds，再平均 suite 内 case，最后四个 suite 等权。两个 sampler seeds 不是两个独立训练重复；本轮只有一个训练配置/顺序。以下事后分组和敏感性分析都是描述性诊断，不是独立验证或显著性检验。

**一、结果是否只是均值、个别 case 或指标汇总造成的？**

| 数据 | 指标 | 原 MSE+NMSE | 加 KL | 改善 / 变差 cases | 配对差中位数 |
|---|---|---:|---:|---:|---:|
| C8 calibration | First10 motor RMSE | 0.0421332 | 0.0415491 | 3 / 5 | +0.00235356 |
| C8 calibration | Future latent NMSE | 0.0544810 | 0.0495689 | 7 / 1 | −0.00687379 |
| Heldout 40 cases | First10 motor RMSE | 0.0442968 | 0.0458895 | 14 / 26 | +0.00113410 |
| Heldout 40 cases | Full32 motor RMSE | 0.0475010 | 0.0493055 | 17 / 23 | 见机器可读结果 |
| Heldout 40 cases | Late22 motor RMSE | 0.0485285 | 0.0503193 | 15 / 25 | 见机器可读结果 |
| Heldout 40 cases | Future latent NMSE | 0.0608774 | 0.0614710 | 18 / 22 | +0.00154110 |
| Heldout 40 cases | First10 gripper RMSE | 0.0570540 | 0.0426101 | 17 / 23 | +0.00039825 |

Calibration 的 motor 均值下降 1.39%，但 case 2019 单独从 0.0681462 降至 0.0481638，足以影响小样本平均。逐一移除任一 calibration case，保持 suite 等权后，motor 平均差范围为 −0.00397708 至 +0.00280877，会翻转方向。因此“训练集 action 已经普遍拟合得更好”不成立。Calibration NMSE 的改善更一致：7/8 改善，逐一移除 case 后平均差始终为负。

Heldout motor 的退步相对稳健：四个 suite 均值都退步；seed0 平均差 +0.00191913、seed1 +0.00126639；移除任一个 case 后，等权平均差始终为正，范围 +0.00125505 至 +0.00193096。这是 leave-one-case-out 敏感性范围，不是 confidence interval。

Heldout NMSE 只退步约 0.98%，但 seed0 改善 −0.00239721、seed1 退步 +0.00358440，移除单个 case 后均值方向也可能翻转。Case 3020 的 NMSE 从 0.0601587 升至 0.191324，是明显的尾部退步。因此对“motor 普遍变差”的判断比对“latent 总体稳定变差”的判断更有把握。

Gripper 平均 RMSE 下降约 25.3%，主要受 case 3031（0.325643 → 0.0218177）和 case 3000（0.325240 → 0.0263553）影响。其余 case 不是普遍改善：40 case 配对中位数反而略升。仅作敏感性诊断，移除这两个最大改善 case 后，剩余 suite 等权平均差为 +0.00069365。两个大改善是真实结果，不能删去主结果；也不能用它们代表所有 case 的 gripper 行为。

**二、是哪一层泛化？**

核对 task_id 后，C8 实际覆盖 8 个任务，每 suite 2 个。Heldout 共 40 个任务，每 suite 10 个；其中 8 个任务出现在 C8 中，但使用另一 demonstration，另 32 个任务没有进入这次 C8 calibration。全部是 original domain。

| Heldout 分组 | First10 motor：baseline → KL | Motor 改善 / 总数 | NMSE：baseline → KL | NMSE 改善 / 总数 |
|---|---:|---:|---:|---:|
| C8 出现过的任务，另一 demonstration | 0.0435800 → 0.0481207（约 +10.4%） | 1 / 8 | 0.0664144 → 0.0771295（约 +16.1%） | 1 / 8 |
| 未进入 C8 calibration 的任务 | 0.0444760 → 0.0453317（约 +1.9%） | 13 / 32 | 0.0594931 → 0.0575563（约 −3.3%） | 17 / 32 |

这不符合“主要是任务没覆盖到，所以新任务失败”的简单解释：已见任务换输入后也明显退步。它与对特定 observation / demonstration / noise 的依赖、跨样本优化干扰相容。分组没有随机分配、任务难度不同、已见组只有 8 cases，不能把两个百分比的差异解释为已证明的机制。这里的“未进入 C8”也不等于 foundation model 预训练未见过这些任务。

原协议中的“same tasks, distinct demonstrations”描述沿用了完整 C80 数据计划；对实际只训练 C8 的本轮，应使用以上更精确的 8 / 32 拆分。冻结主评测不变，也不事后更换 cohort。该 heldout 在过去的实验中已被查看，适合固定开发比较，不能再包装成全新独立泛化验证。

原 baseline 仍可保留作为当前比较基准。它已有 LIBERO-Plus 扰动闭环结果 206/280，BF16 为 216/280；camera_viewpoints 是 17/40，BF16 为 21/40，说明 baseline 本身也有鲁棒性余量。它相对既有 Micro 的 204/280 只多 2 个，paired wins/losses 为 15/13，且两者 calibration banks 不同，不能把差异归因于 latent loss。rho0.1 与 rho1.5 的旧比较也同时改变了 epoch（2 与 1），不能单独用来证明 rho1.5 最优。参见 [原闭环结果](../../fastwam-vq-c8-action-latent-plus/results/RESULTS.zh.md)。本次 KL 没有闭环结果，不能说它的闭环成功率已退步。

**三、参数有没有明显问题？**

当前目标为 L = MSE_action + 0.04245452683 × NMSE_future + 0.05137399516 × KL_anchor。这里 rho=1.5 还要乘冻结 normalization_c=0.02830301789，不能直接比较“1.5”与 beta 的大小。

| 设置或诊断 | 实际结果 | 能支持的判断 |
|---|---|---|
| Adam learning rate | 5e−4；batch 1；1 epoch / 8 updates | 与 baseline 公平匹配，但没有证明适合新增 KL 或已经收敛 |
| Optimizer reset | update1 后清一次 state，与 baseline 一致 | 是继承的优化细节；未单独消融，不能认定为错误或无影响 |
| Gradient clip | 阈值 1；实际总参数 gradient norm 0.00964–0.11941 | 8 步均未触发 clipping，不能把退步归因于被 clip 截断 |
| Temperature | 0.438990，C8 teacher 邻居距离中位数 | 合理初始化规则，尚不是经验证的最优值 |
| Teacher 概率分布 | 归一化 entropy 0.894；平均最大概率 0.271 | 比较软，但没有坍缩为单邻居；不能单凭 entropy 说 tau 错了 |
| Channel 标准差 | 0.570–1.114；floor=0.02，无 channel 触底 | 没有发现接近零方差导致的放大问题 |
| 连通与数值 | 8 步均 614/614 Linears 梯度 connected、finite、nonzero | 没有明显梯度断开或数值爆炸迹象 |

最值得重视的是 beta 的标定：训练前令“平均 beta KL / 平均 lambda NMSE = 10%”，这是 loss 数值比，不是梯度比。对保存的 latent 求偏导后，C8 初始化的平均 ||∇z(beta KL)|| / ||∇z(lambda NMSE)|| = **0.397**，各 case 0.277–0.583；最终平均为 0.447。因此这项附加 loss 对 latent 的推动并不只是 10% 的弱扰动。

两项 latent 梯度 cosine 平均约 0.262（初始）、0.269（最终），方向有明显差异但不是负相关。这只是在输出 latent 空间的偏导；没有计算 ∇θ KL、∇θ NMSE、∇θ action，不能据此声称参数梯度冲突或已经排除冲突。对当前固定 teacher-anchor 形式，KL 在 query 上为凸函数且 teacher query 是极小点，和指向 teacher 的平方误差梯度非负对齐本身也不意外；通过共享网络 Jacobian 后的参数方向关系仍可能不同。

另一个信号是 C8 平均 KL 从未训练初始化的 **0.00565312** 变为最终 **0.00622238**（约 +10.1%），4/8 降、4/8 升，中位数下降。我们没有证据说“KL 项已经被整体优化好”。但多目标总 loss 可以下降而单个 KL 上升，且只有一次 8-step pass，因此这一点也不自动证明 optimizer 有 bug。每步输入不同，日志中的 step loss 不能当作同一目标的收敛曲线。

这个 KL 初始值的参照是未训练 C8 banks，**不是已训练 MSE+NMSE baseline**。原对照缓存没有完整 trained future tensors，无法无新 replay 地重算其 anchor KL；本次没有补跑 baseline。

Temperature 与 beta 要联合理解。经典 distillation 在改变 temperature 时会考虑梯度尺度；本轮 tau 固定，乘一个统一 tau² 可以被 beta 吸收，因此没有额外乘 tau² 不能直接叫作实现 bug。[Hinton et al., 2015](https://arxiv.org/pdf/1503.02531)。空间边与时间边距离中位数的 C8 平均分别为 0.469 与 0.229，当前一个 tau 混合了两种尺度，这是设计上的待验证点，尚不是已证实的失败原因。

**四、KL 到底约束了 latent 的什么？**

本实现不是将有正有负的 latent channel 直接当作分类 logits。它先用 C8 teacher 固定统计量标准化，并在每个 camera 内做 2×2 pooling。对于 48 维 query s_i，以固定 teacher 邻居 t_j 构造：

q_i(j) = softmax_j(−||s_i−t_j||² / (48 tau))，p_i(j) 使用 teacher query t_i；loss 为平均 KL(p_i || q_i)。每个 token 有 4、6 或 9 个邻居，来自同视角空间相邻位置及另一 future time 的相同位置。这是有效的条件邻居分布 KL，利用了时空结构；它不是实际 future density 的 KL，也不是 mutual information 的估计。

关键性质来自本实现的代数推导：

log(q_i(j)/q_i(r)) = [2 s_i·(t_j−t_r) − ||t_j||² + ||t_r||²] / (48 tau)。

所以 q_i 只依赖 query 在 teacher-anchor 差向量张成空间中的投影。若 δ 与所有 (t_j−t_r) 正交，q_i(s_i+δ)=q_i(s_i) 完全不变。局部 anchor span 的 rank 实测为 3–8（平均 6.37），而 query 是 48 维。

在 C8 的标准化误差坐标下，逐 case 计算误差能量比例再取平均：

| 附加 KL 的不可见误差 | 未训练初始化 | KL 分支最终 |
|---|---:|---:|
| 2×2 pooling 消去的块内 residual 能量 / 原始能量 | 46.80% | 48.38% |
| pooling 后、位于 anchor span 正交补的能量 / pooling 后能量 | 69.96% | 68.09% |
| 合并上述两步，完整 KL 不可见能量 / 原始标准化能量 | 84.21% | 83.78% |

前两行分母不同，不能相加；第三行直接按逐 case 正交分解计算。数值检验中，将 query error 的不可见部分去掉，neighbor probabilities 最大变化不超过 1.50e−7，与解析不变性一致。以上仅是 C8 和当前 KL 定义的性质，不是 heldout 或闭环重要性测量。

这并不意味着“总 loss 放任 84% 的误差”，原 NMSE 仍约束所有 latent 坐标，也不意味着不可见方向必然重要。问题在于新增项只偏重少数视觉邻域方向，却没有证明这些方向是 action reader 敏感的方向。

后续机制诊断还需校正维数口径：这一定义在完整 37632 维 future 中的可见 rank 约为 1248，仅占约 3.3%；各向同性误差原本就会有约 96.7% 的能量落入 nullspace。实测可见子空间承载约 16.2% 的标准化误差能量，比其维数占比更高。因此“83.8% 不可见”本身不能作为 KL 方向错误的证据；必须结合按维数归一化的 reader 梯度和实际等 NMSE 干预结果来判读。

在 teacher query 附近，KL 的二阶形式是局部加权平方误差：约为 2/(48² tau²) × eᵀ Cov_p(t_j) e。也就是说，它加强的是 teacher 邻居变化大的方向。对于固定 reader，一阶近似下 action 变化由 J_action e 决定，对应 eᵀJ_actionᵀJ_action e；这两个权重矩阵没有自动一致的理由。这是一个明确的设计限制，而非本次退步的因果证明。Value Equivalence 的工作也强调模型的价值要结合其下游使用来定义，但该论文不是本实现有效性的证据。[Grimm et al., 2020](https://arxiv.org/abs/2011.03506)。

**五、RMSE / NMSE 合理吗，还有哪些指标？**

First10 motor RMSE 与每次 replan 实际执行前 10 个动作相匹配，是有用的 teacher-action fidelity 指标。它在 normalized action 坐标中等权平均 6 个 motor dimensions，仍不等于物理位姿误差或控制代价；它也会掩盖持续同方向偏差、接触时刻的小错误及个别时间步大错误。MSE 是训练目标，RMSE 更方便读；同一 query 上排序一致，但跨 query 的 mean(RMSE) 不等于 sqrt(mean(MSE))。

为排除汇总形式导致误判，本次也检查了 first10 motor 原始 MSE：heldout mean-query MSE 为 0.00201413 → 0.00213978，对应 sqrt(mean-query MSE) 为 0.0448791 → 0.0462577，仍退步。因此当前主结论不是取平方根/平均顺序的产物。

Future NMSE = ||z_Q−z_T||² / ||z_T||²，排除固定 observed frame0，衡量相对 teacher 的整体 future 误差。这是合理的无量纲量化保真度指标；但它不按控制重要性加权，误差可集中在高能量 channel、背景或大面积静态内容。它也不是对真实未来的预测准确度。Teacher 只是参考，并非保证正确的 ground truth。

NMSE 不是无用指标。本轮 heldout 的 motor–NMSE case-level Spearman，baseline 为 0.613、candidate 为 0.499；两者配对变化的 Spearman 为 0.423。18/40 两者均变差、10/40 均改善，但 **8/40 latent 改善而 motor 变差**，另 4/40 相反。它有相关性，无法充分决定 action 结果；这些描述性关联也受 case 难度等因素影响。

之前独立的一次 equal-NMSE 机制诊断也提供了相关证据：在固定 BF16 reader 下，仅缩放 action-only future 的 teacher-relative error，使 NMSE 从 0.093011 降到约 0.071786，first10 motor RMSE 就从 0.041243 降至 0.036074，重现了原 action+latent 收益的大部分。该实验是另一个 22-test-case cohort 的 teacher-assisted 对照，不是本次训练收益或闭环泛化证明。参见 [等 NMSE 原报告](../../fastwam-vq-future-nmse-matched/RESULTS.zh.md)。

| 实际已经使用 | 作用与边界 |
|---|---|
| First10 / Full32 / Late22 motor RMSE | 检查执行区间及其余预测 horizon 的误差 tradeoff |
| First10 gripper RMSE | 连续输出保真度；不是实际离散开合错误率，训练 action loss 也没有直接包含 gripper |
| Future global NMSE、原 MSE+NMSE objective | 检查原目标是否改善，保持同一比较口径 |
| Raw action MSE 与 sqrt(mean MSE) | 排除 RMSE 汇总方式改变排序 |
| 配对中位数、改善/变差计数、suite/seed/task 分组、leave-one-case-out | 判断均值是否有代表性，不能当作独立重复或显著性证据 |
| KL、teacher entropy、最大概率、latent 偏导 norm/cosine、pooling 与 anchor nullspace | 诊断新增 loss 的尺度和几何；不是闭环指标 |
| C8 分 future time NMSE、temporal-difference NMSE | 看时间结构；目前只比较未训练初始化与 KL 最终值，不能声称胜过 trained baseline |

目前没有为这次 candidate 测量 denormalized 分维动作误差、累计动作偏差、实际 gripper sign/开合时刻错误、接触事件、闭环成功率，也没有完整参数梯度的 loss 间 cosine、多训练 seed 方差。连续 gripper RMSE 尤其应与最终 sign/阈值后的命令区分；平均数变小不保证开合决定更一致。

Fixed-input 指标还不能覆盖 action 改变未来 observation 后的分布偏移，序列任务中这种 feedback 是独立问题。[Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html)。因此现在可以说 KL 没通过已有 fixed-input 主指标，但不能断言它已经改善或恶化闭环，也无法仅凭当前指标认定原 action 与 latent 两项谁更错位。

**后续优先级。** 保留 C8 A+Z、rho1.5 为现有 baseline，不把本次 KL 升为 baseline。先诊断新增 loss 是否真正提供了稳定、与 action 有关的梯度，再扩大 calibration 或调更多参数：如果继续实验，最有辨识力的是固定同一小批 C8 输入追踪各项 loss，以及对同一 trainable 参数集合测三项梯度的大小和方向；这会需要新的模型计算，本次没有执行。改变训练步数、learning rate、reset 或 calibration 数量时，原目标也要有相同预算和设置的对照，旧 baseline 仍保留为参考。

若重设计 KL，不能只追求增大 rank 或减少 nullspace：任务无关方向被忽略可能是优点。应先说明概率分布所保留的信息为何会影响 reader，例如检验 teacher-anchor 方向与 action sensitivity 的关系。把 deterministic latent 简单套上同协方差 Gaussian 后再加 KL，本质上只是另一个加权 MSE，不会凭 KL 名称自动解决语义/控制错位。优化和设计选择应使用 calibration / 独立开发数据，已有 heldout 作为回归比较；最终泛化需另留未参与选择的数据。这次没有自动启动新训练、ablation 或 280 episodes。

可复查的机器结果为 [diagnosis_v2/summary.json](diagnosis_v2/summary.json)。补充 CPU 作业 25903851.pbs101（2m36s）和 25903971.pbs101（1m54s）均为 PBS F / Exit_status 0、artifact exit 0、PHASE_COMPLETE，新增模型 forward 为 0。原实验完整结果和公平性说明见 [CONCLUSIONS.zh.md](CONCLUSIONS.zh.md) 与 [ANALYSIS.zh.md](ANALYSIS.zh.md)。
