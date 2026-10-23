# C80 A+Z+frozen-reader KL：结果与停止结论

2026-10-22，按用户最新要求结束本轮：C80 KL 已完整完成；C80 A+Z 在2个已保存更新后取消，依赖双组的原 aggregate 未执行。另用 CPU allocation 分析现有缓存，新增训练更新、模型 forward、teacher forward、闭环均为0。监控 subagent 已停止。

**实际判断：C80 KL 没有超越原 C8 A+Z、rho1.5 的 held-out motor 表现，没有足够收益支持继续投入这版配方。扩大 calibration 缓解了之前 C8 KL 的部分退步，但不能把它解释成“已解决泛化问题”，也不能证明 KL 的独立贡献。** 原 C8 A+Z 保留为 incumbent。

## 实际 student 的完整结果

比较参照为历史训练8步的 C8 A+Z，不是同预算 C80 A+Z。数据来自相同 observation、prompt/proprio、teacher 和 sampler seeds 的缓存。C80 calibration 为80 cases×seed0；held-out 为40 cases×2 sampler seeds。先在 case 内平均 seeds，再在 suite 内平均 cases，四个 suite 等权。表中的 RMSE 是这种汇总下的 query RMSE 均值，不是将所有误差混在一起后开根号。

| Held-out 指标 | 原 C8 A+Z | C80 KL | 相对变化 | 改善/变差 cases |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 0.04429677 | 0.04444355 | +0.33% | 20/20 |
| Full32 motor RMSE | 0.04750098 | 0.04842031 | +1.94% | 23/17 |
| Late22 motor RMSE | 0.04852850 | 0.04961731 | +2.24% | 21/19 |
| Future latent NMSE | 0.06087737 | 0.05436114 | −10.70% | 23/17 |
| First10 gripper RMSE | 0.05705403 | 0.05417848 | −5.04% | 12/28 |

主指标 first10 motor 的配对中位差为 +0.00005103，均值和中位数均没有改善，case 胜负各半。Full32/late22 的配对中位差虽为负，均值仍变差，说明多数 case 的小改善与部分较大退步并存。Gripper 均值下降，但28/40 cases变差，中位差为 +0.00231527，不能概括成普遍改善。

| Held-out suite | First10 motor 相对变化 | Motor 改善/变差 | Latent NMSE 相对变化 |
|---|---:|---:|---:|
| LIBERO-10 | −4.74% | 6/4 | −30.56% |
| LIBERO-Goal | +3.46% | 4/6 | −4.59% |
| LIBERO-Object | +3.85% | 4/6 | −8.25% |
| LIBERO-Spatial | −0.92% | 6/4 | −0.71% |

两个 sampler seeds 的 motor 方向也不一致：seed0 为 0.04364072→0.04321786，seed1 为 0.04495283→0.04566924。这里没有稳定、广泛的 motor 收益。NMSE 改善的23个 cases中，16个 motor 也改善，7个 motor 反而变差：latent 保真与动作保真有关，但不是同一个排序。

完整 C80 calibration 相对历史 C8 A+Z：motor 为0.04414696→0.04364685（−1.13%，45/80改善），NMSE 为0.05473537→0.05223911（−4.56%，46/80改善）。不能将上一轮仅8个 calibration cases 上的 −8.11% motor 改善，直接与这里80个 cases上的 −1.13% 比较；评估集合变了。

## C80 是否解决了 C8 的泛化问题

相对历史 C8 reader-KL，C80 KL 的 held-out motor 从0.04514992降到0.04444355（−1.56%，21/40改善），NMSE 从0.05735647降到0.05436114（−5.22%，19/40改善）。扩大后的完整 recipe 的确比之前 C8 KL 更接近原 baseline。

但 C8→C80 同时改变 calibration 覆盖、更新数8→80、action 标准化统计和一次性标定的 beta，不能单独归因于样本数量。且最终 motor 仍未超越原 C8 A+Z；NMSE 相对 C8 KL 的均值收益也不是多数 case 的一致收益。当前结果支持“覆盖/训练预算可能影响结果”，不支持“问题只是 C8 太小”。

更关键的是，本轮现在有完整 calibration 上的 reader 前后比较：

| 同一 C80 calibration 上的指标 | 原始共同初始化 | 最终 C80 KL | 相对变化 |
|---|---:|---:|---:|
| Frozen-reader soft-bin KL | 0.02748146 | 0.03080744 | +12.10% |
| 标准化 frozen-reader MSE | 0.00913232 | 0.00997667 | +9.25% |
| Future latent NMSE | 0.06169479 | 0.05223911 | 下降 |

这里 reader、teacher、噪声、C80 标准化和 soft-bin 定义均固定，前后可比。**辅助 KL 在训练集合上的均值也未下降**，因此不能采用“先在 calibration 把 KL 学好了，只是 held-out 没泛化”的解释。表中是整体均值，不能据此声称每个 case 的 KL 都变差。

最终 calibration/held-out reader KL 均值分别为0.03080744/0.03065093，标准化 reader MSE 分别为0.00997667/0.00988807。这两个集合的最终 reader 分数接近；但没有原 C8 A+Z 或未训练模型的 held-out reader endpoints，不能声称 held-out reader 相对它们改善或退步。

## 是优化、权重还是 KL 设计的问题

数学实现没有显示明显断梯度或重放不一致：80个更新均覆盖614个目标 Linears，梯度 connected/finite/nonzero；student 的 staged replay action/future 最大差为0，初始共同 banks、teacher-reader 与最终 inference parity 通过，首步部署输出确实变化。分阶段 VJP 通过链式法则传播辅助项，不是只把一个无梯度标量加到 loss 上。这是在既定 STE/可微计算图下的等价，不保证离散量化部署目标随之下降。

但是，这些检查不保证辅助目标被充分优化。beta=0.0095308743，仅使训练前平均 KL 的加权标量值等于加权 NMSE 的10%；它不保证 future 梯度或 parameter 梯度占10%，更不保证 Adam 更新中各项的影响比例。原 rho=1.5、normalization_c、lr5e-4保持不变，不等于它们已经适合80个不同 contexts。

本轮是一遍在线更新，每个 case 只遇到一次。训练 history 中相邻行来自不同 case，不能把后段数值较小直接当成同一分布上的收敛曲线。辅助权重较弱、主辅梯度冲突、顺序遗忘、量化部署中的优化误差都仍可能解释“NMSE下降而 reader loss 上升”。当前没有组件梯度夹角或匹配 C80 A+Z，因此无法给这些解释分配因果份额，也不能断言“beta 调大就会解决”。

设计层面的一个具体限制已经可以从最终缓存看见。令 e=R(o,z_Q)−a_T，g=a_Q−R(o,z_Q)，则实际 student 的误差为 e+g，存在逐 query 的精确恒等式：

    MSE(a_Q,a_T) = mean(e²) + mean(g²) + 2 mean(e*g)

| Held-out 误差分解 | 均值 |
|---|---:|
| 实际 student motor MSE | 0.00205545 |
| Future 经 BF16 reader 后的误差 mean(e²) | 0.00054784 |
| Student 与 BF16 reader 的差异 mean(g²) | 0.00144086 |
| Cross term | 0.00006675 |

实际 student RMSE均值为0.04444355，而固定 BF16 reader 使用同一 student future 时为0.01962550；两者的读出差异 RMSE均值为0.03782241。当前 future 在 BF16 reader 下的行为偏差，小于量化 student 与这个 reader 的差异。g 包含两种计算路径的差异，不能把它全部称为 action head 单独的量化误差。

这支持一个设计疑点：辅助项保护 BF16 reader 认可的 future，但部署使用量化 student，二者的敏感方向和误差补偿未必相同。该辅助项没有直接约束 g。误差分解是描述性恒等式，含 cross term；不能把各项解释成独立、可直接消除的因果份额，也不证明删除 e 就能得到相应闭环收益。

## KL 的数学合理性与真正利用的结构

实现采用冻结 BF16 reader R，以相同 observation/noise 读取 teacher 和 student futures；R(o,z_T)=a_T 的缓存重放验证通过。对每个前10步、6个 motor scalar，将标准化读出 u 通过固定 Gaussian smoothing 和16个 bins 映射成概率 q(u)，计算 teacher→student KL，再对60个 scalars取均值。

这是有效的离散概率 KL，具有非负性。其含义是选定平滑/分箱后的控制读出保真，不是估计真实 latent density、预测不确定性或 mutual information。它逐 scalar 计算，不直接建模跨 motor、跨时间的联合动作分布。

在 teacher 附近，对 reader 的局部平滑可微近似，令 f(z) 是标准化 reader 输出，J 是 f 对 latent 的 Jacobian，则局部二阶形式为：

    L_aux ≈ (1/120) δzᵀ Jᵀ diag(I_bin(f(z_T))) J δz
    I_bin(u) = Σ_b [q'_b(u)]² / q_b(u)

而 global NMSE 在固定 teacher 下是 ||δz||² / ||z_T||²。前者确实利用了 NMSE 没有显式编码的结构：reader 对 latent 方向的敏感性，以及 observation 相关的非线性读出。合理性主要来自 reader 赋予 latent 误差的用途，不来自 KL 这个名称。只覆盖 reader 可见方向本身不是数学缺陷，原 NMSE 也仍保护其他方向。

但相同 reader 上的标准化 MSE 同样具有 JᵀJ 结构。KL 在此增加的主要是 soft-bin 概率几何的 scalar 权重；它没有凭空获得额外观测信息。若不分箱而直接比较同方差 Gaussian，KL 就精确等于标准化 MSE 乘常数；固定分箱会丢弃部分信息，并使不同位置、特别是尾部的敏感性不同。当前没有 reader MSE ablation，不能证明 KL 特有的几何比这个更简单的目标有价值。

因此应区分两件事：数学上有效、对 latent 有 task-dependent 结构；实践中这版固定 reader/soft-bin/权重/80步配方是否值得。前者成立，后者尚未得到实际 motor 收益支持。Calibration KL 本身未降低，也说明本轮不能当成“已成功实现该几何约束，但约束仍无效”的干净反证。

## 指标与证据边界

First10 motor RMSE 与每次 replan 执行10步的协议对应，适合作为固定输入下的主控制保真指标。Full32/late22用于检查 horizon 其余部分；它们不代表这些动作在闭环中一定被执行。NMSE适合作为 latent 数值保真，但 global energy normalization 不能表达所有控制敏感方向，不能单独充当闭环代理。

本轮另外采用 reader KL、标准化 reader MSE、reader motor RMSE、student-reader gap、cross term、case胜负/中位差、suite和seed分层，因此没有只凭 RMSE/NMSE 两个均值判断成功。Gripper仍是连续输出误差，不等于离散开合时刻或抓取成功率。

这里的 held-out 是相同任务、不同 demonstrations 的固定开发/回归集合，已多轮查看；没有 unseen-task、全新独立测试集或闭环泛化结论。两个 sampler seeds 不是独立训练复现，不把80 queries当作80次独立训练实验。不进行显著性优越或统计等价声明。原先“KL 相对匹配 C80 A+Z”的 gate 因该对照取消而不可评定；不能把历史 C8 对照冒充这个 gate 的判定。

## 为什么总时长变长

以下是各训练实现保存的逐步耗时，计时边界和 cases不完全相同，只作描述性比较：

| 实现 | 完整更新记录数 | 排除首步后的中位耗时 |
|---|---:|---:|
| 原 C8 A+Z | 8 | 224.52秒 |
| C80 KL | 80 | 256.00秒 |
| 已取消 C80 A+Z | 2 | 258.56秒（仅剩1个样本） |

记录显示 C80 KL 单步约慢14%，没有显示单步成倍变慢。训练总更新数增加10倍，并增加初始化和完整 student/reader 评估，是总运行时间大幅增加的主要原因。KL作业实际 walltime 为6小时26分50秒。

实现上，本轮 A+Z 也沿用了 staged 路径：一次 no-grad student forward，再一次 differentiable student replay/VJP；原 C8 A+Z 每步只有一次 differentiable student forward/backward。这个额外 forward 对不带 reader 的 A+Z 是可避免开销，我应在提交前去掉。KL还包含 frozen-reader backward 和 dense weights 的CPU/GPU搬移。这里没有组件计时，且 A+Z 只有2步，不能把全部差额归给某一组件，也不能把当前 A+Z 与 KL 接近的耗时解释成“KL 几乎免费”。

## 最终处置与复现文件

停止本版 reader-KL 配方，不续跑 C80 A+Z、不扫权重/学习率/bins，不启动新实验或280闭环。若以后重新考虑 reader 辅助目标，现有证据首先要求说明它如何对应部署时的量化读出，以及辅助目标是否实际降低；单纯再扩大 calibration 不是已有证据支持的充分修复。

KL作业25913982.pbs101：F/Exit0、artifact0、PHASE_COMPLETE，80 updates、160 student endpoints、160 reader endpoints通过。已有训练运行共394个新 student full queries、323个 reader-only queries，teacher新 full queries和闭环均为0。后续缓存分析25920011.pbs101：CPU、2分37秒，F/Exit0、artifact0、PHASE_COMPLETE，完整160对历史/候选记录通过，新增模型计算为0。

A+Z作业25913984.pbs101已取消，F/143，保留2步checkpoint；原 aggregate25914065.pbs101取消且未执行。监控/root/c80_monitor_resume1已确认停止。原始设计仍保存在 protocol.json，范围修改与完成状态见 RUN_STATE.json、completion_evidence.json。

机器结果：[summary.json](cached_analysis/summary.json)；逐case配对：[paired_cases.json](cached_analysis/paired_cases.json)；实际KL运行记录：[worker summary](reader_kl/summary.json)。所有张量与统计汇总在真实PBS compute allocation内完成；本地仅整理报告和小型控制文件。
