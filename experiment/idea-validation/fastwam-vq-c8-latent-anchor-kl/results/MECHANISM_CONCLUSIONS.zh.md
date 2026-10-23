# C8 anchor KL 机制诊断结论

2026-10-22。完整数值见 [MECHANISM_ANALYSIS.zh.md](MECHANISM_ANALYSIS.zh.md)，机器结果见 [mechanism_analysis/summary.json](mechanism_analysis/summary.json)。

**目前最明确的问题是有限步优化与实际量化 forward 之间的落差。单纯“KL 权重太大”缺乏支持；当前 affinity KL 的额外控制价值也没有得到稳定证据。不能把结论简化成 KL 的方向整体反了。**

本轮使用完整 C8，分别在未训练初始化和当前 KL update8 checkpoint 上计算三项参数梯度，完成 22 个可回滚更新条件和 8 个等 NMSE reader 对照。参数 probe 使用当前 decoded-BF16 VQ + dynamic A4、BF16 KV 路径；reader 对照使用原始冻结 BF16 reader。原 C8 A+Z、rho1.5 trained baseline 保留既有结果，没有重放。以下不是独立泛化或闭环结果。

**权重：没有发现 KL 梯度支配模型更新的证据。**

| 完整 C8 平均梯度 | 初始化 | KL 最终 checkpoint |
|---|---:|---:|
| 加权 KL / 原 MSE+NMSE 梯度范数 | 11.64% | 10.16% |
| KL 与 NMSE 的参数梯度 cosine | 0.970 | 0.708 |
| KL 与 action 的参数梯度 cosine | 0.395 | 0.253 |

此前在输出 latent 上测到的约 40% 梯度比，只是相对 NMSE 的输出偏导，不能直接当作相对原目标的参数更新占比。本轮参数结果也不是 Adam 更新比例；Adam 的预条件和历史 moments 还会改变实际步长。

将 beta 降到十分之一，在同一点、同 optimizer state、同 step fraction 的六组配对中，平均 action MSE 是三组改善、三组变差。比如最终 checkpoint、quarter step 下，fresh Adam 时较小 beta 改善 6/8 cases，保留原 moments 时却只有 1/8 改善；整步时结果又不同。因此不能把 beta 降十倍当成已经验证的修复。

保留原 moments 时，beta=1 与 beta=0 的 Adam proposal cosine 约 0.99982；fresh Adam 下约 0.96951。当前 loss 的改变能在多大程度上改变更新，明显依赖 optimizer state。这个比较包含步长大小与方向的共同变化，不单独证明 reset 更好或更坏。

**优化：已有三条直接证据，且并不只发生在新增 KL 上。**

首先，22 个完整 C8 更新条件中，一阶 g·delta 预测 action MSE 下降、实际均值却上升的有 11 个；原 MSE+NMSE objective 有 5 个，KL 有 2 个。这些条件相关，不是 22 个独立样本。beta=0 时也出现 action 预测下降而实际上升，说明这不是 KL 独有的失效。

其次，初始化的八个单步条件中，BF16 部署 row gains 改变数全部为 0；FP32 master gains 明明有非零 delta。以原 beta、整步为例，原目标的一阶预测下降中约 59% 来自 gains，而这些 gains 的部署值在该步完全未变。CPU 重算的 g·delta 与保存值最大相对差约 5.85e−9，支持这一分解。它证明了这些局部 probe 的部署量化死区，不能外推为原八步训练 gains 从未改变；最终 checkpoint 的单步中确实已有一部分 gains 跨过 BF16 格点。

一阶预测落差不全由 gains 死区解释：books 的 BF16 rounding、A4 的 STE 以及非线性有限步变化也同时存在。较小步长也并非总更安全：最终 checkpoint 的原 beta fresh-Adam quarter step 使三项均值都上升，而整步使三项均值都下降。当前证据支持“应验证部署后的真实步长效果”，不支持把问题统一归为 learning rate 太大。

第三，当前 checkpoint 在 calibration 上仍有明显优化余地。从同一个 KL 最终 checkpoint 出发，仅用完整 C8 平均的原 action MSE+NMSE 梯度、fresh Adam 做一个整步诊断，得到：

| C8 指标 | KL 最终 checkpoint | 不含 KL 梯度的一步诊断 |
|---|---:|---:|
| First10 motor MSE | 0.001754010 | 0.001562458 |
| First10 motor RMSE | 0.0415491 | 0.0389636 |
| Future NMSE | 0.0495689 | 0.0370588 |
| Anchor KL | 0.00622238 | 0.00349707 |

Action MSE 和原 objective 均有 7/8 cases 改善，配对中位数也下降。即使不使用 KL 梯度，三项均值仍能同时改善。因此原 update8 不是“只能牺牲一个目标来改善另一个”的充分优化结果。这个 probe 改用了全 C8 平均梯度和 fresh Adam，并已回滚；它不等于原训练顺序再继续一步，也不是新的 trained baseline，更不证明其 heldout 表现会更好。

最终点的 KL 跨 case 梯度有 14/28 对 cosine 为负，初始化为 4/28，说明共享参数下的样本间拉扯值得关注。但本轮没有逐更新固定输入轨迹或顺序消融，不能把它确认为历史遗忘、欠拟合或某个 reset 事件的独立因果作用。

**方向：当前 KL 有一定动作相关性，但没有稳定的增量优势。**

在每个 latent 输出上，KL 与 NMSE 的偏导 cosine 约 0.23–0.34；落到当前 books/log-row-gains 参数后，逐 case cosine 达到 0.82–0.99。两个目标在 latent 坐标中不同，并不意味着它们会给当前可训练参数提供同样不同的方向。全 C8 平均梯度的投影表明，KL 的很大一部分是在加强 NMSE 方向；初始化的局部梯度投影相当于把 rho 从 1.5 加到约 1.719，最终点约 1.681。这只是同一点的梯度投影，不是 loss 全局等价、Adam 更新等价或推荐的新 rho。

其余正交部分与 action 平均梯度的 cosine 为 −0.377、−0.137，提示额外方向未自然对齐 action。但全 C8 正交部分还含有隐含样本权重变化；最终点逐 case 是四个负、四个正，不能据此认定 KL 在所有输入上方向错误。

为直接检验 latent 方向，本轮将保存的 candidate residual 正交分为 KL 可见和不可见两部分，分别作 teacher-assisted 修复，控制两种修复的实际 BF16 future NMSE 相同。所有 8 cases 的匹配误差都通过预设 0.1% 相对容差，最大实际相对误差约 1.02e−5；teacher future 注入后的动作重放误差全部为 0。

| 同一冻结 BF16 reader 的输入 | Future NMSE | Anchor KL | First10 motor RMSE |
|---|---:|---:|---:|
| 原 candidate future | 0.0495689 | 0.00622238 | 0.0168265 |
| 修复 KL 可见误差 | 0.0456759 | 0.00309902 | 0.0160154 |
| 修复 KL 不可见误差 | 0.0456758 | 0.00622368 | 0.0161456 |

两种修复 NMSE 几乎相同，可见修复的 KL 却低约一半。它的 motor 平均值只略好，逐 case 为 4 胜、4 负，full32 motor 均值反而略差；没有稳定的额外 motor 优势。Gripper 连续 RMSE 是可见修复 4 胜、0 负、4 平，这是一个局部有利信号，但尚不是实际开合时刻或闭环成功证据。这些 reader 数值只隔离 future 对原 BF16 reader 的影响，不能与上表量化 student 的 motor RMSE 直接混为同一条性能曲线。

此前“约 84% 标准化 residual 能量位于 KL nullspace”需要谨慎解读：不可见空间本来就占约 96.7% 的维数。按维数归一化后，可见空间的 reader 梯度能量密度反而是不可见空间的 2.22–4.05 倍。因此大 nullspace 本身不是 KL 方向错误的证据。更具体的问题是：可见子空间有动作敏感性，当前 affinity KL 的方向与权重却没有稳定识别出比 NMSE 更值得优先修复的误差。Reader action-loss 梯度与 NMSE 的 cosine 在 7/8 cases 中高于与 KL 的 cosine，也与这一解释相容。

**下一步应优先修正可观察的优化问题，再评价新的信息论目标。**

先保证 FP32 更新在部署路径上产生可控变化，并在固定完整 C8 上检查各 loss 的实际下降及逐 case 分布；按 books 与 gains 分组观察，而非只看梯度连通或 loss 数值。单纯减小 beta、learning rate 或延长顺序训练都还不是已验证方案。

若继续改 KL，应让它表达与 reader 行为相关的条件信息，而不只复现局部邻居 affinity；保留 NMSE 作为完整 latent 的锚。还应检查新的 latent 目标经过当前可训练参数反传后是否仍提供有用的增量方向。任何训练策略或预算改变，都应同时给原 MSE+NMSE 配套同设置、同预算对照，保留原 C8 rho1.5 作为历史参照；设计选择使用 calibration/独立开发数据，已有 heldout 用作回归比较，最终泛化另作确认。

本轮没有自动启动新训练、ablation 或 280 个闭环 episodes。优化不足与泛化问题可以并存；以上 C8 局部证据不能独立解释先前所有 heldout 退步，也不能确定原 action loss 与 latent loss 谁更偏离现实闭环。

**完成证据。** GPU 25905057.pbs101 为 F/Exit0、artifact exit0、PHASE_COMPLETE，耗时 2h44m31s；CPU preflight 25905027 为 1m34s，缓存优化几何 25905822 为 3m02s，最终汇总 25905075 为 2m10s，均 F/Exit0 且 artifact 验证通过。完整 16/16 梯度 records、22/22 panels（176 case queries）、8/8 reader records（32 reader queries）齐全，没有删去不利条件。新增 192 个 student full queries、32 个 reader-only queries，48 个参数 reverse passes、8 个 reader reverse passes；内部 checkpoint recomputation 不另算逻辑 query。全部 tensor/模型计算在真实 PBS allocation 内，GPU 使用每 15 秒写 job log。原 trained baseline 0 新 queries、teacher 0 新 full queries、0 持久训练更新、0 闭环。原 checkpoint 与 banks 未改写，所有临时参数 probe 均回滚。
