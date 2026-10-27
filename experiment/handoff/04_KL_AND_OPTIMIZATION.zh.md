# KL 目标、控制响应与优化落差

返回 [总交接](../HANDOFF.zh.md)。本项目出现过四种不同的 KL：逐 scalar soft-bin、student–student relational、student–teacher-anchor affinity、frozen BF16 future-to-action reader。它们作用对象、可训练参数、初始化和训练预算不同；不能统称“加了 KL”后混合结果。当前主方法用最后一种，并保留原 action MSE 和 future NMSE。

## fastwam-vq-dual-kl

**问题与设计。** 在原 per-channel D+ratio activation 优化上做 2×2loss 替换：MSE+NMSE、action soft-bin KL+NMSE、MSE+latent soft-bin KL、dual-KL。VQ 冻结，共同 C8 起点、C80/80 updates；原 MSE arm 复用并重放 160 端点。KL 坐标与 coefficients 只由 C80 初始统计冻结。

**概率定义。** 每个标准化 scalar 加 N(0,0.375²)，映射到 16 bins，15 内边界 linspace(−3.5,3.5)，两端无限；KL 方向 teacher||student，逐位置平均，不是跨位置 histogram 或真实 policy density。初始 mean loss 配平不等于梯度配平。

**结果。** Held-out motor/latent：MSE **0.031641466/0.055377875**，action-KL **0.031724741/0.059674341**，latent-KL **0.032218684/0.053663947**，dual **0.032329860/0.053033005**。Dual 改善 latent 但 motor 变差，预设 joint screen 失败，保留 MSE reference。完整四臂已结束，不自动追加闭环。

证据：[README.zh.md](../idea-validation/fastwam-vq-dual-kl/README.zh.md)、[ANALYSIS.zh.md](../idea-validation/fastwam-vq-dual-kl/results/ANALYSIS.zh.md)、[RUN_STATE.json](../idea-validation/fastwam-vq-dual-kl/RUN_STATE.json)。

## fastwam-vq-relational-kl

**问题。** 逐点 soft-bin 无法显式描述邻居结构；增加 action temporal 与 latent spatiotemporal conditional-neighbor KL 是否更合适。Action 用 ordered3-step 窗口得到 8 个 18D tokens；latent 按 camera 分别 pool2×2，196 个 48D tokens，含空间邻边与同 camera 时间邻边，排除 camera 接缝。

**公平性。** 复用 soft-bin final80；关系组从相同起点训练 80 updates。Tau 用完整 C80 teacher 正距离 median；beta 用 teacher 预设噪声 0.1 配平，不看 student/held-out。仍只训练 D/ratio，不训练 VQ。此 KL 不估计 action–latent 完整联合密度，存在平移/反号等盲区，需 soft-bin 绝对锚。

**结果。** Held-out softbin→relational：motor **0.032329860→0.031438487（−2.757%）**，future **0.053033005→0.054295895（+2.381%）**，composite **0.005913306→0.005743407（−2.873%）**。Relational 目标更低但 latentNMSE 更差；不能用一个共同 objective 隐藏取舍。CPU`25884638`完整汇总 320 端点。

证据：[README.zh.md](../idea-validation/fastwam-vq-relational-kl/README.zh.md)、[ANALYSIS.zh.md](../idea-validation/fastwam-vq-relational-kl/results/ANALYSIS.zh.md)。

## fastwam-vq-softbin-relational-plus

**问题。** 上述 fixed-input KL 改善能否转为闭环。直接复用两种 final80，每臂 280，13reference 组 3640 slots，无新训练；完整 source action/future、cache、infer parity 均 0。

**结果。** `25886452.pbs101`完成 560 新 episodes：soft-bin **199/280**，relational **198/280**；原 per-channel MSE203、C8 rho1.5 为 206、clip-only207、BF16216。Relational 对 softbin 为 9 胜/10 负/261 平，对 per-channel MSE 为 10 胜/15 负/255 平。

**判断。** 当前全量替换 softbin/relational recipe 没有闭环收益。该负结果不等同于后来“保留 MSE/NMSE，只增加 reader-KL”的主方法；不能用 198 直接评价 208checkpoint。仍是 decoded-BF16/fake-A4 执行。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-softbin-relational-plus/RESULTS.zh.md)。

## fastwam-vq-relational-vq-activation

**问题与阶段。** 将原 C8/8update VQ supervision 也换成 soft-bin+relational KL，再固定该 VQ 训 C80/80update D/ratio。比较旧 MSE-VQ8+KL-activation80，避免把 VQ-only 端点当同预算 final80 对照。

**结果。** 新 pipeline held-out motor **0.034438064**、future **0.057364387**、composite **0.0061686663**；比旧 relational 参照分别+9.54%/+5.65%/+7.40%，两个 split 八项总体指标都更差。Activation 阶段相对新 VQ-only 有恢复，但未追平旧参照。

**C8 补诊断。** CPU`25895158`复用缓存：KL-VQ8 在真实 C8 motor/latent **0.040801177/0.043700444**，优于 MSE-VQ8 **0.042133242/0.054480982**，但 held-out 分别更差 10.33%/12.32%。因此不能说 KL 连 C8 都没拟合；应分清 VQ C8 拟合、activation C80 优化与 held-out 泛化。根报告主 pipeline 已完成，`results/vq_diagnosis`某些梯度诊断只有历史待完成记录，不提升为最终因果证明。

证据：[总分析](../idea-validation/fastwam-vq-relational-vq-activation/results/ANALYSIS.zh.md)、[VQ补诊断](../idea-validation/fastwam-vq-relational-vq-activation/results/vq_diagnosis/ANALYSIS.zh.md)。

## fastwam-vq-control-kl

**问题。** 为什么 soft-bin/relational 的部分 fixed-input 指标更好，sensor-noise 闭环反而差。先重放原 40sensor 初始状态 clean/noisy，再看 5 个 MSE 成功/relational 失败案例的 47 个沿途 states，固定 reader 隔离 future 路径；另做完整 C80 的独立噪声配对与可回滚参数 gate。

**已完成证据。** 初始 4 arms×2inputs×40=320endpoints，所有 noisy full 32 action 精确重现旧输出。Noisy motor MSE-reference/softbin/relational RMSE **0.058782/0.050527/0.053921**，但原 sensor 闭环成功 **26/40、24/40、21/40**。47 沿途 states 包含 376native 端点与 376reader 干预；结果显示初始均值没有覆盖的沿途 response/bias 退化。完整 C80 clean/noisy 诊断也完成。

**当前边界。** 根报告明确“完整目标尚未完成，未启动新训练”；后续 parameter gate 与可回滚 C80 检查是阶段诊断，不是一个已部署 control-KL 训练结果。5 失败 case 按 MSE/relational 成败选出，不能解释 softbin 所有失败，更不能把这些已看 test states 回灌超参选择。

**实现注意。** Captured RNG 的 NumPy state 实际是可序列化 dict，旧 tuple 比较器出错；原失败和修复保留。观察到 future→reader 梯度非零只证明可微，不能当作 D/ratio 能有效优化新目标。接手需读 gate contract，不自动提交训练。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-control-kl/RESULTS.zh.md)、[C80_GATE_CONTRACT.md](../idea-validation/fastwam-vq-control-kl/C80_GATE_CONTRACT.md)。

## fastwam-vq-kl-direction

**问题与设计。** 新 KL final80 甚至在各自完整 C80 目标上不如旧 MSE 可行点，先排除 training/eval 前向差异并检查真实有限步。4 个 checkpoint/objective 单元，每个 C80 平均 gradient；saved/fresh Adam×1/0.5/0.25 步，每候选都从同一 checkpoint 出发并投影到 bounds，结束精确回滚，不保存新模型。

**结果。** 全 320 同参数 checkpointed/no-grad action/future parity0；24 候选中 **16 个** surrogate 预测下降、真实 C80 目标却上升。Dual checkpoint saved quarter step 使 dual objective0.002797668→0.002639068，低于旧 MSE 同公式 0.002743996；而更小步/fresh moments 并不总更好。

**含义。** 存在 surrogate/Adam 提议与实际量化 forward 有限步的落差，不能只归为 eval 路径不同或“学习率太大”。它不是第 79→80 步历史的重建，也不能因某候选 calibration 变好就预言 held-out。后续 full batch-refine 已经另立实验完成；旧 RUN_STATE 的 active 不是无限训练授权。

证据：[README.zh.md](../idea-validation/fastwam-vq-kl-direction/README.zh.md)、[ANALYSIS.zh.md](../idea-validation/fastwam-vq-kl-direction/results/ANALYSIS.zh.md)。

## fastwam-vq-kl-fullbatch-refine

**设计。** 复用上条完整 C80 梯度与候选，按实际 calibration objective 选择至多一个额外 update，并给 MSE 同样精修机会。选择不读 held-out；最终仍完整评估原 40×2。

**结果。** MSE 原/精修 motor **0.031641466/0.032115747**，latent **0.055377875/0.061798732**；dual 原/精修 motor **0.032329860/0.031900538**，latent **0.053033005/0.060566725**。Dual 精修略优于同样退化的 MSE 精修，但相对原 MSE 两项仍差约+0.82%/+9.37%。Motor 主配对 20/40 改善、LOCO 跨 0，gate 失败。

**结论。** 完整 C80 真实 loss 选步可修复 calibration，不保证泛化；不能把“胜过退化对照”写成超过原已知好解。本轮已完成，保留原 MSE，后续 warmstart 单独记录。

证据：[ANALYSIS.zh.md](../idea-validation/fastwam-vq-kl-fullbatch-refine/results/ANALYSIS.zh.md)、[decision.json](../idea-validation/fastwam-vq-kl-fullbatch-refine/decision.json)。

## fastwam-vq-kl-mse-warmstart

**问题。** 从原 MSE80 这一更好的共同起点，复用已算 KL 梯度和选中候选，能否得到更好 held-out。没有新 gradient/teacher，至多一个逻辑 update；calibration 选步后一次完整 held-out 评估。

**结果。** Warmstart dual motor **0.032662221**、latent **0.061515833**，均差于原 MSE0.031641466/0.055377875，也没有超过 matched MSE refinement 的 motor。原 MSE 比较 motor 仅 16/40 cases 改善；不能把 23/40 相对另一退化 dual 的 wins 拿来替换主比较。

**当前决策。** GPU`25882284`、CPU`25882411`完成，但信息论 loss 联合泛化目标未实现。后续只准备 ratio-only 隔离实验；最后明确约束为不启动新训练，不可从`goal_active=true`推出继续提交。

证据：[ANALYSIS.zh.md](../idea-validation/fastwam-vq-kl-mse-warmstart/results/ANALYSIS.zh.md)、[RUN_STATE.json](../idea-validation/fastwam-vq-kl-mse-warmstart/RUN_STATE.json)。

## fastwam-vq-kl-gradient-geometry

**范围。** CPU 读取同一个原 MSE80 点的缓存 MSE/KL 梯度、Adam states，做参数几何与投影步模拟。0forward、0backward、0 updates、未用 held-out。

**结果。** Ratio 组 MSE/KL cosine **0.999955**，channel 组 **0.987747**，整体 **0.996776**。少数坐标反号不等于主要梯度质量冲突；不支持“两个混合目标在这个点整体方向相反”。Saved/fresh Adam 及 bounds 会改变真实 proposal。

**限制。** g·delta 是线性预测，不能据此断言 finite-step 损失；同方向也不证明同输出。CPU`25882486`完整结束，旧 failed attempt 另留。它为固定 D 的 ratio-only 提案提供定位动机，不提供新 loss 有效性结论。

证据：[ANALYSIS.zh.md](../idea-validation/fastwam-vq-kl-gradient-geometry/results/ANALYSIS.zh.md)。

## fastwam-vq-kl-candidate-crosscheck

**问题与范围。** 同一 MSE80 起点两套已有 12 候选，分别用固定 Jm 与 Jk 交叉评分；只读 JSON，C80，不读 held-out、不改模型。

**结果。** 12 候选中，两公式对每个候选的升降方向完全一致：2 个都降、10 个都升，不能只说评价指标排序不一致。Saved/full 下，MSE 梯度候选 Jm/Jk **+8.1265%/+7.7194%**，KL 候选 **−2.8893%/−2.1120%**；投影 update 距离只有 MSE-update 范数的 0.58925%，仍产生明显有限输出差异。

**含义。** 微小 proposal 差异、A4/compensation/非线性共同影响有限步，但没有分离哪一个是根因。RMSE 三角不等式只给 student 间距离下界，不是实际 output tensor 测距。CPU`25882534`完成，0 新模型计算；不据此用已见 held-out 重新选 best。

证据：[ANALYSIS.zh.md](../idea-validation/fastwam-vq-kl-candidate-crosscheck/results/ANALYSIS.zh.md)。

## fastwam-vq-kl-ratio-refine

**设计。** 保留原 MSE80 的 per-channel D，固定 D，仅开放 614ratios，用缓存 C80 梯度比较 MSE 与 dual-KL；目的是隔离 update 子空间，而非预设 D 已被证明导致失败。

**实际状态。** `prepared_not_submitted`、`uploaded=false`、`submitted=false`、`runtime_validated=false`，0 updates/forwards。只有 offline 代码和 preparation checks，没有 GPU 结果，不能将预期改善写入结果表。

**下一步边界。** 原最后约束“不启动新训练”仍记录在此。接手可以审阅实现与 decision，但本次交接文档不构成提交授权；任何实际执行需由后续研究任务明确决定。

证据：[decision.json](../idea-validation/fastwam-vq-kl-ratio-refine/decision.json)、[RUN_STATE.json](../idea-validation/fastwam-vq-kl-ratio-refine/RUN_STATE.json)、[preparation_checks.json](../idea-validation/fastwam-vq-kl-ratio-refine/preparation_checks.json)。

## fastwam-vq-kl-continuation

**角色。** 保存未选用的 continuation controller/worker 草案：从已有 per-channel checkpoint 继续 80 updates，四种 MSE/KL 目标与 20/40/60/80 snapshots。代码的存在只说明准备过执行方案。

**状态与决策。** 本地没有该方案独立 protocol/handle/结果 receipt；上游`kl-direction`明确将它标为 unsubmitted、not selected，不能自动采用 warmstart+fresh Adam 或半 LR。实际后续走的是 full batch-refine、warmstart 及缓存诊断，不是此 80 步 continuation。

**接手方式。** 不运行该目录作为“继续未完成训练”的入口。若将来确需复用，先建立新的冻结合同与当前数值检查，保留原 incumbent；没有可交接的实验分数。

证据：[worker.py](../idea-validation/fastwam-vq-kl-continuation/worker.py)、[上游状态说明](../idea-validation/fastwam-vq-kl-direction/RUN_STATE.json)。

## fastwam-vq-teacher-anchor-kl

**原设计。** 在 MSE/NMSE 上增加 student-to-fixed-teacher-neighbor affinity KL；不同于 student–student relational，teacher anchors 完全 detach。原本计划 C80 per-channel 训练 80 步并完成 paired280，beta 按初始加权 NMSE 的 10%冻结。

**实际状态。** 用户后来更正为 C8 VQ-only、无 activation scaling/clipping、无 280 闭环。此目录`goal_status=superseded_by_user`，GPU pipeline 取消。CPU`25902483`完成 teacher 定义检查：selfKL0、spatial-roll0.657848391、time-swap0.365453346、matched bias0.044196485；这只是定义辨别能力，没有完成原计划的训练/闭环。

**接手意义。** 下一条才是修改后真正跑完的 C8 实验。不能看到这里 README“已提交 GPU”便报告 C80 anchor 闭环结果；取消来源`USER_SCOPE_CHANGE_CANCELLATION.json`保留。

证据：[README.zh.md](../idea-validation/fastwam-vq-teacher-anchor-kl/README.zh.md)、[RUN_STATE.json](../idea-validation/fastwam-vq-teacher-anchor-kl/RUN_STATE.json)。

## fastwam-vq-c8-latent-anchor-kl

**设计。** 从与原 C8 A+Z 相同未训练 banks，614books/log gains，8 updates，增加 teacher-anchor affinity KL；普通 A4，D1/ratio1，不训 activation 参数。只用 C8 定 statistics/tau/beta，复用训练后 A+Z 的 88 条对照，不把未训练 reference_cache 当 trained comparator。

**结果。** Held-out motor **0.0442968→0.0458895**，latent **0.0608774→0.0614710**，保留 baseline。后续完整机制诊断含 16 组梯度、22 可回滚条件、8 等 NMSE reader 对照，0 持久新 updates。

**机制结论。** 加权 KL/原目标参数梯度范数约初始化 11.64%、终点 10.16%，不能说 KL 梯度支配。22 条件中 11 个 action 一阶预测下降而真实上升；初始 8 种单步里 BF16 部署 row gain 全部未变，尽管 FP32 master 有 delta。Beta 降 10 倍有好有坏，quarter step 也不总更安全；存在部署格点/STE/有限步落差，但未唯一分离根因。等 NMSE 下修复 KL 可见/不可见方向的 motor 差很小，额外控制价值未获稳定证据。

证据：[README.zh.md](../idea-validation/fastwam-vq-c8-latent-anchor-kl/README.zh.md)、[MECHANISM_CONCLUSIONS.zh.md](../idea-validation/fastwam-vq-c8-latent-anchor-kl/results/MECHANISM_CONCLUSIONS.zh.md)。

## fastwam-vq-c8-frozen-reader-kl

**新方向。** 不用 latent affinity 近似“信息重要性”，而让固定原始 BF16 reader 实际读 student future，再对 first 10motor 的 soft-bin 输出蒸馏。保留 A+Z，reader 参数冻结、future cotangent 保留；C8 同起点 8 updates，staged VJP 按相同参数/输入/noise 精确重放 student。

**结果。** 完整 40×2held-out：motor 相对原 A+Z **+1.93%**，仅 12/40 cases 改善，future NMSE **−5.78%**。要求 mean/median/motorcase 数及 latent 均通过的 gate 失败，停止此 recipe；没有运行条件性的 reader-MSE ablation，也没有参数 sweep 或闭环。

**意义。** Reader 有梯度不保证真实 deployed action 变好；不能把辅助 reader 分数当最终 policy 指标。后续 C80 与 R128 matched 版本是不同配方，须分别解释。

证据：[README.zh.md](../idea-validation/fastwam-vq-c8-frozen-reader-kl/README.zh.md)、[结论](../idea-validation/fastwam-vq-c8-frozen-reader-kl/results/CONCLUSIONS.zh.md)。

## fastwam-vq-c80-frozen-reader-kl

**设计与修改范围。** 原计划 C80/80 updates A+Z 对 reader-KL，但用户要求先分析已完成 KL 后结束；匹配 A+Z 在 2 updates 后取消，双组 aggregate 未执行。KL 完成 80 updates、160student+160reader 端点，另 CPU 读取缓存，无新 forward。

**结果。** 相对历史 C8 A+Z，held-out motor **0.04444355（+0.33%）**、latent **0.05436114（−10.70%）**，motor20/40 改善，median 仍退化。共同初始化→final 的 calibration reader KL **0.02748146→0.03080744（+12.10%）**，辅助目标本身没有成功下降。Student 与 BF16 reader 的 MSE 差项 0.00144086，大于 future 经 reader 误差 0.00054784；仅优化 teacher-readerfuture 并不能消除量化下游差异。

**结论与接手。** 源 KL`25913982`完整 F/0；A+Z`25913984`取消 F/143；CPU 分析`25920011`完成。缺少 matched C80 control，原同预算 gate 不可评定，不能用历史 C8 冒充。原版停止；后来冻结 R128、batch10、同起点的 loss-pair 另做了干净对照并得到 208/280，详见下一部分。

证据：[README.zh.md](../idea-validation/fastwam-vq-c80-frozen-reader-kl/README.zh.md)、[CONCLUSIONS.zh.md](../idea-validation/fastwam-vq-c80-frozen-reader-kl/results/CONCLUSIONS.zh.md)。
