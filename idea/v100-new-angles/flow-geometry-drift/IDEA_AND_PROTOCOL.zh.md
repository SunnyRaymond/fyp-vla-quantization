# Flow Geometry under PTQ：离线 action drift 排序诊断

2026-09-13，候选协议草案，尚未执行或采集数据。使用裁剪ResearchStudio-Idea：直接先例、C02受控诊断、独立审查和最小可证伪screen。

## 切面与已有证据

不训练新的quantization loss或修复模型，转而检验低比特VLA的**可观测性**：当PTQ改变action时，策略本次已计算出的flow velocities是否足以给出漂移提示？这与旧FRT/PRR的误差训练与恢复是不同问题。

直接复用 [The Geometry of Flow-Matching Uncertainty v3](https://arxiv.org/html/2607.27933v3) 的accel指标，绝不主张发明该proxy。该工作把速度变化用于uncertainty/failure detection，并已讨论自信但错误动作的盲区。待检验的窄问题是：数值量化造成的drift是否能被同一信号捕捉，及量化language backbone或action expert时是否不同。当前有限检索未在该全文找到quantization处理，不等于检索完备或novelty认证。

补充检索已经发现用trajectory curvature生成PTQ校准数据的直接先例，见[补充先例](../../../experiment/idea-validation/v100-new-angles/flow-geometry-drift/ADDITIONAL_PRIOR.zh.md)。本实验的问题限于Q-only几何信号排序drift及两个对照，不能将“curvature用于quantization”概括为新贡献。

## 拟冻结干预与资源

- 模型为官方 [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero)：公开907MB checkpoint，continuous action 7维、内部padding32维、chunk50、10个denoising steps。预先冻结model commit `31d453f7edd78c839a8bbc39744a292686daf0de`，dataset `lerobot/libero` commit `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`（含episode indexing修复）；其余processor/base config/source与实际下载hash由CPU准备记录。model card不足以证明此权重的完整LIBERO success，本实验亦不复现成功率。
- 参考为同一checkpoint数值在V100用FP32执行；若保存权重原为BF16，转换FP32不会恢复此前丢失的信息，也不冒充原始FP32训练权重。不把本screen等同作者BF16配置结果。关闭compile/RTC、eval、固定explicit initial noise。先验证完整state_dict加载、dtype、完整10steps、hook不改变输出，再看数据。
- Arms：FP32、language-backbone-only W4 RTN、action-expert-only W4 RTN。W4均signed[-7,7]、Linear per-output absmax scale、zero-scale=1、round-to-nearest-even，实际eligible module名单与参数数冻结。vision、embedding、norm与其他模块FP32；不是全模型W4。
- 无CAL优化。选样规则在读取任何模型输出前固定：metadata按task_index升序前4个task，各按episode_index升序取前3个不同episode；frame取floor(episode length/4)。12个episode各使用noise seeds1701/1702，所有arm共享完全相同explicit noise。按episode平均重复noise，再做rank比较。metadata字段不匹配时停止查明，不能按error挑选episode或task；演示数据不冒充task-generalization held-out test。
- 记录10步完整velocity与最终action；proxy固定前8个steps、前8个chunk positions、前7个physical action dimensions。accel=8×相邻velocity差的L2 norm之和/velocity L2 norm之和，分母<=1e-12记degenerate。仅用Q数据的读数是候选detector；FP读数仅作为离线对照。
- Target drift：相同observation/noise下，Q与FP前8个action位置、7个维度在模型normalized action space的MSE。只判断FP fidelity，不当作环境失败或物理危险标签。
- Baselines：FP accel控制固有任务/flow难度；Q action norm为同一normalized前8×7 action slice的Frobenius norm，控制简单输出幅值。两种PTQ locus独立报告，不混合成24个独立episodes。保存每noise读数并报告其差异，不只保留平均。

## 草案停止规则

对每个PTQ locus，先每episode平均两个noise的proxy与MSE，再计算12个episode的Spearman rho。Q accel须rho>=0.5，且分别超过FP accel与Q action norm的rho至少0.1，才为该locus preliminary_go；否则mechanism_no_go。若error或proxy近常数、存在不完整样本、engineer gate失败则分别记no_binding_locus或implementation_failure/inconclusive，不臆造rho。两locus均通过才给总方案preliminary_go；只一处通过则明确scope-limited。阈值为经济性筛选，不是显著性。

Spearman使用平均rank处理exact ties，再对rank计算Pearson correlation；任一输入max-min<=1e-12则不定义rho，记no_binding_locus。比较边界统一使用>=，只允许1e-12数值容差，不对连续数值先舍入造ties。accel严格使用raw velocity而不是Euler increments；SmolVLA reverse-time dt=-0.1保留原solver。本screen没有threshold或lead-time验证，所以结果仅为offline drift-ranking diagnostic，不能称online预警detector。7个physical indices由官方action_feature与unnormalizer核实，包含gripper但不额外binarize，以免把representation处理混进PTQ干预。

本方案不校准报警threshold、不实施fallback、不重跑完整rollout，不声称安全保证或新uncertainty算法。无论结果均到此为止；不得按DEV改prefix、bitwidth、metric、frame或noise seed。具体实现错误才允许同协议修复。

CPU准备与GPU各独立bounded SLURM allocation，GPU一次<=45min；不训练。所有模型/data下载、hash、依赖安装、推理和统计仅真实compute node；head只小控制、提交和小结果读取。无需ASPIRE2A连接，保留host-key与凭证边界。
