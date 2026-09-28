# LeWM Goal-query 初步验证

冻结日期：2026-09-26；状态：执行前冻结。研究对象是指定 teacher terminal query 的廉价近似，不是新 Markov world state，也不是 novelty 验证。

## 问题和最小对照

固定当前 LeWM＋PushT teacher、encoder、H=5 和原平方欧氏 terminal cost。四个训练臂：direct scalar、direct scalar 加训练期 projected/tail 辅助监督、projected＋goal-dependent tail（r=32）、small full-latent predictor。前两个区分结构收益与额外监督收益；full-latent 臂也得到同一 teacher-cost 监督。projected-only 用同一 trained projected 权重移除 tail，是诊断而不是第五次训练。

当前远程stage是没有.git的源码archive。FREEZE中的commit是此前冻结baseline的声明provenance，不能声称本次重新验证了HEAD；报告实际source/checkpoint路径及runtime interface和原cost检查。无需为缺少.git扩大hash或完整性审计。

共同 goal-free context/action trunk 为两层 h256 MLP。只用可信当前 latent 与完整 action sequence，直接预测 terminal query；不把预测输出递归当状态。scalar/tail 使用相同 goal-conditioned head。真实参数量和 latency 必须报告，不假装输出维数等于计算量。训练期辅助头不进入 scalar_aux 的部署计时。

损失细节在FREEZE.training.loss_exact唯一冻结：每context-goal bank先沿candidate维度去均值；teacher bank variance使用population mean squared deviation。variance floor为所有训练标签population variance乘1e-6（下限1e-8）；absolute项权重0.1，y/tail/full latent辅助项各权重1。y/full latent MSE分别除以训练terminal标签的平均coordinate population variance。全部归一化量train-only。

每条 teacher terminal latent 只生成一次，所有臂共享 context/action/goal tuples 和1500次更新。使用两个固定 initialization seeds、同一 context schedule，不根据结果选择 checkpoint、rank、loss 或种子。训练目标同时看 candidate-centered cost differences 与低权重 absolute cost；尺度、projection 和 variance floor 只由训练 split 获得。

## 数据边界

复用 job25223859 的512-context anchor-aligned training bank；不是旧student权重，也不把旧结果当新模型证据。每context有64动作候选。目标为own logged goal加同split三个不同parent episode的goal donors，不使用每个候选自身teacher terminal作为goal。跨episode donor goals只用于teacher查询泛化，不能假设物理可达。

新dev为复现selection_seed=20300903排序后的valid[800:808]，新test为valid[824:832]。各8parent episodes，early/middle/late三anchors、两个fresh action seeds、300候选，4goals。必须验证train/dev/test parent episodes及goal来源分离；不足832valid episodes时报告INCONCLUSIVE，不能事后换slice。冻结teacher已有全数据预处理统计沿用，因此仅声称episode/goal来源隔离，不声称全部scaler untouched。

early/middle/late是episode内observation anchors，不是CEM iteration阶段；当前候选来自冻结的logged-action Gaussian bank，尚未覆盖adaptive或method-induced CEM proposals。dev/test仅对本run隔离，不声称全项目所有旧实验均未接触这些episodes。

P用训练teacher terminal targets的PCA拟合并正交化，r32固定。先检查原criterion与terminal平方距离一致、exact projected＋exact tail恢复原cost。失败先修工程问题，不能当研究NO-GO。

## 证据与停止规则

第一层：固定bank的去均值cost error、Spearman、top30、argmin、teacher elite regret、30/31margin和真实elite proposal mean/std漂移，按goal类型、anchor与parent episode聚合。候选是嵌套测量，不作为独立样本。

第二层：exact y＋learned tail隔离tail学习信号；它付了teacher成本，不能计为加速。与tail-zero比较。另检查fully learned projected＋tail相对scalar_aux是否有独立优势。严格区分tail signal、structural signal与deployable predictor gate。

第三层：cached context/goal＋300动作到最终scores的GPU原生计时，每个split按冻结构造顺序取前六个不同row的action-bank0与own goal作为六个固定blocks、各臂warmup3次、平衡随机顺序重复10次、前后CUDA同步。dev用于选择，test单独测量用于final gate；主时钟是含前后CUDA同步的CPU perf_counter，包含Python与launch开销。teacher复用缓存的输入latent/goal，调用官方五步predict和criterion；未应用within-rollout iteration cache，因此这里只是初步predictor对照，不能声称胜过最强cached planner。shadow/oracle调用不进入部署计时。固定active_contexts=1，报告p50/p95及参数量。此结果不覆盖encoder、CEM或闭环速度。

dev先选一个符合quality门的deployable arm，按top30、Spearman、latency排序，选择写入小文件后才看final test。两个训练seeds都需达到median Spearman≥0.95、min≥0.8、median top30≥0.75、min≥0.5，以及scoring p50比teacher降低≥20%，才进入下一层。门槛是工程筛选标准，不是理论保证。

上述predictor median/min是对所有嵌套query banks的描述性覆盖检查，minimum保留最差query约束；不是把banks当独立样本。tail/structure的配对差先按parent episode取median，再按8个episodes计算描述性6/8门；cyclic goal donors跨episode共享，episode摘要之间也非完全独立，禁止据此作显著性或population CI结论。own/donor分组结果分别报告。

若有效执行后所选臂失败，结束本次初步screen，NO-GO仅针对该冻结recipe；不扫更大模型、不延长训练、不换goals。若仅scalar通过，承认cost-only路线，不能宣称projected结构有贡献。若过门，再执行bounded adaptive CEM：保持official300/30/top30、argsort、unbiased std、candidate-zero及paired RNG语义，检查teacher-shadow、proposal/first-action漂移和完整solve计时；过门后才考虑paired closed-loop。本次screen没有闭环成功结论。

## 执行与监控

模型加载、数据读取、PCA、标签、训练、自检和计时全部在真实PBS compute allocation。入口检查PBS_JOBID、非login hostname、hostname属于PBS_NODEFILE、可用GPU。GPU利用率和显存每15秒写job.log。2小时allocation、6900秒自终止，不安装新环境、不hash、不用banked reset。

只上传小控制源码，取回小结果报告；checkpoint和rows留在compute storage。监控使用gpt-6-luna、每轮一次只读qstat；Q/R无实质变化保持安静。终态包含job ID、Exit_status、wrapper状态、gate、summary与证据路径，通知原任务恢复验收。监控不能改作业、重提交或宣布科学完成。

实验设计使用experimental-design skill的parent-episode blocking与nested-measurement原则；软件参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents，https://doi.org/10.48550/arXiv.2609.00065（执行前补核当前v2元数据，引用稳定DOI）。当前pilot不作powered显著性推断。
