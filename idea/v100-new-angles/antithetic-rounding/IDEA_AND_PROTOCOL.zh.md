# Antithetic Rounding Pairs：先检验误差配对能否穿过 world-model rollout

状态：待独立审查与最小实验；2026-09-13。Adapted ResearchStudio-Idea run，省略 Phase 4，不声称 canonical DONE 或 novelty 已认证。

## 切面与已有证据

此前工作优化单个 quantizer 或 mixed-bit map 的损失；本候选改变两份冻结量化模型的 **joint rounding distribution**，不训练、不搜索 bits、不复用旧 DEV 挑选方法。

最接近前作是 [Ex Uno Pluria / LPE-BSR, NeurIPS 2024](https://arxiv.org/html/2411.14860v1)，§3–4 已给出从一份 pretrained model 用 Bernoulli stochastic rounding 构造低比特 ensemble。低比特 ensemble 和无训练生成多成员本身均非新贡献。[SPEQ, AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16839) 已用 stochastic precision ensemble 做 distillation，处理的是另一种训练设置。经典 antithetic variates 是标准 variance reduction，不能声称新的统计原理。

[QuantWM](https://arxiv.org/html/2602.02110v1) 已报告低比特 rollout 与 planning objective 的失配；[Where Bits Matter](https://arxiv.org/abs/2602.11882) 已报告 planner-budget 条件性。这里未知的狭窄问题是：权重层面的反相关 rounding 误差是否会穿过冻结 action-conditioned nonlinear dynamics，使两个 **score predictions 的平均**更适合选择动作，而非只降低 weight MSE。

## 精确定义与前提

冻结 DINO-WM Wall epoch65；encoder 与非 predictor Linear 参数保持 FP32。对 24 个 predictor Linear 使用 symmetric per-output-channel W4，qmax=7，scale=absmax/7；全零行设 scale=1。对 v=w/scale，p=v-floor(v)，u 为逐权重独立 Uniform[0,1)。第一份整数为 floor(v)+1[u<p]，配对第二份用 1-u。独立 negative control 的第二份使用独立 Uniform[0,1) 变量 u_ind，v、p和scale完全相同；两个 arm 共用第一份模型、目标、候选、bit grid、前向次数、三个预先固定 rounding seeds。

每个 replica 完成整条 H=5 rollout 后计算原始 objective；最后平均两份 scores为bar{s}，按bar{s}从小到大取top30，exact tie按原candidate index从小到大。FP reference与所有baselines同样tie-break。不在中间 step 平均 latent，不平均 weights，不中途改变模型。

有限精度RNG的执行定义：使用24-bit midpoint grid，u=(k+0.5)/2^24，k均匀取0..2^24-1；u及1-u以float64表示并比较，保证二者在同一个离散网格，避免[0,1)与(0,1]的端点差。v/p和最终整数、dequant保持FP32。独立u_ind使用同一离散分布。连续Uniform的covariance公式仅作为理想化解释，实际量化误差另行测量。

对一个 scalar，两个 rounding error 的 covariance/scale² 为 max(0,2p-1)-p²，即 -min(p²,(1-p)²)。这仅证明 scalar 的反相关；网络 nonlinearities、共享参数重复使用、rollout 和 elite selection 均不保证保留该优势。该传播前提是本次 falsification target。对于 p=0 的已在 grid 上权重，两份均保持不变；clip 只保护数值边界，不用于缩小不可解释的大误差。

Observation model：6 个新的 Wall validation source episodes，固定标准正态 action pools；它代表初始 CEM 的候选分布，不能代表迭代收缩后的全部 CEM 分布或真实控制成功。FP32 是明确的 fidelity reference，绝非环境 oracle。本轮不对环境正确性作结论。

## Pattern 来源与适用边界

读取 IdeaSpark `ideation-patterns/overview.md` 和实际 `ideation-sub-patterns/C02.md`，使用 `controlled_diagnostic_design`：把 **多一次 forward 的收益**与 **配对方式的收益**分开，保持相同两次 forward，仅替换第二份 rounding random variable 的耦合。选择单一 pattern 是因为这是最小前提诊断，不为凑 composition 增加组件。

C02 的拒绝约束直接保留：小样本单任务不得支持广泛 robotics claim，FP fidelity 不得冒充真实成功，已知 antithetic scalar 效应本身不是新发现。如果只得到 weight-space 的预期负 covariance，而没有 downstream selection 改善，直接 no-go。

## 冻结最小设计

- 新 source dataset indices 96–101；env namespace 930000、candidate namespace 940000；记录真实 underlying episode 映射和 initial fingerprint，验证六个样本互异且不与已有 registry 重叠。未覆盖历史状态需标明。执行前检查PRR manifest发现84–95属于其test_locked保留区，故在任何推理/数据采集前更正原草案84–89；历史及保留0–95均不打开作本轮评估。
- 每 episode 一组 300 条 normalized action sequences，H=5，action_dim=10。每组原始数组冻结并保留；不根据结果替换候选。
- rounding seeds 1301、1302、1303。每 seed 的 marginal random construction 相同；依序交错 independent / antithetic 第二成员执行次序。3 seeds 是同六个 episodes 的重复测量，非18独立 episodes。
- 基线：FP32 reference、单 RTN W4、独立双 SR，以及单 RTN W8的成本背景对照。独立双 SR 是去除 coupling 的负对照，计算预算与 antithetic 双 SR 相同。W8在提交前加入，避免忽略两份W4约等于一份W8的权重预算；只增加六个固定pool评分，不调参。
- Primary：one-shot shortlist regret，R=mean(s_FP[selected top30])-mean(s_FP[FP top30])，理论非负，越低越好。按 episode 先平均三个 seeds，再跨六个 episodes 平均。它不是执行elite mean后的真实cost或success。
- Secondary：elite mean action 与 FP elite mean 的 MSE、score NMSE；诊断 pair score-error covariance 与 weight-error covariance。
- 两份模型的 forward count 相同；只常驻一份 model，逐次精确 restore / apply。两份 W4 存储与单 W8、重复计算成本不同，不能从本轮推断总体压缩收益。

## 事前经济性决策（非显著性门槛）

工程 gate：allocation / GPU / source / quant grid / snapshot restore / finite values /完整6 episodes ×3 seeds 均通过。否则记 implementation_failure 或 inconclusive，不能作 mechanism no-go。

若 antithetic 相对 independent 的平均 primary regret 改善至少5%（(R_ind-R_anti)/R_ind，先对seed/episode聚合再求比），至少4/6 episodes的seed平均R_anti<R_ind，且至少2/3 rounding seeds的六episode平均方向改善，同时不劣于 RTN 的总体primary regret，才满足regret gate。额外必须满足elite-mean action MSE的总体平均不劣于independent与RTN（浮点比较容差1e-12，非容许退化比例）。二者都通过才记preliminary_go，防止shortlist代理改善掩盖action失真。5% 和覆盖数是本轮明确选择的经济性 screen 门槛，不是来源论文结论、统计显著性或数学必然。若基线 regret 近零（<=1e-12），无实际改善空间，记 no-go for expansion。其余工程通过但不达标记 mechanism_no_go。Secondary 不用于事后替换 primary。

若单W8在regret与elite-mean action MSE均优于双W4，且需要的logical tensor bytes与forward数不更多，则另记practical_no_go：即使pairing有机制信号，也不支持经济性扩跑。两份W4的scales等开销如与单W8不同，应据实列出。

所有结果无论 go/no-go 均停止此候选，不追加 full CEM 或闭环 suite。只有具体实现缺陷被定位并修复时才重跑相同实验；不得通过增大 ensemble、改 seeds 或调 bitwidth 挽救失败。

## 执行边界

CCDS SLURM 单 V100 32GB；首 job <=45 min，runner workload deadline2400s。复用 `cem_update_ccds` 的已验证环境、官方模型和数据。所有 model I/O、hash、数值计算在 compute allocation 内，必须检查真实 hostname 与 scontrol job ownership / NodeList。禁止 login/head 重操作与 credentials 输出。大数组只在远端保留，下载小 JSON 结论。

本轮为 FP32 operators 上的 W4 numerical emulation；不会声称 native kernels、真实部署加速、环境成功率或完整研究验证。
