# CEM-Update PTQ 最小机制筛选 v1

2026-09-10。新实验；不覆盖旧 RankCal。阶段 A 本地 CPU 离线分析；阶段 B 先采集与计时，再冻结实际搜索预算；阶段 C 仅在 B 的机制门槛通过后冻结并执行。

## 假设与主要对照

研究假设：在相同 joint search 下，保护实际 CEM 中间 mean/std 更新，比只保护 mean、scalar score 或全候选 ranking 更能保留后续输出 mean。以上是待检验假设。旧 24 test 已参与提出假设，不能作为新独立 test。

四个目标 CEM-Update、MeanOnly、ScoreError、Rank 使用相同旧 ScoreError mapping 起点、同 family swap 邻域（35）、最多两轮 best improvement，每方法包括起点最多 71 配置评价。所有目标共享完整配置 score cache，不能由 site-only effects 合成联合损失。每次完整恢复 FP32，再应用所有 eligible blocks 的 W4/W8。

12 encoder / 6 predictor blocks；W8 配额 3/2。重新核对每 family 的权重与 scale 成本，固定 logical bytes，不作 native peak-memory matching 或 kernel 加速声明。数值 emulation，FP32 activations/reference，RTN 与旧实验一致。

## 数据、语义与聚合：新结果产生前固定

- 新 CAL：4 episodes，dataset indices 42–45，env namespace 600000，CEM seed 为 namespace+10000+local_index。
- 新 DEV：4 episodes，indices 46–49，env namespace 700000，CEM seed 同规则。
- 每例只使用第一个实际 MPC point 的 CEM iterations 1/5，各 split 固定 8 pools。采集仍复用旧 FP32 完整轨迹 adapter；这些 reference success 不代表新方法成功率。
- 旧 targets indices 0–41 排除，并验证已有 40 个 cal/dev/test fingerprints 无重叠。新 CAL 与 DEV 目标也去重。
- CEM5 / 300 candidates / 30 elites / H5，var_scale=1；首个 MPC point 从 mu=0、sigma=1 开始，warm start 只传递 mu，不复用上一 MPC 的 sigma。stable argsort，exact ties 按 candidate index。std correction=1，candidate[0]=mu，无新增 clipping/floor；沿用 normalization，s=1。mu_before/sigma_before 保存在采集 job 每例 pools.npz 中。
- L_update=L_mu+L_sigma 用于 iteration1，iteration5 的 L_update=L_mu。分别保留三项；point/episode 等权，不用 pools 或 candidate pairs 冒充独立样本。
- 接受改善阈值 absolute 1e-10；同损失采用 lexicographic mapping 顺序，不能用 DEV 解 tie 或选配置。
- DEV 的 two-step replay 只用 iteration1 pools：第一步同池；第二步使用各自更新的 mu/sigma 与共享固定 CPU torch.randn noise（seed namespace 810000+episode local_index），candidate[0]=各自 mu。第二步 pool 可以不同，分别通过 reference/quantized model，比较最终 mu。没有环境动作执行，不是 closed-loop success。

## 资源门槛与停止规则

A：旧数据若缺 candidate actions/scores、reference self 不为零、指标不可测/退化，停止 GPU 路径。旧8maps仅探索，不拟合 loss weights。

B preflight 上限：CPU-only runtime preparation 4 CPUs / 20GB / 15min；新 reference pools 1 A100 / 15min；计时 1 A100 / 15min。总 preflight 最多 0.5 GPU-hours + 1 CPU-hour（按实际分配时间报告）。没有下载依赖计划，复用旧 runtime tar，在 CPU allocation 解包。

搜索：先测模型加载与一个完整 mixed configuration 在少量 CAL pools 上的耗时，再冻结总资源；搜索最多 2 GPU-hours（1 A100），超出保守预算即不启动或降低已声明算法轮数并重新冻结，不能偷偷减少 candidates。实际配置数可因各目标路径不同而变化，均受相同 71-config 上限约束，并报告 cache 命中、forwards 与用量。

实测后预算冻结：计时 job 17042287.pbs101 成功，每 pool 约 1.243 秒、加载约 93 秒。维持两轮；MAX_SECONDS=4200，内部全流程 deadline=4170 秒，PBS walltime=4260 秒（1 A100，最多 1.1833 GPU-hours）。预算含 CAL 搜索、DEV scoring、所有 alias 的 two-step calls、模型加载、指标和 I/O 余量；详见 BUDGET_FREEZE.json。预算耗尽不解释为研究 no-go。

B 为小样本机制筛选。进入 C 的预声明必要条件：

1. 工程验证通过，完整 mapping 被 CAL 搜索稳定改善；FP32 replay/restore 不一致属于工程失败。
2. CEM-Update 与 MeanOnly 必须得到不同 mapping，才能支持本配方 sigma 的增量价值；相同 mapping 则本次停止，不追加起点或 loss weight。
3. 相对共享起点，DEV 平均 L_update 和 two-step final-mu MSE 均至少改善 5%，各至少在 3/4 episodes 上改善（每例 loss delta < -1e-10）。
4. 相对同搜索 MeanOnly、ScoreError、Rank，DEV L_update 与 two-step final-mu MSE 都至少降低 5%；同时 L_mu(Update) <= 1.05 * L_mu(MeanOnly) + 1e-10，容许为 sigma fidelity 付出少量 mean 代价。这是偏保守的资源决策阈值，有 false no-go 风险，不是显著性检验或论文有效性标准。

冻结说明：上述 L_mu 的 5% non-inferiority allowance 来自独立设计审查，修订发生在任何新量化配置的 CAL/DEV 评分产生前；初稿的近似 exact no-regression 1e-10 gate 已被替代。修改不依据新结果。Loss 改善 tolerance 与搜索规则不变。

任一必要条件不满足，则记录 weak/no signal 并停止当前配方，不自动扩展到 C。若通过，先依据实测给出新的闭环确认资源估计并冻结 targets 与完整方案，再执行小型 C；未执行 C 前不能声称 idea 已在闭环 work。

## ASPIRE2A 执行与复用边界

旧 screen.pbs 只有间接 PBS 变量约束，没有完整 hostname / allocation 验证；本轮禁用其直接启动路径，使用新 gpu_stage.pbs 与 allocation_guard.py。旧 remote.py 的 bulk transfer/任意命令入口不用于新实验；新 remote_control.py 仅允许 PBS 操作和每文件不超过64KiB的小型控制/结果文件。

实际 hostname 必须非 login 且属于 PBS_NODEFILE，qstat 确认本 job 处于 R 且 exec_host 匹配，才开始重 I/O 或模型加载。不伪造环境变量。所有 runtime 解压、模型和分数计算、完整结果校验在获批 compute allocation。完整 raw arrays 留在 compute 产物目录，小型 JSON 轻量回传；不得在 login 打包/批量下载结果。compute 失败不退回 login。不触碰其他任务进程，不扫描大型目录，不输出 credentials，保留 host-key verification。
