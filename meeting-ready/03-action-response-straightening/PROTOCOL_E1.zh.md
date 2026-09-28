# E1：Frozen LeWM 局部动作响应试测

状态：development-only pilot。首轮 `25536940.pbs101` 已完成；结果必须按下方修订解释，不将原提案的假说当作事实。

## 首轮结果后的预定尺度修订（2026-09-24）

首轮真实 CEM 提案在展平 50 维动作坐标中的非中心 L2 距离，late round 的 p10 仍约 1.2–1.4，early round 的 p10 约 6.1–6.2。初选 `0.05/0.10/0.20` 半径的每个球内仅有 candidate 0，因此没有可估计的真实提案 Taylor residual。追加 `1/2/4/8`，同时保留 `0.05/0.20` 作为局部数学尺度；这仅是同两颗 development seeds 的尺度校准，不是 final test 调参。

Pinned CEM 的候选在 solver 坐标中可超出环境声明的 `[-1,1]` Box，且未找到实际用于 proposal 的 solver clipping bound。中心有限差分因此在未裁剪的 solver proposal 坐标上保持对称；不能把环境 Box 冒充 solver bound。首轮检查程序将“无显式 solver bound”误判为“Jacobian 无效”，使所有 Taylor 指标暂不可解释。修复后先验证 finite-difference JVP，再解释同一组 seeds 的探索性误差。若 Jacobian 仍不通过，仍保留无效结论。

## 问题与比较

在同一个 PushT context 下，以原始 LeWM 的完整五步 rollout 得到 terminal latent `F(c,v)=z_H`。在 CEM 实际访问的动作候选处，比较 full `F(c,v)` 与参考动作 `v̄` 的一阶近似 `F(c,v̄)+A(v-v̄)`，其中 `A=∂F/∂v|v̄`。goal cost 使用相同 checkpoint 的 official criterion。此阶段不训练、不替换 CEM driver、不作任务成功或速度提升声明。

## 固定 pilot

- PushT reset seeds：`4101,4102`；两者及其全部派生 contexts 固定为 **development-only pilot**，不得进入 final test。每个 context 运行原始完整 CEM 一次，保持 300 candidates、30 iterations、top-30 和官方 policy/solver 配置，记录 CEM RNG seed。
- 仅截取第 `1,15,30` 轮的真实候选及生成该轮候选的 `prev_mean/prev_var`；用 candidate 0 与 `prev_mean` 的运行时关系检查轮次对应，不用评分后的 mean 冒充 reference。不以数据集随机 bank 代替。候选优化张量预期为 `[1,300,5,10]`，展平后 `D=50`，但须在 compute node 上核对实际 bounds、clipping 和优化坐标。
- 先记录实际候选到 reference 的 L2 距离分布、合法动作 box，以及按阶段的覆盖率。半径统一指 **展平后动作向量的 L2 距离**。`0.05/0.10/0.20` 只作首轮候选半径；若远小于实际搜索尺度，不能据此声称方法不可行或可行。球内覆盖率与按实际距离分箱的误差应分开报告。
- 每轮对完整 300 candidates 计算 exact 与 linearized cost 和 elite/top-30 差异；Taylor latent residual 每半径最多记录 16 个真实候选。抽样规则、种子、选中 candidate ID 和实际距离必须记录；若只报告球内均值，须明说它不是半径附近的壳层误差。此 pilot 暂不作接触模式分组，该分组属于后续 E5。若抽样不足以估计某一阶段的误差分布，报告覆盖与不确定性，不外推。

## 必需结果

每个 context×CEM 阶段至少报告：实际距离分位数、每半径候选覆盖率、terminal latent 绝对／相对 Taylor residual、action-induced 真实 latent 变化、official cost error、top-30 overlap、第 30/31 名 cost gap、参考 forward/Jacobian/全批 surrogate 与 exact scoring 的时间。exact cost 来自本地锁定 checkpoint 的 native `JEPA.get_cost`；surrogate 使用同一 context、goal 与 `criterion`。真实 latent 变化接近零时，relative residual 可能失稳，须同时给绝对值。计时需 GPU 同步，并将 warm-up 与稳态分开；此 pilot 的局部计时不是完整 solve speedup。保留少量诊断值即可，不落盘整批图像或大 tensor。

在小批动作方向上用中心有限差分检查 autodiff JVP；扰动必须合法且对称，不能将分别 clip 后的点当成中心二阶差分。模型为 `eval()`，参数可冻结；求动作导数时不能使用阻断该路径的 `no_grad`/`inference_mode`，也不能在 imagined rollout 中途 detach。

## Pilot 决策

本轮先判断接口是否正确、Jacobian 是否具有可承担的成本、early/middle/late 的局部范围究竟有多大。两个 reset seeds 不足以得出 GO。扩展开发集的最低条件是：真实候选与 callback reference 对齐、finite-difference JVP 未显示断图、exact replay cost 与 CEM cost 一致；之后依据实际 proposal 距离、elite 误差及 `Jacobian+surrogate` 对 `300-candidate exact` 的耗时比，决定是否值得继续直接 surrogate-CEM 路线。若比值明显无益，记录该路线的成本 NO-GO；仍可独立评估训练能否改善局部几何。扩展前冻结新开发／最终测试 seeds；既有 8 个 held-out contexts 与 4 个 planner cases 只能作开发／回归检查。E2 matched training 和后续 adaptive CEM、paired 闭环保持为独立 gate。

## 集群执行

checkpoint、simulator、模型 rollout、Jacobian、计时都仅在获批 PBS GPU compute allocation 内运行。脚本先检查真实 `PBS_JOBID`、非 login hostname 和 GPU allocation；GPU utilization/VRAM 每 30 秒写入 job log。Login node 仅用于小型控制文件传送、提交与状态查询。不下载、不安装、不做额外 hash。
