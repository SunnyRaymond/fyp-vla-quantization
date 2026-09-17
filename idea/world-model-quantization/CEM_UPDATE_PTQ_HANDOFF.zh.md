# CEM-Update PTQ：从 RankCal 修改后的 idea 与实验交接

日期：2026-09-10。状态：用户同意优先探索；本文件是新假设与建议的验证方案，不是已完成实验或已经冻结的 protocol。本轮仅写文档，没有提交作业、下载模型、训练或重新运行 IdeaSpark。

## 1. 给接手对话的任务

先做最低成本的机制筛选，判断“保留 CEM 更新”是否值得替代 RankCal 的全候选排序目标。不要重新运行完整 IdeaSpark pipeline，也不要直接扩大旧 RankCal 的测试集。先读取本文件和旧 screening 证据，核对可复用数据，再给出分阶段资源估计与冻结的实验 manifest。

主要文件（相对于本目录）：

- `../../experiment/idea-validation/world-model-quantization/dino-wm-wall/SCREEN_RESULTS.zh.md`：实际 RankCal screening 结果，首要证据。
- `../../experiment/idea-validation/world-model-quantization/dino-wm-wall/TEST_FREEZE.json`：旧配置、targets 与 seeds。
- `../../experiment/idea-validation/world-model-quantization/dino-wm-wall/EXPERIMENT_PROTOCOL.zh.md`、`FAST_SCREEN.zh.md`：旧协议及门槛。
- `../../experiment/idea-validation/world-model-quantization/dino-wm-wall/artifacts/screen/joint_verification.json`：旧联合配置的指标核验。
- `../../experiment/idea-validation/world-model-quantization/dino-wm-wall/artifacts/screen/artifacts/`：归档代码、候选池相关产物、轨迹和日志；先看小型 manifest，确认文件布局，避免盲目扫描或解压大文件。
- 项目根目录 `experiment/reproduction/dino-wm-wall/RESULTS.md`：未量化 checkpoint baseline；这不是 RankCal 量化实验结果。
- 项目根目录 `experiment/reproduction/dino-wm-wall/source/planning/cem.py`：原始 CEM 语义；实际运行 adapter 还需从旧实验归档核对。

旧 `phase0` 至 `phase4` 和旧实验保持历史记录。它们描述 RankCal，不应自动当成新方法规范；不要覆盖旧 final_candidate、反证字段或旧 TEST_FREEZE。新运行建议写入 `../../experiment/idea-validation/world-model-quantization/cem-update-ptq/`。

## 2. 为什么修改：已有证据与推测分开

旧 screening：FP32 22/24、all-W4 16/24、RankCal 19/24、LocalMSE 15/24、ScoreError 19/24、Random1 20/24、Random2 17/24。样本小，不能把未显著解释为等效，也不能据两个 random maps 断言随机分配普遍更强。

Development 联合配置中，RankCal 的全候选 E2 disagreement 为 25.35%，略优于 LocalMSE 的 25.92%；但 elite overlap 为 42.40% 对 43.75%，elite-mean action MSE 为 0.044385 对 0.043846，没有同时领先。相对 ScoreError，离线指标更好也没有得到 held-out success 净提升。

**已有证据：** 全候选排序改善未同步转成 elite/action 与闭环优势，当前方案为 weak/mixed signal，旧 screening 的资源决定是 no-go。

**新假设：** CEM 关心由 elites 决定的下一轮搜索分布，而非全部候选的两两顺序；量化可能主要通过改变这个更新过程造成决策偏移。现有数据没有证明新目标有效，也没有证明单点影响不可加。

## 3. 相对 RankCal 改了什么

| 项目 | 旧 RankCal screening | 新 CEM-Update PTQ |
|---|---|---|
| 要保留的对象 | 全候选 pairwise ordering | CEM 更新后的 mean 与 standard deviation；最后一轮主要保留输出 mean |
| calibration 信号 | FP32-background 的 site-only W4/W8 排序差异 | 完整 mixed-bit 配置实际产生的 CEM update 偏差 |
| allocation | 单点收益及固定 family 配额 | 同 family 的 W4/W8 block swap，每次直接评价完整配置 |
| 重要对照 | LocalMSE、ScoreError、random maps | 同搜索器/预算的 ScoreError、mean-only、rank objective；旧 maps 作为历史基线 |
| 新增推断风险 | — | update fidelity 仍可能无法迁移为闭环收益；搜索成本可能高于收益 |

仍保留：DINO-WM Wall、既有 checkpoint、固定 planner、weight-only RTN、固定 W8 配额、无需训练 backbone。属于 PTQ。部署时不需要 reference model；校准成本必须单独报告。

这是一次新研究假设，不是修饰旧实验结果，也不是替旧 RankCal 宣称成功。

## 4. 最小机制定义

固定一个观测/目标、planner state、CEM iteration，以及同一组候选 action sequences U。reference 和量化模型各自打分，按同一排序/tie 规则选 K 个 elites：E_F 与 E_Q。不得让两个模型在该单步 probe 中各自生成不同候选池。

按实际 CEM 实现计算：

```text
mu_F = mean(U[E_F], dim=candidate)
mu_Q = mean(U[E_Q], dim=candidate)
sigma_F = std(U[E_F], dim=candidate)
sigma_Q = std(U[E_Q], dim=candidate)

L_mu    = mean(((mu_Q - mu_F) / s)^2)
L_sigma = mean(((sigma_Q - sigma_F) / s)^2)
L_update = L_mu + w_iter * L_sigma
```

`s` 是每个 action coordinate 的固定正尺度，所有方法共用；优先用已有模型 action normalization/已声明坐标单位，不能按每个候选配置或 test 结果重估。若无合适定义，在新 CAL 上确定并冻结。均值覆盖 horizon × action coordinates。

`w_iter=1` 用于仍有下一次采样的 CEM iteration；最后一轮设为 0，因为旧实现返回 mu，最后的 sigma 不用于后续采样。若 warm-start adapter 实际复用 sigma，应先核对代码并记录协议调整。这是比此前讨论更具体的实现约定，避免校准一个未被使用的量。

当 w_iter=1 时，该目标是动作坐标归一化后的 diagonal Gaussian squared Wasserstein distance 的维度平均版本。这个数学形式不是新贡献，也不意味着 Gaussian 近似反映真实任务风险。非最后一轮的 sigma 描述搜索范围，不能只看 elite set overlap。

实现细节必须核对：

- 原代码是 `topk_action.std(dim=0)`；复刻实际 PyTorch correction 语义，不擅自改 population std，不添加未声明的 variance floor 或 clipping。
- 原代码用 `torch.argsort(loss)`；旧 probe 提及 stable top-K。两者的 exact-tie 行为不能假定一致。比较实际 adapter、reference 与 probe，使用同一明确规则；统计 ties，有差异时披露。
- 采样中的 `action[0]=mu`、warm start、early stopping、动作执行/反归一化均需保留实际 screening 语义。
- 分别报告 L_mu、L_sigma 与 L_update，防止合成指标掩盖某一项退化。
- 沿用 pool → MPC point → episode 等权聚合，不把成千上万 candidate pairs 当独立样本。

## 5. allocation：联合配置上的有限搜索

建议首版保留旧设置：12 encoder blocks 中 3 个 W8、6 predictor blocks 中 2 个 W8，其余 eligible blocks 为 W4。FP32 activations；reference 是 FP32，不要写成原 idea 中的 FP16。

为消除初始配置选择的影响，建议所有搜索目标从同一预先指定的旧 ScoreError map 起步。这仅是共享起点，不是因为它的旧 test 结果而宣称最优；额外起点属于扩展，首轮不做。

一次邻域：在 encoder family 内交换一对 W8/W4，或在 predictor family 内交换一对。按旧配额有 3×9 + 2×4 = 35 个邻居。每个邻居必须完整应用量化并重新打分；不能由 site-only probe 合成 joint score。

建议最多两轮 best-improvement search：每轮枚举全部邻居，接受校准损失改善最大的配置；无超过预先冻结 numerical tolerance 的改善即停止，ties 使用稳定 block-ID 次序。最多 70 次邻域配置评价，每次需要多个 candidate-pool forwards，**不是 70 次 forward**。可以缓存重复 map 的结果。此上限是算法建议，不是已授权的 GPU-hour 额度或耗时承诺。

共同邻域、起点、数据、停止规则与最大评价数必须适用于 ScoreError、mean-only 和 rank-objective 对照。报告实际配置数、forward 数与 walltime，避免把更多搜索算成目标函数优势。

旧 26,817,472 bytes 是 eligible logical weights 含 scales 的成本，不是系统显存。重新核对 registry 和同 family 成本；若不再等价，不能继续用配额代表相同 bytes。首轮沿用 numerical emulation，不声称 native W4/W8 加速。

## 6. 分阶段验证与停止条件

### A. 旧数据离线检查：先不启动 GPU

检查已有产物是否保存完整 candidate actions、每种联合配置的 scores、pool/episode IDs。不能从 aggregate elite overlap 或 action MSE 反推 sigma。数据不足时明确列缺口；不要自动重跑整套。

若数据足够，重算旧完整配置的 L_mu/L_sigma/L_update，复核 FP32 对自身为零，并解释是否出现 E2 与 update fidelity 的不同排序。可结合旧成功结果作描述性检查，但旧 maps 很少，不能拟合一个“最能解释旧成功率”的权重并宣称确认。

这一阶段只检查指标是否可测、是否非退化、是否提供不同信息。没有差异则暂停重新评估；有差异也不是成功证明，不自动触发大规模作业。

### B. 小型新 calibration/development：只验证机制

先估计一个完整配置在少量 pools 上的成本，再冻结搜索上限和 CPU/GPU budget。优先只完成 CEM-Update、mean-only、ScoreError 的相同搜索；rank-objective 在资源允许时加入，以拆分相对 RankCal 的“更换目标”与“更换搜索器”两项改变。不能只对比旧 site-only RankCal 就归因于新 loss。

CAL 用于选择 mapping；独立 DEV 用于单步 update 与短 CEM 链诊断。建议一个两步 CEM replay：第一步同池，第二步让 reference/quantized 分别从各自更新分布采样，但使用相同标准正态 noise。此时第二步的 pools 合理地不同，不能误称同池比较。再检查最终 mu 偏差是否改善。该实验无须执行环境动作，不能替代闭环测试。

停止/复审条件：新目标退化、重复 reference 不一致、joint search 没有稳定改变、DEV 上更新误差改善不能传到后续 mu、或 mean-only/ScoreError 达到相同或更好的表现。小样本相近不证明等效，但可以构成不继续投入的资源理由。

### C. 只有机制值得继续时才做新的小型闭环确认

配置与门槛冻结后，再生成新的 paired episodes/targets。旧 24 held-out 结果已经用于提出新假设，因此对新方案只能是历史探索数据；不要将其重新命名为新 test。

最低方法：FP32、all-W4、CEM-Update、同搜索 ScoreError、mean-only，并保留预先指定的 random allocation 对照。不要为追求超过 random 更换 seeds。全量 episode 数在资源测量后写入 manifest，不能从旧结果倒推挑样本。

报告 paired successes/failures、goal distance、MPC rounds、update/action fidelity、calibration 成本与每次 planning latency。早停造成总时间不同，不能解释为量化 kernel 提速。样本小则只作 exploratory go/no-go；未显著不等于等效。

事先冻结方向性门槛及资源停止点；若 update fidelity 改善但闭环没有相对同预算对照的支持，停止当前配方，不不断调 loss 权重或追加 seed。协议失效、数据泄漏与计算错误属于工程失败，应与假设不支持分开记录。

## 7. 新颖性与边界

候选贡献是 quantization calibration 对齐到实际 CEM update，以及在联合量化背景下选配置。CEM 更新、Gaussian distance、mixed precision、局部搜索、decision-focused optimization 都不是新概念。单纯把 rank loss 换成 mean MSE 不足以保证论文贡献；需要证明保留中间搜索分布尤其 sigma 的增量价值。

直接前作：QuantWM（https://arxiv.org/abs/2602.02110）、Where Bits Matter（https://arxiv.org/abs/2602.11882）；相邻方法思想：Differentiable CEM（https://arxiv.org/abs/1909.12830）。仅做过定向检索，没有完成穷尽性 novelty audit。无需为最低成本机制筛选先重跑完整文献 pipeline。

## 8. ASPIRE2A 操作约束：接手 agent / subagent 必须继承

- Login node 只用于轻量连接、提交、状态查询和小型控制文件操作。禁止大文件下载/传输/复制、hash、批量解压、依赖安装/编译、模型加载、推理、benchmark 和大型目录递归扫描。
- `tmux`、`nohup`、后台运行、低 CPU 或不占 GPU 不是豁免；禁止重启管理员终止的 login-node workload。
- 所有重 I/O 与计算在真实获批 PBS compute allocation 内进行。下载、校验、环境准备优先 CPU-only；确需 GPU 才申请 GPU。compute node 不能联网时暂停查明允许路径，不退回 login node。
- 重操作前检查 `PBS_JOBID` 非空、实际 hostname 不是 login node，并确认获批 allocation；失败即退出，不伪造环境变量。恢复旧脚本前审查并禁用不安全入口。
- 合理 walltime，任务完成或失败及时退出。清理只终止确认属于本任务的进程；一个 login node 的检查不代表其他节点，无从检查则标为未验证。
- 每个目标文件只有一个下载写入者；并发写过的文件隔离，不能凭 apparent size 判完整，校验在 compute allocation 内执行。
- 保留 SSH host-key verification；不把 credentials 写到日志、代码、Git 或交接文档。不额外做冗余 SHA-256 清单。
- 委派详细独立子任务时使用 `gpt-5.6-luna xhigh`，必要时 `max`，核实实际 model/effort；监控使用 `xhigh`。必须转交本节限制和明确范围/验收要求。

## 9. 预期交付

新实验目录应保存：协议/数据 split/配置/搜索预算 manifest、离线指标核对、逐配置搜索轨迹、完整选中 mapping、独立 DEV 诊断；只有执行阶段 C 才需要新 test 的逐 episode 结果和 paired 汇总。另记录 PBS 用量和中文结论，区分工程是否通过、机制是否支持、是否值得继续。没有做过的阶段明确写未运行。

给新对话的开场指令：

> 请先读本文件及所列 RankCal screening 证据，按 CEM-Update PTQ 开展最低成本验证，不重跑 IdeaSpark。先确认旧 candidate-pool 数据能否支持阶段 A，并完成可行的离线分析；后续按实际成本冻结小型实验协议与资源门槛，遵守项目 AGENTS.md 的所有 ASPIRE2A 限制。不要覆盖旧产物，不用旧 test 作新方案的独立确认，不把 emulation 当成部署加速。若证据不支持继续，明确停止并报告原因。
