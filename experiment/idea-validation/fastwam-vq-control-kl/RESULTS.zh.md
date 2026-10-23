# 控制相关 KL：阶段性诊断结果

当前结论：**初始状态的平均 KL/RMSE 改善不足以解释闭环；沿途关键状态的响应失真、持续指令偏差，以及完整 C80 上有限的真实优化收益，是目前更有证据的检查方向。** 在全部40个相同初始状态上，softbin 和 relational 的平均 endpoint 与 response 指标多数优于原 MSE，但原 sensor_noise 闭环成功数分别为24/40、21/40，低于 MSE 的26/40。完整47个沿轨迹状态的诊断见下文；其中5例由 MSE 成功 / relational 失败定义，不能外推为 softbin 所有失败的解释。

完整 C80 calibration 诊断也已完成，结果如下。完整目标尚未完成，未启动新训练。

## 实际覆盖与复现

- 原 paired280 的全部40个 sensor_noise cases，每例使用原 environment seed、sampler seed、prompt、proprio 和 noisy observation。
- clean image 在原 corruption 函数入口截获；仅替换 agentview，wrist 保留。每个原函数调用一次，不增加环境步。
- 四个 arm × clean/noisy ×40例，共320个 endpoint outputs，加1次 BF16 repeat，共321次 full model forward。
- 所有40例、四个 arm 的 noisy full32 action 均逐值复现原始动作，最大差为0。BF16 repeat 的 action、future 和 initial action noise 最大差也均为0。
- GPU `25891886.pbs101`：两半共用一个 allocation，实际 walltime `00:51:13`；PBS F/Exit0、artifact exit0、PHASE_COMPLETE、两半 matching-PBS summary 均通过。运行期有 GPU utilization / memory 记录。
- CPU 汇总 `25894046.pbs101`：F/Exit0、完整 receipt，实际 walltime `00:03:03`。覆盖240个 student endpoint 指标和120个 response 指标；每个 suite 10例，报告四 suite 等权均值。

## 全部40例的初始状态均值

所有误差均越低越好；MSE 指原 per-channel clipping MSE checkpoint。motor RMSE 使用前10步、6个 motor channels 的 normalized action；command 指实际 denormalization 后的执行指令。

| checkpoint | clean motor RMSE | noisy motor RMSE | noisy action KL | noisy latent NMSE | response KL | response command bias RMS | response command prefix error |
|---|---:|---:|---:|---:|---:|---:|---:|
| MSE | 0.035811 | 0.058782 | 0.268192 | 0.119199 | 0.294161 | 0.036422 | 0.715221 |
| softbin | 0.033930 | 0.050527 | 0.196004 | 0.103483 | 0.226421 | 0.027770 | 0.581082 |
| relational | 0.033782 | 0.053921 | 0.230085 | 0.111688 | 0.260519 | 0.031411 | 0.650964 |

response 比较 `student(noisy)−student(clean)` 与 `BF16(noisy)−BF16(clean)`，沿用冻结的 action scale 和 soft-bin 定义。prefix error 是前10步内累计 command 误差的最大绝对值；它是指令偏差代理，不是测得的机器人位移。

因此，response KL 目前也不能作为已经验证的闭环替代指标。该结果支持继续诊断状态依赖和误差分布，不能据此断言增加 response loss 会提高成功率。

## 全部5个 MSE 成功、relational 失败案例

下表完整保留这5例。该子集由已有闭环分歧定义，仅用于机制分析，不能当作新泛化测试或后续训练数据。

| index | BF16 原闭环成功 | MSE noisy motor RMSE | relational noisy motor RMSE | MSE response KL | relational response KL |
|---|---|---:|---:|---:|---:|
| 61 | 是 | 0.040895 | 0.059226 | 0.238000 | 0.236308 |
| 130 | 否 | 0.052732 | 0.034828 | 0.163138 | 0.169213 |
| 131 | 否 | 0.049501 | 0.033652 | 0.185046 | 0.118998 |
| 204 | 是 | 0.050961 | 0.042548 | 0.179870 | 0.122844 |
| 208 | 是 | 0.051886 | 0.142105 | 0.715749 | 1.281854 |

- **61：response KL 近似相同，执行指令的偏差明显不同。** noisy command bias RMS 从 MSE 的0.019664升至 relational 的0.044112，command prefix error 从0.319519升至1.014862。这支持保留绝对动作/指令 anchor，不能只约束 clean/noisy 差分。
- **208：第一步状态已经暴露较大的 relational endpoint 与 response 误差。** noisy command bias RMS 为 MSE 0.027138、relational 0.097485；prefix error 为0.585365和1.915283。这里存在明确的初始状态失真，但仍需轨迹确认它如何演化。
- **204：初始 endpoint 和 response 指标均更好，仍然闭环失败。** 初始状态无法解释全部退化，后续实际访问状态和控制信息读取是必要检查。
- **130、131：BF16 本身失败，而 MSE 成功。** 更接近 BF16 的初始动作不保证成功；BF16 是行为参考，不是成功动作 oracle。这两例保留在完整40例和5例分析中，解释时单独标明 teacher 结果。

这些观察与“初始状态平均指标掩盖少数重要偏差、后续状态分布变化、teacher 自身失败”相容，但尚未证明各因素的因果贡献。

## 完整 C80 calibration 与独立噪声配对

两半 `25894124.pbs101`、`25894125.pbs101` 均正常完成，各40 contexts、320次 native queries，共640次。所有80例的 clean BF16 action/future 均与原 C80 teacher cache 逐值一致；四个 arm 在每个 variant 的初始 action noise 一致。独立噪声图像未用于这些 checkpoint 的原训练。CPU `25894399.pbs101` 完整汇总，实际 `00:03:29`；无新训练和 heldout 输入。

下表使用原冻结统计、系数和相同 teacher。common objective 的绝对值取决于固定系数，只能在同一列比较。

| checkpoint | clean motor RMSE | clean soft-bin objective | clean relational objective | noisy motor RMSE | noisy response KL | response command bias RMS |
|---|---:|---:|---:|---:|---:|---:|
| MSE | 0.029801 | 0.00274400 | 0.00479867 | 0.049779 | 0.210663 | 0.029709 |
| softbin | 0.030608 | 0.00279767 | 0.00494410 | 0.048662 | 0.220292 | 0.026833 |
| relational | 0.030427 | 0.00283075 | 0.00479362 | 0.050914 | 0.247867 | 0.029938 |

这里不能把所有问题都归为过拟合：softbin 在原训练输入上的共同 soft-bin objective 仍不及 MSE；relational 的共同 relational objective 仅比 MSE 低约0.1%，同时绝对 action/future 保真退化。它的 clean action temporal KL 为0.034701（MSE 0.037883），latent spatiotemporal KL 为0.004708（MSE 0.004895），但 clean action KL 为0.071480（MSE 0.068851）、latent KL 为0.146859（MSE 0.142764）。结构项改善并未带来同幅度的整体收益。

独立噪声配对的 response KL 中，softbin 和 relational 均高于 MSE；而前面40个 evaluation 初始状态恰好排序相反。两组 contexts、corruption 配置和状态不同，不能把差异直接解释成泛化差距或因果效应。它说明单一均值排序不能稳定代表实际闭环；calibration 状态覆盖和沿轨迹的变化仍需补齐。softbin 的 command bias 更低、response KL 却更高，也说明两者强调的误差不同。

这80组是同一批 C80 clean contexts 的独立噪声视图，增加了输入变化，没有增加 demonstrations 或访问状态种类，不能称为已经扩充了状态覆盖。

## 全部40例的固定 reader 干预

`reader_0/1` 各完成280项核心干预与3项 student 自重建。CPU `25894440.pbs101` 正常完成完整560项核心干预的汇总，实际17秒。BF16 reader 读取 BF16 future 在全部 clean/noisy states 上精确复现 BF16 native action，误差均为0；student 自重建检查覆盖每半首个 clean case 的三个 student，误差均为0。reader 只运行 action 路径，每个 future 单独重新 prefill KV；未推进环境。

以下各列以对应 clean/noisy 输入、同一 seed 的 BF16 native action 为目标，报告 suite 等权的前10步 motor6 RMSE。clean future 列使用 clean 输入；其余列使用 noisy 输入。

| student checkpoint | 原生 noisy student | BF16 reader + student clean future | BF16 reader + student noisy future | student reader + BF16 noisy future |
|---|---:|---:|---:|---:|
| MSE | 0.058782 | 0.026758 | 0.051988 | 0.023717 |
| softbin | 0.050527 | 0.024203 | 0.044422 | 0.023785 |
| relational | 0.053921 | 0.025105 | 0.045719 | 0.023854 |

用固定 BF16 reader 读取 student noisy future 时，control KL 为 MSE 0.213641、softbin 0.158943、relational 0.174314，command prefix error 为0.702111/0.563321/0.609665。KL 模型在这些控制相关均值上仍然更好。因此，闭环退化也不能简单归因于“KL 的平均 future 控制信息比 MSE 更差”。

把 future 统一替换为 BF16 future 后，三个 student reader 的 noisy RMSE 都在0.0238附近，原生差距明显缩小。这说明在这些初始状态上，上游 future 的变化是动作保真差异的重要来源；它没有证明 future loss 能改善闭环，也不能把两列 RMSE 直接相加或换算成某个组件的成功率贡献。

目前证据把重点推向了：关键状态的误差尾部与持续偏差、后续访问状态，以及 calibration 的输入/状态覆盖。固定 BF16 reader 是可实现的辅助目标，但其初始状态均值已经出现与闭环成功率相反的排序，不能把它当成已验证的成功率替代指标。

## 全部47个沿轨迹状态：初始均值没有覆盖的退化

两半 `25894461.pbs101`、`25894618.pbs101` 均正常完成，实际 walltime 00:32:56 / 00:27:23；共47个实际 states、376个 native endpoints 和376项 BF16 reader 干预，source replay、相同输入与 initial-noise parity 全通过。CPU `25894860.pbs101` F/Exit0 完整汇总，实际00:00:44。只重放保存的 observations，没有推进环境或重新跑闭环。

每列先平均每个 case 的实际 snapshots，再等权平均5个 cases。MSE collector 为22 states，relational collector 为25 states；早成功的 MSE episodes 不存在部分 step200 snapshot。两个 collector 的均值不能作相同状态的直接因果比较；在每个 collector 内，三个模型面对完全相同的 states/seeds。

| collector | checkpoint | noisy motor RMSE | response KL | BF16 reader control KL | command bias RMS | command prefix error |
|---|---|---:|---:|---:|---:|---:|
| MSE | MSE | 0.052606 | 0.243286 | 0.167583 | 0.029976 | 0.604954 |
| MSE | softbin | 0.044382 | 0.210594 | 0.080184 | 0.022518 | 0.496972 |
| MSE | relational | 0.058851 | 0.307519 | 0.194681 | 0.035440 | 0.726122 |
| relational | MSE | 0.075059 | 0.366220 | 0.278518 | 0.045246 | 0.990584 |
| relational | softbin | 0.067230 | 0.370282 | 0.223035 | 0.041420 | 0.893697 |
| relational | relational | 0.076616 | 0.441641 | 0.257565 | 0.048414 | 1.056242 |

relational 的 response KL 在两个 collector 的这5例上都比 MSE 差，支持继续检查沿途输入变化的响应失真。reader control KL 在 MSE collector 上更差，在 relational collector 上却更好；后者的 command bias、prefix error 和 reader motor RMSE 仍更差。因此，reader KL 的作用应是可验证的辅助约束，不能单独替代绝对 action/command 约束或成功率。

逐 state 的完整精简表由 CPU `25894866.pbs101` 输出为 `results/trajectory_trace/summary.json`，正常完成实际00:00:17，0个新 forward。原始452,909-byte details留在远端，没有放宽小文件读取上限。

- **204 的反转出现在后续状态。** 在 relational 自身轨迹 step0，relational 的 noisy RMSE/control KL 为0.0425/0.0463，优于同状态 MSE 的0.0510/0.1242；step100 变为0.0517/0.1179，MSE 为0.0445/0.0573；step200 两者都差，relational 为0.1671/0.6893，MSE 为0.1473/0.5943。初始或全轨迹均值不能定位这些局部变化。MSE collector 的204上，relational在四个保存状态的motor RMSE反而都低于MSE，说明只沿成功collector检查也会漏掉问题。
- **208 在首个状态就有明显失真。** relational 的 RMSE0.1421、bias0.0975、control KL1.0681，MSE为0.0519/0.0271/0.3388。进入relational自身step10状态后，三个模型RMSE都较大，relational0.1437、MSE0.2555。局部更接近BF16并不意味着此前进入该状态的动作路径更好，仍不能由这些重放证明失败因果。
- **61 对 collector 很敏感。** 在MSE collector step10，relational RMSE/control KL0.1270/0.8941，MSE0.0486/0.1107；但在relational自己的step10状态，relational只有0.0584/0.1169。前者是反事实查询，不能当作relational实际执行的动作。
- **130、131 继续完整保留。** 两例BF16原闭环失败，不能把更接近BF16解释为更接近成功。softbin在本表多项均值更好，但本子集按MSE成功/relational失败定义，且没有softbin collector；它不能解释softbin自身全部闭环损失。

这些状态已用于机制分析，不能进入新训练、阈值或超参数选择。候选步、系数与训练数据选择只使用calibration侧信息，最终仍保留原heldout80和paired280。

## Calibration覆盖代理：主要差异来自corruption，未证明clean状态整体脱离C80

CPU `25894861.pbs101` 正常完成，实际00:02:37。特征仅取BF16观察帧0：两相机各48通道的空间均值/标准差，共192维；标准化只拟合原C80 clean。无需新模型forward，未来预测帧未进入特征。

| 输入组 | 到原C80 clean的最近特征距离 | clean/noisy配对距离 |
|---|---:|---:|
| C80 clean（leave-one-out） | 0.641885 | 0 |
| C80 independent noisy views | 1.496867 | 1.479953 |
| 原sensor初始clean | 0.625974 | 0 |
| 原sensor初始noisy | 1.477927 | 1.470838 |
| MSE collector clean states | 0.569825 | 0 |
| MSE collector noisy states | 1.703495 | 1.735549 |
| relational collector clean states | 0.555372 | 0 |
| relational collector noisy states | 1.692377 | 1.726117 |

在这个粗粒度表征中，没有发现evaluation clean或轨迹clean整体离C80更远；corruption引起的变化明显更大。新增calibration噪声视图与sensor初始噪声的平均距离相近，轨迹噪声的配对变化更大，但contexts/severity和时间覆盖不同，不能直接称为分布支持缺失或泛化差距。这些特征会遗漏目标位置、接触和控制语义，不能证明calibration已经充分覆盖。

据此，下一步优先对照同一批calibration clean/noisy输入的数据增广与response辅助项；单纯增加更多相似clean snapshots未必是最有效的第一步。后续状态扩充仍限定在calibration demonstrations/tasks，不能采用上述5例evaluation轨迹。

## 正在推进

- `reader_0/1` 及完整40例 CPU 汇总已完成。真实 checkpoint backward 对 future 的两个预测帧产生37,632个非零梯度，观察帧梯度为0；BF16 model、reader 和 VQ banks 未改变。它只证明 future→reader 可微，尚未证明 student D/r 能优化该目标。
- 两个 MSE capture units 已完成全部5例，逐例成功状态、episode 步数和原始首个 full32 action 均复现。61/130/131/204/208的步数分别为221/273/130/113/107，共87次 native queries。原记录没有全轨迹 action 序列，不能据此声称每个后续 action 逐值相同。Relational 两半已完成全部5例，均复现原失败状态和步数。第一半同状态轨迹诊断 `25894461.pbs101` 已 F/Exit0，通过 source 与 initial-noise parity；实际28/28 states、224/224 native endpoints、224/224 BF16 reader passes，walltime 00:32:56。第二半 `25894618.pbs101` 也已完成实际19/19 states，152/152 native endpoints与152/152 reader passes，实际00:27:23。全部47 states的CPU汇总与覆盖分析见上文。使用实际存在的 step0/10/50/100/200 snapshots，不代表逐步覆盖完整 episode。
- `calibration_0/1` 与 `aggregate_calibration` 已完成全部80个独立 clean/noisy pairs；clean BF16 teacher cache 精确复现；原 heldout demonstrations 未参与。
- Calibration覆盖与全部轨迹状态汇总已完成；仍待实际 D/r 参数gate与完整C80可回滚有限步检查，再冻结新loss的训练实现和系数。原 heldout80 与 paired280 的评估范围保持不变。

`parameter_gate.py` 已实现固定 calibration pair 2000 上的 D/r 检查：动态量化路径、分阶段 chain-rule VJP、原 learning rates 下的可回滚候选、实际 loss 复算与冻结参数检查。小型 toy 中，两种 loss 的 staged VJP 均与直接 joint autograd 一致，正常与异常退出均精确回滚；unused-parameter 检查也在首轮 allocation 内通过。

首轮真实检查 `25894427.pbs101` 在首次 student backward 时 OOM（A10040，PyTorch allocated 38.58GiB，另需168MiB），此前该 clean query 的 action/future 与 native 部署逐值一致。该轮只执行1次 student query，未产生 candidate 或训练 checkpoint。修复将该阶段不用的 BF16 reader 权重保留为不可变 CPU masters，仅在 reader 调用区间装入 GPU；每次核对权重往返数值一致，并增加 clean/noisy 两项 BF16 native reader anchor。原 loss、pair2000、seeds、learning rates、候选步和回滚规则不变。修复作业为 `25894526.pbs101`，时限仍为 `01:59:00`，新检查尚未通过；预计12次 student queries、10次 BF16 reader passes、0次 teacher full query。额外的权重驻留/异常清理 toy 在 allocation 内先运行。

第二次作业 `25894526.pbs101` 已确认 F/Exit1：action response 两组 D/r 各614个 tensor 均连通且非零；固定 pair 的 baseline0.184631，fraction1.0/0.1 的实际 loss 分别0.164716/0.135302，两步均回滚。随后第一次 clean latent-reader gradient forward OOM。这是显存实现问题，不是 loss 的实验证伪。第三次修复 `25894737.pbs101` 在 reader 区间额外把未使用的 student dense weights 放在 CPU，保留 sampler device，并检查 reader BF16 与 student weights 的数值往返；同一 pair、loss、seeds、learning rates 和两个候选步不变。两项完整重跑以取得单一完整的参数恢复证据；新 toy 在 allocation 内先运行。

完整 C80 检查代码和独立 child protocol 已写入并上传，尚未提交：两个 isolated objectives 各在原80 pairs 上求完整平均梯度，再生成固定1.0/0.1可回滚候选，在同80 pairs 上用原生部署前向评价。它不替代后续 joint loss 的真实训练或 heldout/闭环评估。

这仍是单 pair 实现检查，不替代完整 C80 的优化证据；尚无新训练或新 checkpoint。

capture 的首次运行暴露 RNG 记录比较错误：实际 `capture_rng` 将 NumPy state 存为可安全序列化的 dict，比较器误当作 tuple。两个 MSE 作业在首个 native action 返回后失败；另外两个已确认加载同一旧代码的 relational 作业由本任务停止，避免继续空耗。现已按实际 dict 格式修复比较，并补充异常时 query 计数与失败摘要；11项 toy checks 通过，四个作业均在确认终态后续跑，原 observations、cohort 和 seeds 保留。这是实现错误，不能当作 checkpoint 行为或实验结果。reader 与 calibration 不使用该比较入口，保持原作业。

## 证据位置

`results/sensor_0/summary.json`、`results/sensor_1/summary.json`、`results/sensor_batch/summary.json`：覆盖、source parity 与 repeat receipt。

`results/aggregate_sensor/summary.json`：全部40例的 suite 分层和等权均值；`details.json`：逐 case endpoint / response 指标。所有数值来自 CPU allocation，未在 login node 或本机重新运行模型或数值聚合。

CPU `25894758.pbs101` 已 F/Exit0、matching summary、artifact exit0 和 PHASE_COMPLETE 全通过，实际00:02:17。新驻留 helper 的正常/异常退出、staged VJP 与直接 autograd、完整 cohort 梯度均值/投影、candidate PBS/path/seed 绑定拒绝及真实 Deadline API 的小型检查均通过。没有加载模型或访问真实输入；这不证明 GPU 显存修复有效或真实 C80 目标可优化。完整 C80 GPU 检查仍等待单 pair 实际模型 gate。

## 独立训练输入准备完成，未启动训练

CPU `25894948.pbs101` 已F/Exit0、matching summary、artifact0与PHASE_COMPLETE通过，实际00:02:11。`training_data_protocol.json`冻结原C80的80个contexts及独立corruption seeds；新池`data/training_pairs`共80个新生成的noisy views，原clean、wrist、proprio等arrays保留，原diagnostic-only池不变。五种families与severity2/4/7/10按suite平衡。三项plan检查、全部原source case/demo/frame/sampler绑定与非agentview逐数组一致性检查均在allocation内完成。

这些是训练侧的输入增广，不是80个新状态或demonstrations，也不是新development set。不同seed不保证deterministic blur产生不同像素。新teacher/student queries均为0、训练updates为0、没有heldout/evaluation轨迹输入。后续所有对照必须使用相同训练输入与teacher，并分别报告额外训练预算与data/loss的贡献；loss配方和系数仍待实际参数gate及完整C80优化证据。


2026-10-21 09:13 SGT owner 核对：parameter_gate 25894737.pbs101 与共享 teacher 准备 25895227.pbs101 均为 query_rc0 / Q。后者仅生成原 C80 训练输入的 BF16 标签，1GPU、01:59:00；计划复用80 clean与32 noisy，补48个唯一 noisy teacher 输出，实际张量与输入绑定在 allocation 内检查。三份改动源的 AST 与 PBS shell syntax 已通过，真实 Tensor toy 将在 allocation 内先运行。未启动 student 训练，未新增 checkpoint；GPU运行期15秒 telemetry、gpt-6-luna/xhigh 每600秒只读监控沿用。已完成输出可复用；验证终态后才可修复重试，所有 query attempts（含失败与重试）保留独立计数。


2026-10-21 09:24 SGT：组合 loss 已实现于 training_objective.py，未设置实验默认系数。CPU training_loss_selfcheck 25895317.pbs101 已 F/Exit0、artifact0、PHASE_COMPLETE、matching summary，实际00:02:25。四项测试覆盖直接 joint autograd 与分阶段 endpoint/逐variant reader cotangents + paired_vjp 一致；first10motor6、future frame0 的范围；teacher/reader无梯度；response共同bias盲区由absolute anchor惩罚；显式系数拒绝。0真实模型forward、0训练。它不替代真实模型D/r与完整C80检查，训练系数、分组与schedule仍未冻结。


2026-10-21 09:47 SGT：control_training.py 与训练 dispatcher 已实现；正式 training_protocol.json 尚未生成，未提交 train_* 或执行真实更新。入口消费160个共享 BF16 标签，按原C80顺序训练 learned MSE 初始化的D/r，保留原始 sampler dtype，组合 endpoint/逐reader cotangents；每个成功pair只做一次Adam更新并原子保存D/r、moments、RNG和输入prefix。每个新PBS进程的reader首pair重新检查native anchor，运行期利用率/显存继续15秒记录。必须有完整C80两个目标在相同所选fraction上的实际下降证据、完整teacher与CPU检查，且配置本地/远端冻结一致，才允许训练提交。

CPU training_runtime_selfcheck 25895458.pbs101 已 F/Exit0、artifact0、PHASE_COMPLETE、matching summary，实际00:02:33。配置/receipt双objective与80-case覆盖拒绝、foreign PBS拒绝、checkpoint来源与顺序、真实Deadline.near、atomic tensor存取、BF16 endpoint dtype合并小型检查均通过；0真实模型forward/训练。这些不证明真实训练与闭环收益。09:46 fresh status：parameter_gate25894737、training_teacher25895227仍query_rc0/Q。

2026-10-21 11:03 SGT：新 checkpoint 加载与原 heldout80 入口已实现，训练收尾会复用已有 query 保存完整 reference_endpoint；原40 cases×2 seeds与共享teacher不变。CPU training_evaluation_selfcheck25896201 已 F/Exit0、artifact0、PHASE_COMPLETE、matching summary，实际00:02:39，通过小型D/r安装、错误来源拒绝、不可变base、完整cohort及等权汇总检查；0真实模型forward、0训练。真实加载后source/native infer parity仍须在每个实际GPU评估进程通过。未提交 heldout 或新 paired280 作业，paired280入口待接入。两项GPU parameter_gate25894737、training_teacher25895227 此时均 fresh query_rc0/Q，继续原监控，不重提。

2026-10-21 11:19 SGT：paired280续跑入口与CPU配对汇总已实现；CPU training_closedloop_selfcheck25896407 已 F/Exit0、artifact0、PHASE_COMPLETE、matching summary，实际00:02:41。fake循环验证原10步chunk/400或700cap、指令原precision、成功/超时/软停及完整记录复用；真实原manifest与15方法共4200条reference均完整载入。原BF16216、per_channel_clip203、softbin199、relational198均复现。0真实模型forward/环境steps/训练；新checkpoint的GPU闭环仍未验证和提交。两项待运行GPU当前仍fresh query_rc0/Q，未重提。

2026-10-21 11:36 SGT：最终checkpoint在原C80 training_pairs的完整loss评估入口已实现，所有checkpoint共享160个teacher labels，保存未加权absolute/response/latent/control原始项，并在CPU以相同显式loss配方重算。MSE参照预计复用80clean+32匹配noisy native outputs，另外48noisy需新forward；reader控制评估另计。CPU final_c80_selfcheck25896560 已F/Exit0、artifact0、PHASE_COMPLETE、matching summary，实际00:02:53，小型配对/来源/缓存prompt切换/reader权重及80-case汇总检查通过；0真实模型forward/训练。这不证明真实GPU参数可优化或闭环改善。两项GPU当前均fresh query_rc0/Q，尚未冻结新训练config或提交实际final-C80评估。
