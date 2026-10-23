# 控制相关 KL：分阶段验证

首轮40例相同状态结果及全部5个分歧案例的分析见 `RESULTS.zh.md`。

目标是先定位 sensor_noise 下的相同状态行为，再验证 calibration 覆盖、action response 和固定 BF16 reader 的 latent 控制保真。后续训练及原冻结评估仍属于本 goal；完成首轮诊断不代表整个目标完成。

## 第一阶段

- CPU 读取已完成的原 paired280 episodes：保留全部40个 sensor_noise cases，并完整提取 MSE 成功、relational 失败的5例，禁止挑选其中有利子集。
- 对40个初始状态，各 checkpoint 在完全相同的 noisy/clean observation、prompt、proprio 和 sampler seed 下查询。clean 图像在原 sensor corruption 函数入口截获，原函数只调用一次，不增加环境步。真实 Plus 只修改 agentview camera；手腕图像保持原值。
- 对全部5例分别用原 MSE 和 relational collector 复现原轨迹，保存 step0/10/50/100/200 的已存在 observations、simstate、RNG 与 commands。额外 counterfactual model queries 在保存后离线运行，不改变 collector 的原噪声流。
- 在相同状态下比较 endpoint KL、前10 motor command 误差、chunk bias、prefix command 偏差、noisy-clean response 误差；后续固定 BF16 reader 做 teacher/student future 互换。

这些已见闭环状态只做机制诊断，不能加入后续训练或作为未见泛化证据。每个episode内的states、perturbations和action steps均为重复测量。

## 后续阶段

1. 只在原 calibration tasks 的独立 contexts / seeds 中扩充状态覆盖；新输入使用同一输入下的 BF16 teacher，原 teacher 缓存仅在输入绑定完全相同时复用。
2. 保留 outer loss `L_action + lambda * L_latent`，逐项检验 action response KL 与固定 BF16 reader 下的 latent control KL，不同时改动所有因素。
3. reader 权重冻结、student future 保留梯度、每个 future 重新 prefill KV；跳过 video denoising，不复用另一个 future 的 cache。
4. 原40 held-out cases ×2 seeds、原 paired280闭环 cohort 不变；无新 development set、held-out 调参、gripper 或 late22 新训练项。

训练协议和系数在相应阶段采集训练数据前冻结。先验证定义、梯度和 reader parity，再运行真实训练；不能仅凭 loss 下降跳过原冻结评估。

`training_objective.py` 明确了后续组合 loss 的 reduction，所有系数必须由正式实验配置显式传入，没有实验默认值。以 `D_SB(T,S)` 表示沿用原冻结统计的 soft-bin KL：

```
L_endpoint = w_a * mean_clean_noisy D_SB(A_T, A_Q)
           + w_response * D_SB(A_T_noisy-A_T_clean, A_Q_noisy-A_Q_clean)
           + lambda * w_z * mean_clean_noisy D_SB(Z_T, Z_Q)
L_reader   = lambda * w_control * mean_clean_noisy D_SB(A_T, Reader_BF16(o,Z_Q))
L          = L_endpoint + L_reader
```

两种 mean 都给 clean/noisy 各1/2权重；action 仅 first10 motor6，latent anchor 排除观察帧0。absolute action anchor 要求正权重；其他项允许零权重做单因素消融。reader 每个 variant 单独释放计算图后，把其 future cotangent 与 endpoint cotangent 相加，再用既有 `paired_vjp` 分别重建 student 图。teacher 和 reader 参数保持冻结。`training_loss_selfcheck` 只在 CPU allocation 检查小型 joint-autograd 与该分阶段计算的一致性，不替代真实 D/r / 全 C80 检查，也不冻结系数、初始化、更新次数或训练分组。

## 诊断如何决定下一步

- 相同 observation / seed 下同时查看 endpoint、执行 command 的 chunk bias 与累计偏差，以及 noisy-clean response。累计 command 是误差累积代理，不是机器人实际位移。
- 四个 reader/future 组合分别为 BF16/BF16、BF16/student、student/BF16、student/student。后者来自原生输出。中间两项帮助区分 future 控制信息损失和 action 路径损失；二者可能交互，不能直接相加解释成功率。
- 全部5个 discordant cases 必须保留，并明确区分 BF16 原本成功的3例与原本失败的2例。BF16 是冻结行为参照，其动作并非任务成功的真值。
- 轨迹 CPU 汇总先在每个 case 内平均实际采到的 states，再平均5个 cases；MSE 与 relational collector 分开。状态少的早成功 episode 不应因此权重更低。逐 state 明细用于定位误差出现时刻，不能当成独立样本量。
- Calibration 覆盖检查使用原 calibration contexts 的独立 corruption seeds / states；这40个已见 evaluation cases 和5例轨迹不进入训练。增加数量前先检查训练数据是否覆盖噪声类型和模型访问状态，再分离 data-only 与 loss-only 的效果。
- 真实训练前，必须在 calibration 上检查 connected student-parameter gradient，并通过可回滚的小步前后真实目标比较，检查 STE 预测的下降是否真的出现。future 梯度非零只证明 reader 可微，不能代替该检查。

`control.py` 的依赖顺序会阻止跳过首个 source-parity gate 或使用未完成 captures。`aggregate_sensor`、`aggregate_reader` 和 `aggregate_trajectory` 都在 CPU allocation 计算，不在 login node 聚合。

## 已验证的 calibration 数据准备

CPU `25892049.pbs101` 已完成 source metadata inventory：原 C80 覆盖40个 tasks，每 task 两个不同 demonstrations；原 heldout 每 task 另一个 demonstration，重叠为0。每 suite 的20个 contexts 为 early7 / middle7 / late6，每个 demonstration 只提供一个 snapshot；数据声明全部为 original，未包含 sensor corruption 或 student-collected states。40个 sensor evaluation cases 对应的 tasks 都已在 C80 内。这只证明来源覆盖，不能证明输入特征的支持范围。

CPU `25892084.pbs101` 已准备全部80组 calibration clean/noisy pairs：每 suite 五种 corruption 各4例、severity2/4/7/10各一次，使用独立 deterministic corruption seed。直接调用同一 LIBERO-Plus corruption functions，仅改变 agentview，原 clean NPZ 和 wrist 等其他 observation arrays 保留。3组已完成配对通过逐数组一致性检查后复用，新增77组。

首次准备在 fog 的 NumPy2 `np.float_` 兼容处失败；修复复用原 runner 的 `libero_state_loader.ensure_numpy_compat`，保持 float64 语义。两个 CPU 工作均为0 model queries / 0 training updates。配对定义固定在远端 `data/calibration_pairs/definition.json`，当前用途仅为 calibration response 诊断，尚未加入训练。原 heldout inputs 未用于构造配对。

## 资源和状态

首个 GPU allocation `25891886.pbs101` 已实际完成全部40个 sensor cases：两半20/20，分别161次（含1次 repeat）和160次 full forward，共321次；实际 walltime `00:51:13`，PBS F/Exit0、artifact exit0、PHASE_COMPLETE 和两半 matching-job summary 均通过。四个 checkpoint 对所有40个 noisy 初始状态的原始 full32 action 逐值复现误差全为0；首例 BF16 repeat 的 action、future 和 initial noise 最大差也均为0。第二半通过 adopt 登记共享 allocation，没有再 qsub。

这验证了输入与采样复现，并不说明哪种 loss 更好。clean/noisy response 的 CPU 汇总已完成。两半 reader 均已通过，各280项核心干预和3项 student 自重建；reader_0 的真实 future-gradient gate 也已通过。完整 reader CPU 汇总已完成。全部5例的 MSE collector 已完成并严格复现原成功状态和步数，relational collector 也已全部完成并复现原失败状态/步数。

C80 calibration 两半及 CPU 汇总已完成：全部80个 clean BF16 action/future 精确复现原 teacher cache，共640次 native queries；独立噪声配对中，MSE/softbin/relational 的 response KL 分别为0.210663/0.220292/0.247867。完整解释见 `RESULTS.zh.md`。这些是原 clean contexts 的额外噪声视图，没有增加 demonstrations 或状态种类。

固定 calibration pair 2000 的参数检查实现与小型 chain-rule / 异常回滚测试已完成。首轮25894427.pbs101在第一次 backward 显存不足，此前动态 action/future 精确复现 native；修复作业25894526.pbs101将不用的 BF16 reader 权重保留在 CPU，只在 reader 区间装入 GPU，并检查数值往返和 BF16 native anchor。原 loss、MSE checkpoint、D/r learning rates 和固定1.0/0.1候选步不变，所有修改回滚，不保存训练 checkpoint。单 pair 仅验证实现；完整 C80 的真实目标变化仍需独立检查。真实训练尚未启动。

每个 GPU allocation 为 `01:59:00`，soft deadline 6600秒，每15秒将利用率和显存写入 job.log。计算、模型、张量和真实聚合只在已验证的 PBS compute allocation 内。Login 仅处理小文件、提交与状态查询，保留 SSH host-key verification，不输出凭据，不使用 banked reset。

为减少队列等待和模型重复加载，`sensor_0` 入口现在在一个 allocation 内顺序尝试两个原20-case shards：先完成并通过第一半，再用同一 runtime 和剩余全局时间运行 `sensor_1`。两半仍分别保留原320个 endpoint records，外加1次 BF16 repeat，共321次预定模型 forward，其中1次为重复性检查。若第二半未完成，保留其已完成 records，随后可单独续跑。仅当 parent allocation 和 `sensor_1` summary 都成功完成、PBS ID相符时，才用 `control.py adopt --unit sensor_1` 登记共享 allocation 的结果，不重新 qsub；未完成时禁止 adopt。若 allocation 启动时已取得旧源版本，则维持原两次 allocation 的方式，不能凭计划登记第二半完成。

gpt-6-luna/xhigh subagent 每600秒监控权威 handles；无实质变化的 Q/R 保持安静，失败不会自动重提。`RUN_STATE.json` 记录完整目标及未完成阶段，`handles.json` 记录当前作业，`last_status.json` 与 `monitor_state.json` 记录实时观测。

## 工作流参考

实验对照、重复测量和分阶段归因检查采用 experimental-design skill；其工作流来源为 Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)，当前公开版本 v2。该参考仅对应工作流，不构成本项目控制 KL 的实验结果或方法验证。

截至2026-10-21 07:38 SGT：trajectory_0 已正常完成实际28 states，trajectory_1 为25894618.pbs101、expected19 states，正在运行。parameter_gate第二次在latent reader梯度前向OOM，action response组件已通过但整体未过门；第三次修复25894737.pbs101使用两阶段权重驻留。完整C80梯度/有限步检查实现已上传，需单pair完整通过后才提交；真实训练仍为0。

截至2026-10-21 08:07 SGT：两个trajectory单元已完成全部47个实际states；aggregate_trajectory25894860、aggregate_coverage25894861和trajectory_trace25894866均F/Exit0并通过完成receipt。完整均值和全部state精简表已下载，原大details留在远端。参数gate25894737仍Q，旧failed summary不作为新job状态；full-C80 GPU检查与新训练均未提交。

为并行推进非GPU准备，training_data_protocol.json已冻结独立训练输入池：原80 C80 clean contexts、相同五种sensor functions和severity平衡、独立corruption seeds，只改变agentview；不增加states/demos，不建立新development set。CPU25894948.pbs101先运行plan检查再生成data/training_pairs。旧data/calibration_pairs仍diagnostic-only，未改。不同seed不保证deterministic blur的像素不同；本阶段0模型/teacher queries、0训练，完整C80优化验证仍待GPU参数检查。


2026-10-21 09:13 SGT owner 核对：parameter_gate 25894737.pbs101 与共享 teacher 准备 25895227.pbs101 均为 query_rc0 / Q。后者仅生成原 C80 训练输入的 BF16 标签，1GPU、01:59:00；计划复用80 clean与32 noisy，补48个唯一 noisy teacher 输出，实际张量与输入绑定在 allocation 内检查。三份改动源的 AST 与 PBS shell syntax 已通过，真实 Tensor toy 将在 allocation 内先运行。未启动 student 训练，未新增 checkpoint；GPU运行期15秒 telemetry、gpt-6-luna/xhigh 每600秒只读监控沿用。已完成输出可复用；验证终态后才可修复重试，所有 query attempts（含失败与重试）保留独立计数。


2026-10-21 09:47 SGT：control_training.py 与训练 dispatcher 已实现；正式 training_protocol.json 尚未生成，未提交 train_* 或执行真实更新。入口消费160个共享 BF16 标签，按原C80顺序训练 learned MSE 初始化的D/r，保留原始 sampler dtype，组合 endpoint/逐reader cotangents；每个成功pair只做一次Adam更新并原子保存D/r、moments、RNG和输入prefix。每个新PBS进程的reader首pair重新检查native anchor，运行期利用率/显存继续15秒记录。必须有完整C80两个目标在相同所选fraction上的实际下降证据、完整teacher与CPU检查，且配置本地/远端冻结一致，才允许训练提交。

CPU training_runtime_selfcheck 25895458.pbs101 已 F/Exit0、artifact0、PHASE_COMPLETE、matching summary，实际00:02:33。配置/receipt双objective与80-case覆盖拒绝、foreign PBS拒绝、checkpoint来源与顺序、真实Deadline.near、atomic tensor存取、BF16 endpoint dtype合并小型检查均通过；0真实模型forward/训练。这些不证明真实训练与闭环收益。09:46 fresh status：parameter_gate25894737、training_teacher25895227仍query_rc0/Q。

## 新 checkpoint 的评估接入（准备）

训练入口会把收尾时已经执行的首个 C80 clean cached query 保存为 `reference_endpoint.pt`，包含完整 action/future、initial noise、输入身份与训练来源，不增加 forward。新 `trained_deployment.py` 从 immutable decoded base 重建 D 补偿权重，更新内部部署绑定；再次切换或激活不会重复乘 D。完整 parent/training/data/source jobs、80-update completion、原子部署路径和冻结状态必须匹配。

`training_evaluation.py` 实现原 heldout80：每半固定40 cases的一个原 seed index，复用原共享 teacher，并按原 plan row 和 sampler seed 绑定；保留完成的 endpoint tensors。每个 GPU 进程启动时，会检查训练参考 action/future/noise 与实际 native infer 的 action 均逐值一致。CPU 汇总仍按 case 内两 seeds、suite 内10 cases、四 suites 等权，报告原 RMSE/NMSE、共同 softbin/relational KL 和 command 指标，不把它们声明为新 joint loss 的完整评估或闭环结果。

CPU-only `training_evaluation_selfcheck` 为 `25896201.pbs101`，已 F/Exit0、artifact0、PHASE_COMPLETE 与 matching summary，实际00:02:39。小型安装、错误绑定拒绝、不可变 base、原80查询覆盖与汇总检查通过；0真实模型 forward、0训练。正式训练 config 未冻结，未提交 heldout 或新 paired280 作业；真实模型评估尚未验证，paired280 入口仍待接入。11:03 SGT owner fresh status：parameter_gate25894737 与 training_teacher25895227 均 query_rc0/Q，保留原句柄与监控。

2026-10-21 11:16 SGT：`training_closedloop.py` 与 CPU 汇总入口已实现，未提交新模型闭环。两半保留原 parity indices，每半140个episodes；同一 manifest、state0、环境/采样seeds、30步settling、每次执行10步和400/700 cap。正式动作只来自 native `runtime.infer`；环境指令保持原 `runtime.command` 精度，每次检查614个Linears与decoded-BF16/A4/BF16-KV路径。

每个完整episode独立原子保存；软时限在6600秒前留出收尾时间，部分episode不保存为失败，后续复用全部完整结果。部署必须通过既有训练reference及native infer复现检查。汇总要求完整280，并复用原15个方法的4200个配对结果，按overall/suite/dimension/cell给出success与wins/losses/ties；不产生kernel latency结论。

CPU `training_closedloop_selfcheck25896407.pbs101` 已 F/Exit0、artifact0、PHASE_COMPLETE、matching summary，实际00:02:41。小型循环、指令精度、成功/超时/软停、完整记录复用与paired统计均通过，并确认worker与原loader读取同一280-row manifest。原15方法×280=4200条结果完整载入，BF16/MSE/softbin/relational成功数为216/203/199/198。0真实模型forward、0真实环境steps、0训练。GPU两项在11:19仍fresh query_rc0/Q；真实新checkpoint闭环尚未验证，未提交新闭环作业。

## 最终 checkpoint 的完整 C80 loss 评估（准备）

`final_c80_eval.py` 在同一80个 training contexts × clean/noisy两views上读取共享160个 BF16 标签，分别保存最终checkpoint的native endpoints与固定BF16 reader actions；无backward/optimizer。原MSE参照只复用按实际输入绑定一致的80 clean和32 noisy native outputs，另外48 noisy需新查询。Reader对所有views重新读取student future；每个PBS首pair的teacher future必须精确复现该variant的native teacher action，输入、prompt、初始noise与future frame0保持配对。

记录保留未加权的action absolute、response、latent absolute和两variant reader control KL。`final_c80_aggregate.py` 在CPU从保存张量重新计算raw项，并让所有checkpoint接受同一批显式loss configs，避免拿各自训练history或不同公式直接比较。完整80 cases、每suite20、两views等权；额外报告原first10 motor RMSE与future NMSE。这里衡量训练集目标优化，不能代替原heldout80与paired280。

CPU `final_c80_selfcheck25896560.pbs101` 已F/Exit0、artifact0、PHASE_COMPLETE与matching summary，实际00:02:53。小型teacher/input绑定、不同case缓存prompt切换、raw loss重算、reader两variant的1/2权重及完整80-case等权汇总检查通过；0真实模型forward、0训练。未提交实际final-C80 GPU评估。正式training config与loss系数仍待参数gate和完整C80有限步结果；11:36 SGT两项GPU句柄均fresh query_rc0/Q，继续原监控，不重提。
