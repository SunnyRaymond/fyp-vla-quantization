# 第一轮：保留 MSE/NMSE，增加 latent teacher-anchor KL

用户已授权完成第一轮实验，取消此前单作业2小时限制；使用 gpt-6-luna / xhigh subagent 每20分钟只读监控。完整目标包括训练、原固定输入评估、原280个 paired LIBERO-Plus episodes及结果解释，提交成功不等于实验完成。

## 冻结的两组比较

| Arm | 目标 | 训练 |
|---|---|---|
| mse_nmse | first10 motor6 MSE + rho*c*future global NMSE | 复用原相同初始化与80 updates的per-channel checkpoint，完整重放160个端点 |
| anchor_kl | 原MSE + 原NMSE + beta*latent teacher-anchor KL | 从原C8 rho1.5、D=1/r=1重新训练80 updates |

保持原C80 cases2000–2079、固定update order、seed0、共享BF16 teacher、rho=1.5、c=0.028303017887797623、Adam/learning rates/bounds/首步后reset与原可训练参数。仅训练614个clipping ratios及1,806,863个per-channel log2D；books、indices、row gains、B32 exponents、backbone和BF16 KV不变。固定final80，不使用heldout或闭环选checkpoint。

## 新KL的含义

对future latent按view分别2×2 pooling；使用原完整C80 teacher固定channel mean/std。Fast-WAM adapter配置为两个宽14的view、48channels、两个future时间位置；`anchor_loss.py`本身支持其他batch/channel/time/view布局。observed frame0排除。

每个token的候选为同view同时间8邻域、同view同位置相邻future时间；排除self与camera接缝。teacher分布的距离为teacher_i–teacher_j，student分布的距离为student_i–固定teacher_j，二者使用相同softmax(-distance/temperature)，优化KL(teacher||student)。此概率描述条件邻居关系，不是实际未来/动作的生成密度，也不保证自动强调物体或接触点。

温度复用原完整C80 teacher正邻边距离median，只来自calibration。GPU更新前对完整80个初始化contexts计算实际新KL及原NMSE，冻结

`beta = 0.1 * (rho*c) * mean_initial_latent_NMSE / mean_initial_anchor_KL`。

原rho/c不重算。10%是初始化loss值配平，不代表10%梯度贡献。初始化配平forward不做参数更新；之后恢复首个原baseline parity query后的RNG，再训练。已有原baseline八十个输入端点逐一重放，权重/输入/teacher/seeds与原指标必须匹配。

## 评估与证据边界

每组完整C80 calibration80和原heldout40cases×2seeds=80queries；报告motor、future NMSE、新KL、共同原目标/共同新增目标、full32/late22、实际gripper开合和事件。先平均case内seeds，再suite内平均cases、四suite等权。完整action/future端点保存在compute端，后处理无需重新forward。

闭环复用原280-row manifest、state0、环境/采样seeds、30步settling、每次执行前10步、400/700 cap。新方法两个workers各140episodes；原MSE+NMSE的203/280逐episode references复用。报告overall、suite、dimension、cell和paired wins/losses/ties。即使固定输入结果不佳，也完成本轮已授权的完整闭环；不自动启动ablation/sweep。

这些cohorts已用于前期工程诊断，不宣称独立泛化、未见任务或统计显著性。方法接口可迁移不等于已验证其他WAM效果；后续跨baseline确认需另行实验。执行为decoded-BF16 VQ + dynamic fakeA4 + BF16 KV，不是native INT4 kernel速度/显存证据。

## 资源与状态

Remote root：`/scratch/users/ntu/yguo017/fastwam-vq-teacher-anchor-kl-20261021`。

- CPU prepare/aggregate：4CPU、32GB、45分钟，均只在compute allocation运行。
- MSE对照重放、新方法初始化/训练/评估：各1GPU、16CPU、110GB、6小时。
- 新方法闭环：同一allocation中2GPU、32CPU、220GB、24小时，两worker并行，完整episode原子保存；失败或软停后保留完成单位，核实terminal后才能续跑。
- 每15秒采集分配GPU的utilization与VRAM到job.log；监控subagent每1200秒读取权威handles、qstat、进度及完成凭证，普通Q/R静默，查询失败不授权重提。

CPU25902483.pbs101已F/Exit0，3分01秒完成完整C80 teacher定义检查：self KL=0，spatial-roll均值0.657848391、time-swap0.365453346、匹配幅度channel-bias0.044196485；temperature=0.4224328100681305，归一化teacher entropy均值0.892426594。0新增model forwards、0新增teacher queries。

GPU对照25902526.pbs101、新方法25902527.pbs101已提交。后续句柄和fresh状态见handles.json、last_status.json；监控状态见monitor_state.json。所有后续结果以实际完成凭证为准。

Login node仅轻量连接、提交/查询和小控制文件；所有Torch、模型/张量、重IO与真实汇总都要求PBS_JOBID、非login hostname、PBS_NODEFILE成员检查。保留SSH host-key verification，不输出credentials，不使用banked reset。
