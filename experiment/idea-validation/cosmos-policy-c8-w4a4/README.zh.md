# Cosmos Policy C8 W4A4 三组闭环实验

2026-10-16 已完成全部 840 个正式 episodes：BF16 233/280（83.21%）、SpinQuant-adapted 221/280（78.93%）、action+latent-aware Microscaling VQ A4 226/280（80.71%）。续跑 `25838062.pbs101` 已 F/Exit0，完整分母、paired coverage、四个 worker 与完成标记均通过。实际结果由前一作业的 120 个有效 slots 加续跑的 720 个构成。

完整表格、编码预算、性能限制与修复来源见 [RESULTS.zh.md](<D:/Downloads/Final Year Project/experiment/idea-validation/cosmos-policy-c8-w4a4/RESULTS.zh.md>)。VQ 的在线 activation 仍走逐 block FWHT；加载的 batched helper 没有接入此路径，原始 metadata 中的 batched 标签不能用作已优化延迟的证据。低比特臂均为 decoded-BF16/fake-A4 数值评估。

目标：先跑通原始 Cosmos Policy LIBERO direct policy，再在同一冻结的 LIBERO-Plus 280-variant cohort 上比较 BF16、SpinQuant W4A4、相似 effective bpw 的 action+latent-aware B32 Microscaling VQ A4，共840个episodes。

## 用户约定

- Smoke：独立单卡、每次 allocation <=2小时。
- SpinQuant calibration：独立单卡、每次 allocation <=2小时。
- VQ calibration：独立单卡、每次 allocation <=2小时；不与 SpinQuant 合并提交。
- 两个校准任务共享只读 C8 observations，各自保存模型产物和 checkpoints。
- 最终评测：单个4卡、24小时g3任务；按实测episode成本合理分配四卡负载，每完成一个episode即保存，保留完整3x280分母。
- 环境、下载、安装与其他重I/O仅在CPU compute allocation准备；login node只做小型控制、提交和状态查询。GPU利用率、显存每15秒写入job log。

## 固定比较口径

沿用已冻结的4 suites x 7扰动维度 x 每cell10 variants和配对环境/sampler seeds；Cosmos使用官方direct-policy配置：5步denoising、16步action chunk并执行16步、sigma=[80,4]、224双相机和proprio、JPEG/crop/flip预处理。沿用现有cohort的30步settling及400/700步episode上限；本实验是同条件配置比较，不宣称逐项复现paper总体成功率。

C8 source：原始LIBERO四个suites各2个demo snapshots，case IDs为2000、2019、2021、2025、2051、2057、2066、2070。只读取既有observation与metadata，使用Cosmos重新构造teacher actions/futures，不复用Fast-WAM模型输出。C8与Plus存在base-task重合，结果衡量校准后的扰动鲁棒性，不称unseen-task泛化。

BPW须统计两种方法同一target-module集合的weights编码、scales、codebooks、indices与transforms，并另报含未量化参数的DiT整体effective bpw。相似预算容忍范围与recipe在校准前冻结，不依闭环结果调整。当前实现阶段不宣称native INT4/FP4加速。

## 当前状态

2026-10-15 23:38 SGT：CPU准备任务`25828512.pbs101`已F/Exit0，通过CPU imports，policy、用户提供的官方tokenizer（507,609,880 bytes）和T5 snapshot均就绪。单卡任务已补完280条缺失embeddings，289条所需instructions全部覆盖；后续复用该cache。

截至2026-10-16 02:00 SGT，BF16最小闭环`25832557.pbs101`已F/Exit0且PIPELINE_COMPLETE：真实LIBERO Goal原始task0/state0/seed195成功，122个控制steps、8个replans、9次模型query（含1次额外的完整RGB路径parity reference），视频73,290 bytes；action-only与完整路径的动作差异为0。这个1-episode smoke不是正式280分母的一部分。

SpinQuant calibration `25832747.pbs101`、VQ calibration `25832754.pbs101`均独立单卡运行后F/Exit1，未产生量化更新：官方与可微分五步sampler的BF16 endpoint parity失败。两臂case2000/seed910004000数值相同：latent max_abs=0.0401641、RMSE=0.00358210；action max_abs=0.00266075、RMSE=0.000759382，当前rtol/atol=0.002。

02:17 SGT提交的SpinQuant `25833398.pbs101`、VQ `25833399.pbs101`均F/Exit1，0更新。严格A inference、B实际带梯度checkpoint的结果彼此相同；B梯度连接通过，但两者仍与oracle不同。新增诊断发现差异先出现在VAE clean image latents（max_abs=0.03125），而原始图像及归一化数值完全一致。

源码确认官方BTHWC→BCTHW transpose保留channels_last_3d，但原CPU capture强制改成默认contiguous。现全链条保留官方布局，parity/context版本4记录输入strides及PBS_JOBID，仍要求真实A/B均通过原2e-3，failed/旧cache强制重验。CPU检查`25833557.pbs101` F/Exit0，20项全通过，含uint8/BF16 capture→save/load→replay的值、dtype、exact strides回归。两个校准任务始终无闭环依赖。

SpinQuant `25833620.pbs101`的严格v4 A inference / B grad+checkpoint parity均通过：两路generated/original-clean latent max_abs均为0，action max_abs为5.96e-8，B梯度连接有效。8个teacher endpoints与40个timestep packets已保存。该作业首次训练backward时F/Exit1、0更新。CPU检查`25834243.pbs101`进一步证明仅关闭每Linear内层checkpoint仍会失败：learned R128首次lazy Cayley构造在SAC forward内多执行一次mm，重算时命中matrix cache跳过这次mm，导致缓存回放错位。现在在wrapper安装后、resume后、每次optimizer refresh后，在所有block checkpoint之外提前构造可微matrix，保留官方block SAC、FP32 STE与200-update full joint denoiser NMSE。CPU检查`25834371.pbs101` F/Exit0、CHECKS_COMPLETE，21项全通过，覆盖all-mm及官方按16计数策略各连续两个optimizer cycles、缓存重建和有限非零rotation梯度。

截至05:12 SGT，两组独立calibration均完成且PBS F/Exit0、summary complete、PIPELINE_COMPLETE一致：SpinQuant `25834379.pbs101`完成200/200 updates（3326.1秒），VQ `25833623.pbs101`完成16/16 updates（4125.6秒）。VQ自己的严格v4 A/B parity通过（latent max_abs=0，action max_abs=5.96e-8，B梯度连接有效），lambda固定为0.0040433477161343065。Spin实际运行metadata确认28/28官方SAC wrappers及predict2_2B_720 context，关闭每Linear内层checkpoint并在SAC外预热rotation matrix；训练峰值allocated约10.33GiB。

同一280层、1,761,607,680目标weights：SpinQuant 4.125446 bpw，VQ 4.286979 bpw，相差3.9155%，通过预先冻结的5%预算门槛。计入未量化DiT参数后分别5.307830和5.453279 bpw。以上为decoded-BF16/fake-A4数值比较的编码预算，不是native INT4 kernel或实际推理显存节省证据。正式单个四卡24h g3作业按28个10-variant bundles动态分配，三组各280 episodes。

05:14 SGT，正式评估已提交`25834649.pbs101`，PBS确认Q/g3、无hold/dependency，4GPU/24h；提交请求32CPU/220GB，PBS实际资源条目为64CPU/440GB。当前排队原因是node_pool资源不足，尚未产生正式episode结果；等待状态不会触发重复提交。完成判据为三臂各280、完整840覆盖及配对校验通过，并同时具备PBS F/Exit0、complete summary和PIPELINE_COMPLETE。

10:35 SGT，`25834649.pbs101`启动后8秒F/Exit1，首个Python进程启动前Apptainer报错：四个CUDA GPU UUID中的逗号被`--env`当成键值分隔符。没有加载模型、启动worker或产生正式episodes，不能计入成功或失败分母；两组完成的calibration产物继续复用。修复将CUDA列表通过命令局部`APPTAINERENV_CUDA_VISIBLE_DEVICES`传入，保留`--cleanenv`，并在CPU checks `25836549.pbs101`增加真实容器四UUID原样传递预检。预检通过后重提完整的单个4GPU/24h g3评估；controller仍要求旧handle已验证非零终态、明确repair note及所有前置完成证据。

11:11 SGT，CPU checks `25836549.pbs101`已F/Exit0（3分39秒），真实容器四UUID精确传递预检通过、21项unit tests全通过、CHECKS_COMPLETE及summary一致。完整替代评估`25836774.pbs101`（cosp1015evalr1）已提交并确认Q/g3、无dependency，4GPU/24h，仍为同一冻结的三臂各280。普通subagent监控按用户要求改为约20分钟，状态不变静默；四卡实际推理及正式结果待作业启动后验证。

`25836774.pbs101`已实际四卡运行1:26:32，最终F/Exit1；三臂均成功执行过完整闭环，并保存120个有效slots（每臂40/280）。四个worker随后均在sensor_noise的index270/noise35失败：官方Plus fog severity5的plasma_fractal使用NumPy2已移除的np.float_。异常episode未写入分母。任务局部恢复np.float_到同义np.float64，保留原fog算法、severity、cohort、seeds与两组calibration；异常日志补完整traceback。CPU检查`25837897.pbs101` F/Exit0、CHECKS_COMPLETE，23项全通过；其中真实index270的reset、官方state restore及step确实调用fog severity5与plasma_fractal两次，四UUID传递预检也再次通过。续跑按既有atomic episode机制验证并复用120个完整结果，执行剩余720；进度统一按worker_id唯一记录计数，最终仍要求完整840和三臂各280。

13:03 SGT，续跑作业`25838062.pbs101`（cosp1015evalr2）已提交并确认Q/g3、无dependency，仍4GPU/24h。同一840 cohort中复用120个完整slots，剩余720个待执行；监控仍约20分钟，最终结果需完整覆盖后再报告。

已修复loader的绝对config路径、隔离每作业容器内的`/tmp`，并仅在官方配置注册期间延迟两个随后被选定LIBERO路径覆盖的默认checkpoint下载。实际GPU任务已完整加载policy，missing/unexpected/incorrect-shape keys均为空；selector已统一官方`self_attn`/`cross_attn`命名的选择与coverage。实际build manifest包含280个目标：112self-attention、112cross-attention、56MLP，总计1,761,607,680个目标weights；两种attention的q/k/v/output均在scope内。旧VQ任务`25830849.pbs101`曾在初始化期间停止，避免重复已知scope错误。

校准直接读取官方只读C8指令embeddings，闭环任务复用已完成的Plus指令合并cache，三个任务可独立启动。CPU检查`25832553.pbs101`已F/Exit0，13项检查覆盖量化、独立C8缓存、配置路径与注册期resolver、以及官方self/cross-attention命名。先前CPU配置import预检`25832313.pbs101`虽13项unit检查通过，但因CPU无CUDA导致官方flash_attn import断言失败；该限制由实际GPU完整加载证据替代，未把CPU预检记为通过。此前一次未固定seed的bias一致性报错未复现，严格一致性验证仍保留。闭环及两个校准的完成门槛均已通过，正式作业handle见上。

复用Ubuntu22 SIF与LIBERO-Plus assets，不修改Fast-WAM或Cosmos3环境。`protocol.json`保存固定用户范围，各阶段的当前PBS handle与artifacts独立保存。量化评估使用真正导出编码产物解码出的BF16权重及fake A4；固定权重只在切换arm时解码，产物与解码权重缓存在host RAM。SpinQuant训练使用FP32 STE运算，部署使用BF16 operands；保存一层的差异probe，不能将两者称为逐位等价或native INT4加速。
