# 40 visited states：clip-only / joint / BF16 同输入诊断

使用已完成rollout-C80采集的全部40个visited states：原case2001–2079的奇数编号，四suite各10。每个state沿用其原seed_index0和sampler_seed；不采集新状态、不训练、不改原80-query held-out。两个量化checkpoint均为原C80的final80：clip-only和joint。

每个state预先构建一次5份模型输入，三组共享同一份输入tensor与采样噪声：base、两相机raw uint8 brightness ±1、raw EEF xyz沿(1,1,1)/sqrt(3)方向±5mm。Rotation与gripper qpos不改变。所有扰动都先作用于原始observation，再走原crop/resize/normalizer/BF16 preprocessing；记录实际模型输入变化与图像饱和比例。

主动作范围是实际执行的前10步。分别报告六个motor维度的normalized与反归一化command RMSE、MAE、signed bias、max abs。Gripper按实际部署的denormalize、符号翻转、sign阈值路径计算：-1=open、+1=close、0为阈值零指令；报告连续误差、400条配对命令的开合分歧/confusion、chunk内switch时序差。40个state才是case分母，10个动作和5个变体都是同case内重复测量。

局部敏感性使用两个预先固定方向的central finite difference，另保留正/负单侧响应。比较两模型与BF16的响应幅度、方向cosine、gain及响应/方向导数误差，并统计gripper对输入扰动的翻转。较小响应不自动视为更好；对teacher应有响应却保持恒定的模型，指标会记录不匹配。两条方向不代表完整Jacobian、真实camera viewpoint变化或闭环稳定性。

共40×3×5=600份配对端点：BF16 base39份直接复用验证过的teacher，首份fresh query核对teacher parity；另外5次原checkpoint reference或same-input repeat检查，预计566次新model queries。BF16必须先运行；随后只安装一次ClipBanks，分段恢复两套FP32 masters/ratios并冻结全部参数。每组正式评估前验证reference action，阶段结束后对首个输入同seed重复一次，以检查局部响应是否混入运行漂移。

一个GPU allocation：walltime01:59、软截止6840s、每15s记录GPU utilization/VRAM；不自动追加GPU作业。单进程复用模型和预处理输入，避免三次模型加载。gpt-6-luna/xhigh每10分钟读取当前权威handle，Q/R无实质变化时静默，完成/失败时报告。Login仅连接、提交、qstat和小控制文件；所有模型、数据处理与聚合只在真实PBS compute allocation。禁止banked reset，不输出credentials。

9个标准库测试覆盖完整600-record pairing、cross-input/seed拒绝、逐维误差、真实gripper符号、switch、完整160-row敏感性统计、响应方向和恒定student陷阱；传感器数值selfcheck与模型检查在PBS allocation内执行。

协议见protocol.json，当前状态见RUN_STATE.json/handle.json。端点逐条保存于remote results/records，逐case指标在case_metrics与sensitivity_cases；紧凑汇总和结果见results/comparison.json及RESULTS.zh.md。所有证据限于固定状态、decoded-BF16 VQ + fake A4的动作诊断，不据此声称闭环成功率变化。

已完成：GPU 25863989.pbs101，00:55:07、F/exit0；全部600端点和566次queries完成，六项reference/repeat误差均0。CPU报告25864600.pbs101用时8秒，仅压缩JSON与整理既有结果，没有新forward。monitor已停止，完成证据见completion_evidence.json。

先读results/ANALYSIS.zh.md了解结论与边界，再看results/RESULTS.zh.md的逐维表。joint的base motor与gripper整体改善，但proprio方向响应误差变差；case2045出现motor改善而前10步漏掉BF16释放指令的局部反例。不能据此直接归因之前的camera闭环净下降。report所需的compact JSON由postprocess.py在CPU allocation内生成，control.py pull读取compact版本；原始端点与未压缩汇总均保留在远端。
