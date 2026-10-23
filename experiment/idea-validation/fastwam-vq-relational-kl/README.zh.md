# 第一轮 temporal / spatial relational KL

目标：验证关系分布能否区分时间错序、空间错位和同幅整体偏置，再比较 soft-bin 与 soft-bin + temporal/spatial relational KL。

两组共享原 C80、原 held-out 40 cases × 2 seeds、BF16 teacher、初始 D=1/r=1、80 updates、update order、Adam/reset、learning rates 和原可训练参数。soft-bin 训练复用原 dual_kl final80 checkpoint（25878625.pbs101），补齐关系指标；关系组从同一原 baseline 重新训练。仅 first10 motor6 与 future latent（排除 observed frame0）进入目标。

Action 使用有顺序的3步窗口，形成8个18维token；teacher/student分别构造窗口距离的条件邻居概率，匹配 teacher||student KL。Latent 按两路camera分开2×2 pooling，形成2times×2views×7×7=196个48维token；邻接包括同camera同时间的8邻居、以及同camera同位置的另一个时间片。无普通camera接缝邻边，无action/latent时间插值硬对齐。

关系距离采用标准化后的每特征mean squared distance。温度取完整 C80 teacher 正值邻边距离的median并冻结。原soft-bin mean/std、alpha_a/alpha_z、lambda=rho*c全部复用。关系项beta只用teacher受控standardized Gaussian noise(std=0.1)确定：`beta = 0.25 * mean(noise soft-bin KL) / mean(noise relational KL)`，分别计算action/latent；不使用student输出或held-out。

最终目标：

`L = alpha_a*(action_KL + beta_a*temporal_KL) + rho*c*alpha_z*(latent_KL + beta_z*spatiotemporal_KL)`。

Synthetic selfcheck 和完整 C80 teacher 控错在 CPU PBS allocation 内运行。自检保留关系距离对整体平移、反号及部分时间方向变化的盲区反例，验证soft-bin绝对值项能补足；检查matching MSE但不同误差结构的区分能力、teacher detach、student有限非零梯度、ignored区域零梯度。真实teacher控错完整报告时间反转、latent frame swap、空间roll及匹配MSE的per-channel bias。此分布描述固定邻接关系，尚未估计完整action–latent policy联合密度。

两组 final80 均评完整 C80 80queries 和原 held-out80queries，全部输出统一soft-bin objective、两个关系KL和统一composite objective；RMSE/NMSE保留为诊断。仅固定final80，不按held-out选择参数或checkpoint，不创建新development set。新端点张量保存在compute端以便离线分析，避免未来重复forward。

CPU准备成功后才提交GPU。每个GPU作业walltime01:59:00、软截止6840秒，每15秒将GPU利用率和显存写入所属job.log。gpt-6-luna/xhigh subagent每600秒只读监控，Q/R无实质变化安静；不自动重提失败作业。所有计算/重I/O仅在真实PBS allocation，login仅轻量控制、状态和小代码文件。

结果只支持fixed-input decoded-BF16 VQ + dynamic fakeA4/BF16 KV的对比；没有闭环成功率、native低位kernel或完整联合密度结论。单条训练轨迹的变化按原40cases配对描述，80queries不当作80独立cases。
