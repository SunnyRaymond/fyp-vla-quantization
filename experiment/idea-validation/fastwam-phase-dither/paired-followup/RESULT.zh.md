# WAM W4A8：完整 action 输出误差分解

本目录保留最新周报 Main results 5 提及的输出分解。原 paired follow-up 同时包含的 task0 32-episode pilot 未在最新周报提及，其结果和原混合报告已移到可恢复归档。

## 完整输出误差分解

作业 `25667993.pbs101`：PBS F，Exit_status=0，wrapper exit=0，PIPELINE_COMPLETE存在；walltime **00:18:58**。全部194次forward完成：BF16 34、W4-only 32、RTN 64、learned phase 64。重放全部32个cachedBF16标签，最大差异0；repeat/hook-disabled identity最大差异均0，完整6,446-site执行路径匹配。

数据为已看过的TEST IDs8..15，每轨迹2个固定observations、2个sampler seeds；四个新冻结dither draws2101..2104交替配对两个sampler seeds。这是条件性探索诊断，不是新独立留出验证。完整192份BF16/W4/RTN/learned action输出已保存在本地502,143-byte `decompose_actions.pt`。

逐条件定义

$$\delta_W=a_{W4,A_{\rm original}}-a_{BF16},\qquad\delta_A=a_{W4,A8}-a_{W4,A_{\rm original}}.$$

原激活为实际BF16路径，并非均匀A16量化器。逐条件精确检查

$$\operatorname{MSE}_{total}=\operatorname{MSE}(\delta_W)+\operatorname{MSE}(\delta_A)+2\operatorname{mean}(\delta_W\delta_A).$$

最大恒等式残差 **1.08420×10⁻¹⁹**，通过1e-7容差。先轨迹内平均再等权平均8轨迹，所得结果如下；各项单位均为 **×10⁻⁴**。

| Arm | 权重路径差异项 | 激活增量项 | 两倍交互项 | 总action MSE |
|---|---:|---:|---:|---:|
| RTN W4A8 | 3.61510 | 0.13482 | +0.10264 | 3.85256 |
| Locked learned phase W4A8 | 3.61510 | 0.14112 | −0.00310 | 3.75312 |

这一新draw组上，learned相对RTN降低总MSE约 **2.58%**。其激活增量范数没有减少：增量项略升，但交互项从正0.10264降至接近0的轻微负值−0.00310，总MSE随之下降。因此，在本组条件下，收益主要体现为减轻权重路径误差与激活增量的正向耦合，而非单独减少激活增量幅度。

这只是两条完整非线性路径之差的精确代数分解，不将各项解读为独立因果占比，不把W4-only视为不可突破下界，也不把它等同于逐site residual cross-term机制已经被唯一识别。原pilot的learned MSE为3.58978×10⁻⁴、相对RTN改善6.82%；换成新draw组后为3.75312×10⁻⁴、相对同一RTN基线改善2.58%，说明原先给定draws的CI不能被当作跨整组新seeds的稳定性证据。原5.28%数字使用independent作为比较基线，不能直接与这里的2.58%混比。本轮未测新draw组的independent/shared0 arms，因此不声称其仍优于这些controls。

官方controller处理后，前10个执行命令的连续6维MSE：W4-only **1.87574×10⁻⁴**，RTN **1.89703×10⁻⁴**，learned **1.86188×10⁻⁴**；learned相对RTN改善约 **1.85%**。这组observations下所有比较的gripper sign mismatch均为0。它们是同一observation上的条件command误差，不是闭环任务收益。

## 证据与范围

- [冻结协议](PREREG.zh.md)、[运行记录](RUN_STATE.json)
- [完整分解结果](decompose_result.json)、[reference gate](decompose_reference_gate.json)
- [作业日志与 GPU 采样](decompose_job.log)、[实际完成标记](decompose_PIPELINE_COMPLETE)、[wrapper exit](decompose_exit_code.txt)
- 完整 action arrays 的本地文件为 decompose_actions.pt，按项目 *.pt 规则不进入 Git。
- Independent dither 的新 draws 对照后来另测，见 [A8 对照](../new-draw-a8/README.zh.md)。上面关于原分解作业未测该 arm 的声明保持其原始范围。
- 保留 closedloop.py 作为完整 LIBERO-Goal suite 的共享 helper；不保留 task0 pilot 的评分产物。

PBS Stageout_status=1；结果与日志从 scratch 直接取得，计算完成不表示标准 stageout 成功。未提供 native kernel 加速或全路径逐 site 机制的唯一归因。
