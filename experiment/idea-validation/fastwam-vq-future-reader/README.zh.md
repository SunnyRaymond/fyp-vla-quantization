# 固定 teacher reader 的 future 干预

用户授权：固定同一个 teacher 下游，比较 BF16、Parent、action-only、action+latent 的 future；复用 checkpoint/缓存，完整保留原 cohort/paired seeds。每个作业不超过2h。完成分析后，符合预期时继续两个 student 的交叉 future 干预。

本轮使用旧 row-output Parent、action-only epoch2 与 action+latent epoch2，均为同初始化和16 updates。它不是后来 C8/Microscaling 闭环配置的比较，不重训、不重新选择 checkpoint。

范围是8 calibration contexts、4 selection cases×2 seeds、22 test cases×2 seeds，共60 contexts。两个 shards各30，成对seeds不拆开；test先平均case内seeds，再对22cases等权。旧test已经看过，本轮是机制诊断。

第一阶段固定BF16 video prepare/prefill、ActionDiT及proprio，所有reader目标Linear直接走原始BF16 forward，重新构建每个donor的K/V。仅注入final clean future，frame0仍由真实observation clamp。复用旧缓存里的BF16/Parent/action-only futures；仅补生成action+latent future，并与旧selected action核对。每个context用teacher future重放BF16 action，检查相同初始noise与替换入口。

主指标first10 motor RMSE，另报full32/late22/gripper。比较action-only−Parent用于判断训练后的额外漂移，action+latent−action-only用于判断保护是否修复；两者分别报告。

进入交叉阶段的预定工程门槛：固定teacher reader下，action+latent在完整22-case test的first10均值低于action-only，且至少12/22 cases改善。它不是统计显著性判据，也不要求action-only必须劣于Parent；后者若不成立，要明确缩小结论。交叉阶段固定action-only或action+latent reader，分别读取两者future，完整2×2，不能只报告交叉成功的cases。

先以CPU-only20min作业核实旧缓存/所选checkpoint并准备小型索引；GPU每段1卡/2h，逐query保存，安全余量150秒，只有当前handle已终态且RESUME_READY时才能续交；不因查询失败或排队注释重提。每15秒GPU利用率/显存写job log。汇总在CPU-only作业内。Login仅连接、PBS和小文件操作；不读取tensor/模型、不重I/O，不输出credentials、不绕过host-key验证、不动用banked reset。

本轮不提供native低位kernel、闭环成功率、真实未来准确度或独立泛化证据。执行状态与权威handle见本目录的`*_handle.json`；配置见`protocol.json`。

## 第一阶段实测结果

两份GPU作业25839688/25839689均F/Exit0，分别用时40m10s、50m14s；CPU汇总25841012已完成。完整60 contexts、240条reader结果，无缩减。60次BF16-future回放的最大action差异为0。共补生成60次action+latent完整输出，并执行240次future注入reader查询；无新训练、无执行episode。

以下为22个test cases、每case两seed先平均，再对case等权；RMSE均相对同输入同seed的BF16 action。

| Future donor | 固定BF16 reader的first10 motor | full32 motor | late22 motor | first10 gripper |
|---|---:|---:|---:|---:|
| BF16 | 0 | 0 | 0 | 0 |
| Parent | 0.036144691 | 0.048867316 | 0.052088175 | 0.001903076 |
| action-only | 0.041243219 | 0.052907799 | 0.056283144 | 0.001630140 |
| action+latent | 0.035083670 | 0.043542794 | 0.045684393 | 0.001868341 |

action+latent相对action-only的first10 motor均值下降约14.9%，13/22 cases改善，配对差值中位数−0.003881685；full32和late22各14/22改善。gripper均值反而上升，10改善、11变差、1相同。action-only相对Parent的first10均值变差，但case胜负为11:11；不能称为普遍退化。action+latent相对Parent的first10均值略好，但仅9/22 cases改善，不能称为普遍优于未恢复的Parent。

这支持一个有限结论：在这一固定cohort上，action+latent产生的future对同一个原始BF16下游更有利；收益并非只能依靠它自己的student下游读取。这里定位的是final future作为整体对action的影响，尚未定位语义内容、空间区域、时间切片或cache中的具体有害方向。

原定工程门槛已通过，因此继续完整student reader×future的2×2交叉实验。四份作业为25841031/25841032（action-only reader）和25841033/25841036（action+latent reader），均申请单GPU、2h；复用本轮全部future，不再生成future。实际状态见RUN_STATE.json及各handle。

完整第一阶段机器汇总、各split与配对明细见[报告](retrieved/results/analysis_teacher/REPORT.zh.md)、[analysis.json](retrieved/results/analysis_teacher/analysis.json)和[CSV](retrieved/results/analysis_teacher/paired_contexts.csv)。

## 交叉阶段实测结果与解释

上述四份GPU作业均已F/Exit0并验证完成；实际用时30m24s、31m54s、31m11s、30m49s。CPU汇总25842013也已完成，完整240条交叉结果。两种student各60次self-future回放，共120次，相对所选旧checkpoint的action最大差异均为0。

同一行内固定完整下游（future→prepare/prefill K/V→action），仅替换future。下表均为test22 cases/44 contexts的first10 motor RMSE；改善case数相对该行的action-only future。

| 固定下游 | action-only future | action+latent future | 改善cases | 配对差值中位数 |
|---|---:|---:|---:|---:|
| BF16 teacher | 0.041243219 | 0.035083670 | 13/22 | −0.003881685 |
| action-only student | 0.058110442 | 0.053481176 | 10/22 | +0.000549668 |
| action+latent student | 0.058676505 | 0.052951009 | 12/22 | −0.000413754 |

平均motor收益在三个固定reader下均保留。这排除了“只有action+latent自己的reader更会读取自己的future”作为平均收益的唯一解释。尤其固定action-only reader，单独换future就能将0.058110442降至0.053481176；再换成action+latent reader才到0.052951009。这里只是两条实际干预路径的比较，不把它表述为唯一的因果中介占比，也不把nonlinear系统的误差强行相加分解。

收益并不均匀：action-only reader下，12/22 cases变差，中位数为正；action+latent reader下，10/22变差。case1016、1025在两个student reader下都明显改善，而1017、1022在两者下都变差，提示下一步应检查future误差的具体结构和失败条件。仅凭这些case不能指定物体、接触关系或时间语义为机制。

完整horizon结果也有边界。action-only reader换用action+latent future后，full32均值0.069571863→0.060817783（14/22改善），late22为0.073341530→0.062988386（15/22改善）。action+latent reader下full32为0.069477049→0.061331653（12/22改善），late22为0.072971262→0.063866255（11/22改善）。gripper均值在action-only reader下略好（0.018109958→0.017981764），在action+latent reader下反而变差（0.018386650→0.018825540）；不支持统一的全action收益。

当前可支持的发现是：**action-only恢复后的final future包含会被下游读成action偏差的误差；联合latent保护能在本cohort上降低这部分平均motor误差，而且改善能跨reader保留，但有明确的case与action维度异质性。** 这不是VQ优于scalar quantization的证明，也尚未识别“哪类world-representation drift”或证明真实物理预测更准确。

接下来应先做有针对性的机制定位，再搜索latent loss配置：复用这些缓存future，固定reader，将action-only→action+latent的修正按两个future时间切片或预定义空间块逐组加入，再反向移除。用双向patch检查哪部分修正足以改善、移除后是否失去收益；保持frame0、输入和action noise，重新prefill cache。整段替换是本轮已有正负对照。若要比较“关键部分”与随机部分，需增加相同误差能量的随机/非关键修正对照，避免把修正幅度误当方向或语义重要性。只用calibration选择分组/权重；这份已见test继续作为诊断，不再当作独立泛化证据。

定位得到可重复的敏感部分后，再比较dense latent MSE、按calibration敏感度加权的latent loss、同权重预算的随机加权对照。固定同一初始化、16 updates、action loss和checkpoint选择规则，另备未参与选择的新验证数据及闭环评估。当前结果不支持立即开展大范围lambda/grid search，也不支持先写成“必须使用VQ”。

完整交叉汇总与明细见[报告](retrieved/results/analysis_cross/REPORT.zh.md)、[analysis.json](retrieved/results/analysis_cross/analysis.json)和[CSV](retrieved/results/analysis_cross/paired_contexts.csv)。所有本轮reader结果合计480条，另有60次action+latent完整输出生成；监控已结束，无待续跑lane。
