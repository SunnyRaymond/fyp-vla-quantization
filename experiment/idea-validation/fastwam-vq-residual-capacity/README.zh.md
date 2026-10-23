# 按 future / reader 损伤分配 residual VQ 容量

问题：在相同 residual 容量和 metadata 预算下，保护会损伤 future、并经固定reader传播到action的位置，是否优于静态 saliency 或直接 action / joint Fisher？本轮做冻结权重上的容量移除与重分配诊断，不训练新 codebook，不预设结果为正。

**本轮已完整完成，结果不支持当前有符号 future/reader 损伤分配规则。** 组合规则在22/22个test cases上的固定reader first10和native first10均差于uniform；两种Fisher对照明显更好。结果与解释见文末，全部17种分配的原始汇总见[完整结果表](retrieved/results/analysis/REPORT.zh.md)。

## 固定对象与预算

使用 action-only epoch2 / 16 updates 的614个目标Linear，原4D、两本256-entry additive VQ、row scales、Smooth/H128、fake A4及sampler全部固定。第一本book始终保留；对第二本book原有index的保留位置重新分配，不改已有index值或centroid。

一个区域为连续4096个4D vectors，按照原decoder逐row padding。每个tile用固定两bit tag表示0%、50%或100% residual slots；50%使用偶数vector位置。Uniform每tile保留50%；其他策略把同一预算集中到高分tiles，必要时保留一个50% tile。所有策略均预留2048×tile_count个uint8 residual index slots；末tile未使用位置也计入padding。Base indices、两本codebooks、BF16 row scales、相同transform、分配tags及显式layout metadata均计入总bytes。Full two-book action-only只作为缓存参考，位于本轮稀疏匹配预算之外。

## Calibration评分与对照

仅原8个calibration cases/seed0决定评分，4 shards各2contexts。每个实际weight的adjoint先累加整个sampler的重复调用，再平方或与实际BF16 removal delta相乘。移除方向为精确的BF16 base decode减full decode；padding坐标不参与损失。

固定BF16 reader包括future→video prepare/prefill K/V→ActionDiT和proprio。先把缓存action-only future作为可微输入，得到reader action MSE对future的adjoint；完成backward后才切回student权重。然后通过student产生的future做VJP。该reader评分只有经future的路径，不包含直接改reader权重的路径。Action-only Fisher保留完整native action路径，二者分开比较。STE评分需要后续有限BF16干预验证，不能直接当作真实量化系统的导数结论。

17个预定等预算arms（14个全局配置，加3个逐Linear匹配容量的对照）：

| Arm | 分配依据 |
|---|---|
| uniform | 每个tile相同50% residual容量 |
| random0/1/2 | 三份预先固定seed的随机区域排序，分别报告 |
| magnitude | 原始BF16权重在相同transform坐标下的tile能量 |
| reconstruction | residual对局部weight squared reconstruction error的减少量 |
| local_output | 用calibration A4输入second moments加权的局部reconstruction改善，diag-Hessian近似 |
| activation_outlier | calibration变换后输入的channel max/RMS，映射到对应weight区域 |
| action_fisher | native first10 motor distillation gradient的diag empirical-Fisher × removal delta² |
| action_damage | 与future评分相同的一阶有符号准则，但使用native action完整路径，控制评分形式差异 |
| joint_fisher | 先合并g_action+lambda*g_latent，再平方乘removal delta²；lambda复用已有calibration冻结值 |
| latent_damage | 移除residual造成future NMSE上升的一阶有符号预测，跨calibration平均 |
| reader_damage | 经future传到固定BF16 reader的first10 motor MSE上升预测 |
| future_reader_damage | latent_damage与reader_damage的正值分别按总质量归一化，再等权组合 |

三个matched_module对照保留组合方法在每个Linear内full/half/zero tags的完整数量分布，只按random、reconstruction、joint Fisher改变tile位置。它们在查看calibration排名前预定，不增加backward。用它们判断收益是否超出layer/branch容量分配；完全相同的masks仅在CPU上用tensor equality确认，复用同一输出，不做额外model forward。

local_output为diag-Hessian surrogate，不声称完整GPTVQ算法。若latent和reader都没有正的一阶损伤信号，组合方法退回uniform并明确标记no signal。

Joint Fisher对照使用同一rollout distillation信息；它不是QuantWAMs原始co-training loss的复现。前者包含视频/动作目标的梯度交互项，是必要的强对照，不能只与action-only比较。

## 有限干预与完成条件

Allocation冻结后，在原完整60contexts上评估全部17arms：8 calibration contexts、4 selection cases×2seeds、22 test cases×2seeds。8 evaluation shards，paired seeds不拆开。每arm重新产生native future/action，再保持BF16 reader固定并重建K/V读取该future。逐query保存，两阶段分别可恢复；完整1020条结果；至多1020次native生成与1020次reader查询，完全相同的分配复用结果。评价先平均case内seeds，再等权平均22个test cases，不把44个contexts视为44个独立cases。

主指标为固定BF16 reader first10 motor RMSE；同时报告native first10/full32/late22/gripper、reader full32/late22/gripper、future NMSE、每case配对差值及leave-one-case-out均值范围。全部随机arms、失败cases与各split保留。若future/reader更好却native action恶化，报告tradeoff，不称为整体更优。分配Jaccard、calibration leave-one-context-out稳定性和module容量分布用于解释选择差异；排序差异本身不证明更有效。

只有完整8个calibration scores、17种等预算allocation、完整60contexts×17arms和配对分析均完成，才能结束本轮。若新评分未优于uniform、random或joint Fisher，也保留负结果。当前test过去已被查看，本轮属于机制诊断；不提供独立泛化、闭环成功率或native低位kernel性能证据。稀疏bytes是明确layout的容量核算，实际执行为decoded BF16/fake A4，不声称已实现压缩常驻或部署收益。

所有GPU作业单卡、walltime最多2h，CPU作业20min；每15s写GPU利用率与显存。模型、tensor、数值计算和重IO仅在真实PBS allocation中；login只连接、提交、查询和有界小文件。只有已终态、成功RESUME_READY的lane才能续交；不因Q注释或查询失败重提。Host-key verification保留，不输出credentials，不动用banked reset。独立subagent约10min只读监控，Q/R无实质变化静默，root负责代码、提交与结果解释。

## 文献与方法边界

- [QuantWAMs v1 §3.2](https://arxiv.org/html/2607.28405v1#S3.SS2)：joint video–action empirical-Fisher和layer精度分配已有先例。本轮拟检验的是冻结reader、经rollout future路径的residual容量需求，不宣称“joint saliency”本身新颖。
- [VPTQ v2 §3.2.2](https://arxiv.org/html/2409.17066v2#S3.SS2.SSS2)：residual VQ与outlier codebook已有先例。新意不能仅落在多一本book或保护outliers。
- [GPTVQ v2](https://arxiv.org/html/2402.15319v2)：Hessian-aware VQ assignment和局部输出误差优化已有先例。是否超出这些saliency需要本轮同预算有限干预结果，不能由相关性或命名判定。
- 实验设计流程参考 Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)。

当前权威job handles、运行阶段和下一步见RUN_STATE.json；未提交或未完成的部分不视为已有结果。

## 完整结果与判断（2026-10-17）

全部17 arms × 60 contexts = 1020条配对记录完成，8个evaluation shards均通过终态、Exit0、完成标记与完整覆盖检查；CPU分析job为25847002.pbs101，额外model forward为0。Test采用22 cases、每case两组paired seeds，先在case内平均，再等权平均cases。所有arms的目标Linear编码预算均为2,272,020,342 bytes，包括padding、codebooks、scales、tags和metadata。

下表均为test均值，越低越好。两列action RMSE均相对BF16 teacher：fixed-reader列保持整个BF16 reader固定，只替换future；native列同时包含当前容量分配对future和下游权重的影响。Future NMSE比较的是BF16 teacher预测latent，不是真实未来观测。

| 分配 | Fixed-reader first10 motor RMSE | Native first10 motor RMSE | Future NMSE |
|---|---:|---:|---:|
| uniform | 0.068938 | 0.092265 | 0.181304 |
| action_fisher | 0.040995 | 0.056460 | 0.099004 |
| joint_fisher | 0.045683 | 0.061760 | 0.103195 |
| action_damage | 0.160171 | 0.211751 | 0.288660 |
| latent_damage | 0.190910 | 0.172289 | 0.410076 |
| reader_damage | 0.179040 | 0.160330 | 0.312421 |
| future_reader_damage | 0.174579 | 0.189466 | 0.367432 |
| matched_module_random | 0.058538 | 0.106325 | 0.150748 |
| matched_module_reconstruction | 0.049498 | 0.104206 | 0.131813 |
| matched_module_joint_fisher | 0.046393 | 0.100516 | 0.113717 |

Magnitude、reconstruction、local_output和activation_outlier的fixed-reader first10分别为0.088762、0.086330、0.090314、0.113045，均未超过uniform；三份global random分别为0.071792、0.069524、0.072693。完整17-arm表、各horizon/gripper和配对差值保存在下方原始产物中。

### 负结果是否由少数case造成

组合规则相对uniform的fixed-reader first10平均差为+0.105641，配对差值中位数+0.102943，0/22 cases改善。每次排除一个case后，平均差仍为+0.102117至+0.109243；这只是leave-one-case-out敏感性范围，不是confidence interval。Native first10同样0/22改善，平均差+0.097201。单独latent_damage和reader_damage在fixed-reader first10上也均为0/22改善。

失败在calibration上已经出现：uniform、joint_fisher、组合规则的fixed-reader first10分别为0.058761、0.031872、0.187623；selection上分别为0.046360、0.023662、0.115010。因此不能仅用held-out分布变化解释。Calibration排序稳定性中，组合规则leave-one-context-out Jaccard均值约0.860，joint_fisher约0.983；更不稳定是一个现象，但尚不能认定是性能失败的根因。

### 对照把问题定位到了哪里

1. **模块内部的位置选择有明显问题。** 保持组合规则在每个Linear内full/half/zero tags的数量完全相同，只随机替换位置，fixed-reader first10就从0.174579降到0.058538；按reconstruction或joint Fisher排序则降到0.049498、0.046393。三个对照在该指标上均在22/22个cases优于组合规则。这排除了“差异全部来自layer/branch总容量”的解释，也未支持新评分在相同模块预算下识别有害future误差的能力。
2. **模块容量分配仍有native代价。** matched_module_joint_fisher的fixed-reader first10为0.046393，接近global joint_fisher的0.045683，但native first10为0.100516，对比global joint_fisher的0.061760；native first10 gripper也为0.111576，对比0.024353。这个对照显示reader可读性和完整native action保护需要分别约束，单看固定reader不够。
3. **失败与有符号一阶准则有关，不能只归因于future目标。** 使用同一种有符号评分、改为完整native action目标的action_damage也很差：reader/native first10为0.160171/0.211751。相同action目标下，非负action_fisher则为0.040995/0.056460。当前证据支持否定这套评分与大幅容量移除的组合；尚未区分符号抵消、有限步长下的曲率、多tile交互和STE误差各自的作用。

具体地，有符号评分按tile累加g·delta、再跨calibration平均，允许正负贡献抵消；Fisher对照先逐坐标平方(g·delta)，再累加，避免这种符号抵消。删除大量residual slots是有限干预，一阶局部方向未必能预测它。上述数学差异提供待检验的解释，不能由本轮整体mask结果直接当作已证实机制。

Action Fisher在test first10均值上最低，但Joint Fisher在full32、late22和gripper的部分均值上略好：例如native late22为0.073802，对比Action Fisher的0.075329。因此这里保留两者为强baseline，不将某一项均值最低写成全指标最优。缓存full two-book action-only参考的reader/native first10为0.041243/0.058110；它不满足本轮稀疏预算，且first10接近或略低于它不代表其他horizon、gripper或闭环无损。

### 这轮对contribution的含义与下一步

这轮没有建立future/reader-aware residual分配优于已有saliency的证据；也不能据此证明WAM必须使用VQ。它检验的是冻结action-only checkpoint上的容量移除配置，不能据此否定训练中的latent protection loss。Action Fisher本身包含action经future路径回传到world模块的梯度，因此它也可能保护future；“只用action loss评分”不等于“只保护action模块”。

下一步建议先校验评分与真实小块损伤的关系，再决定新allocation：只使用原8个calibration cases，在同一Linear内选取高正分、高负分、近零和高Fisher tiles，做等大小的单tile residual移除，同时测量真实native action、future NMSE和固定reader action变化。再用少量同时移除区分单块预测失败与多块交互。若要检查Taylor局部性，可加入小幅delta插值，明确它是连续权重诊断而不是可部署的VQ编码。只有评分能预测这些真实干预，才值得继续完整cohort或训练配置搜索。

本轮边界：固定输入、已被历史查看的cohort、decoded BF16/fake A4、无新增episodes；不报告闭环成功率、独立泛化、native kernel加速或实际压缩常驻显存收益。没有为了得到正结果再调评分或替换test cases。

原始产物：

- [全部17种分配结果表](retrieved/results/analysis/REPORT.zh.md)
- [完整均值、各horizon/gripper及配对对照](retrieved/results/analysis/primary_summary.json)
- [1020条逐context主指标CSV](retrieved/results/analysis/paired_primary.csv)
- [Calibration分配重合与稳定性](retrieved/allocation_summary.json)
- [运行状态与job receipts入口](RUN_STATE.json)
