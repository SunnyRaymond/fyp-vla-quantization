# Per-channel：为什么 motor RMSE 改善没有换来更多闭环成功

结论：未发现错误 checkpoint、漏加载 D/ratios、重复权重补偿或 first10 执行范围不一致的证据。Per-channel 改善了固定观测的六维 motor 逼近，但同时恶化了未监督的 gripper 连续输出；在 Plus 初始输入上，其平均 motor 收益也与误差尾部、部分方向的 chunk 偏置脱钩。后续状态与局部响应没有被当前 loss 约束。它们是有依据的排查方向，尚不能将本轮净少4个成功全部归因于其中一个。

**实际结果与比较边界**

原始 held-out 为40个 demo contexts×2 seeds=80 queries；Plus 闭环为四 suites×七扰动维度×十 variants=280 episodes。不是同一批输入。所有模型的闭环初始协议、环境和sampler seeds配对。

| 配置 | 原 held-out motor RMSE | 原 held-out latent NMSE | Plus成功数 |
|---|---:|---:|---:|
| C8 rho1.5 baseline | 0.044296774 | 0.060877371 | 206/280 |
| C80 clip_only | 0.043848437 | 0.054848124 | 207/280 |
| C80 per_channel_clip | 0.031641466 | 0.055377875 | 203/280 |

Per-channel相对clip-only的配对结果为11胜/15负/254平，净少4例。相对BF16为8胜/21负/251平；BF16自身为216/280，并非成功oracle。未做重复rollout，不能将203与207的小差异解释成稳定或显著劣势。这里已经足以判断：原RMSE的大幅收益没有在这次cohort上转化成净成功数提升。

**实现核查**

已将下面涉及的四份运行时代码与实际PBS25863760.pbs101保存的artifact文本对照，一致；见`runtime_source_review.json`。没有对大模型文件做重复hash。

- `vq_adapter.py`加载源训练25862598.pbs101的final80，恢复原C8 rho1.5 weights、614个ratios、1,806,863个FP32 per-input-channel log2D。它是per-channel scaling + per-Linear clipping；ratio并非每个channel一个。
- 先在A4前除D，再把原decoded BF16 Wq乘D后转BF16；正式episodes前缓存补偿权重一次。源七项endpoint指标在四个workers均精确复现；源sampler、uncached/cached完整action/future、正式infer的action检查均通过。检查输入为case2000，不能将其描述为对280个输入逐一做了cross-runtime parity。
- 训练实际使用`action[:10,:6]`；闭环预测32步、执行前10步后用真实obs重新查询。前10步覆盖与执行一致，不能把late22缺监督当作本轮主要解释。
- 实际command复用同一父runtime：先denormalize，再对gripper做`2g-1`、invert、`np.sign`。没有发现per-channel单独改变gripper符号或normalizer的代码。
- Timeout按400/700个环境control steps，不按query walltime；当前没有依据把成功率下降解释成scaling增加了推理延迟。

**新增的零forward离线分析**

CPU PBS25868135.pbs101，F/Exit0、artifact0、ANALYSIS_COMPLETE，耗时5秒。只解析已有episode的`first_action_chunks[0]`，没有训练、新teacher、新model forward或新episode。四组各280条记录按variant/task/state/environment seed/sampler seed配对。没有保存原像素供本次重新逐像素核对；起始输入对齐来自既定评测协议。第二轮以后各模型已到达不同状态，因此没有直接相减它们后续的动作。

| Plus起始查询，逐episode算误差再平均 | clip-only | per-channel |
|---|---:|---:|
| First10×6 motor RMSE均值 | 0.054872329 | 0.048012932 |
| Motor RMSE中位数 | 0.046641959 | 0.037668067 |
| Motor RMSE P90，nearest rank | 0.083565356 | 0.084956917 |
| Motor RMSE最大值 | 0.240574510 | 0.241700506 |
| First10 gripper连续normalized RMSE | 0.014513651 | 0.041161356 |

Motor均值仍改善约12.5%，221/280个起始queries更低，四个suites的均值都改善。七个扰动维度的起始motor均值也都更低。因此，“离开original demo后motor完全失效”不成立。不过P90和最大值没有改善，camera/background/sensor-noise下收益也小于其他维度。原始held-out的平均收益不能直接等同于Plus最坏输入的保护。

每个起始query先计算前10步各维平均误差，再取绝对值并平均episodes，用于避免不同episodes正负偏置抵消。尽管六个维度的RMSE均值都下降，dx、dz、rotation_y、rotation_z四维的chunk偏置指标反而上升。例如dx从0.039461343升至0.041754679，rotation_y从0.023093970升至0.024249300。这说明RMSE下降没有同时保护所有方向的持续偏置；不能直接将它称为已测得的闭环漂移。

15个clip成功/per-channel失败的episodes中，11个per-channel的起始motor RMSE更低。该失败子集的平均motor为0.055352939→0.050393292，gripper连续RMSE为0.013804550→0.038780706。更接近BF16的初始六维动作，仍可能对应不同后续成败。

连续gripper RMSE增加不等于已经证明开合指令增加错误：最终sign会丢弃同侧幅度变化。这个离线脚本没有把normalized数值的符号冒充实际denormalized command；实际开合与切换时机需要另外核查。原始held-out上gripper也从clip-only的0.059316845升至0.078307935，说明这项盲区不只出现在Plus起点。

原始记录见`initial_action_diagnostic.json`；执行门槛见`initial_action_analysis_status.json`。

**原因的证据强弱**

1. 已确认的目标盲区。`worker.py`的motor loss与评估指标都只包含前6维，gripper第7维没有直接监督。D作用于各Linear的输入channels，包含最终action head输入；即使Wq冻结，它仍会改变七个输出的实际量化函数。因此“冻结weights”不会自动保护gripper。实际部署将其阈值化，抓取/释放时机可能比连续motor平均误差更影响成功；目前没有本轮失败时刻的观测/指令证据，不能称为已证实根因。

2. 单点拟合不约束邻域响应。`channel_banks.py`使用round/clamp及dynamic absmax。D/r改变量化分区；完整coupled STE检查证明实现符合声明的surrogate，不保证真实forward对相邻输入的有限差分接近BF16。1.8M个D参数提升单点拟合自由度，范围[0.5,2]也不等于局部响应受约束。已有40-state visited诊断确实观察到clip/joint的响应与BF16不匹配以及motor改善却漏释放的例子，但没有评估per-channel，不能移植其数值或病例作为本轮因果证据。

3. 状态覆盖仍有缺口。C80来自original demos的稀疏阶段快照；Plus还包含视觉、proprio和language变化，以及各student执行后诱导出的中后期状态。每次chunk后重新闭环不能消除这种分布差异。DAgger讨论了动作影响后续观测的非i.i.d.问题，这是理论动机，不是我们已经定位了哪次失败：https://proceedings.mlr.press/v15/ross11a.html 。

4. 全局latent NMSE不是控制代价。相对clip-only，Goal/LIBERO10的原held-out latent退化与闭环少4/5例方向一致；但Spatial latent也退化，闭环却多3例。Video/KV可影响同次action推断，但runtime只把action送入env.step，没有用全局NMSE做规划筛选。因此不能把suite相关性或调高rho当作已经确定的修复。

**下一步顺序，GPU工作尚未启动**

先补一个信息量高且能复用大量参考的诊断：把本轮per-channel加入既有40 visited states×5输入变体，新增200个student端点；BF16/clip/joint端点和共享noise已有缓存。使用真实`runtime.command`报告前10步的开合分歧、释放/闭合时刻、逐维command误差与chunk偏置，以及teacher-referenced局部差分。两个扰动方向仍不能代表完整Jacobian或真实camera变换。保持原80-query held-out不变，不新增development set。

若实际gripper命令/时机更差，首个训练修复只增加独立gripper监督，保持D/r、原C80、80updates、从原baseline的D1/r1初始化和其余loss不变。采用`L_motor6 + lambda_g*L_gripper + rho*c*L_latent`，保持原motor6与latent系数；不直接换成七维mean而暗中缩放原motor权重。先使用已有teacher完整第7维的连续监督；需要sign margin时，使用实际阈值前的可微命令而不是`np.sign`参与反传，teacher接近阈值时降低符号标签权重。系数仅由calibration确定并冻结。要求motor收益、gripper开合/时机和latent保护共同报告，最终仍以原paired闭环比较验证。

若gripper命令没有明显问题而输入响应恶化，优先加入teacher-referenced response loss：`||[a_q(x+delta)-a_q(x)]-[a_BF16(x+delta)-a_BF16(x)]||^2`，各模型共享输入与噪声。不要只压小student响应，也不除以接近零的teacher响应。数据/teacher来自calibration侧，不能拿这次测试失败当训练样本。

分支粒度仍是后续消融：action_expert用per-channel，video_expert/其余保留B32，测试motor收益对video侧自由度的依赖。但根据现在的闭环证据，我会把它放在gripper/响应诊断之后；它本身不补齐未监督第7维或局部响应目标。

不建议直接扩训练步数、只调rho，或同时换大数据集/全量codebooks/新loss。已有joint_per_channel已经得到motor0.032668161、latent0.048842217，gripper却为0.080485551，尚无闭环结果；latent变好仍不能推出控制问题解决。已有40anchor+40短程visited的C80 joint训练也没有改善原held-out，不能再笼统承诺“换rollout calibration就好”。若需要增加数据，应保留anchors，补独立calibration中的接触、抓取、释放和恢复阶段及输入邻域，并分别控制contexts和updates。

当前episode只保存前三个action chunks，没有逐replan原始observation/环境state。要定位后期失败，应在后续预先定义的诊断rollout保存可重放states、命令和阶段事件，并在同一student visited state上查询BF16；不要对不同模型各自轨迹上的动作直接算RMSE。故障分析可看这15个失败，但训练样本必须另取calibration轨迹。

本次新工作只有5秒CPU离线分析，新增model forwards=0。所有建议中的GPU诊断、训练和闭环均未自动执行。结果限定为decoded-BF16 VQ+fake A4，不涉及native-kernel收益或显著性结论。
