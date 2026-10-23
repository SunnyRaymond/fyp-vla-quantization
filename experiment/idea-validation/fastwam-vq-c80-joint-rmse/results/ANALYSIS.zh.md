# C80 joint final80：结果解释

## 2026-10-18：已有 episode 的机制诊断与下一步

本次只解析已完成的日志：CPU PBS 25862484.pbs101，walltime 12秒，scheduler F/exit0、artifact exit0、ANALYSIS_COMPLETE。训练、teacher查询、新模型forward均为0。原80个held-out paired queries、seeds及teacher未改变。详细数据见mechanism_diagnostic.json。

关键发现：不能仅凭camera闭环净少5次成功，就断言joint在camera初始输入上的动作逼近变差。利用原episode保存的first_action_chunks[0]，在相同variant、初始状态协议、环境seed和sampler seed下，与BF16第一轮动作比较，得到normalized first10×6 motor RMSE：

| 已有闭环的初始查询 | clip_only | joint |
|---|---:|---:|
| 全280 variants平均 | 0.054872329 | 0.052365216 |
| camera_viewpoints 40 variants平均 | 0.066728711 | 0.063280278 |
| camera中位数 | 0.055298922 | 0.051254596 |
| camera P90，nearest rank | 0.095338812 | 0.085143274 |

这些是闭环已有起始查询的补充诊断，不能替代原80-query held-out分母；也没有比较第二轮以后的BF16和student动作，因为各自执行动作后输入状态已不同。起始状态按既定paired协议对齐，没有保存像素用来额外验证逐像素一致。

camera的5个clip成功/joint失败案例，全部BF16成功，且joint全部达到控制步数上限；分布为spatial 1、object 1、goal 1、libero_10 2，并非单一任务或单一suite。3/5案例的joint初始motor RMSE反而更低：

| 原episode index | 任务简称 | clip初始RMSE | joint初始RMSE | clip/joint控制步数 |
|---|---|---:|---:|---:|
| 17 | ramekin上的碗放到盘子 | 0.172022 | 0.202780 | 351/400 |
| 86 | tomato sauce放入篮子 | 0.067366 | 0.065329 | 159/400 |
| 158 | 打开中间抽屉 | 0.072576 | 0.058180 | 255/400 |
| 222 | cream cheese和butter放入篮子 | 0.055584 | 0.068860 | 332/700 |
| 226 | 两个moka pots放到炉上 | 0.047738 | 0.037550 | 392/700 |

原40个held-out contexts先平均两seed后，motor中位数/P90/最大值也都改善：clip为0.042524/0.051281/0.057055，joint为0.041168/0.049327/0.053091。因此没有证据把矛盾简单归因于“平均值被少数好样本拉低、整个初始误差尾部变坏”。但是相对clip仍有13/40 contexts的motor误差增大；object suite均值也变差。全局latent改善亦不保证每个context的motor改善，例如3018的motor RMSE增加0.010069，同时latent NMSE减少0.063041；这是目标可能发生取舍的描述性例子，不能据此选训练样本或rho。

解释优先级：

1. 当前证据更符合固定观测的数值逼近目标与闭环控制目标不完全一致。动作改变后续状态；接近、接触、抓取、释放等时刻的误差方向和偏置可以比平均平方误差更影响是否成功。现有日志只能证明timeout，不能证明具体发生了漏抓、错放或方向性偏置。
2. Calibration覆盖了40个任务，但每任务只有两个独立demo快照，按0.2/0.5/0.8轨迹位置分层；它既没有覆盖原任务之外的全部Plus扰动，也不保证覆盖量化policy偏离demo后会到达的恢复状态。新增更多相似demo快照未必补上这个缺口。初始camera RMSE已改善，所以应把后续可达状态覆盖放在单纯增加camera初始图像数量之前。
3. 当前loss是first10×6 motor MSE加全局future latent NMSE。六个normalized motor维度的误差不是环境控制代价，latent各位置的误差也不等于动作敏感度。Gripper没有直接监督，但其原held-out平均RMSE实际改善；应先查执行后符号/开合时机，不能先认定它是根因。实际闭环只执行前10步后重规划，late22不作为本次改进方向。
4. 不能把joint简单解释为传统训练集过拟合：calibration和原held-out的motor/latent均改善。它仍可能对原数据的状态分布、共享codebook与ratio的补偿方式形成偏好。joint相对clip改变了weights和最终ratios的共同配置，当前对比未隔离哪个参数组导致闭环差异。
5. 9 wins/14 losses是单个冻结cohort的点估计，camera是看过7个维度后发现的定位线索。不同扰动共享40个基础任务，不能把所有queries/variants当作独立泛化重复，也不宣称统计显著或因果归因。

下一步先改calibration覆盖，再扩大activation参数自由度。建议先固定80个calibration contexts和80 updates、原rho、初始化和optimizer预算，仅比较原demo采样与含policy可达状态的分层采样。保留40任务的覆盖和一部分原demo锚点；优先加入接触/抓取/放置阶段以及预先定义camera扰动下的中后段状态。新采样应来自独立calibration demos/seeds/variants，不能把上面5个测试失败直接搬入训练。发生新观测时需BF16 teacher在该观测上生成新标签，标签只生成一次并由所有候选共享；不能拿原图的teacher输出当新视角/新状态的正确标签。若无法补充teacher，这条路线只能作为后续方案，不能声称现有缓存已经完成了它。

先保持loss不变来隔离数据覆盖的作用；随后若闭环仍脱钩，再单独加入受约束的更新或控制相关目标。可检查执行后的gripper符号、关键阶段motor偏置、同一student可达观测上的BF16动作差异；有证据后再增加相应监督/阶段权重。先不同时改rho、训练数据量、训练步数和channel scaling。DAgger的状态分布讨论提供这一路线的理论动机，但不是我们已证明的失败机制：https://proceedings.mlr.press/v15/ross11a.html 。BF16在camera本身仅21/40成功，teacher匹配并不等于任务最优。

Activation下一步先做受约束的连续B32 scaling+clipping，冻结books/indices/row gains，以同一父checkpoint的clipping继续训练作为匹配对照。D从1初始化，用较小范围或log(D)正则限制偏移；不要第一轮同时释放全per-channel D和全部codebooks。已有离散2^k B32实验只改变13/56466组，并未充分检验连续channel scaling；其held-out motor比clip恶化0.315%，latent改善1.270%，不能把这个双指标取舍判为成功。

连续D的实现需显式保留连续scale并使train/eval使用同一路径，不能声称它可无损折入整数B32 exponent；需计入额外metadata及运行成本。对weight-VQ误差、activation误差和最终Linear输出误差做共同诊断，再看原80-query最终action/latent和闭环。SmoothQuant和OmniQuant支持等价变换/联合量化校准的动机，但不提供我们这个Fast-WAM VQ闭环收益的直接证据：https://proceedings.mlr.press/v202/xiao23c.html ，https://arxiv.org/abs/2308.13137 。

原80 held-out、paired seeds、原teacher和既定闭环cohort保持为比较基准，不新增development set、不用held-out挑checkpoint。之后GPU探索仍须按用户的<2h预算做可行性安排并由gpt-6-luna每10分钟监控；数据采集、teacher和完整原80评估都计入预算，未测耗时前不承诺完整闭环可在该预算内完成。本轮未启动这些未来实验。

## 闭环结果补充：RMSE改善没有转化为成功率改善

来源线程随后完成了同一批两组checkpoint的LIBERO-Plus闭环评估。已只读核对其summary与完成门槛：joint评估job25858606.pbs101为F/exit0、artifact exit0、PIPELINE_COMPLETE；共同summary覆盖560个新episodes和2240个参考slots。本评估线程没有新增模型forward或重跑episode。

以下两类评估的输入分布和任务不同：RMSE来自原始demonstration固定输入40 contexts×2 seeds；闭环来自LIBERO-Plus的280个配对扰动episodes，不能把它们视为同一批样本。

| 配置 | 原held-out first10 motor RMSE↓ | 原held-out latent NMSE↓ | Plus闭环成功率↑ |
|---|---:|---:|---:|
| C80 clip_only80 | 0.043848437 | 0.054848124 | 207/280 = 73.93% |
| C80 joint80 | 0.041846837 | 0.048973336 | 202/280 = 72.14% |

joint相对clip_only的闭环paired计数为9 wins、14 losses、257 ties，总计净少5个成功episodes，差−1.79个百分点。camera_viewpoints为joint12/40、clip_only17/40，paired 0 wins/5 losses/35 ties；其余六个维度的净差总计为0，其中background +1与robot_initial_states −1抵消。因此准确表述是“总净下降由camera维度的净差解释”，不是说其他维度没有单episode变化。

参考点：BF16为216/280（77.14%），共同初始化C8 rho1.5为206/280（73.57%）。本轮闭环点估计不支持joint整体优于clip_only，也不支持将较低RMSE直接当作更高成功率的依据。

后续channel scaling应同时保留clip_only和joint对照，将camera_viewpoints单列诊断。Original固定输入和Plus扰动/闭环状态分布存在差异，可能是误差与控制收益不一致的原因之一，但本次结果尚未隔离视觉分布覆盖、状态偏移或量化误差方向各自的因果作用；不把这5个净差直接定性为确定的过拟合或统计显著退步。

原始闭环summary保存为closed_loop_source_summary.json，紧凑计数为closed_loop_compact.json，完成门槛为closed_loop_completion_evidence.json。下面的RMSE结果仍成立，其结论范围限定于固定输入数值误差。

直接消费来源线程已完成的 final80 checkpoint，不重复训练。来源训练job为25858027.pbs101；本任务GPU评估25858590.pbs101，CPU汇总25859109.pbs101。

## 原 held-out80 queries

同一40 contexts ×2 seeds，原输入、seeds和共享teacher；按case内平均seed、suite内平均case、四suite等权。误差越低越好。

| 配置 | First10 motor RMSE | Future latent NMSE |
|---|---:|---:|
| 原baseline | 0.044296774 | 0.060877371 |
| C80 clip_only80 | 0.043848437 | 0.054848124 |
| C80 joint80 | **0.041846837** | **0.048973336** |

joint相对baseline：action −5.5307%，latent −19.5541%；相对clip_only：action −4.5648%，latent −10.7110%。两项主指标均优于两组对照。

相对clip_only，action在27/40 cases改善、13/40变差；latent在26/40改善、14/40变差。相对baseline分别为25/40和31/40改善。收益并非每个case一致；这些胜负计数先平均同一case的两个seeds，不将80个queries当作80个独立contexts。

## C80 calibration80 queries

| 配置 | First10 motor RMSE | Future latent NMSE |
|---|---:|---:|
| 原baseline | 0.044146961 | 0.054735372 |
| C80 clip_only80 | 0.041858108 | 0.047223291 |
| C80 joint80 | **0.040851498** | **0.043192356** |

joint相对baseline：action −7.4648%，latent −21.0888%；相对clip_only：action −2.4048%，latent −8.5359%。收益同时出现在calibration与原held-out，不能将本轮joint概括为“只改善训练拟合、测试仍比baseline差”。

## 对之前判断的修正

这轮支持C80 joint在当前固定cohort的两项主指标上有实际数值收益，不能用先前较小calibration实验的失败否定joint方案。相对等C80、等80 updates的clip_only，本轮开放codebooks和row gains的joint配置获得额外收益。

但C8到C80的历史实验同时改变了数据覆盖与更新预算，当前结果不能把收益全部归因于增加数据，也不能证明过拟合已完全消失。这里仍是单次训练、已被多次用于诊断的固定cohort；不作统计显著性或独立泛化结论。

Late22、full32和gripper仅保留诊断，不改变本次两项主指标判断。完整七指标及四suite结果保存在comparison.json和RESULTS.zh.md。

后续探索channel scaling时，应保留这个C80 joint80作为重要对照，重新验证能否在其基础上增加收益；本轮没有启动任何scaling或额外训练。

## 执行与完成证据

来源：原C8 rho1.5 update8 weights、ratio=1初始化，C80固定顺序80次额外joint更新，原loss/teacher/seeds不变，固定最终checkpoint。消费时加载614组FP32 books/log row gains及614个ratios，保持clipping forward生效，冻结全部参数梯度。

本任务GPU用时00:47:26，小于2小时；CPU汇总00:00:16。新calibration端点80、原held-out端点80，复用baseline/clip_only320端点，共480。评估实际模型query执行161次：160个端点，其中joint reference query复用为一个calibration端点，另有一个原baseline parity query。训练更新0，新teacher0。

原baseline和joint checkpoint的reference action max_abs都为0。GPU utilization/VRAM每15秒写入job log；本地保留有界读取的首尾31个采样作为记录，不把它们当作全程峰值统计。gpt-6-luna / xhigh每10分钟监控，最终检查通过后已停止。completion_evidence.json保存source及两个消费作业的完成判据。

本任务直接执行的证据范围为fixed-input、decoded-BF16 VQ + fake A4；上方另行记录来源线程已完成的闭环结果。未新增本任务的closed-loop或native kernel评测，不由RMSE推断成功率、速度或部署显存收益。
