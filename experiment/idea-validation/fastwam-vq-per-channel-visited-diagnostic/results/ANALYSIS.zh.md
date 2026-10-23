# Per-channel visited-state诊断：motor收益真实，先补gripper监督

本轮结果支持保留per-channel scaling。它在既有visited states上改善了motor端点误差和两个测试方向的motor响应误差，但没有保护好gripper的开合事件；还留下dx的chunk偏置问题。下一轮最有依据的单因素实验是增加独立的first10 gripper连续监督，保持原motor与latent系数、数据和训练预算不变。暂不同时加入response/bias loss，也不能把此前闭环净少4例全部归因于gripper。

**执行与比较口径。** GPU PBS25868350.pbs101和CPU PBS25868746.pbs101均为F/Exit0、artifact exit0、PHASE_COMPLETE且完整覆盖检查通过。GPU实际walltime为00:37:34，CPU为00:00:39。gpt-6-luna/xhigh每10分钟监控，最终终态已记录并结束监控；GPU utilization/VRAM每15秒写入GPU job log。

没有训练、没有新rollout，也没有改变或重跑原80-query held-out。沿用原40个C8-student短程visited states、原seeds及五种输入：base、两相机raw brightness±1 uint8、raw EEF xyz沿单位对角方向±5mm。保留全部原600个BF16/clip-only/joint端点，新增per-channel 200端点，共800；得到240个case×student×modality响应项。共205次完整WAM queries，包含200端点和5次检查；其中1次fresh BF16仅用于replay parity，没有新增teacher targets。

200份输入元数据均与旧BF16记录完全一致；BF16 visited replay、per-channel源端点、uncached/cached完整action/future、实际infer、同输入repeat检查均通过，所有记录的parity最大绝对差为0。源checkpoint的七项endpoint指标精确复现。所有模型、VQ books、row gains、indices、ratios和D均冻结。旧完整image tensors没有持久化，因此不声称对200份旧image tensor重新逐像素比对。

下述RMSE先逐case计算，再平均40个cases；gripper commands使用真正的denormalize→invert→sign，-1为open，+1为close。每种输入有400个执行位置，切换位置分母为40×9=360。五种输入共享40个cases，不是200个独立样本。BF16是共同参考，而不是闭环成功oracle。

| Base输入，first10 | clip-only | joint | per-channel |
|---|---:|---:|---:|
| Normalized motor RMSE | 0.045833804 | 0.044575345 | 0.034658006 |
| 实际command motor RMSE | 0.033880164 | 0.033009563 | 0.027482118 |
| Gripper连续normalized RMSE | 0.100219555 | 0.058089548 | 0.115395849 |
| 实际开合错误 /400 | 6 | 3 | 6 |
| 出现开合错误的cases /40 | 5 | 2 | 4 |
| 切换位置不一致 /360 | 7 | 3 | 4 |
| 前10步内漏open事件 | 1 | 1 | 2 |
| 额外open事件 | 2 | 0 | 2 |
| 有序切换方向序列不一致的cases | 3 | 1 | 4 |

Per-channel相对clip-only的normalized motor在37/40个cases更低、3个更高；实际command motor为33/40更低、7个更高。相对joint分别为33/40、31/40更低。不是只有平均值被少量容易输入拉低。

**Gripper的具体问题是事件，没有被平均motor改善覆盖。** 全局开合confusion在clip-only/per-channel中恰好相同：teacher要求open而student仍close为4步，反向错误为2步。但错误落在不同状态，事件含义也不同。以下step索引从0开始。

| Case | BF16要求 | clip-only | per-channel | 对应motor RMSE，clip→per-channel |
|---|---|---|---|---:|
| 2013，Spatial | 从step0起open | step0错误close | 同样step0错误close | 0.047204→0.035937 |
| 2031，Object | step9 open | 正确 | 前10步内一直close | 0.044041→0.029839 |
| 2043，Goal | 前10步保持close | step9错误open | step8起错误open | 0.035812→0.028483 |
| 2045，Goal | step8 open | step9 open，晚1步 | 前10步内一直close | 0.074669→0.070434 |
| 2049，Goal | step8 open | 前10步内一直close | 正确 | 0.057426→0.079270 |

2031给出本轮新增的直接反例：motor明显更接近teacher，却漏掉本chunk内的释放。2045也确实存在于本轮per-channel中，不再只是从旧joint结果移植的猜测。反过来，per-channel修复了2037的提前open和2049的漏open；2049修复gripper的同时motor变差，说明两者必须分别检查。

六个base错误位置的teacher阈值前command绝对值均在0.9921875–1之间，teacher并不接近开合阈值。没有student在前10步之后的执行记录，故“漏open”严格指前10步窗口内；不能推出以后永远不释放。只对有序事件方向完全一致的序列计算延迟。Per-channel仅两个可匹配事件，延迟均0，但它漏掉了另外两个open，不能据此宣称切换时机完美。

Gripper也没有在全部扰动输入上整体变差。每格仍以同一40×10=400位置为分母：

| 实际开合错误数 | clip-only | joint | per-channel |
|---|---:|---:|---:|
| base | 6 | 3 | 6 |
| image+ | 6 | 4 | 4 |
| image− | 6 | 1 | 4 |
| proprio+ | 7 | 7 | 6 |
| proprio− | 10 | 4 | 4 |

所以，结论是“未监督的gripper仍有控制相关事件缺口”，不是“per-channel让所有gripper指标全面退化”。它比joint的base表现差；相对clip-only，base错误总数相同、漏open更多，但扰动输入的开合错误更少。

**Motor偏置：主要保留dx问题。** 下面先对每个case的10步误差取均值，再取绝对值，最后平均cases，防止跨case的正负误差抵消。数值为normalized action坐标，不将其称为已测得的闭环位移漂移。

| Mean absolute chunk bias | clip-only | joint | per-channel |
|---|---:|---:|---:|
| dx | 0.017737122 | 0.018799133 | 0.019936714 |
| dy | 0.018512554 | 0.017706280 | 0.018505383 |
| dz | 0.024125366 | 0.023039093 | 0.015094604 |
| rotation_x | 0.016424751 | 0.014017792 | 0.011428070 |
| rotation_y | 0.018627605 | 0.014573193 | 0.014921589 |
| rotation_z | 0.015581436 | 0.013695450 | 0.014670906 |

相对clip-only，六维逐维RMSE均更低，但dx chunk偏置更大，dy基本持平，其余四维改善。实际command中的dx偏置也由0.016628551升至0.018690668。Goal的dx偏置由0.020852356升至0.027950745；其dy也略增。这与原Plus起点分析的部分方向偏置风险相容，但不能将原Plus上四维变差概括成这批visited states的结论。

| Base，四suite各10cases | clip motor | joint motor | per-channel motor | 实际开合错误，clip/joint/per |
|---|---:|---:|---:|---|
| Spatial | 0.052243098 | 0.054049459 | 0.039536519 | 1 / 0 / 1 |
| Object | 0.039717410 | 0.038686204 | 0.027319868 | 1 / 1 / 1 |
| Goal | 0.050936744 | 0.045634068 | 0.044449896 | 4 / 2 / 4 |
| LIBERO10 | 0.040437964 | 0.039931649 | 0.027325740 | 0 / 0 / 0 |

Goal的收益较弱，而且normalized motor比joint低，实际command motor却为0.035508342，高于joint的0.032811079。这提醒我们normalizer后不同维度的权重会变化，normalized motor均值不能代替所有执行坐标的保护。LIBERO10这十个base状态没有开合错误，而原闭环在该suite净少5例，也直接限制了“全是gripper造成”的解释。

**局部motor响应整体改善，不能作为本轮主要退化解释。** 使用原中心有限差分，计算first10×6的60D响应相对BF16的误差。以下RMS先逐case计算再平均；亮度与proprio的扰动单位不同，只在同一行比较各方法。

| 方向 | BF16响应RMS | clip误差RMS | joint误差RMS | per-channel误差RMS | per比clip误差更小cases |
|---|---:|---:|---:|---:|---:|
| image，raw brightness±1 | 0.537469 | 6.117905 | 6.070676 | 4.870094 | 35/40 |
| proprio，EEF xyz±5mm | 0.194193 | 4.187125 | 4.358801 | 3.694820 | 29/40 |

Per-channel在五种输入上的平均motor端点误差均比clip低。局部响应误差在image/proprio分别约低20%/12%；不能说scaling导致了整体更差的输入响应。Goal的proprio是局部例外：误差4.182646→4.256152。所有分suite结果保留在comparison.json。

响应仍远未接近teacher：per-channel的平均方向cosine为image 0.04446（40个defined cases）、proprio 0.06079（39个defined cases）；增益均值为13.1766和20.5909。Teacher响应较弱，不能用巨大相对gain单独定义闭环不稳定，更不能把两条有限差分称为完整Jacobian。Gripper相对各自base因扰动而翻转的步数：image+/−为clip 4/4、joint 1/2、per 4/4；proprio+/−为clip 9/8、joint 6/1、per 4/2；BF16全部为0/400。

**这对过拟合判断意味着什么。** 原held-out、Plus起点及本轮visited motor均有收益，且局部响应总体也改善。现有结果不支持将motor收益失去闭环转化简单归结为“训练集motor过拟合”。更明确的问题是优化/度量未充分保护控制关键量：第7维没有直接监督，chunk内的持续偏置和接触/释放阶段也没有单独保护。有限calibration覆盖与1.8M个D参数仍可能存在泛化风险，但不是已经被这些结果证明的唯一原因。

这批状态由旧C8 student短程采集，不是per-channel自身中后期或失败轨迹；亮度扰动不是真实camera viewpoint变化，proprio扰动也没有物理重渲染。因此，本轮只能确认相同输入上的错误形态，不能把它直接映射为此前280 episodes的失败因果。此前203/280对clip 207/280仅为该cohort的净结果，没有重复rollout或稳定差异结论。

**下一轮只做一个loss改动：补first10 gripper。** 建议对照原per-channel配方，使用同一原C80、teacher、update order、初始化和80 updates，训练参数仍仅D与ratios。重新从原baseline的D=1、r=1起步，采用同一optimizer规则，固定final80比较；不要只把新方法从现有final80续训、却拿旧80-update模型当公平对照。

L_new = L_motor6 + lambda_g * mean((a_q[:10,6] - a_BF16[:10,6])^2) + rho * c * L_future_latent。

保持原rho=1.5及冻结normalization_c不变，不因为新增gripper重新计算c。原motor是60个元素的mean，gripper是10个元素的mean，因此可预先固定lambda_g=1/6作为每个action维度等权的起点：这样原motor的绝对权重完全不变。它是有定义的起始配方，不是已验证的最优lambda。直接改为七维mean会将原motor隐式乘6/7，不利于隔离新增监督的作用。

先使用已经缓存的teacher第7维连续值，继续以真正的开合命令和有序事件评估。np.sign不参与反传；先不增加margin、事件重加权或额外超参数。若连续监督仍修不好开合，再针对实际阈值前的可微command加入teacher sign margin，并从calibration确定阈值与权重，不能从这次诊断病例或原held-out选择lambda。

评价时保留原80 held-out、同一visited诊断和原paired闭环协议，分别报告motor、gripper事件和latent；不新建development set。先看能否在保留motor收益的同时减少漏open/错误open，最终再看闭环是否获益，不能因gripper RMSE下降就宣称成功率修复。

Bias/response暂不与gripper同时改。若gripper修复后仍有特定方向的chunk偏置，可下一步单独测试mean-over-time误差惩罚；普通MSE本来包含bias平方，增加此项是在提高持续偏差的相对权重，不是补一个完全缺失的监督。Response loss优先级更低：只有重放失败轨迹显示teacher-referenced响应问题仍阻碍控制时，再增加匹配输入变化的差分loss，不单纯压小student响应，不除以接近零的teacher响应。

本轮没有启动以上任何训练。若后续需要增加calibration，优先预先定义抓取、释放、接触后恢复等阶段的覆盖，再独立控制contexts与updates；不能把更多数据、更多步数、新loss和分支自由度一次全改。下一次诊断rollout应保存每次replan前可重放state/obs、命令及阶段事件，以便在同一student visited state查询BF16，进一步定位原闭环失效。

原始汇总见comparison.json，逐case base事件与偏置见diagnostic_details.json；完整800端点、600个student endpoint metrics及240个sensitivity rows保留在远程results目录。该实验为decoded-BF16 VQ + fake A4 fixed-input诊断，不提供native-kernel、真实低比特延迟或新增闭环收益证据。
