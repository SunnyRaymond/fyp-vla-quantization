# 单tile损伤预测与多tile交互校验

检验上一轮有符号评分失效的原因：它能否预测单个tile residual移除后的真实损伤；多个tile同时移除的实际影响能否由单块结果相加预测。Root亲自实现与分析，subagent仅每10min只读监控。沿用ponytail、experimental-design和statistical-analysis的配对、分块与证据边界要求。

**已完整完成。当前有符号评分对本次大小的单tile有限移除就预测不稳，多tile也存在明显的输出非加性。因此不能把上一轮失败只归因于同时移除了太多tiles。** 这不否定训练中的latent protection，也没有证明WAM必须使用VQ。实际结果与解释见本文后半部分；[机器汇总](retrieved/results/analysis/primary_summary.json)与原始CSV均已保存。

## 冻结设计

- 原action-only epoch2/16updates、614目标Linear、4D additive VQ、两本256-entry codebook、row scales、Smooth/H128、fake A4和sampler不变；不训练。
- Calibration仅原8cases `[0,2,4,6,10,12,14,16]`、seed_index0。不增加/删减cases，不看selection/test来选tile。
- 预定四个Linears：video blocks0 FFN0、video blocks29 FFN2、action blocks0 FFN0、action blocks29 FFN2。覆盖两条路径的早/晚层；不声称覆盖所有模块类型。
- 每个Linear按已有8case均值评分挑7个互异完整tiles，每tile4096个4D向量，共16384权重坐标；排除尾部不足tile和row padding。
- Video模块：latent高正/高负、reader高正/高负、latent/reader归一化近零、Action Fisher高分、随机。Action模块：action高正/高负/近零、Action Fisher高分、下一Joint Fisher高分、两份随机。Action模块的future/reader评分为结构零时保留为负对照，不伪造正负组。端点标签的实际分数均保留；如果端点没有对应符号，明确以实测分数解释。
- 按上述顺序去重选择，某个极值若已被前面的条件选中，则取尚未入选的下一名；已经选中的极值仍完整保留。例如高Fisher tile可能先以负signed-score标签入选，不把它丢掉。
- 28个single；每模块4个pairs与1个quartet，共20个within-module multis；再加2个跨模块pairs、1个跨模块quartet、1个跨模块octet。总52个真实干预，加初始和末尾各一个不改权重baseline，共54conditions/case、432完整配对记录（224single、192multi、16baseline）。干预顺序以固定seed打乱，初末baseline不打乱。

## 实际运行与公平条件

从完整双book source权重出发，只将选中tile从full BF16 decode替换为base-only BF16 decode。每个条件前恢复上一次改动，再施加当前集合；base、codebooks、scales、indices均不重拟合。小块诊断比较的移除数量由集合大小决定，不属于上一轮全模型相同预算竞赛。

每个case先完成全部native生成，保存action和future；再固定完整BF16 reader，根据每个future重新生成K/V并读出action。所有输入、描述、sampler seed/action noise、观测frame0保持一致。初始full native action/future和固定reader输出必须与原缓存**逐数值精确一致**，否则停止，不能用大的parity tolerance掩盖微小tile效果。末尾不改权重baseline重复用于噪声检查；损伤在重复差异的2倍与1e-12之上才判正负。

梯度预测与真实损伤使用同一目标：native first10 motor **MSE**、future **NMSE**（排除frame0）、固定BF16 reader first10 motor **MSE**。主要比较有符号预测的符号正确率、相关性、预测误差，以及先按8cases平均后的tile排序；不拿MSE梯度直接预测RMSE差。Action/Joint Fisher只是非负排序proxy，分别与native MSE变化、native MSE+冻结lambda×future NMSE变化的signed/absolute排序比较，不当作数值校准的损失预测。

输出还保留full32/late22/gripper RMSE。Action-module干预必须单独报告future/reader是否保持不变，以检查路径隔离；这些结构零不应被用来夸大评分预测能力。报告每个模块和case的结果，重复测量不算独立总体样本，不做事后挑选显著性检验。

## 交互的两个层面

Single额外做不增加forward的精确分解。令e为baseline输出与teacher的误差，dy为实际单块输出变化；MSE/NMSE损伤等于`2<e,dy>/normalizer + ||dy||²/normalizer`。前一项减去原梯度预测，就是有限输出响应与梯度/STE预测之间的残差；后一项是非负平方loss项。实际损伤减预测，精确等于这两部分之和。此分解能区分loss平方项与梯度预测偏差，但不能再把后者单独归因于STE或网络的有限步长非线性。真实loss与分解使用float64计算，保持模型输出本身不变。

对集合S，真实损伤为DeltaL(S)，加性预测为各single DeltaL之和；交互I为两者差。归一化使用single损伤绝对值之和，避免有符号抵消产生错误分母。

同时保存输出：y_add=y_baseline+sum(y_single-y_baseline)，检查实际y_multi是否等于y_add。把loss交互精确拆成：

1. `L(y_add)-L(y_baseline)-sum(single DeltaL)`：即使输出完全相加，平方loss也会出现的交叉项。
2. `L(y_multi)-L(y_add)`：实际输出不相加带来的额外loss变化。

报告native action、future和fixed-reader action的输出非加性norm/ratio、损伤交互、上述分解和加性预测符号翻转。按集合大小与video/action/mixed路径分开报告，避免action-only的future结构零稀释结果。

## 执行与完成条件

先CPU prepare冻结plan并跑小型生产helper测试；随后GPU pack仅做一次与源算法完全相同的CUDA/BF16 decode，保存10个可恢复chunks。这样每个评估job只需顺序读取full权重、切换选中的小切片，减少重复614个bank解码。Pack不做model forward，使用GPU是为保留源CUDA log-scale exp和BF16 decoder的精确结果。

8个GPU评估分片各负责1个case；每job最多2h，15s记录GPU利用率/显存。逐query原子保存并可恢复，只有verified F/Exit0/RESUME_READY才续交。CPU prepare/analyze 20min。所有模型、tensor、数值计算和重IO在真实PBS compute allocation内；login仅可信SSH、小控制文件、PBS查询/提交，不下载/传输模型、不安装、不做hash或推理。保留host-key verification，不输出credentials，不动banked reset。Subagent不提交、不改实验、不分析数据，只约10min监控当前handle；Q/R无实质变化保持安静。

只有4个模块×7tiles×8cases的全部single、多块组合、baseline重复和CPU配对分析均完成才结束。统计范围是这些预定模块和选中calibration端点，不能据此宣称独立泛化、闭环性能、native低位kernel或训练中latent protection loss的有效/无效。

设计流程参考 Scientific Agent Skills (2026), [arXiv:2609.00065](https://arxiv.org/abs/2609.00065)。当前job handles和阶段见RUN_STATE.json；运行前/排队/partial均不作为已有结果。

## 完整结果与解释（2026-10-17）

这轮要区分两个问题：第一，梯度评分是不是连单块的真实影响都预测不准；第二，即使知道单块的真实影响，多块同时改变时是否仍不能相加。**实测两个问题都存在。** 这里的“损伤”是相对完整双book source的loss变化，可以为负；负数表示更接近BF16 teacher，并不要求减少容量必然变差。

全部8cases、28个single与24个multi组合均完成：224条single、192条multi、16条baseline，合计432条配对记录。每条都包含native与固定BF16 reader查询，共864次模型查询；没有新训练或新backward。Prepare、pack和CPU分析的model forward均为0。Single CSV有672行、multi CSV有576行，是每条干预分别展开三个目标后的行数，不能当作672或576个独立样本。

### 工程对照是否通过

- 八个case的初始native action、future和固定reader action，都与原缓存精确一致，最大差异为0。
- 八个case的末尾重复中，native action、future以及reader action与初始输出的最大差异均为0；三个loss也精确一致。本轮判方向的noise floor因此都是预定的1e-12下限。
- 112条action-module single负对照中，future与固定reader action的最大变化都为0。Multi中纯action分支的future/reader结果也为结构零，单独保留，不用于夸大预测准确率。
- 所有GPU作业F/Exit0/PIPELINE_COMPLETE，逐15秒GPU利用率/显存写入各job.log；八个评估作业均在2h以内，无需续跑。CPU分析25851398.pbs101同样F/Exit0，完整覆盖门通过，用时5m48s。

| Case | GPU job | 实际walltime |
|---:|---|---:|
| 0 | 25848467.pbs101 | 01:33:58 |
| 2 | 25848994.pbs101 | 01:34:38 |
| 4 | 25848995.pbs101 | 01:10:03 |
| 6 | 25848998.pbs101 | 01:26:28 |
| 10 | 25848999.pbs101 | 01:07:10 |
| 12 | 25849000.pbs101 | 01:19:33 |
| 14 | 25849003.pbs101 | 01:16:55 |
| 16 | 25849005.pbs101 | 01:24:16 |

Subagent只做约10min只读监控；代码、提交、终态验收和本节分析由root完成。全部作业验收后监控已停止，没有待续跑或待监控handle。终态及baseline receipts见retrieved/evaluate-*-baselines.json，当前完成状态见RUN_STATE.json。

### 单块：当前评分不是可靠的有限损伤预测器

下面每行是同一个模块内7个tiles×8cases的描述性结果；Spearman衡量排序是否一致，方向正确数衡量预测的改善/恶化方向是否与真实变化一致。56条是重复测量，不能当作56个独立总体样本，也不据此做显著性判断。

| 目标 | 模块 | Spearman | 方向正确 / 非零记录 |
|---|---|---:|---:|
| Native motor MSE | Video block0 FFN0 | -0.027 | 24/56 |
| Native motor MSE | Video block29 FFN2 | -0.331 | 21/56 |
| Native motor MSE | Action block0 FFN0 | 0.076 | 30/56 |
| Native motor MSE | Action block29 FFN2 | -0.071 | 26/56 |
| Future NMSE | Video block0 FFN0 | 0.093 | 30/56 |
| Future NMSE | Video block29 FFN2 | -0.036 | 25/56 |
| Fixed-reader motor MSE | Video block0 FFN0 | -0.038 | 26/56 |
| Fixed-reader motor MSE | Video block29 FFN2 | -0.138 | 30/56 |

只移除一块时，native action方向正确率为37.5%至53.6%，fixed-reader为46.4%至53.6%；排序也没有稳定的一致性。逐case结果同样有明显正负波动，例如video block29的native排序相关在case0为+0.500、case12为-0.786，不能把跨case混合后的相关当作每个case内都成立的关系。所有逐case结果见[per_case_predictive.json](retrieved/results/analysis/per_case_predictive.json)。

我们还按原评分的选择方式，先对8cases平均，再比较每个模块内的7个tiles：

| 目标 | 模块 | 8case均值的Spearman | 均值方向正确 |
|---|---|---:|---:|
| Native motor MSE | Video block0 FFN0 | 0.321 | 1/7 |
| Native motor MSE | Video block29 FFN2 | -0.571 | 3/7 |
| Native motor MSE | Action block0 FFN0 | -0.286 | 1/7 |
| Native motor MSE | Action block29 FFN2 | -0.107 | 4/7 |
| Future NMSE | Video block0 FFN0 | 0.464 | 4/7 |
| Future NMSE | Video block29 FFN2 | 0.464 | 2/7 |
| Fixed-reader motor MSE | Video block0 FFN0 | 0.321 | 3/7 |
| Fixed-reader motor MSE | Video block29 FFN2 | 0.250 | 4/7 |

因此不能说所有目标都完全没有排序信号：future的7点均值排序有中等正相关。但方向、数值尺度和跨case稳定性仍不可靠，不能据此恢复上一轮的大规模allocation规则。端点是按预测分数预先选中的，只有7点；这些相关不能泛化到全部tiles。

一个预先选择的例子是video block0的reader高正分tile t02。8case平均预测为**+9.40e-7**，实际fixed-reader MSE变化却为**-2.98e-4**：评分认为移除会损伤，实际均值反而改善。相邻的近零评分tile t04也有明显真实变化，不能把“梯度分数近零”解释成“有限移除安全”。这不是因为把RMSE变化与MSE梯度比较，本轮两边严格使用同一MSE/NMSE目标。

### 为什么不能简单说只是漏掉了loss曲率

对平方loss，实际损伤可以精确写为：

`真实DeltaL = 2<baseline误差,实际输出变化>/normalizer + ||实际输出变化||²/normalizer`

前一项使用实际有限输出变化，后一项为非负平方项。它们来自已保存输出，没有额外forward。它们的和与真实loss变化在预定数值检查内一致。

同一个t02的fixed-reader例子中，8case均值为：

| 项 | 数值 |
|---|---:|
| 原梯度预测 | +9.40184e-7 |
| 实际输出变化对应的线性loss项 | -6.54273e-4 |
| 非负平方loss项 | +3.56754e-4 |
| 两项相加的真实loss变化 | -2.97519e-4 |

这说明**权重变化到输出变化的一阶/STE预测已经与有限响应不符，平方loss项本身也不可忽略**。仅给原来的正预测加一个非负平方项，不可能解释这里实际为负的结果。其他模块的平均平方项和平均绝对有限响应残差也都显著非零，详见机器汇总；不能把全部误差只归因于其中一项。

此分解仍不能将有限响应残差进一步拆成STE误差、activation rounding阈值变化或网络有限步长非线性。一个tile含16384个权重坐标，移除整个residual是有限变化；本轮没有验证无穷小扰动下的真实导数，也没有加入delta插值或关闭A4的额外对照。

### 多块：即使使用单块实测结果，也不能简单相加

这里已经不用旧梯度分数，而是把各single的**实测**DeltaL相加来预测联合移除。交互相对量定义为：

`|联合DeltaL - sum(single DeltaL)| / sum(|single DeltaL|)`

输出非加性相对量定义为：

`||联合输出变化 - sum(single输出变化)|| / sum(||single输出变化||)`

这些比例不是解释方差或任务成功率。Native action没有纯action干预导致的future结构零混入，按组大小的结果如下：

| 同时移除 | case×组合记录 | 交互相对量中位数 | 输出非加性相对量中位数 | 相对单块实测和的方向翻转 |
|---:|---:|---:|---:|---:|
| 2 tiles | 144 | 0.629 | 0.702 | 43/144 |
| 4 tiles | 40 | 0.609 | 0.669 | 14/40 |
| 8 tiles | 8 | 0.436 | 0.659 | 4/8 |

即使两个tile，联合loss变化与单块实测和的偏离已经不小；43/144条记录甚至把改善/恶化方向翻转。8-tile仅一个预定组合×8cases，不是8种独立组合；不同size用了不同组合，因此不能从这张表推断交互随size单调变化。

Future和reader必须剔开纯action干预的结构零解释。只看纯video分支组合（88条重复测量），future的交互相对量中位数为0.637，输出非加性为0.671，方向翻转18/88；fixed-reader分别为0.651、0.647，方向翻转23/88。纯action分支的future/reader全为零，符合路径隔离预期。Mixed分支的native/future/reader交互相对量中位数分别为0.496/0.230/0.339；不同分支组合构成不同，不作为可直接比较的随机处理组。

我们还将loss交互精确拆成“输出相加时平方loss已有的交叉项”和“实际输出不相加引入的loss变化”。两者均保留，且分解通过数值检查；真正的输出非加性也明显非零。因此不能将本轮现象全部解释为平方loss自带交叉项。

### Fisher对照也不能被过度解释

上一轮Action/Joint Fisher的整体allocation较好，不代表它们已被验证为精确的单块损伤预测器。本轮同模块56条重复记录中，Action Fisher对native绝对损伤的Spearman分别为0.140、0.138、0.197、0.086；Joint Fisher对其对应joint绝对损伤分别为0.137、0.124、0.197、0.086。这些描述性相关也较弱。Fisher只比较排序，不把非负分数当作有符号或数值校准的DeltaL预测。

### 对当前研究方向的判断

这轮缩小了上一轮负结果的解释范围：不能只说大量tiles一起改变才造成失败，因为当前评分在单个完整tile的有限移除上就不可靠；多块同时改变又额外引入明显非加性。它没有量化这两种问题分别解释了上一轮全模型失败的百分之多少，也没有将2/4/8块的结果外推到大规模容量移除。

我不建议继续直接用当前signed damage分数做大范围allocation或lambda搜索。若继续修评分，应先做少量delta插值，并配合明确的A4/量化梯度对照来区分有限变化与梯度近似；插值是连续权重机制诊断，不是可部署VQ编码。若优先推进“action-only留下哪类有害world-representation drift”这个contribution，更直接的下一步仍是固定reader，在已有future上做预定时间/空间分组的双向patch，并加入同修正能量的对照，定位哪些修正足够带来收益、移除后收益是否消失。上述是后续建议，本轮没有额外运行这些实验。

训练中的latent protection是否有益，与冻结权重上如何移除residual，是两个不同的问题。本轮不否定此前future替换的正结果，也没有识别语义drift或证明VQ优于scalar quantization。证据限于原8calibration cases、四个预定FFN模块、选中的端点及有限移除；执行是decoded BF16/fake A4，没有新episodes、独立泛化或native低位kernel性能证据。

完整原始产物：

- [全部汇总、精确分解、分支与size结果](retrieved/results/analysis/primary_summary.json)
- [逐case预测表现](retrieved/results/analysis/per_case_predictive.json)
- [224个single×三个目标](retrieved/results/analysis/single_tile.csv)
- [28个tile的8case均值×三个目标](retrieved/results/analysis/mean_tile.csv)
- [192个multi×三个目标及精确交互分解](retrieved/results/analysis/interactions.csv)
- [Action/Joint Fisher及对应真实目标](retrieved/results/analysis/fisher.csv)
- [CPU原始简表](retrieved/results/analysis/REPORT.md)
