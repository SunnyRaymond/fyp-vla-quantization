# C8 bilevel 失败分析与下一步方案

2026-10-20。状态：仅分析与设计，未提交新实验。新模型forward=0，新训练updates=0，新PBS submissions=0。本轮只读取现有72份endpoint小JSON（confirmation48、outer24）、现有search操作日志与本地汇总；不加载模型、tensor或checkpoint。原实验协议与结果保持不变。

## 结论

confirmation 的 first10 motor RMSE 退化高度集中在case5007，但其余cases也没有形成一致改善。单步虚拟更新与最终常数rho重训的目标在数学定义上不一致；现有数据还发现probe覆盖只占C8的一半、outer batch与probe context固定关联、最终rho没有incumbent接受检查。这些是可确认的结构问题，尚不能量化各问题对confirmation退化的因果贡献。

## 退化集中度

所有数值均为rho≈2减去development预先冻结的rho=1.5对照；每case先平均两个inference seeds。均为相对BF16 teacher的误差。

| Case | Suite | Phase | 固定rho1.5 first10 RMSE | 固定rho2 first10 RMSE | 相对变化 |
|---|---|---|---:|---:|---:|
| 5000 | libero_spatial | early | 0.042852 | 0.040328 | -5.89% |
| 5001 | libero_spatial | middle | 0.036165 | 0.041857 | +15.74% |
| 5002 | libero_object | middle | 0.038525 | 0.044049 | +14.34% |
| 5003 | libero_object | late | 0.034563 | 0.036430 | +5.40% |
| 5004 | libero_goal | early | 0.048341 | 0.047137 | -2.49% |
| 5005 | libero_goal | middle | 0.046629 | 0.038618 | -17.18% |
| 5006 | libero_10 | middle | 0.034060 | 0.035474 | +4.15% |
| 5007 | libero_10 | late | 0.063981 | 0.155920 | +143.70% |

case5007来自LIBERO-10的“把黄白色杯子放进微波炉并关门”任务，late phase，demo27/frame298。其first10 motor RMSE从0.0639805升到0.1559205（+143.70%），case差值0.091940；八个case的总差值0.094697。它因此贡献97.09%的净增量，或86.38%的所有正向退化之和；前者含其他case改善带来的抵消，二者不能混为一谈。

两个inference seeds都退化：seed0为0.067970→0.113054（+66.33%），seed1为0.059991→0.198787（+231.36%）。大幅退化不只存在于一个seed；但两个seeds仍不足以估计广泛的seed稳定性。

该case的future latent NMSE从0.0296527升到0.1523089（+413.64%），贡献confirmation future NMSE净增量的99.31%；full32/late22 motor的净退化也几乎全由它贡献。这里同时出现motor与latent退化，不能解释成已证明的二者之间简单trade-off；gripper并非该case的主要退化项。

仍有5/8 cases的first10 motor RMSE变差，3/8改善；逐case配对差值中位数为+0.001640。为了诊断集中度，事后排除5007并保留四suite等权时，first10 motor仍为0.0393995→0.0399209（+1.32%）。若错误地改成剩余7 cases等权会得到+0.98%；这里不以这种不同权重替代原口径。case5007的排除只用于敏感性分析，原8-case cohort和No-Go结论不变。

原outer development中已有相同task/late phase、不同demo的case4007。该case在常数rho2下motor已退化39.90%，future NMSE退化107.22%。最终固定rho2在完整outer上也比rho1.5差10.52%。因此失败信号在最终development endpoint中已经可见，不能仅归于confirmation碰到了一个development完全没覆盖的任务。

## 单步代理与最终训练的目标错配

令s包含VQ参数、Adam一阶与二阶矩、RNG及训练位置，T表示一次inner Adam更新。当前搜索每轮比较：

$$\widetilde J_t(u)=J_{D_t}\!\left(T_{\exp(u),C_{2t}}(s_{2t})\right),$$

其中s_2t来自此前不断变化的rho历史；比较时把它冻结，只改变下一次更新的rho，并只在四个交替outer contexts上打分。实际最终模型需要优化：

$$F_{16}(u)=J_D\!\left(T_{\exp(u),C_{15}}\circ\cdots\circ T_{\exp(u),C_0}(s_0)\right).$$

后者从共同初始状态出发，16步均使用同一rho，Adam历史也随这个rho完整变化，并在完整outer数据上比较。当前方法没有估计rho对前15步训练与Adam历史的影响，而最后一轮搜索只回答“现有历史下，下一步更适合哪个rho”。取最后一轮rho再从头训练，会丢弃搜索产生的参数/Adam路径；简单把warm-start unroll从1改到4步仍不能等同于F16。

这是目标定义的结构差别。bilevel优化文献会显式把inner优化动态纳入目标；truncated hypergradient也是有条件的近似，不能直接把相应收敛结论套用到本项目的非凸Adam、量化更新与大跨度有限差分上。[Franceschi et al. 2018](https://proceedings.mlr.press/v80/franceschi18a.html)、[Shaban et al. 2019](https://proceedings.mlr.press/v89/shaban19a.html)。

现有结果提供的支持：最终rho2在outer batch A比rho1.5改善1.49%，在batch B退化22.92%，完整outer退化10.52%。最后一轮的warm-state单步probe却在batch B偏好rho2。这里比较的起点、更新长度及另一候选rho不同，不能把这些数值直接当作“同一hypergradient符号已被反证”；它们足以说明短步信号没有保证最终常数rho endpoint会更好。

尚缺两项用于因果区分的结果：adaptive搜索最终checkpoint的完整评测，以及同一对rho候选从同一s0分别训练1/8/16步的outer排名。两项都需要未来新增forward，当前没有推算或伪造。

## 虚拟probe的覆盖与选择规则

每轮提交2个真实updates，却只在本轮第1个context做probe；8轮probe只覆盖2000、2021、2051、2066，各出现两轮（每轮正负两支）。2019、2025、2057、2070从未进入任何虚拟probe，但最终训练均使用它们。四个suite各探测1个context，C8中唯一late context2019恰好被漏掉。不能据此断言它导致5007的late退化，但它说明代理方向没有平均全部C8 contexts。

outer batch A固定配对spatial/goal的probe，batch B固定配对object/LIBERO-10的probe，无法从本次轨迹中分开outer batch与inner context的影响。实际rho方向在7次相邻转换中改变4次，并三轮到达上界3；这不足以证明收敛到局部最优。

最终rho按预设规则取round8末值，没有要求该rho从头重训后优于已有incumbent。严格遵守了本轮冻结协议，但下一轮应修改这条选择定义。数值恢复、确定性重放和teacher sentinel均已通过；search各步gradient norm最大约0.0877，小于clip=1，因此本次search并未触发gradient norm clipping。未见证据把本次失败归因于这些机制，也没有activation clip实验可供归因。

## 下一步分阶段方案

目标是先区分路径变化与有限训练长度的影响，同时维持C8、16步、共同初始banks、Adam、c、训练顺序、损失与数值设置。先保持原probe factor=1.5，不同时引入新的robust loss、activation clip或额外训练contexts。

### A 评估已有动态搜索终点

读取已保存的search最终checkpoint s16，使用完全相同的fake A4/decoded BF16部署方式，评估完整outer8×seed0及已观察的旧confirmation8×2。与固定rho2和rho1.5的缓存逐case配对比较；只作为诊断，不参与新rho选择。无需训练或新teacher，共24F、0G。

如果动态终点比固定rho2明显好，尤其4007/5007恢复，支持训练路径被丢弃是重要因素；如果动态终点也差，则单纯保留schedule终点不足以解决问题。两种结果都不是泛化结论，旧confirmation已被观察。若项目目标转为训练动态schedule，应直接验证该schedule终点；若目标仍是选择一个常数rho，则继续B。

### B 直接检验完整16步的标量外层目标

围绕冻结的incumbent rho1.5，以相同log扰动delta=log1.5选rho_minus=1.0、rho_plus=2.25；它们也与initial preflight的探针一致。两支均从完全相同的s0出发，branch内rho保持不变，使用完整C8顺序两轮共16步。

在同一条训练轨迹的update1、8、16保存checkpoint并分别评估完整outer8×seed0；每支为16G+24F，两支总32G+48F。无需为短horizon另训，也不需要新的teacher。所有checkpoint保留per-case五项指标，最终两个update16模型可直接复用。

分别计算g_H=[J_H(rho_plus)-J_H(rho_minus)]/[2log1.5]，H=1,8,16，检查方向是否随训练长度改变，并检查末端的batch A/B、suite及每case差值。H=8和16覆盖完整C8周期；H=1仅代表第一context。这里得到的是给定两个端点的割线方向，不是精确的无限小导数。

以F16最低的合格候选更新rho；固定rho1.5始终保留为incumbent，不以最后一轮rho自动替代。候选只有在完整outer first10 motor至少改善1%、其他指标均不超过5%退化时才进入下一阶段；否则保留rho1.5并停止该路线。原rho0.1、rho1.5、rho2的update16 endpoint与outer结果保留为参考，不重训。

这一步属于full-horizon标量bilevel/HPO诊断；对一维rho，两条完整训练轨迹的预算与原32G搜索相同。若方向一致且候选仍不胜incumbent，不强行推断增加unroll一定有效；若方向翻转，则有相同候选、数据和初始状态下的直接证据支持短步偏差。之后才考虑更小log步长或多轮连续hypergradient，不在这一轮混改。

### C 有development改善后才做新的泛化检查

旧confirmation及历史held-out均已被观察，后续只作回顾性诊断。新confirmation最低仍为每suite2 cases，共8×2 inference seeds；按冻结task/phase规则选独立demonstrations，排除原C80、原40 held-out及两组旧development的全部demonstrations，不按本次误差大小挑选样本。若需更可信的尾部风险结论，应在后续单独增加独立demonstrations或training seeds，不能将当前两个inference seeds当成训练复现。

冻结rho后，比较固定rho0.1、固定rho1.5与新候选，使用相同输入、teacher及seeds；所有最终模型均16步。主比较继续对冻结的rho1.5，保留原cohort均值与次指标门槛，不删除5007，不采用trimmed mean把尾部问题藏起来。worst-case、worst-suite、case wins与seed间差异作为预先规定的风险指标完整报告。 新confirmation与完整历史held-out的first10 motor均需至少改善1%，且其他四项在两份数据上均不得退化超过5%，才允许考虑后续扩展；新数据的8 cases仍只是工程pilot，不能作统计显著性声明。

复用所有已匹配的teacher与对照结果。新confirmation需16个teacher F及三组各16F，共64F；新候选在旧confirmation与完整历史40×2评测另需16+80F，合计160F、0G。历史对照及teacher均可复用，旧confirmation不再被称为盲测。

## 成本与停止条件

| 阶段 | 新训练操作G | 新inference操作F | 预计GPU·h |
|---|---:|---:|---:|
| A 已有动态终点评测 | 0 | 24 | 0.1–0.4 |
| B 两支完整16步及1/8/16步outer评测 | 32 | 48 | 1.3–1.8 |
| C 通过development门槛后独立确认与回顾性评测 | 0 | 160 | 0.5–1.0 |

以上为未运行的估计，包含模型加载的粗略余量；参考原search相近的32G+64F实际1.541 GPU·h及单个16步训练实际0.55–0.82 GPU·h。A+B约1.4–2.2 GPU·h；只有B出现合格改善才增加C。所有将来的PBS请求仍须小于2小时，保存可恢复状态并定期记录GPU利用率与显存。当前没有提交任何作业。

本轮可以确认集中度、覆盖范围和目标定义差异，不能确认具体layer/timestep、误差来自activation还是weight-VQ、或短步错配占退化的比例；现有小JSON没有逐坐标张量和对应中间模型评测。完整C8训练复现仍是后续推广所需证据。

## 数据与方法来源

- 本地原始结果：[comparison.json](../results/comparison.json)、[搜索轨迹](../results/selected_rho.json)、[GPU成本](../results/GPU_COST.json)。
- 本轮诊断：[diagnostics.json](diagnostics.json)、[72份现有endpoint记录](existing_endpoint_records.json)、[搜索操作日志](search_operations.json)。
- 实现依据：[search与fixed_training](../worker.py)，训练选择和有限差分定义来自本项目源文件，不由文献代替。
- 理论背景：Franceschi et al. 2018, [Bilevel Programming for Hyperparameter Optimization and Meta-Learning](https://proceedings.mlr.press/v80/franceschi18a.html)；Shaban et al. 2019, [Truncated Back-propagation for Bilevel Optimization](https://proceedings.mlr.press/v89/shaban19a.html)。
- 方法流程采用statistical-analysis与experimental-design skills的配对单位及控制变量规范。软件方法来源：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner. 2026. [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。该引用只说明分析流程来源，不作为本实验结论的证据。
