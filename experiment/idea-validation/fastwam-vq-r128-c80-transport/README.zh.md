# C80：配对 weight 运输诊断与补偿联合训练

**已完成并验证成功（2026-10-23）：Comp8与完整运输probe均已完成，最终单卡作业25921642为PBS/artifact Exit0、PHASE_COMPLETE。补偿相对原Joint8将heldout mixed降低5.251%，收回原Joint8−Seq8差距48.488%，未达到预设50%效应门槛；仍比Seq8 mixed高6.256%。详细机制、BF16残差限制和资源记录见[最终分析](ANALYSIS.zh.md)，原始完整结果见[结果表](results/25921642.pbs101/RESULTS.zh.md)。**

**当前执行调整（2026-10-22 23:02 Singapore，用户明确要求）：停止Uncomp8重跑，双卡作业25919309已qdel并确认PBS F/Exit141；Comp8保留3 updates/30 exposures，改为单卡16CPU/110GB、04:00:00续训至final8并完成cal80/heldout80。未完成的Uncomp8不再训练，正式对照直接复用成功作业25912619的原Joint8完整结果。**

实际提交配置与授权记录在`execution.json`。为逐字段匹配已有checkpoint中的训练配置，`protocol.json`保留原始内容；当前allocation和执行臂由明确的execution amendment控制，不能用原protocol中的双卡8小时字段描述当前请求。数据、loss、LR、Adam状态、RNG、batch顺序及最终8次更新目标均不变。完整运输probe和数值gate已完成并复用。

表格中`Uncomp8`是原Joint8历史结果的明确别名，所有复用行保留source binding；其零差值仅检查记录复制一致性，**不算同期完整训练replay通过**。保留原50%恢复/motor与latent不退/3个suite改善的效应门槛，但不声称拥有原计划的同期重放归因强度。原Uncomp8已发生的3次更新记录保留为历史，不计作完成的新臂。

用户授权先复用现有 Seq8/Joint8 检验补偿机制，再实现补偿训练；单节点双卡、walltime大于2h，使用g2并由subagent监控。协议为 `fastwam-vq-r128-c80-transport-v1`。

## 已完成运输探针的判断（2026-10-22）

八个端点均完成 calibration80 与 heldout80，后者为40 contexts×2 seeds；按seed→context→suite、四suite等权汇总。独立CPU汇总作业 `25920427.pbs101` 在compute node上完成，PBS/artifact Exit0、PHASE_COMPLETE；本次汇总新增teacher/student forward均为0。[完整聚合数据](results/probe_analysis/probe_comparison.json)。`probe_report.py`复用原aggregate校验与统计函数，其合成自检逐项比对原完整聚合器；仅输出probe结果，不代填尚未完成的Comp8/Uncomp8。

这里R0已是C80 W16A4训练后的旋转，RJ是原Joint8进一步更新后的旋转，不是R与H的比较。运输比较固定某个端点的未量化压缩函数，只更换R并配对运输decoded weight。结果支持：**RJ相对R0的最终变化在calibration上有收益，在heldout主要指标上有害；不能解释为“旋转方向本来很好，只差weight跟上”。** 这不证明每一步或每个局部方向均有害。

| Heldout端点 | First10 motor RMSE | Future latent NMSE | Mixed loss |
|---|---:|---:|---:|
| Seq原函数，R0 | 0.044207680 | 0.051400843 | 0.004175274 |
| Seq原函数，带补偿换RJ | 0.045392461 | 0.054024152 | 0.004404567 |
| Joint原函数，RJ | 0.045749947 | 0.059470054 | 0.004682388 |
| Joint原函数，带补偿换R0 | 0.044858449 | 0.056623538 | 0.004460030 |

Seq→RJ的heldout motor/latent/mixed分别变差2.68%/5.10%/5.49%，mixed为16胜24负；Joint→R0分别改善1.95%/4.79%/4.75%，mixed为23胜17负。Calibration上，Seq→RJ的mixed改善1.24%，Joint→R0则变差4.48%，两种anchor都出现calibration/heldout的旋转排序反转。

Suite差异不可忽略：Seq→RJ的mixed在spatial/object/goal/10分别变化−0.23%/+24.94%/−5.01%/+4.06%；Joint→R0分别为−18.53%/−6.26%/−5.06%/+15.29%。总体支持R0更适合heldout主指标，不能说每个suite都如此。胜负数是配对描述，不是显著性检验。

关闭A4后的两条运输mixed变化为+0.0000056892、−0.0000213312；最大绝对残差是原Joint−Seq gap（0.0005071143）的4.21%，低于预设10%门限。A4打开时对应变化为+0.0002292928、−0.0002223583；扣除noA4变化的描述性差分仍为+0.0002236036、−0.0002010271。主要指标上的方向不由该平均BF16残差解释，但这不是逐层误差的精确分解。

**平均mixed通过残差门限不等于完整输出逐位等价。** Seq路线heldout的noA4 action最大绝对差达1.9765625，gripper平均RMSE变化+37.26%；当前聚合没有定位该最大差的具体动作维度。不要用此门限宣称所有输出的数值扰动均小。A4下RJ还改善Seq函数的gripper RMSE19.62%，而Joint换R0的gripper变差13.84%；“有害”限于当前first10六维motor与latent主指标，不能推广成所有输出。

对“缺少补偿是否主因”的证据更新：

- 原Joint相对Seq的A4 mixed gap为+12.15%。把Joint函数带补偿换回R0，仍比Seq差6.82%。换坐标只能回收约44%的现有端点gap；这是端点操作结果，不是“补偿训练的因果贡献率”。剩余部分也不能命名为纯VQ误差，它包含Joint已学得的整个函数及其与A4的相互作用。
- 关闭A4后，Joint相对Seq的motor更好6.52%、latent更差5.55%、mixed略好1.23%。这不支持“Joint的底层函数已经全面学坏”。当前证据更指向：函数适配与量化坐标共同变化后，对A4的适应在heldout上失效；calibration上的误差抵消是可能机制，尚未独立验证。
- 缺少weight补偿是已确认的参数化差异，也会改变R的梯度；它可能正是RJ走向不良量化几何的上游原因。故不能从RJ不佳推导“补偿无用”，也不能只凭post-hoc运输证明它是主要原因。

继续完成预先冻结的Comp8/Uncomp8干预：同R0/VQ0、同8×10训练顺序与loss，唯一主要变化为可微weight补偿。待Uncomp8完整重放原Joint8通过后，按原门限检查Comp8是否恢复至少50% gap、motor/latent不差于Uncomp8、至少3/4 suites mixed改善。当前probe报告不对该训练判据作结论。

改进优先级：先完成上述配对干预，暂保Seq8作为incumbent；不要直接把旧RJ带补偿移植到Seq8作为“升级”。若Comp8仍不佳，再独立评估围绕R0限制旋转幅度或缩短R更新预算的方案，保留当前loss并使用独立validation选择超参，避免继续用已检查的heldout调参。native VQ的weight目标随R更新/重编码仍是另一项设计问题，decoded transport本身不解决原d4/B32编码族的可表示性。

## 固定问题与证据边界

原 Joint8 的 R 只作用于 activation，decoder 不含 R。本轮区分：

1. 保持同一个浮点压缩模型时，Seq 与 Joint 的旋转坐标怎样影响 A4 error？
2. 在原 R0/VQ0、原训练配方下，加入可微 weight 补偿能恢复多少 Joint8 相对 Seq8 的 heldout gap？

运输探针自身是端点机制对照；它不能单独证明训练回退的原因。后续 Comp8/Uncomp8 同期训练是设计干预对照。仅有一个训练seed，结论限定在本冻结配方，不作唯一原因或一般统计因果断言。

## 来源与数据复用

- 原始成功双臂：`/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-dual-20261022`，作业 `25912619.pbs101`。
- Seq8、Joint8 的 deployment、源query records，以及共同 R0/VQ0 banks全部复用；不重新拟合moments或banks。
- 同一 C80 cases2000..2079、seed index0；heldout cases3000..3039、seed indices0/1，共40contexts×2=80queries。每个端点完整评cal80/heldout80。
- 同一原 LIBERO demonstrations、sampler seeds、BF16 action/future teachers；新增teacher queries=0。
- source成功summary、artifact Exit0/PHASE_COMPLETE、deployment/plan/query绑定及源重放均为前置门限。执行时从原成功作业归档复制runtime源码，避免引用后来改动的依赖。

## 运输公式与实现

代码保留固定Smooth/sign/H，R=H U，原decoded weight定义在anchor U_a坐标。相对输入逆变换采用：

    B = I + solve(U_current, U_anchor - U_current)
      = U_current^-1 U_anchor.
    V_current = Decode(C,g,I,e) B^T.

FP32下 `(Z U_current) V_current^T = Z U_anchor Decode(C,g,I,e)^T`。逆矩阵写法补偿实际浮点U，避免以U^T近似U^-1时引入已有正交残差；中心化保证current=anchor时B精确等于I。当前U保持可微，anchor停止梯度。完整128块执行相对变换，原固定tail不变。

实际A4/无A4端点仍使用BF16 activation operands、BF16 transport后dense weight和BF16 KV。任意R128运输后的weight通常不能直接折叠回原d4/B32编码族，所以这是 **decoded-weight transport reference**：编码保留anchor坐标，deployment额外保存anchor/current rotations及明确transport语义；没有同native算子、bpw、延迟或closed-loop声明。

训练期每个Linear对补偿计算使用checkpoint，避免长期保留全部旋转后的dense weight图；无梯度评估只缓存当前冻结端点的补偿weight。原未补偿A4对照直接走归档的legacy forward。其余梯度与优化规则沿用原配方。

## 完整运输探针

| 名称 | 浮点函数的anchor | activation坐标 | Weight补偿 | A4 |
|---|---|---|---|---|
| Seq_A4 | Seq8 | R0 | 无，复用源端点 | 开 |
| Seq_to_Joint_A4 | Seq8 | 原Joint8的RJ | 有 | 开 |
| Joint_A4 | Joint8 | RJ | 无，复用源端点 | 开 |
| Joint_to_Seq_A4 | Joint8 | R0 | 有 | 开 |
| Seq_noA4 / Seq_to_Joint_noA4 | Seq8 | 各自同上 | 各自同上 | 关，仍BF16 |
| Joint_noA4 / Joint_to_Seq_noA4 | Joint8 | 各自同上 | 各自同上 | 关，仍BF16 |

每个anchor/样本保存无A4完整action/future的配对残差，源无A4张量缓存在allocation内生成并保留供恢复复用。BF16 folding/rounding残差与A4误差变化分开报告；不能把浮点代数等价直接当作BF16整模型逐位等价。

GPU0/1按context交错均分：各40calibration contexts及20heldout contexts×2seeds；同context两seeds及全部配对端点由同一GPU负责。每卡6个新probe端点×80queries=480新查询，另2个源A4端点只复用缓存。source replay和数值预检属于额外有记录的检查查询。

## 数值门限

- 随机非初始anchor/current U、128与257宽度的FP32输出不变性，最大误差<=1e-5；R梯度最大绝对值<=1e-5。同时确认未补偿路径在该fixture上确实改变输出/梯度。
- 实际BF16 decoder与嵌套sampler/per-Linear checkpoint：初始forward及VQ gradients与legacy逐元素一致；补偿使R梯度发生变化，各梯度finite/nonzero。每次query先在sampler checkpoint外、相同grad mode下建立当前R与补偿矩阵缓存，避免首次forward与重计算的缓存命中改变计算图。
- 两GPU各重复同一完整context forward/backward/Adam step，比较完整outputs、1230组gradients/deltas与更新后outputs，重复更新全部丢弃。
- 将补偿起点的1228组VQ gradients/deltas与原成功作业R0/VQ0探针对照，确保不是偷偷改变VQ初始化或首步更新。R梯度允许按设计变化。
- 最终formal infer parity、保存deployment重载后的完整action/future parity<=1e-6；orthogonality<=5e-5。任何实质数值门限失败都停止正式训练，不静默放宽。

## 原始同期训练方案（已按上述用户要求取消Uncomp8续训）

探针和数值gate全部完成后：

| 项目 | GPU0 Comp8 | GPU1 Uncomp8 |
|---|---|---|
| 初始 | 原R0+VQ0 | 同一原R0+VQ0 |
| 唯一主要设计变化 | 当前R的可微weight补偿 | 原Joint8 legacy路径 |
| VQ / R LR | 5e-4 / 1e-5 | 相同 |
| 更新预算 | 8个batch，每batch10contexts平均梯度 | 相同batch/顺序/80exposures |
| Loss | first10×6 action MSE + 0.04245452683169643×future latent NMSE | 相同 |
| 其余规则 | fresh Adam；两组分别clip norm1；update1后只清VQ Adam state | 相同 |

原模型、Smooth/signs、indices、B32 exponents、D=I、r=1、BF16 KV冻结。无新增loss、无warm-up/anchor/LR sweep；固定final8，不按heldout选checkpoint。Uncomp8逐步记录与原Joint8训练的差异，最终对原Joint8完整cal/heldout逐query七指标重放检查。

deployment保留足够的anchor/current R、C/g、source banks与执行语义，并通过实际文件重载检查。不是在训练结束后临时给未补偿模型加一个变换。

## “主要原因”的预设判读

主指标为heldout mixed。令 G=Uncomp8−Seq_A4，恢复量 E=Uncomp8−Comp8。

- G>0时报告E/G；G<=0时不定义恢复比例。
- 同期Uncomp8必须在全部cal/heldout逐query七指标重现原Joint_A4，最大绝对差<=2e-6；否则训练归因标为inconclusive。
- 至少恢复50%正gap、Comp对Uncomp的motor/latent均不退、至少3/4suite mixed改善时，支持“本固定配方中补偿设计是主要贡献因素”。这不是普遍或唯一原因结论。
- 运输几何的强解释还要求无A4的绝对mixed残差不超过原Joint_A4−Seq_A4正gap的10%；否则明确标记BF16数值混杂。无论通过与否，完整端点和残差都报告。
- 聚合为seed→context→suite→四suite等权；heldout独立配对单位40contexts。报告七项指标、逐suite和配对胜负，gripper/late22不因mixed改善而隐藏。

## Allocation、恢复与监控

原双卡阶段请求单节点2GPU、32CPU、220GB、08:00:00，通过normal routing实际进入g2；共享诊断按context均分，随后两训练臂独立并行。当前改为1GPU、16CPU、110GB、04:00:00，normal routing进入g1，仅继续Comp8；完成即退出。完整probe/gate只复用并重新绑定元数据，不重复模型计算。

每15秒写双卡utilization/VRAM日志。任何配对worker失败即停止同阶段其余工作，保留原子写入的queries、probe tensors与每次optimizer更新checkpoint；确认终态失败后才可修复和resume。监控subagent使用gpt-6-luna/xhigh，每20分钟读取当前handle，无实质变化保持安静。

Login仅轻量连接、qsub/qstat、小型源码/JSON控制；所有模型、Torch、tensor读写及真实结果聚合都在真实PBS allocation内。保留host-key验证，不处理banked reset，不输出credentials。

## 实现修复记录

- Attempt0：`25917991.pbs101`，g2，PBS F/Exit1，artifact Exit1，walltime 00:11:22。FP32非初始旋转及实际BF16单层补偿检查在两卡均通过，但完整梯度probe在backward发生checkpoint tensor-count mismatch（3589 vs 3585）；正式probe endpoints和训练更新均未产生。
- 原因是相对补偿矩阵首次在外层sampler checkpoint内lazy计算，重计算时命中缓存。修复将可微R/补偿缓存提前到每次query的checkpoint之外；数学补偿、loss、数据、预算与门限不变。增加缓存顺序的stdlib回归测试及嵌套checkpoint数值fixture。新fixture和完整probe必须在恢复allocation中重新通过，不复用旧unit通过结果代替新检查。
- Attempt1：`25918274.pbs101`，g2，PBS F/Exit1，artifact Exit1，walltime 00:45:18。修复后的嵌套checkpoint及完整重复梯度probe通过；两卡各1228组初始VQ gradients/deltas和原action/future输出与源probe的差值均为0。Seq anchor各子阶段记录已生成，但切换Joint anchor时旧补偿dense缓存尚未释放，新weight解码触发OOM。将缓存释放提前至replacement decode之前，并以stdlib回归测试约束该资源顺序；保留已有绑定正确的Seq queries/pairs供恢复复用。未产生正式训练更新。
