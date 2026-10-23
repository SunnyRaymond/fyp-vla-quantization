# C80 R128：补偿运输与 Comp8 完整结果分析

完成时间：2026-10-23 Singapore。最终作业 `25921642.pbs101` 已确认 PBS F/Exit0、artifact Exit0、PHASE_COMPLETE；完整性验证通过。以下数值来自 compute allocation 内生成的 [comparison.json](results/25921642.pbs101/comparison.json) 与 [训练记录](results/25921642.pbs101/training.json)，本分析没有新增模型计算。

## 结论

**weight 未随 R 配对补偿确实是应修正的参数化问题；补偿后的 Comp8 有收益，但未达到预设“主要贡献”的效应门槛，也没有超过 Seq8 的主指标。** Comp8 相对原 Joint8 将 heldout mixed 降低5.251%，收回原 Joint8−Seq8 指标差距的48.488%；预设门槛为至少50%，不能四舍五入宣称通过。这是指标差距的恢复比例，不是已识别的因果贡献比例。

运输探针还发现：在保留同一个浮点压缩模型并补偿 weight 后，原 Joint8 的旋转 RJ 对 heldout A4 依然不如 R0，而 calibration 倾向相反。因此，回退不能只解释为“weight 没跟上 R”；还存在旋转对 activation 量化几何的泛化/适配问题。Comp8 的 hard-quantized 同 batch loss 也并非稳定下降，保留了优化问题的证据。

用户要求停止 Uncomp8 重跑后，本次正式对照直接复用成功作业 `25912619.pbs101` 的原 Joint8 全部结果。生成表中的 `Uncomp8` 只是该历史结果的别名。没有同期完整 Uncomp8 重训证据；记录复制的一致性不算 replay gate 通过。

## 结果与公平条件

相同 R0/VQ0、相同 C80 cases2000–2079/seed0、8个同序 batch×10 contexts、相同 Adam/LR/reset 规则，loss 固定为 first10×6 motor MSE + 0.04245452683169643×future latent NMSE。Smooth/signs、indices、B32 exponents 冻结，D=I、r=1、BF16 KV。没有新增 teacher queries。

Heldout 为 cases3000–3039×seeds0/1，即40个独立配对 contexts、80 queries；先均值 seeds，再 contexts，再四 suite 等权。不是未见任务泛化，也不是 closed-loop 成功率。

| Heldout arm | First10 motor RMSE↓ | Latent NMSE↓ | Mixed↓ | Full32 motor RMSE↓ | Late22 motor RMSE↓ | Gripper RMSE↓ |
|---|---:|---:|---:|---:|---:|---:|
| Seq8 | 0.044207680 | 0.051400843 | 0.004175274 | 0.047952118 | 0.049053147 | 0.038028946 |
| 原 Joint8 | 0.045749947 | 0.059470054 | 0.004682388 | 0.049379018 | 0.050440249 | 0.049313403 |
| 补偿 Comp8 | 0.044157657 | 0.057518565 | 0.004436499 | 0.049629828 | 0.051252111 | 0.034355929 |
| Comp8 相对 Joint8 | −3.480% | −3.281% | −5.251% | +0.508% | +1.610% | −30.331% |
| Comp8 相对 Seq8 | −0.113% | +11.902% | +6.256% | +3.499% | +4.483% | −9.658% |

Comp8 相对 Joint8 的 mixed 配对为24改善/16变差；相对 Seq8 为21改善/19变差。后者胜负接近而平均 mixed 更差，不能用略多的 context 胜数代替总体指标。

注意 mean RMSE 与 mean MSE 不可互相平方换算。Comp8 相对 Seq8 的 motor RMSE 低0.113%，但 motor MSE 为0.001994576，对比0.001993076，反而高0.075%。因此不能把 motor loss 说成整体改善。

| Suite | Comp8 vs 原Joint8 mixed | Comp8 vs Seq8 mixed |
|---|---:|---:|
| libero_spatial | −21.520% | −6.328% |
| libero_object | −1.775% | +11.430% |
| libero_goal | −5.252% | −0.143% |
| libero_10 | +11.550% | +24.282% |

补偿相对 Joint8 达到3/4 suites改善、motor/latent总体不退；但48.488%的恢复比例仍未到50%。收益存在明显 suite 差异，尤其 libero_10 仍回退。

Gripper 均值改善也不能称为普遍改善：Comp8 vs Joint8 是5个 contexts改善、35个变差，均值下降主要来自少数较大退化的修复。完整输出需要保留这些尾部风险。

## 配对运输说明了什么

对每个固定 decoded-weight anchor，使用相对补偿将 activation 坐标从 R0 换到 RJ，或反向换回；不是只交换 R 而保持 weight 不动。FP32 中固定 VQ、关闭A4时保持原函数。

| Heldout 对比 | Motor RMSE变化 | Latent NMSE变化 | Mixed变化 |
|---|---:|---:|---:|
| Seq8 保留浮点函数，R0→RJ，开启A4 | +2.680% | +5.104% | +5.492% |
| Joint8 保留浮点函数，RJ→R0，开启A4 | −1.949% | −4.786% | −4.749% |

两种 anchor 均显示 RJ 的 heldout A4 主指标较差。与此同时，calibration 的 Seq anchor mixed 从0.003954003降至0.003904830；Joint anchor 从RJ换回R0反而由0.003889337升至0.004063572。**保持浮点模型的前提下仍出现 calibration/heldout 排序反转，支持 activation 量化几何本身的泛化问题。** 这不是单靠让 VQ 自动补偿坐标漂移就能解释的现象。

关闭A4但保留BF16时，Seq/R0与Joint/RJ的 heldout mixed 分别为0.001539290、0.001520416，motor RMSE分别0.023100950、0.021595245。旧Joint的无A4端点并没有在这两项上整体更差。因此“未补偿导致浮点压缩模型全面劣化”也过于简单；A4与旋转/weight适配的相互作用不可忽略。noA4切换不是可加的独立误差分解，不能把这些差值直接相加归因。

### BF16 残差的边界

两条 noA4 运输路线的 heldout mean mixed 变化为 +5.6892e−6、−2.1331e−5；最大绝对值小于预设门槛5.0711e−5（原正gap的10%）。这支持**总体主指标尺度上的**几何比较。

但它不证明逐样本完整输出一致：Seq anchor heldout 的全action最大绝对差为1.9765625，最大action NMSE为0.0664855；Joint anchor对应最大绝对差为0.0185547。Seq anchor的gripper均值也发生变化。当前保存的残差摘要不能定位该极值属于哪一维/哪一步，不能擅自断言它只来自gripper。

自动聚合文件中的“可作强几何归因解释”只由 mean mixed 门限触发，应按上述限定范围阅读；不能推广到全部action/latent张量或gripper。FP32代数等价不等于BF16逐位等价，R0/RJ对无A4的细微数值改变在迭代模型中仍可能被放大。

## 补偿修好了什么，为什么还不够

当前实现把 C/g 保留在固定anchor坐标，使用可微的相对逆变换：

    B = I + solve(U_current, U_anchor − U_current)
    V_current = Decode(C,g,I,e) B^T

完整128块上，FP32满足 `(Z U_current) V_current^T = Z U_anchor Decode(...)^T`。当前U经过 activation 与 weight 两支反向，anchor停止梯度；中心化使初始相对变换精确为I。模型在权重transport后再转BF16。

已验证：非初始FP32输出最大误差≤1.79e−7，R梯度最大绝对值≤2.08e−9；嵌套checkpoint检查通过。完整起点的1228组VQ gradients、Adam deltas及action/future与原probe差值均为0，跨GPU的1230组完整gradients/deltas及重复outputs差值也为0。Comp8最终formal infer与实际deployment重载后的完整action/future差值均0，最大正交误差2.3842e−6。

补偿改变了R的梯度：首batch R gradient norm从旧Joint的0.0466505变为0.00601738；VQ起点梯度保持一致。**梯度norm变小不等于Adam后的实际旋转步长变小**，本轮未保存完整逐步旋转距离/角度，不能据此声称已限制函数敏感度。

| Calibration arm | Motor RMSE↓ | Latent NMSE↓ | Mixed↓ |
|---|---:|---:|---:|
| Seq8 | 0.043426336 | 0.047780080 | 0.003954003 |
| 原Joint8 | 0.043118979 | 0.046901237 | 0.003889337 |
| Comp8 | 0.043238925 | 0.050030558 | 0.004043222 |

Comp8的calibration latent/mixed也不如Seq8，说明剩余问题不能只归结为heldout数据不足。训练记录中，Comp8仅updates2/5/6/8降低同batch mixed；update3从0.004063476升至0.005830444，update7也明显上升。旧Joint为7/8次下降，Seq为4/8次。不同batch之间不能连成收敛曲线，但这些同batch before/after足以说明现有STE/Adam步并不稳定改善真实hard量化目标。

可能机制包括：R与VQ同时变化、hard A4阈值敏感、迭代采样放大，以及固定indices/exponents限制适配；本轮未逐一分离。补偿修正了等价旋转的计算图，却不会自动使当前LR、更新频率或STE成为好的优化配方。

更重要的是，post-decode运输固定VQ编码主要提供干净的activation量化优化方向；它不会让weight量化目标自动变成 `Q_VQ(W_s R)`。任意R128一般也不能折叠回当前d4/B32编码族。因此本实现是有完整保存/重载语义的 **decoded-weight transport reference**，不是已经保持相同native VQ格式、bpw或延迟的部署改进。

## 后续优先级（未启动新实验）

1. 当前保留Seq8为A+Z主参考；不要因Comp8 motor RMSE略低就替换mixed/latent更好的Seq8。
2. 优先处理补偿路径下的hard-loss更新稳定性：以固定VQ的R小步诊断区分R与VQ更新，再独立比较更小R步长/较低R更新频率或trust region。先用calibration-validation选规则，不依本轮反复查看的heldout挑参数。补偿应一直保留，不能用小LR替代它。
3. 若目标是R同时改善weight-VQ与A4几何，明确weight目标随当前R变化，考虑小步交替R更新、重新编码/投影、固定R的VQ恢复；最终必须评估实际hard编码模型。该路线工程成本高于当前reference，应单独立项，不与本轮结果混淆。
4. 暂不同时加入D、activation scaling或新loss。Gripper与late22的保护属于另一个目标设计问题，不能用当前first10主指标改善替代验证。

## 执行与资源记录

用户调整后，Comp8从3updates/30exposures恢复，参数、Adam、RNG及原配置逐字段匹配；完成同一8次更新目标，lineage为25919309→25921642。原Uncomp8只做过3次更新，随后按要求停止，没有被计作完整新臂；正式比较复用原Joint8 final8。

| 作业 | GPU数 | 实际walltime | 终态与用途 |
|---|---:|---:|---|
| 25917991 | 2 | 00:11:22 | Exit1；修复嵌套checkpoint缓存 |
| 25918274 | 2 | 00:45:18 | Exit1；已完成Seq记录，修复切换anchor的缓存释放顺序 |
| 25919309 | 2 | 01:54:24 | 用户要求qdel；完整probes/gate，Comp8与Uncomp8各3updates |
| 25921642 | 1 | 02:35:43 | Exit0；Comp8续训、完整评估与聚合 |

按allocation walltime×请求GPU数计约8.30 GPU-hours，包含失败尝试及停止前的Uncomp8部分训练，不是实测活跃计算时间；不含既有来源实验。GPU utilization/VRAM按15秒写入各作业日志。最终作业已释放资源，监控nextcheck=null，无后续任务自动提交。

原始完整表格：[RESULTS.zh.md](results/25921642.pbs101/RESULTS.zh.md)。实际执行修订：[execution.json](execution.json)。数值证据：[unit_receipt.json](results/25921642.pbs101/evidence/unit_receipt.json)、[Comp8 summary](results/25921642.pbs101/evidence/arms/Comp8/summary.json)、[断点恢复receipt](results/25921642.pbs101/arms/Comp8/resume_receipt.json)。
