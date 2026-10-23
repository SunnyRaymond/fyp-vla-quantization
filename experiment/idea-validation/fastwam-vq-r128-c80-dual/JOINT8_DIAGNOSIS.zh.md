# Joint8 vs Seq8：设计、训练行为与泛化的分层诊断

分析日期：2026-10-22。依据已完成作业 `25912619.pbs101` 的训练记录、完整端点评估、receipts，以及该作业归档的实际执行源码。本次只读取小型源码/JSON并进行数学分析，没有新训练、模型 forward、tensor 计算或 PBS 提交。

## 判断

**当前 Joint8 对“学习等价旋转以改善量化”的目标存在确定的设计偏离：R 仅改变 activation 坐标，没有在 weight 分支做配对补偿。** 它是可以优化的 compressed-student 参数化，但其 R 同时承担改变模型函数的作用。这个设计偏离应先处理，再考虑增加 calibration、延长训练或调小 LR。

现有结果同时显示 calibration/heldout 的排序反转，支持泛化问题的存在；但它不能区分普通过拟合、未补偿的函数漂移、受限 VQ 的补偿不足和离散量化优化误差各自贡献。不能把 heldout mixed +12.15% 全部归因于一个因素，也不能保证补偿后必胜 Seq8。之前仅将问题概括为“泛化/适配”不够准确：那是表现，下面的计算图差异才是已确认的结构事实。

## 实际执行路径

已读取该作业 `artifacts/25912619.pbs101` 内的 `rotation_bank.py`、`c8_train_banks.py`、`micro_codebook.py`、`worker.py` 和 `selfcheck.py`；它们与本次检查的本地源码逐行一致。小型源码副本在 [executed_sources](results/25912619.pbs101/evidence/executed_sources)。

| 路径 | activation | weight | R 的梯度路径 |
|---|---|---|---|
| 原 W16A4 rotation 训练 | A4((X/S) H U) | W S H U | activation 与 weight 两支 |
| 当前 Seq8 | A4((X/S) H U0) | Decode(C,g,I,e) | R 冻结 |
| 当前 Joint8 | A4((X/S) H U) | Decode(C,g,I,e) | 只有 activation 一支 |

证据位置：

- `rotation_bank.py:131–147`：W16A4 使用 `transformed(..., weight=True)`，两支共享当前 U。
- `rotation_bank.py:180–186`：VQ forward 的 R 只出现在 activation；`DecodedWeight` 没有接收 R。
- `c8_train_banks.py:30–63,121–132`：weight 的前向使用刷新后的 BF16 dense decoder，反向只返回 books/log gains 梯度；refresh 按当前 C/g 重新解码，固定 I/e，未做 R 补偿。weight 并非忘记 refresh，也并非完全不更新。
- `worker.py:375–379,543–550`：两组 Adam 分别更新，VQ LR=5e-4、R LR=1e-5；没有约束 VQ 更新必须等于 R 引起的坐标补偿。

这也是 C8 Joint8 文档所描述的 compressed-student co-adaptation 路线。它不是一次运行偶然漏执行了补偿，而是原联合方案的参数化选择。需要修正的是设计与结果解释，而非将其包装成数值随机性。

## 为什么这改变了学习问题

以下按 row-vector Linear 写法，省略 bias；Smooth 与固定 signed H 均已配对。记 Z=X/S，R 为 H 与 learned U 合成的 block-orthogonal transform，W_s=W S。V(C,g;I,e) 是 VQ 解码后的 weight，初始化 V0 近似 W_s R0。

W16 的等价关系为：

    (Z R) (W_s R)^T = Z W_s^T = X W^T.

当前 Joint 是：

    Y = Q_A(Z R) V(C,g;I,e)^T.

冻结 V，关闭 A4 后，单独把 R0 改成 R：

    Y(R) - Y(R0) = Z (R - R0) V^T.

这个差异一般不为零，而且即使 V0 精确等于 W_s R0、原始 VQ 误差为零，也仍然存在。因此当前 R 梯度包含直接改变模型函数的方向，不能解释成只在降低量化误差。

用一阶展开也能看清两组参数应该如何配合。对正交 R 的切向变化 dR=R Omega、Omega^T=-Omega，等价补偿要求 dV=V Omega。于是：

    dY = Z dR V^T + Z R dV^T
       = Z R Omega V^T - Z R Omega V^T = 0.

独立 Adam 更新 C/g 与 R，并不保证 dV=V Omega。当前 weight 梯度链也缺少与 activation 分支抵消的 R 导数。原 W16A4 训练和 Joint8 第二阶段虽然都叫 R 训练，实际学习的梯度方向已经改变。

这还可以解释为什么补偿不仅影响forward，也改变训练信号。记当前activation为A=ZR；在正交切向更新dR=R Omega下，若A4的STE把量化导数近似为identity，未补偿路径的一阶变化是`dY ≈ A Omega V^T`。配对weight满足`dV=V Omega`时，两支合起来成为`dY ≈ (A-Q_A(A)) Omega V^T`。后一项随activation量化残差消失，前一项即使没有量化误差也存在。这里是理想浮点与该STE假设下的局部推导；实际BF16 round、共享R的多层累积及decoder surrogate仍需数值检查。因此补偿不只是“帮weight追上R”，它把R的优化方向重新约束到量化引起的差异上。

SpinQuant 的 [Appendix B.1](https://arxiv.org/html/2405.16406v4) 也以“去掉量化后 R 的梯度归零”解释等价 rotation learning。该论据用于区分学习问题；不能直接把 LLM 上的结果当成本项目证据。

注意这里的等价性要求是：**固定 C/g，只改变 R 时保持浮点函数。** C/g 本身做 task-aware 训练时当然允许模型函数改变；不能要求整个 QAT 过程中输出始终等于原模型。

## 为什么现有 VQ 难以自动承担补偿

本轮每层的 decoder 是 d4、M2、K256，共享 codebooks，带 row gain 和 B32 power-of-two exponents。每个 d4 vector 的形式为：

    v[row,group] = g[row] * 2^e[row,B32(group)]
                   * (C0[I0[row,group]] + C1[I1[row,group]]).

任意 R128 的相对旋转会混合一个128-channel块中的32个 d4 vectors，以及4个 B32 scale blocks。旋转后一个 d4 vector 往往是多个旧 vectors、多个不同 scale 与 codeword 组合的线性组合；固定原 indices/exponents，仅更新共享 codebooks 与 row gain，通常无法精确表示全部补偿矩阵。

所以“一起训练了 VQ，VQ 就会自动把 R 补偿回来”没有保证。它既要拟合原量化误差，又要适应 R 引入的函数变化，可能利用 calibration 上较容易的方向降低 loss，却在独立 contexts/seeds 上回退。此机制在数学上成立，但本轮没有保存补偿残差/分层输出漂移，尚不能量化其实际占比。

## 训练与结果支持什么

| Split / arm | First10 motor RMSE | Future latent NMSE | Mixed loss |
|---|---:|---:|---:|
| Calibration / Seq8 | 0.043426336 | 0.047780080 | 0.003954003 |
| Calibration / Joint8 | 0.043118979 | 0.046901237 | 0.003889337 |
| Heldout / Seq8 | 0.044207680 | 0.051400843 | 0.004175274 |
| Heldout / Joint8 | 0.045749947 | 0.059470054 | 0.004682388 |

Joint 的 calibration 三项略好，heldout 三项分别回退3.489%、15.699%、12.146%；四个 suite 都出现这三项回退。Mixed 的配对胜负为15改善/25变差，而不是80个独立样本。Heldout是相同tasks的独立demonstrations、两个sampler seeds；calibration只有seed index0。这里是context/demo及采样条件的泛化，不能称为未见任务泛化。

Gripper还显示另一种风险：Joint相对Seq的平均RMSE回退29.673%，但40个contexts中35个改善、5个变差。均值与胜负反向说明少数较大回退拉高了总体误差；不能据此说大多数contexts的gripper都变差，也不能用35胜就忽略尾部退化。当前loss没有直接保护gripper，故它是额外自由度影响未监督输出的警示，尚不是缺少补偿的独立因果证据。

同 batch 的 before/after 记录显示 Joint 有7/8次更新降低 mixed，Seq有4/8次。Joint 并非整体没有优化成功；另一方面，实际 hard-quantized loss 并非每步下降，说明不能只用STE梯度的存在判断真实更新收益。各update使用不同contexts，不能把跨batch的原始loss连接成收敛曲线。

第一次更新最有诊断价值：两臂从相同 R0/VQ0、同batch出发，记录的 VQ gradient norm 同为0.00682159094。更新后已经出现不同的取舍：

| 首个batch | Motor RMSE | Latent NMSE | Mixed loss | Gripper RMSE |
|---|---:|---:|---:|---:|
| 更新前，共同起点 | 0.047082585 | 0.075814973 | 0.005493702 | 0.031524678 |
| Seq 更新后 | 0.045758804 | 0.078717442 | 0.005495148 | 0.030899591 |
| Joint 更新后 | 0.049125340 | 0.066317880 | 0.005295616 | 0.150716499 |

Joint 的第一步降低 latent 和 mixed，却提高 motor，gripper也大幅波动。R 的额外自由度从第一步就改变了拟合取舍，而非只有最后才出现回退。不过本轮只记录了跨臂 VQ 梯度norm，不能用相同norm声称完整梯度/更新逐元素相同；也没有保存该步R-only/VQ-only反事实结果。

Loss仅保护first10×6 motor和future latent；gripper、late22 motor没有直接项。两臂lambda相同，不能把Joint回退简单归因于lambda设置，但新参数方向与既有loss可能发生不同的取舍。没有分量梯度，不能断言latent梯度“压倒”了motor。

## 已排除到什么程度，仍缺什么

| 项目 | 已有证据及边界 |
|---|---|
| 数据/初始化/预算不匹配 | 相同R0/VQ0、8×10的同批顺序、相同loss；源重放和初始端点parity通过。Joint额外更新R，计算预算不完全相同。 |
| 梯度断连或NaN | 每步614层VQ记录均connected/finite/nonzero；Joint两路R均有非零有限梯度。证明连通，不证明梯度对应等价rotation目标。 |
| clipping导致两臂行为不同 | 两臂记录的pre-clip group norm均小于1；R最大约0.191，VQ最大约0.0422，本轮norm clip未触发。 |
| 正交约束失效 | 最终最大orthogonality error约2.62e-6；没有支持正交性崩坏的证据。正交并不限制偏离R0的角度。 |
| 部署/训练不一致 | formal-infer probe max_abs=0；只说明同一模型的两条路径一致，未检查模型是否仍为等价旋转。 |
| 数值不可复现 | 固定probe在进程内/跨GPU的forward、1230组梯度及Adam delta均一致；只覆盖该probe，不能据此排除所有优化问题。 |
| 现有selfcheck为何没发现设计差异 | Joint测试覆盖U=I的H/VQ0起点、梯度连接、checkpointing一致性；没有测试非初始R下，固定VQ、关闭A4后是否仍等价。W16的配对旋转测试不能替代VQ路径的测试。 |
| R LR是否过大 | LR=1e-5不足以判断实际函数变化；没有逐步R相对R0的距离、最大旋转角、有效weight漂移或量化翻码率。Adam的参数步长也不能由raw gradient norm直接推断。 |
| 是否只是过拟合/训练不足 | calibration/heldout排序反转是事实，但没有固定validation全程轨迹、跨训练seed或补偿对照。80个contexts是否不足、是否需要更多updates均未被识别。 |
| VQ/R optimizer状态不对称 | update1后只清VQ Adam state，R继续保留；可能影响后续共同适配。Reset本身不改变已更新参数，不能解释第一步输出分化。Seq也清VQ状态，单凭这一点不能归因。改变它应单独比较。 |

## 改进顺序

### 1. 保留Seq8；先确立正确的补偿参考实现

把 C/g 解释为 R0 坐标中的编码，当前 R 的补偿 weight 应为：

    V_comp(R,theta) = Decode(theta) R0^T R
    Y = Q_A(Z R) V_comp(R,theta)^T.

关闭A4、固定theta后：

    (Z R) [V(theta) R0^T R]^T = Z R0 V(theta)^T.

实际代码 R=H U，H固定，故只需每个完整128块右乘 U0^T U，尾部不加learned旋转。**不能直接乘U**：VQ0已在U0坐标中拟合，直接乘U会重复包含原旋转。

补偿必须处在可微forward中，当前U不能detach，也不能只在`no_grad refresh()`里修改weight；否则forward可能补偿了，R的反向仍缺少weight分支。U0是固定anchor，应停止梯度。C/g更新可改变浮点函数，R-only更新则应通过FP32不变性及零梯度检查。

这个版本适合先回答机制问题，但尚不是同一原生VQ算子的直接替代。V_comp通常已超出原d4/B32编码族。decoded-BF16验证可以计算它；要保持原格式部署，需要重新编码或明确增加变换算子，并重新验证最终hard编码模型。不能只保存原C/I/e却声称已融合补偿，也不能仅在最终导出时补偿，因为那会改变训练过的函数。

还要区分两件事：对固定decoded weight进行补偿，会保持其未量化模型函数；它不会凭空降低weight VQ的重建误差。它主要为R提供干净的activation-quantization优化方向。真正改变weight VQ几何，还需要下一项。

### 2. 若目标是完整R+VQ共同优化，让weight量化目标跟随R

概念上应为：

    Y = Q_A(Z R) Q_VQ(W_s R; codebooks, scales, assignments)^T.

当前方法只在初始化时拟合了W_s R0，之后不再重新定义或投影到W_s R。较稳妥的工程路线是小步交替：更新受约束的R；在新坐标下重新分配/拟合I、B32 exponents及必要的C/g；固定R后做VQ恢复训练；每个阶段都按实际hard编码端点评估。也可以transport已有训练后decoded weight，再投影回VQ编码族，以保留已有task适配，但需明确锚定的是这个压缩模型而非原始W。

这是需要重新设计的交替/双层优化，并非把一行STE接到R就完成。若采用可微量化近似，真实forward必须使用当前旋转目标对应的量化结果；既要保留C/g的梯度，也要明确R经过weight量化支路的surrogate，不能只添加反向补偿却维持旧forward。

实施时需保留原始W_s或明确选定的浮点anchor。当前`mod.weight.data`已被decoded VQ覆盖，不能在VQ安装后直接把它当原始W再调用`weight=True`变换，否则Smooth/R会被重复应用。

另一种低容量、较易保持格式的路线是限制相对旋转为每个d4组重复同一个Q4。理想浮点下它可通过所有codewords右乘Q4折叠，且不混合B32 scales；BF16舍入仍需验证。但这只允许很小的旋转子空间，不能把它称为完整R128能力的替代。

### 3. 参数化明确后，再做优化稳定措施

- 先VQ warm-up，再解冻R，减少两个坐标/权重目标同时移动。
- 在既有action MSE+latent NMSE之外，单独测试对R0的trust region/anchor，例如归一化的||R-R0||_F^2；同时记录实际步长、函数漂移和hard量化翻码率。小LR与anchor都是稳定手段，不能代替等价补偿。
- 对R/VQ使用不同更新频率，或交替更新；VQ optimizer的update1 reset规则也应在独立对照中评估。
- 维持原A+Z/rho1.5主对照。Gripper/late22保护项属于另一个loss问题，不应与补偿修正一起改变，否则难以归因。
- 选择早停、LR或正则强度使用预先划定的calibration-validation；不继续依当前反复查看的heldout挑选。

## 下次计算前，最有区分度的最小诊断

以下是本诊断形成时提出的最小诊断；后续执行状态见文末，不能把待验证假设当成结果。

1. FP32 unit gate：固定一个decoded weight，关闭A4与新增BF16 round，改变非初始R。补偿路径应保持输出不变，loss对R的梯度接近零；当前路径一般不会。只验证计算图即可，不需要先重训整模型。
2. 使用现有Seq8 weight与Joint8 R做成对运输探针：对比Seq8原模型与 `Q_A(Z R_J)[V_Seq R0^T R_J]^T`，分别启用/关闭A4。两者表示同一个浮点压缩模型，可检验Joint学到的旋转方向本身对activation量化是否有益。局部Linear探针只能支持局部机制，完整输出结论需要原cohort的固定输入评估。
3. 同时记录三类量：未量化函数漂移、activation量化输出误差、投影/重新编码后的weight与最终输出误差。把最终原生格式端点单列，避免decoded reference好却导出后变差。
4. 若第1项成立且第2项值得推进，再在同一C80、同一VQ预算、同一loss下比较冻结R与补偿版本；旧Joint保留为未补偿参数化对照。

不要只把Joint的R替换回R0而保持其weight不动，就称为“纯R贡献”：Joint的codebooks已与R共同适配，直接交换也会引入坐标/函数变化。也不要在任意weight上事后补偿而不重训，就把得到的指标当作“原训练只差一行”的证据。

## 来源

- [训练记录](results/25912619.pbs101/training.json)
- [完整聚合及配对比较](results/25912619.pbs101/comparison.json)
- [实际执行源码](results/25912619.pbs101/evidence/executed_sources)
- [数值重复性receipt](results/25912619.pbs101/evidence/repro_receipt.json)
- [SpinQuant论文](https://arxiv.org/html/2405.16406v4)：等价旋转及Appendix B.1的梯度性质。

本次没有用新增数值实验验证任何改进收益。已确认的是计算图与等价性条件的差异；其对12.15% heldout mixed回退的定量贡献仍待机制对照。

## 后续验证状态

2026-10-23已完成配对运输probe与Comp8训练/完整评估，最终作业`25921642.pbs101`验证成功。用户中途要求停止Uncomp8重跑并改为单卡Comp8续训，故正式对照复用本作业的原Joint8。补偿收回原heldout mixed差距的48.488%，未达到预设50%效应门槛；运输还显示RJ的heldout A4几何不如R0，不能只归因为weight未补偿。完整机制、BF16输出残差与历史对照的边界见[后续最终分析](../fastwam-vq-r128-c80-transport/ANALYSIS.zh.md)。本文件此前各节保留为新计算前的源码诊断，不应把其中的待验证假设当作最终实验结论。
