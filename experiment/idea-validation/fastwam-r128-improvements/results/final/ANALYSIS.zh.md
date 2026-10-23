# R128 改进实验：完整结果与原因分析

三项实验均已完成并验收。第3项按用户最新要求只训练C80，与已有C8-R8和H比较；不要求等样本处理量。全部预先固定的臂均完整报告。

**主要发现：C80 recipe修复了原C8-R8的motor/latent平均指标退化，heldout相对H的motor RMSE为-0.38%、latent NMSE为-10.50%、mixed loss为-5.36%。但motor收益很小，而且gripper RMSE为+25.08%，37/40个context变差。当前证据支持扩大覆盖并平均梯度后，既定motor+latent目标的表现改善；尚不足以宣称整体action质量稳定优于H。**

## 1. 固定协议与证据范围

- W16A4、无VQ；冻结原始BF16 model weights、Smooth0.5/signs、BF16 KV、D=I、clipping ratio=1。只训练action/video两路共享128×128 Cayley raw矩阵，最终update8全部冻结后评估。
- Loss为前10步×6维motor MSE + 0.04245452683169643×clean-future latent NMSE；latent排除clamped frame0。沿用原rho1.5和normalization c，不从heldout重调loss权重。
- C8为固定8个contexts；heldout为40个contexts×2 seeds，先seed→context→suite聚合，再四suite等权。配对单位是40个contexts，不能将80条queries当80个独立样本。
- C80使用80个不同contexts、40个tasks、每suite20个contexts，8次更新，每次取10个context的平均梯度，LR1e-4。原C8-R8为8次单context更新、LR1e-3。因此C80比较同时改变覆盖、LR和梯度平均方式，是整体recipe对照。
- C8小LR臂与C80也可作描述性比较，但不能由此分离“覆盖更多context”和“平均10个梯度”的贡献。每个recipe只有本次一条训练轨迹，未评估跨训练重复性。
- C80覆盖heldout中的全部40个task，使用不同demonstration；这提供跨demo/context的固定输入证据，不是heldout新task泛化证据。下文“原C8未覆盖task”是相对原C8而言，不是相对C80而言。
- 未按heldout挑选t、LR、checkpoint或追加训练。没有closed-loop、native kernel或延迟结论；全部teacher复用，新增teacher queries=0。

## 2. 所有固定端点

下表均为相对H的误差变化，负数更好。C8列只用于观察拟合，不用于选臂。

| Arm | C8 mixed | Heldout motor RMSE | Heldout latent NMSE | Heldout mixed |
|---|---:|---:|---:|---:|
| H，t=0 | 0 | 0 | 0 | 0 |
| 原C8-R8，grouped LR1e-3，t=1 | -17.98% | +4.65% | +5.12% | +7.67% |
| 原R8方向t=0.25 | -13.01% | +1.27% | +9.22% | +5.96% |
| 原R8方向t=0.5 | +32.57% | +0.18% | +11.09% | +5.75% |
| C8 grouped LR1e-4 | -18.22% | +4.07% | -0.07% | +4.37% |
| C8 interleaved LR1e-3 | +20.39% | +12.55% | -0.95% | +13.30% |
| C8 interleaved LR1e-4 | -23.42% | +3.08% | -2.29% | +2.21% |
| C80，8×10平均梯度，LR1e-4 | -1.32% | -0.38% | -10.50% | -5.36% |

完整绝对值、七项metric、逐suite/逐context结果见[RESULTS.zh.md](RESULTS.zh.md)与[comparison.json](comparison.json)。H heldout motor RMSE=0.0426045556，latent NMSE=0.0383806173，mixed loss=0.00347420612。

### 旋转幅度不能单独解释结果

沿原R8方向减小幅度可以减轻motor退化，但t0.25/t0.5的latent反而比原R8更差，固定四点均未同时超过H。C8 loss沿路径也明显不单调。较小的最终R不等价于小LR重新训练。

C80的action/video ||U-I||F为0.12448/0.08628，C8 grouped1e-4为0.09672/0.08431，interleaved1e-4为0.09968/0.08019。C80的action旋转反而略大，heldout主指标却更好。因此收益不能简单归为“R离H更近”，所学方向和训练过程同样重要。

### 小LR有帮助，交错顺序不保证有帮助

在本次观测中，grouped降LR使heldout motor相对原R8低0.55%；interleaved降LR使motor低8.41%。但在LR1e-3时，interleaved比原grouped的motor高7.55%；LR1e-4时则低0.95%。不能把交错顺序当成独立、稳定的修复。

C8 grouped1e-4/interleaved1e-4的case2019分别贡献105.74%/99.29%的C8净mixed改善。其余七个context的mixed变化分别为+1.57%/-0.25%，motor为+9.21%/+6.03%。这解释了为什么较漂亮的完整C8拟合数字仍不等于更好的跨context表现；这些敏感性统计不替换完整C8主指标。

## 3. C80与现有H、C8-R8的直接比较

| Heldout metric | H | 原C8-R8 | C80 | C80 vs H | C80 vs C8-R8 |
|---|---:|---:|---:|---:|---:|
| First10 motor RMSE | 0.04260456 | 0.04458512 | 0.04244386 | -0.38% | -4.80% |
| Future latent NMSE | 0.03838062 | 0.04034388 | 0.03434905 | -10.50% | -14.86% |
| Mixed loss | 0.00347421 | 0.00374074 | 0.00328802 | -5.36% | -12.10% |
| Full32 motor RMSE | 0.04500986 | 0.04693537 | 0.04477837 | -0.51% | -4.60% |
| Late22 motor RMSE | 0.04576608 | 0.04770224 | 0.04547724 | -0.63% | -4.66% |
| First10 gripper RMSE | 0.03015418 | 0.03452837 | 0.03771624 | +25.08% | +9.23% |

C80相对H，motor改善18/40、变差22/40；latent和mixed均改善21/40、变差19/40。motor逐context差值的中位数为+0.00033477，而平均值为-0.00016070，故平均motor优势并不代表多数context改善。相对原C8-R8，motor改善28/40，mixed改善24/40，说明修复原R8退化的证据比“稳定胜过H”的证据更强。

### 四个suite仍有差异

| Suite，C80 vs H | Motor | Latent | Mixed | Gripper |
|---|---:|---:|---:|---:|
| libero_spatial | +3.79% | -16.12% | -4.82% | +45.87% |
| libero_object | -0.46% | +13.99% | +6.48% | +17.32% |
| libero_goal | -2.29% | -2.09% | -3.86% | +29.59% |
| libero_10 | -2.28% | -36.40% | -18.61% | +16.21% |

motor有3/4个suite改善，latent/mixed也有3/4改善；object latent/mixed和spatial motor仍回退。Gripper四suite全部回退，逐context为37/40回退，不能用平均motor的微小改善掩盖这项代价。当前loss仅直接监督六维motor，没有gripper项，这是明确的objective覆盖缺口，但尚未验证增加gripper监督能修复多少。

### 泛化判断如何更新

C80在原C8覆盖task的新demo上，motor/latent/mixed为-0.71%/-17.10%/-9.32%；原C8未覆盖的32个task上为-0.29%/-8.45%/-4.27%。改善并不限于某一个分组；两组在C80训练时均已有task覆盖。

与此同时，C80在旧C8端点上motor为+3.03%、latent为-4.92%、mixed仅-1.32%，拟合旧C8明显弱于原C8-R8，heldout却更好。这支持“把少数C8 context拟合得更好不等于更好的跨demo表现”。它不证明原退化全部由过拟合造成，因为LR、梯度平均方式以及数值轨迹也改变了。

即使对比同LR的C8臂，C80的heldout motor仍低3.36%～4.27%，latent低8.41%～10.44%，mixed低7.41%～9.32%。因此，小LR本身不能解释本次C80相对这些C8端点的全部观测差异；覆盖与梯度平均仍需作为组合因素理解。

完整C80最终端点为motor=0.04463564、latent=0.03837086、mixed=0.00366653。本轮没有同一完整C80 cohort的H/R8端点，因此不据这些绝对数值宣称完整C80上的相对拟合收益。C80在旧C8上净mixed收益很小，case2019贡献比例达413.49%，分母很小且其余case净回退，不能解释为收益更稳健。

## 4. 已发现的训练数值问题与尚未定位的原因

最关键的限制是：两条LR1e-4臂的第一个context都为case2000，raw R=0、fresh Adam、相同seed，首步七项已记录前向metrics与H source replay完全相同；但global gradient norm为0.00826246664与0.00826714467，相差0.0566%。首步Adam之后，同case latent NMSE却为0.0208138432与0.0317301117，相差52.45%。

这是相同已记录前缀之后出现数值分化的直接证据。相同前向标量不等于完整tensor逐元素相同；当前记录没有逐元素gradient和首步raw参数delta，不能认定差异仅发生在backward，也不能定位为某个具体算子错误。详见[顺序/LR诊断](../optimization/25905200.pbs101/INTERPRETATION.zh.md)。

只读检查确认每arm重置R/RNG并新建Adam，每query刷新Cayley cache；实际sampler使用固定seed、CPU随机数并恢复临时patches。关闭TF32不等于启用全部deterministic algorithms。CUDA非确定性、BF16数值敏感性和共享Runtime隐藏状态仍是候选，没有证据确认stale Cayley cache。

全部梯度连通/finite/nonzero检查通过。C80最大pre-clip norm=0.02932，C8新臂最大为0.19022，均未触发阈值1的裁剪。C80的8个batch有4个在本batch更新后mixed loss反而升高；这说明当前更新也没有稳定降低真实量化端点loss，尚不能在STE、Adam动量、量化分箱变化和数值因素之间作唯一归因。

因此，顺序×LR只能作本次recipe的描述性比较；C80的-0.38% motor优势尤其不宜在没有训练重复性证据时称为稳定胜过H。所有工程门限通过也不等于已经排除了全模型BF16舍入与训练数值问题。

## 5. 后续优先级

1. **先定位首步可复现性。** 在后续单独协议中固定一个context和初始化，核对完整输出、两路gradient、Adam参数delta及相同checkpoint的重复推理，再决定需要确定性设置、独立Runtime还是算子级处理。当前没有追加这些forward或训练。
2. **为gripper设显式目标或回归门限。** 当前motor+latent loss的收益不能替代完整action质量；不能直接把此C80 checkpoint视为H的全面替代。是否改loss应在下次实验前声明。
3. **保留C80+平均梯度作为有希望的recipe。** 相比继续把C8 loss压得更低，本次方向更值得验证；但不把覆盖数量、batch平均和LR的作用混成单因素结论，也不宣称已经选定部署臂。
4. **随后再评估R+VQ联合。** 本轮无VQ，已说明R的训练与目标覆盖本身仍有问题；不能从W16A4结果推出联合训练一定胜过H+VQ。后续若进入VQ，应保留H+VQ与冻结R+VQ对照，并使用独立验收条件。

这些是下一轮建议，没有因此增加实验、调整本轮loss或启动新的VQ训练。

## 6. 执行与验收

| Lane | PBS job | Queue | 实际单GPU allocation walltime | 状态 |
|---|---|---|---|---|
| Path | 25905198.pbs101 | g1 | 01:00:59 | F / Exit0 / verified |
| Optimization | 25905200.pbs101 | g1 | 01:57:51 | F / Exit0 / verified |
| C80 | 25905203.pbs101 | g1 | 02:42:36 | F / Exit0 / verified |

三项合计5小时41分26秒单GPU allocation。每15秒记录GPU utilization与显存。模型、tensor和所有forward/backward均在真实PBS compute allocation内完成；login只做轻量控制、小文件/receipt读取。各臂达到完整预声明分母，614 target Linears、最终全部冻结、source replay和formal inference parity均为0、最大orthogonality误差低于5e-5。三个作业均已完成，20分钟subagent监控已停止。
