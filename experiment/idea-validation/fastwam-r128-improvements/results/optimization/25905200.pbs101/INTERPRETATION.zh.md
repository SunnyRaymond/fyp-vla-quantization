# 顺序与 LR 对照：结果解读

作业25905200.pbs101已F/Exit0，walltime 01:57:51；父代理验收通过。三条新臂各完成8次单context Adam更新、8次训练曝光、完整C8 8/8与heldout80/80，614 Linears；source H重放与formal inference parity均为0，最大orthogonality误差5.37e-6，最终参数全部冻结，新增teacher为0。

固定输入W16A4、无VQ；heldout为40 contexts×2 seeds，先seed→context→suite平均，再四suite等权。误差越低越好。以下描述本次固定update8端点，不作训练随机性、显著性或最佳配置选择结论。

| Arm | C8 motor vs H | C8 latent vs H | C8 mixed vs H | Heldout motor vs H | Heldout latent vs H | Heldout mixed vs H |
|---|---:|---:|---:|---:|---:|---:|
| 原grouped 1e-3，C8-R8 | -8.26% | -12.09% | -17.98% | +4.65% | +5.12% | +7.67% |
| grouped 1e-4 | -0.41% | -31.74% | -18.22% | +4.07% | -0.07% | +4.37% |
| interleaved 1e-3 | +21.88% | -8.67% | +20.39% | +12.55% | -0.95% | +13.30% |
| interleaved 1e-4 | -5.54% | -30.57% | -23.42% | +3.08% | -2.29% | +2.21% |

## 结果支持什么

1. **小LR缓解部分退化，但没有让motor超过H。** grouped降LR后heldout motor比原R8低0.55%，interleaved降LR后低8.41%；两个小LR臂的latent都接近或优于H，motor与mixed仍高于H。
2. **交错顺序不是普遍的修复。** LR1e-3时，interleaved相对原grouped的heldout motor高7.55%；LR1e-4时，interleaved相对grouped低0.95%。观察到明显的顺序/LR交互，不能将交错本身解释为稳定收益。
3. **C8拟合收益仍集中于case2019。** grouped1e-4与interleaved1e-4中，case2019分别贡献105.74%与99.29%的净mixed改善；其余七个context的mixed变化分别为+1.57%与-0.25%，motor分别为+9.21%与+6.03%。这七个context仅是诊断，不替换完整C8主指标。
4. **不是只有未覆盖task出现问题。** interleaved1e-4在C8覆盖task的新demo上，motor/latent/mixed分别为+7.31%/+18.74%/+17.57%；未覆盖task上分别为+2.03%/-8.83%/-2.03%。C8顺序和LR调整仍未解决跨context表现。
5. **没有触发梯度裁剪，梯度连通检查也通过。** 三臂最大pre-clip norm分别为0.1902、0.0485、0.0722，均小于阈值1；逐context两路R梯度均connected/finite/nonzero。不能把退化解释为clip大量截断或R完全没有收到梯度。

## 首步数值差异限制了因果解释

grouped1e-4与interleaved1e-4都从raw R=0、fresh Adam、相同seed和case2000开始。首步七项已记录前向metric逐键完全相同，H+U=I replay误差均为0，但：

| 首步记录 | grouped1e-4 | interleaved1e-4 |
|---|---:|---:|
| Pre-update mixed loss | 0.00257624686 | 0.00257624686 |
| Global gradient norm | 0.00826246664 | 0.00826714467 |
| Action gradient abs_sum | 0.283085674 | 0.282947630 |
| Video gradient abs_sum | 0.796890855 | 0.797422171 |
| Post-update latent NMSE | 0.0208138432 | 0.0317301117 |
| Post-update mixed loss | 0.00276651746 | 0.00322053256 |

Global norm仅相差0.0566%，首步后的latent却相差52.45%。这是同一已记录前缀后出现数值分化的直接证据。JSON没有保存首步完整输出tensor、逐元素gradient或参数delta，故不能证明差异仅发生在backward，也不能定位具体算子。

只读代码检查确认：每个arm重置raw R和RNG并新建Adam；每次query清Cayley cache并预计算matrix；实际执行的sampler固定seed、CPU采样随机数，并在finally恢复临时patches。worker关闭TF32，但没有显式启用deterministic algorithms。CUDA非确定性、BF16数值敏感性或共享Runtime的隐藏状态仍是未证实候选，尚无证据确认stale Cayley cache或某个具体实现错误。实际sampler小型源码存于source_diagnostic_refs/sampler.py。

因此，这个2×2应解释为本次实现下的observed recipe contrasts；最终差异不能单独归因于样本顺序或LR。没有为此追加forward、重复训练、修改当前C80配置或扩展实验。

C80已按预先冻结的80个唯一context、8次10-context平均梯度更新、LR1e-4完成。它是整体recipe与原H/C8-R8的比较，不是等处理量或单因素calibration数量对照。整体判断见[最终分析](../../final/ANALYSIS.zh.md)。
