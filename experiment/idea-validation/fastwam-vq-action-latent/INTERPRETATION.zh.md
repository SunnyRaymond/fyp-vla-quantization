# Fast-WAM action + latent-aware VQ：第一步验证的解释

## 结论

在相同 calibration、selection、test contexts 下，给原 first10 motor action distillation 增加一个小权重的 BF16 future latent preservation loss，改善了完整 22-case test 的平均 action 与 future fidelity。这支持继续研究 **action-aware + latent-aware**，但还不足以支持用 latent-aware 替代 action-aware，也不能宣称全部动作维度、horizons 或 case 都改善。

GPU 实验 `25798439.pbs101` 与 CPU 汇总 `25799647.pbs101` 均已 F/Exit0，并通过完成凭证与覆盖检查。原固定监控已按用户要求删除。

## 这轮到底改变了什么

从原 `row_output_vq_a4` parent 重新开始，训练同一组 614 个 Linear 的 VQ codebooks 与 row scales，indices 与其余量化配置固定。训练是原 first10×6 motor MSE 加上 `w × future latent NMSE`，并未把 teacher future 注入 student 输入；student 始终使用自己生成的 future。

Future loss 监督最终 denoising 的 clean video latent，排除被 observation clamp 的 frame 0，包含两个 future latent frames。它比较量化模型与 BF16 模型想象的未来，不比较真实未来视频。NMSE 用 teacher latent 能量归一化；temporal-delta NMSE 则比较相邻 latent 的变化，并用 teacher 变化的能量归一化。

`w=0.0029918822946277566` 在训练前只用 8 个 calibration contexts 冻结，使初始化 calibration 均值上 latent 项的 loss 数值约为 action 项的 10%。这个比例不代表参数梯度贡献为 10%，也没有利用 test 调整权重。

其余条件沿用旧 action-only 实验：相同 8 个 calibration cases、4 个 selection cases×2 seeds、22 个 test cases×2 seeds，2 epochs/16 updates，相同 optimizer schedule 与采样设置。每个 case 先平均两个 seed 的指标，再对 22 个 case 等权平均。16 次更新的 614 个 target Linears 均有 connected、finite、nonzero gradients。

## 完整 test 的主要结果

所有误差越低越好，参考都是同 context、同 seed 的 BF16 输出。

| 指标 | Parent row-output VQ | 旧 final-action distilled | 新 action + latent |
|---|---:|---:|---:|
| First10 motor RMSE | 0.057256 | 0.058110 | **0.052951** |
| Full32 motor RMSE | 0.066828 | 0.069572 | **0.061332** |
| Late22 motor RMSE | 0.069300 | 0.073342 | **0.063866** |
| First10 gripper RMSE | 0.018110 | 0.018110 | 0.018826 |
| Future latent NMSE | 0.084435 | 0.093011 | **0.071787** |
| Temporal-delta NMSE | 0.153072 | 0.168856 | **0.133841** |

Calibration 上 action 与 future 也都改善。Future NMSE 从 parent 的 0.074410、action-only 的 0.082009 降到 0.059389；test 同方向下降，收益没有仅停留在训练 contexts。但 test cohort 此前已被观察过，因此这仍是已知 cohort 上的 recovery diagnostic。

First10 motor 是本轮直接监督与选择 checkpoint 的指标，新版本比此前几个 weight-quantized 配置都低，包括 Scalar W4A4 的 0.054994、row-scale VQ 的 0.057052、action-sensitive VQ 的 0.061030 和 original VQ 的 0.071311。它仍高于 BF16 weights + A4 的 0.048559。

收益也有明确边界：Scalar W4A4 的 full32/late22 为 0.059341/0.060688，row-scale VQ 为 0.059709/0.060271，都低于新版本。因此“first10 进步”不能写成“所有 action horizons 最优”。本轮没有直接监督 gripper，其 test 平均误差略升，也应保留在最终结果中。

完整 7 个 source arms、旧 action-only 与新版本的比较见 [原始汇总报告](retrieved/results/analysis-25799647.pbs101/REPORT.zh.md)。

## 平均收益是否普遍

| 指标 | 相对 parent 改善 cases /22 | 相对 action-only 改善 cases /22 |
|---|---:|---:|
| First10 motor | 15 | 12 |
| Full32 motor | 14 | 13 |
| Late22 motor | 13 | 10 |
| Future latent NMSE | 17 | 16 |
| Temporal-delta NMSE | 15 | 15 |

相对 parent，future 改善覆盖较多 case；Original 的 10 个 cases 的 future NMSE 全部下降，Plus 为 7/12。相对 action-only，late22 的平均值虽然更好，但改善数只有 10/22，median paired delta 为正。也就是说，部分较大的修复抵消了更多较小的退化，不能把均值改善解释为多数 cases 的 late22 都改善。

Case 1016 是明显的恢复例子：action-only 的 first10 motor RMSE 为 0.169303，新版本为 0.088431；future NMSE 从 0.387126 降到 0.103047。新版本也低于该 case 的 parent。这与“保护想象的未来有助于防止严重 action drift”的解释一致，但单个 case 的共同改善不能证明因果中介。

Case 1022 则是反例：相对 action-only，first10 motor 从 0.043425 升到 0.075457，future NMSE 从 0.033564 升到 0.129422。Case 1021 相对 parent 的 first10 motor 下降了 0.005132，而 future NMSE 却上升了 0.039648。这表明 future 与 action 误差的变化并不总是同步。

### 尾部敏感性：避免把大幅修复误读成全面改善

额外 CPU 后处理 `25799776.pbs101` 已完整通过；只读取现有 `analysis.json`，没有新增 model forwards。以下临时省略 case 的视图都是 **post hoc 敏感性分析**，不替换完整 22-case 主结果，也未用于训练、权重调整或 checkpoint selection。百分比为同一视图内新旧均值的相对变化，负值表示误差下降。

| 对比 | 完整 22-case 相对变化 | 敏感性视图 | 该视图相对变化 |
|---|---:|---|---:|
| Full32 vs action-only | -11.84% | 省略 1016，21 cases | **+4.23%** |
| Late22 vs action-only | -12.92% | 省略 1016，21 cases | **+6.15%** |
| Late22 vs parent | -7.84% | 省略 1015，21 cases | **+1.62%** |
| First10 vs action-only | -8.88% | 省略 1016、1025，20 cases | **+1.90%** |
| First10 vs parent | -7.52% | 省略 1016、1025，20 cases | **-4.17%** |
| Future NMSE vs parent | -14.98% | 省略 1016、1025，20 cases | **-10.92%** |
| Future NMSE vs action-only | -22.82% | 省略 1016、1025，20 cases | **-8.66%** |

这使结论更精确：相对 parent 的 first10 与 future 改善在上述省略视图中仍然保留；相对 action-only 的 action 均值优势则更依赖对严重退化的修复。尤其不能把 full32/late22 的平均下降解释为普遍提高远期 action fidelity。

Action 与 future 的共同改善有一定覆盖：相对 parent 为 14/22 双改善、1/22 仅 action 改善、3/22 仅 future 改善、4/22 双退化；相对 action-only 为 12/22 双改善、4/22 仅 future 改善、6/22 双退化。它们支持继续研究两者的耦合，但不能证明 future 改善是 action 改善的唯一原因。

完整敏感性表见 [tail_sensitivity.zh.md](retrieved/results/tail-25799776.pbs101/tail_sensitivity.zh.md)。

![逐 case 的 action 与 future 误差变化；负值为改善](<D:/Downloads/Final Year Project/experiment/idea-validation/fastwam-vq-action-latent/retrieved/results/tail-25799776.pbs101/paired_recovery_case_ids.png>)

## 对 latent-aware idea 最有帮助的两个观察

**收益更多出现在较远的 future latent。** 相对 parent，第一 future latent 的 NMSE 从 0.061944 到 0.061869，变化很小；第二个从 0.103767 到 0.080341，改善明显。较远的 temporal-delta NMSE 也从 0.229776 降到 0.190173。当前小权重约束主要修复了较远 future 的偏移，而非平均地改善每个时间位置。这提示后续可以分析 temporal weighting 或 motion-aware latent preservation，但本轮没有训练这些新目标，也不能声称它们已经有效。

**更好的 future 指标不保证更好的 action checkpoint。** Selection set 上，epoch 1 的 future NMSE 为 0.032728，比 epoch 2 的 0.033895 更低；但 first10 motor RMSE 为 0.055259，明显差于 epoch 2 的 0.042003。最终按 action 指标选择 epoch 2。只按 latent NMSE 选 checkpoint，会在这轮选出 action 更差的版本。

因此本轮更合适的解释是：latent preservation 对共享生成路径增加了辅助约束，使 codebooks 在拟合 calibration actions 时也保留 BF16 future 的结构。这是一个与结果一致的机制假设；实验没有分离 video-path 与 action-path 的贡献，也没有证明 action 改善全部由 future 改善中介。

## 下一步优先级

先保留 action + latent 这个组合，重点解释 case 1017/1019/1021/1022 等退化，以及 late22 与 gripper 的取舍；不要仅凭平均 future NMSE 改用 latent-only 训练或选择模型。较远 latent 与 motion preservation 是有数据线索的下一步，但 future 改善相对较广、远期 action 改善仍对尾部 case 敏感，说明当前还缺少对 action horizon 的稳定约束。Temporal weighting、弱 late22 action loss 等改动应分别测试，避免一次加入多个新项后无法判断收益来源。

这些结论来自 Fast-WAM 当前实现。推广到 Cosmos Policy、LingBot-VA 或其他联合生成结构，仍需要各自模型上的验证。本轮为 decoded-BF16 VQ/fake A4、BF16 KV 的 fixed-input 输出保真度实验；没有 closed-loop episodes、native kernel 性能或真实未来准确度证据。
