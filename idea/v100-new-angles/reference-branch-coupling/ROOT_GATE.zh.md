# Reference-branch coupling：root design与prior边界

2026-09-13；GPU结果产生前记录。

本次C02干预是同一个visual encoder的跨分支map assignment；C04把propagated current error、goal reference error、residual fidelity与scalar objective fidelity分开。Shared/Different完整3×3保证两侧边际精确一致，不要求不同draw各自误差一样。主张是当前checkpoint的有限诊断，不是新SR算法。

原goal-coordinate gate中的“predictor不等变所以不能识别交互”论证过强：使用实际输出Δy后，残差误差可以直接分解。这个纠正不等于旧四臂已经证明common-mode机制；新protocol另外引入共享map/不同map的受控干预，并以相对FP的两项误差联合为目标。原先只比较quantized objective变小的草案不采用。

这也不同于已完成screen的简单seed变更：Antithetic研究两个predictor score平均，Persistence研究同一action流的积分累积，Broadcast研究同一activation在patch场内的扰动聚合；本次研究encoder跨两条不对称分支后在reference subtraction处的误差抵消。共享相关性预期的方向可以与前两项相反。它们仍属correlated-rounding同一方法家族，不能把每个application算成独立原创quantizer。

## Primary grounding

- [DINO-WM](https://arxiv.org/abs/2411.04983)：current/goal共享visual encoder、预测latent与goal比较是原模型结构。源码固定commit 0a9492fa12044b852ae9e001cc74604b79c8bb0c。
- [QuantWM](https://arxiv.org/abs/2602.02110)：已覆盖DINO-WM encoder/predictor PTQ不对称与planning-objective失配；本次不重提module-sensitive mixed-bit方法。
- [Where Bits Matter](https://arxiv.org/abs/2602.11882)：已研究paired-goal mixed-bit与planner budget；arXiv v1标注workshop submission，不据此声称正式接收。
- [Correlated Quantization](https://proceedings.mlr.press/v162/suresh22a.html)：相关/分层randomness是已有quantization方法；其distributed mean目标不同于这里的reference subtraction，但宽方法新颖性不成立。
- [Siamese object tracking PTQ](https://pmc.ncbi.nlm.nih.gov/articles/PMC10970761/)：共享backbone或双输入网络的PTQ本身也不新；该文采用Vitis-AI固定rounding/PTQ，没有为本次三draw匹配比较提供结论。

定向检索覆盖world model goal encoder PTQ、shared/common randomness distance和Siamese PTQ；不是穷尽novelty search。`generic_method_novelty_no_go / narrow_application_unverified`先于实验固定。可以做一次资源很小的受控screen，但不因数值go升级成论文贡献或追加完整验证。

## 执行前接受条件

以PROTOCOL.zh.md为唯一科学规则，独立CONDITIONAL_GATE复核，再核对实际48个encoder Linear、数据与source身份、quantizer及输入/恢复receipt接口。所有输出仅在实际CCDS allocation产生。未通过执行前接口审查不提交；不把当前编写代码当作已运行结果。
