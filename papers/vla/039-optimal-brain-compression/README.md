# 01 — Optimal Brain Compression

## 基本信息

- 完整标题：*Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning*
- 作者：Elias Frantar、Sidak Pal Singh、Dan Alistarh
- 发表会议：NeurIPS 2022
- 本地来源：[官方会议论文 PDF](paper-neurips-2022.pdf)
- 在本阅读路线中的定位：second-order foundation

## 一句话要点

本文将经典的 Optimal Brain Surgeon 转化为一个高效的 layer-wise framework，用于 one-shot pruning 和 quantization；其核心是衡量每次 weight update 对 reconstruction 的影响，并对这种影响进行补偿。

## 阅读前准备

理解 post-training quantization 与 retraining 的区别，并熟悉写成 `Y = WX` 的 linear layer。

## 问题与目标

给定一个训练完成的 model 和一个小型 calibration set，在不进行 retraining 的情况下压缩 model。对于 weights 为 `W`、calibration inputs 为 `X` 的 layer，局部目标是最小化：

`||WX - W_hat X||²_F`

这是一个 local reconstruction objective，并不等同于直接优化 end-to-end task loss。

## 方法脉络

1. 将 Optimal Brain Surgeon 专门化到 row-wise layer reconstruction problem。
2. 使用来自 input Hessian 的 second-order information，估计 pruning 或 quantizing 某个 weight 所造成的损失。
3. 选定一次 weight update，并用 closed-form correction 补偿其余尚未压缩的 weights。
4. 重组计算，使精确的 OBS-style procedure 只需要约为 layer width 三次方的时间和二次方的内存，避免不切实际的 naïve scaling。
5. 用同一套机制处理 pruning、quantization，以及二者的组合应用。

其中 quantization variant 通常称为 OBQ，更广义的 framework 称为 OBC。

## 重点检查的证据

- Calibration 使用 1,024 个随机 training samples；适用时加入 ImageNet augmentation，并且不进行 retraining。
- Table 4 报告：ResNet-18 的 dense Top-1 accuracy 为 69.76；4-bit OBQ 为 69.56，3-bit 为 68.69，2-bit 为 64.04。
- 对 ResNet-50，相应结果为 dense 76.13，4-bit 75.72，3-bit 75.24，2-bit 70.71。
- 在论文报告的 ResNet settings 下，joint pruning and quantization 达到约 12–14× BOP reduction，同时 relative accuracy loss 约为 2.5%。
- CPU demonstration 报告：在 1% 和 2% accuracy-loss operating points 下，speedup 分别约为 4× 和 5×。

不要把这些 speedup 与后续 LLM token-generation results 直接比较；二者的 models、kernels、hardware 和 workload 均不相同。

## 局限与证据边界

- 实验早于当前的 LLM deployment scale，主要聚焦 ResNet、YOLO 和 BERT-style workloads。
- Objective 是 layer-local；良好的 reconstruction 并不能证明 task performance 达到 global optimum。
- 该 procedure 是 greedy 的，并依赖 calibration inputs。
- 实际 speedup 需要匹配的 sparse/quantized execution support；algorithm 本身并不能保证加速。

## 对你的 FYP 有何意义

这篇论文最适合作为起点，帮助你理解 GPTQ-style methods 为何使用 activation statistics、Hessian inverse 和 error compensation。它也提醒你，“quantization error” 必须针对具体 layer 和 calibration distribution 来定义。

## 阅读路线

### 20 分钟

阅读 Abstract、Section 3、OBQ algorithm、Table 4 和 Conclusion。

### 75 分钟

从 reconstruction objective 出发，推导 weight update 和 compensation rule；随后比较 pruning 与 quantization 如何作为两种约束被纳入同一 framework。

## 阅读问题——阅读后自行回答

1. 为什么 layer-wise objective 使用 `WX`，而不是只计算 `W` 与 `W_hat` 之间的距离？
2. 哪个 approximation 使 Hessian 能够在一个 layer 的不同 rows 之间复用？
3. Compensation update 与简单地 round 一个 weight 后继续处理有何不同？
4. 哪些结果证明 algorithmic accuracy，哪些结果证明真实 execution speed？
5. 从 ResNet 或 BERT 扩展到 70B-parameter Transformer 时，哪些 assumptions 会变得脆弱？

## 讨论速记

- 需要解释的核心论点：高效而精确的 OBS-style updates 能够改善 one-shot pruning 和 quantization。
- 需要复现的公式：layer-wise reconstruction objective，以及一次 compensation update。
- 需要引用的表格：Table 4。
- 应主动说明的局限：local reconstruction 和 compression ratio 并不是 end-to-end deployment evidence。

