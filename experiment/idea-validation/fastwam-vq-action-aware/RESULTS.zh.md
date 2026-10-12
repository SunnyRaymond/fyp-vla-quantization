# Fast-WAM 敏感度加权 VQ 实验结果与后续验证

这轮完整实验支持保留 row-scale VQ 作为后续初始化方案。相对于原始 VQ，它在新观察上的前 10 步 motor action RMSE 下降约 20%，22 个 case 中有 19 个改善。但局部输出拟合没有继续降低 test 平均误差，当前 action-sensitive 拟合进一步恶化了 test 误差。最关键的机制发现是：训练用的 action projection loss 几乎归零，真实 action 误差却仍然很大。因此，下一步首先需要验证敏感度 proxy 的可靠性，再决定如何修改 codebook。

**实验和数据。** GPU 作业 `25743693.pbs101` 成功完成，PBS 为 F、Exit_status 为 0，walltime 为 9 小时 6 分 54 秒。614 个目标 Linear 全部拟合。412 次查询包括 40 次 calibration、64 次 selection、308 次 test；保存的 action 和 query 覆盖一致。CPU 汇总 `25748947.pbs101` 也成功完成，从保存的 action 向量独立复算了指标。第一次 CPU 汇总仅在 Markdown 格式化时因字段名错误失败，原始数据与失败作业快照保留。

Calibration 使用 8 个旧输入，每个 1 个 sampler seed。Selection 使用 4 个熟悉的 original tasks 的新状态，每个 2 个 seeds。Test 使用 10 个 original 输入和 12 个新 Plus variants，每个 2 个 seeds。两个 seeds 先在 case 内平均，再对 case 等权平均。Plus variants 可以共享底层任务，44 个 contexts 不能当作 44 个独立场景；本报告使用描述性的配对比较。

所有量化版本固定复用 Smooth alpha 为 0.5、signed Hadamard block 为 128、scale bound 为 16 的既定变换。没有根据本轮结果重新选择 Smooth 强度。A4 使用动态 per-row scale；KV、bias、norm、Text encoder 和 VAE 等保持既定精度。新 VQ 表示为 4 维向量、两个各含 256 项的 BF16 additive codebooks、两个 uint8 indices，加 BF16 per-output-row scale。局部和 action-sensitive 版本各自从同一 row-scale bank 出发，保持相同编码容量。

**Held-out action 误差。** 主指标是相同 observation、相同 sampler seed 下，相对于原始 BF16 action 的前 10 步、6 个 motor 维度的 RMSE，越低越好。它不是成功率下降比例。

| 版本 | Selection motor RMSE | Test motor RMSE | Test gripper RMSE | Test 32 步 motor RMSE |
|---|---:|---:|---:|---:|
| BF16 weights + A4 | 0.03910 | 0.04856 | 0.01546 | 0.05961 |
| Scalar W4A4 | 0.04465 | 0.05499 | 0.01251 | 0.05934 |
| 原始 VQ + A4 | 0.06352 | 0.07131 | 0.04050 | 0.07463 |
| Row-scale VQ + A4 | 0.04740 | 0.05705 | 0.03180 | 0.05971 |
| 局部输出拟合 VQ + A4 | 0.04265 | 0.05726 | 0.01811 | 0.06683 |
| Action-sensitive VQ + A4，lambda 为 0.1 | 0.04117 | 0.06103 | 0.02126 | 0.06660 |

BF16 reference 的所有相对误差均为 0，故未重复列入表中。Lambda 候选为 0.1 和 1；selection 分别为 0.04117 和 0.04445，所以冻结 0.1 后才进入 test。Test 前查询数为 0，没有在 test 上重新挑选 lambda。Lambda 1 没有进行 test，不能判断它在 test 上的结果。

Row-scale 配置相对原始 VQ 改善 19.995%，19/22 cases 更好；但仍未超过 Scalar，只有 10/22 cases 比 Scalar 好。相对于同一 row-scale 初始化，局部输出拟合的 test 主指标增加 0.357%，10/22 cases 改善。Action-sensitive 相对局部输出拟合增加 6.592%，相对 row-scale 增加 6.972%，相对 Scalar 增加 10.975%，只有 7/22 cases 比 Scalar 好。

这些对比不能把 row-scale 配置的全部收益归因于 scale 单独一个因素：新 bank 还重新计算了 actual A4 trajectory 的 input moments，并重新初始化和拟合。局部拟合与 action-sensitive 的增量比较则使用相同初始化、格式和容量。

局部输出拟合对 gripper 有改善，但没有改善主 motor 指标，32 步 motor 误差还增加了。不能用其中一个次级指标的收益替代主指标的结果。

**变化在哪些输入上发生。**

| Test 输入组 | Cases | Scalar | 原始 VQ | Row-scale | 局部输出拟合 | Action-sensitive |
|---|---:|---:|---:|---:|---:|---:|
| Original | 10 | 0.04566 | 0.05737 | 0.04771 | 0.04812 | 0.05166 |
| Plus 全部 | 12 | 0.06277 | 0.08293 | 0.06484 | 0.06487 | 0.06884 |
| Plus camera | 2 | 0.07235 | 0.07554 | 0.05950 | 0.05312 | 0.08062 |
| Plus robot | 2 | 0.09834 | 0.14865 | 0.11858 | 0.10709 | 0.11697 |
| Plus layout | 2 | 0.04135 | 0.05186 | 0.04609 | 0.04415 | 0.04930 |
| Plus language | 2 | 0.04845 | 0.06654 | 0.04421 | 0.04377 | 0.04225 |
| Plus light | 2 | 0.04457 | 0.07815 | 0.04485 | 0.05569 | 0.05325 |
| Plus background | 2 | 0.07154 | 0.07684 | 0.07581 | 0.08537 | 0.07062 |

Action-sensitive 在 Original 和 Plus 两个大组上都更差，所以不能只归咎于 Plus 扰动。Camera 是相对局部拟合恶化较明显的一组；language 和 background 则有收益。每个 Plus 维度只有 2 个 cases，这些分组用于定位问题，不能当作整个维度的稳定性能结论。Selection 只包含 Original，不能充分代表 Plus；但 Original test 同样恶化，selection 分布不匹配也不能单独解释全部现象。

**优化确实工作了，但训练指标不能预测真实 action 改善。** 下表把 row-scale 的拟合指标归一化为 1。局部损失是逐层在采样 teacher-path 输入上计算的输出 MSE，再按层各自的初始损失归一化，最后对 614 层平均；它不是整个模型的输出 MSE。

| 拟合版本 | 平均归一化局部 MSE | 全局 sampled projection loss 相对 row-scale | Calibration 实际 motor RMSE | Calibration 平均 proxy RMSE |
|---|---:|---:|---:|---:|
| Row-scale | 1.00000 | 1.00000 | 0.05215 | 0.27232 |
| 局部输出拟合 | 0.83211 | 0.77944 | 0.04671 | 0.23510 |
| Action-sensitive lambda 为 0.1 | 0.83873 | 0.000048014 | 0.04851 | 0.00192 |
| Action-sensitive lambda 为 1 | 0.84766 | 0.000011589 | 0.05145 | 0.00096 |

局部拟合的 614/614 层局部 MSE 都下降，所有拟合版本的 614 个已接受 coordinate objectives 均未增加。当前负结果不能简单解释为没有优化到目标。

Action-sensitive lambda 为 0.1 的全局 proxy loss 已降至初始值的约 0.0048%，真实 calibration action RMSE 却仍为 0.04851，甚至比局部输出拟合的 0.04671 更高。Lambda 为 1 时 proxy 更低，真实误差又更高。失配在训练输入上就存在，不只是 held-out 泛化问题。

Lambda 为 0.1 时，5 个局部 MSE 增加的模块全部位于 Video expert 的 text/time conditioning 路径：text_embedding.0、text_embedding.2、time_embedding.0、time_embedding.2 和 time_projection.1。Lambda 为 1 时，time_projection.1 和 time_embedding.2 的局部误差分别达到初始的约 2.13 和 1.80 倍。这是敏感度目标发生局部取舍的位置，尚不是这些层导致最终 test 恶化的因果证明。

**为什么少量投影可以变得很好，而动作仍然不准确。** 当前拟合使用每个 calibration context 的 4 个固定 Rademacher action probes，每次 Linear 调用采样 2 个 token rows，并在同一 context 内把跨调用、跨层的一阶贡献相加后平方。

对一个 60 维 motor action 误差，4 个混合投影至多直接约束 4 个独立方向。即使四个投影都为零，其他方向仍然可能存在误差。例如只观测向量两个坐标的和，误差 `[1, -1]` 的观测值是 0，向量误差却没有消失。这里的 4 个 probes 是随机混合方向，并不是只观察了 4 个 action 坐标。

全局目标可写作：

\[
L_{\rm proxy}=\mathbb E_{c,r}\left[\left(\sum_l\langle g_{c,r,l},E_{c,l}\rangle\right)^2\right].
\]

其中 c 是 calibration context，r 是 action probe，l 是共享 Linear 的调用，g 是 BF16 teacher 的 action adjoint，E 是采样的 quantized Linear 输出误差。跨调用先求和对于共享权重和真实一阶传播是必要的，但有限 probes 和 token rows 下，codebook 也可能学到只对这些样本成立的抵消。

此外，一阶近似建立在 BF16 teacher trajectory 上，实际学生经过 614 层量化后，后续 activation、attention 和 denoising trajectory 都可能变化。当前结果已直接证明 proxy 与实际效果失配；固定方向过拟合、采样导致的抵消以及一阶近似失效，是需要进一步区分的解释。

**接下来怎样建立更可靠的敏感方向保护。** 先保留现有 banks，做诊断，不先增加 lambda 或改变 Smooth 强度。

1. 用未参与拟合的新 action probes 评估同一批已冻结 banks，并把 probe seed 与 token-row sampling seed 分开。先固定 rows 只换 probes，再固定 probes 只换 rows，避免把两种误差来源混在一次变化里。
2. 检查小扰动到完整量化的曲线。在全部目标 Linear 上，用 BF16 输出和 quantized 输出按 t 插值，t 从小值逐步到 1；同 observation、同 sampler seed 下比较真实 action 和一阶预测。若只在小 t 对齐，后续需要限制拟合的扰动范围或更新 linearization point。
3. 上述检查通过后，再考虑方向 metric。对局部输出方向 u，可定义 \(s_l(u)=\mathbb E\|D^{1/2}J_lu\|^2=u^TG_lu\)，其中 J_l 把局部误差传播到最终 action，D 定义 action 坐标的重要性，\(G_l=\mathbb E[J_l^TDJ_l]\)。Codebook 使用该 metric 衡量重建误差，使敏感方向的同等误差付出更高代价。这里比较的是单位方向和相同误差幅度；若两方向的平方 action 损失比为 100，可解释为这一 metric 下的敏感度比为 100。
4. 对少量样本估计的 G 使用局部重建项作为约束，并检查独立方向上的效果。可加入局部误差上限或 trust region，尤其避免 conditioning 模块为降低全局 proxy 而承担过大的局部误差。具体阈值应由 calibration/selection 决定，不能在已看过的 test 上重新挑选。

这些是后续待验证的设计，不是本轮已经有效的算法，也不构成 novelty 结论。推广到其他 WAM 时，应先验证该 baseline 的最终 action 输出和梯度路径，再重复同样的 proxy 可靠性检查；本轮尚无其他 baseline 的数值结果。

**存储和部署边界。** 三个新 VQ 版本的编码容量相同，含 books、indices、row scales、tails 和 transform 的 target-weight payload 都是 3,023,856,089 bytes，effective bpw 为 4.020873。各 bank 的序列化模块文件大小为 3,025,501,840 bytes；上述 payload 已包含 9,034,315 bytes 的 transform tensors，不应再次相加；序列化大小是另一个存储度量。Scalar 的对应 bpw 为 4.137026，原始 VQ 为 4.015358，所以与 Scalar 的比较接近相同存储预算，但不是精确相等。

推理时使用 decoded BF16 reference，没有 packed VQ compute kernel。编码文件变小不等于实际推理显存或延迟按同样比例下降。这是 fixed-input action fidelity 实验，执行 episodes 为 0，不能推导闭环成功率或 native latency。Smooth 强度保持固定，结果也不能回答更激进的 Smooth 是否更好。

完整数值见 [CPU 汇总报告](retrieved/results/analysis/RESULTS.zh.md)、[配对指标](retrieved/results/analysis/summary.json) 和 [拟合及 proxy 诊断](retrieved/results/analysis/diagnostics.json)。作业证据见 [Full terminal receipt](full_terminal_evidence.json) 和 [Analysis terminal receipt](analysis_terminal_evidence.json)。
