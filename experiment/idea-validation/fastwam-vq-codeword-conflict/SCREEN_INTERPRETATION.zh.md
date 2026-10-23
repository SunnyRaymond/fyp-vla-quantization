# Codeword conflict：第一阶段解释

状态：24/24 条 capture 已完成，CPU 分析 `25845607.pbs101` 已验证。下表只覆盖 8 个 calibration contexts、seed0；不是 held-out 或 closed-loop 结果。

| Checkpoint | Future NMSE | 抽样组内抵消 C | 随机分组抵消 C | Action/latent 方向相反的 codeword 比例 |
|---|---:|---:|---:|---:|
| Parent | 0.07440975 | 0.77407376 | 0.77410879 | 24.40% |
| Action-only A | 0.08200883 | 0.77482930 | 0.77457203 | 36.47% |
| Action+latent Z | 0.05938866 | 0.77435822 | 0.77422103 | 28.66% |

## 目前能说明什么

组内抵消定义为 `C = 1 - ||sum_i u_i|| / sum_i ||u_i||`，其中 `u_i = stored_row_scale_i * decoded_weight_gradient_i`。先把每个 weight vector 在全部 sampler 调用中的梯度相加，再做分组。每个 occupied codeword 最多固定抽样 32 个成员；这些成员统计不等于全体成员比例。共享 codeword 的总梯度另用全体成员精确计算。

三个 checkpoint 的全局抵消都约为 0.774，与五次随机分组对照几乎相同。因此，高抵消本身没有显示当前 VQ assignment 存在特别异常的结构。`C=0.774` 也不表示丢失了 77.4% 的模型性能。

A 的局部 action/latent 方向冲突比例高于 Parent，Z 较低。但在这三个 checkpoint，精确 action gradient 汇总后的 latent 一阶方向导数仍为负：

| Checkpoint | 每个 context 的方向导数均值 | 先跨 calibration 平均梯度后的方向导数 |
|---|---:|---:|
| Parent | -0.00035147 | -0.00005687 |
| A | -0.00012130 | -0.00001444 |
| Z | -0.00016798 | -0.00001860 |

这里考察的是 codebook 参数、scale 固定时的 `-dot(g_action, g_latent)`；负号表示无穷小 action 梯度步预测降低 latent loss。它不是实际 BF16 有限更新，更不能解释历史训练轨迹。因此，“部分 codewords 方向相反”目前不等于“整体 action 更新损害 latent”。

各成员的梯度项也不是互相独立的 latent losses。所谓 opposed member fraction，只表示该梯度项对共享更新的一阶贡献方向相反，不是观察到了该成员对应的实际任务损害。

## 接下来验证什么

固定使用 A 的全部 8 个 calibration contexts，已按预先规定的 energy/cancellation 规则选择四个模块、每个模块 8 个 book0 codewords：

- `video_expert.time_embedding.0`
- `video_expert.text_embedding.0`
- `video_expert.blocks.24.ffn.0`
- `video_expert.blocks.17.ffn.0`

选择结果和全部 32 个 codeword IDs 见 [selection.json](retrieved/analysis_capture/selection.json)。选择未使用 test 数据。部分选中分组的抵消明显低于随机对照；它们被选中是因为预先规定的 energy/cancellation 总分，而不是因为它们比随机分组差。

下一步先检查这 32 个 codewords 的全部成员，在全体梯度之和与真实 autograd codeword gradient 一致后，构造同 support、实际 decoded-BF16 norm 匹配的 shared / untied / random 扰动。随后覆盖原定全部 60 contexts，并固定 BF16 reader 测量 future 改动的下游影响。

这个有限扰动实验固定 book1 和 scales；untied 增加自由度。即使 untied 更好，也只能先支持局部容量约束的诊断，不能直接证明整个 VQ 格式无法修复、当前 assignment 比其他同 bits assignment 差，或新方法已经有效。零 scale 行的梯度能量会单独报告，防止把解除零 scale 限制的收益误归于去除 codeword 共享。扰动 norm 匹配发生在全部选中 support 上，未逐模块匹配；CPU 报告另记录每个模块的扰动能量及一阶预测贡献，帮助区分方向变化和模块间能量重分配。

当前决定：继续完整的精确及有限扰动诊断，暂不根据抽样抵消大小选择 assignment 算法。

原始汇总见 [analysis.json](retrieved/analysis_capture/analysis.json)，分 case 数据见 [paired_cases.csv](retrieved/analysis_capture/paired_cases.csv)。
