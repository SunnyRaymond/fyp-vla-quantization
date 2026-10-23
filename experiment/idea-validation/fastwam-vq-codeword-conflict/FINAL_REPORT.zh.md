# Codeword sharing 与 latent protection：完整诊断结论

2026-10-17；protocol `fastwam-vq-codeword-conflict-v1`。本轮诊断已完成，未实现新的 assignment 算法。

**结论：存在小步解除共享约束的正面信号，但没有得到“当前 assignment 产生异常梯度冲突，并稳定阻碍 latent protection”的充分证据。暂不直接推进新的 conflict-aware assignment。** 0.005 扰动下，untied 的 held-out future 和固定 reader action 均优于 shared；0.015 下则明显变差，未通过事先规定的双步长推进条件。下一步应先区分更新幅度、模块间能量分配与共享自由度的作用，再决定是否改变 assignment。

## 1. 实际比较了什么

在 action-only checkpoint A 上，仅用原 8 个 calibration contexts 决定诊断位置及更新方向。先覆盖全部 614 个目标 Linear、全部 occupied codewords，再按预定规则选出 time_embedding.0、text_embedding.0、blocks.24.ffn.0、blocks.17.ffn.0；每个模块选择 8 个 book0 codewords。

完整成员检查覆盖 32 codewords × 8 contexts，共 256 条记录；所有成员梯度之和均与实际 autograd codeword gradient 一致。选中 support 含 1,118,810 个 weight vectors、4,475,240 个有效 weight coordinates；零 scale 成员为 0。

有限干预的三个方向是：

- **Shared**：indices、book1、scales 固定，将 calibration latent 梯度投影到选中 book0 可实现的 decoded-weight 子空间，再实际更新 BF16 codebook。
- **Untied**：相同 support 上独立更新 decoded weights，方向来自相同 calibration latent 梯度。它增加了自由度，是容量放开的诊断，不是同 bits 可部署模型。
- **Random**：相同 support 上的固定随机方向对照；只有一个预先固定的随机方向，不是随机方向的统计分布。

三者匹配实际 decoded-BF16 扰动的总范数，epsilon 定义为 `||delta W_support|| / ||W_support||`，不是全模型相对范数，也不是 optimizer learning rate。两个目标值为 0.005、0.015。实际范数匹配最大相对误差约 0.126%，低于预定 5% 容差；四个分片使用同一份预先生成的更新。

完整有限评估包含 8 calibration contexts、4 selection cases × 2 seeds、22 held-out test cases × 2 seeds，共 60 contexts；6 个条件共 360 行。每行包含 native generation 和固定 BF16 teacher-reader future replay，共 720 次模型查询。先在 case 内平均两个 seeds，再对 cases 等权平均；test 分母始终为 22 cases，而非 44 个独立样本。

Future NMSE 衡量与 BF16 teacher future 的偏差。Teacher motor RMSE 衡量固定原始 BF16 reader、只替换生成 future 后，相对于 BF16 reference action 的前 10 步、6 个 motor dimensions 的偏差。原始 video prepare/prefill、ActionDiT、proprio encoder 全部固定为 BF16；因此这条 action 差异来自注入的 future。两者都是 teacher fidelity，不是真实未来状态或任务成功率。

## 2. 有限 forward 的主要结果

以下均为 held-out 22 cases、配对 seeds 平均后的结果；两项指标均越低越好。

| 相对扰动范数 | 条件 | Future NMSE | 固定 BF16 reader first10 motor RMSE |
|---|---|---:|---:|
| 0 | 原 action-only A | 0.0930110 | 0.0412432 |
| 0.005 | Shared | 0.0832547 | 0.0391120 |
| 0.005 | Untied | **0.0730161** | **0.0333009** |
| 0.005 | Random | 0.0822009 | 0.0390221 |
| 0.015 | Shared | **0.0777560** | **0.0359131** |
| 0.015 | Untied | 0.1218054 | 0.0590210 |
| 0.015 | Random | 0.0863554 | 0.0414430 |

**0.005 的信号值得保留。** Untied 相比 shared：future 平均差为 -0.0102386，中位差为 -0.00903509，14/22 cases 改善；motor 平均差为 -0.00581106，中位差为 -0.00358046，16/22 改善。12/22 cases 两项同时改善，2 个仅 future 改善，4 个仅 motor 改善，4 个两项均变差。它支持“放开这个受限共享子空间，可以找到在部分 held-out 输入上更好的小步修正”。

与 random 相比，小步 untied 的 future 平均差为 -0.00918483，但仅 11/22 cases 改善，中位差为 -0.000128708。这一对照的优势在 cases 间不均匀，不能只突出均值。小步 random 本身也改善了 test 均值，因此“任何改动后均值更好”不足以证明 latent 梯度方向独特有效。

**0.015 的反向结果同样明确。** Untied 相比 shared：future 平均差 +0.0440494，motor 平均差 +0.0231078；两项各仅 2/22 cases 改善，19/22 两项同时变差。Calibration future 也从 shared 的 0.0767219 变为 untied 的 0.1055960。因此不能把它只解释为 held-out 过拟合。当前两个有限步长并不显示“解除共享越多越好”。

Selection 组也未呈现一致收益：0.005 下 untied 的 future 均值优于 shared，但 motor 为 0.0198723，高于 shared 的 0.0192720，也高于原 A 的 0.0162318。该组只有 4 cases，不能作为统计结论，但不应忽略其与 test 均值不同的方向。

事先规定的条件要求两个步长都满足：calibration/test future 均值改善、至少 12/22 test future 改善、固定 reader motor 均值不变差、test future 均值优于 random。**0.005 全部通过；0.015 全部未通过；整体推进条件为 false。** 这是工程决策条件，不是显著性检验。不能看完 test 后只保留小步长并宣称原假设已证实。

## 3. 梯度抵消能解释多少

全部 614 modules 的抽样分析只使用 calibration，Parent/A/Z 各 8 contexts。每个 occupied codeword 最多固定抽样 32 个成员，先累加整个 sampler 内重复调用的成员梯度，再计算组内指标。

| Checkpoint | 抽样 cancellation | 随机分组 cancellation | 局部 action/latent 方向冲突比例 |
|---|---:|---:|---:|
| Parent | 0.774074 | 0.774109 | 24.40% |
| Action-only A | 0.774829 | 0.774572 | 36.47% |
| Action+latent Z | 0.774358 | 0.774221 | 28.66% |

Cancellation 定义为 `1 - ||sum_i u_i|| / sum_i ||u_i||`，其中 `u_i = stored_scale_i * decoded_weight_gradient_i`。三个 checkpoint 都接近随机分组对照，未发现当前 assignment 在该汇总指标上存在异常抵消。数值 0.774 不是性能损失比例；近一半成员的一阶项方向相反，也不等于观察到近一半独立样本受害。

A 的局部 action/latent 冲突比例较高，但精确汇总 codebook 梯度后，三个 checkpoint 的 action 梯度步对 latent 的一阶方向导数仍为负。当前 raw gradient 诊断不支持“整体 action 更新必然损伤 latent”的断言，也不能复原历史 Adam、scale 更新和训练轨迹的因果关系。

全成员投影能量很低，但每个模块仅放开 8 个 4D codewords，即 32 个共享方向。相对于数万到数百万个独立 weight coordinates，低投影能量有很强的维数因素。四个模块的实际保留能量与 isotropic `rank(P)/D` 参考之比分别为 2.206、1.207、1.156、0.564；并非全部低于这一参考。这个参考只解释维数限制，不是替代 assignment 的实测对照，也不是显著性检验。

## 4. 为什么现在不直接改 assignment

现有诊断仍有两个未拆开的因素：

1. **模块间更新分配发生变化。** 匹配的是全 support 总 norm。Shared 的扰动平方能量约 99.5–99.6% 在 time_embedding；untied 约三分之二在这里，其余进入 text/FFN。小步收益可能同时包含方向、自由度和跨模块分配变化，不能单独归因于 codeword 成员的角度冲突。
2. **一阶方向不能保证有限 BF16 forward。** Untied 两个步长的一阶 latent 预测均为下降，实际较大步长却连 calibration 都变差。非线性、A4/BF16 离散变化及 sampler 的影响尚未分离；两个步长不足以确定是哪一种机制。

此外，当前只检查 book0，book1/scales 固定。放开独立权重优于这个局部子空间，并不证明两本 codebook/scales 的正常联合优化无法修复，更不证明重新分配同样 bits 的 indices 必然有效。

文献也限制了 novelty：[VQ4DiT v1 §3.2、§4](https://arxiv.org/html/2408.17131v1) 已讨论共享 codeword 的成员梯度不一致，并通过候选 soft assignments 与 codebook 联合优化进行校准。因此，“发现梯度冲突 + 优化 assignment”本身不能作为我们的新意。其余文献和候选方法见 [README.zh.md](README.zh.md)。

## 5. 下一步的具体决定

**保留 assignment 作为候选，但先做最小的归因实验，不直接实现一套新算法。** 下一轮优先处理：同一份 latent 信息，哪些修正能够通过当前 VQ 参数稳定实现，并保留固定下游 reader 的可读性。

建议按以下顺序推进，均属于下一轮计划，本轮没有执行：

1. **固定每个模块的实际 decoded 更新 norm，再比较 shared/untied/random。** 保持已有 support 和 book1/scales 配置，用 calibration 决定模块预算，先去除模块间能量重分配这个混杂。若差距明显缩小，应优先研究 latent-aware update allocation，而不是 assignment。
2. **用 calibration 的实际有限 forward 约束更新幅度。** 将一阶梯度仅用于给出候选方向，步长由 calibration 的 future 及固定 reader fidelity 决定；记录实际部署 BF16/A4 下的更新效果。若差距主要被步长控制消除，应优先改优化过程。当前已读取 test 的 22 cases 后，只能继续用于内部开发；最终泛化声明需要新的独立测试。
3. **若差距仍稳定存在，先增加“同 indices、两本 books/scales 可动”的容量对照。** 只有这一对照仍不足时，才更有理由实现 latent 功能兼容性的 assignment，并与相同 latent loss 的普通 assignment 优化比较。格式、calibration、候选数和训练预算一致；如改变资源分配，匹配 codebooks/indices/scales/metadata 的实际总 bytes。

可以继续争取的贡献是具体机制及对应方法：哪些 action 恢复未约束的 future drift 会损伤固定 reader，以及如何在既定量化表示下稳定保护这些方向。当前结果尚未证明 VQ 必须优于 scalar，也尚未证明一个新的 assignment 方法有效。普通 trust region、gradient-aware grouping 或增加 latent loss 本身都不能自动构成新意。

## 完成记录与证据边界

- Capture：24/24 checkpoint-context records，614 Linears；exact：8/8 contexts、256/256 全成员加和检查通过。
- Finite：完整 60 contexts、360/360 conditions、720 次查询；四个 GPU 分片的范数记录一致，固定 BF16 reader 和 future injection 检查通过。汇总 JSON/CSV 未发现 NaN/Infinity。
- 所有成功 GPU 作业 walltime 均小于 2h；最长 capture 为 1:26:43，最长有限干预为 0:36:29。2h 指单个 GPU 作业上限；整个多阶段流程另有排队及串行分析时间。
- 最终 CPU `analyze`：`25846654.pbs101`，F/Exit0、matching summary、PIPELINE_COMPLETE，walltime 0:04:46。监控在本轮完成后停止。
- 本轮有 24 capture + 8 exact 成功梯度查询，720 有限干预查询；另有一次修复前失败的 capture 查询。这里的 query 不是单次 Linear/DiT forward 调用数，backward checkpoint recomputation 也不应混为额外独立样本。
- 没有新增 assignment 训练、closed-loop episodes 或 native low-bit kernel benchmark。Untied 不能部署为当前压缩格式；action 指标仅是固定输入上的 teacher fidelity。

原始完整表：[有限干预报告](retrieved/analysis/REPORT.zh.md)、[analysis.json](retrieved/analysis/analysis.json)、[360 行配对记录](retrieved/analysis/paired_contexts.csv)。梯度阶段解释：[抽样分析](SCREEN_INTERPRETATION.zh.md)、[全成员分析](EXACT_INTERPRETATION.zh.md)。
