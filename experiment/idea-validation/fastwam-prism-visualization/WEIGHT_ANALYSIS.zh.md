# Fast-WAM weight 机制分析

本分析已完成，复用原 Prism visualization 的 checkpoint、量化 bank、frozen transforms 和已经保存的 BF16 paired inputs。没有训练、重新拟合或构建整模型，新增整模型 forward 为 0。

从[中文图集](retrieved/results/full/weight_analysis/plots/gallery.html)开始查看：先看 Micro 相对 Row 的汇总，再展开各层的 weight 分布、B32 block 误差和固定输入输出误差。图集含 80 组 figure、112 页预览；[数值结果](retrieved/results/full/weight_analysis/weight_analysis_data.json)和[图表清单](retrieved/results/full/weight_analysis/plots/manifest.json)也可直接查看。

范围固定为两个 expert 的 `self_attn.q` / `ffn.2`，block 0、10、20、29，共 16 个 Linear。输入采用 cases 1004 / 1014、seed index 0、各自 denoising stage 的 step 0 / 4 / 9，每个 module/case/stage/step 选最低 call index，共 96 个 primary inputs。三个方法是 learned SpinQuant-adapted、Row-scale VQ、Microscaling VQ，共 48 个 weight comparisons 和 288 个 fixed-input output comparisons。原 conditioning/prefill 数据仍保留，但不计入这 96 个主输入。

这是配对的描述性机制分析：比较单位是同一 Linear 和相同输入上下文。层、token、channel 及重复 step 之间存在依赖，不把矩阵元素作为独立重复样本，不据此计算显著性或对未知输入人群作推断。主对比是 Micro/Row 的 weight 与 output NMSE 差异，SpinQuant 用来判断不同 frozen basis / quantization recipe 的分布和误差；跨方法存储格式与开销不同，误差排名不代表等 bit-budget 优势。

## 三组比较

1. **Weight 分布**：从源 checkpoint 读取选定矩阵，转换到原模型实际使用的 BF16 精度，再分别应用 frozen transform。统计覆盖完整矩阵的 input/output-channel RMS、CV 和幅值分布。3D 图采用共同的前 128 output channels × 512 input channels 窗口，明确区分 Raw、同基底 FP reference 和 quantized/decoded effective weight。
2. **B32 block 量化误差**：每个方法将 effective weight 与自己的同基底 reference 比较；B32 为共同的诊断分组，SpinQuant 的实际 quantization group 仍为 G128，VQ 的 vector size 仍为 4。B32 NMSE 是该 block 的 squared error sum 除以 reference energy；全矩阵统计与显示窗口分别标注。
3. **固定输入的输出误差**：从已保存的 `paired_raw` 输入在 FP32 中重建各自 transformed basis，计算 bias-free Linear 矩阵乘法。保持输入不做 A4，独立观察 weight 的作用。误差为 `X_basis @ (W_quantized - W_reference).T`，NMSE 以对应 `X_basis @ W_reference.T` 的总能量归一化。统计使用完整 input/output width 和已保存 token rows；Video 图采用前 64 tokens × 128 output channels，Action 实际只有 32 个已保存 token，图使用全部 32 tokens × 128 output channels。Weight 的两条坐标是 output/input channel，只有输入与输出图具有 token 坐标。

BF16 source windows 与已有 Raw weight capture 应一致；各方法 decode/reference windows 与已有 capture 核对。另核对 transformed FP reference 输出与 Raw basis 输出的数值等价性。验证只涉及离线矩阵运算，不能当作新整模型推理或 native kernel 验证。

## 执行与结果

所有 checkpoint 加载、tensor 运算、解码和绘图均放在真实 CPU-only PBS allocation 内。单次上限两小时、soft deadline 6600 秒；按完成的 module 保存 checkpoint，续跑使用同一结果目录。Login node 只进行 PBS 查询及小型控制文件操作，不加载或传输大型 checkpoint/NPZ，不进行 hash 或数值计算。

结果目录：`/scratch/users/ntu/yguo017/fastwam-prism-visualization-20261012/results/full/weight_analysis/`。完整矩阵来源、signed 派生数组、PNG/PDF 留在 compute-accessible 存储；本机图集使用计算节点生成的小型 JSON/JPEG。

数值分析和最终绘图均已完成。计算节点生成的 `weight_analysis_data.json` 为 177027 bytes，16 个 module、48 组 weight、96 份 primary inputs、288 组 output comparisons 的覆盖全部通过；实际 weight 与旧 capture 窗口逐元素一致，reference 窗口在容差内，全部 FP32 basis-equivalence 检查通过。最终 job `25774454.pbs101` 的 PBS 状态为 F、Exit_status=0，wrapper exit=0，存在 PIPELINE_COMPLETE；用时 00:08:42。

最终 manifest 为 complete，80/80 组 figure、112/112 页预览齐全。已修正 3D reference 叠加遮挡 effective weight 的问题，distribution/block_error 的 32 组修订图均为 render_version=2；两种 expert 的 3D weight、RMS、B32 heatmap 和输出误差图均已目视检查。对比图使用共同的高度或颜色范围，Row-scale 向量按 output channel 展示。112 张小型 JPEG 已取回，本机中文图集的图片和链接均可定位。任务监控已停用。

## 已确认的数值观察

Micro 相对 Row 的百分比均为 `(Row NMSE - Micro NMSE) / Row NMSE`，正数表示 Micro 误差更小。Weight 比较每组有四个不同层；output 比较为四层 × 两输入 × 三 steps，共 24 个配对上下文，不能当成 24 个独立场景。

| Expert / 位置 | Micro weight 更准的层数 | Weight NMSE 改善中位数 | Micro output 更准的上下文 | Output NMSE 改善中位数 |
|---|---:|---:|---:|---:|
| Video / self_attn.q | 3/4 | +0.153% | 12/24 | +0.099% |
| Video / ffn.2 | 3/4 | +0.623% | 15/24 | +0.571% |
| Action / self_attn.q | 2/4 | +0.238% | 11/24 | −0.082% |
| Action / ffn.2 | 3/4 | +0.515% | 12/24 | −0.039% |

Micro 的完整 weight 重建通常有小幅改善，但收益没有稳定传到固定输入的 Linear 输出。Video FFN 的结果较一致，Action 两个位置的 output 改善中位数接近零且略为负值；不同层与不同 step 的方向会改变。

主展示位置为 block 10、ffn.2，输出上下文 case 1004 / step 4：

| Expert | 方法 | 完整 weight NMSE | 固定输入、完整 output width 的 NMSE |
|---|---|---:|---:|
| Video | SpinQuant | 0.013731 | 0.006965 |
| Video | Row-scale VQ | 0.012818 | 0.006425 |
| Video | Microscaling VQ | 0.012725 | 0.006370 |
| Action | SpinQuant | 0.013682 | 0.002510 |
| Action | Row-scale VQ | 0.019019 | 0.002464 |
| Action | Microscaling VQ | 0.018729 | 0.002493 |

该 Action 层中，Micro weight NMSE 相对 Row 下降 1.523%，但 output NMSE 上升 1.161%。因此，重建误差的大小和它落在哪些输入方向上，是两个不同问题。

完整矩阵的 channel RMS 也说明 rotation 的作用方向：Video 此层的 input-channel RMS CV 从 Raw 0.08818 降到 learned SpinQuant FP reference 0.01849，Action 从 0.13838 降到 0.03789；output-channel RMS CV 分别保持约 0.18893 与 0.21089。Micro 的 input-channel CV 没有继续压低：Video Row 0.02204、Micro 0.02214，Action Row 0.04746、Micro 0.04786。更平坦的表面不能代替同基底的重建误差与固定输入输出误差。

## B32 block：典型误差与尾部误差分开看

以下同样使用 block 10 / ffn.2 的完整矩阵。每个诊断 block 是一个 output channel 上连续 32 个 input channels；Video 共 1376256 个 block，Action 共 131072 个 block。P95 表示 95% block 的 NMSE 不超过该数值，最大值用于定位少数热点；它们与按完整 reference energy 归一化的整体 weight NMSE 是不同统计量。

| Expert | 方法 | Block NMSE 中位数 | P95 | 最大值 |
|---|---|---:|---:|---:|
| Video | SpinQuant | 0.013230 | 0.025545 | 0.108394 |
| Video | Row-scale VQ | 0.011264 | 0.022006 | 0.131716 |
| Video | Microscaling VQ | 0.011172 | 0.022250 | 0.140340 |
| Action | SpinQuant | 0.013182 | 0.025532 | 0.091405 |
| Action | Row-scale VQ | 0.016157 | 0.035327 | 0.216292 |
| Action | Microscaling VQ | 0.015999 | 0.035517 | 0.208943 |

Micro 在这两个层的典型 block 误差都略低，但 P95 都略高；Video 的最差 block 也更差，Action 的最大值则下降。误差热点位置也发生变化。因此 microscaling 的收益更接近局部误差重新分配，现有结果不支持“所有 block 都更好”或“尾部被统一消除”。

机制上的解释是：rotation 主要让输入方向的能量更均匀，block scaling 则改变局部量化的分辨率；最终输出还取决于已有输入是否激发这些误差方向。上述 Action 层 weight 改善、output 变差就是这一差别的具体例子。下一步研究可优先关注输出敏感的误差热点及其在不同 step 的变化；本轮没有据此追加实验。

Weight 和固定输入局部输出指标不能单独证明 action RMSE 或 closed-loop success 的改善；本分析使用 bias-free FP32 矩阵乘法、不做 A4，将 native SpinQuant bank 解码后计算，VQ 使用 effective decoded BF16 weight，不测 native kernel 性能。两份 frozen VQ banks 的 codebooks 和 scaling recipe 都可能不同，以上比较没有单独隔离 B32 scaling 的因果贡献。
