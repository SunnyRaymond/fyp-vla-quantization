# Fast-WAM 多 seed 教师数据分析

## 数据范围与完整性

本报告只分析 **Fast-WAM** 原始 BF16 teacher 数据，不是 Cosmos 数据，也没有改变 Cosmos 的训练协议。cohort 覆盖 80 个 case（ID 2000–2079），libero_10、libero_goal、libero_object、libero_spatial 四个 suite 各 20 个。每个 case 有 16 个 seed，共 1,280 条 teacher 记录。seed 0 的 80 条复用既有 teacher 输出；seed 1–15 的 1,200 条由 PBS job `25965429.pbs101` 新采集。输入取原本已归一化的 action 张量 `action[0:10, 0:6]`，即 10×6 共 60 维，不做额外缩放或反归一化；这不是全动作维度之外推，也没有视频或 Cosmos policy 量化结果。

数据摘要确认 cases、seeds 与记录数完整。模型为原 BF16 teacher，参数未参与梯度更新；训练更新数为 0，optimizer step 为 0，held-out 数据未使用。16 次 replay 检查均通过：8 条 cached replay 的 action 与 future 最大绝对差均为 0；8 条 same-seed repeat 的 action 最大绝对差也为 0。Replay 的比较范围与字段以随附 JSON 的原始明细为准。

## 描述性结果

按 summary 的定义，case 内 seed 波动的 normalized within-state 标准差 RMS 为 **0.02358**。80 个 case mean 之间的 normalized 标准差 RMS 为 **0.32473**，约为 case 内数值的 **13.77 倍**。前者概括同一 case 的重复采样差异；后者是 80 个描述性 case 均值之间、逐坐标计算 population standard deviation 后的 RMS。这个比值说明在本 cohort 中 case 间差异较大，但 case 并非从已定义的总体分布随机抽出的统计单位，所以它不能单独证明真实任务的分布稳定或不稳定，也不是置信区间。

分布估计使用 88 个固定投影：60 个 action-coordinate axis、6 个 time mean、6 个 first-half-minus-second-half 与 16 个固定随机投影（RNG seed 20261021）。所有组共用从 C80 full16 数据得到的标准化 ruler，并同时汇总 pooled C80 full16 与 per-case full16 ruler。固定 ruler 便于组间比较，但投影、分箱和带宽均属于本分析设定，数值不能脱离这些设定解释为绝对分布误差。

Pooled 汇总的 K=4 与 K=8 covariance relative Frobenius error 均值分别为 **1.2130** 和 **0.98766**。Effective rank 均值为 **6.2136**，numerical rank 为 **15**；chunk-mean coherence fraction 均值为 **0.65423**，shuffle-null 均值为 **0.31604**。这些数字反映当前投影、汇总规则和固定 cohort 下的描述性结构；没有预设显著性检验，不能从 coherence 与 shuffle-null 的数值差直接宣称显著，也不应将其概括为多模态证据。

这些统计不替代 closed-loop task success、quantization fidelity、latency、memory 或部署测量；本数据链没有测量这些指标。

pooled kernel 标准化带宽 σ=0.25 时，**98.75%** 的投影宽度大于相应 case 的 within-state 标准差。这个比例说明该带宽在多数投影上宽于观测到的 case 内波动。它提示平滑估计可能跨过一些条件差异、掩盖部分条件不确定性，但并不直接证明实际决策模型一定产生这种偏差。`predeclared_stability_criteria` 为 `null`，因此没有预先登记的稳定性门槛可用于判定通过或失败；本文只报告描述性统计。

## 来源与结果文件

Prepare、collection、analysis 的 PBS job 分别为 `25965426.pbs101`、`25965429.pbs101`、`25895961.pbs101`；seed 0 的原始复用来源为 `25895959.pbs101`。实际 replay 文件位于 `/scratch/users/ntu/yguo017/fastwam-teacher-multiseed-20261021/results/replay.json`，collection summary 在同一 scratch 根目录下的 `results/collect/summary.json`，analysis summary 在 `results/analysis/summary.json`。PBS collector 报告 `Stageout_status=1`，但 scratch 中完整 summary、replay 与完成标记均存在。随附分析 JSON 副本为 `retrieved/results/analysis/summary.json`；其数值来自已完成的 analysis job，没有在本地重新计算。
