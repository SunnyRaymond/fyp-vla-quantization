# Semantic Input Scales — mechanism no-go

2026-09-13。最终结论：**mechanism_no_go**。GPU **64738** 在 TC1N04 V100 完成，elapsed **35s**、exit 0:0；CPU **64740** 完成原始score/action独立复算。至此停止，不改partition、CAL规模、bitwidth或gate。

## 最小结果

固定epoch65 DINO-WM Wall、predictor-only W4；CAL 108–111、DEV 112–117。4个CAL episodes共享FP32 channel absmax统计，6个DEV episodes每组64个H5 candidates。按原始terminal score取top6，primary仅为first-action fidelity proxy。

| 输入activation scheme | first-action MSE ↓ | 全H5 elite-mean MSE ↓ | FP-score regret ↓ |
|---|---:|---:|---:|
| W4A32 | 0.06082061 | 0.07505019 | 0.01898216 |
| FullInput-A8（1 scale） | 0.06102718 | 0.06813129 | 0.01900731 |
| Semantic-A8（3 scales） | 0.06082061 | 0.07505019 | 0.01898216 |
| Permuted-A8（同3 scales、同组尺寸） | 0.06102718 | 0.06813129 | 0.01900731 |
| PerChannel-A8（404 scales） | 0.04818120 | 0.06438185 | 0.01383643 |

Semantic对FullInput和Permuted的primary改善均仅 **0.3385%**，远低于冻结5%门槛。独立FP64复算的严格改善为 **1/6 episodes**，低于4/6门槛；全H5 action MSE反而更高。这个接口上的3个语义scale没有表现出值得继续推进的增量。普通PerChannel背景更好，不能把它的收益归因于本候选的3组机制。

## 工程与数值边界

实际 `num_hist=1`、`concat_dim=1`、384+10+10接口、source/checkpoint identity、no-op exact scores/top6/action shape和hook移除检查通过。observer只使用CAL，DEV没有重新估计scale。所有模型/统计工作均在真实compute allocation。

GPU汇总原先报告2/6改善；CPU用FP64重算为1/6。原因是GPU汇总中的FP32 action-mean归约对同一elite集合的排列产生极小舍入差异，在1e-12 strict比较下多计了一个改善。**这是一处指标报告的数值实现问题，不能挽救该方法**：mean指标在容差内一致，5%和4/6两个gate在两种计算下均失败，no-go结论一致。采用CPU的1/6作为最终数学指标；保留原始GPU汇总，不重跑模型。

CPU仅从raw arrays复算，不独立重做模型、observer或weights；这些工程证据来自GPU记录。该screen没有环境success、native A8/W4加速、真实内存节省或统计显著性结论。一个固定permutation只支持本对照，不代表所有随机分组。

## 保留证据

- [协议](../../../../idea/v100-new-angles/semantic-input-scales/IDEA_AND_PROTOCOL.zh.md)、[独立审查](INDEPENDENT_AUDIT.zh.md)、[GPU汇总](artifacts/64738/summary.json)、[CPU复算](artifacts/64740/verification.json)。
- 完整 `raw_scores_actions.npz`、`scale_definitions.json`、目标fingerprints、运行脚本和协议快照留在 `/tc1home/UG/yguo017/v100_newangles_ccds/artifacts/64738`。
- Raw SHA256：`d2ac9ee246c868fe2af44c14bff048811b099654001cc6ae2020fd42eab9d079`。

不继续pipeline改方法或完整验证；所有negative与对照结果保留。
