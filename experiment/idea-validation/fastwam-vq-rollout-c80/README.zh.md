# 原 C80 与 rollout-aware C80 等预算对照

比较对象为原 C80 joint final80 和新的 rollout-aware joint final80。原结果直接复用，不重训对照。两组都从原 C8 rho1.5 update8、ratio=1 开始，训练80个 contexts、80 updates，保持原 Adam、两组学习率、首次update后清空optimizer state、gradient clip、rho=1.5及 normalization_c。不增加 channel scaling 或 late22 loss，不按held-out选择checkpoint。

每个基础任务的两个原calibration contexts中，偶数case保留为anchor，奇数case替换成固定C8 student从原demo snapshot出发、执行最多10个动作后到达的观测；若提前terminal则保留当时观测，不丢弃或重抽。共40 anchors +40 visited states，四suite每组各20。保留原任务、相机、prompt、environment seed与训练sampler seed，没有同时引入camera扰动。它检验的是局部student可达状态覆盖，不等同于完整长期rollout数据聚合。

HDF5 state/XML读取在CPU allocation内进行。GPU准备先核对恢复起点的camera/proprio，再由冻结C8 collector采集；另起独立BF16进程生成40个新teacher标签，40个anchor teacher复用旧缓存。新旧teacher都绑定完整plan row和sampler seed。所有采样规则预先固定，不使用held-out状态或闭环失败筛选。

训练后评估原40 held-out contexts ×2 seeds，共80 paired queries，复用原输入和BF16 teacher；按case内平均seed、suite内平均case、四suite等权聚合。报告first10 motor RMSE和future latent NMSE，并保留full32/late22/gripper诊断，不由固定输入结果推断闭环成功率。

每个GPU作业walltime为1:59，软截止6840秒，保留完成的训练updates和评估queries供精确续跑；不缩小原80分母。采集/teacher成本与80-update训练成本分别报告，因此“等预算”指训练contexts和updates相同，并非把rollout采集称为免费。GPU利用率和显存每15秒写入job log；gpt-6-luna/xhigh subagent每10分钟检查当前权威handles，状态未实质变化时保持安静。

当前协议见protocol.json，当前作业见handles.json，监控见monitor_state.json，结果完成后见results/RESULTS.zh.md。Login node仅处理提交、qstat和小控制文件；所有数值工作、模拟器和数据读取均在真实PBS allocation内执行。

## 已完成：原 held-out 的两项指标未超过原 C80 joint

| 配置 | First10 motor RMSE ↓ | Future latent NMSE ↓ |
|---|---:|---:|
| 原 baseline | 0.044296774 | 0.060877371 |
| C80 clip_only80 | 0.043848437 | 0.054848124 |
| 原 C80 joint80 | 0.041846837 | 0.048973336 |
| Rollout-aware C80 joint80 | 0.042925590 | 0.053785088 |

新方案相对原 C80 joint，motor +2.578%、latent +9.825%；case 内先平均两个 seeds 后，分别为17/40、14/40改善。相对 clip_only 两项仍降低2.105%/1.938%，相对 baseline 降低3.095%/11.650%。原 C80 joint 继续保留为当前固定输入数值参考，不能把新方案称为比原 joint 更好的替代。

变化在各 suite 间不一致：Object 的两项均改善，Spatial 和 Goal 两项均退步，Libero_10 motor接近、latent增加。Gripper RMSE相对原joint降低20.715%，只作为诊断；late22/full32也未用于改变原两项主判断。本次没有新的闭环结果，因此既不能说rollout calibration已解决闭环退步，也不能由这组原分布RMSE否定它对闭环的潜在作用。

原80个held-out queries、seeds和teacher不变。新增训练固定80 contexts/80 updates，只有40个奇数case的观测及其teacher标签改变；没有channel scaling、late22 loss或新的development set。完整七指标及配对计数见[结果表](results/RESULTS.zh.md)，原始精度见[comparison.json](results/comparison.json)。

## 如何解释与继续

原C80 joint在calibration和原held-out均有收益，但Plus闭环为202/280、clip_only为207/280；paired9 wins/14 losses，净差由camera的0 wins/5 losses解释。已有camera初始查询的平均、中位数和P90 motor误差仍改善，5个丢失成功中3个初始RMSE更低。因此，优先检查闭环状态覆盖与控制敏感误差；现有日志只能确认timeout，未证明具体抓取/释放原因。详细已有证据见[闭环分析](../fastwam-vq-c80-joint-rmse/results/ANALYSIS.zh.md)。

本轮只检验了一个action chunk之后的局部visited states，使用原相机，没有camera augmentation，也没有长期恢复状态聚合。50%替换同时减少了原demo观测的训练覆盖。下一步更有信息量的calibration检查是：用两组现成final80 checkpoint，在同一40 anchors与40 visited states上交叉评估，复用现成80份teacher、不重训，分开报告两类状态。若visited误差降低而anchor误差增加，提示分布取舍；若visited也没有收益，应先检查优化与目标匹配。这是calibration内机制诊断，不是新的独立泛化测试。尚未提交该评估。

在确认数据取舍前，暂不同时扩大样本总数、训练updates和activation自由度。后续可以单独检验较低替换比例；扩大训练数据时应明确计算成本与等updates对照。闭环收益仍需原paired闭环cohort验证。

Activation方面，已有coupled B32相对clip_only的motor/latent仅改善1.0345%/1.4237%，且没有闭环证据；这项实验同时改变了输入backward。更直接的一组对照是固定D=1、只训练ratios，同时保留完整dynamic-absmax input coupling，以原C80/80updates匹配已有coupled B32结果。先区分梯度规则与scales的收益，再考虑受约束B32 scales与weights的分阶段联合训练。暂不直接释放约32倍的per-channel scales及全部codebooks。后续同时记录activation、weight-VQ和Linear输出误差，不能仅凭activation重建误差推进。

这些方向是下一步建议，本轮没有自动追加GPU训练或闭环。

## 完成与限制

CPU sources `25862581.pbs101`用时00:05:43；GPU prepare `25862620.pbs101`用时00:27:20；GPU train/eval `25862999.pbs101`用时01:07:44。GPU实际累计01:35:04，小于2小时。三项均通过PBS F/exit0、artifact exit0、PHASE_COMPLETE和matching complete summary。新teacher标签40、collector queries40，原teacher锚点40复用；训练80次、正式新held-out80 queries、复用对照240 records。每次更新的weight/clip均有614个模块梯度连接、有限、非零；初始reference和最终runtime/sampler action parity均为0。

GPU utilization/VRAM每15秒写job log；gpt-6-luna/xhigh每10分钟监控至完成后停止。长progress JSON会挤出短tail中的GPU行，后续新鲜采样已确认遥测正常。未用短tail估计全程峰值。完成证据见[completion_evidence.json](completion_evidence.json)。

本轮复用了历史原C80训练对照。相同前四个anchor中，第一步forward完全一致，但首次gradient norm有约0.3%差异，后续数值前缀分叉。代码有CUDA index_add_的已知非确定性机制，另有parity warmup与LIBERO环境入口差异；没有隔离各因素的实际贡献。不能把这次差异全部因果归于calibration数据。详见[训练方差说明](results/training_variance_note.json)。这是每配方一次训练、反复使用固定cohort的工程比较，未新增闭环、native kernel或统计显著性证据。
