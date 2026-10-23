# C8 frozen-reader KL：本轮结论

2026-10-22。本轮已完整完成，固定 gate 未通过，停止这版配方；不运行 reader MSE ablation、不扫参数、不进入280闭环。原 C8 A+Z、rho1.5 baseline继续保留。

主评测是实际量化 student 在原40 cases ×2 sampler seeds上的结果。先平均case内两个seeds，再平均suite内case，最后四suite等权。Baseline的88条记录、C8 teacher和heldout teacher全部复用。

| Heldout指标 | 原A+Z | Frozen-reader KL | 相对变化 | 改善/变差cases |
|---|---:|---:|---:|---:|
| First10 motor RMSE | 0.0442967744 | 0.0451499174 | +1.93% | 12/28 |
| Full32 motor RMSE | 0.0475009795 | 0.0495732677 | +4.36% | 17/23 |
| Late22 motor RMSE | 0.0485285040 | 0.0508427202 | +4.77% | 15/25 |
| Future latent NMSE | 0.0608773710 | 0.0573564747 | -5.78% | 22/18 |
| First10 gripper RMSE | 0.0570540305 | 0.0408736746 | -28.36% | 25/15 |

Motor配对中位数为+0.0012749056，两个sampler seeds分别汇总均退步。四个suite有三个motor均值退步；四个suite的motor配对中位数均为正。object suite虽均值略改善，也仅2/10 cases改善。主指标失效不是只因一个suite或一个sampler seed。

Calibration motor RMSE从0.0421332419降至0.0387150622（-8.11%，6/8改善）；future NMSE从0.0544809815降至0.0454062065（-16.66%，6/8改善）。因此本次有较明确的calibration拟合收益，未转移为heldout motor收益；heldout latent保真也确有改善，但不足以保证实际动作更接近teacher。Gripper的连续输出有改善，但大部分平均收益来自case3029（0.6283368170→0.0109467842）；它不是离散开合时刻或闭环收益。

固定gate四项为motor均值下降、case配对中位数下降、至少21/40改善、future NMSE均值不退步。本轮前三项均未通过，最后一项通过。按预先规则停止，不因其他指标有利而进入ablation。

这不是明显梯度断开或初始化/重放错误：8步均614/614目标Linears connected/finite/nonzero；每步staged student重放的action/future最大差均0；teacher-reader重放、原C8初始化和最终native inference parity均0；首个参数更新确实改变部署输出。beta=0.005622540816813253，由C8初始化数据一次性标定，未用heldout选择参数或checkpoint。

上述检查不证明训练已充分收敛，也不能确定calibration大小、泛化、有限步优化各自的因果份额。没有对最终全部C8额外重放reader loss，因此不声称该辅助KL已整体优化充分。结论限定为这套冻结C8/8-update/frozen-reader soft-bin KL配方没有带来所需的实际student heldout motor收益，不否定所有reader辅助目标或KL方法。

运行证据：CPU prepare25910452.pbs101（1m19s）、GPU25910456.pbs101（g1，A10040GB，52m52s）、CPU aggregate25910457.pbs101（12s）均F/Exit0、artifact0、PHASE_COMPLETE和完整覆盖通过。GPU利用率与显存每15秒写job.log。新增106个student full queries、18个reader-only queries、8次参数reverse passes、8次reader reverse passes；内部checkpoint recomputation不另计逻辑queries。持久训练更新8，baseline新queries0、teacher新full queries0、闭环0。全部张量/模型计算在真实PBS allocation内。

10分钟只读subagent监控已停止。完整分suite/seed数据见[机器结果](aggregate/summary.json)，逐case配对见[paired_cases.json](aggregate/paired_cases.json)，基础表见[ANALYSIS.zh.md](ANALYSIS.zh.md)，完成核对见[completion_evidence.json](../completion_evidence.json)。
