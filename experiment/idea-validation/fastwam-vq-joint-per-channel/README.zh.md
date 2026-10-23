# Joint VQ + per-channel scaling / clipping

在已完成的 per-channel 配方中，同时开放原 joint 的 VQ codebooks 和 log row gains。原 per-channel 的含义保持不变：每个物理 input channel 一个 FP32 log2D，每个 Linear 一个 clipping ratio；不是给每个channel单独一个clip ratio。

两组从共同 C8 rho1.5/update8、D=1、ratio=1 初始化，新增80 updates。沿用原C80的80 contexts、原顺序、sampler seeds、共享BF16 teacher和原first10×6 motor MSE + 1.5×0.028303017887797623×future latent NMSE loss。只用固定最终update80，不新增development set，不按held-out挑checkpoint。

相对原per_channel_clip，本组增加books/log row gains的训练及其所需weight adjoint。相对更早的C80 joint，除了新增D，也采用了per-channel配方已有的完整input absmax coupling，因此后者是配方参考，不能把其差异完全归因为D。

本组训练 books、log row gains、ratios和per-channel log2D；indices、原B32 integer exponents、Smooth0.5/H128、bias、其他模型参数及BF16 KV冻结。学习率分别为weights .0005、ratios .01、log2D .02；保留原Adam首步后清state、gradient norm clip1、ratio范围[.5,1]、log2D范围[-1,1]。

Forward使用 `F.linear(Q_A4(Z/D;ratio), BF16(Wq*D), bias)`，Wq每步由实际BF16 books/row gains解码更新。保留已有per-channel的dynamic-absmax/input/D surrogate，只增加从实际BF16 compensated F.linear回到未补偿Wq的weight adjoint，再通过既有DecodedWeight更新FP32 masters。D的已有surrogate已经包含权重补偿贡献，不重复添加另一条D梯度。

GPU allocation内，先通过原clipping和per-channel检查，再用独立native autograd BF16 compensated F.linear核对新增weight adjoint；覆盖D=1/非1、H128/tail、多维batch、仅weight可导、books/log gains连通和零输入。原x/ratio/D梯度需与已完成per-channel实现逐位一致。检查通过后才加载模型训练。

最终checkpoint保存614组FP32 books/log gains、ratios和log2D；checkpoint另存Adam/RNG。训练和正式评估使用同一实际BF16补偿，完整uncached/cached action/future parity通过后冻结全部参数。补偿rounding误差的参照是最终未补偿joint Wq，不是原baseline，也不是完整精度weights的总VQ误差。

新增最终calibration80及原held-out40 contexts×2 seeds=80 queries。复用baseline、clip_only80、原C80 joint、原per_channel_clip四组640条记录，合计800。报告first10 motor RMSE、future latent NMSE、原40个case内平均seed后的胜负及四suite变化；full32/late22/gripper仅诊断。没有新teacher、closed-loop或native kernel评估。

GPU总预算<2h，单作业walltime01:59、软截止6840秒，保留逐步checkpoint和端点但不自动续提GPU。GPU利用率和显存每15秒写job log；gpt-6-luna/xhigh subagent每10分钟只读监控，普通Q/R保持安静，终止/失败/需要处理时通知。所有模型、Torch、数值运算与重I/O均在已验证PBS allocation内；login仅连接、提交、状态和小文件控制。

本轮为固定cohort、decoded-BF16 VQ + fake A4数值比较。原held-out已反复用于工程分析，每配方单次训练，不能据此宣称统计显著、闭环改进或native kernel收益。

## 已完成结果

GPU `25864060.pbs101`：F / Exit 0，walltime **01:27:15**。CPU aggregate `25865058.pbs101`：F / Exit 0，walltime 00:00:26。两组均通过 completion gates；新 arm 完成80 updates、calibration80、held-out80，共160条新端点；640条旧端点按原plan、seeds、teacher binding复核并复用。新teacher为0。

614/614组books和row gains均发生更新；初始baseline parity和最终uncached/cached的完整action、future latent parity均为0。

| 原 held-out：40 contexts × 2 seeds | First10 motor RMSE | Future latent NMSE |
|---|---:|---:|
| baseline | 0.044296774 | 0.060877371 |
| clip_only | 0.043848437 | 0.054848124 |
| joint_c80 | 0.041846837 | 0.048973336 |
| per_channel_clip | 0.031641466 | 0.055377875 |
| joint_per_channel | 0.032668161 | 0.048842217 |

相对per_channel_clip，joint_per_channel的motor RMSE **+3.245%**，latent NMSE **-11.802%**；40个case内先平均两seed后，motor为15改善/25退步，latent为23改善/17退步。相对原joint，motor **-21.934%**、latent **-0.268%**；latent平均值几乎持平，17 case改善/23退步，不能描述成稳定的大幅latent改善。

Calibration：per_channel_clip为0.029800742 / 0.044684950，joint_per_channel为0.033144291 / 0.051131065（motor / latent）。新joint在calibration上两项也更差，因此这轮不是典型“训练集更好、held-out更差”的过拟合图景。当前证据更适合把它视为联合优化的取舍，尚不能据此确定具体梯度冲突或学习率原因。

First10 gripper RMSE为0.080485551：比per_channel_clip高2.781%，比原joint高81.890%。尽管motor相比原joint改善明显，这仍是后续闭环验证需要关注的差异；本轮没有跑closed-loop，不能宣称成功率提高。

完整suite结果及诊断项见 [RESULTS.zh.md](results/RESULTS.zh.md)，原始摘要见 [comparison.json](results/comparison.json)，终态证明见 [completion_evidence.json](completion_evidence.json)。最终部署checkpoint保存在 `/scratch/users/ntu/yguo017/fastwam-vq-joint-per-channel-20261018/results/train_eval/deployment.pt`，包含FP32 VQ masters、ratios和per-channel log2D。
