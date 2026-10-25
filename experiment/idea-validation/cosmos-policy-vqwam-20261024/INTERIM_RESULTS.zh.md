# Cosmos Policy 量化：固定 endpoint 性能与闭环结果

六种方法的固定 endpoint 性能均完成，每臂覆盖 80 个固定 query、720/720 次测量；闭环评估已验证 1,680/1,680 条记录，六臂各 280 条。下表和分组结果来自完整报告 JSON。

## 固定 endpoint 性能

| 方法 | 闭环成功 | 五 NFE denoising mean (ms) | Spd = BF16 mean / 方法 mean | Peak allocated max (GB) | Whole DiT packed storage (GiB) |
|---|---:|---:|---:|---:|---:|
| BF16 | 233/280 (83.21%) | 367.886 | 1.000× | 5.7094 | 3.6441 |
| SpinQuant W4A4 | 221/280 (78.93%) | 780.141 | 0.472× | 5.7095 | 1.2089 |
| QuaRot W4A4 | 227/280 (81.07%) | 1,310.805 | 0.281× | 5.7101 | 1.2094 |
| SVDQuant adapted, rank-16 weighted residual | 78/280 (27.86%) | 891.628 | 0.413× | 5.7580 | 1.2541 |
| ViDiT-Q mixed W4/W8A4 | 232/280 (82.86%) | 1,467.195 | 0.251× | 5.7128 | 1.2424 |
| ours VQ-WAM | 230/280 (82.14%) | 566.490 | 0.649× | 3.6079 | 1.2421 |

性能使用 NVIDIA A100-SXM4-40GB、PyTorch 2.7.0+cu128、CUDA 12.8。Latency 是同步的完整五 NFE denoising wall interval，包含 per-query condition-cache setup 与方法的在线变换/解码，排除 VAE/T5 和 observation preparation；它不包含完整 endpoint 调用中的其他拷贝与输出提取。Spd 按表头的均值比计算，小于 1 表示比 BF16 慢。Peak 是 720 条测量中的最大 allocated GB；storage 是整套 DiT packed inference weights，包含未量化部分与 metadata。

ours 的 566.490 ms 仍慢于 BF16，但在本次量化实现中最快；比 SpinQuant 少 213.651 ms。ours 比 BF16 少 3 次成功（−1.07 个百分点），比 SpinQuant 多 9 次（+3.21 个百分点），比 ViDiT-Q 少 2 次（−0.71 个百分点）。这些是本固定 cohort 的描述性结果，不代表统计显著性或纯算法因果效应。

## 闭环成功率分组

每个 suite 含 70 条 paired episode；每个扰动维度含 40 条。单元格为成功数/分母（成功率）。

| Suite | BF16 | SpinQuant | QuaRot | SVDQuant adapted | ViDiT-Q | ours VQ-WAM |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-10 | 58/70 (82.86%) | 60/70 (85.71%) | 57/70 (81.43%) | 11/70 (15.71%) | 61/70 (87.14%) | 58/70 (82.86%) |
| LIBERO-Goal | 54/70 (77.14%) | 46/70 (65.71%) | 51/70 (72.86%) | 14/70 (20.00%) | 51/70 (72.86%) | 50/70 (71.43%) |
| LIBERO-Object | 62/70 (88.57%) | 61/70 (87.14%) | 61/70 (87.14%) | 28/70 (40.00%) | 59/70 (84.29%) | 62/70 (88.57%) |
| LIBERO-Spatial | 59/70 (84.29%) | 54/70 (77.14%) | 58/70 (82.86%) | 25/70 (35.71%) | 61/70 (87.14%) | 60/70 (85.71%) |

| 扰动维度 | BF16 | SpinQuant | QuaRot | SVDQuant adapted | ViDiT-Q | ours VQ-WAM |
|---|---:|---:|---:|---:|---:|---:|
| Background textures | 31/40 (77.50%) | 29/40 (72.50%) | 31/40 (77.50%) | 6/40 (15.00%) | 32/40 (80.00%) | 32/40 (80.00%) |
| Camera viewpoints | 29/40 (72.50%) | 26/40 (65.00%) | 26/40 (65.00%) | 3/40 (7.50%) | 29/40 (72.50%) | 28/40 (70.00%) |
| Language instructions | 38/40 (95.00%) | 36/40 (90.00%) | 35/40 (87.50%) | 13/40 (32.50%) | 37/40 (92.50%) | 37/40 (92.50%) |
| Light conditions | 40/40 (100.00%) | 40/40 (100.00%) | 40/40 (100.00%) | 24/40 (60.00%) | 40/40 (100.00%) | 39/40 (97.50%) |
| Objects layout | 36/40 (90.00%) | 34/40 (85.00%) | 35/40 (87.50%) | 16/40 (40.00%) | 34/40 (85.00%) | 35/40 (87.50%) |
| Robot initial states | 22/40 (55.00%) | 19/40 (47.50%) | 21/40 (52.50%) | 6/40 (15.00%) | 21/40 (52.50%) | 21/40 (52.50%) |
| Sensor noise | 37/40 (92.50%) | 37/40 (92.50%) | 39/40 (97.50%) | 10/40 (25.00%) | 39/40 (97.50%) | 38/40 (95.00%) |

ours 对各方法的 paired 结果：

| 对比方法 | ours wins | losses | ties | 成功率差 (百分点) |
|---|---:|---:|---:|---:|
| BF16 | 8 | 11 | 261 | −1.07 |
| SpinQuant W4A4 | 15 | 6 | 259 | +3.21 |
| QuaRot W4A4 | 10 | 7 | 263 | +1.07 |
| SVDQuant adapted, rank-16 weighted residual | 153 | 1 | 126 | +54.29 |
| ViDiT-Q mixed W4/W8A4 | 9 | 11 | 260 | −0.71 |

包含 28 个 suite×dimension cells 的完整分层数据与 paired 分母保留在完整报告 JSON 中，没有在本文重复展开。SVDQuant 的 27.86% 仅对应本项目的 Cosmos-adapted、rank-16 weighted-residual 版本，不能外推为官方 SVDQuant 的结果。

## 方法与证据边界

ours 使用 Smooth α=0.5、signed H128 与共享 learned U128；权重为 B32 microscaling additive VQ，vector dimension 4、两套 K=256 codebook。先以 W16A4 更新 rotation 8 次并冻结，再更新 codebook 与 row gain 8 次；每次使用 10 个 C80 context。indices、block exponents、base model 与 transforms 在 VQ 阶段固定，采用最终 update 8，不通过测试集选 checkpoint。

SpinQuant、QuaRot、SVDQuant、ViDiT-Q 使用各自 Cosmos-adapted 的 eager decoded-BF16/fake-A4 数值执行；没有使用 ours engine，也不是 native INT4 kernel 性能或官方方法的完整复现。SVDQuant 采用 calibration-only smoothing 搜索及 rank-16 weighted residual。ViDiT-Q 在预算内对 9 个完整 Linear 配置 W8A4。ours 使用 packed Triton decode、BF16 GEMM 与 net graph，per-Linear 权重以 decoded BF16 参与计算；native_int4_kernel_used=false，同样不能称为 native INT4。

SpinQuant 复用了原 C8 cohort 的 update 200 校准产物；其他新校准使用 C80。因此校准覆盖和更新预算不同，闭环差异不能归为单一量化算法效果。QuaRot、SVDQuant、ViDiT-Q 分别完成 280/280 Linear 校准，encoded target 为 4.128125、4.345833、4.288839 BPW；ours deployment 覆盖 280/280 Linear、4.287277 BPW。ours 训练作业 25968675 完成 R 的 8 次更新和 VQ 的 8 次更新；engine 作业 25969437 完成同一 artifact 的 full-80-query action/future 一致性验证，性能作业 25969497 完成 80 query、720/720 样本。

## 闭环记录恢复与来源

GPU 闭环作业 25969569.pbs101 以 F/Exit 1 结束，source_execution_complete=false、source_pipeline_complete=false；worker 3、4、5、7 因等待 claim_queue.json.lock 超时退出。该 GPU 作业没有被标记为成功，也没有伪装成八个 worker 全部成功。

作业结束前已原子写入全部 1,120 条新记录。CPU 恢复作业 25971643.pbs101 以 F/Exit 0 和 PIPELINE_COMPLETE 完成，逐条验证这 1,120 条新记录及复用的 560 条旧记录；没有新增 model forward 或 episode（new_model_forwards=0，new_episodes=0）。因此这里的 data_complete=true 表示记录完整且验证通过，不改变源 GPU 作业的失败状态。queue ledger 与短锁修复仅用于后续 control 运行，不改写本次模型数据或 worker 结果。

完整报告保留 1,680 条覆盖、六臂各 280 条、40 个分层 strata、ours 对五方法的 paired 结果及六臂性能记录。初次完整核验由 CPU 作业 25971643 完成；含 gripper 诊断的最终报告由 CPU 作业 25971904 导出，完整 JSON 位于 retrieved/reporting.summary.complete.json（127,837 bytes）。原始核验版保留在 retrieved/reporting.summary.before_gripper.json，其中与最终版的主表、闭环分组和 paired 数据完全相同。

## 单 DiT 的 reader-KL 适配与训练证据

Cosmos reader 使用同一 joint DiT 的冻结 BF16 suffix。在第 3 次 denoising evaluation 前切开轨迹，只用量化轨迹的 noisy future-image slots 6/7 替换 teacher 对应位置；action slot 4、历史及其他 slots 保持 teacher 值并 detach。随后从相同的真实 solver state 和 multistep history 继续计算最终 action，比较 teacher future 与量化 future 经过同一冻结 suffix 后的 motor-action 读出；模型没有独立 action decoder。

连续 motor action 先用 C80 的 16×6 mean/std 归一化（std floor 0.02），再通过 16-bin Gaussian-CDF soft bin（sigma 0.375）定义 KL。这是 soft-bin consistency KL，不是模型原生的 action probability distribution KL。训练损失为 full-16×6 motor MSE + λ·future-view latent NMSE + β·reader-KL；C80 初始化后固定 λ=0.0645743742、β=1.7317817103，再开始 VQ update 1。

记录显示 C80 reader 输入梯度连接为 80/80；8 次 VQ 更新均记录 560 个有梯度的 parameter tensors、有限梯度，以及 staged replay max-abs=0。560 是 Tensor 数量，不是标量参数量。结合上述 full-80 same-artifact action/future 验证，这些证据验证了训练和执行路径；本轮没有 A+Z 对照或消融，不能把结果归因于 reader-KL 单独提升闭环成功率。

## Gripper 独立诊断

仅比较每个 paired 初始状态下已经保存的首个 16-step action chunk，第七维为原始连续 gripper scalar。每臂覆盖 280 chunks、4,480 个预测值，参考为 BF16；该范围不包含模型轨迹分叉后的后续 action。指令诊断使用原始第七维的 sign（零单独保留），对应当前 PandaGripper 输入方向规则；没有增加 OpenVLA 翻转或缩放。

| 方法 | Gripper MAE | Gripper RMSE | Sign 指令差异 / 4,480 |
|---|---:|---:|---:|
| BF16 | 0 | 0 | 0 |
| SpinQuant | 0.024493 | 0.027243 | 0 |
| QuaRot | 0.014807 | 0.017338 | 0 |
| SVDQuant adapted | 0.166701 | 0.172734 | 0 |
| ViDiT-Q | 0.007970 | 0.009699 | 0 |
| ours VQ-WAM | 0.008073 | 0.009917 | 0 |

这些零 sign 差异只适用于生成的首个 chunk，不能据此推断全轨迹 gripper 指令一致。诊断作业 25971904 为 CPU-only、F/Exit 0，使用已有记录，新增 forward 和 episode 均为 0。完整数值另见 retrieved/gripper.diagnostics.complete.json。

## 来源

性能作业：BF16/SpinQuant 25965506；QuaRot 校准/性能 25965507/25965819；SVDQuant 25965880/25966458；ViDiT-Q 25966507/25967221；ours VQ-WAM 25969497。闭环 GPU source 25969569（F/Exit 1）；CPU 记录恢复 25971643、最终报告与 gripper 诊断 25971904（均 F/Exit 0）。

性能汇总和逐臂记录：retrieved/performance.six_arm_summary.json；retrieved/performance.bf16.json、retrieved/performance.spinquant_w4a4.json、retrieved/performance.quarot_w4a4.json、retrieved/performance.svdquant_w4a4.json、retrieved/performance.viditq_w4a4.json、retrieved/performance.ours_vqwam.json。校准记录：retrieved/quarot.complete.json、retrieved/svdquant.complete.json、retrieved/viditq.complete.json、retrieved/ours.calibration.complete.json。闭环完整结果及 28 个 cells：retrieved/reporting.summary.complete.json。ours full-80 engine 验证：retrieved/ours.engine_validation.full80.json；存储：retrieved/ours.bundle.storage.json；闭环资源决策：evaluation_dispatch.json。

