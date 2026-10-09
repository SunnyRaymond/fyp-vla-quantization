# Fast-WAM 三组 rotation baseline

本次在同一个 Optional-IDM checkpoint 上对比 BF16、QuaRot-adapted W4A4、SpinQuant-adapted W4A4。Plain W4A4 已表现很差，不再列入本轮。固定 rotation 与学习 rotation 使用相同初始化、相同位置、相同 W4/A4 quantizer，区别是是否优化 rotation。

## 样本和配对

从已有官方 LIBERO-Plus pilot manifest 中，不使用任何历史成绩，按 suite × perturbation dimension 分层选取 10 个 variants，同时尽量均衡 original task、subtype 和 difficulty。4 suites × 7 dimensions × 10 variants = **280 episodes/arm，三组共 840 episodes**。每个 variant 只采用 state 0 和一个确定的 sampler/environment seed；并非每个 variant 另跑 10 次。三组共享冻结 manifest、initial state、env seed、每次 replan 的 sampler seed、episode cap 和动作执行规则。

此前 pilot 的设计为每 cell 50 个 variants，Spatial 是 7 × 50 = 350 slots/arm；本轮 Spatial 是 7 × 10 = 70 slots/arm。设计分母不代表此前所有 slots 均成功完成。

## 适配范围

Fast-WAM 的 LayerNorm、timestep modulation、channel gating 不允许照搬 LLaMA 全局 residual rotation。这里在每个目标 Linear 的输入端做局部右乘：`x' = xR`，`W' = WR`；正交时 `xR (WR)^T = xW^T`。Rotation 位于上游 norm/modulation/nonlinearity 之后，未穿过这些运算。

每个 video/action/proprio stream 共享一个 128 × 128 rotation；不足 128 的尾块使用固定的 power-of-two signed Hadamard。QuaRot-adapted 保持固定 randomized signed block-Hadamard。SpinQuant-adapted 从同一矩阵开始，只学习 Cayley 参数，`R = R0 @ Cayley(A - A.T)`。原模型权重冻结。

这是局部 R128 的适配比较，不是原论文的 LLaMA global R1 或完整 R1/R2/R3/R4 复现。训练采用 Adam 优化 Cayley 参数，也不称作原作者的 Cayley SGD。两组均用 RTN；本轮不加入 GPTQ，从而不会把 rotation 学习收益与更换 weight quantizer 混在一起。

W4 为 native packed signed INT4、G128、BF16 scales；A4 为动态 per-row absmax、FP32 scales、signed INT4。执行真实 `S4 × S4 mma.sync`，不是以 fake quantization 代替评估。KV、T5/VAE、attention arithmetic、norm、bias、scheduler 和非目标参数保留浮点。

## Calibration 和学习

采用原始 LIBERO 四个 suites 的 task IDs 0、1 作 calibration，task ID 2 作 selection：共 8 条 training observations、4 条 selection observations，各为 state 0 + 30 settling steps。Plus 评估 observations 不参与学习或选 checkpoint。

每条 observation 先运行官方 BF16 IDM query，采集 video/action 的 denoising indices 0、3、6、9，以及对应完整 denoiser 输出。优化目标为输出对 BF16 teacher 的 normalized MSE；video/action 交替抽取，按固定 seed 对各自全部 packets 作循环排列。Context 的原始 text 部分固定，proprio token 用当前 quantized encoder 重新计算；action packets 使用固定 teacher BF16 video KV。它不是局部 Linear reconstruction loss，也没有展开整个 sampler 对最终 action 做反传。

SpinQuant rotation 的历史训练预算为 Adam lr=0.001、200 steps、gradient norm clip=1；每 40 steps 在独立 selection packets 上评估。当前复用的 `learned_rotation.pt` 来自 PBS job `25726472.pbs101`，原训练已选中 step 200。本轮准备阶段原样复用该 checkpoint，不重训、不重新 selection，也不覆盖 `rotation_training/` 或已有 banks。Step 0 仅是历史诊断，不作为 learned arm 候选。即使 learned rotation 没改善，也记录结果并运行完整三组闭环比较。

当前 fixed-input gate 的 STE reference 使用 FP32 activation/weight 解码值、FP32 bias 和 GEMM，Linear 输出才转回 BF16，以匹配 native integer dot 的 FP32 scale/累加边界；准备阶段关闭 TF32。历史训练 `25726472.pbs101` 的 STE operands 曾先舍入到 BF16，随后 QuaRot native-vs-STE gate 得到 action RMSE 0.055302（门槛 0.05）并停止，未运行 evaluation episodes。此轮保留其原有 step-200 learned rotation、两份 bank、96-packet BF16 teacher trace、observations 和 manifest；修正的是 gate reference 的 FP32 一致性。准备 PBS 重新加载模型，对现有旋转和 banks 执行 corrected FP32-STE full-query numeric checks 及原 native gates；不在 FP32 STE 下重训，也不重新选择 checkpoint。准备脚本会在 compute allocation 内保存旧 protocol/progress 快照，不移动或覆盖训练、bank、teacher、observation 或 cohort 文件。

## 计算安排和检查

流程拆成两个独立 PBS jobs。第一阶段 `prepare.pbs` 请求 1 GPU、8 CPUs、55 GB、walltime `01:55:00`；作业 `25727738.pbs101` 已从 `normal` 路由至 `gdev`，server 实际配置为 16 CPUs、110 GB。它只做 frozen rotation 的验证：运行 CPU rotation selfcheck 和 native quantization/STE fixtures，加载一次完整 pipeline，完成 corrected FP32-STE full-query numerical checks 与原 native gates。通过后只写 `PREPARATION_COMPLETE`；不写 pipeline 完成标记、不启动 evaluation worker。这次准备作业的 transformed-BF16 checks 与 tiny native fixtures 完成，但 QuaRot case 2 的 native-vs-corrected-dense-STE action32 RMSE 为 0.05439452，高于原 0.05。用户明确放行完整测评；PREPARED 保留 native check passed=false 和具体 waiver，不重训、不追加诊断任务。

第二阶段 `run.pbs` 请求 4 GPUs、32 CPUs、220 GB、walltime `24:00:00`，只做正式 paired evaluation，不训练或重新准备模型。4 个单卡 worker 各分配 70 个 variants、210 个 episodes，覆盖所有 suite/dimension，每 cell 2–3 条，SensorNoise 和 Long 均分散到四张卡。每个 worker 按 BF16 → QuaRot → SpinQuant 顺序运行；T5/VAE 与 pipeline 保留，组间只换目标 denoiser 的 prepared bank。Worker CPU thread budget 为 8。两个阶段均每 15 s 把 allocation 中 GPU utilization 和 memory 写入各自 `job.log`；worker 失败时停止其他 worker，且不聚合为成功完成。

无重试时 preparation 加载一次完整 pipeline，评估加载四次，合计 **5 次完整加载**。评估有 12 次 arm installations，其中首次 BF16 并不加载额外 quantized bank。准备阶段可在 teacher trace 已存在时复用原 96 个 packets；不得因 protocol/progress 快照而移动或覆盖 frozen teacher、observations、manifest、原训练 checkpoint 或 banks。

正式 episodes 前已有 PREPARED gate 必须确认 rotation orthogonality、first-10 transformed-BF16 drift、corrected FP32-STE/native full-query action RMSE、PTX/call count/packed-weight 证据。四条独立 selection observations 的 first-10 motor RMSE 上限 0.02，gripper RMSE 上限 0.05；native-vs-STE full-query action RMSE 上限 0.05。原定不通过则停止。2026-10-08 的明确用户决定如下：接受当前 native consistency 误差，直接运行四卡完整评测。原 0.05 gate 不标记为通过；通过有记录的用户 waiver 放行。

两个 PBS jobs 各自每 15 s 将分配到的 GPU utilization 和 memory 写入 job log。Login node 只作连接、提交、状态和小控制文件操作；模型加载、fixed-input gate、native bank 验证、评估和聚合均在真实 PBS compute allocation 内运行。

## 报告和完成标准

主要结果是各组在配对 variants 上的 success count/rate，另按 suite（70/arm）、dimension（40/arm）、cell（10/arm）列明准确分母；报告同一 variant 上的成败差异。Timeout 为 policy failure，基础设施错误是缺失 slot。

分别报告连续 denoising GPU latency、完整 action-query wall time、包含环境的 episode wall time；first query 与 steady queries 分开。不能用本轮 W4A4 denoiser 结果声称全模型 4-bit、KV4、端到端 acceleration 或原论文复现。

需同时具备 840 个唯一有效 terminal episode slots（各 arm 280、各 cell 10）、aggregation complete、PBS Exit_status=0 和 PIPELINE_COMPLETE 才算实验完成。排队、运行或 scheduler X 单独均不算完成。


## 2026-10-08 用户放行记录

用户指令：差不多得了，这点小误差不会影响最终结果，直接跑四卡完整评测实验即可

正式实验复用原 step-200 SpinQuant、原两份 banks、冻结的 280-variant manifest。0 个新增训练 steps，不重选 checkpoint。放行只针对准备阶段 native consistency gate；full evaluation 的 native PTX、实际 GEMM、目标 BF16 weight absence、episode 有效性与 840-slot 完成条件照常检查。未完成的其余 native selection queries 不宣称通过；当前误差是否影响闭环结果由本次测评观察，不预先保证。两次准备已经各加载一次 pipeline；正式四个 workers 各加载一次，若无评估重试总计六次。Grouped-reference 后续验证方案未提交。
