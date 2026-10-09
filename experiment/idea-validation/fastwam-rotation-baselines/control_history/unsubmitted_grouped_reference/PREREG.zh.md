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

历史训练作业 `25726472.pbs101` 的 STE operands 曾先舍入到 BF16，QuaRot native-vs-STE gate 的 action RMSE 为 0.055302（门槛 0.05），因此停止且没有运行 evaluation episodes。随后 reuse 验证作业 `25727738.pbs101` 使用 dense FP32-dequant STE（先完整反量化，再沿 whole K 做 FP32 reduction），其 native 对 reference 的 action RMSE 为 0.05439452，仍高于 0.05。这个结果继续保留为诊断；dense reduction 的舍入顺序不同于 G128 grouped integer dot 与 scale accumulation，不能单凭该差异归因为 native kernel bug。

Preparation validation v2 仍复用 `25726472.pbs101` 训练并在 step 200 选中的原 rotation、两份原 bank、96-packet BF16 teacher trace、observations 和 manifest。helper 会在分配的 GPU 上先分别对两个原 bank 执行 online rotation + quantization，并逐 bit 检查与 packed bank 的一致性；之后在相同 full-query inputs 上用独立的 S8×S8 int32 dot、G128 分组和 FP32 scale/partial-sum accumulation reference 对比真实 native S4 MMA。新的 0.05 gate 明确命名为 native-vs-grouped-reference；dense FP32-dequant RMSE 只作诊断。该 reference 仅用于 preparation gate，不进入 closed-loop evaluation runtime，也不增加 native counters。此次不重训、不重新 selection、不覆盖 checkpoint 或 banks，样本、阈值及其余科学控制均保持冻结。旧 protocol/progress 快照由准备 helper 在 compute allocation 内保存。

## 计算安排和检查

流程拆成两个独立 PBS jobs。第一阶段 `prepare.pbs` 请求 1 GPU、8 CPUs、55 GB、walltime `01:55:00`，队列为 `normal`（路由到 `gdev`）；失败作业 `25727738.pbs101` 的 server 实际配置为 16 CPUs、110 GB。Preparation validation v2 保持相同资源申请，仅运行 CPU rotation selfcheck、native quantization/STE fixtures、一次模型加载、两个 bank 的在线逐 bit coherence check、同输入 grouped-reference full-query gate 和原 native gates。它不重训、不重选、不覆盖旧 rotation/banks。通过后只写 `PREPARATION_COMPLETE`；不写 pipeline 完成标记、不启动 evaluation worker。只有 PREPARED 的 protocol、ready、numeric、native INT4 gates 和两份 bank 文件均通过后，才提交第二阶段。

第二阶段 `run.pbs` 请求 4 GPUs、32 CPUs、220 GB、walltime `24:00:00`，只做正式 paired evaluation，不训练或重新准备模型。4 个单卡 worker 各分配 70 个 variants、210 个 episodes，覆盖所有 suite/dimension，每 cell 2–3 条，SensorNoise 和 Long 均分散到四张卡。每个 worker 按 BF16 → QuaRot → SpinQuant 顺序运行；T5/VAE 与 pipeline 保留，组间只换目标 denoiser 的 prepared bank。Worker CPU thread budget 为 8。两个阶段均每 15 s 把 allocation 中 GPU utilization 和 memory 写入各自 `job.log`；worker 失败时停止其他 worker，且不聚合为成功完成。

无重试时 preparation 加载一次完整 pipeline，评估加载四次，合计 **5 次完整加载**。评估有 12 次 arm installations，其中首次 BF16 并不加载额外 quantized bank。准备阶段可在 teacher trace 已存在时复用原 96 个 packets；不得因 protocol/progress 快照而移动或覆盖 frozen teacher、observations、manifest、原训练 checkpoint 或 banks。

正式 episodes 前已有 PREPARED gate 必须确认 rotation orthogonality、first-10 transformed-BF16 drift、same-input grouped-reference/native full-query action RMSE、PTX/call count/packed-weight 证据。四条独立 selection observations 的 first-10 motor RMSE 上限 0.02，gripper RMSE 上限 0.05；native-vs-grouped-reference full-query action RMSE 上限仍为 0.05。Dense FP32-dequant diagnostic 不替代该 gate。不通过则停止，不改变样本或阈值绕过；准备 gate 失败不得启动 4-GPU worker。

两个 PBS jobs 各自每 15 s 将分配到的 GPU utilization 和 memory 写入 job log。Login node 只作连接、提交、状态和小控制文件操作；模型加载、fixed-input gate、native bank 验证、评估和聚合均在真实 PBS compute allocation 内运行。

## 报告和完成标准

主要结果是各组在配对 variants 上的 success count/rate，另按 suite（70/arm）、dimension（40/arm）、cell（10/arm）列明准确分母；报告同一 variant 上的成败差异。Timeout 为 policy failure，基础设施错误是缺失 slot。

分别报告连续 denoising GPU latency、完整 action-query wall time、包含环境的 episode wall time；first query 与 steady queries 分开。不能用本轮 W4A4 denoiser 结果声称全模型 4-bit、KV4、端到端 acceleration 或原论文复现。

需同时具备 840 个唯一有效 terminal episode slots（各 arm 280、各 cell 10）、aggregation complete、PBS Exit_status=0 和 PIPELINE_COMPLETE 才算实验完成。排队、运行或 scheduler X 单独均不算完成。
