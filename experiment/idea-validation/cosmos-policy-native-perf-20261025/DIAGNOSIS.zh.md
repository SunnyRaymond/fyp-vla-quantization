# Cosmos Policy reader-KL 与性能问题诊断

本文区分已完成的旧实验、已取得的 profile，以及尚待 GPU 验证的新 native 路径。新工作固定已有 checkpoint，new training updates = 0，new closed-loop episodes = 0。

## 1. 单 DiT reader-KL 确实使用 future-only intervention

当前实现与原设计一致。在第 3 次 denoising evaluation 之前取得 teacher 的真实 solver boundary，detach/clone teacher state，再只替换 future-image slots 6/7：

```python
state = _clone(capture.state)
teacher_x = state[0]
hybrid = torch.where(future_mask(teacher_x), student_noisy_state, teacher_x)
state = type(state)((hybrid, *state[1:]))
```

mask 只在 `[:, :, 6:8]` 为真。Action slot 4、其他 slots、condition 及切点处 sampler history 来自 teacher；student 的 action 没有注入 reader。随后由冻结 BF16 joint DiT 的真实 sampler suffix 读出最终 action。

“保持 teacher 状态”指 intervention 的初始条件。进入 suffix 后，hybrid state 按原 solver 自然演化，action 会受到所替换 future 的影响；没有每一步再把 action clamp 回 teacher。否则该 loss 就无法测量 future 对最终 action 的作用。实际使用的 solver2mid 是 midpoint/RK 路径，虽保留真实 state 的其余字段，不应把它描述成依赖历史 x0 的通用多步 solver。

连续 motor action 用固定统计量归一化后经过 Gaussian-CDF soft bins，计算 consistency KL。它不是模型原生 action probability distribution 的 KL。此前已验证 teacher identity、future-only cotangent、80/80 梯度连通和 staged replay parity。这些验证支持实现正确，不构成 reader-KL 单独提升闭环的消融证据。

代码依据：[reader.py](../cosmos-policy-vqwam-20261024/reader.py)、[test_reader_math.py](../cosmos-policy-vqwam-20261024/test_reader_math.py)、[train.py](../cosmos-policy-vqwam-20261024/train.py)。Reader suffix 只用于 calibration/training；部署仍为五次 NFE，推理变慢不是多跑了 reader。

## 2. 旧效率表反映的是当时的执行实现

旧实验每臂 80 queries、720 次测量，A100-SXM4-40GB，Torch 2.7.0+cu128，五 NFE sampler wall time：

| 方法 | Mean ms | Spd. | Peak allocated GB | Packed whole-DiT GiB |
|---|---:|---:|---:|---:|
| BF16 | 367.886 | 1.000× | 5.7094 | 3.6441 |
| SpinQuant | 780.141 | 0.472× | 5.7095 | 1.2089 |
| QuaRot adapted | 1310.805 | 0.281× | 5.7101 | 1.2094 |
| SVDQuant adapted, rank16 | 891.628 | 0.413× | 5.7580 | 1.2541 |
| ViDiT-Q adapted, mixed W4/W8A4 | 1467.195 | 0.251× | 5.7128 | 1.2424 |
| Ours | 566.490 | 0.649× | 3.6079 | 1.2421 |

四个 baseline 当时都以 decoded BF16 weights + fake A4 的 eager 路径执行。它们的 packed storage 是压缩模型载荷，实际 GPU GEMM 仍用 BF16 weights，所以显存几乎没降，且额外承担在线变换和量化开销。该表不能证明这些方法的 native INT4 inference 慢于 BF16。

Ours 已经使用 packed resident VQ、Triton layer decode、共享 scratch 和局部 CUDA Graph，但 `native_vq_kernel=False`、`native_integer_gemm=False`。每个 Linear 先解码 BF16 weight，再走 BF16 cuBLAS；A4 也没有因此变成 INT4 Tensor Core GEMM。相比 BF16，原有 GEMM 工作仍在，另加 Smooth/H128、FP32 U、A4 和 decode。

这里还有一个表示层面的区别：当前 d4、两本 K256 的 additive VQ 是每四个 weight 保存两个 8-bit index，因此仅 index 部分为 4 bpw；重构值为 microscale 乘以两个 codebook vector 之和，并非 16 个均匀整数等级。不能把这些 index 当作 signed INT4 weight 交给 QuaRot/Nunchaku GEMM。我们的专用 native 路径需要直接消费 codebook 表示，或融合 lookup/decode 与矩阵乘；单纯接入通用 INT4 GEMM 会改变量化模型。4 bpw 的存储收益不能直接推导出 INT4 Tensor Core 的吞吐收益。

Ours 的旧实测比 BF16 慢约 54.0%，同时 Peak 下降约 36.8%、packed storage 下降约 65.9%。现有实现已兑现压缩与显存收益，尚未兑现低比特计算吞吐收益。此前用这张表比较各算法的部署速度，证据不充分，应纠正这一解释。

旧结果来源：[完整旧表](../cosmos-policy-vqwam-20261024/INTERIM_RESULTS.zh.md)、[旧六臂 JSON](../cosmos-policy-vqwam-20261024/retrieved/performance.six_arm_summary.json)。旧闭环结果仍属于原 checkpoint/reference 执行路径，不能自动移植成新 native backend 的闭环结果。

## 3. 新 profile 已定位的开销

GPU job `25974929.pbs101` 的四个代表 query 再次观察到 warm sampler 约 BF16 0.40 s、ours 0.57 s。首条冷启动约 4.5 s，包含准备/graph capture，不纳入正式 steady-state 性能结论。

实际 graph 路径中，ours 前两次 NFE 仍为 eager，后三次 replay CUDA Graph。条件/cache 的一致性检查还会触发 host-device 同步。CPU trace 中，ours 的 `cudaStreamSynchronize` 为 143 次，`_local_scalar_dense` 为 119 次；BF16 分别是 59 次和 35 次。同步的等待时间包含它所等待的 GPU 工作，不能全部当成可删除的额外 CPU overhead。

单独的 forced-eager attribution 确认 H128、FP32 U、A4 和 decode 都产生了可观 GPU 工作。这一 repeat 专门用于分项归因，不能把它的耗时当成实际 graph 路径 latency，也不能把 nested annotation、ATen 和 CUDA kernel 的时间加总成“总开销”。VAE 的卷积出现在完整 endpoint profiler 中，但不在 sampler 计时段内。

无训练优化把五次 NFE 都纳入 graph，并在 graph 内重算 cross K/V，以消除前两次 eager 和相关 cache 匹配检查。GPU job `25975121.pbs101` 已通过全 80 queries 的完整 action（含 gripper）/future `max_abs <= 1e-6`。同一 allocation 的四代表 query × 三次测量中，sampler mean 从 616.052 降到 436.737 ms（相对旧 engine 1.4106×，约少 29.1%）；query Peak 从 3.6072 降到 3.3757 GB。一个 graph 覆盖全部五次 NFE。该结果支持采用这一调度优化，仍不是 native VQ GEMM。这组 12-sample 诊断与旧 BF16 720-sample 表的统计 panel 不同，不能直接拼接为正式 speedup；限制来自测量 panel，并非要求同一物理 GPU。正式六臂表仍待统一重测。

来源：[BF16 profile](retrieved/profile.initial.bf16.json)、[ours profile](retrieved/profile.initial.ours_vqwam.json)、[trace analysis](retrieved/profile.analysis.json)、[优化实测](retrieved/optimized.summary.json)、[候选实现](engine_optimized.py)。

原 graph-path trace 中，主要两类 BF16 GEMM kernel 的合计 device duration 在 BF16 与 ours 均约 172 ms；ours 额外记录到 Smooth/H128 约 64 ms、FP32 transform GEMM 约 27 ms、主要 VQ decoder 约 23 ms。这些是一次含 profiler overhead 的 endpoint trace 中对应 kernel 的累计 device duration，用于定位实现开销，不能直接相加还原 sampler latency，也不是优化后候选的正式分项表。它们说明仅消除 Python/同步还不够：保留 BF16 GEMM 时，变换与解码的 GPU 工作仍在。

## 4. SteerQuant Table 2 的差异

本地论文 v1 的 Table 2、Appendix D 明确：RTX 4090；FP-captured inputs；DiT replay；CUDA Graphs + CUDA events；5 warm-up sequences、20 timed sequences；报告 median。Cosmos-Policy 的 BF16 为 261.460 ms / 4.241 GB，QuaRot 为 122.657 ms / 1.604 GB，SVDQuant 为 131.093 ms / 1.755 GB，SteerQuant W4A4 为 117.322 ms / 1.645 GB。Table 2 包含 Atom；没有直接列出 SpinQuant 或 ViDiT-Q 的效率行。

我们的旧表是另一张 A100 上的 sampler wall-time 表，且 Peak 包含 resident tokenizer/VAE 和 endpoint temporaries。硬件、计时范围、mean/median、显存 residency 都不同。论文也明确将没有对应 low-bit kernels 的 PTQ4DiT、Q-DiT 排除出效率表，不能拿我们的 fake-quant baseline 去对应它的 native 性能行。

算法配置也不同。论文明确披露 SteerQuant 的 weight quantization 是 symmetric per-output-channel；当前冻结适配 checkpoint 使用 G128 scales、local H128、SVD rank16 或部分 W8A4 保护。论文没有公开足够的逐算子部署细节，不能凭表中数字断言其每个 baseline 使用了哪个具体 extension、哪些 fusion 或与我们完全相同的 group/rank 配方。

来源：[论文 PDF](../../../papers/wam/014-steerquant/paper-arxiv-v1.pdf)，pp.8、15。

## 5. 本轮 native 重测的实现边界

| 臂 | 新接入的核心路径 | 仍需明确的适配部分 |
|---|---|---|
| QuaRot | 官方 CUTLASS INT4 GEMM；官方 FHT H128 已编译，待 GPU gate；另有 public Nunchaku full-K 候选 | G128 逐组 scale/add 与 full-K 两种 adapter 均保留 checkpoint scales，需验证后选择 |
| SpinQuant | 原 learned rotation checkpoint + QuaRot/CUTLASS INT4 backend | 这是 SpinQuant 的 CUDA adapter；官方公开 ExecuTorch 部署支持 W4A8，不能标成其原生 Cosmos W4A4 全引擎 |
| SVDQuant adapted | 官方 Nunchaku INT4 GEMM + official MMA layout packing；另有 official LoRA-down/up fusion 候选 | G128 scales 无损复制为两个 G64；core 与 fused 候选均需通过固定 checkpoint fidelity gate |
| ViDiT-Q adapted | 其仓库官方 Atom W4A4 kernel | affine zero-point correction、padding；已有 W8 模块用两次 signed-INT4 精确分解 |
| Ours | 固定 VQ checkpoint 的 graph/cache 调度优化 | 当前仍为 packed decode + BF16 GEMM |

这些 native core 已编译不等于 GPU gate 已通过，更不等于完整官方 engine 已移植。正式表必须保留 adapter 名称/上游 commit、kernel 证据和算术误差结果。

SVDQuant 的 rank16 满足官方融合的 rank 对齐要求，但官方 activation-quantize/down-fusion 按 K/64 分组计算 activation scales；冻结适配 checkpoint 的 activation 是全 K 的 per-token 单 scale。直接替换会改变量化函数。当前 fused 候选保留原 A4 scale/codes，使用官方 helper 的 LoRA-down 输出及 GEMM 的 LoRA-up epilogue；helper 自带的另一套 A4 输出丢弃。它仍承担这部分重复 quantization 开销，且 FP16 factor/operand 路径与 reference 的 rounding 差异必须经过 gate，尚不能称为已验证的完整官方 engine 移植。

`25976174.pbs101` 的第一次 all-native primitive probe 已明确失败，Nunchaku adapter-packed GEMM 的 max-abs error 为 373.274。根因定位到 task-local activation-scale packer：每个 32-row warp 只有 16 个有效 half2 lane，destination stride 错用 32，导致写出 scale buffer。已改为官方布局的 16，并同时安排 core/fused 两个 CPU rebuild（25977147/25977151）；新 probe 25977166 依赖两者成功，先检查 layout 和 one-hot/random GEMM，再执行全部 primitive tests。修复通过 GPU 实测之前，不把这些 native 路径计为数值有效。

官方资料：[QuaRot kernel](https://github.com/spcl/QuaRot/blob/main/quarot/kernels/gemm.cu)、[SpinQuant ExecuTorch 导出](https://github.com/facebookresearch/SpinQuant#3-export-to-executorch)、[Nunchaku](https://github.com/nunchux-ai/nunchaku)。

重测固定六臂、batch1、五 NFE、80 queries × 9 repeats = 每臂每口径 720 samples，硬件要求相同 A100-SXM4-40GB、SM80、Torch/CUDA 环境；按用户澄清，不要求同一物理 GPU。不同 UUID 的完整有效样本可续跑和汇总，保留每条样本的实际 UUID/host/PBS producer，报告实际设备列表。分别测真实 sampler 与固定输入 DiT replay；后者将非 DiT 模块 offload，并给所有臂相同的通用 CUDA Graph 计时条件，不会把我们的 VQ engine 装给 baseline。Peak 按实际 allocated bytes，Storage 按实际 unique packed tensor payload（含 padding/scale/transform）统计。完整门槛通过前，不发布新的 native speedup 结论。

正式 panel 已接入单次 runtime load 的 driver，先完成六臂 sampler，再 offload 非 DiT 模块并完成六臂 DiT replay。每臂结束恢复 BF16、释放其 graph/packed/rotation 引用，保留 CPU BF16 shadow；样本数和 measurement workers 保持原协议。149 项 stdlib tests 已通过，尚无该 driver 的完整 GPU panel 结果。

ViDiT adapter 后续修正了 affine correction 的 G/N 布局，把原先逐 forward 解包整块 W 计算的静态项移到 bank 安装阶段，生产 forward 不再持有这些 dense weight 临时量。其 H128 复用已编译的官方 FHT；新增静态 correction tensors 纳入实际 packed payload。新 adapter 版本为 `cosmos_viditq_atom_w4a4_g128_native_h128_precomputed_affine_v2`，仍待 GPU fidelity gate。

capture `25977017.pbs101` 最终为 PBS F/-29（30:22），manifest 仍为 0/80；不能依据 wrapper 的错误 success marker 认定成功。后续 wrapper 已补 TERM/INT 非零退出处理。CPU-only import 诊断 `25977419.pbs101` 与启动阶段定时 Python stack trace 用来定位长启动；不把容器/import 等待计入 inference latency。最近完整 stdlib suite 为 162 项通过；其后 gate-resume 硬件 receipt 修正的定向 39 项测试通过。

后续 capture `25977542.pbs101` 已完整生成原 80 queries 的真实五次 DiT forward 输入/输出，PBS F/0（18:33），无需重跑。CPU import 诊断无模型 forward，观测到 Torch import 约 245 s，LIBERO evaluator 及其依赖 import 约 1522 s；不同 allocation 的启动耗时会变化。Gate 和正式 panel 都已改成每个 worker 只加载一次 runtime。

`25978617.pbs101` 已通过全部 native primitive GPU tests（F/0，1:46）：QuaRot 12 shapes 为整数精确一致，official FHT 与 fused scaled-add 通过，Nunchaku core 与 ViDiT general core 均在预先固定的低精度容差内。这个结果仅证明算子实现，尚不替代完整 checkpoint 的 full80 gate 或正式性能表。

Atom 最终需要两处适配：A-scale 存的是每个 lane 的 `(row r, row r+8)` half2 pair 的四次复制；dequant 还必须为相邻 N 列分别使用 scale。独立 general module 保留官方 INT4 MMA、load/store，只把 c0/c1/c2/c3 的 scale 改为对应两行×两列的四个乘积。GPU 对照中，原 module 的两个独立 weight-scale group 最大误差为 84/60，general module 对应误差均为 0；正确 paired-half2 和 production v4 A-scale 测试也均为 0。MMA fragment 的行列对应依据 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/archive/11.6.0/parallel-thread-execution/index.html)，原实现见 [pinned Atom source](https://github.com/thu-nics/ViDiT-Q/blob/43657d11b32192ec0f50f0dac8d3a16c3d841196/kernels/csrc/qgemm/w4a4/atom.cu)。这些是适配冻结 G128/逐输出列 scale 的实现修正，没有重新量化权重或训练。最近完整 stdlib suite 为 191 项通过。
