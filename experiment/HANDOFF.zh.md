# Microscaling VQ WAM 与 inference engine：完整技术交接

初稿日期：2026-10-26；最近增补：2026-10-27（Asia/Singapore）。当前覆盖 **85 个 idea-validation 目录和 2 个 reproduction 目录，共 87 个目录**。本次更新 AQLM fixed-input/停止后的闭环分析，并新增标准 LIBERO 400-episode engine 评测；其余历史实验沿用初稿证据快照。本稿由 Codex 亲自阅读本地实现、协议、完成 receipts 与报告后整理；没有新增训练、模型推理、环境 episode 或统计实验。它是内部技术交接，不是已经过作者审核的投稿稿件。

## 1. 接手者首先需要知道什么

我们研究的是 **WAM backbone weights 的低比特 vector quantization**：在尽量保留动作与想象未来的前提下压缩模型，并提供真正降低部署显存、控制在线开销的 inference engine。这里的 VQ 不是 action tokenizer，也不是把 latent/action 先离散成行为词表。

当前算法由三部分组成：

1. **表示与预处理**：冻结 Smooth0.5、signed Hadamard-128 与学习后的局部 R128；weight 使用 d4、两个 K256 additive codebooks、BF16 row gains、B32 power-of-two exponents；activation 使用 dynamic per-token scalar A4。
2. **任务末端校准目标**：保留原始 BF16 teacher 的 motor action、clean future latent，并增加冻结 BF16 future-to-action reader 的 soft-bin KL，即 A+Z+reader-KL。只训练压缩表示的 codebooks/log row gains，底模、indices、exponents、R0 均冻结。
3. **部署**：压缩编码常驻，每层在线解码当前 BF16 weight 到共享 scratch，再用 cuBLAS BF16 Linear；结合条件缓存、稳定 workspace、手动 CUDA Graph 与 shape-tuned decoder。

**当前 Fast-WAM 算法参考是 `25921795.pbs101 / reader_kl / final8`，简称 Q0。** 严格配对 LIBERO-Plus 闭环 A+Z 为 204/280，A+Z+reader-KL 为 208/280；BF16 为 216/280。推荐 runtime 是 **C+decoder+buffer**，同 Q0，完整 80-query action 与原量化路径差异为 0，独立 action/future anchor 也为 0；policy-call p50 0.887096 s，query peak allocated 16.914255 GB。

**当前跨模型结果已不止旧 Cosmos C8 A+Z。** 最新 Cosmos 六臂中的 ours 已使用 C80、R8→VQ8、joint-DiT suffix reader-KL 与 engine，闭环 230/280，BF16 233/280。它没有 matched A+Z 消融，不能将变化单独归因 reader-KL。最新 native-performance 表还存在 mixed measurement designs 与 baseline fidelity failed/missing 的限制。

**2026-10-27 新增两项结果。** 同一 Q0 配合 C+decoder+buffer 在标准 LIBERO 完成 **391/400（97.75%）**，四卡完整 action/future anchor 差异均为 0；这是原始 LIBERO、每 task 前 10 个官方 initial states 的一轮评测，与此前 LIBERO-Plus 208/280 是不同面板，不能互换。

**AQLM loss-pair 已得到负结果，并按用户指令停止后续实验。** Fixed-input 两臂均完成 final8；held-out motor RMSE 为 reconstruction **0.051755376**、ours **0.116873852**，future NMSE 为 **0.069989021 / 0.239592169**。后续闭环 reconstruction 完成 280、ours 完成 245，共保留 525 条；共同 245 条中为 **179/245 vs 125/245**，ours 5 胜/59 负/181 平。原 560 目标未完成，缺少的 35 条不计失败；监控和自动恢复已停止。原 owner chat 为 `01a17756-c858-7142-b39a-619f26b9eae0`。旧状态的 `active`、`remaining_required_work` 或 `next_action` 不能覆盖最新用户停止指令。

核心证据：[Fast-WAM matched 闭环](idea-validation/fastwam-vq-r128-c80-loss-pair-plus/RESULTS.zh.md)、[最终 engine](idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/RESULTS.zh.md)、[最新 Cosmos](idea-validation/cosmos-policy-vqwam-20261024/INTERIM_RESULTS.zh.md)、[AQLM fixed-input](idea-validation/fastwam-aqlm-loss-pair-20261026/pulled/26008761.pbs101/results/report.zh.md)、[AQLM 停止后分析](idea-validation/fastwam-aqlm-loss-pair-plus-20261026/STOPPED_ANALYSIS.zh.md)、[标准 LIBERO 完整结果](idea-validation/fastwam-libero-engine-20261027/RESULTS.zh.md)。

## 2. 这套交接材料怎样阅读

| 文档 | 内容 |
|---|---|
| 本文 | 研究目标、实际表示/损失/训练/部署、数据合同、主要结果、实现地图、仍缺的证据 |
| [01 基础与 baselines](handoff/01_FOUNDATIONS_AND_BASELINES.zh.md) | 13 个早期诊断/基线目录及两个 reproduction；说明实验为何开始、为何停止 |
| [02 VQ 与 latent](handoff/02_VQ_AND_LATENT.zh.md) | 16 个表示、蒸馏、future-reader、NMSE-matched、容量/梯度机制实验 |
| [03 Scaling、clipping 与数据](handoff/03_SCALING_CLIPPING_AND_DATA.zh.md) | 17 个数据覆盖、activation 参数、visited states 与 gripper 实验 |
| [04 KL 与优化](handoff/04_KL_AND_OPTIMIZATION.zh.md) | 16 个 KL 设计、有限步/梯度诊断及未提交方案 |
| [05 R128、matched loss 与 AQLM](handoff/05_R128_MATCHED_LOSSES_AND_AQLM.zh.md) | 13 个当前主模型的前驱、联合优化、Q-Clip 与 AQLM 负结果/停止后分析 |
| [06 Engine、标准 LIBERO 与 Cosmos](handoff/06_ENGINE_AND_COSMOS.zh.md) | 6 个 runtime 阶段、标准 LIBERO 400 episodes、3 批 Cosmos 结果及精度/性能边界 |
| [07 接手操作手册](handoff/07_OPERATIONS.zh.md) | 环境、数据/模型位置、checkpoint 来源、只读命令、恢复顺序、完成标准、常见错误 |
| [HANDOFF_INDEX.json](HANDOFF_INDEX.json) | 87 个目录到逐项说明和原始证据入口的机器可读映射 |

每个逐项条目都应连同原始报告阅读。根 README 经常保留“刚提交/正在等待”的历史正文，优先级是：明确的用户范围修订 → 对应阶段的最终结论/完整 receipt → 同 job 的 summary/records → 实际 artifact 源码 → 初始 protocol/README。不能只凭文件修改时间或单个 `complete` 字段判定。

## 3. 模型与任务的实际边界

### 3.1 Fast-WAM 主平台

主线使用 released Optional-IDM clean checkpoint `libero_optional_idm_2cam224.clean.pt`，上游 FastWAM revision `7faa71108368fbb3b6885649f112af607427a2d4`。不要将这个模型、`first_frame` 路径、`idm` 路径和其他 FastWAMJoint 变体混称为同一实验。

当前主结果是 `idm`：先 10 次 video denoising 得到 imagined future，准备 video K/V，再 10 次 action denoising，输出 32×7 normalized action chunk。闭环通常执行前 10 个 actions 后 replan；第 7 维 gripper 单独走部署后处理。当前 teacher/reader 的 clean latent 形状为 `[1,48,3,14,28]`，frame0 是 observed clamp，frame1/2 是预测 future；14×28 中包含双相机布局。9 个 video frames 与 3 个 latent time positions 是不同层级，不能互相替换。

量化集合为 **614 个 canonical Linear / 6,016,317,440 个 weight values**：video 306、action 307、root proprio encoder 1。Bias、normalization、nonlinearities、text encoder、VAE、KV 等不因这组 weight 编码自动变成低精度。未量化 text encoder 与 VAE 是 whole-model payload 的主要部分。

`first_frame` 没有 future-video generation，历史 profiling/cache/dither 多用这个模式；它不是主算法的廉价等价替代。实验同样叫“Fast-WAM”，并不保证 mode、steps、precision、calibration 和计时范围相同。

代码证据：[frozen_reader.py](idea-validation/fastwam-vq-c80-frozen-reader-kl/frozen_reader.py)、[主协议](idea-validation/fastwam-vq-r128-c80-loss-pair/protocol.json)、[基础 smoke](reproduction/fastwam-smoke/RESULT.md)。

### 3.2 Cosmos Policy 跨模型平台

使用 `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` 的官方 direct-policy 方式，不是 Cosmos3-Edge。它把 action、proprio、future/value 等放进一个 shared video DiT 的 latent slots；没有独立的 Fast-WAM ActionDiT。

当前配置 5 NFE、16-step action chunk、每 replan 执行 16 步，双相机 224，官方 normalization/JPEG/crop/flip，非 planning mode。量化 280 个 transformer attention/MLP Linear，共 1,761,607,680 target weights；patch/final/time/conditioning projection 等有明确排除。Action 从 slot4 的重复体积还原并平均；future-image slots 为 6/7。不能把 slot4 的 latent MSE 当作最终 action MSE，也不能把 direct-policy success 当成 planning 结果。

代码证据：[Cosmos protocol](idea-validation/cosmos-policy-vqwam-20261024/protocol.json)、[runtime.py](idea-validation/cosmos-policy-vqwam-20261024/runtime.py)、[reader.py](idea-validation/cosmos-policy-vqwam-20261024/reader.py)。

## 4. Microscaling additive VQ 的精确定义

### 4.1 Linear 的预处理坐标

对 `Y=XW^T+b`，记冻结正对角 Smooth scales 为 S，局部正交变换为 R。实数配对形式是：

`X_t = X S^{-1} R`，`W_t = W S R`，因此 `X_t W_t^T = XW^T`。

R 在实际实现里包含保存的 signs、H128，再接 stream-shared Cayley residual U；不能穿过上游 norm、timestep modulation 或 nonlinearities。H、U 的原 FP32 运算和 BF16 cast 顺序在部署中有意义，实数等价不保证改写 kernel 后 bitwise 等价。尾宽不足 128 时保留既有小块变换；不要假设所有层都能补齐后随意换同一种矩阵。

Cayley 参数化：`K=A−A^T`，`U=solve(I−0.5K, I+0.5K)`，R 的完整 blocks 为 H128 后 U。Fast-WAM 主路径只有合格的 video/action stream 共享 128×128 residual，proprio 的小 input width 不因此拥有第三个同形 R。当前 Q0 中 U/R0 已冻结，独立存储约 131,072 bytes。

证据：[rotation_bank.py](idea-validation/fastwam-vq-learned-r128-c8/rotation_bank.py)。Joint 同时更新 R 但不补偿 weight 的历史版本会改变函数，详见 [R/VQ诊断](handoff/05_R128_MATCHED_LOSSES_AND_AQLM.zh.md#fastwam-vq-r128-c80-dual)。

### 4.2 编码、解码与开销

每层有两本 `C0,C1 ∈ BF16[256,4]`，沿 input dimension 每 4 个 weights 一个 vector，保存两个 uint8 indices。对 output row r、vector v：

`W_hat[r,4v:4v+4] = BF16( (FP32(C0[i0]) + FP32(C1[i1])) × S_block[r,floor(v/8)] )`。

每 32 个 input-channel weights 共享一个 block scale，即 8 个 d4 vectors；`S_block = ldexp(FP32(row_gain), stored_exponent−127)`。`row_gain`为 BF16，exponent 存 uint8，合法真实 exponent 范围−127..127，编码 255 保留且检查拒绝。Tail padding 按实际 shape 裁回，零 row gain、尾 block 有效长度都在 fitter 里处理。

初始 block scales 由 row RMS 与 block RMS 的 log2 比值 round 确定；拟合度量包含真实 scale²与校准 A4 输入 second moment。之后真实 sampler 目标微调 books/log gains，通常不再重分配 indices 或改 B32 exponents。训练用 FP32 masters，forward 用实际 BF16 stored values；不是 FP32 books 训练完最后才统一量化一次。

仅 indices 为 `(2×8bits)/4 = 4bpw`；B32 的 uint8 exponents 额外 0.25 bpw，再加 books、row gains、padding、Smooth/sign/U。当前 target 含 transforms 的实际约 **4.271052 bpw**。B32 不是 vector length32；codebook vector length 是 4。两 uint8indices 也不是两个 INT4 weights，不能直接交给通用 S4×S4 Tensor Core GEMM。

实现证据：[micro_codebook.py](idea-validation/fastwam-vq-microscale/micro_codebook.py)、[存储实账](idea-validation/fastwam-vq-r128-c80-inference-profile/EXPORT_STORAGE.zh.md)。

### 4.3 Activation 与训练导数

当前主路径的 activation 是逐 token row 的 signed symmetric scalar A4：`scale=max(abs(z))/7`，`q=clamp(round(z/scale),−7,7)`，输出 BF16 的`q×scale`；零行有安全分支。它没有在线 activation VQ/codebook 搜索。

训练的 A4 backward 为既定 STE；codebook/gain 的 FP32 master 通过 stored-BF16 decoder 反传。STE/VJP 连通只是约定计算图里的导数，不是硬 round、BF16 格点及整个 closed-loop 系统的精确导数。过去多项有限步诊断已看到 g·delta 预测下降而真实 forward 上升，不能因为梯度非零就声称该目标已被优化好。

Clipping/per-channel 路线另有 LSQ、absmax coupling 与 D compensation，但它们不属于当前 Q0；不能把其 1.8M D 参数、ratio 边界或梯度规则混到 Q0 训练里。

## 5. A+Z+reader-KL 的损失与梯度实现

### 5.1 原始两个目标

同一 observation、prompt/proprio 与 sampler seed 下，teacher 输出 `(a_T,z_T)`，student 输出 `(a_Q,z_Q)`：

`A = mean((a_Q[:10,:6] − a_T[:10,:6])²)`。

`Z = sum((z_Q[:,:,1:] − z_T[:,:,1:])²) / max(sum(z_T[:,:,1:]²),1e−12)`。

Fast-WAM 当前 `L_AZ=A+lambda×Z`，`lambda=0.04245452683169643 = rho1.5×c0.028303017887797623`。该 c 沿用先前冻结 C8 定义，不在 R128 loss-pair 每个 batch 或 held-out 重新估计。A 是 motor6，不含 gripper；仅 first 10 直接监督，但 student action 通过自身 future 的路径保留，Z 也不比较真实环境未来。

代码证据：[reader_loss.py](idea-validation/fastwam-vq-c80-frozen-reader-kl/reader_loss.py)、[worker.py](idea-validation/fastwam-vq-r128-c80-loss-pair/worker.py)。

### 5.2 Reader KL 的含义

Frozen BF16 reader 记为 `R_T(o,z)`，包含原 video prepare/prefill 与完整 action denoising，接受 student clean future；observed frame0 仍由真实 input clamp。Teacher future 重放要求 `R_T(o,z_T)=a_T`。Reader 参数不训练，但 future 输入保留梯度。

将 reader 输出的每个 motor scalar 按固定 C80 每 motor 坐标 mean/std 标准化。当前统计每坐标 800 值（80 contexts×10 steps），不是为每个 time position 分别拟合一个分布；std floor0.02。用 Gaussian-CDF 构造 16-bin 概率，sigma0.375，内边界 15 个点覆盖[−3.5,3.5]，两端无限，再算逐位置 `KL(p_teacher || p_reader_student)` 并平均。

`L = A + lambda Z + beta K_reader`，`beta=0.006083668576461056`。规则是初始完整 C80 上 `beta = 0.1×lambda×mean(Z_initial)/mean(K_initial)`，一次冻结后不再调。10%是 loss 数值尺度，不是 10%的参数梯度或最优权重。

这项 KL 描述平滑 scalar 读出的一致性，不是原 flow policy 原生 probability density、mutual information 估计，也不是完整 action–latent 联合密度。它将 future 质量与一个实际控制读出联系起来，但不会自动解决量化 student reader 本身的误差。

代码证据：[frozen_reader.py](idea-validation/fastwam-vq-c80-frozen-reader-kl/frozen_reader.py)、[softbin_loss.py](idea-validation/fastwam-vq-dual-kl/softbin_loss.py)、[冻结statistics与beta规则](idea-validation/fastwam-vq-r128-c80-loss-pair/protocol.json)。

### 5.3 为什么分阶段 VJP

不能同时持有完整 student 和 teacher-reader 的所有反向激活。主 worker 按每个 10-context batch 执行：

1. 当前 student 先 no-grad 得到 action/future，缓存输出，在 FP32 叶子上求 A/Z 对输出的 cotangents。
2. 进入`reader_session`，临时 park/恢复相关 banks/weights，用原 BF16 reader 分别读这些 future，求 beta×reader-KL 对 future 的 cotangent，检查 finite、非零且 frame0 梯度严格 0。
3. 恢复 student，不改变参数/input/noise，grad-enabled 重放；完整 action/future 与第一遍要求逐值一致。把原 A/Z 与 reader 的 future cotangents 在 FP32 合并，边界只 cast 一次。
4. 对 10 个 contexts 累积各自梯度的 1/10，再做一次 norm clip1、Adam step；第 1 个 update 后清 VQ optimizer moments。每次保存 masters、optimizer、RNG、cursor 与 lineage。

因此“80 exposures”是训练数据预算，不等于 80 次完整 forward。KL 臂有 preparation、student replay 和 reader-only 额外调用。训练 receipt 中的 1228 是参数 Tensor 数（614 组 books/gains），不是 1228 个 scalar 参数。

证据：[worker.py 的 preparation/kl_batch/train](idea-validation/fastwam-vq-r128-c80-loss-pair/worker.py)、[reader_runtime.py](idea-validation/fastwam-vq-c80-frozen-reader-kl/reader_runtime.py)。Reader 只用于训练/诊断，不在 inference bundle 内，也不让部署 sampler 多跑 reader。

## 6. 当前主模型的来源链与冻结合同

| 阶段 | 权威来源 | 产物与作用 |
|---|---|---|
| 数据与 teacher | `fastwam-vq-calibration-suites`、`fastwam-vq-c80-activation-clipping` | C80/held-out plan、同 seed BF16 actions/futures；后续复用 |
| W16A4 R0 | `25905203 / coverage_c80 / C80_once_lr1e4` | C80、8×10 平均梯度、R LR1e-4、final8；当前固定 R0 |
| R0 下 VQ0 | `25912619 / fastwam-vq-r128-c80-dual` | 614 共同 banks、indices/exponents、source/repro receipts；matched 两臂共同起点 |
| 两臂 loss | `25921795 / fastwam-vq-r128-c80-loss-pair` | 同 R0/VQ0、C80、8 updates/80 exposures、fixed final8；只差 readerKL |
| 闭环比较 | `25925636 / loss-pair-plus` | AZ204/280、KL208/280，15 组 reference 共 4200 slots |
| 导出 | `inference-profile / export` | Q0 inference-only bundle；不含训练 masters/reader |
| 最终 runtime | `25959865 decoder`、`25960283 buffer`、CPU`25961057` | 同一 Q0，80action 精确 parity、独立 future anchor；最终推荐 C+decoder+buffer |

不要把 R128 dual 的 Seq8 作为 matched loss-pair 的预训练起点；两者用的是 dual 里共享**未微调 VQ0**。不要把 Q-Clip 或旧 C80 clip-only 替换成 Q0，因为两者恰好 207 成功就混用来源。实际远端路径及恢复顺序见 [操作手册](handoff/07_OPERATIONS.zh.md)。

## 7. Inference engine 的实际数据流

每次新 query 先建立 prepared GPU datum，处理当前 text/image-latent/condition。对目标 Linear：固定 Smooth/sign/H128/U 产生 FP32 变换，dynamicA4 输出到独立 BF16 buffer；对应编码在 GPU 上常驻，当前 weight 由 shape-tuned Triton decoder 写入共享 BF16 scratch；cuBLAS 消费 BF16 activation 与 BF16 weight，bias 保持原语义。之后 scratch 可被下一层复用。

query 内不变的 text embeddings、condition projections 和各层 cross K/V 缓存起来；因为含当前 proprio 和 observation，每次 query 重建/失效。该优化使 614 种目标模块的动态调用数从 6436 降到 5258/query；模块数和调用数是两回事。

三个手动 Graph 分别记录 video step、video prefill、action step。固定 geometry/地址下刷新输入 buffers 再 replay，不能复用上个 query 的内容或把 Tensor 地址变化后仍当相同 Graph。Graph 减少 host launch 成本，没有减少 denoising steps，也没有把 BF16 GEMM 变成低比特 GEMM。

最终 decoder 保持 FP32 book sum、power-of-two scaling 和 BF16 写出顺序；H/U 可借用 weight scratch，因为在 weight decode 前已不再使用，A4 operand 不能借同一片即将被覆盖的 scratch。当前只验证单 request/单 CUDAstream；跨请求并发、可变 shapes 或多 stream 必须重新设计 workspace 与 Graph ownership。

### 7.1 阶段结果

| Runtime | policy-call p50 s | Query peak allocated GB | 数值状态 |
|---|---:|---:|---|
| 历史 BF16 reference | 1.450439 | 25.130502 | 原始 BF16；未做同等 cache/Graph 优化 |
| Q0 eager dense | 6.527626 | 25.139422 | 原量化参考 |
| Input-fused dense | 3.503950 | 25.139422 | full 80 action 精确 |
| Layer-decode FP32 input | 3.532909 | 16.550766 | full 80 action 精确 |
| B：workspace+condition cache | 3.047032 | 16.688547 | full 80 action 精确 |
| C：B+CUDA Graph | 0.956914 | 16.949808 | full 80 action 精确 |
| C+decoder+buffer | **0.887096** | **16.914255** | full 80 action0；独立 action/future anchor0 |

计时从 prepared GPU datum 到 infer_action 返回并同步，包含内部 text/image-latent 编码及 denoising，排除外部 observation preprocessing、环境 step 和机器人运动；batch1、32×7、9 frames、10+10 steps、3 warmups。主 timing 没有 profiler，load/setup/capture 另报。

2026-10-27 又用该 engine 完成标准 LIBERO 400 episodes：391 成功、6415 次 warm policy calls，p50 **0.899527 s**。这是闭环访问 states 上的实际执行计时，不覆盖同一固定 80-query 输入面板；因此不替换上表 0.887096 s，也不声称两次 p50 差异由算法改变造成。该轮没有同设置 BF16 效率对照。

历史参考有跨 allocation 复用。约 1.64×相对上述 BF16 是本地 runtime 组合的描述，不是公平隔离 VQ 算法贡献，也不证明优于同样优化过的 BF16。通用 packed fused GEMM 更慢且改变输出，BF16 U/input 转换也失败；torch.compile 没有有效结果。不能只展示成功路线而丢掉这些失败边界。

### 7.2 存储、内存和未量化部分

| 项目 | 当前实账 |
|---|---:|
| Target encoding 含 Smooth/sign/U | 2.991409 GiB |
| 非量化 model parameters | 11.902288 GiB |
| Whole bundle unique tensor payload | 14.893702 GiB |
| 实际 bundle archive | 14.895614 GiB |
| Original dense whole payload | 23.108552 GiB |
| 最终 query peak allocated | 16.914255 GB（10^9） |
| 共享 weight scratch | 113,246,208 bytes |

非量化部分中 text encoder 约 10.581520 GiB，VAE 约 1.312585 GiB。State_dict 含 aliases，必须按底层 storage 去重；logical keys 相加会重复计数。Archive 大小不包括全部外部 cache/tokenizer assets。Allocated、reserved、nvidia-smi used、tensor payload、archive bytes 不能互换。

证据：[engine最终报告](idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/RESULTS.zh.md)、[ABC组件](idea-validation/fastwam-vq-r128-c80-runtime-abc/RESULTS.zh.md)、[export ledger](idea-validation/fastwam-vq-r128-c80-inference-profile/EXPORT_STORAGE.zh.md)。

## 8. 数据、cohort、指标与公平条件

### 8.1 不能混用的几个面板

| 面板 | 实际组成 | 用途与限制 |
|---|---|---|
| 最早 Smooth/VQ | Spatial22 观察内拆 C8、selection4、test10 | 10-case test 不是后来 22-case test |
| 旧 action-aware/latent | 8cal、4selection×2、22test×2 | Spatial Original10+Plus12；已多次查看，是机制/recovery 集合 |
| 四 suite C8 | 2000/2019/2021/2025/2051/2057/2066/2070，各 seed0 | 8 tasks，phase2early/5middle/1late，C80 子集 |
| C80 | 2000..2079，40 tasks×2distinctdemos，seed0 | 每 suite20，phase 合计 28/28/24；不是 80 独立 tasks |
| 40held-out | 3000..3039，每 task 第三个 demo，2sampler seeds | demonstration 分离，不是 unseen-task；已反复工程分析 |
| Visited40 | 原 C80 奇数 case 的 student 短程可达 states | 40states×5inputs 不是 200 独立 states；不自动进入训练 |
| 当前 Plus 闭环 | 4 suites×7dimensions×10variants=280 | 每 variantstate0、一个配对 trial；不是整个 LIBERO-Plus |
| 标准 LIBERO engine | 4 suites×10 tasks×10 official initial states=400 | state 索引 0–9、环境/sampler seed42；无 Plus 扰动，未重校准 |
| AQLM 停止后共同子集 | reconstruction280、ours245；共同 245 | 前三 suites 各 70、Long35；不是完整 280 配对，也不是四 suite 等权分数 |
| 旧大 pilot | 原 1400variants×4 arms=5600 目标 | 实际只有明确 partial 完成，不能冒充当前 280 全量结果 |
| AgileX/Pepper 原始数据 | 根`agilex pepper training data/`两 runs | 本地整目录 ignore；尚未在本交接里验证 schema/标注/机器人评估 |

C80 的 sampler seed index0 不是每个 case 同一个数值 seed；真实 sampler seed 由 plan row 提供。不能重新 random seed 后仍称冻结 paired 比较。Calibration observations 可跨模型复用，teacher actions/futures 必须由各自模型生成，不能将 Fast-WAM cache 当 Cosmos teacher。

### 8.2 指标计算

First10 motor RMSE 是每 query 对 60normalized scalars 先 MSE 再开根；两个 seeds 先在 case 内平均指标，再 case→suite→四 suite 等权。**不能把宏平均 RMSE 平方还原训练 MSE**。某些历史旧记录只有逐 query RMSE，允许逐 query 平方后的有限精度复算，但必须注明来源，不能宏平均后平方。

Future NMSE 是排除 observed frame0 后的 error energy/teacher energy；不是对所有元素先各自除 teacher 值，也不是真实 future prediction accuracy。Full32/late 22/gripper 是独立诊断；未进入 loss 就不能假装直接监督过。

Gripper 需区分 continuous scalar、denormalize、Fast-WAM 符号翻转和 sign command；漏 open、额外 open、方向事件序列、switch 位置不一致不能互换。Cosmos 遵循自身原第 7 维规则，不加 OpenVLA 反转。Action query throughput 也不是控制频率；horizon 执行与环境/机器人耗时另算。

Closed-loop 有效 timeout 算 policy failure；import/render/NaN/load/资源错误算缺失或基础设施失败，不能硬计为失败 episode 补足分母。Wins/losses 按同一 slot 双方成败，ties 含共同成功和共同失败。相同总分不等于相同成功集合。

### 8.3 公平性底线

比较时登记共同初始化、checkpoint、量化覆盖、effective bytes、calibration tasks/contexts、updates/exposures、optimizer/reset、sampler seeds、mode/steps、部署 backend 和 timing scope。C8 与 C80 常同时改变 coverage、LR、batching 和 updates，不能都解释为单纯数据量效应。Paired inference seeds 不是独立训练 repeats；本地单次 cohort 差异不自动构成显著性、SOTA 或 unseen-task 泛化。

## 9. 已完成的主要结果应怎样理解

Fast-WAM 同 280 面板的关键闭环如下；各行是完整 recipe，除明确 matched loss 外，不解释为单因素因果比较。

| 配置 | 成功/280 | 主要出处 |
|---|---:|---|
| BF16 | 216 | rotation-baselines |
| QuaRot/SpinQuant adapted | 195 / 203 | rotation-baselines |
| Smooth/H Scalar | 200 | smooth-scalar-plus |
| Row/Micro VQ | 202 / 204 | vq-plus |
| C8 A+Z rho0.1 / rho1.5 | 201 / 206 | c8-action-latent-plus；updates 不同 |
| 旧 C80 clip-only / joint | 207 / 202 | c80-clip-joint-plus |
| Per-channel / softbin / relational | 203 / 199 / 198 | 各自 plus |
| Adapted SVDQuant / ViDiT-Q | 108 / 203 | diffusion-baselines；非官方完整 recipe |
| H-W16A4 / C80 R0-W16A4 | 204 / 203 | r128-w16a4-plus |
| Matched R0-MXVQ AZ / AZ+readerKL | **204 / 208** | loss-pair-plus；当前核心 ablation |
| Q0+Q-Clip | 207 | qclip-plus；不替换 208 的 Q0 |

这些结果共同支持的研究叙事是：平衡 activation 与 weight 表示、用任务末端 A/Z/reader 约束压缩损伤，并提供 memory-efficient 执行路径。它们也说明更低 local proxy、future NMSE、motor RMSE、KL 或更多训练 loss 下降，都不保证 closed-loopsuccess 提升。

负结果已经限制了一些更强的机制 claim：NMSE-matched 未稳健支持额外误差结构价值；codeword conflict 与 signed tile allocation 未证明需要新的 assignment；Joint 补偿只收回 48.488%gap 未通过 50%gate；全量 KL 替换、per-channel、Q-Clip 没有成为更强 closed-loopincumbent。写 paper 时保留这些边界，不能删掉它们后声称所有改进都单调有效。

### 9.1 标准 LIBERO：当前 checkpoint 与 engine 的实际闭环

`26020749.pbs101 / fwlib400` 完成 **391/400（97.75%）**；四个 suite 各 100，均为原始官方 initial states 索引 0–9、固定环境与 sampler seed42。

| Suite | 成功/episodes | 成功率 |
|---|---:|---:|
| Spatial | 98/100 | 98.00% |
| Object | 99/100 | 99.00% |
| Goal | 97/100 | 97.00% |
| Long (`libero_10`) | 97/100 | 97.00% |
| 合计/四 suite 等权平均 | **391/400** | **97.75%** |

本轮复用 Q0 `25921795 / reader_kl / final8` 与 C+decoder+buffer，新增校准/训练 updates 为 0。四个常驻 GPU workers 全部通过完整 action/future anchor 的精确比较；400 个唯一 slots、40 tasks、四个 worker exits、产物及 PBS F/0、PIPELINE_COMPLETE 完整。本地最终 monitor 保存于 **2026-10-27 15:45:38 SGT**，实际 walltime 01:22:44，监控已停止。

它补上了“推荐 engine 实际运行闭环”的一项证据。它没有补上同 400 slots 的 BF16 对照、engine on/off 配对闭环、三 training/evaluation seeds 或完整官方 2000-episode 结果。C80 与评测覆盖相同 40 tasks，但 calibration 来自 demonstrations 的 snapshots；这不是 unseen-task 泛化。标准 LIBERO 391/400、Plus 208/280、旧 fixed-input 80-query profiling 必须分别报告。

证据：[完整结果](idea-validation/fastwam-libero-engine-20261027/RESULTS.zh.md)、[400-episode summary](idea-validation/fastwam-libero-engine-20261027/pulled/26020749.pbs101/summary.json)、[当前完成状态快照](idea-validation/fastwam-libero-engine-20261027/monitor_status.json)。

### 9.2 AQLM：loss 迁移消融的负结果

两臂共用官方 AQEngine 的 Fast-WAM adaptation 初始化：614 banks，其中 22 heavy/12 light/580 minimal；g8/M4/K256，含共同 preprocessing 的 target effective **4.031080 bpw**。两臂同 C80、batch10、8 updates/80 exposures、LR5e-4、fixed final8，只改变训练 loss。该 bpw 与当前 MXVQ 的 4.271052 并非精确相同，跨 quantizer 不能当作完全同表示的单因素比较；本实验的严格对照是 AQLM 内部两臂。

| Held-out（40 contexts×2 seeds） | Reconstruction | Ours | Ours 相对变化 |
|---|---:|---:|---:|
| first10 motor RMSE | 0.051755376 | 0.116873852 | +125.82% |
| future NMSE | 0.069989021 | 0.239592169 | +242.33% |
| 共同 mixed=A+lambda Z，不含 KL | 0.005762744 | 0.025813018 | +347.93% |

三项均 0/40 context 胜。完整 C80 从初始化到 ours final8，motor RMSE **0.048906→0.118843**、future NMSE **0.057176→0.250011**，80/80 mixed loss 变差，说明训练拟合本身已失败，不能仅称 held-out 过拟合。

后续 Plus 闭环按用户要求在 525/560 条时停止：reconstruction 完整 **202/280**；共同 245 条为 reconstruction **179/245（73.06%）**、ours **125/245（51.02%）**，ours 5 胜/59 负/181 平，差 **−22.04 pp**。Long 只完成 35 个配对 episodes，不能写成完整 280 配对结果或四 suite 等权结果，不能把缺失 35 条硬算失败。CPU `26020010` 仅分析既有缓存，新增训练/模型 forward 均 0。

Ours codebooks 相对初始值的全局 L2 漂移 **8.903%**，reconstruction **2.057%**；第一次更新后，下一批相同 cases 的总 loss 相对其初始值升至 **6.030×**。参数步幅是优先排查的候选原因，但没有经过固定 batch 小步真实方向验证，不能宣称已定位 LR、KL 或 STE 的因果责任，也不能据此否定全部任务末端 loss。保留负结果；当前不自动补跑、训练或 sweep。

证据：[fixed-input 报告](idea-validation/fastwam-aqlm-loss-pair-20261026/pulled/26008761.pbs101/results/report.zh.md)、[停止后完整分析](idea-validation/fastwam-aqlm-loss-pair-plus-20261026/STOPPED_ANALYSIS.zh.md)、[CPU 完成 receipt](idea-validation/fastwam-aqlm-loss-pair-plus-20261026/stopped_analysis/26020010.pbs101/completion_receipt.json)。

## 10. Cosmos 的 reader、engine 与结果如何接入同一故事

Fast-WAM reader 读取 final clean future；Cosmos reader 读取第 3NFE 前的 student noisy future slots6/7，再运行真实原 sampler suffix。这是模型结构适配，不是同一个 reader 函数换名字。Teacher action slot4 只在切点处作为 detached initial state，后续自然演化；若每一步强制把 action clamp 回 teacher，会破坏 future 对 action 的可微影响。

新 Cosmos 的 lambda0.0645743742/beta1.7317817103 与 Fast-WAM 不同；action 监督完整 16×6 而不是 first 10。只有 CosmosC80 teacher 统计能用于它。80/80cotangent 连接、560 个 parameter Tensors 有梯度、staged replay0 支持训练接线，但没有 matched AZ control，不证明 KL 独立提升 230/280。

当前六臂 233/221/227/78/232/230 的旧性能表使用 eager decoded baselines 与 ours engine，不能作为公平 native benchmark。新 native 表将 sampler 和 FP-capture DiT replay 分开：BF16 N720/80 queries 与其他 N40/40 queries 混合；ours sampler435.695ms 接近 BF16432.566ms，DiT replay433.901ms 仍慢于 BF16277.356ms。四 native baselines 有 failed representative/missing full 80 fidelity，用户允许性能 report-only；不能把它们称精度 gate 全通过，更不能把旧闭环分数移植为新 backend 闭环。

Cosmos 已经证明当前方法和存储/显存方案可跨到另一 WAM 结构；还没有证明通用 WAMnative 低比特吞吐提升、所有 backends accuracy-preserving 或在多硬件/真实机器人上成立。

证据：[六臂完整报告](idea-validation/cosmos-policy-vqwam-20261024/INTERIM_RESULTS.zh.md)、[reader数学检查](idea-validation/cosmos-policy-vqwam-20261024/retrieved/reader_math.json)、[native mixed表](idea-validation/cosmos-policy-native-perf-20261025/RESULTS_mixed.zh.md)。

## 11. 实现地图：应该改哪一层

| 工作 | 首先读的实现 | 不应混淆的层级 |
|---|---|---|
| VQ 编码/尾宽/exponent/decode | [micro_codebook.py](idea-validation/fastwam-vq-microscale/micro_codebook.py) | 表示与数值语义，不是端到端训练目标 |
| H/U、Cayley、W16A4、VQBanks | [rotation_bank.py](idea-validation/fastwam-vq-learned-r128-c8/rotation_bank.py) | 旧可训练 R 路径与当前冻结 R0 不同 |
| 当前 loss/calibration/batchVJP | [loss-pair worker.py](idea-validation/fastwam-vq-r128-c80-loss-pair/worker.py) | 不含正式 closed-loopcontroller |
| Frozen FastWAM reader/session | [frozen_reader.py](idea-validation/fastwam-vq-c80-frozen-reader-kl/frozen_reader.py) · [reader_runtime.py](idea-validation/fastwam-vq-c80-frozen-reader-kl/reader_runtime.py) | 原 BF16 reader，不是 student 自身 reader |
| Gaussian-CDF KL | [softbin_loss.py](idea-validation/fastwam-vq-dual-kl/softbin_loss.py) | 真实 flow density 不在这里 |
| Closed-loop 安装/分片/聚合 | [loss-pair-plus runtime.py](idea-validation/fastwam-vq-r128-c80-loss-pair-plus/runtime.py) · [evaluate.py](idea-validation/fastwam-vq-r128-c80-loss-pair-plus/evaluate.py) | 源 checkpoint 与 manifest 绑定优先于文件名 |
| Bundle/direct loader/按层 decode | [export_runtime.py](idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/export_runtime.py) · [layer_runtime.py](idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/layer_runtime.py) | 单层 BF16 解码与 fused GEMM 是两条路线 |
| Cache/workspace/Graph | [abc_runtime.py](idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/abc_runtime.py) · [condition_cache.py](idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/condition_cache.py) · [graph_runtime.py](idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/graph_runtime.py) | input 刷新、query 失效、地址 ownership |
| Decoder 调优 | [decoder_runtime.py](idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/decoder_runtime.py) | 保持原 book sum/scaling/cast 顺序 |
| Cosmos reader/training/engine | [reader.py](idea-validation/cosmos-policy-vqwam-20261024/reader.py) · [train.py](idea-validation/cosmos-policy-vqwam-20261024/train.py) · [engine.py](idea-validation/cosmos-policy-vqwam-20261024/engine.py) | Shared DiT adapter，不是 FastWAM 代码直接复用 |
| 标准 LIBERO engine 评测 | [worker.py](idea-validation/fastwam-libero-engine-20261027/worker.py) · [analysis.py](idea-validation/fastwam-libero-engine-20261027/analysis.py) · [control.py](idea-validation/fastwam-libero-engine-20261027/control.py) | 400 个官方 initial-state slots；不使用 Plus 扰动与 280 manifest |
| AQLM 停止后只读诊断 | [stop_analysis.py](idea-validation/fastwam-aqlm-loss-pair-plus-20261026/stop_analysis.py) | 只读缓存与已有参数；不代表授权恢复模型 forward |
| 五 NFE 优化 | [engine_optimized.py](idea-validation/cosmos-policy-native-perf-20261025/engine_optimized.py) | 新 native performance 不自动继承旧闭环 |
| AQLM 表示/fit/两 loss | [aqlm_banks.py](idea-validation/fastwam-aqlm-loss-pair-20261026/aqlm_banks.py) · [worker.py](idea-validation/fastwam-aqlm-loss-pair-20261026/worker.py) | Official AQEngine 初始化与末端 loss 训练分开 |

很多 worker 通过`run.pbs`复制并重命名旧成功 job 的 artifact 源码，再设置 PYTHONPATH。它们不是彼此完全独立的 Python packages。复现原结果优先使用原 job artifact；准备新实验才使用 review 后的当前 workspace 源码。不能从文件名相同推断两者内容相同，也不能直接在本机运行 worker 来“试一下”。

## 12. 当前还缺什么，以及接手后的顺序

1. **保留 AQLM 负结果与用户停止状态。** Fixed-input 两臂已完整，闭环仅共同 245 配对；525 条记录、初始化和全部 checkpoints 只读保留。现阶段不补剩余 35 条、不重训、不新增模型 forward。若以后明确重开，先用固定 batch 分解 A/lambda Z/beta KL 梯度及小步真实 loss，验证参数步幅；这是候选诊断，尚未执行。
2. **固定算法、runtime 与两类闭环证据。** Q0 的 Plus 208/280 与标准 LIBERO engine391/400 均保留，各自引用协议。标准 400 已证明推荐 engine 实际闭环可运行，但没有同 slots BF16 或 engine on/off 对照；不能改写为已证明量化零精度损失。
3. **解决正式效率公平性。** Fast-WAM 缺同等 cache/Graph 优化 BF16，Cosmos 缺同一 measurement design/精度资格通过的 native 全表；当前不能从几张口径不同的表作 SOTA claim。
4. **独立确认泛化与训练稳定性。** 现有 held-out 多次查看，paired inference seeds 不是 training seeds；若新方法选择和 paper 结论需要强泛化，应先冻结独立 protocol，再采新证据，不能把现有 regression 面板改名为新测试集。
5. **Native VQ GEMM 仍是工程缺口。** 当前直接 decode+cuBLAS 稳定；旧通用 fused kernel 更慢且偏移。要继续必须同时考察 shape 特化、reduction/epilogue 语义、full policy parity 和实际 latency，不只看单层 microbench。
6. **真机评估仍需独立完成。** AgileX/Pepper 原始数据和任务建议报告已保存，但本交接没有验证数据 schema、训练 pipeline 或 robot 闭环。不要把文献硬件、simulation 成功或 raw data 存在当作真实机器人结果。报告见[真机任务建议](../research/real-robot/2026-10-26/2026-10-26_FastWAM_RealRobot_Task_Recommendations.docx)。

上述是待办与建议的分层，不是本次自动申请资源的清单。已停止的 bilevel/affinity/ratio-only/continuation 与旧大 pilot 不因本交接而恢复。

## 13. 文档证据与维护方式

主结论均链接到本地实现/结果，不用记忆摘要替代实验 receipt。逐项台账区分 complete、partial、cancelled/superseded、prepared-not-submitted 和仅 protocol；没有结果的项目明确保留缺口。操作手册列出的远端位置来自保存的协议/launcher，**未重新扫描远端存储或确认每个大文件今天仍在**；这是恢复时需要做的小范围检查，不应先重下载或重训。

后续只更新发生变化的实验条目、主表和当前状态，保留旧 producer、失败 attempt、用户范围修订及结果分母。不要用后来成功的后处理覆盖 GPU 失败事实；不要用未来推荐方案改写“当时实际做了什么”。

本稿采用本地`ponytail`与`scientific-writing`流程：以最小必要检查确保链接、目录覆盖、来源和相互引用一致；没有建立额外 hash 审计或重新运行模型。写作流程来源为 Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*，[arXiv](https://arxiv.org/abs/2609.00065)、[DOI](https://doi.org/10.48550/arXiv.2609.00065)。2026-10-26 核对记录为 v2；此引用说明写作流程，不是本项目量化方法或实验结果的证据。
