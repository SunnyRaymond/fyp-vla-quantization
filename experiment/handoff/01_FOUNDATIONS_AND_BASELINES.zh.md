# 基础管线、历史诊断与 baselines

这是 [总交接文档](../HANDOFF.zh.md) 的第一部分。数值来自下列本地原始报告；这里重组问题、设置、结果与接手意义，没有重新运行实验。`first_frame`、`idm`、fixed-input 与 closed-loop 分开解释。下文的“完成”指对应报告记录的覆盖和验收，不表示重新核查了远端文件是否仍存在。

## fastwam-libero-bottleneck

**问题与设计。** 在优化模型前，先测清 FastWAM Optional-IDM 的 `first_frame` 实际时间花在哪里。冻结 checkpoint、observation、proprio、prompt 与 RNG，比较原生调用、逐层 hooks、较轻计时及单独 profiler；另测 OSMesa 环境步进。它服务于判断 attention sparsity/cache 的整段收益空间，不是量化实验。

**结果。** 原生暖态 action chunk 约 637–640 ms；10 次 action denoising 合计约 523.12 ms，约占暖态 chunk 的 82%。VAE 当前图像编码约 9.99 ms，text encode 约 33.15 ms，video K/V prefill 约 52.04 ms。所谓 action mixed-attention 约 55.4 ms 是 inclusive 诊断区间，含多种边、layout 和 launch 开销；单次 profiler 的相关 GPU kernels 累计仅约 11.29 ms，不能把前者当作可删除的 observation QK/AV 独占成本。

**局限与后续。** 逐层 hooks 使整段由约 638 ms 变成 1212 ms，说明详细 profiling 本身会扰动性能。OSMesa 的每步约 150 ms 是 CPU rendering 环境开销，不是真机控制频率。当前 inference engine 的完整 policy-call 与这里的 first_frame 不是同一模式和计时口径。接手时用本实验理解历史瓶颈，不能拿它直接作新 engine 的 BF16 分母。

证据：[RESULT.zh.md](../idea-validation/fastwam-libero-bottleneck/RESULT.zh.md)、[PROTOCOL.zh.md](../idea-validation/fastwam-libero-bottleneck/PROTOCOL.zh.md)。实现入口为该目录 `runner.py` 与相应 profiling 脚本，具体 artifacts 由报告索引定位。

## fastwam-idm-libero-bottleneck

**问题与设计。** 用同一 Optional-IDM checkpoint、相同输入与随机源，比较 `first_frame` 和 `idm`。前者不生成 future，后者多出 10 次 video denoising，再用 current/future video 做 action conditioning；两者 action 均为 10 步。

**结果。** 12 对原生完整调用中，IDM/first_frame 的配对比值均值为 1.8039。完整调用 median 分别为 1158.59 / 643.19 ms；单轮 video/action forward 的 CPU wall median 约 52.00 / 52.99 ms。实际等待时间上两段相近，而 profiler child kernels 的 video/action 合计约 413.26 / 158.10 ms；这两种观察并不矛盾。

**异常与恢复。** GPU 作业 `25663625.pbs101` 完成了 measurement，但简报汇总器读错 nested keys，原始 F/Exit1 保留。CPU `25663764.pbs101` 修正后处理并验收，未重跑模型和 episode。不要把 GPU wrapper 改写成成功，也不要因后处理失败自动重跑昂贵采集。

**接手意义。** 现有 joint video/action 量化必须同时保护 video 与 action 链路；first_frame profiling 不能替代 IDM profiling。本实验也没有证明 action-context sparsity 能得到大幅整段加速。

证据：[结果及恢复记录](../idea-validation/fastwam-idm-libero-bottleneck/RESULT.zh.md)、[协议](../idea-validation/fastwam-idm-libero-bottleneck/PROTOCOL.zh.md)。

## fastwam-action-parallel

**问题。** 分开测试两种“并行”：固定 text/proprio context 的 K/V 预计算，以及依赖性的 denoising 时间步能否用 prefix-Picard 近似并行求解。前者是精确复用，后者会改变求解过程，不能混为一种优化。

**设置。** BF16、Optional-IDM/first_frame、10-step、32×7 actions、A100；固定 context pilot 用三个冻结观察。Cache 比较 native、text embedding cache、逐层 K/V cache、跨层投影 batching。时间线比较 W=2/5/10、共 11 个 window/round 配方，预先要求 action max-abs 与 relative-L2 均不超过 1e-3。

**结果。** 逐层固定 K/V cache 的完整 infer_action 约 642.77→591.82 ms，配对约 1.0864×，逐步及最终输出一致。跨层 batching 让构建更快，但整段额外收益约 0.4%，且增加显存。所有 11 个 prefix-Picard 配方均未过 numerical gate；W10/R1 看似约 9.88×，输出误差很大；W10/R10 约 0.99×，仍未通过。Batch10 仅多约 47.4 MiB，故当前限制是数值/收敛与额外工作量，并非模型复制导致的巨大显存压力。

**当前决策。** 保留精确 K/V cache 的实现思想；冻结的时间并行 recipe 为 NO-GO。当前 engine 的条件缓存仍需在当前模型模式内独立验证，不能继承这三个观察的覆盖结论。

证据：[RESULT.zh.md](../idea-validation/fastwam-action-parallel/RESULT.zh.md)、[context 协议](../idea-validation/fastwam-action-parallel/CONTEXT_PROTOCOL.zh.md)、[时间并行协议](../idea-validation/fastwam-action-parallel/TIME_PROTOCOL.zh.md)。

## fastwam-phase-dither

**问题与表示。** 在 W4A8 activation subtractive dither 中，通过共享随机 draw 的 phase 调整跨位点量化残差相关性，尝试降低最终 action error。固定 8 个 phase groups，保持原模型权重与求解器；这是独立历史方向，不是当前 additive weight VQ 的组成。

**四段证据。** `mechanism-screen/` 在完整 first_frame 路径上校准 phase 与 matched direct-PTQ，比较 7 arms；`new-draw-a8/` 检查新 draws；`paired-followup/` 做输出级误差分解；`libero-goal-suite/` 做完整 Goal suite 闭环。它们的任务、分母、seed 和 precision 不能合并成一张无标签表。

**结果与结论。** 原 screen 对 independent dither 的完整 action MSE 改善 5.27696%，低于预先规定的 10% 扩展 gate，因此 `NO_GO_FOR_EXPANSION`。新 draws 下 phase/independent 的排序改变。完整 Goal 10 tasks×50 states、每臂 500 episodes：BF16 487，RTN 486，independent 483，learned phase 481；learned 对 RTN 为 7 胜/12 负/481 平。当前 recipe 没有闭环优势。

**边界。** 该闭环用 first_frame、20 action steps、W4A8 fake-quant floating GEMM；当前 Microscaling 主结果用 IDM 10+10，不能直接比较 latency 或成功率。NO-GO 只约束这套粗粒度 recipe，不等于证明 dither 全家族无效。原始二进制输入依赖旧 scratch；本地 `new-draw-a8` 只有从混合结果提取的 A8 字段，不能声称 raw data 全部齐全。

证据：[总入口](../idea-validation/fastwam-phase-dither/README.zh.md)、[screen](../idea-validation/fastwam-phase-dither/mechanism-screen/RESULT.zh.md)、[suite 结果](../idea-validation/fastwam-phase-dither/libero-goal-suite/RESULTS.zh.md)。

## fastwam-libero-plus-pilot

**最初目标。** 原设计为 4 suites×7 扰动×50 variants=1400 variants，每 variant 比 BF16/W4A8/W4A4/W4A4 KV4，合计 5600 slots；这是后续 280-variant 面板的来源，不是同一个分母。W4A4 要求真实 packed INT4 Tensor Core GEMM；KV4 要求真实打包存储后读取，不接受 fake quant 冒充。

**实际进度。** `RUN_STATE.json` 明确记录：保留的 78 shards 已完成，CPU 汇总验证的是 Spatial 6 个维度、300 variants×4 arms=1200 slots；A4 与 A4 KV4 均为 0/300。原 5600 目标、完整 performance/storage 均未完成，后续提交已禁用。状态文件里仍有 `completed_episodes=0` 等历史字段，与自然语言状态并不一致，不能机械把任意一个字段当最终评分。

**为什么保留。** 这批证据建立了 plain per-row A4 大幅失真的背景，并促成 scope factorial 和 Smooth/Hadamard 路线。接手时不能恢复成“继续把 5600 跑完”；当前主线不依赖补齐这个已停止 pilot。若引用其分数，只引用已经验证的具体子面板与对应分析，不能写成 full LIBERO-Plus。

证据：[PREREG.zh.md](../idea-validation/fastwam-libero-plus-pilot/PREREG.zh.md)、[RUN_STATE.json](../idea-validation/fastwam-libero-plus-pilot/RUN_STATE.json)。`analyze_completed.py`、`completed_all_sources.json`、`artifacts/` 保存 partial 分析来源；旧 queue 快照不代表现在还在排队。

## fastwam-a4-phases12

**问题与设计。** 识别 A4 的主要敏感 scope，并排除“只有 native kernel 出错”的解释。在 10 个 Original Spatial 与 12 个 Plus 输入、每输入 2 seeds 上，冻结 W4 G128，只对 video/action/proprio 的 A8/A4 做完整 2³ factorial；加独立解包 FP32 Linear reference。总 440 个 full queries，0 闭环。

**结果。** Original/Plus first 10 motor RMSE：全 A8 0.02034/0.02852；仅 video A4 0.32548/0.31233；仅 action A4 0.11299/0.12854；全 A4 0.34717/0.34535。Video 的 paired 主效应约 +0.25888，action +0.06263，proprio 接近零。Video×action 存在非加性交互。

**机制证据。** Video conditioning/denoising 的 A4 零编码比例约 65–72%；同输入独立 reference 同样有大偏差。局部 native/reference output NRMSE 约 2.5e-5–3.0e-5，但两条完整轨迹最终 first 10 RMSE 仍相差约 0.04551，说明小数值差也可能经多步放大，不能把这部分简单加总为 kernel 因果贡献。

**接手意义。** 支持优先改善 video/action activation basis 与量化步长，促成 Smooth/H128；不支持只降低 proprio 精度来修复主问题。44 contexts 是 22 个输入内的配对重复，不是 44 个独立场景。

证据：[RESULTS.zh.md](../idea-validation/fastwam-a4-phases12/RESULTS.zh.md)。

## fastwam-smooth-vq-phases12

**问题。** 先用 Smooth/Hadamard 改善 A4，再看简单 weight VQ 能否接住被迁移到 weights 的难度。614 Linears、约 6.016B target weights；固定 8 calibration contexts、4 selection cases×2 seeds、10 held-out cases×2 seeds。Selection 前没有使用 test；0 episodes。

**实现。** 局部 Linear 配对变换后保持实数等价；Smooth alpha 与 Hadamard 强度只在预设候选中选。VQ 为 d4、两本 256-entry additive books，每层独立拟合；原始版尚未有后来完善的 row-scale/Microscaling。

**结果。** BF16 W+A4 的 held-out motor RMSE 从 identity 0.33724 降至 Smooth0.5+H 的 0.04529，约下降 86.6%。同一 S0.5H 基底下 Scalar W4A4 为 0.05745，原始 VQ 为 0.07831，10/10 test cases 的 VQ 更差。Selection 上更激进 smoothing 的 activation error 更低，并不保证 action 更好；weight peak ratio 随 smoothing 可显著增大。

**决策。** 固定适中的 Smooth0.5+H128 作为后续共同 preprocessing，再改 VQ 表示/拟合。这个实验不能与后来的 22-case test 数字直接混表；`VISUALIZATION.zh.md` 是解释性展示，不是额外闭环验证。

证据：[RESULTS.zh.md](../idea-validation/fastwam-smooth-vq-phases12/RESULTS.zh.md)、[README.zh.md](../idea-validation/fastwam-smooth-vq-phases12/README.zh.md)。

## fastwam-rotation-baselines

**问题与公平性。** 在同一个 Optional-IDM 模型、冻结 280 variants 上比较 BF16、固定 QuaRot-adapted、learned SpinQuant-adapted W4A4。局部右乘在 norm/modulation/nonlinearity 后执行，不把 LLaMA 的全局 residual rotation 直接穿过 WAM。Learned rotation 复用历史 step200，不重训、不重新选 checkpoint。

**结果。** 正式作业 `25728285.pbs101` 完成 840/840 slots，BF16 **216/280**、QuaRot **195/280**、SpinQuant **203/280**。SpinQuant 对 QuaRot 为 16 胜/8 负/256 平；两者分别比 BF16 少 13/21 个成功。量化 episodes 有实际 native INT4 调用证据。

**边界与特殊放行。** 原 preparation 的部分 native-vs-reference 数值差被用户明确放行；未完成的 selection 检查不能写成全通过。正式 native runtime、完整覆盖和结果有效性仍验收。它是本地 adapted recipe，既不能叫官方完整复现，也不能保证 native 与 decoded BF16 输出相等。

**接手用途。** 它提供后续反复复用的 280-variant manifest 和 BF16/rotation reference。优先复用已经完成的 slots，不重新生成 seeds 或抽样。

证据：[PREREG.zh.md](../idea-validation/fastwam-rotation-baselines/PREREG.zh.md)、[正式完成 receipt](../idea-validation/fastwam-rotation-baselines/formal_completion_receipt.json)、[summary](../idea-validation/fastwam-rotation-baselines/results/summary.json)。

## fastwam-smooth-scalar-plus

**问题与设计。** 把已冻结 Smooth0.5+Hadamard Scalar bank 接入真实 native W4A4 闭环，与已有 BF16/QuaRot/SpinQuant reference 比较。2 GPUs 各 140 episodes；不新增 calibration 或训练。

**结果。** `25734678.pbs101` 完整 280/280，Scalar **200/280（71.43%）**；对 BF16 为 9 胜/25 负/246 平，对 SpinQuant 为 14 胜/17 负/249 平。实际 native 完整 query mean 约 6.124 s，query allocated peak 约 16.23 GB；压缩降低 storage/allocated memory，没有形成速度优势。

**边界。** Scalar 的历史 calibration/selection 只覆盖 Spatial，rotation baseline 覆盖四 suites；因此不能把成功率差全部归因于 Smooth。Native 与旧 decoded reference 的启动 action 存在差异，这个闭环只属于该 native 实现。其 bank 来源固定，不因 Plus 结果重新调参。

证据：[RESULTS.zh.md](../idea-validation/fastwam-smooth-scalar-plus/RESULTS.zh.md)。

## fastwam-spinquant-rmse

**问题。** 将已训练的 native SpinQuant bank 放到与 VQ 相同的固定输入 action-fidelity 面板中，避免只比较不同轨迹的 closed-loop queries。4 selection cases×2 seeds 加 22 test cases×2 seeds，共 52 个新 query，另一个 BF16 replay。

**结果。** Test first 10 motor RMSE **0.057373**；历史 Scalar 0.054994、row-scale VQ 0.057052、local-output VQ 0.057256、action-sensitive VQ 0.061030。正式每 query 记录 614 targets、6436 次 native INT4 GEMM；没有回退 decoded SpinQuant。Job `25756064.pbs101` F/Exit0。

**解释。** 这是完整配置间的误差比较，不是 rotation 的纯因果效应。构造 runtime 时的旧 JSON 显示 native calls=0，是 BF16 启动阶段；正式 query receipts 才是实际执行证据。此轮没有同输入 native latency benchmark，也没有新 episodes。

证据：[RESULTS.zh.md](../idea-validation/fastwam-spinquant-rmse/RESULTS.zh.md)、[native receipt](../idea-validation/fastwam-spinquant-rmse/native_execution_receipt.json)。

## fastwam-diffusion-baselines

**问题与设计。** 在同一 614-target、相近 effective bpw 下加入 diffusion quantization 的 SVDQuant/ViDiT-Q adaptations。C80 calibration，两个新 arms 各 280 episodes，复用 2240 reference slots。SVD 保留 BF16 low-rank residual；ViDiT 保留 channel balancing、rotation 和 token-dynamic A4。

**结果。** `25878591.pbs101` 完成 560/560 新 episodes：SVDQuant **108/280（38.57%）**、ViDiT-Q **203/280（72.50%）**。Target effective bpw 为 4.4250 / 4.3250；SVD global rank16。ViDiT 与 SpinQuant 成功数相同，但仍有 9 胜/9 负的不同 slots。SVD 的低分是该适配版事实，未做归因。

**不可混淆。** 本轮低比特路径为 decoded BF16/fake-A4，不是两者官方 native backend。SVD 没有完整 smoothing grid search/迭代 refinement；ViDiT 没有官方 mixed precision protection，部分宽度使用 dyadic fallback。完整 query mean 约 2.304 / 5.855 s 是本实现成本，不是论文 kernel 速度。后来的 Cosmos native benchmark 是另一项实验。

证据：[方法说明](../idea-validation/fastwam-diffusion-baselines/README.zh.md)、[RESULTS.zh.md](../idea-validation/fastwam-diffusion-baselines/RESULTS.zh.md)。

## fastwam-prism-visualization

**问题。** 用相同输入的 activation/weight 图解释 transformations 与 Microscaling 的作用，避免把更平坦的图面直接当成最终控制收益。采集 2 个输入×5 arms，共 10 次 full forward，得到 752 份 activation/paired-local/weight 记录；16 个目标 Linears 覆盖两个 experts 和多个深度。

**结果。** 代表性 video block10 ffn.2 的 channel RMS CV：raw 0.492、SpinQuant R128 0.130、Smooth+H128 0.122；但 action 位置的相对排序不同。Row/Micro 共享相同 activation recipe；其差别应看相同基底下的 weight/output error，而非说 Micro 又额外“平滑 activation”。

**接手边界。** 图窗口不是全模型统计。后续离线 weight analysis 复用已有 captures，不额外跑整模型；no-A4 FP32 bias-free Linear 的 output NMSE 不能写成 action RMSE、native latency 或 closed-loop 证据。原始 signed NPZ 与 JPEG 展示分开保存。

证据：[README.zh.md](../idea-validation/fastwam-prism-visualization/README.zh.md)、[WEIGHT_ANALYSIS.zh.md](../idea-validation/fastwam-prism-visualization/WEIGHT_ANALYSIS.zh.md)、[本地图集](../idea-validation/fastwam-prism-visualization/retrieved/results/full/plots/gallery.html)。

## fastwam-teacher-multiseed

**问题与设计。** 在构造概率/soft-bin 目标前，先测 BF16 teacher 的条件采样波动。C80、每 case16 seeds，共 1280 记录；seed0 复用 80 条，新增 1200 条。仅 first 10×6 normalized motor；0 训练、0 held-out、0 闭环。

**结果。** Case 内 seed 波动 RMS std **0.02358**，case mean 间 std RMS **0.32473**，约 13.77 倍。使用固定 88 投影比较 K4/K8 子组；covariance relative Frobenius error mean 分别 1.2130/0.98766，effective rank 约 6.2136。16 次 cached/repeat replay 检查通过。

**结论限制。** Full16 是有限样本参照，不是真分布；16 seeds 不等于 16 独立环境状态，也不能证明多模态。共享 Gaussian bandwidth 可能大于 case 内波动；σ0.25 时 98.75%投影宽度超过相应 within-state std。没有预先 stability threshold，不能写成“分布估计已稳定通过”。此数据不自动改变 Cosmos 训练协议。

证据：[RESULTS.zh.md](../idea-validation/fastwam-teacher-multiseed/RESULTS.zh.md)、[DESIGN.zh.md](../idea-validation/fastwam-teacher-multiseed/DESIGN.zh.md)。

## reproduction/fastwam-smoke

**角色。** 原始 FastWAM、LIBERO、OSMesa、checkpoint、容器/host 库兼容性的基础入口。上游 revision `7faa71108368fbb3b6885649f112af607427a2d4`；主仓库只保存内部 FastWAM 的 gitlink，恢复源码需检出登记 revision。

**已完成。** 2026-10-02 的 `25658646.pbs101`，first_frame，LIBERO-goal task0，**1/1 成功**、F/Exit0、PIPELINE_COMPLETE。单 episode 约 57.45 s，不含完整模型加载，不能作为模型 latency。旧 2026-09-09 smoke 记录也保留。

**实际实现约束。** PBS 分配 GPU UUID 与 EGL 全局 index 不兼容，当前采用 OSMesa CPU rendering、分配 GPU 推理，不修改 CUDA mask 去探测别人的卡。旧 EGL 全局探测入口已禁用。该 smoke 仅证明一条管线能跑通，不是完整 suite/IDM 结果。

证据：[RESULT.md](../reproduction/fastwam-smoke/RESULT.md)、[README.md](../reproduction/fastwam-smoke/README.md)。

## reproduction/cosmos3-edge-robolab120

**角色。** Cosmos3-Edge-Policy-DROID 与 RoboLab-120 的后续迁移准备，不是当前 Cosmos Policy/Predict2-2B 实验，也没有完成 Microscaling 量化。

**保存内容。** README 登记了 policy/framework/RoboLab/VAE 精确 revision 与来源；保留 CPU 依赖准备、policy server 与 episode launcher。A100 可作为 policy-server 候选，但 RoboLab/Isaac simulator 需要兼容 RTX 环境，不能把 A100 支持 CUDA 等同于支持 Isaac rendering。

**本地最新证据。** 原文末尾明确：依赖安装未确认完整，CUDA 环境未在获批 A100 验证，没有 model load、render、inference 或 RoboLab episode。旧`25660513.pbs101`和网络 probe 仅是历史状态；不能当成今天仍在运行。下一步是核实既有安装 receipt 与可用 RTX runtime，再由负责该方向的人决定是否推进；不要自动安装已明确跳过的 simulator。

证据：[README.md](../reproduction/cosmos3-edge-robolab120/README.md)。
