# Inference engine、标准 LIBERO 与 Cosmos 跨模型证据

返回 [总交接](../HANDOFF.zh.md)。这里把 algorithm checkpoint、runtime 实现、storage、显存、latency 和 closed-loop 分开。当前推荐 Fast-WAM 路径是**packed VQ residency → 按层 BF16 decode → cuBLAS BF16 Linear + 条件 cache + CUDA Graph**；不是已验证的 native INT4/VQ GEMM。

## fastwam-vq-r128-c80-inference-profile

**问题与产物。** 冻结`25921795/reader_kl final8`，导出 inference-only bundle 并定位部署成本。Bundle 保存 614 目标的最终编码、transforms/U、非量化 state、normalizer 和 resolved config；排除目标 denseBF16、training masters/Adam、frozen reader。Factory 仍依赖 FastWAM 源码及 VAE/T5/tokenizer 外部 cache，并非一个文件就能离线搬到任意机器。

**设计与状态。** Legacy、BF16、inference_dense 三个独立进程各完整 80 queries，另 stream_decode 两个诊断 queries；3 warmups，主 latency 无 profiler，detailtrace 独立。`25938083.pbs101`成功，source/export action/future parity0。原 load 字段的计时范围不一致，不能据其算公平 cold-start speedup。

**存储账。** 目标 6,016,317,440 weights，编码含 transforms/U **2.991409 GiB / 4.271052 bpw**；未量化 parameters **11.902288 GiB**；whole unique tensor payload **14.893702 GiB**，archive **14.895614 GiB**，densewholepayload **23.108552 GiB**，whole 压缩 **1.5516×**。Target 压缩 3.7569×不能写成 whole model 压缩 3.7569×。

证据：[README.zh.md](../idea-validation/fastwam-vq-r128-c80-inference-profile/README.zh.md)、[EXPORT_STORAGE.zh.md](../idea-validation/fastwam-vq-r128-c80-inference-profile/EXPORT_STORAGE.zh.md)、[最终报告](../idea-validation/fastwam-vq-r128-c80-inference-profile/pulled/25938083.pbs101/FINAL_REPORT.zh.md)。

## fastwam-vq-r128-c80-fused-kernel

**设计。** 依次实现 Smooth/sign/H/U/A4 input fusion、uint8codebook lookup+decode+BF16 MMA/FP32 acc、direct/meta loader。保持 source checkpoint、完整 80panel；614decoderbitwise、代表 shapebench、full policy parity 分别检查。

**实际进度。** 初始 constexpr/globalconstant 编译错误无 modelquery；r3 通过 synthetic 与 614decoder 检查，但真实 input-fused anchor 的 A4 位模式 gate 失败，只有 eager80 完成，没有完整 packed/direct E2E 表。修复定位到 PyTorchscalar /7 与 Triton div_rn 算术差异，转入下一条 fused-tuning。

**接手意义。** 这个目录是失败 attempt 与初版 source，不是最终推荐 engine。不能把“614weight decoder 相同”当作整个 GEMM 或 policy 相同；activationoperand 和 reduction/epilogue 也需独立验证。

证据：[README.zh.md](../idea-validation/fastwam-vq-r128-c80-fused-kernel/README.zh.md)、[fused_input.py](../idea-validation/fastwam-vq-r128-c80-fused-kernel/fused_input.py)、[fused_vq.py](../idea-validation/fastwam-vq-r128-c80-fused-kernel/fused_vq.py)。

## fastwam-vq-r128-c80-fused-tuning

**最终测量。** `25943391`同一 A100 五模式各 80 queries；source 不变，0training/closed-loop。Inputfusion p50 **6.528→3.504 s**，action 误差 0；通用 packed fused VQ p50 **7.093 s**、peak16.330 GB；direct/meta loader 约 7.095 s、peak16.438 GB、loadpeak16.139 GB，0dense targetweights。

**精度失败。** Packed 相对 eager first 10 motor RMSE **0.0449623**，gripper 分歧 6/800；direct loader 相对同 packed arithmetic 误差 0，不等于对原 checkpoint0。独立 precisionprobe 提示主要偏差仍存在于 denseTritonGEMM，不能全部归因 VQdecode。Pipeline 验收完成表示结果完整，不表示候选数值等价或值得采用。

**当前决策。** 保留精确 inputfusion；不采用这条更慢且改变输出的通用 packed GEMM。后续回到 cuBLAS BF16 主 Linear，仅优化 decode/residency。该目录虽有“native VQ”历史命名，不能用于当前主路径的 native acceleration claim。

证据：[README.zh.md](../idea-validation/fastwam-vq-r128-c80-fused-tuning/README.zh.md)、[完整表](../idea-validation/fastwam-vq-r128-c80-fused-tuning/pulled/25943391.pbs101/RESULTS.zh.md)。

## fastwam-vq-r128-c80-layerdecode-bf16

**实现。** VQ 编码常驻，单个 BF16 scratch，每 Linear 由 Triton 完整解码当前矩阵，再调用原 cuBLAS `F.linear`；保留 inputfusion/directmeta loader，不常驻 614 张 denseweights。

**结果。** `25954757`第一轮 full 80action 误差 0，p50 **3.532909 s**、allocatedpeak **16.550766 GB**，对原 dense25.139 GB 形成真实 runtime 显存节约。第二轮 BF16 U 为 4.166 s、更广 BF16 input 为 3.982 s 且都改变输出，不推荐。沿用历史 eager/inputfused/BF16 对照，跨 allocation 比较明确来源。

**用途与限制。** 保留 FP32 H/U/A4 语义和 BF16 cuBLAS，数值兼容优于前一条 fused GEMM；decode 临时矩阵是当前层 scratch，不是“没有解码”。单请求与固定 geometry 为后续 Graph 基础。

证据：[README.zh.md](../idea-validation/fastwam-vq-r128-c80-layerdecode-bf16/README.zh.md)、[完整结果](../idea-validation/fastwam-vq-r128-c80-layerdecode-bf16/long/pulled/25954757.pbs101/RESULTS.zh.md)。

## fastwam-vq-r128-c80-runtime-abc

**组件设计。** A 预分配 FP32 H/U 与 BF16 A4 workspace，保持算术；B 增加 query 内 text/context 及各层 cross-attention K/Vcache；C 在 B 上为 video_step、video_prefill、action_step 建立三个手动 CUDA Graph。缓存含当前 proprio，只能按 query 重建，不能跨不同 observation 盲用。

**完整 80 query 结果。** A/B/C p50 **3.693649/3.047032/0.956914 s**，peak **16.592381/16.688547/16.949808 GB**，action 与独立 future anchor 均 0 误差。A 本身没有 E2E 收益，但提供 stable buffer addresses；B 使逻辑 targets/query 从 6436→5258；C 逻辑工作量不变，通过减少 host submission 开销大幅降时。Video/actionGraph 各 800 replays、prefill80。

**公平边界。** 历史 BF16 p50 **1.450439 s**尚无同等 cache/Graph 优化，C 相对它约 1.52×是完整 runtime 组合结果，不是 VQ 算法独立加速。Profiler nestedstages 不相加；Graph 增加 setup 和 memory 成本，报告已单列。

**compile 失败。** 首 job 完成 ABC 后 compile 卡住被精确停止，保留 ABC；独立重试`25958068`准备阶段 segfault139。没有有效 compilelatency 或精度结论，也不再自动尝试。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-r128-c80-runtime-abc/RESULTS.zh.md)、[abc_runtime.py](../idea-validation/fastwam-vq-r128-c80-runtime-abc/abc_runtime.py)、[graph_runtime.py](../idea-validation/fastwam-vq-r128-c80-runtime-abc/graph_runtime.py)。

## fastwam-vq-r128-c80-c-decoder-buffer

**当前推荐实现。** 固定 Q0 与 C，独立 decoder 有限 5 配置 shape-tuning，保持 FP32 codebook sum→ldexp→rowmultiply→BF16 write 顺序；17 真实 shapes 中 14 选择新配置、3 回退。再使 FP32 H/Uintermediates 复用 113,246,208-byteweight scratch，A4 保留独立 BF16 buffer，顺序 H→U→A4→weight decode→cuBLAS，避免覆盖仍在使用的 operand。

**主结果。** C / C+decoder / C+decoder+buffer：p50 **0.956914 / 0.890160 / 0.887096 s**；最终 mean **0.886964 s**、p95 **0.894262 s**、query peak allocated **16.914255 GB**。全 614weight/activationbitwise 检查、full 80actionmax/RMSE0、独立 anchoraction/future0。Decoder 主要解释降时；buffer 比 decoder p50 仅少 0.34%，跨 allocation 不解释成可靠速度收益，显存少约 35.55 MB。

**成本与存储。** 仍 5258decodes/query，累计写出约 107.159 GiB BF16 weights，是多次写出量，不是常驻显存。未量化 11.902 GiB 主要为 text encoder10.5815 GiB、VAE1.3126 GiB；aliases 按 storage 只计一次。Bundle 与编码不变，无训练/闭环。

**异常与验收。** Decoder`25959865`、buffer`25960283`测量完整，但原 wrapper 错误要求跨 jobproducer 相同而 F/1；CPU`25961057`只读已有 actions/receipts 修正来源绑定并验收，未重跑 forward。不能改写原 GPU 失败。推荐单 request/单 CUDAstream；多并发需独立 workspace/Graph ownership，尚未验证。

证据：[RESULTS.zh.md](../idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/RESULTS.zh.md)、[decoder_runtime.py](../idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/decoder_runtime.py)、[abc_runtime.py](../idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/abc_runtime.py)、[graph_runtime.py](../idea-validation/fastwam-vq-r128-c80-c-decoder-buffer/graph_runtime.py)。

## fastwam-libero-engine-20261027

**问题。** 将当前算法 checkpoint 和推荐 inference engine 真正接到标准 LIBERO 环境，检查冻结 C80 reader-KL 模型在原始 40 tasks 上的表现，以及 engine 在大量闭环 queries 中能否正常执行。这不是重校准、AQLM 实验或完整 LIBERO-Plus 评测。

**冻结模型与环境。** Q0 `25921795.pbs101 / reader_kl / final8`，released Optional-IDM base，`idm`，video10+action10 steps、sigma_shift1、32×7 action chunk、每 replan 执行 10。使用原 `fastwam-smoke/libero` 的官方 initial states 索引 0–9，环境/sampler seed42、每 task10 trials，4 suites 各 10 tasks，共 400 唯一 slots。30 settling steps；Spatial/Object/Goal cap400 控制步，Long700；沿原 normalizer、图像和 gripper 后处理。C80 来自同 40 tasks 的两条不同 demonstration snapshots，新增校准/训练 updates0；不是 unseen-task 测试，不使用 Plus 扰动。

**实际 engine 与并发。** Packed MXVQ 常驻、按层 BF16 decode、cuBLAS BF16 GEMM，condition cache、三 Graph 区域、H/U 借 weight scratch、A4 独立 buffer。复用原 decoder shape choices，不重新 tuning，不是 native INT4/VQ GEMM。四常驻 workers 各加载一次，通过原子动态 task queue 领任务，Long 优先；与旧 Plus 按偶奇固定 140 切片不同。最终各完成 **100/100/90/110** episodes、成功 **98/98/87/108**，总 model loads4；每 worker 加载/安装/anchor 约 304 秒。

**完成状态。** `26020749.pbs101 / fwlib400`，2026-10-27 **15:45:38 SGT**保存的 PBS 状态 **F/Exit0**、实际 walltime **01:22:44**，artifact 和四 workers exit0、PIPELINE_COMPLETE。原请求 normal→g3、4 GPU/32 CPU/220 GB、6h；scheduler 实际改为 64 CPU/440 GB、仍 4 GPU/6h。每 15 秒 GPU telemetry；30 分钟监控在完成后停止。没有为此次补文档再查询/提交作业。

**结果。** 标准 LIBERO **391/400（97.75%）**；四 suite 各 100，因此 overall 也等于四 suite 等权均值。

| Suite | 成功/episodes | 成功率 |
|---|---:|---:|
| Spatial | 98/100 | 98.00% |
| Object | 99/100 | 99.00% |
| Goal | 97/100 | 97.00% |
| Long (`libero_10`) | 97/100 | 97.00% |

9 失败来自 7 tasks（task IDs 从 0 开始）：Spatial3/5 各 1，Object5 为 1，Goal3 为 2/6 为 1，Long2 为 1/8 为 2；完整 40-task 表见 summary。本轮不保存 rollout 视频以减少 I/O。

**数值与性能证据。** 四张卡完整 action/future anchor 相对冻结 checkpoint 的 max abs 均 **0**；每 worker 执行覆盖 614 目标 modules 和 `video_step`/`video_prefill`/`action_step` 三 Graph，video/action replay 数各为 query 数×10。全轮 **6415 policy calls**，warm policy-call p50 **0.899527 s**，不含 observation preprocessing、model loading/Graph capture、环境步进。Telemetry 采样显存约 17161 MiB/卡，这是采样值，不能替换原 profile 的 peak allocated/reserved。闭环 states 和调用数量不同，不用 0.899527 覆盖原 80-query profile 的 0.887096。

**能说明什么。** 当前 Q0+推荐 engine 已在这 400 个官方 slots 实际闭环完成；相对初稿仅 fixed-input engine 验证，这是新增 deployment evidence。没有同 400 个 slots 的 BF16、其他量化 baseline、engine on/off 配对臂或多 seed 重复；不能计算本轮量化造成的成功率损失/加速，也不能直接对比官方 2000-episode 分母。保留 Plus208/280 作为另一套扰动面板结果，不能因 97.75%而替换它。

**实现与接手。** `worker.py`负责官方任务/initial states、常驻模型、engine 安装、startup anchor、原子任务队列与逐 episode 记录；`analysis.py`要求 400 唯一 slots、40tasks、相同协议及四 worker 通过后聚合，`control.py`管理 handle/status。产物在 `/scratch/users/ntu/yguo017/fastwam-libero-engine-20261027/`，本地保存 summary、task_queue 和实际 job artifacts。本轮已完成，无待续 episodes。

证据：[RESULTS.zh.md](../idea-validation/fastwam-libero-engine-20261027/RESULTS.zh.md)、[protocol.json](../idea-validation/fastwam-libero-engine-20261027/protocol.json)、[完整summary](../idea-validation/fastwam-libero-engine-20261027/pulled/26020749.pbs101/summary.json)、[最终monitor](../idea-validation/fastwam-libero-engine-20261027/monitor_status.json)、[实际worker](../idea-validation/fastwam-libero-engine-20261027/pulled/26020749.pbs101/artifacts/worker.py)、[实际聚合代码](../idea-validation/fastwam-libero-engine-20261027/pulled/26020749.pbs101/artifacts/analysis.py)。

## cosmos-policy-c8-w4a4

**第一批跨模型验证。** `Cosmos-Policy-LIBERO-Predict2-2B`官方 directpolicy，sharedDiT，5NFE、chunk16 执行 16。共享 OriginalC8 观察与指令，但 teacheraction/latent 由 Cosmos 生成，不能使用 Fast-WAM targets。280 targets 约 1.762Bweights；d4/twobooks/B32MXVQ，只有 A+Z，**没有 reader-KL**。SpinQuant200 updates，VQ16 updates，目标及预算不同。

**结果。** 840 episodes 完整：BF16 **233/280**，SpinQuant **221/280**，VQ **226/280**；VQ 对 Spin14 胜/9 负/257 平。Targetbpw16/4.125446/4.286979；wholeDiTbpw 另算，不是同一个分母。

**性能限制。** 当时 decoded-BF16 eagerwarmquery 约 BF16.594 s、Spin1.057 s、VQ10.042 s。Batchedhelper 虽加载，但实际 activation 路径没有接上，不能信 summary 里 batched 标签。这个慢结果不能代表后来的 engine，也不能悄悄用优化后 timing 替换其原始 receipt。完整闭环属于旧 A+Zcheckpoint。

证据：[RESULTS.zh.md](../idea-validation/cosmos-policy-c8-w4a4/RESULTS.zh.md)、[README.zh.md](../idea-validation/cosmos-policy-c8-w4a4/README.zh.md)。

## cosmos-policy-vqwam-20261024

**当前跨模型方法。** C80 先 W16A4 学共享 R8，再冻结 R、训 VQ8，每 update 平均 10 contexts；训练完整 16×6motorMSE、future-viewNMSE 和**joint-DiT suffixreader-KL**，不是仅旧 A+Z。Lambda **0.0645743742**、beta **1.7317817103**按 Cosmos 自身 C80 冻结，不沿用 Fast-WAM 数字。

**Reader 接法。** 在第 3 次 evaluation 前取得真实 solverboundary；只用 student noisy future 替换 slots6/7，actionslot4、其他 slots/condition/cutstate 来自 detachedteacher；然后由冻结原 DiT 的原 solver suffix 自然演化，不逐步 clampaction。Action 从 slot4 恢复并对重复 volume 平均。该模型没有独立 ActionDiT，不能照抄 Fast-WAM reader 类。`reader_math.json`是原 sampler+toydenoiser 的数学验证；真实 80/80cotangent 与 staged replay 另由训练 receipt 支持，二者分清。

**六臂结果。** 完整 1680 records：BF16 **233/280**、SpinQuant **221**、QuaRot **227**、SVDQuantadapted **78**、ViDiT-Qmixed **232**、ours **230**。Ours 对 BF168 胜/11 负/261 平；没有 matchedA+Zcontrol，不能把 230 相对旧 226 的变化单因归 KL。

**旧效率表。** 各臂 80 queries、720 timings；五 NFEdenoisingmean(ms)为 **367.886/780.141/1310.805/891.628/1467.195/566.490**。Ourspeak3.6079 GB、wholeDiTstorage1.2421 GiB；它有 engine，其他旧 arms 为 eagerdecodedBF16，不能据此评价算法的公平 native 速度。Ours 本身仍 BF16 Linear。

**恢复与有效性。** GPU`25969569`因 claim_queue.json.locktimeoutF/1，1120 新 records 已原子写完；CPU`25971643`验证它们和 560 复用 records，0 新 forward/episode。Finalgripper 报告`25971904`来自首个 chunk：4480gripperscalar/sign 全部相同只覆盖首 chunk，不证明全轨迹 gripper 一致。

**代码入口。** `runtime.py/common.py`管 officialendpoint；`train.py/calibrate_ours.py`管 R/VQ；`reader.py/softbin_loss.py`管 KL；`export_bundle.py/engine.py`管 packedresident、scratch、cache/Graph；`evaluate.py/evaluation_multinode.py`管分片/ledger；`aggregate.py/result_extract.py`管恢复与报告。

证据：[INTERIM_RESULTS.zh.md](../idea-validation/cosmos-policy-vqwam-20261024/INTERIM_RESULTS.zh.md)、[reader.py](../idea-validation/cosmos-policy-vqwam-20261024/reader.py)、[engine.py](../idea-validation/cosmos-policy-vqwam-20261024/engine.py)、[protocol.json](../idea-validation/cosmos-policy-vqwam-20261024/protocol.json)。虽然报告名含 INTERIM，正文及最终 JSON 已是完整六臂数据。

## cosmos-policy-native-perf-20261025

**目标与 scope。** 固定已有 checkpoint，测真正接入 native baseline 后的性能，并分别报告真实 sampler wall 和 FPcaptured DiT Graph replay CUDAevent。0 训练、0 新闭环。Ours 仍 codebookdecode+BF16 GEMM，native 一词描述 baseline backend/native sampler，不是宣称 oursINT4。

**Engine 改进。** 原 ours 前两 NFEeager、后三 Graph，优化为五 NFE 单 Graph，减少 host/cache 同步；full 80action/future≤1e-6。四代表 query×3repeat 仅是先期诊断，随后按用户要求复用旧结果、每新 arm40 contexts×seed0 各测一次，形成 mixedpanel。

**最终表与边界。** BF16 行 80 queries×3blocks×3repeats=N720；其他行 40 queries/seed0=N40，**不能称相同 cohort 等重复 benchmark**。Sampler BF16/oursmean **432.566/435.695ms**，peak **5.7094/3.3082 GB**；DiT replay **277.356/433.901ms**，peak **5.0787/2.6453 GB**。Oursstorage1.2421 GiB，sampler 近 BF16 但 DiT replay 仍慢，不能只报一个相近端点概括 backbone 加速。

**Baseline 资格。** Spin/Qua/SVD/ViDiT sampler means **3823.556/2217.626/859.117/1387.795ms**；四 baseline 的 representative fidelity 均有 failedreceipt、full 80 missing。用户明确允许把误差列为 report-only 继续收集性能，因此表格完整不等于 fidelity 通过，更不能沿用旧 eager 闭环分数冒充新 nativebackend 闭环。

**接手优先。** 若要正式效率结论，先决定统一 measurement design 和精度资格，再补 matchedBF16/cacheGraph、native 闭环。不能拿论文 RTX4090DiT replay 与本机 A100policycall 直接论 SOTA，也不自动发起额外 sweep。

证据：[RESULTS_mixed.zh.md](../idea-validation/cosmos-policy-native-perf-20261025/RESULTS_mixed.zh.md)、[DIAGNOSIS.zh.md](../idea-validation/cosmos-policy-native-perf-20261025/DIAGNOSIS.zh.md)、[engine_optimized.py](../idea-validation/cosmos-policy-native-perf-20261025/engine_optimized.py)。
