# Fast-WAM + LIBERO 瓶颈测量

日期：2026-10-03。本文的 `first_frame` 主测量和 CPU rendering 细分诊断均已完成。同 checkpoint 的 `idm` 路径和同观测模式比较也已完成，详见 [IDM 报告](../fastwam-idm-libero-bottleneck/RESULT.zh.md)。

## 结论

现有 FastWAMOptionalIDM checkpoint 的 `first_frame` 路径，原生稳态 action chunk 约 **637–640 ms**。完整的 **10 轮 action denoising** 是模型内部的主要耗时；ACSM 所针对的 action-query→observation QK/AV 只占其中一部分。

较轻计时下，所有 action mixed-attention 合计约 **55.4 ms/chunk**；这个范围还包含 action→action、mask/layout 处理、CPU launch/stream gaps 和计时扰动。单次 profiler 中这 300 次 SDPA 的 GPU kernel 累计约 **11.29 ms**，其中 efficient-attention backend 约 **6.38 ms**。不能把 55.4 ms 当作 observation QK/AV 的原生独占耗时。

将整个 55.4 ms 路径都假设为可删除、忽略首次 dense、评分和 packing，代入同批原生调用 640.3 ms，得到约 **1.095×** 的宽松理想估算。它不是严格测得的 ACSM 加速上界：分子来自受扰动的区间，且干预实际只删部分 observation edges。当前证据支持先关注完整 action 执行路径，而不是把 ACSM 当成该模式的大幅加速主线。

闭环环境每步约 **150 ms**；执行 10 个动作对应约 **1.50 s** 的环境耗时，超过一次稳态模型调用。这里使用 OSMesa CPU rendering，这项延迟不能代表真实机器人或 EGL/GPU rendering。

## 1. 固定 baseline 与证据

- Source revision：`7faa71108368fbb3b6885649f112af607427a2d4`，沿用准备阶段的固定 archive/revision marker。
- Checkpoint：`libero_optional_idm_2cam224.clean.pt`，12,041,735,545 bytes；对应 dataset stats。
- Mode：`FastWAMOptionalIDM / first_frame`，调用 Fast-WAM direct action 路径；不生成 future video。**这不是 FastWAM-Joint 的测量。**
- NVIDIA A100-SXM4-40GB，allocated UUID `GPU-751e6130-e59d-f893-b230-bee77bd9c48b`；节点 `x1000c1s7b0n0`。
- BF16，seed 42，sigma shift 1.0，CFG 1.0，compile=false，10 denoising steps；预测 32×7 的 action chunk，每次最多执行 10 个动作。
- `libero_goal` task0 / trial0：`open the middle drawer of the cabinet`，30 个 no-op waiting steps。
- PBS `25663213.pbs101`：F / `Exit_status=0` / wrapper exit=0 / walltime 00:10:10；`PIPELINE_COMPLETE`、`TASK_SUCCESS` 存在，rollout MP4 为 228,479 bytes。
- PBS `Stageout_status=1`：PBS 自动 stageout 有异常；scratch 内日志、结果、计时、完成标记已核实，并已取回小型报告产物。
- 单 episode 1/1 success，仅证明本管线执行成功，不提供全 suite 任务成功率。

原始小型产物见 `artifacts/25663213.pbs101/`；完整 raw spans 与 profiler trace 留在 compute-side artifacts，未传回大型 trace/模型文件。

## 2. 测量扰动与输出一致性

从真实 episode 的 replan 0、6、12 取三个固定观测。每个 context 3 warmups，4 对原生/逐层细分重放，交替 AB/BA；再做每 context 2 对原生/较轻计时重放。观测、指令、proprio、RNG 和模型/solver 保持固定。

| 完整调用 | 样本数 | mean ± SD（ms） | median（ms） | 相对配对原生扰动 |
|---|---:|---:|---:|---:|
| 原生（逐层组的参照） | 12 | 638.21 ± 6.79 | 636.51 | — |
| 逐层细分 | 12 | 1212.15 ± 110.59 | 1162.98 | +89.93% |
| 原生（较轻组的参照） | 6 | 640.33 ± 5.24 | 639.42 | — |
| 较轻计时 | 6 | 687.42 ± 4.34 | 685.42 | +7.35% |

12 对逐层和 6 对较轻重放的 raw/postprocessed outputs 均通过 finite/allclose；与原 episode 对应动作一致，已检查的最大绝对差为 0。计时改变了性能，没有改变本次输出。

因此：完整原生延迟以 native 为准；主要阶段采用较轻计时；逐层数据只用于定位；profiler kernel sums 来自一个独立、有扰动的 trace。不能把不同 passes 的各行拼成一个原生时间分区。

## 3. 冷启动与闭环 episode

| 范围 | 耗时 |
|---|---:|
| 模型构造/组件加载 | 254.62 s |
| Checkpoint override | 15.96 s |
| 进入 episode 前的 runner 总时间（包含上述项） | 272.55 s |
| Episode 的首次 action chunk | 19.475 s |
| 后续 12 个 action chunks | mean 637.16 ms，median 636.61 ms |
| 13 次模型完整调用合计（包含首次冷调用） | 27.121 s |
| 155 次 environment.step 合计（含 30 个等待 steps） | 23.298 s |
| Env create / reset / set_init_state | 3.465 / 1.629 / 0.147 s |
| MP4 编码保存 | 3.184 s |
| Env close | 0.208 s |
| 官方 episode duration | 59.187 s |

这些 episode 外层区间基本是相邻阶段；完整初始化总量与其子项不能再相加。

已有 raw spans 在 CPU allocation `25663473.pbs101` 内解析，13 个 chunks 的每轮/每阶段数据见 `episode_chunks.csv`。首次调用的 CPU inclusive 明细：预处理 **1.204 s**、VAE **10.562 s**、文本 **4.583 s**、video prefill **0.392 s**、10 轮 action forwards 合计 **0.734 s**；`model.infer_action` 为 **18.255 s**，完整 chunk 为 **19.475 s**。模型父区间包含未单列操作与同步，不能把父子项重复相加。仅凭这些边界计时不能确定冷延迟由哪一个内部编译/初始化机制造成。

## 4. 稳态 action chunk 的主要阶段

原 episode 暖态取后续 12 个 chunks 的 CPU wall 边界；较轻计时取 6 个固定观测重放的 **inclusive** CUDA event 区间（包含 stream gaps）。两列来自不同 passes，不组成同一个时间分区。

| 阶段 | 原 episode 暖态 CPU wall（ms） | 较轻计时 CUDA inclusive（ms） | 含义 |
|---|---:|---:|---|
| 图像/proprio 预处理 | 3.80 | — | 原生输入路径 |
| VAE 当前图像编码 | 9.99 | 9.76 | 当前两相机图像 |
| encode_prompt | 33.15 | 33.15 | 每个 chunk 重新编码同一任务文本 |
| Video K/V prefill | 52.04 | 56.17 | 30 层，整个 chunk 仅一次 |
| 10 轮 action denoising 合计 | 523.12 | 569.42 | CPU 区间约为原 episode 暖 chunk 的 82% |
| 其中：300 次 action mixed-attention | — | 55.42 | 是上一行的子项，不能相加 |
| 其中：30 次 video prefill attention | — | 5.52 | 是 video prefill 的子项 |
| 动作反归一化 / gripper inversion | 0.129 / 0.005 | — | 数值后处理 |

各阶段以同 pass 的统计均值归并；尚有 schedule/noise、video.prepare、proprio embedding、D2H 同步、Python gap 等未单列部分。CPU enqueue wall 和 CUDA event 是不同视角，不能相加。

### 10 轮 denoising

| 轮次（0-based） | 较轻 CUDA inclusive（ms） |
|---:|---:|
| 0 | 56.05 |
| 1 | 55.97 |
| 2 | 57.13 |
| 3 | 57.08 |
| 4 | 56.94 |
| 5 | 57.55 |
| 6 | 57.66 |
| 7 | 57.06 |
| 8 | 57.35 |
| 9 | 56.63 |

没有某一轮明显独占延迟。逐轮明细可见 `denoise_steps.csv`。

原 episode 暖态每轮 CPU wall 平均约 **51.85–52.90 ms**；较轻事件计时则约 56–58 ms。首次 chunk 的第 0 轮为 263.29 ms，后续轮约 47–53 ms。这三种范围在 CSV 中分列。

### CPU 环境渲染细分

CPU-only PBS `25663473.pbs101`：F / Exit_status=0 / wrapper exit=0 / PIPELINE_COMPLETE，walltime 00:02:55。节点 `x1001c3s3b0n1`，8 CPUs、16 GiB、GPU=0；同任务/init state/seed、OSMesa、256×256 两相机，OMP/OPENBLAS/LP threads 均为 8。30 个 no-op warmup 后测 10 个 no-op steps；在 sim class 上挂 render hook，已验证 reset 重建 instance 后依然生效，捕获 20 次真实 render。

| 同一 CPU 诊断的相邻组成 | mean（ms/step） |
|---|---:|
| agentview render | 99.71 |
| eye-in-hand render | 82.50 |
| 两路 render 合计 | 182.21 |
| 去掉上述 render 的残余（物理/控制/其余观测等） | 20.29 |
| 完整 environment.step | 202.50 |

该诊断中渲染约占 **89.98%**，说明本软件渲染路径的环境开销主要在图像生成。原 GPU allocation 的真实 episode 为 150.31 ms/step；两次节点、状态轨迹和采样范围不同，不能用 89.98% 精确拆分原 episode 的 23.298 s，亦不能把剩余 20.29 ms 全部叫“纯 physics”。

## 5. Action block 内部：三类计时放在各自范围内理解

下表每行合计 30 层×10 轮。前两列来自高扰动的逐层计时；末列来自单次 profiler 的 CPU record scope 关联 GPU kernel 累计，排除独立 CUDA annotation ranges。

| Action 操作 | 逐层 CPU inclusive（ms） | 逐层 CUDA inclusive（ms） | 独立 trace GPU kernel 累计（ms） |
|---|---:|---:|---:|
| Q projection | 58.14 | 46.44 | 4.03 |
| K projection | 36.34 | 25.00 | 3.92 |
| V projection | 36.65 | 25.43 | 3.87 |
| Output projection | 37.65 | 27.91 | 4.63 |
| FFN parents | 119.85 | 109.74 | 11.00 |
| Text/proprio cross-attention parents | 348.54 | 338.05 | 54.12 |
| Observation+action mixed SDPA | 64.71 | 56.32 | 11.29 |
| 整个 action denoising | 1011.77 | 1011.50 | 150.31 |

Text cross-attention parents 包含其 Q/K/V/O、attention 与 normalization；不能再与其 children 相加。整层还有 norm、modulation、RoPE、gate、packing 等操作。逐层记录不能直接解释为“原生调用里 cross-attention 占 338 ms”。

本次 mixed SDPA 实际选择 `aten::_scaled_dot_product_efficient_attention`。源码函数虽然叫 `flash_attention`，本次并未因此证明使用 FlashAttention backend。

`layers.csv` 提供 360 行：video/action 各 30 层的 Q/K/V/O、FFN、text cross-attention。Action 的每行包含 10 轮；video 的每行包含一次 prefill。

## 6. ACSM 的真实干预范围

实测 attention shapes：

- Observation prefill Q/K/V：`[1,98,3072]`，24 heads×128。
- Action query：`[1,32,3072]`。
- Mixed K/V：`[1,130,3072]` = **98 observation-cache tokens + 32 action tokens**。
- Text/proprio cross-attention：query 为 32，context 为 129；这是另一条 attention 路径。

每层每轮 action→observation QK/AV 理论 FLOPs（MAC 记 2）为：

`4 × 1 × 24 × 128 × 32 × 98 = 38,535,168`。

30 层×10 轮合计 **11.56 GFLOPs/chunk**。整个 mixed QK/AV 为 **15.34 GFLOPs/chunk**；observation edges 占约 75.4% 的 attention 边计算，但这个比例不等于完整模型时间占比。

ACSM 保留首次 dense evaluation、QKV/FFN、text/proprio cross-attention、原生 solver。它删的是后续 action queries 读取部分 observation K/V 的 QK/AV edges；不减少上述整个 569 ms 的 action 路径，不免除 video prefill，也不解决 OSMesa 环境每步 150 ms 的延迟。

在这份 `first_frame` baseline 上，**目标路径存在且反复执行，但可优化的时间份额有限**。首步评分、group provenance、compact kernel 与 packing 成本仍未实测，不能声称已经获得 ACSM 的净收益或任务保持。

## 7. 后续最值得检查的范围

从当前证据出发，优先调查 action loop 的 Python/kernel launch 与小算子执行，以及 text/proprio cross-attention 的重复投影。源码中 chunk 内 context 固定，K/V 是否能精确复用值得做单独机制验证；同一任务文本的 encode_prompt 也存在重复调用。这些是后续候选，尚未实施或测得加速。

如果研究目标仍是 joint video-action WAM，必须得到匹配 Joint checkpoint，再按其原生 video/action schedule 重新测量。当前 direct-action baseline 的占比不能迁移为 Joint 的瓶颈结论。

## 8. 产物索引与限制

- `PROTOCOL.zh.md`：冻结协议。
- `artifacts/25663213.pbs101/report_metrics.json`：native/paired/shape/stage/profiler 统计。
- `artifacts/25663213.pbs101/resolved_config.yaml`、`results.json`、`job.log`、`terminal_receipt.json`：配置、任务、GPU sampling、终态。
- `denoise_steps.csv` / `layers.csv`：细分表。
- `episode_chunks.csv` / `artifacts/25663473.pbs101/episode_detail.json`：已有 episode 的完整 CPU 边界明细与独立 render 诊断。
- Remote 完整证据：`/scratch/users/ntu/yguo017/fastwam-libero-bottleneck/artifacts/25663213.pbs101/`，包含 raw_spans、profile_summary 与 torch_profiler_trace。

原始 episode 的 instance render hook 在 reset 后未捕获新的 sim；environment.step 时间有效。补充 CPU 诊断用 class hook 验证并细分了实际 render；其时延范围单独报告。

## 9. IDM 补测结论

在新的同 A100、同 checkpoint、同观测的 12 对原生模式比较中，first_frame / IDM 完整调用 median 为 **643.19 / 1158.59 ms**，mean 为 **666.61 / 1202.63 ms**；配对比值均值为 **1.804**。不能把本报告的另一个闭环 episode 637 ms 直接当作配对分母。

IDM 原生 video/action denoiser 单轮 CPU wall median 为 **52.00 / 52.99 ms**，各执行十轮；实际等待时间由一个主要 action 阶段变成两个接近的阶段。较轻计时的十轮合计为 **519.56 / 566.19 ms**，该 pass 完整调用扰动 +3.67%。独立 profiler 的 GPU child kernels 则为 video **413.26 ms**、action **158.10 ms**，GPU 运算的最大阶段变为 video。

IDM action attention 的 K/V 为 **98 current-observation + 196 future-video + 32 action tokens**。ACSM 当前只减少其中部分 current-observation 边，不减少新加入的十轮 video denoise；整个 action mixed SDPA 仍约 **55.67 ms**。这继续限制了其在现有路径上的整段加速份额，不能迁移为 Joint 的结论。

GPU 补测作业 `25663625.pbs101` 在 measurement 全部完成后因简报汇总键名错误退出，原始 F / Exit_status=1 与缺失 PIPELINE_COMPLETE 保留；CPU-only `25663764.pbs101` F / Exit_status=0 完成恢复汇总，全部实验 gates 通过。详见 IDM 报告的终态、CSV 与原始产物索引。
