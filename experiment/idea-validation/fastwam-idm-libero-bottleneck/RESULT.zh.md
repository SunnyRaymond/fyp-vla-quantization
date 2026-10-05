# Fast-WAM IDM：多轮 video denoise 与单轮 forward 的延迟

日期：2026-10-03。GPU 模型测量已完成，后处理由 CPU allocation 恢复完成；所有输出一致性、配对数量和 runtime geometry 检查通过。主 GPU 作业的后处理失败状态保留，见第 8 节。

## 结论

在同一 A100 40GB、同一 Optional IDM checkpoint、相同 observation/proprio/text/RNG 的 12 对原生调用中，**IDM 的整段耗时约为 first_frame 的 1.804 倍**。完整调用中位数分别为 **1158.59 ms / 643.19 ms**；均值为 **1202.63 ms / 666.61 ms**。

一轮原生 video denoiser 的 CPU wall 中位数约 **52.00 ms**，一轮 action denoiser 约 **52.99 ms**。两者都执行 10 轮，所以 video 阶段约 520 ms、action 阶段约 530 ms；这里的十轮数值是每轮中位数乘 10 的近似。较轻计时的实际逐轮区间合计为 **519.56 ms / 566.19 ms**，其完整调用扰动为 +3.67%。

因此，按实际等待时间看，**IDM 出现了两个接近的大阶段，action 仍稍高**；按单次 profiler 的 GPU child-kernel 累计看，**video denoise 为 413.26 ms，action 为 158.10 ms，GPU 运算的最大阶段转向 video**。两种计时口径回答不同的问题。

当前 ACSM idea 只减少 action queries 读取部分 current-observation K/V 的边，不触及新增的十轮 video denoise。IDM 的 action mixed-attention 区间仍约 **55.67 ms**，其 current-observation 边又只是这个范围的一部分。此次测量不支持把 ACSM 当作该 IDM 路径的大幅整段加速方案。

## 1. Baseline 与执行路线

- Source revision：`7faa71108368fbb3b6885649f112af607427a2d4`，固定 archive/revision marker。
- 模型 class：`FastWAMOptionalIDM`；checkpoint 为已有的 `libero_optional_idm_2cam224.clean.pt`，对应原 dataset stats。
- `action_infer_mode=idm` 调用官方 `FastWAMIDM.infer_action` 路径。**这是同一 Optional IDM 权重的模式比较，不是另一个专门训练的 IDM checkpoint 的性能复现，也不是 FastWAM-Joint。**
- A100-SXM4-40GB；GPU UUID `GPU-6e6d271e-6af6-836c-f753-82043b78acfe`，compute node `x1000c0s0b0n0`。请求 1 GPU、16 CPUs、110 GB、30 分钟；GPU/VRAM 每 15 秒写日志。
- BF16，seed 42，sigma shift 1，CFG 1，compile=false，两个模式各保留原生 10 轮 action denoise。预测 32×7 actions，每次最多执行 10 步。
- `libero_goal` task0/trial0：`open the middle drawer of the cabinet`；30 waiting steps；OSMesa CPU rendering。

| 模式 | Video denoise | Video cache prefill | Action denoise | Future video |
|---|---:|---:|---:|---|
| first_frame | 0 轮 | 当前图像，1 次 | 10 轮 | 不生成 |
| idm | 10 轮 | 当前图像与生成的未来 latent，1 次 | 10 轮 | 生成 latent，不 decode pixels |

IDM 先生成未来 video latents，再冻结它们作为 action denoising 的 condition。配置为 9 个 video frames，VAE 后为 3 个 latent frames；输出 latent shape 为 `[1,48,3,14,28]`。

**“Single forward”在本文指原生十轮中的一次 `_denoise_video` 或 action denoiser 调用。** 它不等于完整 policy 只执行一次网络；没有改成 K=1，也没有验证单轮 policy 的任务质量。

## 2. 同观测配对：完整调用增加多少

从真实 IDM episode 的 replan **0、6、11** 取三个观测，每个模式每个观测先 warmup 3 次，再按 ABBA 顺序做 4 对调用。观测、任务文本、proprio、RNG 相同；比较模式耗时时移除逐层 hooks。跨模式动作可以不同，只对同一模式的重复与 instrumentation 做 parity。

| 原生完整调用 | n | CPU wall mean ± SD（ms） | CPU median（ms） | CUDA event mean（ms） |
|---|---:|---:|---:|---:|
| first_frame | 12 | 666.61 ± 34.99 | 643.19 | 666.60 |
| idm | 12 | 1202.63 ± 66.25 | 1158.59 | 1202.61 |
| 每一对 IDM / first_frame 比值 | 12 | 1.8039 ± 0.0050 | 1.8013 | 比值均值 1.8039 |

均值差为 **536.01 ms**，配对比值约 **+80.4%**。数据是三个固定观测上的重复测量，不是 12 个独立任务。

Context 0 的两个模式都较慢：first_frame 约 713–715 ms，IDM 约 1291–1293 ms；其他两个 context 约 642–645 / 1156–1160 ms。全部样本保留，配对比值仍集中在 1.799–1.812。现有记录不能确定这种共同变慢来自 GPU clocks、运行阶段还是其他因素，因此不作原因归因。不能拿此前另一个 episode 的 637 ms 与本次 IDM 直接当作配对比值。

## 3. Single forward 与十轮循环

### 原 episode 的 CPU 边界

| 阶段 | 单次区间中位数（ms） | 调用数 | 解释 |
|---|---:|---:|---|
| VAE.input_encode | 10.02 | 12 | 当前图像；统计包含首次冷调用 |
| text.encode_prompt | 33.27 | 12 | 每个 chunk 重新编码文本 |
| video.denoise_step | 52.00 | 120 | 每个 chunk 10 轮 |
| video.scheduler_step | 0.051 | 120 | 单独 scheduler 更新 |
| video.cache_prefill | 51.40 | 12 | 每个 chunk 一次 |
| action.denoise_step | 52.99 | 120 | 每个 chunk 10 轮 |
| 图像/proprio 预处理 | 3.79 | 12 | CPU 边界 |
| 动作反归一化 | 0.13 | 12 | CPU 后处理 |

以上是所有 episode 调用的中位数，包含首次冷调用；不把各项中位数相加当作真实 chunk 总时间。首次完整 chunk **18.379 s**；后续 11 个 chunk 的原生 mean **1165.90 ms**、median **1165.51 ms**。

### 较轻计时：逐轮合计

较轻 pass 共 6 个 chunks，每个 denoise step 有 6 个观测重放样本。以下为 inclusive CUDA event 区间，包含 stream gaps；只列相应阶段与其明确标记的 children。

| 阶段 | 每个 chunk 合计（ms） | 每轮均值（ms） |
|---|---:|---:|
| 10 轮 video denoise | 519.56 | 51.96 |
| Video scheduler 10 步 | 0.57 | 0.057 |
| Video cache prefill 1 次 | 55.09 | — |
| 10 轮 action denoise | 566.19 | 56.62 |
| 其中：action mixed SDPA 300 次 | 55.67 | — |
| 其中：cache prefill SDPA 30 次 | 5.50 | — |

Video 第 0–9 轮分别为 **52.17、51.93、51.85、51.93、51.96、51.97、51.91、51.97、51.95、51.92 ms**；action 各轮约 56.50–56.75 ms。没有某一轮独占延迟。

十轮 video forwards 的工作约为其中单轮的 **10 倍**；整段 policy 的比例却只有 **1.80 倍**，因为两个模式都保留图像编码、文本编码、cache prefill、十轮 action denoising 和后处理。不能把一次 forward 的 52 ms 当作完整单轮 policy 的端到端 latency。

## 4. Instrumentation 的代价

| IDM 完整调用 | n | CPU mean（ms） | 相对同批 native 扰动 |
|---|---:|---:|---:|
| 逐层组的 native 参照 | 12 | 1159.12 | — |
| Fine hooks | 12 | 2323.70 | +100.47% |
| Thin 组的 native 参照 | 6 | 1161.35 | — |
| Thin hooks | 6 | 1204.00 | +3.67% |

IDM 的 12 对 native/fine、6 对 native/thin，以及 first_frame 的 3 对 native/fine 检查通过 finite/allclose；同模式重复与 episode 对应动作的检查也通过。BF16 只在诊断副本转换为 FP32 后交给 NumPy 比较，模型推理 dtype 保持 BF16。

Fine 层数据用于定位，不能作为原生 latency share。Thin 主要增加 action mixed-attention 的边界计时代价；因此原 episode 的 action 每轮约 53 ms，thin 下约 56.6 ms。CPU enqueue wall、CUDA event 与独立 profiler kernel sums 分别报告，不能拼接或相减得到精确 CPU 开销。

## 5. 最大耗时部分有没有变

### 实际等待时间

此前 first_frame 原生暖 chunk 约 637–640 ms，十轮 action denoising 的 CPU 边界约 523 ms，是主要阶段。本次 IDM 增加了约 520 ms 的 video denoising；action 仍约 530 ms，**从一个明显大阶段变成两个接近的大阶段**。按较轻 pass，action 的 566 ms 仍稍高于 video 的 520 ms。

### GPU 运算

单次独立 profiler 的 CPU record_function scope 所关联的 GPU child-kernel 累计如下。聚合前排除了 `FW/` 的独立 CUDA annotation events；各父子项 inclusive，不能重复相加。

| Scope / 主要模块 | Video denoise 10 轮（ms） | Action denoise 10 轮（ms） |
|---|---:|---:|
| 整个阶段的 GPU child kernels | 413.26 | 158.10 |
| FFN parents，30 层合计 | 120.99 | 10.97 |
| Text/proprio cross-attention parents | 101.41 | 55.28 |
| Self-attention Q projection | 17.22 | 4.04 |
| K projection | 17.22 | 3.92 |
| V projection | 17.20 | 3.80 |
| Output projection | 17.25 | 4.59 |
| Video self SDPA / action mixed SDPA | 19.70 | 14.68 |

Cache prefill 的 GPU kernels 为 **40.72 ms**，其中 FFN **12.09 ms**、text cross-attention **10.13 ms**；VAE 与 prompt scopes 分别为 **9.20 / 19.30 ms**。

**GPU 运算最大阶段确实变成 video denoising。** 在已单列的 video 主要模块中，FFN 和 text cross-attention 最大；action 内部仍是 text/proprio cross-attention parents 更大。FFN 和 cross-attention 都包含自己的线性投影与其他 children，不能再加上这些 children。

两个阶段每轮等待时间接近，但 GPU kernel 工作不同。小矩阵与许多算子的 launch/组织开销是 action 路径值得调查的候选解释；现有 trace 有扰动，不能据此精确声称有多少毫秒能由 CPU 或 launch 优化消除。

## 6. 对 ACSM idea 的含义

Runtime geometry 已检查：

- 每个 latent frame **98 tokens**；IDM video cache **294 tokens**。
- Video self-attention Q/K/V：`[1,294,3072]`。
- Video text cross-attention：Q `[1,294,3072]`，K/V `[1,129,3072]`。
- Action mixed attention：Q `[1,32,3072]`，K/V `[1,326,3072]`，24 heads×128。
- Mixed K/V 的来源为 **98 current-observation + 196 generated-future-video + 32 current-action tokens**。不能把全部 294 video tokens 都称为当前 observation。

按 MAC=2，action-query→current-observation 的 QK/AV 仍为每层每轮 **38,535,168 FLOPs**，30 层×10 轮共 **11.56 GFLOPs/chunk**；整个 mixed QK/AV 为 **38.46 GFLOPs/chunk**。Current-observation edges 的比例从 first_frame 的 75.4% 降为 IDM 的 **30.1%**，这是边计算比例，不是 latency 比例。

在同一个 thin pass 中，整个 mixed SDPA 约 55.67 ms，相对完整 1204.00 ms 为约 **4.6%**；它还包含 future-video/action edges、mask/layout 与 timer/stream gaps。Current-observation 子矩阵没有独立 timer，不能按 30.1% 线性切分其耗时。

ACSM 不减少十轮 video forwards，不删 future-video/action edges，也不免除 action 的 QKV/FFN、text/proprio cross-attention、首次 dense selection、packing 与原生 solver。该 IDM 路径中，idea 的可干预范围更窄；目前没有测得 ACSM 净加速、任务保持或 formal NO-GO。

## 7. 闭环环境与冷启动

本次真实 IDM episode：**1/1 success**，duration **63.862 s**，149 environment steps（含 30 waiting steps）、12 action chunks。单 episode 只作为运行和计时证据，不构成全 suite 成功率或两种 policy 的质量比较。

Environment.step mean **153.09 ms**、median **152.66 ms**；十个动作约对应 1.53 s 环境时间。图像来自 OSMesa 软件渲染，这不是机器人执行延迟或 EGL/GPU rendering 的速度。原 first_frame 诊断中的独立 CPU render 细分见相邻目录报告，其比例不能精确套到本次轨迹。

首次 chunk 18.379 s 与暖态约 1.166 s 必须分开。Episode 总时长还包含 environment create/reset、等待动作、模型冷调用、MP4 编码等，不能拿它直接评价稳态模型加速。

## 8. 作业终态与后处理恢复

| PBS job | 终态 | 说明 |
|---|---|---|
| 25663567.pbs101，GPU | F / Exit_status=1 | Episode 成功；BF16 tensor 的 NumPy 诊断转换报错，重放未完成 |
| 25663625.pbs101，GPU | F / Exit_status=1 | Episode、全部重放及 profiler 完成；profile_summary.status=complete；简报汇总器读错两个 nested parity keys |
| 25663741.pbs101，CPU | F / Exit_status=1 | 真正实验 gates 全通过；额外 512KB 报告大小限制导致退出 |
| 25663764.pbs101，CPU | **F / Exit_status=0** | 修正汇总键名，取消非实验性的字节限制；全部 gates 通过，生成最终 compact 报告 |

**主 GPU job 原始 wrapper exit=1、没有 PIPELINE_COMPLETE，这些状态没有被修改。** 完成证据来自 GPU 内完整的 measurement artifacts 与 CPU-only 恢复作业的 `DIAGNOSTIC_POSTPROCESS_COMPLETE` / exit=0。恢复只读取现有 summary/results/MP4，未重跑模型或 episode，也没有放宽输出一致性或 geometry 检查。

主 GPU `TASK_SUCCESS`、成功 results JSON 和非空 MP4 已检查；manifest 中 MP4 为 **219,800 bytes**。GPU util/VRAM samples 在 job.log。模型测量、完整 raw spans（约 74 MB）和 Chrome trace（约 175 MB）保留在 compute-side artifacts，未通过 login node 传回大型文件。

## 9. 产物

- `PROTOCOL.zh.md`：固定 source/checkpoint、两种模式、测量与 allocation 协议。
- `retrieved/finalization-25663764.pbs101/report_metrics.json`：最终统计与检查证据，1,645,771 bytes。
- 同目录 `terminal_receipt.json`、`job.log`、`exit_code.txt`、`DIAGNOSTIC_POSTPROCESS_COMPLETE`：CPU 恢复终态。
- `retrieved/25663625.pbs101/`：GPU job 原始配置、results、log、manifest、失败 terminal receipt。
- `paired_modes.csv`：12 对逐次完整调用。
- `denoise_steps.csv`：20 行，video/action 各十轮的 thin CPU/CUDA inclusive 均值。
- `layers.csv`：540 行，action denoise、video denoise、video cache prefill 各 30 层的 Q/K/V/O、FFN、cross-attention；来自高扰动 fine pass。
- `kernel_parents.csv`：同一独立 trace 的主要层 parent kernel 累计，保留 stage。
- `native_episode_stages.csv`：原 episode CPU 边界统计，含 cold call，不能把 mean 误写成暖态 mean。
- Remote 完整证据：`/scratch/users/ntu/yguo017/fastwam-idm-libero-bottleneck/artifacts/25663625.pbs101/`。

此前 first_frame 和 CPU rendering 报告：`../fastwam-libero-bottleneck/RESULT.zh.md`。
