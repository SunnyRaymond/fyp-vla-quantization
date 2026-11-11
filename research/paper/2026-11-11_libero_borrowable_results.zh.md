# 标准 LIBERO：Fast-WAM-IDM 与 Cosmos-Policy 可借用的已发表结果（2026-11-11 检索）

检索范围：本地 `papers/wam`、`papers/vla` 全部 WAM/VLA 量化论文；Semantic Scholar 上 Fast-WAM（478 篇）与 Cosmos-Policy（452 篇）的全部引用论文按标题筛量化/加速/部署相关后逐篇查正文；arXiv API 按摘要关键词（quantization × WAM/LIBERO/Cosmos/Fast-WAM）兜底。截止 2026-11-11。

我们的两个模型：Fast-WAM **Optional-IDM 检查点，`idm` 模式**（= Fast-WAM 论文的 Fast-WAM-IDM）；Cosmos-Policy **`nvidia/Cosmos-Policy-LIBERO-Predict2-2B`**（280 个 block Linear）。

## 1. Fast-WAM-IDM

**全精度（可借）**

| 来源 | Spatial | Object | Goal | Long | Avg | 回合 |
|---|---|---|---|---|---|---|
| Fast-WAM 论文 Table 2（arXiv 2603.16666v2） | 98.8 | 97.8 | 97.8 | 97.6 | 98.0 | 2000（40 task × 50） |
| WAMachine Table 2（arXiv 2609.34608，09-28），Native | 99.27 | 99.67 | 98.47 | 97.00 | 98.60 | 6000（50 初始状态 × 种子 195/196/197） |

**量化对比方法：没有任何工作在 IDM 变体上跑过。** 现有 Fast-WAM 量化数都在别的变体上，不能放进 IDM 行：

- QuantWAMs（2607.28405，07-30）：**action-only Fast-WAM**（测试时不生成未来），FP16 98.2/99.8/97.0/95.2/97.6；GPTQ W4A16 96.0、SmoothQuant W8A8 96.4、SVDQuant W4A4 73.7 / SVDQuant* 75.3、Atom W4A4 76.2 / Atom* 82.1、QuantWAMs W4A4 97.4（3 个 protocol seed，共 6000 回合）。注意其表头写作 Goal/Spatial/Object，按 FP16 行对照 Fast-WAM 原表，实际顺序是 Spatial/Object/Goal。
- PreDE（2609.19441，09-16）：Fast-WAM-6B，官方评测脚本默认模式（未切 IDM），只有四 suite 平均：bf16 95.55、SVDQuant W4A4（nunchaku int4）95.75、HQQ W4 g64 95.55、HQQ W3 90.15。
- SteerQuant（2609.39056，09-30）：**FastWAMJoint**（未来与动作联合去噪），FP 99.2/99.4/98.6/98.4/98.9；W4A4 QuaRot 98.7、Atom 98.2、SVDQuant 98.2、SteerQuant 98.4（2000 回合）。
- Q-WAM（2609.33269）：只有 RoboTwin 2.0，没有 LIBERO。

本项目自己：没有 Fast-WAM-IDM 的 BF16 标准 LIBERO 全量结果；`fastwam-libero-engine-20261027` 的 391/400 是 C80 MXVQ 量化模型，不是 BF16。

## 2. Cosmos-Policy

**全精度（可借）**

| 来源 | Spatial | Object | Goal | Long | Avg | 回合 |
|---|---|---|---|---|---|---|
| Cosmos-Policy 论文 Table 1/4（2601.16163） | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 | 6000（500/suite × 3 种子） |
| SteerQuant Table 1（FP Original） | 98.3 | 100.0 | 97.9 | 98.0 | 98.5 | 3000（40 task × 25 × 3 种子） |
| PreDE Table I（bf16） | — | — | — | — | 98.38 | 3 种子 |
| WAMachine Table 2（Native） | 97.40 | 99.60 | 97.67 | 97.53 | 98.05 | 6000，种子 195–197 |
| MARS Table 1（2610.17141，10-12，同期） | 97.47 | 99.87 | 98.20 | 96.67 | 98.05 | 2000 × 3 种子 |
| SmolWAM Table 1（2610.13740，10-09，同期） | 97.0 | 100.0 | 99.0 | 97.0 | 98.2 | 400，单种子 |

不同论文的全精度复现差到 0.45 pp，与量化损失同量级。

**量化对比方法（可借）**

SteerQuant Table 1（按 PDF 坐标还原，Cosmos-Policy 列）。设置：量化 self/cross-attn 的 q/k/v/o 与 FFN up/down（与我们的 280 Linear 范围一致），**KV cache、Norm、RoPE、Softmax、GELU 保持 BF16**；320 条 LIBERO 训练观测校准；3000 回合，方法间初始状态与种子配对。

| 精度 | 方法 | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|---|
| FP | Original | 98.3 | 100.0 | 97.9 | 98.0 | 98.5 |
| W4A8 | SmoothQuant | 98.0 | 99.7 | 98.4 | 95.6 | 97.9 |
| W4A8 | PTQ4DiT | 98.8 | 99.6 | 96.1 | 97.7 | 98.1 |
| W4A8 | Q-DiT | 97.5 | 99.9 | 96.8 | 96.7 | 97.7 |
| W4A8 | SteerQuant | 98.3 | 99.9 | 98.1 | 98.0 | 98.6 |
| W4A4 | QuaRot | 98.2 | 99.2 | 96.1 | 98.1 | 97.9 |
| W4A4 | Atom | 96.2 | 98.7 | 96.6 | 97.3 | 97.2 |
| W4A4 | SVDQuant | 98.4 | 99.7 | 96.7 | 97.7 | 98.1 |
| W4A4 | SteerQuant | 98.3 | 99.7 | 97.7 | 97.5 | 98.3 |

SteerQuant 自身权重为对称 per-output-channel W4；附录未写 QuaRot 如何处理 AdaLN，也未写各基线的权重粒度。

PreDE Table I（只有四 suite 平均；官方检查点 commit 18a2acc，280 个 DiT Linear，embedding/adaLN/final 保持 bf16，50 trials/task，3 种子）：bf16 98.38、ViDiT-Q W8A8 98.12、SVDQuant W4A4（nunchaku）97.98、HQQ W4 98.20、HQQ W3 97.87。仓库 `quant/results/libero_logs/` 的日志未公开，拿不到逐 suite 数。

**对照我们 Table 1 的基线行**：QuaRot ✓（SteerQuant）、SVDQuant ✓（SteerQuant 逐 suite；PreDE 只有平均）、ViDiT-Q 只有 W8A8（PreDE），没有 W4A4、SpinQuant ✗、AQLM ✗。另外可加 Atom 与 SteerQuant 本身（专用 WAM 量化器，无代码，借数是唯一的比较途径）。

## 3. 查过但不可用

VLAQuantBench（2609.25376，仅 VLA）、LoopQ（2611.05093，仅 VLA）、LatentQuant（2610.03959，WALL-WM 的 VAE）、Embodied.cpp（Cosmos3 与 LingBot-VA）、Long-WAM（自训模型，Fast-WAM 行抄原文）、WAMJET（我们两个模型只报延迟；FP8 只用在 DreamZero/Cosmos3/OpenWAM）、The Speedup Paradox（Cosmos-Policy 的 W8A8/W4A4 只在 RoboCasa）、WAM-Cache / C3ache / RealtimeWAM / GlanceWAM（只有 action-only Fast-WAM 或抄原文的 BF16，无量化）、Complementary Future Token Pruning（Cosmos3）、VLA-HWBench、PhyAI。

## 4. 借数时的约束

- 标准 LIBERO 饱和：p≈0.98 时 95% CI 半宽在 2000 回合约 ±0.61 pp、3000 回合约 ±0.50 pp、6000 回合约 ±0.35 pp；SteerQuant 表里 W4A4 各方法 97.2–98.3，基本落在彼此 CI 内，排不出名次。
- 借来的行只能和它自己论文的 FP 对比（报 Δ），不能和我们自己的 BF16 混算；也无法和我们的回合配对。
- 时间：SteerQuant、PreDE、WAMachine、QuantWAMs 均早于 2026-10，按 CVPR 规则属于已有工作；MARS、SmolWAM、WAM-Cache、RealtimeWAM 属于同期。
