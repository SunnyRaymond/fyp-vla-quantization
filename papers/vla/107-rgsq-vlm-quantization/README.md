# 107 — RGSQ: Riemannian Geometry-Sensitive Quantization for Large Vision-Language Models

- 作者：Zhiping Wu, Dongdong Ren, Yangchengyu Zhou, Zhengjie Zhang, Wenbin Li, Hongbing Pan, Yang Gao
- 固定版本：arXiv:2609.25492v1，2026-09-21；14 页。按 arXiv preprint 阅读；PDF 页眉的 `VOL. 14, NO. 8, JULY 2027` 不能据此当成已发表期刊信息。
- 原文：[本地 PDF](paper-arxiv-v1.pdf) · [arXiv 摘要](https://arxiv.org/abs/2609.25492v1) · [arXiv HTML](https://arxiv.org/html/2609.25492v1) · [作者代码](https://github.com/RL-MIND/RGSQ)
- 定位：通用 VLM 的 post-training quantization（PTQ）；实验对象是图文理解，不是 VLA action head、world model 或机器人 closed-loop 控制。

## 一句话抓手

同样大小的量化误差，落在模型敏感的方向上可能造成更大损失。RGSQ 用图像和文本 token 各自的 activation/gradient 统计估计这个“方向敏感度”，据此选择量化坐标与校准目标，让低比特误差尽量避开敏感方向。

## 先建立直觉

把等误差曲线想成椭圆：沿短轴稍微移动就可能明显改变输出，沿长轴移动则影响较小。普通平方误差把所有方向都看成圆。本文的 Fisher/Kronecker 近似试图找出椭圆的方向和尺度；Riemannian whitening 再把椭圆变成圆，使原有 Euclidean PTQ solver 能在变换后的坐标中工作。这里的“曲率”是校准数据上的近似敏感度，不能理解成精确的全模型 Hessian。

## Background / Problem

VLM 把 vision tokens 和 text tokens 送入共享的 language backbone。它们的分布、数量及对任务损失的敏感度可能不同。只按统一的 Euclidean reconstruction error 或每个 modality 一个 scalar weight 校准，可能忽略同一 modality 内部的方向差异；低 bitwidth 时尤其明显。论文比较的对象包括 RTN、GPTQ、AWQ、SmoothQuant、MBQ、MQuant 等（Sections II、V）。

## Method：按论文 Section IV 的顺序读

1. **RMAM：建立 modality-aware metric（Eqs. 8–11，Algorithm 1）。** 对每层的 vision/text tokens 分别估计 activation covariance `A^(m) = XXᵀ/N_m` 和 gradient covariance `S^(m) = GGᵀ/N_m`。用平均 token gradient L1 norm 得到 modality 权重 `π^(m)`，融合成 `A = Σπ^(m)A^(m)`、`S = Σπ^(m)S^(m)`，以 Kronecker-Fisher 形式近似权重扰动的敏感度。校准需要 forward **和 gradient collection**。
2. **ROW：把加权误差改写为普通 Frobenius error（Eqs. 12–15）。** 核心式是 `||S^(1/2) ΔW A^(1/2)||²_F`。在变换后的坐标中评估误差，可借用现有 PTQ solver。这个等价针对论文定义的二次代理目标；不表示量化后的真实 task loss 与它严格相等。
3. **GGES：用 sparse Givens rotations 选坐标（Eqs. 19–23，Algorithm 2）。** 一次 2×2 rotation 混合一对 input channels，论文用四层稀疏旋转与离散角度搜索。全精度时 `W'X' = WX`；量化后则因坐标不同而产生不同方向的误差。Activation scale 另按 modality 权重校准（Eq. 24，Algorithm 3）。

`W3A16` 表示 3-bit weights、16-bit activations；`W4A8` 和 `W2A8` 以此类推。Table III 的默认 weight group size 是 128，activation 采用 symmetric per-token；读结果时不要把 bitwidth 不同的行直接相减。

## Main Results：先看测量对象，再看增益

| 位置 | 原文结果 | 该结果支持什么 |
| --- | --- | --- |
| Table II，LLaVA-OneVision-7B，W4A8 | 六 benchmark 均值：RTN 56.1、MBQ 63.1、MQuant 66.0、RGSQ 68.1；FP16 67.5 | 在这组任务和设置下，RGSQ 的报告均值最高；高于 FP16 的 0.6 点不应解读为普遍提升模型能力。 |
| Table II，Qwen2-VL-7B，W4A8 | MBQ 68.4、MQuant 71.3、RGSQ 73.8；FP16 73.8 | 同一六任务均值中恢复到报告的 FP16 水平。 |
| Table V，LLaVA-OneVision-72B，W4A8 | MBQ 70.8、RGSQ 74.2；FP16 74.3 | 大模型上也有报告的精度恢复。 |
| Table IV，LLaVA-OneVision-7B，W4A8 ablation | 基线 56.1 → RMAM 64.1 → +ROW 66.6 → +GGES 68.4 | 各组件在这个 ablation 配置中的增量；完整行与 Table II 的 68.1 不一致。 |
| Table IX，LLaVA-OneVision-7B，W4A8 | RGSQ calibration 2.1 h / 38 GB；MBQ 45 min / 24 GB，MQuant 1.2 h / 28 GB | 较高的离线校准成本；不是 inference latency 或节省显存的测量。 |

Table VII 还报告 W2A8 / W3A8 在两个大 VLM 上优于 MBQ；Table VIII 是静态图文校准后的视频理解评测。它们都仍是 VLM benchmark evidence，不能外推为机器人动作质量。作者公开的 repository 明确将当前复现路径限定为 **Qwen2-VL-7B W4A8**；论文更广的模型、bitwidth 与视频结果不能由这一个公开路径自动视为可复现。

## Limitations / 原文核对点

- **表格内部不一致。** Table II 的 LLaVA-OneVision-7B W4A8 RGSQ 为 68.1，Table IV 完整 ablation 为 68.4；Table IX 同配置 RGSQ 为 68.1，但 MBQ/MQuant 的均值 65.8/66.6 与 Table II 的 63.1/66.0 不同。正文也写 InternVL2-26B W4A8 RGSQ 为 74.5，而 Table V 是 74.6。可能有设置差异或排版错误，原文没有在这些位置给出清晰解释；引用时注明表号。
- **极低 bitwidth 的参照值需核对。** Table VII 的 Qwen2-VL-72B W3A8 为 82.4、InternVL2-26B W3A8 为 83.8，均高于 Tables V 中相应模型的 FP16 六任务均值 78.1、74.6。若 benchmark 集、协议或统计口径不同，不能直接并列；论文 Table VII 写的是六 benchmark average，需要进一步追查。
- **Runtime 说法需拆开。** Section IV-C 明说 activation rotation `X' = RᵀX` 必须在线应用，每 channel 有约 `4K` multiplications；Section V-H 又称 whitening 不增加 runtime operator。较窄的解释是 whitening 仅用于 calibration，不能据此说整个 RGSQ 推理零 overhead。正文没有给出完整 end-to-end latency/throughput 对照来量化在线 rotation 成本。
- **校准依赖数据与梯度。** 图文配对 calibration 来自 ShareGPT4V/COCO 风格 captions；若目标是 action-conditioned VLA 或分布不同的视频/机器人输入，需要重新验证 metric 与 bitwidth 效果。原文的 `Fisher` 是因子化 empirical proxy，并非精确 Fisher 或 task-loss 保证。
- **效果与部署指标分开。** Table IX 报的是 calibration time/peak memory。正文关于“hardware-friendly”的说法主要由稀疏 rotation 的复杂度论证支撑，不能替代压缩后模型大小、inference latency、吞吐或功耗实测。
- **公开代码范围。** 作者 README 当前仅提供 Qwen2-VL-7B W4A8 pipeline；其中还提醒评估时必须启用 `--w4a8 --a_group_size 128`，否则会用 FP16 activations。其余模型和低 bitwidth 的独立复现状态仍待核实。

## Why It Matters for the FYP

这是一篇 **VLM quantization 的几何敏感度 prior**：它给出用图像/文本 token 分区统计、gradient-aware calibration、sparse rotation 和 whitening 降低低比特损失的组合。若考虑迁移到 LeWM / PushT 或 VLA，先明确 `token modality`、训练/规划目标及真正要保护的输出是什么，再设对应的 predictor/action/planner gate。VLM 六任务 accuracy 的提升本身不是 world-model rollout、CEM ranking、first action 或 closed-loop success 的证据。

## Reading Route

### 20 分钟：理解主张和边界

1. 看第 1 页 Abstract / Fig. 1，再看第 2 页 Fig. 2：用自己的话解释“误差大小相同，方向不同”。
2. 看 PDF 第 9 页 Tables II–IV 和第 10 页 Table V：锁定 model、bitwidth、benchmark average。
3. 看 PDF 第 12 页 Table IX 与第 13 页 Conclusion：区分 calibration cost 和 inference claim。

### 90 分钟：追踪方法是否闭合

1. 读 Sections III-B、IV-A（Eqs. 3–11）：`A`、`S`、`π` 从哪些数据与梯度得来？
2. 读 IV-B（Eqs. 12–15）：手推一次 `||S^(1/2)ΔWA^(1/2)||²_F`；注意它是局部代理目标。
3. 读 IV-C–D（Eqs. 19–24，Algorithms 2–3）：标出 offline 权重旋转、online activation 旋转、whitening 各自的位置。
4. 对照 Tables II、IV、V、VII、IX，记下上述数字不一致与实验边界。

### 3 小时：形成 FYP prior-art card

1. 阅读 [公开代码 README](https://github.com/RL-MIND/RGSQ)，区分“论文实验覆盖”与“公开复现覆盖”；只做阅读，不启动集群实验。
2. 写一张比较卡：`calibration signal / metric proxy / error orientation / quantized modules / offline cost / runtime cost / downstream metric`。
3. 在纸面上判断，若把同一 metric 用在 LeWM，哪个下游 gate 能检验“敏感方向”确实保留 planner 决策？不要把 VLM accuracy 当作替代指标。

## Reading Questions（留给你回答）

1. 为什么 `A` 和 `S` 要按 vision/text 分开估计，却又融合成单个 metric？什么信息会在融合时丢失？
2. `π` 来自 token gradient L1 norm；gradient 对 calibration loss、caption 模板、图文 token 数量有多敏感？
3. Fisher/Kronecker 代理与真实 task-loss curvature 的差距可能在哪些层最大？
4. Eq. 15 的 whitening 等价成立在怎样的变量变换和可行量化集合下？标准 PTQ solver 的约束是否保持不变？
5. 为何 rotation 要同时作用于 `W` 和 `X`？量化后为什么等价关系不再保证误差相同？
6. Givens pairing、四层深度与角度 grid 会怎样影响结果和校准时间？
7. 本文结果中，RMAM、ROW、GGES 哪个贡献最大？Table IV 能否证明三者在其他模型上也有相同排序？
8. Table II 与 Table IV 的 68.1/68.4 差异，原文是否提供不同设置的解释？
9. Table VII 的 W3A8 均值为什么高于 Table V 的 FP16？两表协议真的可比吗？
10. Section IV-C 的 online activation rotation 与 Section V-H 的 no-runtime-operator 说法如何同时成立？
11. Table IX 的 2.1 GPU-h、38 GB 只覆盖哪些 calibration 步骤；是否包含完整模型下载、数据准备和评测？
12. 公开 Qwen2-VL-7B W4A8 路径能检验论文中的哪些表项，哪些不能？
13. 若用于 VLA/WM，梯度应针对 caption likelihood、action error、rollout error，还是 planner decision；各选择会改变什么？
14. 这篇论文的 VLM benchmark evidence 离“机器人闭环仍可靠”还缺哪两层验证？

## Meeting Card

- **Paper**：Wu et al., *RGSQ*, arXiv:2609.25492v1（2026-09-21）。
- **Core idea**：以 modality-partitioned activation/gradient 因子构建 Kronecker-Fisher proxy，用 whitening 和 sparse Givens rotation 指导 VLM PTQ。
- **Strongest reported result**：LLaVA-OneVision-7B W4A8 的六任务均值 Table II：RGSQ 68.1 vs MBQ 63.1、MQuant 66.0；代价是 Table IX 中 2.1 h / 38 GB 的校准。
- **Critical caveat**：原文表间数字、低比特与 FP16 比较口径、在线 rotation 的 runtime 描述均需核对；公开复现只覆盖 Qwen2-VL-7B W4A8。
- **FYP connection**：可借鉴“按目标敏感方向”校准，但必须在 LeWM/PushT 上重新定义并验证 predictor、planner 与 closed-loop 指标。
- **讨论问题**：若目标是保护最终动作，VLM caption loss 的 Fisher proxy 是否选错了方向？
