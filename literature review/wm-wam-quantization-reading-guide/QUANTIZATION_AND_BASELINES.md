# 量化 prior art、baseline 与可探索问题

2026-09-08 · [返回总指南](README.md)。这是针对本次检索的选题地图，不是完成的 novelty certification，也未运行复现实验。

## “做得少”应怎样表述

可以说：**本次检索中，专门研究 latent planning / joint WAM 的低比特方法数量有限，尚未形成统一评估协议；但直接 WM/WAM PTQ 与工程低精度部署都已经存在。**

不应说：第一个 WM quantization、第一个 WAM PTQ、第一个考虑 rollout 的 WAM calibration，或第一个按 encoder/predictor 分配 precision。以下 prior art 已覆盖这些宽泛方向。

| 已有工作 | 已覆盖部分 | 尚不能由它证明 |
|---|---|---|
| [QuantWM](papers/03-quantwm/README.md) | DINO-WM PTQ、granularity、encoder/predictor sensitivity、planning loss 与 success | 所有 JEPA/WAM 的普适规律；真实端到端 kernel 收益 |
| [Where Bits Matter](papers/24-where-bits-matter/README.md) | paired mixed-bit、模块分配、planner budget | 多任务多平台普适最优；packed INT4 部署 |
| [QuantWAMs](papers/23-quantwams/README.md) | shared-basis pooling、joint saliency、closed-loop rollout auditing、混合精度、Blackwell block kernels | 所有模块纯 W4A4；端到端机器人 1.6×；本次未核实其公开实现 |
| [DreamZero](papers/07-dreamzero/README.md) | 大 WAM 的系统加速中包含 PTQ 等策略 | 整体收益等于单项 quantization 收益 |
| [SANA-WM](papers/16-sana-wm/README.md) | FP8/NVFP4 video WM deployment、camera-conditioned generation | robot manipulation 闭环 success |
| [Matrix-Game 3.0](papers/17-matrix-game-3/README.md) | INT8 路线与 distillation/decoder 优化 | 低比特独立贡献或跨 embodiment robot control |

相邻检索还发现 [MotuBrain](https://arxiv.org/abs/2604.27792) 的系统 recipe 含 FP8；它列入观察，不把多组件加速当作新的独立 quantizer。VQ/FSQ latent/action tokenizer 工作、world-model-assisted LLM offloading 和 WorldCache caching 不等于本专题的 weight/activation PTQ，但可能构成相关效率对照。

## Baseline 选择建议

| 目标 | 第一选择 | 第二对照 | 为什么 |
|---|---|---|---|
| 低成本解释量化为何破坏 planning | LeWM + PushT | DINO-WM + Wall/PushT | LeWM compact；DINO-WM 直接连到已有 PTQ |
| 更强 latent planning 的可迁移验证 | JEPA-WMs | 原 DINO-WM / corrected 2-AC | 已有系统 recipe 和多种公开模型；必须注明版本及 license |
| 与现有 VLA/LIBERO 经验衔接 | Fast-WAM 或 Cosmos Policy | LingBot-VA Long / Faster-WAM | 公开 checkpoint 与 simulation 入口；QuantWAMs 是直接 prior art |
| 研究 multimodal frontier model | Cosmos 3 | 指定的较小 policy family | 先检查具体 checkpoint 与 action interface；不是直接从全模型预训练起步 |
| 研究 video WM 的低精度/streaming | SANA-WM / Matrix-Game 3.0 | 同模型 BF16、同 step/decoder | benchmark 应含 camera/action fidelity 与 long-horizon consistency |

LeWM 小，适合验证机制，但可节省的绝对显存也小。想做有部署价值的 FYP，应证明方法能迁移到更大 latent model 或 joint WAM，或者利用省出的算力在固定 wall-clock 下获得更好的 planning。仅在 15M 模型上跑 W8A8 并保持 success，贡献可能较弱。

ASPIRE2A 若采用 A100，不能直接复现 QuantWAMs 的 Blackwell NVFP4/FP8 kernel 数字。可以先研究可在目标硬件实现的 weight-only INT4/INT8 或数值量化对照，但 **fake quant / dequantized floating GEMM 只说明数值影响，不说明真实低比特加速**。具体 kernel 支持、形状和数据布局要在实施时 profile。

## 值得进一步验证的三个假设

这些是待做 prior-art 深查和实验的 hypothesis，不声明为新方法，也不预先填答案。

### 1. 保留候选动作排序，比保留平均 latent MSE 更重要

LeWM/DINO-WM 在一批 candidate actions 中依据 latent goal cost 选择行为。量化可能保持平均 reconstruction error，却反转最优候选。可比较 ordinary MSE calibration 与 candidate-pair ranking preservation，再检查实际环境成功率。

与 QuantWM 的差别必须落到**明确的 ranking objective、calibration 数据产生方式和跨模型闭环验证**；单纯发现 loss 与 success 不一致已不是新发现。与 QuantWAMs 的 joint training-loss saliency 也要区分，不能只换名字。

证伪点：在相同 bits/bytes、相同 calibration episodes 下，ranking agreement 虽改善，闭环 success 和固定时间下的 planning 却没有改善。

### 2. 长时量化误差与 planner budget 的共同作用

区分 encoder goal-geometry bias、predictor recursive drift 和 CEM search variability。研究哪类误差会被更多搜索缓解、哪类会被优化器 exploit；比较固定 search budget 与固定 time budget 两种结果。

“模块混合精度 + 不同 budget”已被 #24 覆盖。进一步贡献需有可预测的误差机制、可执行的 budget/precision policy，以及更多任务与实际性能证据。

证伪点：在控制 total bytes、sampler seeds、goal pairs 后，结论不稳定或收益只是增加了计算量。

### 3. Future conditioning / memory 的量化与 OOD robustness

Fast-WAM 与 Faster-WAM 给出一个结构对照：省略未来计算，或保留稀疏未来 representations。量化是否会首先损伤 future-conditioned K/V 或 action-critical residual？可在 LIBERO 与 LIBERO-Plus 配对评估。

Closed-loop calibration、denoising-step protection 已被 QuantWAMs 覆盖；不能只提出“用闭环状态校准”。更具体的对象可以是缓存长期复用的误差、future/action 信息交互，或同部署预算下的 risk of losing future conditioning。

证伪点：future branch 的高精度保护不能优于同 bytes 的其他模块保护，或仅在一个特定训练 seed 上有效。

## 最小实施路线（准备建议，尚未执行）

1. 确定一个 checkpoint、一个 environment 和一批固定 goals；跑通 FP，不进行整模型重训。
2. 少量 episodes 做 smoke test，核对 action shape、normalization、frameskip、goal/termination。小样本只用于查错。
3. 做 encoder-only、predictor-only、joint 三种数值损伤对照；不要一开始扫几十个方法。
4. 若出现稳定 failure mode，再比较 simple RTN 与一个校准 baseline，使用分离的 calibration/test episodes。
5. 只有真实 kernel 路径可用后才报告 speedup；测 full planning/observation-to-action critical path，并报告失败、packing/scale overhead。
6. 通过后扩到第二环境或第二模型，才讨论 generality；paired trials 与多 seed 的正式预算另定。

## 近期观察与同名边界

- [Faster-WAM / DoT，2608.02365](https://arxiv.org/abs/2608.02365) 与本地 [Faster-WAM / Future Conditioning，2608.04404](papers/11-faster-wam-future-conditioning/README.md) 是两篇不同论文，不能互换 code/数字。
- [LAWA，2608.24882](https://arxiv.org/abs/2608.24882) 用 latent action 表达未来意图；摘要仍写 code/models will be released，本次列观察。
- [LingBot-World 2.0，2607.07534](https://arxiv.org/abs/2607.07534) 是 interactive world generation，区别于 LingBot-VA 2.0 和 LingBot-VLA 2.0。
- [DIAMOND](https://diamond-wm.github.io/) 是 diffusion WM/RL 基础支线；需要 Atari 路线时再补读。
- Dreamer4 有多个 third-party implementations，不能因名字相同就称为原作者 official code。本次保留清楚可核验的 DreamerV3 基础材料。
