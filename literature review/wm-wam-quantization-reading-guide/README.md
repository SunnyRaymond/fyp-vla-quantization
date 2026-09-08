# World Model / World Action Model 量化：阅读与 baseline 选择

来源核验：**2026-09-08**。本库包含 **24 份本地论文 PDF、24 篇中文导读**，并按核心、扩展、复现配套和 benchmark 分组。保留 English technical terms，Reading Questions 与 Meeting Cards 留待你读后填写。论文已下载；模型权重、训练数据、GPU 实验不包含在本次交付中。

**先读结论：LeWorldModel 很可能是推荐者所指的小型 baseline，但单凭“Yann LeCun 的开源模型”不能唯一定位；DINO-WM、V-JEPA 2-AC、JEPA-WMs 也符合。WM/WAM 量化已有直接论文，尤其要先看 QuantWM 与 QuantWAMs，再确定 novelty。**

[PDF 与版本清单](PDF_INVENTORY.md) · [Benchmark 地图](BENCHMARKS.md) · [量化 prior art 与选题边界](QUANTIZATION_AND_BASELINES.md) · [来源与开放程度核验](SOURCE_AUDIT.md) · [校验结果](VALIDATION.md)

## 你被推荐的工作，如何识别

| 推荐者提到的线索 | 最吻合的工作 | 用途 |
|---|---|---|
| 约 15M、单 GPU 数小时、Gaussian regularizer、从 pixels 端到端训练 | [01 LeWorldModel](papers/01-leworldmodel/README.md) | 小规模 latent WM 量化与机制验证 |
| 冻结 DINOv2、预测 patch features、PushT / Wall、zero-shot planning | [02 DINO-WM](papers/02-dino-wm/README.md) | 与已有 WM PTQ 对照 |
| Meta、看大量 internet videos、DROID、Franka 机械臂 | [04 V-JEPA 2 / 2-AC](papers/04-v-jepa-2/README.md) | 大规模 video encoder 与 latent robot planning |
| 系统研究何种 encoder / predictor / planner 更好 | [06 JEPA-WMs](papers/06-jepa-wms/README.md) | 更强、公开的 latent planning recipe |

这些均有 Yann LeCun 作者信息。LeWM 的官方仓库是 **lucas-maes/le-wm**；搜索也会返回保留同一 README 的 fork，不能仅凭 README 自称 official 认定来源。V-JEPA 2.1 是独立的新训练 recipe，见 [05](papers/05-v-jepa-2-1/README.md)。

## 先把 WM 与 WAM 的实验对象说清楚

| 家族 | 预测什么 | 决策发生在哪里 | 本库代表 |
|---|---|---|---|
| Latent predictive WM | 下一步或多步 latent state，条件含 action | CEM/MPC 搜索 action sequence | LeWM、DINO-WM、JEPA-WMs、V-JEPA 2-AC |
| Model-based RL | dynamics、reward、value 等 | imagination 中训练 policy，或部署时 planning | DreamerV3、TD-MPC2 |
| Joint video-action WAM | future video/latent 与可执行 action | 模型输出 action，可能附加 planning | DreamZero、LingBot-VA、Cosmos Policy |
| 使用 world-model co-training 的高效 policy | 训练时视频与动作，测试可省略 future generation | action head / policy inference | Fast-WAM |
| Interactive video/world generator | RGB/video、camera-conditioned future | 输入可能是键鼠或 camera motion | SANA-WM、Matrix-Game 3.0 |

WAM 不是完全统一的架构标准，作者使用范围不同。请在报告中直接写清 training outputs、inference outputs、action conditioning 和是否闭环。**VQ/FSQ tokenization 是表示离散化；本专题研究的 low-bit weights/activations/KV 是另一种“量化”。**

## 优先阅读：先建立问题，再选择模型

| 顺序 | 阅读项 | 本轮要拿到的判断 |
|---|---|---|
| 1 | [LeWM](papers/01-leworldmodel/README.md) + [TwoRoom 独立复现](papers/18-lewm-reproduction/README.md) | 小型实验是否可做，评估协议有什么坑 |
| 2 | [DINO-WM](papers/02-dino-wm/README.md) → [QuantWM](papers/03-quantwm/README.md) | 现有 latent WM PTQ 已经回答什么 |
| 3 | [QuantWAMs](papers/23-quantwams/README.md) | WAM PTQ 已有 joint saliency、closed-loop calibration 和 mixed precision |
| 4 | [JEPA-WMs](papers/06-jepa-wms/README.md) | DINO-WM 之外更强的 latent planning 对照 |
| 5 | [Fast-WAM](papers/10-fast-wam/README.md) + [Faster-WAM](papers/11-faster-wam-future-conditioning/README.md) | 效率与 OOD robustness 是否冲突 |
| 6 | [Cosmos Policy](papers/12-cosmos-policy/README.md) / [LingBot-VA](papers/08-lingbot-va/README.md) | 选择可接到机器人 benchmark 的 WAM |

QuantWAMs 本地 v1 引用了未随 PDF 附上的 Appendix A–E，项目页尚未给出可用的外部 Code link。本库把它列为**必须读的 prior art**，但不标成已核实可复现的开源工具。

## 其他强方法与前沿阅读

“SOTA 级”在这里表示有明确强结果或重要架构贡献，**不表示在统一、同数据同预算的全领域排行榜上排名第一**。

| 工作 | 推荐定位 | 公开资源与主要边界 |
|---|---|---|
| [V-JEPA 2 / 2-AC](papers/04-v-jepa-2/README.md) | 大规模 latent WM | code / weights；真实机器人协议与 LIBERO 不同 |
| [V-JEPA 2.1](papers/05-v-jepa-2-1/README.md) | 更新的 dense video features | 不能只换 encoder 就假设旧 predictor 兼容 |
| [DreamZero](papers/07-dreamzero/README.md) | 14B joint WAM、泛化与跨 embodiment | 有 code / weights；当前官方 server 路线计算要求高 |
| [LingBot-VA](papers/08-lingbot-va/README.md) | causal video-action、async control | 有 code / weights；LIBERO-Long 与全四套需区分 |
| [LingBot-VA 2.0](papers/09-lingbot-va-2/README.md) | native pretraining、MoE、tokenizer | 报告公开，完整 2.0 可执行 release 本次未确认 |
| [Cosmos Policy](papers/12-cosmos-policy/README.md) | 视频模型到 action/state/value | code / weights / data；base 与 planning 两种成本 |
| [Cosmos 3](papers/13-cosmos-3/README.md) | omnimodal frontier family | 公开模型和工作流；具体 policy checkpoint 与 backend 单独核对 |
| [DreamerV3](papers/14-dreamerv3/README.md)、[TD-MPC2](papers/15-td-mpc2/README.md) | RL 基础与强 baseline | 成熟开放实现；不是当前全 benchmark 的统一最新冠军 |
| [SANA-WM](papers/16-sana-wm/README.md)、[Matrix-Game 3.0](papers/17-matrix-game-3/README.md) | 视频 WM 的低精度与 streaming | 已有 low-precision 路线；不同于机器人 manipulation WAM |

## 分时路线

**2 小时定位**：本页与 benchmark 地图 15 分钟 → LeWM 25 分钟 → QuantWM 25 分钟 → QuantWAMs 30 分钟 → TwoRoom 配套 10 分钟 → 填组会卡 15 分钟。目标是选家族，不是读完 24 篇。

**1 天（约 6 小时）**：上述 2 小时 → DINO-WM/JEPA-WMs 对比 75 分钟 → Fast-WAM/Faster-WAM 对比 60 分钟 → Cosmos Policy/LingBot-VA 选一篇 60 分钟 → 检查 code/weights/benchmark 对应关系并写下一步 45 分钟。

**1 周（约 12–16 小时）**：Day 1–2 完成 latent WM 与 quantization 核心；Day 3 精读 QuantWAMs 与 #24 mixed-bit companion；Day 4 读一种 WAM；Day 5 比较 ID/OOD benchmark；Day 6 查选定 baseline 的 inference modules 与 calibration data；Day 7 用以下组会卡做选择。大规模 pretraining 和所有 24 篇通读不在这条一周路线内。

## 组会卡（自己填写）

- 我准备量化的是 latent predictor / video backbone / action head / KV / 其他：
- 我的 baseline 与具体 checkpoint：
- 现有 QuantWM / QuantWAMs / mixed-bit study 已覆盖的部分：
- 一个比“WM 量化”更具体的 hypothesis：
- 成功指标与 failure mode：
- 主 benchmark 与一个 orthogonal/OOD benchmark：
- 需要的数据与校准信息；是否超出对照方法：
- 模型级收益与 end-to-end 收益分别怎么测：
- 第一项能证伪 idea 的低成本实验：
- 想向推荐者确认的模型名称或项目链接：

## 全部目录与原阅读库

所有 24 项含本地 PDF 与导读，见 [PDF_INVENTORY.md](PDF_INVENTORY.md)。新增四篇 benchmark 为 [WorldArena](papers/19-worldarena/README.md)、[WorldArena 2.0](papers/20-worldarena-2/README.md)、[WorldScore](papers/21-worldscore/README.md)、[Physics-IQ](papers/22-physics-iq/README.md)；mixed-bit companion 为 [24 Where Bits Matter](papers/24-where-bits-matter/README.md)。

[返回主 Reading Guide](../reading-guide/README.md) · [原 simulation evaluation 库](../sim-eval-vla-reading-guide/README.md) · [原 control/world-model 方向](../vla-ideas-reading-guide/03-control-distillation-and-world-models/README.md)。本库编号独立，不改变主库 24 篇 core 的含义。
