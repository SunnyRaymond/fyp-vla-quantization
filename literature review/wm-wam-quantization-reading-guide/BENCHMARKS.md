# WM/WAM 常用 benchmark：按任务选择

核验日期：2026-09-08。[返回阅读入口](README.md)。这里的“常用”依据本次核对的代表性论文与官方实现，未做全领域 publication-frequency 统计；新兴 benchmark 单独标注。

## 机器人控制与 latent planning

| 类型 / benchmark | 测什么 | 哪些本库模型使用 / 适配 | 对量化实验的价值与边界 |
|---|---|---|---|
| PushT | 推动 T-shaped object 到目标姿态 | LeWM、DINO-WM、QuantWM | 便宜、接触敏感；goal-image planning 版与其他 policy 版的 success threshold 不能默认相同 |
| Wall / PointMaze / TwoRoom | 绕墙、过门、目标导航 | DINO-WM、LeWM、QuantWM、mixed-bit study | 很适合查 cost geometry 和 action ranking；三个名字不等于同一环境/数据 |
| OGBench-Cube | 3D cube manipulation、goal-conditioned behavior | LeWM | 补充复杂视觉/3D 接触；LeWM 用特定单 cube、pixels 设置，不是全 OGBench |
| [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) | 连续控制、locomotion、reaching | DreamerV3、TD-MPC2；LeWM 的特定 Reacher 变体 | episode return、sample efficiency；pixels 与 state-based 要分开 |
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | 多任务机械臂 manipulation | JEPA-WMs、TD-MPC2 等 | success、task transfer；任务集合、版本、goal/state 信息明确记录 |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | language-conditioned manipulation 与 transfer | Fast-WAM、Cosmos Policy、Faster-WAM；LingBot-VA 的 Long | 是当前机器人 WAM 最实用的共同入口之一；常见比较 Spatial/Object/Goal/Long 各 10 tasks，但完整 LIBERO 还包含其他 suite，不要把 40 tasks 称为所有 LIBERO 任务 |
| [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) | 视觉/语言/布局等扰动下 robustness | Faster-WAM 与近期高效 WAM | 高基线成功率下更易暴露量化问题；明确 perturbation、severity、trial 数 |
| [RoboTwin 2.0](https://robotwin-platform.github.io/) | 双臂操作、多任务与 domain randomization | Fast-WAM、LingBot-VA、Faster-WAM、QuantWAMs | 区分 Clean/Randomized、demo 数、robot embodiment；不能只报平均成功率 |
| [RoboCasa / RoboCasa365](https://robocasa.ai/) | 厨房与 household manipulation、组合长任务 | Cosmos Policy、JEPA-WMs 相关实验；更大规模路线 | 旧 subset 不等于 365 全套；官方 2026-05 更新 horizon，版本漂移会改变成功率 |
| Real robot：Franka / ALOHA / AgiBot 等 | sim-to-real、真实时延、未知物体与接触 | V-JEPA 2-AC、DreamZero、LingBot-VA、Cosmos Policy、QuantWAMs | 不是一个统一 benchmark；固定 task、reset、物体、trials 和人工 intervention 规则 |

PushT/TwoRoom/OGBench-Cube 的具体设置见 [LeWM](papers/01-leworldmodel/README.md)；Wall/PointMaze 的设置见 [DINO-WM](papers/02-dino-wm/README.md)。LIBERO/RoboTwin 的跨模型使用证据见对应 paper 的实验与已保存官方 README。

## Model-based RL 与 learned simulator

| Benchmark | 常见指标 | 适用对象 | 容易误读的地方 |
|---|---|---|---|
| Atari 100k | 每游戏 return、human-normalized score、median/IQM、置信区间 | DIAMOND、IRIS 类、Dreamer 相关研究 | 100k 环境交互预算、frame skip 和 raw frames 区分；不能把 mean HNS 当成所有游戏普遍改善 |
| DMControl | normalized return、sample efficiency、多 seed | DreamerV3、TD-MPC2 | 同环境步数不等于同算力；observation modality 要匹配 |
| Crafter / Minecraft | achievements、reward、sample efficiency | Dreamer 类 | 长时行为能力；训练 policy 的成本和部署 world model 的成本是两件事 |
| [WorldArena 2.0](papers/20-worldarena-2/README.md) 的 interactive RL track | 在 learned environment 中训练的 policy，回真实 simulator/机器人表现 | learned simulator | agent 可能 exploit WM error，不能只在模型自身世界里评成绩 |

[DIAMOND 官方项目](https://diamond-wm.github.io/) 与 [DreamerV3 code](https://github.com/danijar/dreamerv3) 是这条支线的入口。想做 manipulation WAM 时，不必为了显得全面把 Atari 强行加入主实验。

## 物理理解、预测与视频质量

| Benchmark / metric | 主要测量 | 能否作为 LeWM/WAM 量化主指标 |
|---|---|---|
| [IntPhys 2](https://github.com/facebookresearch/IntPhys2) | plausible / implausible 视频的 violation-of-expectation | 可作 latent representation 保真辅助；不能替代闭环成功 |
| [MVPBench](https://github.com/facebookresearch/minimal_video_pairs) | minimal video pairs，抵抗 shortcut 的 physical Video QA | 需要相应 QA interface；不是 action-conditioned control |
| [CausalVQA](https://github.com/facebookresearch/CausalVQA) | counterfactual、hypothetical、anticipation、planning、descriptive QA | 需要 language readout，数据许可/隐藏 test labels 会影响本地完整评测 |
| Something-Something v2 / EPIC-KITCHENS / Ego4D | action recognition / anticipation | V-JEPA encoder retention 指标；classification/anticipation 不能宣称 robot planning 保留 |
| [Physics-IQ](papers/22-physics-iq/README.md) | 需要物理原理的真实视频 continuation | 适合 video WM，latent-only model 需合理适配 |
| [WorldScore](papers/21-worldscore/README.md) | camera controllability、quality、dynamics | world generation 重点；不是机器人任务成功率 |
| [WorldArena](papers/19-worldarena/README.md) | 视频质量 + data engine / policy evaluator / planner utility | 新兴 embodied WM 专门评测，比仅视频指标更贴近功能 |
| [WorldArena 2.0](papers/20-worldarena-2/README.md) | 增加 visuotactile、interactive RL、real-robot 多平台 | 新协议，锁定 track 与 scoring revision，不混报 |
| [VBench](https://github.com/Vchitect/VBench) | 多维视频生成质量与一致性 | 辅助；高分不证明物理正确、action fidelity 或 control success |
| FVD / FID / LPIPS / PSNR / SSIM | 分布距离、perceptual / pixel reconstruction | 是 metrics，不是完整 benchmark；不应作为 manipulation 量化唯一主结论 |

WorldArena 已揭示 perception-functionality gap，Physics-IQ 强调 realism 与物理理解可以分离。因此“FVD 几乎不变”远不足以证明 WAM 量化无损。

## 对这个 FYP 的建议组合

**路线 A：latent WM quantization（建议先做）**

1. **LeWM + PushT**：先固定 checkpoint 和 FP reference，查 planning pipeline；TwoRoom 可作 smoke test，但别只凭这个简单环境发表结论。
2. **DINO-WM + Wall/PushT**：复核 QuantWM 与 mixed-bit study 的基本趋势，建立直接对照。
3. **LeWM + OGBench-Cube 或 JEPA-WMs + 一个官方 simulation task**：检验跨任务/跨模型是否成立。

主指标为 paired closed-loop success 与 end-to-end planning latency / peak memory；辅助看多步 latent drift、候选 ranking agreement、true environment goal distance。CEM population、iterations、rollout horizon、frameskip 和动作执行长度都要固定并分别记录。

**路线 B：joint WAM quantization**

1. **Fast-WAM 或 Cosmos Policy + LIBERO**：先复现适配 checkpoint 对应的 suite。若选 LingBot-VA 的公开 Long checkpoint，先只报告 Long。
2. **LIBERO-Plus**：测试低精度下的 OOD degradation，而非只追高饱和 ID 分数。
3. **RoboTwin 2.0**：补充双臂、不同动作空间和 domain randomization。

主指标为 per-suite/per-task success、P50/P95/P99 observation-to-action latency、peak resident memory、有效反馈频率；补充 action/video consistency。WorldArena 可作功能扩展，WorldScore 不宜代替主闭环结果。

## 报告中必须保留的最小条件

记录 checkpoint revision、env/asset revision、task IDs、train/calibration/test splits、seed/initial states、episode horizon、image resolution/cameras、precision scope、kernel/backend、GPU、batch size、planner/denoising budget、失败与 early stop 定义。使用相同 initial states 做 FP/quantized paired comparison，报告绝对百分点差异和不确定性。混合精度同时报告 parameter-weighted bytes 与实测 memory，不只报 layer-count average bits。

“相同调用次数”与“相同 wall-clock budget”回答不同问题：前者检验损伤，后者检验省出的时间能否通过更多搜索提高任务表现。两种实验分开，不能让 quantized 版本暗中多做 planning 又称无损。
