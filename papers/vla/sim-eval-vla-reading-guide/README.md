# VLA Simulation Evaluation Reading Guide

Updated: 2026-08-25

## 结论先行

你提到的 Chen Tianxing 新工作，最可能是 **RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies**。Chen Tianxing 是第一作者；同一条技术路线还包括他共同第一作者的 **RoboTwin** 与 **RoboTwin 2.0**。

`sim eval network for VLA` 不是一个统一术语。当前相关工作最好分成三类：

1. **physics-based benchmark / environment**：RoboDojo、RoboTwin 2.0、LIBERO-PRO、LIBERO-Plus、LIBERO-X、LIBERO-Para、SIMPLER、VLABench、RoboCasa365；
2. **evaluation harness**：vla-eval；
3. **learned world-model evaluator**：WorldEval、WorldGym、dWorldEval。

没有一个系统已经证明自己在所有意义上“全面优于 LIBERO”。更准确的说法是：它们分别修补 LIBERO 的不同 **evaluation validity gap**。

## 最值得先读的结论

| Work | 类型 | 作者对 LIBERO 缺口的主要判断 | 它声称补上的能力 | 证据边界 |
|---|---|---|---|---|
| [RoboDojo](papers/01-robodojo/README.md) | sim-and-real benchmark | 现有 benchmark 的 capability coverage、bimanual evaluation、sim-to-real protocol 与 anti-gaming 机制不足 | 42 simulation tasks + 18 real-world tasks、5 capability dimensions、hidden verification layouts、统一远程 real evaluation | 作者表中 LIBERO 有 130 tasks，RoboDojo 只有 60 total tasks；优势是覆盖与 protocol，不是 raw task count |
| [RoboTwin 2.0](papers/02-robotwin-2/README.md) | bimanual benchmark + data generator | clean simulation 容易对 appearance 与 layout 过拟合，且缺少 scalable synthetic data | 50 dual-arm tasks、5 embodiments、structured domain randomization、100k+ trajectories | 没有同一训练 protocol 下直接击败 LIBERO；与 LIBERO 的比较属于方向性推断 |
| [LIBERO-PRO](papers/03-libero-pro/README.md) | LIBERO OOD extension | train 与 evaluation task 近乎相同会抬高 memorization-based score | object、position、instruction、task、environment 变化；作者报告某些 generalized settings 从 >90% 跌到 0% | 仍继承 LIBERO simulator、assets 与 tabletop domain |
| [LIBERO-Plus](papers/04-libero-plus/README.md) | LIBERO robustness extension | standard success rate 隐藏 camera、initial state、lighting 等脆弱性 | 10,030 task instances、7 perturbation factors、5 difficulty levels、>56k scenarios | CVPR 2026 final；仍是 LIBERO-derived benchmark |
| [LIBERO-X](papers/05-libero-x/README.md) | training + evaluation benchmark | 只改变很小的 initial state、单 scene 少 task、homogeneous demonstrations 不足以测 compositional OOD | 600 tasks、100 scenes、2,520 demonstrations、5 cumulative difficulty levels | 重新训练数据与 time limit 不同，不能把分数直接与 standard LIBERO 对比 |
| [LIBERO-Para](papers/06-libero-para/README.md) | language robustness diagnostic | identical train/eval instructions 无法测 paraphrase robustness | 43 linguistic variation types、PRIDE metric、trajectory divergence diagnosis | 只覆盖 LIBERO-Goal 的 language shift |
| [SIMPLER](papers/07-simplerenv/README.md) | real-to-sim evaluator | pure simulated success 未证明能代表 real-world ranking | 用 paired sim/real trials 验证 relative ranking，报告高 Pearson correlation 与低 MMRV | 仅少量 embodiments/tasks；作者明确说不是 real evaluation 的替代品 |
| [VLABench](papers/08-vlabench/README.md) | semantically rich VLA benchmark | LIBERO 对 world knowledge、implicit language、logic、long horizon 与 object diversity 覆盖不足 | 100 task categories、2,000+ objects、strong randomization、cross-embodiment | 不是同-policy、同-training 的 LIBERO head-to-head |
| [RoboCasa365](papers/09-robocasa365/README.md) | large-scale household benchmark | tabletop benchmark 对 room-scale mobile manipulation 与 task/data scaling 覆盖不足 | 365 kitchen tasks、2,500 scenes、500k+ demos、约 2,200 hours data | kitchen-only；它更像 training ecosystem，不是专门的 robustness diagnostic |
| [vla-eval](papers/10-vla-eval/README.md) | evaluation harness | 不同 benchmark 的 dependencies 与 undocumented protocol 会制造巨大复现误差 | versioned isolation、canonical protocols、14 benchmark integrations、最高 47x speedup | 不增加任务难度，也不消除 LIBERO saturation |
| [WorldEval](papers/11-worldeval/README.md) | learned world-model evaluator | 手工搭建 simulator 难以随 embodiment/task 扩展 | Policy2Vec action conditioning；作者报告平均 Pearson r=0.942、MMRV=0.044 | 与 SIMPLER 不能直接比较，只复用了 real-to-sim 技巧；有 hallucination 与 action-fidelity 风险 |
| [WorldGym](papers/12-worldgym/README.md) | learned world-model environment | handcrafted simulator 难以快速构造 OOD image/language setting | single initial frame + action-conditioned rollouts + VLM reward；per-task Pearson r=0.78 | 对 Bridge evaluation 的三个 policies/17 tasks 验证；部分 interaction 不真实 |
| [dWorldEval](papers/13-dworldeval/README.md) | discrete diffusion world evaluator | video-pretrained prior 会覆盖 action signal 并 hallucinate success | unified action/vision/language tokens、sparse keyframe memory、progress token；LIBERO multi-view r=0.910 | 2026 v1 preprint；关键 metric 与 progress labels 由作者体系定义，尚待独立复现 |

## “相比 LIBERO 好在哪”应该怎样解读

### 1. 直接修补 LIBERO 的 train-test leakage / robustness gap

优先看 **LIBERO-PRO、LIBERO-Plus、LIBERO-X、LIBERO-Para**。它们的优势是可以最大程度复用 LIBERO tooling 与 model checkpoints，并通过 controlled distribution shifts 揭露 memorization、camera sensitivity、scene variation 或 language paraphrase failure。缺点是仍未跳出 LIBERO 的 MuJoCo/robosuite physics、assets 和 tabletop task family。

### 2. 扩展 task、embodiment 与 capability coverage

优先看 **RoboDojo、RoboTwin 2.0、VLABench、RoboCasa365**。它们分别强调 bimanual/sim-and-real protocol、domain randomization、world knowledge/long horizon、room-scale household scaling。这里“更好”通常不是 standard LIBERO score 更高，而是测到了 LIBERO 没有充分覆盖的能力。

### 3. 验证 simulation score 是否能预测 real-world performance

优先看 **SIMPLER**。它把目标从“在 simulator 中完成任务”改为“simulator 能否保持 real-world policy ranking”。这是比单纯 success rate 更强的外部效度问题，但目前任务与 embodiment 范围仍小。

### 4. 用 learned world model 替代手工 simulator

优先看 **WorldEval → WorldGym → dWorldEval**。它们主张更容易跨 task/scene/embodiment 扩展，并可在部署前安全筛查 policy；但 world model 自身会产生 action ignoring、hallucinated success、object deformation 与 reward-model bias，因此现在更适合作为 physics/real evaluation 的补充证据，而不是单独裁决模型好坏。

## 推荐阅读路线

### 2 小时：抓住最关键争论

1. [LIBERO baseline note](papers/00-libero/README.md) — 15 min
2. [RoboDojo](papers/01-robodojo/README.md) — 35 min
3. [LIBERO-PRO](papers/03-libero-pro/README.md) — 20 min
4. [LIBERO-Plus](papers/04-libero-plus/README.md) — 20 min
5. [SIMPLER](papers/07-simplerenv/README.md) — 20 min
6. [dWorldEval](papers/13-dworldeval/README.md) — 10 min，先看 claim 与 caveat

### 1 天：形成可用于 FYP 的 evaluation taxonomy

按以下顺序读 PDF 的 Abstract、Introduction、Benchmark/Protocol、Main Results、Limitations：

1. RoboDojo；
2. RoboTwin 2.0；
3. LIBERO-PRO + LIBERO-Plus + LIBERO-X；
4. SIMPLER + VLABench；
5. vla-eval；
6. WorldEval + WorldGym + dWorldEval。

### 1 周：准备实验设计

- 保留 **standard LIBERO**，用于与已有 VLA literature 对齐；
- 加一个 **LIBERO-derived OOD suite**：优先 LIBERO-Plus；若重点是 memorization claim，选 LIBERO-PRO；若需要训练集与 progressive evaluation 同时扩展，选 LIBERO-X；
- 加一个 **orthogonal capability benchmark**：single-arm semantic/long-horizon 选 VLABench，bimanual 选 RoboDojo 或 RoboTwin 2.0；
- 若 claim 涉及 deployment，加入 SIMPLER 或少量 real-world paired evaluation；
- 若研究 World Action Model，加入 dWorldEval，但必须同时报告 physics-sim 或 real execution ground truth；
- 能用时通过 vla-eval 固化 environment version、preprocessing、action mode、seed、episode count 与 termination rule。

## Chen Tianxing 工作线

- [RoboDojo](papers/01-robodojo/README.md)：最新、最直接对应你的描述，Chen Tianxing 第一作者。
- [RoboTwin 2.0](papers/02-robotwin-2/README.md)：更强调 scalable data generation、domain randomization 与 multi-embodiment bimanual benchmark。
- [RoboTwin](papers/02a-robotwin/README.md)：前身，提出 Generative Digital Twins 与 dual-arm sim/real benchmark。

## Library integrity

- 15/15 PDFs 通过 `%PDF-` signature、`pypdf` reopen、page-count 与 encryption check；
- 15/15 PDFs 完成 first-page visual inspection；
- 所有文件均未加密；
- SHA-256、版本与页数见 [PDF_INVENTORY.md](PDF_INVENTORY.md)。

## 术语与证据标记

- **Authors claim**：论文或项目明确提出的结论；
- **Direct comparison**：同一论文中有明确 protocol/table 的比较；
- **Synthesis**：基于任务设计与证据边界的综合判断，不等同于作者完成了 head-to-head experiment。
