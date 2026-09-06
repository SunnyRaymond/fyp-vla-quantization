# 方向二：让 VLA 先试探，再动手

原文：[VLA_idea.pdf p.2](../../VLA_idea.pdf) · [三个方向总览](../README.md)

**8 个编号条目，8 份 PDF。** 建议把四个部件分开阅读：unknown variable / belief、measurement model、probe selection、执行任务的 policy。Failure detector 只提供触发信号，不能代替其余部件。

| # | Paper / notes + PDF | 贡献或必要对照 | 阅读层级 |
|---:|---|---|---|
| 1 | [Shielding-Aware Dual Control](01-shielding-aware-dual-control/README.md) | uncertainty reduction 与 task control 的联合 formulation | 理论基础；本次补充 |
| 2 | [Push to know!](02-push-to-know/README.md) | 用 active push 辨识物理参数 | **先精读**；IROS 2023 |
| 3 | [Predictive Visuo-Tactile Interactive Perception](03-predictive-visuo-tactile/README.md) | 多探索动作、GNN filtering、N-step information gain | **先精读**；原总结列出 |
| 4 | [ActiveVLA](04-activevla/README.md) | 3D active view selection / zoom-in | 观测选择对照；2026 preprint |
| 5 | [CoMe-VLA / Act, Sense, Act](05-come-vla/README.md) | human-data-driven active perception 与 memory | **必要 VLA prior art**；本次补充 |
| 6 | [PI-VLA](06-pi-vla/README.md) | uncertainty-triggered horizon adaptation / replanning | 被动响应对照；Symmetry 2026 |
| 7 | [Perturbation-Based Failure Detection](07-perturbation-failure-detection/README.md) | low-rank perturbation 的 epistemic score | 触发器对照；2026 preprint |
| 8 | [PhysReflect-VLA](08-physreflect-vla/README.md) | feasibility checking 与在线 reflection | 可靠性对照；2026 preprint |

## Primary route（约 3–4 hours）

Push to know! 60 min → Predictive Visuo-Tactile 45 min → CoMe-VLA 45 min → Dual Control 45 min。随后对照 ActiveVLA、PI-VLA、PFD，逐个判断它是在“改变 observation”“估计参数”“检测不确定”还是“选择 probing action”。

## 不宜直接照搬的判断

- 不能笼统说“VLA 不会主动获取信息”：CoMe-VLA 已覆盖主动感知行为。
- physical parameter inference 已经有 learned differentiable models，不能把旧工作都归为纯解析模型。
- PI-VLA 的 AURD 名称有 “Active”，实际应按 Algorithm 1 判断；horizon adjustment 不自动等于 information-gain probing。
- 只预测未来的 model 无法凭空创造观测中不存在的信息；但 historical interaction / prior 可能提供额外信息，应在 experiment 中明确可见变量与可用历史。

## 本方向对照卡（留空）

| Paper | Hidden variable | 可用 sensors | Probe 是否由任务决定 | Belief update | Information gain 的收益 | 探测时间 / 风险 |
|---|---|---|---|---|---|---|
| Push to know! | | | | | | |
| CoMe-VLA | | | | | | |
| PI-VLA | | | | | | |
| 我的构想 | | | | | | |

## Cross-links

原总结列在本方向的 [LaWAM](../03-control-distillation-and-world-models/08-lawam/README.md) 放在方向三以保持完整的 latent world-model 主线；此处仍可直接打开。可比较 `no probe / random probe / informative probe / oracle parameter`，但问题与方案留待你阅读后确定。
