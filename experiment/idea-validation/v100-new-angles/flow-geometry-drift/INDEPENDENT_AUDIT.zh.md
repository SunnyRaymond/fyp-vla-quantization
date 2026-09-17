# Low-bit flow geometry 能否提示 PTQ action drift：独立审查

**结论：`revise` 后可推进；`accel` 本身 `novelty_no_go`，窄诊断问题未被完全覆盖。** 未选 episode、未读数据。

## 先例边界

[原论文](https://arxiv.org/html/2607.27933v3) 已定义 Eq.11 的 `accel_p`，用 prefix/Spearman 评估 flow uncertainty，也承认 geometry 对“自信但错误”会盲。因此这些设置都不是创新；该文未测 PTQ 或同 noise 的 FP/Q drift。

[QuantVLA](https://arxiv.org/abs/2602.20309) 已分析 VLA 的 quantization-induced drift，并比较 backbone、action head 与联合量化。“模块选择影响 drift”是强近邻。可保留的问题是：固定 noise 时，Q `accel_8` 能否排序相对 FP 的 drift，以及两个 isolated locus 是否有不同盲区。不得声称新 proxy、完整 detector 或 joint-PTQ 结论。

## 可识别性与阻断

同一 observation、solver、scaling、初始 noise 跨三 arm 配对，能识别关联诊断，但不能推断联合量化或 closed-loop success。冻结 `D` 为前 8 chunk×7 physical dims 的 normalized-space Q/FP action MSE；`accel_8` 按 Eq.11 用 raw `v_0...v_7`（分子 `8×t=1..7`，分母 `t=0..7`）。这是 physical-slice adaptation，须确认 32-d padding 索引与 gripper 规则；不能用 Euler increment、末两步或 Q action norm 代替 Eq.11。

两 noise 先按 episode 等权平均，最终只以 12 episodes 做 Spearman，不能把 24 runs 当独立样本。冻结 `rho(Q_accel,D)>=.5`，且分别比两 baseline 高 `.1`；写死 ties、常数向量/undefined rho、边界及 Q action norm 的向量范围。

标题中的“预警”没有 threshold、提前量或 online FP-free test；本 screen 只能叫 offline drift-ranking diagnostic。两 locus 不说明 joint PTQ。

## 最小执行修订

1. 冻结两个 locus 的 module allowlist、W4 RTN scale、FP32 范围及 source/checkpoint hashes，禁止隐藏 mixed precision。
2. 同 seed 跨三 arm，保存前 8 steps raw velocities、denoising vectors、7-d action slice、每臂 `accel_8`/drift；零分母、非有限或不完整样本记 `inconclusive`。
3. 报告 12-episode paired table、noise 离散度及 rho/delta 原值；不称 detector。
