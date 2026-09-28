# E1 Frozen LeWM／PushT pilot 结果（2026-09-24）

## 状态与证据

这是两个 development-only reset seeds (`4101,4102`) 的探索性 pilot，不是 GO 实验。原始 `25536940.pbs101` 证实了 CEM callback、candidate 0／`prev_mean`、exact cost replay，但初选半径 `0.05/0.10/0.20` 没有任何非中心真实 proposal；且有限差分检查误把未显式裁剪的 solver 坐标判为不可检查。`25537017.pbs101` 修复了 solver 坐标检查，`25537027.pbs101` 统一 math SDPA 差分前向和 Jacobian 前向，追加 `1/2/4/8` 的 L2 半径。可引用的逐项原始值在 `artifacts/25537027.pbs101/e1_pilot_summary.json`。

## 主要观察

官方 CEM 每轮 300 个候选，优化张量 `[1,300,5,10]`，动作维度 50。其候选可以超出环境 `[-1,1]` action Box，runner 没找到对这些 proposal 生效的 solver clipping bound。扰动和距离均以实际 solver 坐标计。

| CEM round | 两个 context 的非中心 proposal 距离 p10 | 能覆盖大部分 proposal 的观察半径 | Taylor sample 终点相对误差 | top-30 overlap | Jacobian／完整 300-candidate cost 耗时 |
|---|---:|---:|---:|---:|---:|
| 1 | 6.07–6.22 | 8（269/299、276/299） | 0.58、0.64（各 16 个） | 0.10、0.10 | 29.2×、6.9× |
| 15 | 3.05–4.07 | 4–8 | 半径 4 为 0.31、0.40（各 16 个） | 0.77、0.67 | 6.0×、6.4× |
| 30 | 1.22–1.38 | 2（286/299、289/299） | **不作确认性解释** | 0.90、0.70 | 6.3×、6.1× |

上表相对误差是选中真实提案的 `||F(v)-[F(v̄)+J(v-v̄)]|| / ||F(v)-F(v̄)||` 平均值；每个有效半径最多 16 个，属于球内按距离排序的抽样，不是壳层误差。Jacobian 计时包含完整 autoregressive rollout 的 forward-mode 构造，完整 cost 计时来自官方 batched `get_cost`。首个 context 的 round 1 有明显 cold-start 开销，因此速度判断主要看其余行。局部计时仍不等于完整 solve 计时。

与 `torch.func.jacfwd` 相同的 math SDPA 前向下，中心差分相对 JVP 的四方向检查：round 1 全部 <0.1%；round 15 全部 <3.5%；round 30 的最差方向分别为 5.2%、6.2%，未满足预设的 5% 门槛。round 30 的差分在步长 0.01 时更不稳定，提示 fp32 消减可能参与，但目前不能据此宣布 Jacobian 通过。

## 阶段判断

**直接每轮以完整动作 Jacobian 替换 300-candidate full rollout：成本 NO-GO（在这个 pinned 实现与 A100 pilot 上）。** 即使忽略候选评分后的其他 solver 成本，稳态 Jacobian 本身约为一次完整候选评分的六倍；early round 的 elite 重合率又只有 10%。现有证据不支持继续把该直接 surrogate 接入 official adaptive CEM。训练是否能改善局部几何是独立的 predictor-level 问题，交由 E2 matched pilot 检验；E2 的结果不能自动推翻这个 Jacobian 成本判断。

本 pilot 不提供 closed-loop 成功率、整体 planner speedup 或跨 context 统计精度。late-round Taylor 指标因有限差分 gate 未通过而保持未确认。没有根据两个开发 seeds 宣称普遍不可行。
