# Antithetic Rounding Pairs：独立机制审查

审查对象：`IDEA_AND_PROTOCOL.zh.md`（按新版 source indices `96–101`）。本审查不改 protocol/script，不运行训练或完整验证；scalar arithmetic 仅手算，未执行 stdlib trace。

## Verdict

**revise before execution**。机制问题足够狭窄，尚无 `novelty_no_go`：Ex Uno Pluria 的 LPE-BSR 已覆盖单 checkpoint、training-free、Bernoulli stochastic-rounded low-precision ensemble，但其成员分布是逐参数独立 product、最终组合的是 predictions；没有 antithetic pair、world-model rollout 或 action shortlist。QuantWM 已在 DINO-WM 的 Wall/PushT candidate rollout 中报告低 bit 误差累积、planning mismatch 和 FP reference 边界；本候选可定位为“固定两次 W4 forward 的 rounding coupling 是否改善一次候选选择”，不能声称新统计原理、通用 ensemble 或 planner 改进。

## 机制核对

令 `v=w/s`、`p=v-floor(v)`、`u~Uniform[0,1)`，第一成员为 `floor(v)+1[u<p]`，第二成员使用 `1-u`。对 `0≤p<1`，误差 covariance 为 `max(0,2p-1)-p² = -min(p²,(1-p)²)`；公式正确。`p=0` 行固定不变，clip 只应是边界保护。这个 scalar 结论不推出 nonlinear rollout 或 ranking 优势，protocol 已正确把后者作为 falsification target。

## 执行前阻断与最小修复

1. “独立 negative control 的 `v_random`”不是可验收定义。必须改成同一 `v`、同一 scale/grid、同一第一成员，第二成员另取独立 `u_ind`；antithetic 才取 `1-u_0`。否则 marginal 不匹配，coupling 对照失效。
2. 明确每 candidate 的选择分数：`bar{s}_c=(s_c^1+s_c^2)/2`，在同一固定 300-candidate pool 上用 `bar{s}` 选 top30，固定 tie-break；再用同一 pool 的 FP score 计算 regret。称为 **one-shot initial-pool shortlist regret**，不要称 CEM elite 或 actual action。
3. `elite mean action MSE` 必须成为 go gate：定义 `A_m=mean(action[Top30(bar{s}_m)])`，`M_m=||A_m-A_FP||²/d`；要求 aggregate `M_anti≤M_ind+τ` 且 `M_anti≤M_RTN+τ`，否则 `mechanism_no_go`，其中 `τ` 预先冻结。现有 primary 只能作为 shortlist proxy。
4. 写清成本矩阵：antithetic/independent 各 2 个 W4 forward，RTN 1 个，FP32 reference 另计；只宣称配对比较，不宣称 latency/memory/compression gain。FP32 必须是同 checkpoint、同 targets/pools 的 full-precision fidelity reference，不是 environment oracle。
5. 预先定义 `R` 的方向、5% 分母与 episode/seed aggregation（3 seeds 是重复测量，不是 18 个独立 episode）；保留 `96–101` fingerprint/registry exclusion gate。

修复上述定义后，工程 gate 全通过且 primary、action gate 同时通过，才可记 `preliminary_go`；任一失败均停止，不追加 ensemble/CEM/闭环 suite。

## 原始来源

- [Ex Uno Pluria, arXiv:2411.14860](https://arxiv.org/html/2411.14860v1)：§3–4 Eq.2/Eq.5 的 stochastic rounding 与独立低精度 ensemble，Eq.3–4 的 prediction averaging。
- [QuantWM, arXiv:2602.02110](https://arxiv.org/html/2602.02110v1)：DINO-WM Wall/PushT candidate rollout、planning horizon、FP32/RTN 对照及低 bit rollout 失配。
