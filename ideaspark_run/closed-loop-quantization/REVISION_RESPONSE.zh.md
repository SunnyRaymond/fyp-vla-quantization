# 对独立 review 的逐项回应（2026-09-12）

本文件对应 `INDEPENDENT_IDEA_REVIEW_2026-09-12.zh.md`。原 STRC 不再推进，历史文件保留；修订候选为 `Frozen-Residual Transport PTQ`。

| review 要求 | 落实 |
|---|---|
| 不把 `E[e_{t+ell}r_t^T]` 当因果 kernel | 删除该对象及 causal wording。新候选只用同 history/action 下的 `delta=F_Q0(X,a)-F_FP(X,a)`，并在 `F(X+delta,a)-F(X,a)` 上作 finite-difference comparison；不声称 causal/Jacobian。 |
| 避免 `sum ||K_l||^2` 丢符号或 telescoping | 删除 signed cumulative、lag shuffle、cancellation 目标。新目标只比较一步 transport difference，另以 short closed-loop physical xy drift 做 held-out endpoint。 |
| 不做泛用 observer/accumulator | 删除 delta-action accumulator、online observer、error-feedback deployment；部署不带 FP teacher 或额外 state。 |
| 不退回 single-site additive score | 固定同一 W4 map，不做 mixed allocation；对所有 target blocks 联合拟合标准 AdaRound `theta={(s_g,alpha_g)}`，不加 site scores，也不把 two-block interaction 设为必要 gate。 |
| 使用实际可执行的 history/state 接口 | 明确 `X` 是完整 history，`shift/concat` 只把 `delta` 写入新 latent 槽，旧 slots 保持；接口 gate 要求可恢复 history/action/RNG，physical xy 只在独立 environment endpoint 读取，不假设 latent→xy 线性 readout。缺失则停止。 |
| 冻结 residual 防止自证循环 | `delta` 只从先收集的 frozen Q0 paired rollout 产生，`stop-gradient`；DEV 另收 fresh Q0 residual，并在拟合后让 Qtheta 生成一次 fresh residual，二者只作迁移检查。 |
| 与 Sobolev/GAD/PD-Quant/QDrop 边界清楚 | 新增 `phase0/DIRECT_PRIOR_SUPPLEMENT.zh.md`。承认 derivative/JVP 和 prediction-difference 总原则已存在；random same-norm delta 是 GAD/Sobolev-like control，QDrop/input-noise、PD-Quant clean difference、AdaRound hard rounding 均为 matched controls。 |
| 修正 DINO/QuantWAMs 不匹配 | 删除 QuantWAMs denoising schedule 对照。QuantWAMs 只作 reachable-state/Fisher prior boundary；DINO 首 pilot 仅做 predictor one-step PTQ。 |
| 缩减资源 | 规划改为 ≤12 A100 GPU-hours，1–4×A100-40GB；A100-80GB 不作为超时 fallback。包含 controls、启动和失败成本。 |

## 当前判定

这仍是 conditional candidate，不是 canonical DONE，也不是独立审查通过。最小可证伪点只有一个：在相同 W4 bytes 和评估预算下，真实 Q0 residual direction 的 transport 对齐能否比 random same-norm、local MSE、QDrop/input-noise、PD-Quant-style clean difference 和 direct two-step matching 更好地预测 held-out physical xy drift。若 Q0 direction 与 random same-norm 等效，或 fresh DEV 检查失败，立即 no-go。
