# Frozen-Residual Transport PTQ：fresh-context audit input

请把 `phase2_generate/frozen_residual_transport_ptq.json` 和 `IDEA.zh.md` 作为当前候选；同目录旧 STRC 只作 `rejected_attempt` 历史，不得合并名字或机制。审查目标是判断“conditional candidate / no-go”，不是把它写成 canonical DONE。

## 必查接口

1. `X` 是否真的是 DINO-WM Wall 的完整 history state，`F(X,a)` 是否可作为纯函数重放；`shift/concat` 是否只把 `delta` 写到预测新 latent 槽，旧 history 不变。
2. 是否可以在相同 history/action/RNG 下得到 FP32 与 `Q0` 的 `delta=F_Q0(X,a)-F_FP(X,a)`，并在拟合前 `stop-gradient` 冻结；DEV 是否能在拟合后各生成一次 fresh `Q0` residual 与 fresh `Qtheta` residual。
3. loss 是否严格是 latent finite difference `F_Q(X+delta,a)-F_Q(X,a)` 与 FP 对照的 transport contrast；不得引入 terminal cumulative、cross-lag kernel、causal/Jacobian/小噪声线性假设。physical `xy` 只能是独立 environment closed-loop endpoint。
4. 量化器是否是标准 AdaRound hard materialization：`q_int=clip(floor(w/s)+h(alpha))`、`h(alpha)` 最终冻结为 `{0,1}`、部署丢弃 `alpha`；是否保持同一 W4 logical map，未变成 mixed precision 或 per-weight offset 存储。

## 必查对照与停止

- 同 calibration/evaluation budget：RTN Q0、local MSE/AdaRound clean、PD-Quant-style clean difference、direct two-step、QDrop/input-noise、random same-norm `delta`（GAD/Sobolev-like），及 FRT。
- 若 random same-norm 与 Q0 residual direction 的 transport/physical endpoint 等效，或 FRT 只改善 CAL 不改善 fresh DEV，或 joint fitting 没超过 local controls，则 no-go。
- 检查 DA-PTQ、QuantWAMs、Feedback World Model、RPIQ、When Can Depth Replace Precision? 的机制边界；不接受“trajectory-aware”“closed-loop residual”这类宽泛 novelty。
- 检查 ≤12 A100 GPU-hours、仅 1–4×A100-40GB pilot 约束；A100-80GB 不得成为超时 fallback。没有模型/实验/PBS 结果可以被写成 evidence。

## 证据边界

Phase 0 是 bounded/degraded connector pool；补充的 AdaRound/QDrop/PD-Quant/Sobolev/GAD 取自 primary web full-text/official repository，见 `phase0/DIRECT_PRIOR_SUPPLEMENT.zh.md`。没有 exact FRT collision 不等于不存在。请将审查结论和任何 no-go 理由写入 fresh 独立文件，避免改写本文件和历史产物。
