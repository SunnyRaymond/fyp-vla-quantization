# 子任务状态（供 root）

**最新：FRT 原配方已完成 A/B 并判 mechanism_no_go；用户要求更新方向，新候选 [Paired-Rollout Recovery](<D:/Downloads/Final Year Project/ideaspark_run/closed-loop-quantization/revisions/paired-rollout-recovery/IDEA.zh.md>) 已形成方案，未运行实验。下文旧状态按历史记录保留。

## 当前状态（2026-09-12，修订后）

旧 STRC 已标记为 `rejected_attempt`，并复制到该子目录，历史文件保留但不再推进：signed accumulation 可能 telescope 到 terminal drift，原 phase-shuffle action injection 不能保持同一 policy，且 DINO-WM 没有 QuantWAMs denoising schedule。当前候选改为 **Frozen-Residual Transport PTQ**，详见 `IDEA.zh.md`、`phase2_generate/frozen_residual_transport_ptq.json`、`REVISION_RESPONSE.zh.md` 与 `AUDIT_INPUT.zh.md`。

机制：固定 FP32 与 baseline W4 `Q0`，在同一完整 history/action/RNG 下收集 `delta=F_Q0(X,a)-F_FP(X,a)`，`stop-gradient` 冻结，只写入 shift/concat 后的新 latent 槽；在下一步纯函数 map 上比较 `T_Q=F_Q(X+delta,a)-F_Q(X,a)` 与 `T_FP`，联合拟合同一 W4 配置的标准 AdaRound scalar scales/rounding（最终只保留 hard W4）。encoder/policy/readout 保持 FP32；部署不带 teacher、observer、额外 state 或 error-feedback。physical xy 只在 environment closed-loop endpoint 测量，不假设 latent→xy 线性 readout。

novelty 仅是 conditional：它检验真实 self-induced Q0 residual direction 是否比 local MSE、direct two-step、QDrop/input-noise、PD-Quant-style clean difference 和 random same-norm/GAD-like control 更能预测 held-out physical xy drift。没有 causal/Jacobian/strict-linearity、mixed precision 或 native deployment 声称。新增的 QDrop/AdaRound/PD-Quant/Sobolev/GAD primary-web 补检见 `phase0/DIRECT_PRIOR_SUPPLEMENT.zh.md`；RPIQ 与 Depth Replace Precision 的 residual/error-feedback 碰撞已记录。

预算已收窄为 ≤12 A100 GPU-hours、仅规划 1–4×A100-40GB conservative smoke；不保证模型一定放得下，A100-80GB 不是超时 fallback。当前无模型/实验/SSH/PBS 执行，必须先过完整 history + physical xy interface gate；random same-norm 等效或 fresh-Q0/fresh-Qtheta DEV 迁移失败即 no-go。

## 历史初稿记录（仅供追溯）

2026-09-12：Phase 0 已完成真实但降级的小 pool 检索（OpenAlex/Semantic Scholar；arXiv/OpenReview 因依赖缺失跳过；429/504 已记录），关键全文已取到 DA-PTQ、QuantWAMs、Feedback World Model、MARR、SQIL 等。Phase 1 输出已写入 `phase1/phase1_output.json`。

Tentative non-VQ candidate：**Temporal Residual Feedback Calibration (TRFC)**。对同一初始状态配对 FP/Q rollout，估计跨控制时刻的 residual-transfer kernel `K_r(ell)=E[e_{t+ell} r_t^T]`，以 persistent/canceling temporal mode 而非单点幅度、Jacobian motion score 或 one-step consequence 形成固定 W4/W8 map；可选另测量一个 calibration-derived delta-action error-feedback accumulator。Phase-shuffle、LocalMSE、DA-PTQ-style motion score、QuantWAMs-style reachable-state schedule、static on-policy recalibration 与通用 Feedback World Model 是必须对照。

关键边界：DA-PTQ 已覆盖 trajectory-level motion/Jacobian；QuantWAMs 已覆盖 reachable-state replay 与 schedule repair；Feedback World Model 已覆盖通用 transition-residual latent observer；`Calibrate Where You Deploy` 已报告 on-policy recalibration negative result。TRFC 只有在 temporal lag/sign/cancellation 的独立可复现预测、并在 phase shuffle 后消失时才可成立；不能把 generic observer 或 on-policy calibration 改名为新贡献。

已生成：`phase0/lit_table.md`、`phase0/RETRIEVAL_LIMITATIONS.zh.md`、`phase1/phase1_output.json`、`TENTATIVE_MECHANISM.zh.md`。当前未运行模型/实验/集群。

最终审查已完成：见 ../FINAL_IDEA_REVIEW_2026-09-12.zh.md，结论为 revise / pilot-ready conditional，非 novelty 认证。root 已补齐 fixed-index 优化预算或 action-slot/hard-W4 定义；以本目录 IDEA.zh.md 与最新 candidate JSON 为当前规格。未运行实验，原生 navigator 不标 DONE。
