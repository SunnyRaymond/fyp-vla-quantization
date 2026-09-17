# ResearchStudio-Idea / IdeaSpark 状态

日期：2026-09-12。用户允许省略弱相关 tagging、三语出卡和 PDF 排版；本目录保留实际运行记录与偏离原因。

## 已完成

- 新目录为 `idea/closed-loop-quantization`，验证结果归档在 `experiment/idea-validation/closed-loop-quantization`，没有改动旧 `world-model-quantization` 或共享文档。
- `phase0` 已用 `run.py next` 执行 bounded real connector pool；OpenAlex/Semantic Scholar 有结果，arXiv/OpenReview 因缺少依赖跳过，429/504 和原始 sentinel 均保留。没有重复整池检索。
- 已取 DA-PTQ、QuantWAMs、Feedback World Model、MARR、SQIL 等关键近邻全文；本轮只额外补了 AdaRound、QDrop、PD-Quant、Sobolev、GAD 的 primary web full-text/official repository 摘要和方法边界，见 `phase0/DIRECT_PRIOR_SUPPLEMENT.zh.md`。
- 初始 Phase 1/2/3 产物均保留；STRC 根据独立审查复制归档到 `rejected_attempt/`（含旧 `TENTATIVE_MECHANISM.zh.md` 与 `phase2_generate_output.json`），不删除历史文件。
- 新候选的精简 Phase 1、candidate、IDEA 卡、修订回应和 audit 状态已写入：`phase1_revised/phase1_revised_output.json`、`phase2_generate/frozen_residual_transport_ptq.json`、`IDEA.zh.md`、`REVISION_RESPONSE.zh.md`。
- fresh-context 审查输入已写入 `AUDIT_INPUT.zh.md`；要求逐项检查完整 history/shift/concat、标准 AdaRound hard W4、Q0/Qtheta fresh residual、random same-norm control 和 endpoint boundary。

## 当前候选和偏离

当前 non-VQ 候选是 **Frozen-Residual Transport PTQ**：冻结 baseline W4 `Q0` 在同 history/action 下产生的 `delta`，只在新 latent 槽执行真实 shift/concat，以 `F_Q(X+delta,a)-F_Q(X,a)` 的 finite-difference transport 对齐 joint W4 scalar scales/标准 AdaRound hard rounding。encoder、policy/readout 保持 FP32；physical xy 只在 environment endpoint 测量，不假设 latent→xy 线性 readout；部署不含 teacher/observer/extra state。

本轮没有继续运行旧 STRC 的 phase-shuffle、signed cumulative、mixed-map swap 或 QuantWAMs schedule；它们分别因 terminal telescoping/归因复杂、single-site 风险和 DINO 接口不匹配而停止。也没有执行模型、实验、SSH、PBS、重 I/O、安装或下载模型。

## 未完成与 gating

候选 novelty 仍为 conditional，不能写 canonical DONE。必须先过 interface gate（完整 `X`、真实 shift/concat、FP/Q0 replay、environment physical xy endpoint），再以单一 Wall environment 中 CAL/DEV/TEST=`6/6/12` episodes、3 fit seeds 做 ≤12 GPU-hour bounded pilot；只要求 episode/initial-state disjoint，不宣称 task-disjoint。A100-40GB 只是保守 smoke allowance，不保证模型一定放得下，不能用 A100-80GB 延长预算。必要对照为 local MSE、direct two-step、QDrop/input-noise、PD-Quant-style clean difference、random same-norm delta、fresh-Q0-DEV residual 和拟合后一次 fresh-Qtheta residual。若 random same-norm 等效或 fresh DEV 迁移失败，立即 no-go。

所有未来 heavy I/O、模型加载和计算仍须在真实 PBS allocation 中核验 `PBS_JOBID`、非-login hostname 和 GPU allocation；login node 只做轻量控制。本轮无集群执行、无凭证写入。

最终审查已完成：见 ../FINAL_IDEA_REVIEW_2026-09-12.zh.md，结论为 revise / pilot-ready conditional，非 novelty 认证。root 已补齐 fixed-index 优化预算或 action-slot/hard-W4 定义；以本目录 IDEA.zh.md 与最新 candidate JSON 为当前规格。未运行实验，原生 navigator 不标 DONE。
