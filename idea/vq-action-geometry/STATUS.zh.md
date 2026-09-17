# VQ candidate status（修订版，非 canonical DONE）

候选暂定为 **Trajectory-Relational Product VQ (TR-PVQ)**：固定 action probes 只作 input；对 FP32/Q model 的真实 paired rollout 得到 predicted outputs，直接调用 reproduction/dino-wm-wall/source/planning/objectives.py 的原始 objective_fn 作为 score anchor，并另以同坐标的 auxiliary representation 做 goal-anchored unnormalized off-diagonal Gram。首 pilot 固定 indices、只更新 codebook entries，沿用 VPTQ/AQLM packed layout；定义共同 actual-byte ceiling 并报告各 variant 的真实 bytes/slack，不做 rate allocator、dynamic schedule 或 custom accelerator。SCREEN anchor 使用已核对的 CEM5 配置（H=5、topk=30、num_samples=300、var_scale=1、opt_steps=5），不能把 source 默认 CEM10 当作本候选结果。

最小可证伪点是：在 held-out goals/action neighborhoods 上，固定 ceiling 的 L_abs+off-diagonal 必须超过 L_abs、local-MSE/VPTQ-like、ordinary RKD-style VQ（同 trajectory/goal anchor；含 QATMA/TPSD (CR-QAT label) pairwise-similarity 类强碰撞）及 same-target scalar/independent-codebook，并把关系收益传递到 planner first action 与 closed-loop return/success；若 off-diagonal 只重复实际 score/diagonal、stratified pair shuffle 等效、或 codebook coupling 不必要，则降级为领域适配/negative result，不声称新优化原理或现已 novel。

当前限制：phase0 为 bounded connector pool，arXiv/OpenReview 缺依赖、Semantic Scholar 429、2 个 host refs unresolved；bulk tagging、完整 fulltext sentinel、phase3 collision/canonical navigator 未完成。已写 REVISION_RESPONSE.zh.md、IDEA.zh.md 和待 fresh-context 审查的 audit input；本轮未运行模型、实验、SSH/PBS 或 native kernel。


最终审查已完成：见 ../FINAL_IDEA_REVIEW_2026-09-12.zh.md，结论为 revise / pilot-ready conditional，非 novelty 认证。root 已补齐 fixed-index 优化预算或 action-slot/hard-W4 定义；以本目录 IDEA.zh.md 与最新 candidate JSON 为当前规格。未运行实验，原生 navigator 不标 DONE。
