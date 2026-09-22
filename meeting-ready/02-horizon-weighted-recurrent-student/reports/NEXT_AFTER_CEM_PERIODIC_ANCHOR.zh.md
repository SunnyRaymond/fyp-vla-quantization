# CEM periodic anchor 之后的下一步

## 结论先行

推荐 **E：停止 DINO special-purpose line，把结论收束回 LeWM + PushT**。

这不是因为 periodic anchor 没有相对收益，而是因为它没有解决 planner-facing 的绝对 fidelity 问题：

- `student_only` 到 `periodic_teacher_anchor_P5` 的 iteration-30 first-action RMS median 为 `0.7317 -> 0.6430`，trajectory AUC median 为 `0.6153 -> 0.5493`；两项 relative gate 均通过，且 treatment `4/6` cases 不 worse。
- 但冻结的 absolute gates 仍失败：iteration-30 RMS 要求 `<=0.15`，实际 `0.6430`；coordinate absolute maximum 要求 `<=0.25`，实际 `1.9762`。
- round-30 teacher top-30 overlap median 只有 `1/30`。`shuffled_anchor_P5` 没有同时复现 RMS 与 AUC 的下降，因此只支持“anchor 的 cost-action correspondence 有作用”，不支持已经得到可部署 replacement。
- 这是 fixed-observation mechanism diagnosis；按 protocol，Stage 2 closed-loop 不运行。

因此，下一步不应再用另一个 DINO CEM 变体消耗 GPU。若必须给 A 一个标签：现有 case JSON **支持做字段级离线检查，但不支持可靠的 deployable proxy-validity 结论**，所以 A 的 proxy gate 为 **NO-GO，且不解锁 GPU experiment**。

## 为什么 A 不能被可靠解锁

六个 `case_*.json` 确实包含了 A 所需的原始量：

- `student_only.rounds[*].pre_update.mu/post_update.mu`：可计算 full-trajectory 与 first-action update magnitude；
- `pre_update.sigma/post_update.sigma`：可计算 signed contraction、expansion 和 absolute change；
- `selected_indices` / `student_top30_indices`：可计算 selected-index churn；
- checkpoints `1,5,10,15,20,25,30` 上的 `first_action_rms`、`teacher_cost_regret`、`teacher_top30_overlap`：可作为 oracle outcomes。

但这些材料不能把 A 变成可靠的 adaptive fallback：

1. 独立 unit 只有 6 个，而且是预先挑出的 teacher-success/student-failure cases；没有 healthy/low-risk cases 来估计 false-positive trigger rate。
2. oracle outcomes 只有 7 个 checkpoints；中间 23 个 rounds 只有 student trajectory，没有 teacher label。因此可以测 association，不能离线重演任意 trigger round 的 teacher counterfactual。
3. 不能把 `6 x 7 = 42` 个观测当作 42 个 replicate。按 case 作为 replicate 的处理，candidate/round 都是 nested measurements。
4. 轻量的只读关系检查没有给出稳定的单调 proxy：`mu_step` 和 `first_action_step` 随 CEM 收缩而变小，但在 case 内与 `teacher_cost_regret` 的方向与 time-matched cross-case 关系相冲突；`sigma` change 方向不稳定；selected churn 大多挤在约 `0.889--0.983`，动态范围很小。故不能从这 6 cases 冻结阈值而不追着样本过拟合。

这足以判定 **A = proxy-validity 不成立/不可可靠判定，不能进入 GPU**，而不是把一个相关性弱、且缺少 healthy negatives 的指标包装成 deployment trigger。

## 候选比较

| 候选 | 当前证据 | 决策 |
|---|---|---|
| A adaptive teacher fallback | 字段可提取，但只有 failure-selected cases、稀疏 oracle checkpoints，且 proxy 关系不稳定；无法冻结可信 trigger | **NO-GO** |
| B trust-region / damped student CEM updates | 可能减缓 feedback amplification，但不能修复 iteration-1 的 elite mismatch；需要新的 GPU mechanism run，当前没有独立支持 | 暂不做 |
| C P=2/P=3 scheduled anchors | 只是 P=5 同 recipe 的 cost boundary；P=5 已在绝对 gates 上大幅失败，频率 sweep 不是新机制 | 不做 |
| D proposal-state / unrolled trajectory training | 需要重新训练，成本最高；已有 CEM-DAgger 与 score-distill 结果不足以证明应继续训练扩展 | 不做 |
| E 收束回 LeWM + PushT | 不新增假设、不追加 DINO 成本，符合项目主线和现有证据边界 | **推荐** |

## 最小输入、arms、cases/seeds 与预算

E 是停止决策，不是新的 GPU experiment，因此没有新的 arms、cases 或 seeds：

- 输入：保留现有 `cem_periodic_anchor_summary.json`、6 个 `case_*.json`、freeze/protocol 和三份相关 CEM reports；不读取 model、dataset 或 runtime archive。
- 新作业：`0`；新 teacher calls：`0`；GPU budget：`0`；closed-loop budget：`0`。
- 已有 P5 与 shuffled negative control 作为 closure evidence，不重跑 P=5、不补跑 P=2/P=3。

## Primary endpoints / gates / negative control

停止决策沿用已冻结的 P5 gates，不放宽任何阈值：

- iteration-30 first-action RMS median `<=0.15`；
- iteration-30 coordinate absolute maximum `<=0.25`；
- 相对 student-only 的 RMS、trajectory AUC、iteration-30 regret gates；
- shuffled negative control 不得同时复现两个 relative gains。

P5 已经是“relative gains 通过、absolute fidelity 失败”；因此不存在一个尚待补跑的 endpoint 可以把本结论改成 GO。现有 shuffled anchor negative control 未复现两项 relative gain，也不抵消 absolute gate failure。

## Stop rule 与后续边界

现在停止 DINO CEM periodic-anchor 线：不运行 closed-loop，不扩大 cases，不调 threshold，不做 P=2/P=3，不转入 B 或 D。只有在用户明确提出新的独立 evidence、不同 research question，或重新冻结一个有充分外部/held-out proxy 支持的 recipe 时，才重新开启 DINO experiment design。

本 memo 支持的结论仍限于：六个预选 PushT cases 上的 fixed-observation CEM mechanism diagnosis。它不支持 population-level success、closed-loop improvement、native latency/memory 或 deployable adaptive fallback claim。

项目后续应回到 **LeWM + PushT**；这表示研究线的收束方向，不是对尚未运行的新 LeWM experiment 预先宣称结果。

## 本地依据

- [periodic-anchor summary](../artifacts/25158912.pbs101/cem_periodic_anchor_summary.json)
- [periodic-anchor freeze](../config/CEM_PERIODIC_ANCHOR_FREEZE.json)
- [periodic-anchor protocol](../config/PROTOCOL_CEM_PERIODIC_ANCHOR.zh.md)
- [CEM trace diagnosis](RESULT_CEM_TRACE_DIAGNOSIS.zh.md)
- [CEM multi-fidelity](RESULT_CEM_MULTIFIDELITY.zh.md)
- [CEM score distill](RESULT_CEM_SCORE_DISTILL.zh.md)
- [CEM-DAgger](RESULT_CEM_DAGGER.zh.md)
