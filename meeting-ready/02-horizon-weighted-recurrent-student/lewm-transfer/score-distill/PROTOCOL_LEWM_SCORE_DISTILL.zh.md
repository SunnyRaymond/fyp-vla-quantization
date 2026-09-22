# LeWM PushT：Bounded Score Distillation vs Pairwise Rank 冻结协议

状态：`frozen / predictor-level only`

## 1. 目的与不变量

本实验只替换 training objective，不改变 student inference architecture。三臂均为
原始 `LeWMCompactRecurrentTransitionStudent`：`192-D` latent、`10-D` packed action、
h256 shared recurrent、predicted-latent feedback、无 attention、无 conditioner、无
hidden-size expansion、无 goal input。

复用正式 dense-rank job `24908446.pbs101` 已生成的 frozen `prepared_rows.pt`，其中
包含同一批 256 个 episode-disjoint train contexts、每个 context 的 64-candidate
teacher bank，以及同一 16-block held-out bank。不得重新生成或下载该大文件。

三臂从同一 initial state、同一 context schedule、同一 1500-step budget 开始：

| arm | latent loss | external objective loss |
|---|---|---|
| `pairwise_rank` | horizon-weighted free-running latent MSE | 已验证的 teacher top-6 vs ranks 7–12 pairwise logistic，temperature 0.5，weight 0.1 |
| `score_distill` | same | 全 64 candidates 的 context-normalized teacher-score SmoothL1 |
| `shuffled_score` | same | 相同 score loss，但在 context 内用 frozen seed 置乱 teacher scores |

Score distillation 对每个 context 计算：

```text
t_norm = (teacher_cost - mean(teacher_cost)) / clamp(std(teacher_cost), 1e-6)
s_norm = (student_cost - mean(student_cost)) / clamp(std(teacher_cost), 1e-6)
L_score = weighted_mean(SmoothL1(s_norm, t_norm, beta=1.0))
```

teacher ranks 1–12 的 weight 为 2，其余 52 个 candidate 的 weight 为 1，最后按
权重归一。`goal` 只进入 frozen official external criterion；`official_model` 的
parameters 保持 `requires_grad=False`，但 student prediction 对 criterion 保留 input
gradient。`shuffled_score` 仅置乱 teacher score labels，不改变 candidate bank。

## 2. Screening gate（收集结果前冻结）

主比较为 `score_distill` 相对已验证的 `pairwise_rank` concurrent reference。step 1500
的 held-out 16 blocks 必须同时满足：

- Spearman median `>= 0.701261`；
- top-30 median `>= 0.433333`；
- relative latent MSE median `<= 0.0175`；
- positive top-30 blocks **恰好为 16/16**；
- minimum top-30 overlap `> 0`；
- `shuffled_score` 不能同时达到同一组 score-distill screening 条件。

这些是 screening 条件，不替代每一臂独立的原 absolute predictor gate；任意 arm 的
absolute ranking/fidelity gate 失败时，只报告对应 `NO-GO`。统计单位是 held-out
block，不能把 candidates 当成独立 replicates。

## 3. 报告与边界

报告三臂 step 500/1000/1500 snapshots、step1500 paired deltas、relative latent
MSE、causality、predictor-only latency、GPU telemetry、screening gate 与 absolute
predictor gate。仅拉取 summary、job.log、job_status；`prepared_rows.pt` 不回传。

本实验停在 predictor-level；不运行 official CEM、planner viability 或 closed-loop。
