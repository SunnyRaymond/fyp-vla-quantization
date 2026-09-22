# DINO-WM PushT：CEM proposal score distillation 冻结协议

状态：`frozen / predictor-level first`

## 1. 边界与输入

本实验只比较固定 CEM proposal 数据上的 recurrent student training objective。
不创建 environment，不执行 student closed loop，不运行官方 planner。输入是已有
`24928207.pbs101` 生成的 `cem_dagger_dataset.pt`：8 个 collector contexts、5 个
CEM checkpoints（`1/5/10/20/30`）、每个 context/checkpoint 120 行，共 4800 行。
不得重新运行 CEM collector 或重新采集这些 teacher labels。数据字段为
`context_visual/context_proprio`、`actions`、`target_visual/target_proprio`、
`teacher_cost` 及 `context_index/label_iteration`。

为了保留旧 offline replay 对照，复用既有 256 train contexts 的 immutable 四槽
query-slate bank（logged action、两个 Gaussian proposal、one-step CEM proposal），
并使用冻结的 `replay_schedule_seed=20261201`；每个 update 独立选择 8 个 old replay
contexts。replay contexts 不声称与 8 个 CEM collector contexts 相同，也不改变
CEM-DAgger label dataset。

## 2. 三臂与 grouped update

三臂从同一 horizon-weighted step-1500 state dict 开始，使用相同的 500 updates、
AdamW、learning rate、weight decay、horizon weights、256-context replay slate、
replay schedule、group schedule 和 candidate schedule：

| arm | CEM proposal objective | replay objective |
|---|---|---|
| `latent_only` | 0.5 × DAgger grouped latent loss | 0.5 × exact old replay latent loss |
| `score_distill` | 同上 + `0.1 × score loss` | 同上 |
| `shuffled_score` | 同上 + `0.1 × shuffled score loss` | 同上 |

每个 update 选择一个完整 checkpoint group（按 step modulo 五个 checkpoint 循环），
并对 8 个 contexts 各从该 context 的 120 candidates 用预计算 CPU permutation 取
32 行。因此 DAgger batch 的逻辑形状是 `[contexts=8, candidates=32]`（实现时可展平
为 256 rows）；三臂完全共享这些 row indices。每个 update 的 latent loss 先分别
计算 DAgger 与四-query replay 的 weighted horizon MSE，再取两者平均，避免来源行数
把 DAgger 权重放大。

`score_distill` 只在完整的 `[8,32]` context slab 上计算 score loss，不能把 candidate
当成独立 replicate。对每个 context：

```text
t_norm = (teacher_cost - mean_context(teacher_cost)) /
         clamp(std_context(teacher_cost), 1e-6)
s_norm = (student_cost - mean_context(student_cost)) /
         clamp(std_context(teacher_cost), 1e-6)
L_context = weighted_mean(SmoothL1(s_norm, t_norm, beta=1.0))
```

teacher-cost ranks 1–8 的 weight 为 2，其余 24 个为 1；先求每个 context 的
weighted mean，再对 8 个 contexts 求 mean。`shuffled_score` 只在该完整 context slab
内用冻结 seed 置换 teacher-cost labels，actions、targets、latent loss 和 schedule
均不变。`goal` 只进入 frozen official objective，teacher parameters 不反传。

## 3. Stage 1 predictor gate

在既有 8 held-out contexts、2 action-prefix seeds、16 blocks、300 candidates 上，报告
三臂各自的 absolute predictor gate、finite/causality、step-500 losses，并报告
`score_distill - latent_only` 与 `shuffled_score - latent_only` 的 paired block-level
deltas（Spearman、top-30、terminal relative latent MSE）。统计单位是 held-out block，
不是 candidate。

`score_distill` 只有同时满足以下条件才可进入 Stage 2：

- absolute gate：Spearman median/minimum `>= 0.95/0.85`，top-30 median/minimum
  `>= 0.80/0.50`，causality `<= 1e-6`，outputs finite 且无 silent fallback；
- 相对 `latent_only`：Spearman median delta `>= 0.005` 且至少 `10/16` blocks 改善，
  top-30 median delta `>= 1/30` 且至少 `10/16` blocks 改善；
- `shuffled_score` 不能同时满足同一组 paired improvement 条件。

条件在作业提交前冻结；不因结果调阈值。若不满足，Stage 2 必须为 `SKIPPED`。

## 4. Stage 2 fixed-observation CEM mechanism gate

只有 Stage 1 通过才运行既有六个 failed cases `[0,1,2,4,5,7]`，使用既有
`M=300/K=30/30`、eval seeds、checkpoints `1/5/10/30` 和 CPU innovations。
历史 teacher/full 与 student-only trace 只读，不重跑。Stage 2 primary arm 是
`score_distill`，沿用既有 support gate：iteration-30 first-action RMS median
`<=0.15`、coordinate absolute max `<=0.25`、teacher-cost support regret median
不高于历史 student-only，且至少 4/6 cases 不 worse；另外 score 相对
`latent_only` 的 iteration-30 first-action drift median 必须严格更低，且至少 4/6
cases 不 worse；teacher-cost support regret median 也必须严格更低且至少 4/6 cases
不 worse。`shuffled_score` 不得同时复现 score 的 paired predictor gain 与上述
mechanism gains。三臂 case records 均须报告，但不产生 closed-loop success 或
deployment claim。

## 5. PBS 与数据边界

PBS wrapper 只在 compute node 解压已 staged runtime，拒绝 login/head/submit node，
并每 30 秒把 GPU telemetry 写入 `job.log`/`gpu_usage.csv`。login node 不下载、安装、
编译、解压、加载模型或运行 benchmark。成功输出只包含本实验 summary、三臂 step-500
checkpoint、case records、status 与 telemetry；源 `cem_dagger_dataset.pt` 不回传。
