# LeWM PushT Horizon-Weighted Recurrent Student：Predictor-Level 结果

## 结论

最终有效作业为 `24564619.pbs101`，`Exit_status=0`。当前 best recurrent student
在 LeWM 上带来很大的 predictor-only 加速，但没有保住 planner-facing candidate
ranking，因此冻结的 predictor feasibility gate 为 **NO-GO**。

最重要的结果不是“student 学不会 latent”。恰恰相反，latent cosine 很高、relative
latent MSE 很低；问题是这些小误差仍足以改变候选 action prefixes 的 terminal-cost
顺序。

## 固定实验边界

- backend：official LeWM PushT，source commit
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`；
- teacher：约 `10.79M` 参数的 6-layer action-conditioned causal Transformer；
- student：`775,872` 参数的 shared h256 residual recurrent transition；
- policy 初始 history：`H=1`，predictor 最大 history：`3`；
- action prefix：`25×2` primitive actions，打包为 `5×10` tokens；
- train contexts：`256`；held-out contexts：`8`，episode-disjoint；
- evaluation：`8 contexts × 2 fresh seeds = 16 blocks`，每 block `300`
  candidates，top-k `30`；
- training：`1500` updates，predicted-latent free-running feedback，horizon weights
  `[1/3, 2/3, 1, 4/3, 5/3]`；
- timing：cached current latent + five packed action tokens → five-step predictor；
  不包含 encoder、CEM、environment；
- 按用户要求，official CEM Stage B **未运行**。

## Primary gate

| 指标 | 冻结要求 | step 1500 | 判定 |
|---|---:|---:|---|
| median Spearman | `≥ 0.95` | `0.415374` | FAIL |
| minimum Spearman | `≥ 0.80` | `-0.404356` | FAIL |
| median top-30 overlap | `≥ 0.75` | `0.266667` | FAIL |
| minimum top-30 overlap | `≥ 0.50` | `0.000000` | FAIL |
| median relative latent MSE | `≤ 0.25` | `0.012116` | PASS |
| positive Spearman blocks | `≥ 12/16` | `12/16` | PASS |
| positive top-30 blocks | `≥ 12/16` | `15/16` | PASS |
| predictor latency reduction | `≥ 20%` | `92.1251%` | PASS |
| causality | all prefixes `≤1e-6` | prefixes 1–4 均 `0` | PASS |

综合结论：`predictor_feasibility = NO-GO`。

## 主要收益来自哪里

Teacher 对每个未来 step 都重新运行最多 3-token history 上的完整 6-layer
Transformer。Student 把它替换为一个五步共享的 residual MLP transition，并把自己的
192-D prediction 反馈到下一步：

| predictor | 参数量 | 300 candidates、5-step median latency |
|---|---:|---:|
| official LeWM teacher | `10,791,360` | `23.5295 ms` |
| recurrent student | `775,872` | `1.85293 ms` |

Student 约小 `13.9×`，该冻结 timing boundary 上约快 `12.70×`，即 latency reduction
`92.1251%`。这是本实验的主要收益；不包含 observation encoder，因此不能解释成完整
planner 或 end-to-end control 的 `12.70×`。

## 为什么 latent 很像，ranking 仍然失败

step 1500 的 held-out median per-horizon 结果为：

| horizon | latent MSE | cosine |
|---:|---:|---:|
| 1 | `0.004032` | `0.998058` |
| 2 | `0.008407` | `0.995885` |
| 3 | `0.011723` | `0.993980` |
| 4 | `0.012982` | `0.992990` |
| 5 | `0.020116` | `0.989661` |

这些数值说明 student 能复现 teacher latent 的大尺度位置，但 ranking 依赖的是候选之间
很小的 terminal goal-distance 差异。共同偏移或 action sensitivity 的局部失真可能只占
很小 latent MSE，却足以交换 elite candidates 的顺序。因此“cosine 接近 1”不能作为
planner-facing replacement 的充分条件。

16 个 blocks 中 Spearman 有 12 个为正，但分布从 `-0.4044` 到 `0.7067`；top-30
overlap 从 `0` 到 `0.5333`。这不是稳定、可替换的排序保持。

## 训练是否正常

- weighted training MSE：step 1 `4.20265`；最后 10 steps median `0.0127881`；
- `last10 / first = 0.003043`，远低于收敛 gate `0.8`；
- 所有 evaluation 输出 finite；
- causality prefixes 1–4 的最大差异均为 `0`；
- snapshot ranking 从 step 500 → 1000 → 1500 持续提高：median Spearman
  `0.0360 → 0.3189 → 0.4154`，但离 `0.95` gate 仍很远；top-30 median
  `0.1000 → 0.2333 → 0.2667`。

所以本次 NO-GO 不是训练崩溃、NaN、future-action leakage 或 latency 不够。单次冻结
实验不能严格区分“模型容量不足”和“训练覆盖不足”，但它明确说明：当前 h256、256
contexts、1500 updates 的 best DINO-derived mechanism，直接迁移到 LeWM 后不足以保留
planner-facing ranking。

## 证据与失败记录

- authoritative interface probe：`24554356.pbs101`，证明 `H=1`、五步 official
  target contract；
- final experiment：`24564619.pbs101`，`PREDICTOR_LEVEL_COMPLETE`；
- `24562554.pbs101`：在读取第一段 pixels 时因 HDF5 plugin 未注册而退出；
- `24563988.pbs101`：在第一次 encode 前因缺 time axis 而退出；
- 上述两个失败作业均未进入 teacher-target generation 或 student training，不是实验臂。

权威小型 artifacts：

- [summary JSON](artifacts/24564619.pbs101/lewm_recurrent_student_summary.json)
- [context manifest](artifacts/24564619.pbs101/context_manifest.json)
- [job log](artifacts/24564619.pbs101/job.log)
- [job status](artifacts/24564619.pbs101/job_status)

## Claim boundary

可以说：当前 recurrent student 在 LeWM compact latent 上实现了约 `12.70×`
predictor-only speedup，并保持很高的 latent cosine，但没有通过 frozen ranking gate。

不能说：它可以安全替换 official LeWM predictor、保持 official CEM behavior、提升
closed-loop PushT success，或已经优于 Fast-LeWM。按当前实验范围，official CEM 与
environment 均未运行。
