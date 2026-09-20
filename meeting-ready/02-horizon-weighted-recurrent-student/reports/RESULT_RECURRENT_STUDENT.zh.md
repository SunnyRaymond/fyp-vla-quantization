# DINO-WM PushT：Shared Recurrent Latent-Transition Student 结果

## 结论先行

这次实验完整运行并通过了实现完整性、收敛、causality、latent non-inferiority 和 predictor-only latency 检查，但没有通过冻结的 architecture-effect 或 absolute-fidelity gates。因此，shared recurrent latent-transition student 目前不能作为可靠的 predictor replacement。

最重要的结果是：它确实比 direct baseline 更小、更快，并且 ranking 大多有正向趋势，但收益没有达到预先冻结的幅度和稳定性要求：

- 16 个 paired held-out blocks 上，Spearman 的 median Δ 为 `+0.037226`，正向 `15/16`；
- top-30 overlap 的 median Δ 为 `+0.066667`，正向 `12/16`；
- 冻结 architecture-effect gate 要求分别达到 `+0.05`、`+0.10`，因此为 `FAIL`；
- recurrent student 只有 `732,554` 个参数，是 direct control 的 `2,413,450` 个参数的 `30.35%`；
- predictor-only latency 为 `10.805 ms`，相对 teacher reduction 为 `99.6914%`，相对 direct control latency ratio 为 `0.6553`；
- latent-MSE ratio 为 `0.7041`，满足 non-inferiority；
- frozen overall：`NO-GO`。

这说明当前 recurrent architecture 方向有工程价值和一定的 ranking 改善趋势，但本次证据还不足以宣称它稳定改善 candidate ranking，更不能外推为 closed-loop 或跨模型结论。

## 1. 实验边界与 control

这是 DINO-WM PushT 的 action-conditioned predictor-level paired experiment。唯一新变量是 predictor architecture：

- treatment：`RecurrentNativeLatentTransitionStudent`，shared recurrent latent transition；
- control：job `24494756.pbs101` 中已经完成的 read-only direct slate-MSE student；
- 两边复用相同的 context schedule、四-query slate bank、teacher targets、held-out candidate bank 和 paired block keys；
- 训练为 `1500` updates，snapshot 为 `500/1000/1500`；
- held-out 为 `2` 个 action-prefix seeds × `8` 个 contexts，共 `16` 个 paired blocks，每个 block `300` 个 candidates；
- 不包含 `encode_obs`、observation encoder、CEM 执行、environment interaction 或 closed-loop success。

本次 PBS job：`24503839.pbs101`。`job_status.txt` 显示 `completed`，runner 和 final exit code 均为 `0`；`NO-GO` 来自冻结 scientific gates，不是运行崩溃。

## 2. Student architecture

Student 从 cached native observation latent 的最后一个 context state 开始，之后每个 horizon step 都只输入自己的上一状态和当前 packed action token：

```text
(visual_k, proprio_k, packed_action_k)
          ↓ shared residual transition
(visual_{k+1}, proprio_{k+1})
          ↓ feedback into the next step
```

具体约束保持冻结：

- hidden size：`256`；
- 同一个 residual MLP 在 `k=0..4` 五步共享；
- visual patches 只做 mean pooling 以形成 transition summary；
- 无 goal input、无 `encode_obs` call、无 teacher forcing；
- 无 teacher/source encoder/source predictor parameters；
- 无 attention、per-horizon cell copy、rank loss 或额外 self-consistency loss；
- 训练目标仍是 dense native-latent MSE。

参数量如下：

| architecture | total/trainable parameters |
|---|---:|
| recurrent treatment | `732,554 / 732,554` |
| direct control | `2,413,450 / 2,413,450` |
| recurrent / control | `0.303530` (`30.35%`) |

## 3. Integrity 与收敛

Integrity gate：`PASS`。

- manifest：`PASS`；旧 `128` 个 context 保持原值和顺序，新 `128` 个 context 追加；每 episode `8` 个 context；未引入新 trajectory；
- context schedule 与 query-slate schedule 一致：`true`；
- action slate 与 query-slate bank 一致：`true`；
- held-out candidate bank 按 key 一致：`true`；
- slate group 完整且 query 两两不同：`true`；
- goal input hidden：`true`；
- `encode_obs` entry points：空；
- source/teacher parameter reference：`false`；
- recurrent outputs、training、evaluation 均 finite；
- causality integrity：`true`；
- baseline 只读：`true`；
- 未见 OOM、NaN 或 silent fallback。

训练收敛 gate：`PASS`。

| 指标 | 数值 |
|---|---:|
| first latent-MSE | `3.281877` |
| median last-10 latent-MSE | `0.140877` |
| last-10 / first latent-MSE | `0.042926` |
| 冻结上限 | `0.8` |

因此 ranking 失败不能归因于 recurrent student 没有完成训练。

## 4. 16-block recurrent − direct-control deltas

正值代表 recurrent treatment 更好。正式判定使用 step `1500`。

| action seed | context | ΔSpearman | Δtop-30 |
|---:|---:|---:|---:|
| 20264925 | 0 | `+0.033495` | `+0.133333` |
| 20264925 | 1 | `+0.044684` | `+0.166667` |
| 20264925 | 2 | `+0.049029` | `+0.266667` |
| 20264925 | 3 | `+0.077655` | `-0.066667` |
| 20264925 | 4 | `+0.025050` | `-0.066667` |
| 20264925 | 5 | `+0.030590` | `+0.133333` |
| 20264925 | 6 | `+0.015715` | `+0.033333` |
| 20264925 | 7 | `+0.004643` | `0.000000` |
| 20264926 | 0 | `+0.040956` | `+0.200000` |
| 20264926 | 1 | `+0.043932` | `+0.100000` |
| 20264926 | 2 | `+0.048459` | `+0.066667` |
| 20264926 | 3 | `+0.048426` | `+0.066667` |
| 20264926 | 4 | `+0.050791` | `+0.100000` |
| 20264926 | 5 | `+0.020499` | `+0.066667` |
| 20264926 | 6 | `+0.017680` | `+0.033333` |
| 20264926 | 7 | `-0.002817` | `0.000000` |

汇总：

| metric | mean Δ | median Δ | positive blocks | frozen requirement |
|---|---:|---:|---:|---:|
| Spearman | `+0.034299` | `+0.037226` | `15/16` | median `>=+0.05`，positive `>=12/16` |
| top-30 overlap | `+0.077083` | `+0.066667` | `12/16` | median `>=+0.10`，positive `>=12/16` |

两项的 positive-block 数量达到要求，但 median effect size 都不够。因此 architecture-effect gate 为 `FAIL`。这是一个“方向一致但幅度不足”的结果，而不是完全没有信号。

作为训练过程参考，recurrent treatment 在 step `500/1000/1500` 的 absolute ranking 逐步变化，但 architecture-effect 的正式结论只使用冻结的 step `1500`，没有根据中间 snapshot 选择更有利的结果。

## 5. Absolute fidelity

Absolute fidelity gate：`FAIL`。

step `1500` recurrent student 的 absolute ranking：

| metric | mean | median | minimum |
|---|---:|---:|---:|
| Spearman | `0.955651` | `0.966345` | `0.881070` |
| top-30 overlap | `0.770833` | `0.800000` | `0.533333` |

冻结要求为 median Spearman `>=0.99`、minimum Spearman `>=0.95`、median top-30 `>=0.95`、minimum top-30 `>=0.80`。因此虽然 recurrent 比 direct control 有明显正向 paired trend，但仍未达到“可安全替换 teacher ranking”的绝对精度标准。

## 6. Logged latent fidelity

在相同 held-out logged action prefixes 上：

- recurrent mean teacher-relative MSE：`0.042467`；
- direct control mean teacher-relative MSE：`0.060312`；
- recurrent / control ratio：`0.704108`；
- 冻结上限：`1.25`；
- latent non-inferiority：`PASS`。

这个结果表示 recurrent student 的 latent reconstruction 在本次 paired measurement 中没有退化，甚至低于 control 的 logged MSE。它支持“ranking 正向趋势不是由 latent fidelity 崩坏导致”的解释，但不能把较低的 logged MSE 自动解释为 planner ranking 已经可靠提升。

## 7. Causality

Causality gate：`PASS`。

对 unchanged prefix length `1/2/3/4`，未来 action 改变对 unchanged prefix 输出的最大绝对差均为 `0.0`，容差为 `1e-6`。因此 recurrent feedback 没有把未来 action 的影响泄漏到已经确定的 prefix 输出中。

## 8. Predictor-only latency

边界为 cached native observation latent + normalized action prefix → predictor rollout；不包括 `encode_obs`、CEM、environment 或 closed-loop 控制。测量为 batch `300`、warmup `3`、technical repeats `10`，并包含 CUDA synchronization。

| arm | teacher median | student median | speedup | reduction |
|---|---:|---:|---:|---:|
| recurrent | `3501.403 ms` | `10.805 ms` | `324.05x` | `99.6914%` |
| direct control | `3513.662 ms` | `16.488 ms` | `213.10x` | `99.5307%` |

Recurrent / direct-control student latency ratio：`0.655339`，即约低 `34.47%`。因此它确实减少了 direct compact student 的 predictor-only latency，同时参数量也降到约 `30.35%`。这两个是明确的 engineering tradeoff 结果，但不能替代 ranking gate。

GPU telemetry 为 A100，运行期间未见 OOM/Xid；峰值显存约 `26,343 MiB`。远端 checkpoints 和 `plan_targets.pkl` 未取回。

## 9. Decision levels

| decision | result | reason |
|---|---|---|
| `architecture_effect` | **FAIL** | positive blocks 足够，但 median ΔSpearman=`+0.037226`、median Δtop-30=`+0.066667` 未达到冻结 effect-size gates |
| `full_replacement` | **NO-GO** | absolute fidelity 未通过 |
| `recurrent_supported_replacement` | **NO-GO** | architecture effect 与 full replacement 未同时通过 |
| `overall` | **NO-GO** | 冻结 overall gate |

## 10. Predictor-level claim boundary

当前证据支持：在冻结的 DINO-WM PushT action-conditioned predictor cell 中，一个不调用 `encode_obs`、不含 teacher/source 参数的 shared recurrent latent-transition student，能够以约 `30.35%` 的参数量和约 `65.53%` 的 direct-control latency，保持通过 causality、收敛和 logged latent non-inferiority，并呈现一致但不足以过 gate 的 ranking 改善趋势。

当前证据不支持：

- 宣称 recurrent architecture 已稳定改善 candidate ranking；
- 宣称它已经可以 full replacement；
- 宣称 closed-loop CEM、environment success 或 control frequency 改善；
- 把 predictor-only reduction 当作包含 `encode_obs` 的端到端 latency；
- 将该结果直接外推到 LeWM、Fast-LeWM 或所有 JEPA-style world models；
- 将 paired effect 与旧的 context-density、rank-distillation 或 query-slate treatment 混为同一个 causal claim。

## Artifacts

- summary：[artifacts/24503839.pbs101/recurrent_student_summary.json](../artifacts/24503839.pbs101/recurrent_student_summary.json)
- log/status：[artifacts/24503839.pbs101/job.log](../artifacts/24503839.pbs101/job.log)、[job_status.txt](../artifacts/24503839.pbs101/job_status.txt)
- runner/final status：[runner_exit_status.txt](../artifacts/24503839.pbs101/runner_exit_status.txt)、[final_exit_status.txt](../artifacts/24503839.pbs101/final_exit_status.txt)
- GPU telemetry：[gpu_info.csv](../artifacts/24503839.pbs101/gpu_info.csv)、[gpu_usage.csv](../artifacts/24503839.pbs101/gpu_usage.csv)、[gpu_telemetry.jsonl](../artifacts/24503839.pbs101/gpu_telemetry.jsonl)

本地只取回上述小型 artifacts；远端 recurrent checkpoints 与 `plan_targets.pkl` 未取回。
