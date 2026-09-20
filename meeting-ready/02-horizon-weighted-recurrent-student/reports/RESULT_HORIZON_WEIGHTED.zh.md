# DINO-WM PushT：Horizon-weighted Recurrent Student 结果

## 结论先行

这次实验完整运行成功，integrity、训练收敛、causality、latent non-inferiority 和 predictor-only latency 均通过；但冻结的 horizon-weighted effect 和 absolute-fidelity gates 未通过，因此当前方法不能作为可靠的 predictor replacement。

核心结果如下：

- 16 个 paired held-out blocks 上，horizon-weighted treatment 相对 uniform recurrent control 的 Spearman median Δ 为 `+0.004083`，`16/16` 个 block 为正；
- top-30 overlap median Δ 为 `+0.033333`，`9/16` 个 block 为正；
- 冻结 effect gate 要求 median ΔSpearman `>=+0.05`、median Δtop-30 `>=+0.10`，因此 effect 为 `FAIL`；
- absolute ranking 为 Spearman median/minimum `0.970596/0.894798`，top-30 median/minimum `0.833333/0.600000`，未达到冻结 replacement gate；
- weighted/control 的 logged latent MSE ratio 为 `0.927539`，满足上限 `1.25`；
- predictor-only latency 相对 teacher reduction 为 `99.6936%`；weighted 与 uniform recurrent control 的参数量相同，均为 `732,554`；
- 三个关键决策为：`horizon_weighted_effect=FAIL`、`full_replacement=NO-GO`、`horizon_weighted_supported_replacement=NO-GO`。

因此，按后验排序是安全的，但本次结果不支持“仅通过提高长 horizon loss 权重就能改善 planner candidate ranking”的 claim。

## 1. 实验边界与 control

这是 DINO-WM PushT 的 action-conditioned predictor-level paired experiment。唯一新变量是训练 loss 的 horizon weighting：

- treatment：`horizon_weighted_recurrent_shared_latent_transition`；
- control：job `24503839.pbs101` 的 uniform recurrent student，read-only 复用；
- 两边使用相同的 recurrent architecture、context schedule、四-query slate bank、teacher targets、held-out candidate bank 和 paired block keys；
- 训练 `1500` updates，snapshot 为 `500/1000/1500`；
- held-out 为 `2` 个 action-prefix seeds × `8` 个 contexts，共 `16` 个 paired blocks，每个 block `300` 个 candidates；
- 不包含 `encode_obs`、observation encoder、CEM 执行、environment interaction 或 closed-loop success。

PBS job `24510395.pbs101` 的 `job_status.txt` 显示 `completed`，`final_exit_status=0`。最终 `NO-GO` 来自冻结 scientific gates，不是程序崩溃。

## 2. Per-horizon loss 定义与权重

每个 horizon 的基础误差先定义为 visual 和 proprio latent MSE 的平均：

```text
per_horizon_mse[h] = 0.5 * (visual_mse[h] + proprio_mse[h])
```

五个 horizon 的冻结权重为：

```text
[1/3, 2/3, 1, 4/3, 5/3]
```

权重均值为 `1`，训练 loss 为：

```text
weighted_loss = mean_h(weights[h] * per_horizon_mse[h])
```

因此它提高了较远 horizon 的训练梯度，但没有改变 architecture、数据、slate、seed 或 evaluation rule。

训练中的 per-horizon MSE 从 step 1 到 step 1500 为：

| horizon | step 1 | step 1500 | weight |
|---:|---:|---:|---:|
| 1 | `0.467473` | `0.050515` | `0.333333` |
| 2 | `1.442675` | `0.093045` | `0.666667` |
| 3 | `2.869886` | `0.125439` | `1.000000` |
| 4 | `4.660779` | `0.159103` | `1.333333` |
| 5 | `6.968569` | `0.181989` | `1.666667` |

weighted loss 从 `4.363229` 降至 `0.143952`；median last-10 / first weighted loss ratio 为 `0.036670`，训练收敛 gate 为 `PASS`。

## 3. Integrity 与 paired setup

Integrity gate：`PASS`。

- recurrent architecture 与 uniform control 完全一致；
- context schedule、action slate bank、teacher target bank 和 held-out candidate bank 均按 key 配对一致；
- 256 train contexts、8 held-out contexts、每 block 300 candidates 的冻结数量满足；
- horizon weights 与 freeze specification 完全一致；
- `encode_obs` 未被调用，student 未引用 source encoder、teacher predictor 或 teacher 参数；
- outputs、training、evaluation 均 finite；
- causality check 通过；
- baseline 为 read-only；
- 未发生 OOM、NaN 或 silent fallback。

## 4. 16-block weighted − uniform-recurrent deltas

正式判定使用 step `1500`，正值表示 horizon-weighted treatment 更好：

| metric | mean Δ | median Δ | positive blocks | frozen requirement |
|---|---:|---:|---:|---:|
| Spearman | `+0.007431` | `+0.004083` | `16/16` | median `>=+0.05`，positive `>=12/16` |
| top-30 overlap | `+0.027083` | `+0.033333` | `9/16` | median `>=+0.10`，positive `>=12/16` |

这不是完全没有信号：Spearman 在所有 paired blocks 都为正。但 effect size 很小，top-30 也没有稳定达到要求，所以不能把它解释成 planner ranking 的实质性改善。冻结 horizon-weighted effect gate 为 `FAIL`。

## 5. Absolute fidelity

Absolute-fidelity gate：`FAIL`。

step `1500` 的 treatment ranking 为：

| metric | mean | median | minimum |
|---|---:|---:|---:|
| Spearman | `0.963083` | `0.970596` | `0.894798` |
| top-30 overlap | `0.797917` | `0.833333` | `0.600000` |

冻结要求为 median Spearman `>=0.99`、minimum Spearman `>=0.95`、median top-30 `>=0.95`、minimum top-30 `>=0.80`。因此加权 loss 没有把 student 推到可安全替换 teacher ranking 的精度区间。

## 6. Logged latent fidelity 与 causality

在相同 logged action prefixes 上，weighted/control logged latent MSE ratio 为 `0.927539`，冻结上限为 `1.25`，因此 latent non-inferiority：`PASS`。这说明加权训练没有造成总体 latent reconstruction 退化，但该指标没有转化成 candidate ranking 的明显提升。

Causality：`PASS`。未来 action 不影响已经确定的 prefix 输出，最大差异在冻结容差内（summary 标记为 `PASS`）。

## 7. Predictor-only latency 与参数量

测量边界是 cached native observation latent + normalized action prefix → predictor rollout；不包括 `encode_obs`、CEM、environment 或 closed-loop control。batch `300`、warmup `3`、technical repeats `10`，包含 CUDA synchronization。

| arm | median teacher | median student | speedup | reduction |
|---|---:|---:|---:|---:|
| horizon-weighted | `3498.452 ms` | `10.719 ms` | `326.37×` | `99.6936%` |
| uniform recurrent control | `3501.403 ms` | `10.805 ms` | `324.05×` | `99.6914%` |

两者 architecture 和参数量完全相同（各 `732,554`），所以本实验不测试 model-size 或 architecture gain；它只测试 loss weighting 是否能在相同 student 上改善 ranking。

GPU telemetry 显示 A100，峰值显存约 `26,343 MiB / 40,960 MiB`，未见 OOM/Xid。远端 checkpoints 与 `plan_targets.pkl` 未取回。

## 8. Decision levels

| decision | result | reason |
|---|---|---|
| `integrity` | **PASS** | weights、pairing、student isolation、finite outputs 和 causality 均通过 |
| `convergence` | **PASS** | median last-10 / first weighted loss ratio `0.036670`，低于 `0.8` |
| `horizon_weighted_effect` | **FAIL** | ΔSpearman 和 Δtop-30 的 median effect size 均不足 |
| `full_replacement` | **NO-GO** | absolute fidelity 未通过 |
| `horizon_weighted_supported_replacement` | **NO-GO** | effect 与 replacement 未同时通过 |
| `overall` | **NO-GO** | 冻结 overall gate |

## 9. Predictor-level claim boundary

当前证据支持：在冻结的 DINO-WM PushT recurrent action-conditioned predictor 中，使用 mean-normalized、随 horizon 增长的 loss weights 可以稳定训练，并保持 latency、causality 和 logged latent non-inferiority；它产生了 Spearman 方向一致但幅度很小的正向趋势。

当前证据不支持：

- 宣称 horizon weighting 已实质改善 candidate ranking；
- 宣称 weighted recurrent student 可以 full replacement；
- 宣称 closed-loop CEM、environment success 或 control frequency 改善；
- 把 predictor-only latency reduction 当成包含 `encode_obs` 的端到端 latency；
- 将结果直接外推到 LeWM、Fast-LeWM 或所有 JEPA-style world models；
- 把本实验与 context-density、query-slate 或 recurrent-architecture treatment 混为同一个 causal claim。

## Artifacts

- summary：[artifacts/24510395.pbs101/horizon_weighted_summary.json](../artifacts/24510395.pbs101/horizon_weighted_summary.json)
- log/status：[artifacts/24510395.pbs101/job.log](../artifacts/24510395.pbs101/job.log)、[job_status.txt](../artifacts/24510395.pbs101/job_status.txt)
- runner/final status：[runner_exit_status.txt](../artifacts/24510395.pbs101/runner_exit_status.txt)、[final_exit_status.txt](../artifacts/24510395.pbs101/final_exit_status.txt)
- GPU telemetry：[gpu_info.csv](../artifacts/24510395.pbs101/gpu_info.csv)、[gpu_usage.csv](../artifacts/24510395.pbs101/gpu_usage.csv)、[gpu_telemetry.jsonl](../artifacts/24510395.pbs101/gpu_telemetry.jsonl)

本地只取回上述指定 artifacts；远端 checkpoints 与 `plan_targets.pkl` 未取回。
