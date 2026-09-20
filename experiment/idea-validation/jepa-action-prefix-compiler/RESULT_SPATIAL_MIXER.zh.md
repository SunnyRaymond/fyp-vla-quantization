# DINO-WM PushT：Spatial Token Mixer Student 结果

## 结论先行

本次 `24516588.pbs101` 完整运行成功，integrity、训练收敛、causality、logged latent non-inferiority 和 predictor-only latency 均通过；但加入 spatial/token mixer 没有改善 planner candidate ranking，反而在最终 snapshot 上略有下降。因此冻结 overall gate 为 `NO-GO`，不能把该模型宣称为可靠的 predictor replacement。

核心结果：

- 16 个 paired held-out blocks 上，spatial-mixer treatment 相对 horizon-weighted recurrent control 的 Spearman median Δ 为 `-0.002755`，只有 `2/16` 个 block 为正；
- top-30 overlap median Δ 为 `0.000000`，只有 `1/16` 个 block 为正；冻结 effect gate 要求分别为 `>=+0.05`、`>=+0.10` 且至少 `12/16` 个 block 为正，因此 effect 为 `FAIL`；
- treatment 的 absolute ranking 为 Spearman median/minimum `0.965205/0.878174`，top-30 median/minimum `0.783333/0.533333`，未达到 frozen replacement gate；
- treatment/control 的 logged latent MSE ratio 为 `1.136222`，低于 `1.25` 的 non-inferiority 上限；
- predictor-only latency 为 `26.024 ms`，相对 teacher `3504.172 ms`，speedup `134.65×`、reduction `99.2573%`；
- spatial mixer 参数量 `996,234`，baseline recurrent 参数量 `732,554`，为 baseline 的 `1.3599×`；
- 决策为：`integrity=PASS`、`convergence=PASS`、`spatial_mixer_effect=FAIL`、`full_replacement=NO-GO`、`spatial_mixer_supported_replacement=NO-GO`、`overall=NO-GO`。

换句话说：这次实验验证了一个干净的结构变量——在当前 recurrent student 上增加 patch-token mixing 是可运行、可因果、仍然极快的；但在当前 PushT paired protocol 下，它没有把 latent 预测质量转化成 planner ranking 收益。不能仅凭该结果宣称 spatial mixing 对 JEPA-style world models 普遍有效。

## 1. 实验边界与 control

这是 predictor-level paired experiment，唯一 treatment 结构变量是一个共享的 token mixer：

- treatment：`SpatialMixerNativeLatentStudent hidden_dim=256`；
- control：job `24510395.pbs101` 的 horizon-weighted recurrent student，read-only 复用；
- 两边使用相同的 256 个 train contexts、context schedule、四种 query slate、teacher targets、held-out candidate bank、初始化 seed 和 1500-step schedule；
- held-out 为 `2` 个 action-prefix seeds × `8` 个 contexts，共 `16` 个 paired blocks；每个 block `300` 个 candidates；
- 不包含 `encode_obs`、observation encoder、CEM 执行、environment interaction 或 closed-loop success；
- treatment 的 horizon loss 与 control 相同：每个 horizon 先计算 `0.5 * (visual MSE + proprio MSE)`，再使用 mean-normalized 权重 `[1/3, 2/3, 1, 4/3, 5/3]` 聚合。

Spatial mixer 为一个共享的 pre-norm、4-head `MultiheadAttention`，作用在 projected visual patch tokens 上，并使用 residual；随后仍使用共享 residual transition MLP 和 predicted native-latent feedback。它不调用 `encode_obs`，不包含 goal 输入，也不包含 teacher/source encoder 参数。

## 2. Integrity 与收敛

Integrity gate：`PASS`。

- treatment/control 的 paired context schedule、action slate、teacher target bank 和 held-out candidate bank 一致；
- 4 种 query slate 在每个 context 内保持 pairwise distinct；
- student 未引用 teacher/source predictor 或 `encode_obs`；
- outputs、training、evaluation 均 finite；无 OOM、NaN 或 silent fallback；
- causal feedback 检查通过；baseline 为 read-only；
- A100 telemetry 正常，峰值显存约 `26,351 / 40,960 MiB`。

训练 weighted loss 从 step 1 的 `5.019744` 降至 step 1500 的 `0.143311`，median last-10 / first ratio 为 `0.032300`，因此 convergence gate：`PASS`。

## 3. Paired ranking effect

正式判定使用 step `1500`，正值表示 spatial-mixer treatment 优于 recurrent control：

| metric | mean Δ | median Δ | positive blocks | frozen requirement |
|---|---:|---:|---:|---:|
| Spearman | `-0.005182` | `-0.002755` | `2/16` | median `>=+0.05`，positive `>=12/16` |
| top-30 overlap | `-0.020833` | `0.000000` | `1/16` | median `>=+0.10`，positive `>=12/16` |

snapshot 也没有显示晚期反转：

| snapshot | Spearman median Δ | top-30 median Δ | positive Spearman | positive top-30 |
|---:|---:|---:|---:|---:|
| 500 | `-0.003706` | `0.000000` | `4/16` | `5/16` |
| 1000 | `-0.000051` | `0.000000` | `8/16` | `2/16` |
| 1500 | `-0.002755` | `0.000000` | `2/16` | `1/16` |

因此不是“训练尚未收敛”导致 effect gate 未过：loss 已收敛，而且 paired effect 在三个 snapshot 都没有达到冻结幅度要求。

## 4. Absolute ranking fidelity

Absolute-fidelity gate：`FAIL`。

| metric | mean | median | minimum | frozen requirement |
|---|---:|---:|---:|---:|
| Spearman | `0.957901` | `0.965205` | `0.878174` | median `>=0.99`、minimum `>=0.95` |
| top-30 overlap | `0.777083` | `0.783333` | `0.533333` | median `>=0.95`、minimum `>=0.80` |

虽然整体 ranking 已有较高相关性，但 worst-case block 仍明显低于安全替换要求；而且 spatial mixer 相比 control 没有提高这个瓶颈。

## 5. Latent fidelity 与 causality

在相同 logged action prefixes 上，spatial-mixer/control 的 logged latent MSE ratio 为 `1.136222`，满足 `<=1.25` 的 non-inferiority gate。这表示额外 token mixing 没有造成不可接受的 latent reconstruction 退化，但该结果没有转化为 candidate ranking 改善。

Causality gate：`PASS`。修改未来 action 后，已确定 prefix 的输出保持不变，summary 中所有冻结 prefix cases 均通过容差 `1e-6`。

## 6. Predictor-only latency 与参数量

测量边界为 cached native observation latent + normalized action prefix → predictor rollout，batch `300`、horizon `5`、warmup `3`、technical repeats `10`，包含 CUDA synchronization。它不包含 `encode_obs`、CEM、environment 或 closed-loop control。

| arm | median teacher | median student | speedup | reduction |
|---|---:|---:|---:|---:|
| spatial mixer | `3504.172 ms` | `26.024 ms` | `134.65×` | `99.2573%` |
| horizon-weighted recurrent control | `3498.452 ms` | `10.719 ms` | `326.37×` | `99.6936%` |

加入 token mixer 使 student latency 约为 control 的 `2.43×`（`26.024/10.719`），参数量增加约 `36.0%`；虽然仍远快于 teacher，但在没有 ranking 收益的情况下，这个额外开销没有被当前 frozen protocol 证明值得。

## 7. Decision levels

| decision | result | reason |
|---|---|---|
| `integrity` | **PASS** | paired setup、student isolation、finite outputs、causality 均通过 |
| `convergence` | **PASS** | last-10 / first weighted MSE ratio `0.032300` |
| `spatial_mixer_effect` | **FAIL** | ranking deltas 接近零或为负，且 positive blocks 很少 |
| `full_replacement` | **NO-GO** | absolute ranking fidelity 未通过 |
| `spatial_mixer_supported_replacement` | **NO-GO** | effect 与 replacement 未同时通过 |
| `overall` | **NO-GO** | frozen overall gate |

## 8. Predictor-level claim boundary

当前证据支持：在冻结的 DINO-WM PushT action-conditioned predictor protocol 下，一个共享的 pre-norm patch-token mixer 可以被接入 recurrent student；它满足 causality、训练收敛、logged latent non-inferiority，并保持相对 teacher 的大幅 predictor-only latency reduction。

当前证据不支持：

- 宣称 spatial/token mixing 已改善 planner candidate ranking；
- 宣称该 student 可以 full replacement；
- 宣称 closed-loop CEM、environment success 或端到端 control frequency 改善；
- 把 predictor-only latency reduction 当作包含 `encode_obs` 的端到端 latency；
- 将结果直接外推到 LeWM、Fast-LeWM 或所有 JEPA-style world models；
- 将本实验解释成 observation encoder、context density、rank loss 或 horizon weighting 的 causal 结果。

## Artifacts

- summary：[artifacts/24516588.pbs101/spatial_mixer_summary.json](artifacts/24516588.pbs101/spatial_mixer_summary.json)
- log/status：[artifacts/24516588.pbs101/job.log](artifacts/24516588.pbs101/job.log)、[job_status.txt](artifacts/24516588.pbs101/job_status.txt)
- runner/final status：[runner_exit_status.txt](artifacts/24516588.pbs101/runner_exit_status.txt)、[final_exit_status.txt](artifacts/24516588.pbs101/final_exit_status.txt)
- GPU telemetry：[gpu_info.csv](artifacts/24516588.pbs101/gpu_info.csv)、[gpu_usage.csv](artifacts/24516588.pbs101/gpu_usage.csv)

PBS job `24516588.pbs101` 的 `status=completed`、`runner_exit_code=0`、`final_exit_code=0`。本地只取回上述小型 artifacts；远端 checkpoints 与 `plan_targets.pkl` 未取回。
