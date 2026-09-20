# Query-slate planner-aware rank paired cell：结果

## 结论先行

这次实验没有证明 planner-aware rank loss 能改善 action-conditioned predictor 的 held-out candidate ranking。与完全相同 slate、初始化、context schedule 和 teacher target 的 latent-MSE control 相比，rank treatment 在最终 step 1500 反而略低：

- Spearman：median Δ = `-0.002639`，16 个 block 中仅 `6/16` 为正；
- top-30 overlap：median Δ = `-0.033333`，仅 `4/16` 为正；
- frozen paired-effect gate：`FAIL`。

因此这条具体的 rank-loss 配方不能作为 predictor 的替换方案。它保留了 latent fidelity（logged teacher-relative MSE ratio `1.0970 <= 1.25`），并且两臂都保持了约 `99.53%` 的 predictor-level latency reduction；但没有带来 planner ranking gain。

## 实验边界与固定条件

这是 DINO-WM PushT 的 predictor-level paired cell，比较：

- `control`：slate-trained latent-MSE student；
- `treatment`：同一 student、同一 slate 和同一 teacher target，加入 listwise planner-rank loss；
- 每个 context 使用 `4` 个 pairwise-distinct action queries；共 `256` 个 train contexts、`1500` updates、`16` 个 held-out blocks，每 block `300` candidates；
- 两臂共享 initialization、context schedule、action-slate bank、held-out candidate bank 与 seeds；rank weight=`0.1`，temperature=`1.0`。

本实验不测 `encode_obs`、CEM 执行、environment interaction 或 closed-loop task success。

## 1. Paired integrity

paired integrity 为 `PASS`：

- initial state equal：`true`；
- context schedule equal：`true`；
- slate bank equal：`true`；
- held-out candidate bank equal：`true`；
- 1500 个 updates 中每次的 `8` 个 slate groups 都完整，slate 内 query 两两不同；
- held-out split unchanged：`true`；
- student goal input hidden：`true`；
- no OOM / NaN / silent fallback：`true`；
- manifest integrity：`PASS`，旧 `128` 个 contexts 保序保值，新 `128` 个 contexts 追加，未引入新 trajectory。

PBS 本身正常结束：`Exit_status=0`；runner 与 final status 均为 `0`。这次的 `NO-GO` 是冻结 scientific gate 的结果，不是运行崩溃。

## 2. 两个 arm 的收敛

| arm | first latent-MSE | median last-10 latent-MSE | latent last-10 / first | 收敛 gate |
|---|---:|---:|---:|---|
| control | `0.321521` | `0.087937` | `0.273502` | PASS |
| treatment | `0.321521` | `0.091933` | `0.285931` | PASS |

两臂均满足冻结的 `<= 0.8` 收敛阈值。因此不能把 paired-effect 的失败归因于某一臂没有训练完。

## 3. 最终 step 1500 的 16-block treatment − control deltas

下面的每一行对应一个 held-out block；正值代表 treatment 更好。

| seed | context | ΔSpearman | Δtop-30 |
|---:|---:|---:|---:|
| 20264925 | 0 | `+0.002661` | `+0.000000` |
| 20264925 | 1 | `-0.010581` | `-0.133333` |
| 20264925 | 2 | `+0.001737` | `+0.033333` |
| 20264925 | 3 | `+0.020950` | `-0.100000` |
| 20264925 | 4 | `-0.002409` | `-0.033333` |
| 20264925 | 5 | `-0.008762` | `-0.033333` |
| 20264925 | 6 | `-0.002684` | `-0.033333` |
| 20264925 | 7 | `-0.002595` | `+0.033333` |
| 20264926 | 0 | `-0.008232` | `+0.000000` |
| 20264926 | 1 | `-0.013696` | `-0.066667` |
| 20264926 | 2 | `-0.011312` | `-0.066667` |
| 20264926 | 3 | `+0.001782` | `-0.066667` |
| 20264926 | 4 | `+0.005548` | `+0.066667` |
| 20264926 | 5 | `-0.016957` | `-0.033333` |
| 20264926 | 6 | `-0.010893` | `+0.033333` |
| 20264926 | 7 | `+0.000018` | `-0.033333` |

汇总：

- Spearman：mean `-0.003464`，median `-0.002639`，positive `6/16`；
- top-30：mean `-0.027083`，median `-0.033333`，positive `4/16`；
- 冻结阈值是 median Spearman `>= +0.05`、median top-30 `>= +0.10`，且两项各至少 `12/16` positive，均未达到。

作为过程对照，step 500 的 paired median 分别为 Spearman `-0.017502`、top-30 `-0.083333`；step 1000 分别为 `-0.002227`、`-0.033333`。训练变长没有把 treatment 转化为稳定的正收益。

## 4. Absolute treatment

最终 rank treatment 的 absolute ranking 为：

- Spearman：mean `0.917888`，median `0.927941`，minimum `0.835827`；
- top-30 overlap：mean `0.666667`，median `0.666667`，minimum `0.533333`。

absolute fidelity gate 仍为 `FAIL`（冻结阈值为 median Spearman `0.99`、minimum Spearman `0.95`、median top-30 `0.95`、minimum top-30 `0.80`）。这说明 treatment 虽然仍能学习可用的 latent predictor，但距离可安全替换 teacher ranking 还有明显差距。

## 5. Logged MSE 与 latent non-inferiority

在同一 held-out logged-action teacher-relative MSE 上：

- control mean relative MSE：`0.060312`；
- treatment mean relative MSE：`0.066163`；
- treatment / control ratio：`1.097006`；
- 冻结上限：`1.25`；
- latent non-inferiority：`PASS`。

也就是说，rank treatment 没有造成不可接受的 latent-MSE 退化，但“没有明显退化”不等于“改善了 planner ranking”。

## 6. Causality（两臂分别验证）

两臂均 `PASS`。对 unchanged prefix length `1, 2, 3, 4`，每个 case 的最大绝对输出差均为 `0.0`，容差为 `1e-6`：

- control：`4/4` prefix cases pass；
- treatment：`4/4` prefix cases pass。

这确认 student 的 action-prefix 因果结构没有被该训练改动破坏。

## 7. Predictor latency

这是 predictor-only、cached native observation latent、normalized action prefix 的测量，不包括 `encode_obs`、CEM、环境或 closed-loop 控制：

| arm | teacher median | student median | speedup | reduction |
|---|---:|---:|---:|---:|
| control | `3513.662 ms` | `16.488 ms` | `213.10x` | `99.5307%` |
| treatment | `3514.305 ms` | `16.523 ms` | `212.69x` | `99.5298%` |

两臂均通过 predictor-level reduction `>=20%`。GPU telemetry 为 A100，19 个 usage samples、14 个 telemetry records，峰值显存约 `26435 MiB`，未见 OOM 或 Xid error。

## 8. Authoritative Dhigh diagnostic boundary

context-density job 的 authoritative `Dhigh` 只作为外部 descriptive reference，不是本实验的 paired control，也不构成 causal comparison。作为诊断：

- treatment − Dhigh median Spearman：`+0.006454`；
- treatment − Dhigh median top-30：`-0.016667`；
- Dhigh 自身 absolute median Spearman：`0.921487`；
- Dhigh 自身 absolute median top-30：`0.683333`。

因此不能把 treatment 与 Dhigh 的差异写成 rank loss 的因果收益，也不能用它绕过本实验的 paired gate。

## 9. Decision levels

| decision | result |
|---|---|
| `paired_effect` | **FAIL** |
| `full_replacement` | **NO-GO** |
| `slate_supported_replacement` | **NO-GO** |

## 10. Predictor-level claim boundary

当前证据允许的表述是：在冻结的 DINO-WM PushT predictor-level cell 中，query-slate planner-aware rank treatment 满足 paired integrity、两臂收敛、latent non-inferiority、causality 和 predictor latency gates，但没有建立相对于 identical slate-trained latent-MSE control 的 ranking improvement；因此不能宣称它是 predictor replacement，也不能宣称 closed-loop CEM 或 environment success 改善。

可以保留的独立工程结论是：该 compact student predictor 在此测量边界内约有 `99.53%` 的 teacher predictor latency reduction。这个 latency claim 不应被解释为包含 `encode_obs`，也不能直接外推到 LeWM/Fast-LeWM 或所有 JEPA-style world models。

## Artifacts

- summary：`artifacts/24494756.pbs101/query_slate_rank_summary.json`
- job log/status：`artifacts/24494756.pbs101/job.log`、`job_status.txt`
- runner/final status：`artifacts/24494756.pbs101/runner_exit_status.txt`、`final_exit_status.txt`
- GPU telemetry：`artifacts/24494756.pbs101/gpu_info.csv`、`gpu_usage.csv`、`gpu_telemetry.jsonl`

远端 checkpoints 与 `plan_targets.pkl` 未取回。
