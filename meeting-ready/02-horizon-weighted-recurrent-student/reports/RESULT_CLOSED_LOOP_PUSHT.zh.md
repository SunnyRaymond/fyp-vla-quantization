# DINO-WM PushT：Horizon-weighted Recurrent Student 闭环结果

## 结论先行

已训练完成的 `horizon_weighted_recurrent_step1500.pt` 已经接入 official DINO-WM PushT 的 CEM + MPC + environment closed loop，并与 official DINO-WM teacher 在同一批 targets、seeds 和 planner settings 下做了 paired comparison。

冻结的 8-case exploratory pilot 结果是：

| arm | success | success rate |
|---|---:|---:|
| official DINO-WM teacher | `8/8` | `100%` |
| horizon-weighted recurrent student | `2/8` | `25%` |

paired success table：

| outcome | cases |
|---|---:|
| both success | `2` |
| teacher only | `6` |
| student only | `0` |
| both fail | `0` |

预先冻结的 progression gate 要求 student 至少 `7/8` success，且相对 teacher 最多落后 1 例。实际为 `2/8`，落后 6 例，因此 **FAIL**。按 gate 停止扩张，不继续自动运行 official 50-case 规模。

这把此前 predictor-level 的证据边界补齐了：student 的 held-out ranking 虽然较高、predictor-only forward 也很快，但不足以在闭环 CEM 中保留 teacher 的 action selection 与 task success。

## 1. 实验身份

- valid pilot job：`24544733.pbs101`；exit status `0`；A100 40 GB；walltime `00:18:06`；
- engineering smoke：`24542653.pbs101`，结果 `teacher 2/2`、`student 1/2`，只用于验证 runner；没有与 pilot 相加；
- student checkpoint：job `24510395.pbs101` 的 `horizon_weighted_recurrent_step1500.pt`；
- teacher checkpoint：official DINO-WM PushT `model_latest.pth`；
- pilot seeds：`[1, 100, 199, 298, 397, 496, 595, 694]`；
- `goal_source=dset`，`goal_H=5`；
- CEM：`300` candidates、top-`30`、`30` optimization steps、horizon `5`；
- MPC：每轮执行 `5` 个 model actions，最多 `12` rounds，即失败 case 最多 `300` environment steps；
- objective：official terminal latent objective，`alpha=1`、`base=2`、`mode=last`。

这不是 official 50-case reproduction：`n_evals` 从 `50` 收缩到 `8`，并把 official unbounded `max_iter=null` 改成 bounded `12` rounds。两臂使用完全相同的 bounded protocol。Teacher 的 8 例全部在首个执行边界成功，因此这个 cap 没有限制 teacher；student 的 6 个失败只表示“在 12-round/300-step budget 内失败”，不排除更长 budget 下可能恢复。

## 2. Student 如何接入 official planner

闭环 adapter 没有重新训练 student，也没有修改 official DINO-WM source。

每个 MPC observation 都经过 official teacher observation encoder：

```text
official transformed observation
  -> official DINO-WM encode_obs
  -> cached current native visual/proprio latent
  -> horizon-weighted recurrent student + packed [B, 5, 10] actions
  -> five predicted native latent states
  -> prepend current latent
  -> official objective, CEM, MPC and PushT evaluator
```

Teacher arm 仍调用 official `VWorldModel.rollout()`。两臂只关闭 decoder-based diagnostic rendering，避免视频 I/O；这不改变 observation encoding、latent rollout、objective、CEM、MPC、environment actions 或 success predicate。

## 3. Per-case 结果

| case | eval seed | teacher | student | teacher env steps | student env steps | teacher state distance | student state distance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | success | fail | 25 | 300 | 36.81 | 90.47 |
| 1 | 100 | success | fail | 25 | 300 | 26.03 | 34.48 |
| 2 | 199 | success | fail | 25 | 300 | 94.87 | 126.27 |
| 3 | 298 | success | success | 25 | 25 | 25.39 | 18.23 |
| 4 | 397 | success | fail | 25 | 300 | 44.44 | 30.56 |
| 5 | 496 | success | fail | 25 | 300 | 38.92 | 46.13 |
| 6 | 595 | success | success | 25 | 25 | 40.15 | 22.12 |
| 7 | 694 | success | fail | 25 | 300 | 41.69 | 23.37 |

`state_distance` 是 full state-vector diagnostic，不是 PushT success threshold 本身。Official success 同时检查 position difference 和 wrapped angle difference；因此不能根据上表的单个 `state_distance` 数字重判 success。

## 4. Runtime 与 memory 的正确解释

| metric | teacher | student |
|---|---:|---:|
| planner walltime | `271.46 s` | `784.98 s` |
| PyTorch peak allocated | `18,432.94 MiB` | `1,843.92 MiB` |

Student 的总 planner walltime 更长，不代表 student predictor forward 更慢：teacher 的 8 例全部一轮成功，而 student 有 6 例跑满 12 rounds，实际 planner work 完全不同。这个闭环结果不能用于固定-work latency speedup claim；predictor-only 的 `~10.7 ms` 仍是另一条 measurement boundary。

同理，student arm 的 PyTorch peak allocated 较低只反映本进程内重置后的 arm-local allocation。整个 job 同时保留 teacher encoder、student 和 CUDA cache；30 秒 telemetry 的最高 `memory.used` 为 `24,831 MiB`。不能把这里的数字写成 native deployment memory reduction。

GPU telemetry 共 `36` 个样本，mean utilization `75.19%`、maximum `100%`；采样写入 `gpu_usage.csv`，同时进入 `job.log`。

## 5. 结论与后续边界

当前最窄、证据支持的结论是：

> Under an eight-case, 12-round paired DINO-WM PushT protocol, the trained horizon-weighted recurrent student succeeds on 2/8 cases versus 8/8 for the official teacher. The frozen progression gate fails, so the recipe should not be expanded to a 50-case run without a new mechanism or training change.

不能声称：

- official DINO-WM 50-case baseline 已复现；
- student 的 population-level success rate 精确为 `25%`；
- student predictor 本身导致 2.89× latency regression；
- 更长的 unbounded MPC budget 一定无法恢复失败 cases；
- 该结果可外推到 LeWM、Fast-LeWM 或其他 environments。

若以后继续，应把它作为新的 recipe，而不是偷偷扩展本次失败配置。新实验需要明确改变 training objective、planner-aware supervision 或 candidate sensitivity，并重新冻结 checkpoint、seeds、budget 和 progression gate。

## 6. 本地证据

- valid pilot summary：`artifacts/24544733.pbs101/closed_loop_summary.json`；
- per-arm results：同目录下 `official_dino_wm_teacher.json` 与 `horizon_weighted_recurrent_student.json`；
- status：`job_status.txt`、`final_exit_status.txt`、`runner_exit_status.txt`；
- telemetry：`gpu_info.csv`、`gpu_usage.csv`、`job.log`；
- runner：`src/run_dino_pusht_closed_loop.py`；
- frozen pilot：`config/CLOSED_LOOP_PILOT_FREEZE.json`。
