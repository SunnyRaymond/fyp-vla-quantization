# DINO-WM PushT CEM-DAgger：结果与边界

## 结论

正式作业 `24928207.pbs101` 正常完成（runner/final exit status 均为 `0`）。
`treatment` 通过冻结的 Stage 1 predictor gate，但 matched `replay_control` 的
top-30 median 为 `0.783333`，低于阈值 `0.80`，因此 Stage 2 fixed-observation
CEM 按协议跳过。

paired block 结果也不支持 CEM-DAgger 优于同预算 replay：treatment 相对 control
的 Spearman 为 `7/16` 改善、`9/16` 回退，median delta `-0.000253`；top-30 overlap
为 `4/16` 改善、`6/16` 持平、`6/16` 回退，mean delta `-0.004167`。因此这不是
值得通过放宽阈值继续推进的边缘成功，而是本 frozen recipe 的机制层面 **NO-GO**。

## 实验身份与执行完整性

- backend：official DINO-WM PushT；A100-SXM4-40GB；
- collector：8 个 train-split episodes，iterations `1/5/10/20/30`，每个 checkpoint
  取 student top-120，共 `4800` teacher-labelled rows；
- treatment：每 update `16` CEM-DAgger rows + `16` exact old replay rows；
- control：每 update `32` exact old replay rows；
- 两臂共享 warm start、optimizer、500 updates、context schedule 与 held-out bank；
- outputs finite，causality max absolute delta `0`，无 silent fallback；
- GPU telemetry 正常写入，训练阶段可见 100% utilization，显存最高约 `29.0 GiB`；
- 未运行 environment interaction、student closed-loop 或 historical teacher-full 重算。

## Stage 1 absolute gate

| arm | Spearman median / min | top-30 median / min | causality | gate |
|---|---:|---:|---:|---:|
| CEM-DAgger treatment | `0.971651 / 0.890883` | `0.800000 / 0.566667` | `0` | **PASS** |
| matched replay control | `0.970651 / 0.904681` | `0.783333 / 0.600000` | `0` | **FAIL** |

control 只在 top-30 median 上低于阈值，但这不能被解释为 treatment 的稳定优势：
两臂的 paired deltas 没有同向支持，且 treatment 的平均 Spearman 与 top-30 均略低。

## Stage 2 与决策

冻结协议要求两臂都通过 Stage 1 才运行 Stage 2。由于 control 失败，Stage 2 状态为
`SKIPPED: one or both stage1 arm gates failed`；没有产生 case-level CEM artifacts。

决策：

1. 不降低 `0.80` top-30 threshold，不补跑同一 CEM-DAgger recipe；
2. 不把 treatment 单臂 PASS 写成 planner improvement；
3. 不运行 official planner deployment 或新的 closed-loop；
4. 将后续算力转向已经独立冻结并完成 preflight 的 LeWM tail-robust EMA 实验。

## Claim boundary

本结果只支持：在固定 train-split CEM proposal distribution 和 held-out predictor
protocol 下，CEM-DAgger treatment 没有显示出相对 matched replay 的稳定优势。

本结果不支持新的 closed-loop success rate、student environment rollout、official
planner deployment、LeWM/Fast-LeWM transfer 或 native low-bit deployment claim。
