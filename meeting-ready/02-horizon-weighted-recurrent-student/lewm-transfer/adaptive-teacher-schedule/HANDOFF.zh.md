# Adaptive teacher schedule handoff

状态：已完成。冻结配置、runner 与 PBS wrapper 未改动；唯一 bounded GPU PBS 作业 `25432397.pbs101` 已正常结束（`F`, exit 0），整体 gate 为 `PASS`。

- 模型路由：用户指定当前主模型为 `gpt-6-sol`，本执行 subagent 指定为 `gpt-6-luna`。此工具环境不向 subagent 暴露运行时 model ID 或 reasoning effort，故无法独立核验实际值。
- 目录：`meeting-ready/02-horizon-weighted-recurrent-student/lewm-transfer/adaptive-teacher-schedule/`。
- main checkpoint：`cem-distribution-distill/artifacts/25239551.pbs101/treatment_step1000.pt`；summary provenance 为 `cem_distribution_distill` / `treatment` / 1000 extra updates。
- fresh evaluation：`valid[592:600]`，selection seed `20300903`，action-prefix seeds `20301105/20301106`，8 episodes、48 paired trajectories。
- 四臂为 student-only、uniform 7-call、late 7-call、teacher-only；每 pair shared innovations，30 rounds × 300 candidates × top-30。
- Quality：uniform 和 late schedule 相对 student-only 的 primary median delta 分别为 `-1.099` 与 `-2.316`，均 8/8 episodes 严格改善。
- Latency：uniform 与 late schedule 相对 teacher-only 的 mean latency 分别减少 `68.49%` 与 `68.54%`，均超过 30% 冻结门槛。详见 `RESULT.zh.md`。
- 本地证据在 `local-status/results/25432397.pbs101/`；模型、dataset、banks 和完整 checkpoint 仍留在 cluster。
- `official CEM` 与 `closed-loop` 为 `NOT_RUN_BY_SCOPE`，不从 fixed-observation adaptive CEM 结果外推 planner deployment 或任务成功。
- 执行期间 Codex primary usage 保持 98% used；没有使用 banked reset credit。
