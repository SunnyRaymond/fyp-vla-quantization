# Stage 1 结果：25476081.pbs101

PBS 作业于 2026-09-23 在 `x1000c0s3b0n1` 完成，状态 `F`、`Exit_status=0`，walltime `00:03:18`。PBS `job_status` 与 runner 状态均为 0，job log 有 40 条 GPU telemetry 采样。完整 summary、call records 和日志保留在远端 `artifacts/25476081.pbs101/`。

**冻结 gate：`FAIL_CLOSED`。** Native teacher 与 routed teacher-only 在 8 个 reset seeds 的每一轮 CEM trace、final plan 和 first action 均精确相同；30 轮、各 teacher schedule、quality 和 latency gate 通过。Validity 失败：所有 8 个 seeds、所有 6 条策略/保真路径中，`pixels`、`goal`、`state`、`goal_state`、`proprio` 均与配对基线精确相同，但 prepared `action` 键仍不相同。作业已在每次 reset 前调用具体 `world.envs.envs[0].action_space.seed(seed)`，因此该差异尚未解释。

冻结计算的描述性固定 observation 结果为：late7 相对 student 的 standardized final teacher cost 中位数 `-2.5027053`，8/8 seeds 更低；late7 与 teacher-only 平均 solve time 分别为 `0.494719 s` 和 `2.177604 s`，按冻结公式降低 `77.28%`。由于配对 observation gate 失败，这些数值不能作为有效的 paired quality/latency 结论，也不能说明 closed-loop PushT 表现。

Stage 2 未提交。后续只进行冻结的 CPU action-reset 诊断，记录各路径 reset 前后的 wrapper/`action_space` 对象、seed 与 `sample` 调用、以及 reset action 的低维值；不会删除 `action` 键或放宽配对 gate。
