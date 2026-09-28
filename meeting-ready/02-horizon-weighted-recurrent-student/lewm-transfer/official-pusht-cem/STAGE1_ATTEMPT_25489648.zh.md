# Stage 1 结果：25489648.pbs101

作业于 2026-09-23 在 compute node `x1000c0s0b0n1` 完成，PBS 状态 `F`、`Exit_status=0`，walltime `00:03:23`；runner 与 PBS 状态均为 0。GPU telemetry 在 `job.log` 中每 5 秒记录，共 41 条。完整 summary 与 48 条 call records 保存在 `artifacts/25489648.pbs101/`。

**Stage 1 总 gate：`PASS`。** 五项冻结检查 `native_equivalence`、`schedule`、`quality`、`latency`、`validity` 全部通过。8 个 reset seeds 的六条路径中，prepared `pixels`、`goal`、`state`、`goal_state`、`proprio`、`action` 全部通过成对检查；action 按 freeze revision 3 比较 shape/dtype、NaN 与正负 infinity masks，并精确比较所有 finite 值。Native teacher 与 routed teacher-only 在 8/8 seeds、全部 30 轮的 candidates、costs、elite indices/candidates、mean/variance updates 以及 final plan/first action 均逐字段 bitwise 相等。

late teacher 的调用恰为 rounds 24–30，每 solve 7 次；`uniform_teacher7` 恰为 rounds 4、8、12、16、20、24、28；student-only 为 0 次，teacher-only 为 30 次。质量上，late7 减 student-only 的标准化 final teacher cost 中位数为 `-2.5027053`，8/8 seeds 都严格改善，超过冻结门槛。

冻结 latency 口径下，late7 平均完整 `CEMSolver.solve` 为 `0.495418 s`，teacher-only 为 `2.175061 s`，平均降低 `77.22%`，超过 30% gate。teacher-only 第一个 solve 为 `11.074841 s`，它恰好是 seed `260922001` 的随机化执行首臂；其余七次为 `0.902216–0.906247 s`。这一首调用大幅拉高 teacher-only 均值，原因不能从本次记录单独确认，可能包含首调用开销；冻结估计仍保留该值。仅作描述性敏感性计算，去掉这一个首调用后 teacher-only 均值为 `0.903664 s`，late7 仍快 `45.18%`，高于冻结阈值。该敏感性不改变预注册 gate 或主要结果。

源码修正与这次通过的 action-pair gate相符：reset wrapper 的 finite `action_space.sample()` 会立即被覆盖成 all-NaN action placeholder，policy preprocessing 保留该 NaN mask。先前 `25469445.pbs101` 和 `25476081.pbs101` 仍原样保留为 `FAIL_CLOSED`，不追溯修改它们的结果或指标；本次是按 revision 3 语义比较规则重新运行后首次通过 Stage 1。

Stage 2 按冻结方案现在具备运行条件：50 个 fresh paired PushT simulator seeds，主比较 `student_only` / `late_teacher7` / `teacher_only`，`uniform_teacher7` 为次要同 teacher-budget 对照；每个 seed 四臂共享初始 simulator/action-space seed 与配对 CEM innovations，逐 episode flush 结果，最大 2 小时 GPU PBS allocation。Stage 2 尚未提交。
