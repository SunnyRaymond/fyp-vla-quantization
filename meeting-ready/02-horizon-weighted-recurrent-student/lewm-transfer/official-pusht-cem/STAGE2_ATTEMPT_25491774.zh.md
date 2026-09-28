# Stage 2 结果：25491774.pbs101

Stage 2 按冻结 revision 3 完成。此前作业仍可在 PBS history 查询时，权威状态为 `F`、`Exit_status=0`；本次刷新 `qstat -x -f` 已返回 `Unknown Job Id`，`tracejob` 也找不到近三天的记录。保存下来的 `job_status` 与 `final_exit_status.txt` 均记录 PBS/runner exit status 为 0，`stage2_summary.json` 标记 `COMPLETED`。执行身份记录 job `25491774.pbs101`、compute node `x1000c0s3b0n1`；`job.log` 有 85 条五秒 GPU telemetry 采样。PBS 作业没有重提或修改。

**基准边界：本结果不是 upstream official LeWM PushT benchmark。** 本次在 pinned PushT simulator 上按冻结方案用独立随机 reset seeds、`dataset=None`、50-step episode 做 paired evaluation。Pinned upstream `le-wm/config/eval/pusht.yaml` / `eval.py` 的评估为 `num_eval=50`、`seed=42`、`goal_offset_steps=25`、`eval_budget=50`、dataset `pusht_expert_train`，并使用 dataset-driven sampled start steps、state/goal callables 与 preprocess；环境 `max_episode_steps=100`，但 `eval_budget=50` 控制每个 rollout 的有效步数。因此本次与上游的有效 action budget 都是 50 步。除了 expert-dataset starts/goals 与本次 simulator random reset 的协议差别，本次 runner 还使用 `process={}`；upstream `eval.py` 则在完整数据集上为 `action`、`proprio`、`state` 拟合 `StandardScaler`（忽略含 NaN 的行），并把对应 scaler 用于 goal 字段。因此不能把共同的 0/50 结果单因果归结为随机 reset；本次结果只适用于冻结协议，不得称为官方 LeWM benchmark 成绩。

**冻结总 gate：`FAIL`，不是 `INVALID`。** `complete`、`pairing`、`finite` 均通过；`late7_benefit_vs_student` 未通过；`late7_practical_margin_vs_teacher` 通过。

完整性复核得到 200 条 episode 记录：冻结的 50 个 seed `260923001–260923050` 与结果完全一致，四臂各 50 条，无重复 `(seed, arm)`、无缺项、无崩溃行或非有限输出。全部 episode 都运行到规定的 50 步并以 `truncated=true` 结束；没有 simulator success/termination。50 个 seed block 的初始 state/goal、全部 policy-prepared observation keys、solver innovations 与 schedules 配对检查均通过；逐 episode 检查没有不一致。四臂成功数都是 0/50。

主要配对比较 `late_teacher7` vs `student_only`：late-only success `0`，student-only success `0`，discordant pairs `0`，两侧 exact McNemar `p=1.0`，late7 多出的成功数为 `0`。冻结门槛要求至少多 `5/50` 个成功且 `p<0.05`，因此失败。`late_teacher7` 与 `teacher_only` 都是 `0/50`，差值为 `0`，满足冻结的“最多落后 5 个成功”实用 margin；这不是有统计 power 的 formal non-inferiority 结论。所有臂都零成功构成明显 floor effect，`p=1` 不能解释成策略等价。

次要计算量/时间指标（每臂 50 episodes 的冻结汇总均值）：student-only 每 episode 0 次 teacher call、planner `0.943 s`；late7 14 次、`0.998 s`；uniform7 14 次、`1.012 s`；teacher-only 60 次、`1.835 s`。late7 相对 teacher-only 少 46 次 teacher call/episode，平均 planner walltime 低约 45.6%；相对 student-only 则平均 planner walltime 高约 5.8%。平均 episode walltime 分别为 `1.235 / 1.195 / 1.206 / 2.029 s`（按 student-only / late7 / uniform7 / teacher-only）。这些是次要描述指标，不能抵消主成功率 gate 失败。

本次冻结协议使用 pinned PushT simulator 的 success 判据；没有任何 episode 达成。复核的 step metrics 中没有 episode 曾达到 `position_error < 20`，虽有 26/200 episode 曾短暂达到 `angle_error_rad < pi/9`，但从未满足完整 success 条件。尤其 teacher-only 同样 `0/50`，提示所有臂共享明显 floor effect；当前结果仅支持“本次冻结 random-reset、50-step 协议下未观察到 late7 成功率收益”，不足以判定策略等价，也无法单凭这些数据把失败归因给 student 或 CEM 路由。

下一步先决定后续问题要复现 upstream official LeWM benchmark，还是继续研究本次 random-reset、50-step 协议；再按所选 protocol 检查 unmodified teacher-only 的 action/goal 处理与逐步轨迹。只有解释清楚 teacher-only 也为 0/50 的 floor 后，再评估是否设计新的独立冻结实验。不要延长本次 50-seed 结果，不把它称为官方 LeWM benchmark，也不要据此宣称成功率非劣。

证据：`artifacts/25491774.pbs101/stage2_summary.json`、`stage2_episodes.jsonl`、`stage2_progress.json`、`job_status`、`final_exit_status.txt`、`job.log`、`gpu_info.csv`、`execution_identity.json`。
