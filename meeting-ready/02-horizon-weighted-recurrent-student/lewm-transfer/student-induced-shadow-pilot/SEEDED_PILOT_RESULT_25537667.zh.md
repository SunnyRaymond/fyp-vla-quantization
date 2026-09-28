# Student-induced shadow pilot：受控 seeded 协议下 NO-GO

`25537667.pbs101` 正常结束（PBS `Exit_status=0`）。四个预设 shadow-off/on 配对全部通过精确 non-interference gate；16 个 collection episode 全部评估，reserved holdout 未用于评估、打分或判决。沿用官方全数据 scaler 会读取训练数据列，包括 holdout 所在行，故这里的 holdout 仅指任务层面的保留。

其中 15/16 个 episode 到达 t25 replan，超过预设的最少 12 个 matched episode。Episode `15946` 在第 25 个真实 transition 后发出 `terminated=true, truncated=false`，当时任务成功；World 在下一次 CEM solve 前停止。它被预先修订的边界规则归为 `terminal_at_t25_before_replan`，未替换，也未计入配对 regret。全 16 个 episode 的 success 为 1/16；这是**固定 reset seed=42 的受控协议**，不能直接与未固定 reset 的 official 50-task 成功率比较。

主指标是同 episode 的 round-30 standardized teacher-elite regret `t25−t0`，越大表示 student 选择的 elite 相对 teacher 变差。15 个 matched episode 的中位增量为 `+0.0860738`，达到预设 `≥+0.05`；但只有 `9/15=0.60` 个 episode 增加，未达到预设的 `≥0.75`。冻结决策是 **NO-GO**，不进入该 blanket student-induced state aggregation 训练配方。原始 episode 增量范围为 `−1.01936` 到 `+0.97589`，显示明显异质性。

次级诊断：round-10 的中位增量 `−0.41766`（5/15 增加），round-20 为 `−0.13842`（7/15 增加），round-30 才转为 `+0.08607`（9/15 增加）。Round-30 teacher top30 在 student top120 中的 recall 中位数 t0=`0.2333`、t25=`0.2667`，没有与 regret 同向恶化。t0 与 t25 的 planner context、原生 CEM candidate draws 和 RNG 位置均不同；这些差值描述联合 on-policy shift，不能单独归因于 observation-state 误差累积。

本 pilot 没有训练新 student，也没有测试部署加速或 official closed-loop 提升。结果不支持把“后续状态普遍更差”作为下一轮训练的宽泛假设；值得转向 late-CEM、episode-dependent 的错误来源，并在新的独立 freeze 中检验可观测的触发信号，而不是重复既有失败的固定起点 candidate-tail 蒸馏或无条件 teacher screening。

证据：`results/25537667.pbs101/seeded_pilot_summary.json`（PBS summary）和 `paired_gate.json`（四对精确 gate）；job log 记录 51 条 GPU 利用率/显存采样。
