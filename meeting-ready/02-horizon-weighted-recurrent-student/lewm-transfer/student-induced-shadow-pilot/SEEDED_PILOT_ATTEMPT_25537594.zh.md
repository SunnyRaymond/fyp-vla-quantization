# Seeded student-induced shadow pilot：t25 边界有效性停止

`25537594.pbs101` 在 gdev compute node `x1000c0s2b0n1` 运行，PBS `Exit_status=1`、墙钟 `00:04:27`。受控 seeded protocol 的前四个 shadow-off/on 精确配对（episode `9136`、`12704`、`4340`、`5233`）全部通过。随后八个 collection episode 记录了 `t25_status=reached`、实际 50 步、success=false；第九个 episode 触发 `RuntimeError: 25 transitions were committed but the expected t25 CEM solve is missing`，作业按有效性规则停机。

没有生成 `paired_gate.json`、`pilot_summary.json` 或正式 16-episode 聚合，不能将前八个日志片段当作 pilot 结果。`job.log` 保存 54 条约 5 秒间隔的 GPU 利用率/显存采样。

已从 pinned `World._run_iter` 核实：dataset evaluation 在 `envs.step()` 后处理 `terminateds/truncateds`，如果第 25 步结束了唯一环境，会在下一次 `_get_actions()` 之前退出，因此合法地没有 t25 solve。现有作业未记录该步的 done 信号，不能追认第九个 episode 的具体终止原因。后续 runner 仅在第 25 步有明确 done 信号时分类为非 matched 的 `terminal_at_t25_before_replan`；否则继续 fail-closed。提交时原 freeze 已另存于 `submitted-freezes/25537594.pbs101/`；新 seeded freeze 只扩充边界状态，任务、regret 指标和 GO 门槛不变。
