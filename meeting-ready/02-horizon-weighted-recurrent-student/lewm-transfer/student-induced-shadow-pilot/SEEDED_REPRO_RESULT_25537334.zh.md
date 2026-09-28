# 固定 reset seed 的 control-control 复现性结果

`25537334.pbs101` 正常结束（PBS `Exit_status=0`，墙钟 `00:04:07`）。冻结 episode `12704` 两次均为 shadow-off student-only，独立创建 World/solver；dataset 没有 `seed` 列，请求的 reset seed 为 `None`，诊断钩子在唯一一次 World.reset 调用中传入实际 seed `42`。

结果 `COMPLETED_EXACT_SEEDED_REPRODUCIBILITY`：两次 prepared `pixels/goal/proprio/state/goal_state/action`、simulator 的 agent/block 位置速度与形状、solver RNG、首轮 CEM candidates/costs/mean/var、首个 committed action 及首个 post-step state 均完全相同，最大差值为 0。两次都运行 50 步且 success=false。`job.log` 保存 49 条约 5 秒间隔的 A100 利用率/显存采样。

这证明**受控 seeded protocol variant** 在该单个 episode 上可复现，不等于原官方 dataset-start reset 语义，也不证明 teacher shadow 非干扰或 student 规划质量。下一步需单独冻结 seeded shadow pilot，并按原任务集和阈值先通过四对 shadow-off/on 精确 gate。
