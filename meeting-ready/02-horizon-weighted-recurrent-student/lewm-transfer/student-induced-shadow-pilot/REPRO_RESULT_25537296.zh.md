# Control-control 基线复现性结果

`25537296.pbs101` 正常结束（PBS `Exit_status=0`，墙钟 `00:03:55`）。仅对冻结的 episode `12704`、start step `3` 做两次 shadow-off student-only 运行，每次新建 World/solver。结果为 `COMPLETED_DIVERGENCE_OR_MISSING_API`，即基线自身没有满足精确复现性要求。

该 HDF5 dataset 没有 `seed` 列，World dataset reset 两次都收到 `None`。两次 t0 的 prepared `pixels/goal/proprio/state/goal_state/action` 六项完全相同（224×224×3 图像的最大差值 0），solver RNG 起止状态相同，第一轮 CEM candidates/costs/mean/var 完全相同，首个 committed action 也完全相同。

然而 simulator 内部 block 物理状态在规划前已不同：block position 从 `[341.3865026562, 246.8645374151]` 对比 `[341.3491516113, 246.2164154053]`，angle `0.8313802392` 对比 `0.8435469270`，block velocity 亦不同；agent 状态及 shape/scale 相同。第一个环境 step 后的 state 最大绝对差 `0.6265411377`。因此先前 shadow-off/on 的轨迹差不能归因于 teacher shadow；隐藏 reset 状态差本身就足以造成分叉。

本次 `job.log` 保存 47 条约 5 秒间隔的 A100 利用率/显存采样。下一步仅验证受控固定 reset seed 对同一 episode 的复现性，不能直接修改原 pilot 判据或宣称累计误差机制成立。
