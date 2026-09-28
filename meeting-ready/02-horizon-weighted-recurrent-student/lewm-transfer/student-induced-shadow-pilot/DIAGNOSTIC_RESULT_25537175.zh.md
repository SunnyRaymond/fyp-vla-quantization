# 两个失败 episode 的非干扰性诊断

`25537175.pbs101` 正常结束（PBS `Exit_status=0`，墙钟 `00:03:50`），仅运行冻结的 episode `9136` 和 `12704` 的 shadow-off/on 配对，没有评估其余 collection 或 holdout 任务。输出 `diagnostic_two_pair.json` 标记 `COMPLETED_MISMATCH`；这是**有效性诊断**，不是 16-episode pilot 结果。

Episode `9136` 在本次复跑中通过原精确 gate：50 步的提交动作、mask、后状态均完全一致，两次 success 均为 false。它在先前 `25537036` 中失败，说明 gate 结果本身会随独立复跑变化。

Episode `12704` 在本次复跑中仍失败：mask 完全一致；提交动作首差在第 26 步，最大绝对差 `0.4262429476`；后状态首差已在第 1 步，最大绝对差 `159.5560150`。其 t0 CEM round arrays 也不同，而 solver 独立 generator 的起止状态相同；t25 solver 输出动作与 round arrays 亦不同。两次轨迹均完成 50 步且 success 均为 false。首个状态差早于首个提交动作差，不能把状态偏差简单归因于这条已记录的动作差。初始记录目前只含 `state/goal_state`，尚未证明渲染 pixels 或 simulator 内部 reset 状态一致。

`job.log` 有 48 条 5 秒间隔的 A100 利用率/显存采样，峰值约 23% / 989 MiB。下一步先验证同模式重复运行与完整 reset 观测是否稳定，再谈 teacher shadow 的非干扰性；不得放宽原 gate 或解释 pilot regret。
