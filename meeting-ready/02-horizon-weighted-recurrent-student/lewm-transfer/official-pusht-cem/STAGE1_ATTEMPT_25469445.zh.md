# Stage 1 首次完整尝试：25469445.pbs101

该作业于 2026-09-23 在 ASPIRE2A gdev 完成，PBS `Exit_status=0`，墙钟时间 `00:03:21`。结果保存在远端 `artifacts/25469445.pbs101/`；summary 为 799,849 bytes，含 8 个 seed block 和所有 gate 输出。`job.log` 保存了 41 条 GPU 利用率/显存采样。

冻结检查结果：native teacher 与 routed teacher-only 的 8/8 solver traces、final plan 和 first action 均 bitwise 相等；teacher-call schedule、quality 和 planner latency gates 通过。描述性质量值为 late7 相对 student-only 的标准化 teacher cost 中位数 `-2.5027`，8/8 seed late7 cost 更低；规划 solve 均值为 late7 `0.4395 s`、teacher-only `2.3215 s`，相对降低 `81.07%`。这些数值因 pairing validity 失败，不能作为有效的成对质量/延迟结论。

整体状态为 `FAIL_CLOSED`：8/8 block 的 policy-prepared observation exact-equality 检查失败。summary 中每个 block 保存的 raw simulator `state` 和 `goal_state` 跨路径完全相同，但旧 runner 未保存逐 key equality，所以不能从此尝试确认具体哪一个 prepared tensor 不同。

后续只读源码核对发现，pinned `EverythingToInfoWrapper.reset` 在 simulator reset 后把 `reward=np.nan` 与 `action=action_space.sample()` 写进 info；旧 comparator 不比较 reward，但比较 action。Reset action 是单环境 action space 生成的占位值，且 `World.reset(seed=...)` 不会为该 action space 单独设 seed。因旧 summary 未保存逐 key equality，无法仅从这次输出证明该 placeholder 是唯一不一致字段。Pinned `CEMSolver.prepare_init_action` 对非 `Actionable` 的 JEPA zero-pad，官方 `JEPA.get_cost` 随后移除/覆盖 reset action，因此该占位值不影响候选评分；不过为了保持全部 prepared keys 精确配对，新版本在每次环境 reset 前用同一 simulator seed 显式 seed action space，并输出逐 key equality 标志。该修正不改变实验 arm、seeds、schedule、estimand 或 gates。

首次结果与所用 freeze snapshot 保留在 [FREEZE.stage1-attempt-25469445.json](FREEZE.stage1-attempt-25469445.json) 和远端 `artifacts/25469445.pbs101/`。Stage 2 没有提交；后续 Stage 1 重跑只有在全部冻结 gate 通过后才可继续。
