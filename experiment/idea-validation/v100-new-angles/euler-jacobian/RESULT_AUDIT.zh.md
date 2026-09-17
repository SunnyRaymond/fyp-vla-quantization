# Euler Jacobian 结果解释审查

审查对象：`RESULT.zh.md`、小型独立复核 `job-64836/verification.json` 和冻结 `PROTOCOL.zh.md`。未读取 `raw.npz`，未运行数值或集群任务。

## 结论

结果的 `mechanism_no_go / STOP` 与冻结 gate 一致，没有需要改写的科学结论。六个 FP state 均 eligible，六个 Q state 均未触发两类 flag；`flagged=0`，低于预注册的 `3` 个门槛。`verification.json` 的 receipt、source/checkpoint、V100 allocation、112-layer quantization、restore、RNG、AD、FD、closure 和 raw-completion checks 全部通过。

1. `sigma_min`、condition ratio 和 determinant sign 是同一 32D first-token local Euler map 的诊断量；结果没有把普通 velocity drift 当成 topology failure，解释正确。
2. 文档明确 32D 含 25 个 padding coordinates，且 7D physical slice 不闭合。因此表中的正 determinant 和 singular values 没有被错误改称 7D physical topology 结论。
3. `det>0` 仅排除本协议定义的 robust orientation-reversal gate，不等于证明 global invertibility；结果未作 global folding 推断。
4. `sigma_min` 远高于 `.005`，ratio 远高于对应严重病态阈值；没有事后更换 timestep、noise 或 tolerance。
5. FD、future-token closure、tail probe、FP grad/no-grad no-op 与 K5 path reconstruction 属于实现/数值合同；结果把它们作为复核证据，没有把它们包装成 task performance。
6. 结果明确不声称 physical action 成功率、global K10 行为、robot closed-loop success 或 native W4 deployment/performance；`FP32 fake quantization` 边界也已写清。
7. 复用固定 observations、共享新 noise，以及只分析共同 FP path 的 `t=.5` 局部 map 都已在协议和结果中披露，结论范围没有被扩大。
8. `FP/Q velocity MSE` 只是附加观测；在“仅 velocity error 不支持假设”的协议边界下，不影响 no-go 判定。
9. 结果中链接使用实际的 `job-64835/` 与 `job-64836/` 目录；用户所述 `artifacts/64836/` 本地不存在，但证据文件路径和链接一致，不构成结论问题。
10. `v100-new-angles/INDEX.zh.md` 已列 Euler；`ACTIVE_STATE.zh.md` 已更新为 “Twelve completed empirical screens STOP”，因此 Euler 已计入第 12 个已完成 screen。

未发现 blocking issue。应保持现有 `STOP`，不增加 task、noise、timestep、padding/physical 重解释或 task-success 验证。
