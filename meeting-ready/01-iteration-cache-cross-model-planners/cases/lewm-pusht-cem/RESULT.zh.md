# LeWM PushT iteration cache：验证结果

## 研究问题

在官方 LeWorldModel PushT 的 CEM planner 中，能否跨 CEM iterations 复用不随
candidate action 改变的 initial-observation embedding 与 goal embedding，同时
保持 action-conditioned rollout、candidate ranking 和最终 action 完全不变？

## 实现边界

`iteration_cache.py` 提供 `IterationCacheModel`。第一次
`get_cost(info_dict, action_candidates)` 时，它只执行一次 initial/goal encode，
并把 detached embedding 放入当前 solve-call 的 cache。之后仍执行官方的
candidate-axis expansion、action encoder、predictor、criterion 与 CEM 更新。
cache 以当前 `info_dict` identity 为 scope；不会跨 solve-call 泄漏状态。

## 冻结协议

- official LeWM commit：`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`
- stable-worldmodel CEM commit：`10c26dbd5677083fa31dba69eb738b973845e9a4`
- horizon `5`，`num_samples=300`，`n_steps=30`，`topk=30`
- batch size `1`，seed `42`
- `pusht_obs_00`、`pusht_obs_01`
- 每个 paired unit：5 warmups + 10 technical repeats，path order 交错
- full timing 包含 solve wrapper、preprocessing、cache setup/重复 encode、完整
  CEM loop、top-k/update 和 solver overhead；CUDA 前后同步
- exactness trace 包含 final actions、first actions、costs、每轮 top-k indices/
  values、mean 和 variance

完整协议见 `LEWM_PUSHT_ITERATION_CACHE_FREEZE.json`。

## 证据

GPU benchmark：`24382364.pbs101`；verifier recovery：`24389764.pbs101`。
本 bundle 保留了可复核的轻量输出：

- `artifacts/24382364.pbs101/verifier.json`
- `artifacts/24382364.pbs101/verifier_recovery_summary.json`
- `artifacts/24382364.pbs101/qstat_24382364_24389764.txt`

原始 GPU job 已完成 benchmark，但第一次 verifier 直接用
`allow_nan=False` 打印包含 `inf` 的失败诊断，因此 PBS exit status 为 `1`。
这不是模型 benchmark 重新失败；CPU recovery job 针对同一
`system_summary.json` 重跑修复后的 verifier，结果为：

```json
{
  "status": "PASS",
  "decision_gate": {"status": "PASS", "mode": "bitwise_exact"},
  "full_planner_reduction": 0.2940910375,
  "plan_section_reduction": 0.2943577700,
  "cem_loop_reduction": 0.2975388903,
  "peak_memory_max_ratio": 0.9708293208,
  "comparison_count": 30,
  "error_count": 0
}
```

## 结论

在这个 frozen benchmark 上，iteration cache 通过 exactness gate，并超过 10%
full-planner reduction gate，同时没有增加 peak memory。它证明了一个可复用的
planner-call cache 机制；它不证明跨任务、跨模型或 closed-loop success 的普适性。
