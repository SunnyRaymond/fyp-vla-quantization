# DINO-WM PushT transfer result

## Decision

`23986275.pbs101` 完成了冻结的 two-observation paired benchmark，但 progression gate 为 **FAIL**。不重跑、不调阈值，也不进入 LeWorldModel PushT。

## Frozen-gate outcome

- decision gate: **FAIL**；30/30 paired calls 均非 bitwise exact，且均超过 `1e-5`。
- full `CEMPlanner.plan()` paired median latency reduction: `4.8150%`，低于冻结的 `10%`。
- maximum cached/baseline peak-memory ratio: `1.0000157`，通过 `<=1.10`。
- PBS walltime: `01:48:58`；最终 exit status 为 `1`，原因是 verifier 返回 FAIL。

因此，这个 PushT result 不能支持 observation-prefix reuse 可安全迁移、也不能作为 full-framework 已验证组件。action-prefix predictor Stage A 可以独立继续，但不能把两者组合写成已验证结论。

## Instrumentation note

verifier 还暴露了一个不影响上述 no-go 的 schema bug：runner 把两个 frozen observations 分成两个独立 planner calls，所以每条 record 的 `n_evals=1`；verifier 却用 observation 总数 `2` 检查每轮 trajectory trace，产生了大量“must contain 2 trajectory traces”错误。即使忽略这些结构错误，30/30 decision comparisons 与 `4.8150% < 10%` 仍分别否定 decision 和 latency gates，因此没有为修 verifier 而重跑 A100 的必要。

## Local evidence

- `artifacts/23986275.pbs101/verifier.json`
- `artifacts/23986275.pbs101/job.log`
- `artifacts/23986275.pbs101/observation_manifest.json`
- `artifacts/23986275.pbs101/qstat_final.txt`
- `artifacts/23986275.pbs101/gpu.txt`

这次旧 job 只留下启动时 GPU identity，没有周期 utilization/VRAM log；以后所有新 GPU PBS jobs 必须持续采样。
