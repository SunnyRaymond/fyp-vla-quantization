# DINO-WM Wall：GD iteration cache portability

这个 case 把官方 `GDPlanner.plan` 内固定的 start/goal observation prefix
移到一次 planner call 开始时编码；action encoder、predictor、objective、
`backward` 和 action-gradient path 均保持官方实现。

## 继续运行

```bash
export DINO_CACHE_BUNDLE_ROOT=/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/01-iteration-cache-cross-model-planners
qsub cases/dino-wall-gd/gd_portability.pbs
```

需要已有的 DINO-WM Wall reproduction root、runtime、dataset 和 checkpoint；PBS
不会在 login node 做重 I/O。`common/cache_core.py` 是本 case 的共享 helper。

## 证据

- authoritative job：`23926079.pbs101`
- decision：PASS，frozen comparison 为 bitwise exact
- full planner latency reduction：`10.7966%`
- max cached/baseline peak-memory ratio：`1.00054`
- overall progression gate：**PASS**
- scope：2 个 Wall observations，direct GD planner-call benchmark；不包含环境交互

这是目前 DINO-WM 上最干净的 cache portability PASS，但不能外推到 PushT、LeWM
或 closed-loop task success。冻结设置与 verifier 分别见
`GD_PORTABILITY_FREEZE.json` 和 `verify_gd_portability.py`。
