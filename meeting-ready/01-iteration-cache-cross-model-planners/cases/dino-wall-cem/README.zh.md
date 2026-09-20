# DINO-WM Wall：CEM factorized observation-cache screen

这是 DINO-WM Wall 上的 CEM case。它测试把固定 observation prefix 做成
`factorized_cache`，同时保留 CEM 的 candidate rollout、ranking、elite update
和 chained rounds。运行入口是 `approx_screen.pbs`；共享实现位于
`../../common/cache_core.py`。

## 继续运行

先将整个 bundle 放到远端，并准备官方 DINO-WM Wall reproduction root。然后在
compute allocation 内提交：

```bash
export DINO_CACHE_BUNDLE_ROOT=/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/01-iteration-cache-cross-model-planners
qsub cases/dino-wall-cem/approx_screen.pbs
```

`DINO_WM_ROOT` 可指向已有的 Wall reproduction root；PBS 只读取已准备的
runtime、dataset 和 checkpoint，不在 login node 下载或解压。

## 证据

- authoritative job：`23435258.pbs101`
- decision：10/10 Wall observations 通过 exact chained-CEM decision screen
- latency：paired median reduction `14.3485%`
- memory：最大 factorized/baseline peak-memory ratio `1.00007`
- overall decision：**FAIL**，因为冻结 system latency gate 是 `20%`
- closed-loop shadow：未授权

这个 case 说明 decision trace 在这组 screen 中保持一致，但当前 approximate
factorized implementation 尚不足以作为已通过的 system-level replacement。不要把
它写成 20% latency gate 通过，也不要把它写成 closed-loop 成功。

轻量结果在 `verifier.json`、`summary.json` 和 `system_summary.json`；冻结协议
在 `APPROX_FREEZE.json`。
