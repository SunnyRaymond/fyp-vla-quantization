# DINO-WM PushT：CEM iteration cache

这个 case 在官方 DINO-WM PushT CEM 上跨 iteration 缓存 candidate-batched
observation encoding；action encoding、predictor rollout、objective、ranking、
elite update 和 RNG schedule 都保持官方路径。

## 继续运行

```bash
export DINO_CACHE_BUNDLE_ROOT=/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/01-iteration-cache-cross-model-planners
qsub cases/dino-pusht-cem/pusht_iteration_cache.pbs
```

官方 PushT assets 和 pinned dependency overlay 必须先通过本目录的
`pusht_assets_prep.pbs`、`pusht_deps_prep.pbs` 在 CPU-only allocation 准备。
A100 PBS 只读取这些标记和资产；不在 login node 下载、解压、安装或编译。

## authoritative evidence

- authoritative job：`24054689.pbs101`
- decision：PASS，30/30 paired units bitwise exact
- full-plan latency reduction：`4.6681%`
- plan-section reduction：`4.6681%`
- max peak-memory ratio：`1.00470`，PASS
- overall progression gate：**FAIL**，latency gate 要求 `>=10%`

因此这个 DINO-WM PushT CEM case 只能声称“行为等价但收益不足”，不能写成
通过的 full replacement。较早的 `pusht_transfer` job `23986275.pbs101` 也是
失败结果（30/30 非 exact、latency `4.8150%`），不能把它合并或改写成成功。

冻结协议、runner、verifier 和小型结果在本目录内；共享 helper 在
`../../common/cache_core.py`。
