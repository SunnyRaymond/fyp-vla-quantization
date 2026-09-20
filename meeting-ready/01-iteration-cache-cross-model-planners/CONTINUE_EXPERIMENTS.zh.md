# 继续实验指引

把整个 `01-iteration-cache-cross-model-planners/` 目录 staging 到 ASPIRE2A
后，按 case 的 README 设置外部 asset root 和 bundle root。不要从旧的
`experiment/idea-validation/...` 路径补文件。

## DINO-WM cases

```bash
export DINO_WM_ROOT=/scratch/users/ntu/yguo017/dino-wm-wall
export DINO_CACHE_BUNDLE_ROOT=$DINO_WM_ROOT/meeting-ready/01-iteration-cache-cross-model-planners
```

然后选择一个 planner：

```bash
qsub cases/dino-wall-cem/approx_screen.pbs
qsub cases/dino-wall-gd/gd_portability.pbs
qsub cases/dino-pusht-cem/pusht_iteration_cache.pbs
```

DINO-WM jobs 要求官方 reproduction root 已经有 `ASSETS_READY`、
`runtime-complete.tar`、对应 checkpoint 和 dataset。PushT 还要求既有
`PUSHT_ASSETS_READY`、`PUSHT_DEPS_READY_V5` 及 overlay；若缺失，可先使用
`cases/dino-pusht-cem/pusht_assets_prep.pbs` 和 `pusht_deps_prep.pbs`，资产准备
只能通过 CPU-only PBS 完成。

## LeWM PushT case

```bash
export LEWM_BUNDLE_ROOT=/scratch/users/ntu/yguo017/dino-wm-wall/meeting-ready/01-iteration-cache-cross-model-planners
export LEWM_REPO_ROOT=/scratch/users/ntu/lewm-pusht-iteration/le-wm
export STABLEWM_HOME=/scratch/users/ntu/lewm-pusht-iteration/stablewm_home
export LEWM_PYTHON_BIN=/scratch/users/ntu/yguo017/lewm-pusht-iteration/venv/bin/python
qsub cases/lewm-pusht-cem/lewm_pusht_iteration.pbs
```

缺少 LeWM source/checkpoint/dataset 时，先在 compute allocation 运行
`cases/lewm-pusht-cem/lewm_asset_prep.pbs`；结果 verifier 需要 recovery 时，
使用 `lewm_verifier_recovery.pbs` 指向同一 GPU summary。不要在 login node
下载、解压、安装、编译、模型加载或 benchmark。

## 结果解释顺序

先看对应 case 的 `artifacts/<job>/verifier.json`，再看 `RESULT.zh.md`。Decision
gate 与 latency/memory gate 分开记录；只有 overall system/progression gate
PASS，才可以把 case 写成已验证的 replacement。任何新改动都应复制 case 并新建
freeze，不要覆盖现有 authoritative artifacts。
