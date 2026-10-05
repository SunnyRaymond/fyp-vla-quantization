# W4A8 new-draw 固定观测结果

本目录保留周报提及的 W4A8 新 dither-draw 比较。`summary.json` 只含 A8 汇总和每轨迹结果；它是从本地混合位宽汇总中的 A8 字段提取而来。A8 原始评估文件没有保存在本地，因此这里不声称从 raw rows 重新聚合或可仅凭这些文件重跑模型。

新 draws 下，独立 dither action MSE 为 `3.6718629e-4`，旧 learned phase 迁移为 `3.7531191e-4`，RTN 为 `3.8525557e-4`。该 phase 比 independent dither 高约 2.21% MSE，但比 RTN 低约 2.58%。初筛的冻结 draws `1101..1104` 曾显示 learned phase `3.58978e-4`，比 independent dither `3.78977e-4` 低 5.28%；新 draws 显示该优势依赖 dither draws。两轮都复用了 task0 TEST IDs 8..15 的同一组 observations，故这是 draw-sensitivity 证据，不是新的轨迹或状态泛化验证。

协议为 `fastwam-activation-bit-screen-v1`：8 条 TEST 轨迹、每条 2 个 observations、sampler seeds `2026/2027`、dither draws `2101..2104`；每轨迹先平均 observations/draws，再对 8 条轨迹等权平均。配置为 `first_frame`、20 inference steps、`sigma_shift=1`、`compile=false`、32-action chunk、W4 group size 128、A8 row-maxabs/126。结果与 seed/误差分解限制见 [`../mechanism-screen/RESULT.zh.md`](../mechanism-screen/RESULT.zh.md)、[`../paired-followup/RESULT.zh.md`](../paired-followup/RESULT.zh.md) 和冻结协议 [`../mechanism-screen/PREREG.zh.md`](../mechanism-screen/PREREG.zh.md)。

`summary.json` 的来源是本地原 `activation-bit-screen/aggregate_result.json` 中 `a8_summary` 与 `a8_per_trajectory` 字段；其记录的原始 A8 source job 为 `25678987.pbs101`，原始结果路径是 `/scratch/users/ntu/yguo017/wam-activation-bits-20261004/artifacts/25678987.pbs101/result.json`。该 raw result 和其 `runtime_config.json` 当前不在本地目录。`extract_summary.py` 可用 `--input`、`--output` 从仍可访问的混合 aggregate 生成同样的 A8-only 摘要；它不读或复制 A4 字段。它不是 raw-result aggregator。

混合原件现位于 `D:\Downloads\FYP-unreported-experiments-20261005\activation-bit-screen\aggregate_result.json`；字段提取脚本的 `--input` 使用此路径即可。最新摘要的 provenance 记录了该归档位置。

保留的 `activation_screen.py` 与 `quant.py` 是未改动的历史通用执行源码快照，支持 A4/A8 两种位宽。模型执行时还依赖同级的 [`../mechanism-screen/runner.py`](../mechanism-screen/runner.py) 和 [`../paired-followup/decompose.py`](../paired-followup/decompose.py)。历史执行还需要旧 pilot 的 `locked_selection.json`、`frozen_observations.pt`、模型 checkpoint/statistics、FastWAM source、容器与缓存。`locked_selection.json` 和 observations 当前不在本地；项目 `.gitignore` 的 `*.pt` 规则排除 observations。旧 phase 的八个自由组可从 [`../mechanism-screen/pilot_phase_calibration.json`](../mechanism-screen/pilot_phase_calibration.json) 最后两个相同的文本 checkpoint 恢复：`action.condition=.25`、`action.early=.25`、`action.late=.5`、`action.middle=.75`、`proprio=0`、`video.early=0`、`video.late=.75`、`video.middle=0`；`video.condition=0` 是固定 gauge。

`source_run.pbs` 是原始的双位宽 PBS launcher 快照，保留其历史配置，不是独立的 A8 launcher。它含固定 scratch 路径，并依赖外部模型、container 和观察数据；单凭本目录不能重跑模型，也不构成 native kernel、latency 或 memory-acceleration 证据。
