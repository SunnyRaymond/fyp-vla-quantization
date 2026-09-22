# LeWM temporal-balanced training confirmatory predictor protocol

## 目的与冻结边界

本轮是 Phase 3 fresh multi-anchor observation 之后的单次 confirmatory
predictor-only experiment。新 hypothesis 是：原来的 `anchor=0` training
distribution 可能没有充分覆盖 middle/late temporal states；在完全不改变
architecture、loss、optimizer、candidate bank、batch、3000 updates 或 compute
budget 的条件下，按 temporal strata 平衡 training anchors 可能改善 fresh
multi-anchor predictor fidelity。

`step0_train` 是 concurrent control：复用原 Phase 2 formal 的 512 episode IDs，
每个 episode 使用 `anchor=0`，online terminal `step_3000`。`balanced_temporal_train`
使用完全相同的 512 episode IDs 与 context ordinals，仅将 ordinal `mod 3` 固定映射
到 `early/middle/late`，目标 counts 为 `171/171/170`：`early=0`、
`late=episode_length-26`、`middle=floor(late/2)`。不按数据或结果挑选 anchor。

两臂共享初始化、AdamW、training seed、context schedule、architecture、objective、
batch size、3000 updates 和 candidate slate。为保证 paired comparison，control
prepared bank 的 64 candidate actions 按 context ordinal 原样复用于 treatment；
treatment 只在新 anchor state 上重算 official teacher targets/objectives。所有
HDF5/model/teacher/training/evaluation/prepared rows 都只在 PBS compute node
处理；prepared rows 和 checkpoint 不回传。

## Fresh confirmation evaluation

沿用 `selection_seed=20300903` 的 valid episode shuffle，但排除旧 heldout
`valid[:8]`、原 512 train `valid[8:520]` 和上一轮 fresh `valid[520:528]`，本轮
固定使用全新 `valid[528:536]`。每个 episode 建立 early/middle/late 三 anchors，
每个 anchor 使用新 action-prefix seeds `20300927` 和 `20300928`，共 24 contexts、
48 blocks，每 block 300 candidates。episode 是 independent replicate；anchor 与
seed 是 nested，candidate 不是 replicate。

两臂均评估 terminal `step_3000`；`500/1000/1500/3000` 只作 descriptive snapshots，
不 early stop、不用 test 选择 snapshot。

## Gates 与 scope

balanced treatment 的 primary gate 要求 inherited overall absolute gate 与
early/middle/late stratum gate 同时通过。Overall 要求 Spearman median/minimum
`>=0.95/0.80`、top-30 median/minimum `>=0.75/0.50`、relative latent MSE median
`<=0.25`、positive Spearman/top-30 `>=36/48`、latency reduction `>=0.20`，并通过
integrity/convergence/causality。每个 stratum 要求 median Spearman/top-30
`>=0.95/0.75`、positive blocks 各 `>=12/16`；minimum thresholds 只在 overall 检查。

Secondary mechanism comparison 以 episode 为 replicate，先聚合每 episode 的 6
blocks，再比较 balanced-control deltas；两项 episode-median delta 均需 `>=0`，且
至少 `5/8` episode 达到一项严格改善、另一项不恶化。secondary 不能替代 absolute
gate。

本轮明确不运行 official CEM、planner viability 或 closed-loop；均标记
`NOT_RUN_BY_SCOPE`。方法与 reporting provenance 参考 Scientific Agent Skills
官方 arXiv 记录：<https://arxiv.org/abs/2609.00065>（不带 `v` 后缀）。
