# LeWM PushT latent-delta oracle：组会与后续实验包

状态：`COMPLETE`。Step 1 / Step 2 已由 corrective PBS job `24844057.pbs101` 完成，退出码为 `0`。权威结果为 [summary.json](artifacts/24844057.pbs101/summary.json)，运行记录为 [job.log](artifacts/24844057.pbs101/job.log)。

## 先看什么

1. [MEETING_CARD.zh.md](MEETING_CARD.zh.md)：组会前的最小说明卡。
2. [Step 1 结果](reports/RESULT_STEP1_COMPRESSIBILITY.zh.md)：observed/model reusable PCA、cross-basis、per-trajectory oracle SVD 与 structure gate。
3. [Step 2 结果](reports/RESULT_STEP2_ORACLE_RANK.zh.md)：candidate ranking、controls、CEM first-action drift 与 planner gate。
4. [PROTOCOL.zh.md](PROTOCOL.zh.md) 和 [LEWM_DELTA_ORACLE_FREEZE.json](LEWM_DELTA_ORACLE_FREEZE.json)：冻结问题、split、rank、gate 和 claim boundary。

## 一句话结果

- Step 1 primary model-delta reusable PCA：冻结 `rank≤64` structure gate **FAIL，formal passing ranks 为空**；rank 64 的 held-out worst-case retained/MSE 为 `0.5728 / 0.4059`，忽略 rank 上限时 rank `192` 仅是 reconstruction diagnostic。
- Step 2 candidate diagnostic：model-PCA 最小通过 rank 为 `64`；per-trajectory oracle rank `3` 起通过 candidate gate，但它不是 reusable basis。
- Step 2 fixed-observation CEM：选入的 model-PCA rank `64/96` 均 **FAIL** first-action planner gate；rank-192 full control **PASS**。

## 研究问题与边界

在 official LeWM PushT 的 `192-D` planner-facing latent 中，基于 calibration transitions 得到的 reusable low-dimensional subspace，能否在 episode-disjoint held-out transitions 和 candidate actions 上重建 latent delta，并保留 planner-facing ranking 与 fixed-observation CEM decision？

这里检验的是 representation bottleneck 的 oracle diagnostic，不是 cheap predictor、真实 inference acceleration 或 closed-loop control experiment。Step 2 每条 approximate path 都先完整运行 LeWM predictor；CEM 没有 environment interaction。

## 冻结设置

- baseline：official LeWM PushT；latent `192-D`，packed action `10-D`；
- split：`256` calibration contexts、`8` held-out contexts，episode-disjoint；
- candidate banks：`16` blocks（`8 × 2`），每 block `300` candidates；
- official fixed-observation CEM：horizon `5`、`300` samples、`30` iterations、`topk=30`；
- 不运行 cheap delta predictor、speed benchmark、closed-loop PushT 或 end-to-end success。

## 当前结论

`rank≤64` 的 global model-PCA reconstruction gate 没有通过；corrective summary 的 observed/model `passing_ranks` 均为空。candidate ranking 的通过不能外推到 CEM action preservation。后续若继续，应先明确是否接受超出冻结 rank 上限的 representation 诊断；本 bundle 本身不授权把该结果写成 acceleration 或 control success。
