# LeWM EMA temporal confirmatory predictor protocol

## 目的与冻结边界

本轮是 Phase 3 tail-robust `NO-GO` 之后的单次 confirmatory predictor-only
实验。只测试同一条训练 trajectory 的 `online` 与 `EMA(decay=0.999)`；不再测试
tail weighting，不扩大 hidden size，不运行 official CEM、planner viability 或
closed-loop。`EMA` 是 primary，`online` 是 concurrent control。

训练严格复用 Phase 2 formal 的 512 train prepared rows、teacher targets 和
candidate bank；训练仍为 h256 shared recurrent/no attention/no conditioner、同一
initialization、same context schedule、AdamW、3000 updates 与原 score-distill
mean objective。EMA 在 step 0 clone online state，并在每个 optimizer step 后更新。

## Fresh evaluation selection

PBS compute node 读取官方 HDF5 的 `ep_len`，按 frozen `selection_seed=20300903`
将 valid episode IDs shuffle 一次。旧 heldout 为 `valid[:8]`，旧 train 为
`valid[8:520]`；本轮只选择 `valid[520:528]`，因此同时排除旧 heldout 和原 512
train，且不使用任何新结果选择 episode。

每个 fresh episode 建立三个 temporal anchors：`early=0`、
`late=episode_length-25-1`、`middle=floor(late/2)`。在官方 HDF5 接口下，
`current frame=anchor`、25 个 primitive actions 为 `[anchor, anchor+24]`、
`goal step=anchor+25`；因此三者均严格落在同一 episode 内。每个 episode 的
3 anchors 各使用 action-prefix seeds `20300917` 和 `20300918`，共 6 blocks；
8 个 episode 共 24 contexts、48 blocks、每 block 300 candidates。新 rows、teacher
targets 和 candidate banks 只在 PBS compute node 生成，不返回本地。

## Primary gate 与 secondary comparison

EMA terminal `step_3000` 在全部 48 blocks 上必须满足 inherited absolute gate：
Spearman median/minimum `>=0.95/0.80`、top-30 median/minimum `>=0.75/0.50`、
relative latent MSE median `<=0.25`、positive Spearman/top-30 `>=36/48`、
latency reduction `>=0.20`，且 integrity/convergence/causality 通过。

同时，early/middle/late 各自 16 blocks 必须满足 median Spearman `>=0.95`、
median top-30 `>=0.75`、positive Spearman/top-30 各 `>=12/16`。stratum 不重复
overall minimum thresholds；overall minimum 只在 overall gate 检查。

EMA-vs-online 的 secondary mechanism comparison 以 episode 为 replicate：先对
每个 episode 的 6 blocks 取 median，再得到 8 个 EMA-online deltas。要求 Spearman
episode-median delta `>=0`、top-30 episode-median delta `>=0`，并至少 5/8
episode 满足“EMA 在一项指标上严格改善，另一项不恶化”的规则。这个比较不能
替代 EMA absolute gate。

## Snapshots 与报告

只记录 500/1000/1500/3000 snapshots 的 descriptive metrics；terminal 固定为
step 3000，不 early stop，不用 fresh evaluation 选择 snapshot 或 hyperparameter。
报告必须分别列出 online 与 EMA 的 overall/stratum metrics、terminal paired
deltas、latency、causality、convergence、GPU telemetry、scope 与负面结果。

方法与报告格式参考 Scientific Agent Skills 的官方 arXiv 记录：
<https://arxiv.org/abs/2609.00065>（citation 不带版本后缀）。
