# LeWM state-action prefix GRU confirmatory predictor protocol

## 目的与冻结边界

本轮是 temporal-balanced train 之后的单次、两臂、predictor-only confirmatory
experiment。`balanced_base` 复现 Phase 5 temporal-balanced training：512 个相同
contexts、early/middle/late anchor counts `171/171/170`、3000 updates、原始
`LeWMCompactRecurrentTransitionStudent` h256、free-running latent MSE 加
`0.1 * context-normalized teacher-score SmoothL1`。`state_action_prefix_gru`
只新增一个 64-D shared prefix hidden state，其他 architecture、loss、optimizer、
batch、candidate slates、teacher targets、schedule、seed 和 terminal snapshot 均固定。

Treatment 的 prefix 初值为 `tanh(Linear(192,64)(initial current latent))`；每个 horizon
step 用 `GRUCell(concat(packed_action[10], Linear(192,64)(current predicted state)),
prefix_hidden)` 更新；再以 `Linear(64,256)` 加入原始
`history_condition + latent_hidden + action_hidden`，之后执行原 shared transition。
GRU 与三组 linear 在五个 horizon steps 之间共享；不使用 attention、goal、teacher、
encoder input、per-horizon copies 或 hidden-size sweep。新增模块使用标准 PyTorch
初始化；不把整个 prefix projection zero-init。control 与 treatment 的 base-owned
parameters 在初始化后必须 bitwise equal。

两臂使用同一 Phase 5 balanced training bank；control candidate slates 按 ordinal
复用。所有 HDF5、official checkpoint、teacher target、training、evaluation 和 fresh
rows 仅在 PBS compute node 发生，prepared rows 与 checkpoint 不回传。

## Fresh confirm test

沿用 `selection_seed=20300903` 的 valid shuffle，排除旧 heldout `valid[:8]`、正式
512 train `valid[8:520]`、旧 fresh `valid[520:528]` 与上一轮 fresh
`valid[528:536]`，固定使用全新的 `valid[536:544]` 八个 episodes。每个 episode 使用
early/middle/late 三 anchors：`0`、`floor((length-26)/2)`、`length-26`；每个
anchor 使用 action-prefix seeds `20300937`、`20300938`，共 24 contexts、48 blocks，
每 block 300 candidates。episode 是 independent replicate，anchor/seed 是 nested，
candidate 不是 replicate；不根据 test 结果挑选 snapshot 或参数。

## Gates 与 scope

Treatment terminal step 3000 必须同时通过 inherited absolute predictor gate 与
early/middle/late stratum gate：overall Spearman median/minimum `>=0.95/0.80`、
top-30 `>=0.75/0.50`、relative latent MSE median `<=0.25`、positive Spearman/top-30
至少 `36/48`，latency reduction `>=0.20`，并通过 integrity/convergence/causality；
每个 stratum 的 median Spearman/top-30 至少 `0.95/0.75`，positive blocks 各至少
`12/16`。`500/1000/1500/3000` 只作 descriptive snapshots，不 early stop。

Secondary mechanism 先对每个 episode 的六个 blocks 取 median，再比较
`state_action_prefix_gru - balanced_base`；两项 episode-median delta 均需 `>=0`，
且至少 `5/8` episodes 在一项严格改善、另一项不恶化。另报告 frozen latency guard：
GRU predictor 相对 balanced-base overhead `<=35%`，同时 teacher reduction `>=20%`。
Secondary 不得替代 absolute/stratum primary gate。

本轮明确不运行 official CEM、planner viability 或 closed-loop；三者均为
`NOT_RUN_BY_SCOPE`。方法与 reporting provenance 参考 Scientific Agent Skills 官方
arXiv 记录：<https://arxiv.org/abs/2609.00065>（不带 `v` 后缀）。
