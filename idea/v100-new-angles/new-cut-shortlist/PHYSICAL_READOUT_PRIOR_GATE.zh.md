# Physical-state readout × WM PTQ：prior gate

日期：2026-09-13。本文只做 source/prior/可识别性审查；未读取真实数组、加载模型、做本机数值或连接 cluster。

## 判定

**`identifiability_no_go / GPU=0`；novelty `unresolved`。** “量化误差是否主要落在任务不可观测 feature 方向”不能由当前 proposed readout 唯一识别。原因是 Wall 的 physical 2D state 已经是模型输入的一部分，并在 latent predictor 中有专用输出位置；对 full latent、predicted proprio 或拼接后的 predictor output 做 linear readout 会重复这条已有路径。当前 teacher-bias 的 0/6 结果只否定六条 recorded-future visual-feature 上的 RTN-W4 改善，不是 physical-state 证据，不能反过来修饰为本候选的经验 no-go。

## Source 事实（避免把维数猜错）

- `reproduction/dino-wm-wall/source/datasets/wall_dset.py:28–30` 读取 `states` 后直接 `self.proprios = self.states.clone()`；`:50–51` 从已加载 tensor 推断 `state_dim/proprio_dim`，而不是在这里固定为 2。`:85–94` 同时返回 raw `state` 与输入用的 normalized `proprio`。当前冻结 CPU manifest 已核实本 Wall asset 的两者实际为 2D；不要用文件顶部未参与该分支的 4D常量推断维数。
- `models/visual_world_model.py:96–134` 对 visual 与 proprio 分别编码；`concat_dim=1` 时在每个 visual patch 后拼接 proprio/action embedding。`:137–148` 的 `predict` 返回相同 latent 结构，`:172–187` 的 `separate_emb` 能直接取回 proprio slice。因而 full-latent readout 可以直接读取状态 embedding；它不是观测性新证据。
- 现有 teacher-bias protocol 已把 primary 限定为 recorded-future visual feature，并明确 raw state/proprio 与 learned 10D embedding 的边界；其结果也明确不支持 physical truth 或 planning claim。

## 假设、先例与不可识别处

可检验的窄假设应是：在同一 frozen visual encoder feature 上，Q−FP 的大 visual MSE 可能主要位于一个固定、state-predictive readout 看不见的子空间。这个对象最多是 **readout-visible component**，不能命名为“任务不可观测方向”：probe 的 nullspace 由 calibration 数据、正则化、坐标缩放和 feature 选择决定，未必等于环境状态或控制任务的不可观测子空间。二维 target 也只能约束二维投影，不能证明其余 382/patch 维度对任务无关。

固定 visual representation 后做 downstream linear probing 已是通用评估范式（[Light-weight probing of unsupervised representations for RL](https://arxiv.org/abs/2208.12345)；[R3M](https://arxiv.org/abs/2203.12601)）。DINO-WM/QuantWM 已覆盖 visual feature fidelity、rollout 与 planning sensitivity（[DINO-WM](https://arxiv.org/abs/2411.04983)；[QuantWM](https://arxiv.org/html/2602.02110v1)）。本候选可能新增的只是一个物理-state diagnostic estimand，未形成新的 quantizer 或 task-aware allocation；本次 targeted prior 不足以宣称被 exact paper 完全覆盖，也不足以支持 novelty。

## 若要重开，最低接口条件

1. readout 输入只能是 **visual DINO patch tokens**；排除 raw/encoded proprio、action、predictor proprio slice 和任何按 arm 重拟合的预测头。
2. 在与 screen 分离的固定 calibration trajectories 上拟合一次同一个 384→2 linear map（含明确 normalization、正则化和 fit hash），所有 FP/Q arm 在 held-out fresh trajectories 上共用；没有该 calibration asset 就先 `resource_blocked`，不能在 screen 样本上拟合。
3. target 必须绑定 WallDataset 的真实 raw 2D state，并同时记录 normalized input 与 source trajectory/frame identity。主结果只称 `visual-feature-to-state readout error`，另报 feature MSE、readout-visible delta 与 residual；不得把低 readout error 当作 physical truth、control success 或 deployment 保真。
4. 至少需要一个预注册的 feature-only null/readout control（例如 frozen random projection或独立 calibration split）来说明 probe 没有因容量/尺度造成假改善；否则“Q error 在 nullspace”不可反驳。若 full-latent/readout、per-arm refit、target leakage 或 calibration 与 test 不分离，立即 `identifiability_no_go`。

在上述条件具备前，不申请 GPU、不修 teacher-bias、不追加样本。即使条件满足，最小结果也只能是 fixed Wall checkpoint、fixed RTN recipe 下的 bounded representation diagnostic；它不能把 visual readout 的二维投影提升为“任务可观测性”或真实闭环因果证据。

检索边界：本次只用两条 targeted primary queries 核对 linear probing/task-based representation 先例，未作 exhaustive novelty search。
