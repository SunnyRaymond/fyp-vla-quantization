# Wall broadcast-coupling 实验 1：fresh multi-step pre-gate

本文件在读取新 scientific outputs 前冻结。目标是用最小成本决定是否继续扩大实验 1、执行 fixed-pool 实验 2 与 closed-loop 实验 3。

## 范围与样本

- 仅 DINO-WM Wall，官方 epoch-65 checkpoint、固定 source commit。
- fresh valid-local indices 固定为 `130..135`；`0..129` 保守视作已用/保留，不能换样。
- 每条 trajectory 从 frame 0 出发，使用 recorded normalized actions；评估 model-action horizon `H=1,5`，对应真实 future frames `5,25`。官方 Wall recorded trajectory 不提供 H=10 所需的 frame 50；该结构边界由零 scientific record 的首次 fail-closed allocation 暴露并在重试前写入本协议。
- 不运行 planner、不优化 actions、不使用 PushT。

## Arms 与干预

- `FP32`、A4 `RTN`。
- `shared_stochastic`：三组 A4 stochastic rounding draws，各自复制到 196 patches。
- `balanced_broadcast`：复用同三组 draws，以固定 patch offset 重排；逐 patch 跨三 draw multiset 与 Shared exact equal，总 input MSE exact matched。
- `iid_patch_stochastic`：每 patch 独立 SR，仅作非 matched control。

干预只发生在初始 `z[0,0,:,384:404]` 的 exact-broadcast 20D tail。后续 autoregressive predictor calls 保持 FP32；因此这是“初始 broadcast error 的 causal propagation”，不是 recurrent quantization 或 native A4 deployment。

Primary metric 是 prediction 与 recorded real-future DINO visual features 的 MSE；同时报告相对 FP rollout MSE。统计单位是 episode，三个 stochastic draws 先在 episode 内平均。

## Gate

全部 allocation/source/checkpoint/shape/finite/hook/model-state/raw matched-budget checks 通过后，才解释 science result。Primary gate 固定为 H=5：至少 5/6 fresh episodes 满足

`(Shared real-future MSE - Balanced real-future MSE) / Shared real-future MSE >= 10%`。

- 通过：`pre_gate_go`，才允许扩展正式实验 1，随后依次进入实验 2、实验 3。
- 未通过：`mechanism_no_go_stop`，停止扩展，不提交实验 2/3。
- 工程或 binding 失败：`inconclusive_binding`，不得按正/负 scientific signal 解读。

STaMP 当前没有可复现的本地实现或 adapter，不伪造 comparator，也不作优于 STaMP 的声明。
