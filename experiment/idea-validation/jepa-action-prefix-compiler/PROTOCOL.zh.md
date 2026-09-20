# JEPA Action-Prefix Compiler：DINO-WM PushT Stage A

## 目的与边界

本阶段只回答一个问题：冻结 official DINO-WM PushT checkpoint 后，能否把其五步 autoregressive action-conditioned predictor post-hoc compile 成一次并行 action-prefix student，同时保留 candidate objective ranking 并得到真实 predictor latency reduction。

这是 predictor-level capacity/fidelity/latency smoke：不运行 CEM、MPC、environment action execution 或 closed-loop evaluation。已有 observation-prefix cache 的 PushT B arm 已 FAIL（decision gate FAIL，full-plan reduction `4.815% < 10%`），因此 Stage A 不依赖 B，也不能声称 full framework 或 composition 已验证。

## Frozen model contract

- backend：official DINO-WM PushT checkpoint；source encoder、action encoder、predictor 与 planner全部冻结。
- anchors：official `PlanWorkspace` dset path 产生的 `pusht_obs_00`、`pusht_obs_01`。
- horizon：`H=5`。
- teacher target：同一 native cached observation latent 与 action prefix 经原始 predictor autoregressively rollout 得到的 `z[t+1:t+5]`。
- student input：native encoded context `{visual, proprio}` 与 raw normalized actions `[B,5,A]`。
- student output：native observation latents `visual [B,5,P,Dv]`、`proprio [B,5,Dp]`；不输出 action dimensions。
- student 内不得调用 `encode_obs`；不引入跨模型 canonical latent space。

## Frozen execution

- train seed：`20260919`；held-out seeds：`20261920`、`20261921`；timing seed：`20270920`。
- training：500 steps，batch 32，AdamW `3e-4`，hidden dim 128。
- held-out：每个 seed、每个 anchor 各 300 action prefixes。
- timing：batch 300，3 warmups + 10 synchronized repeats。
- 所有 model loading、training 与 benchmark 只能在 guarded A100 PBS allocation 内完成；job 每 30 秒记录 GPU utilization 与 VRAM。
- 不下载、不安装、不做 checksum/hash；已 staged checkpoint/data/runtime 直接复用。

## Controls and metrics

1. **Future-action leakage**：对每个 cut `k=1..4`，固定前 `k` 个 actions、只改 suffix；student 前 `k` 个 outputs 的 max-abs 必须 `<=1e-6`。
2. **Capacity**：输出 finite 且 shape/dtype/device 符合 native contract；训练末 10 steps median loss / first loss `<=0.8`。
3. **Latent diagnostics**：逐 horizon 报 relative MSE 与 cosine similarity；latent error 本身不是通过条件。
4. **Planner-relevant fidelity**：用 official terminal latent-to-goal objective 比较 teacher/student candidate scores；跨两个 anchors 与两个 held-out seeds报 Spearman 与 top-30 overlap。
5. **Predictor timing**：同一 cached anchor latent、同一 batch-300 actions 下，比较五步 frozen teacher rollout与一次 student forward。该 timing 排除 encoder、CEM 与 environment，不能表述为 full-plan speedup。

## Frozen gates

Stage A 只有在以下条件全部满足时才 GO：

- capacity/control PASS；
- held-out objective Spearman aggregate `>=0.99`；
- held-out top-30 overlap aggregate `>=0.95`；
- predictor-level median latency reduction `>=20%`。

任一 gate 失败即 no-go，不调阈值、不追加 steps/repeats 来挽救当前 recipe。结果可以用于定位 failure mode，但不进入 planner integration。

## 后续边界

Stage A GO 后，下一步应先做 A/C full `CEMPlanner.plan()` comparison，验证 elite/top-k、`mu/sigma`、final first action 与 full-plan latency。只有 predictor planner gate 和一个重新成立的 observation-cache B arm 都通过，才有资格运行 A/B/C/D 2x2；当前 B arm 的失败结果禁止被包装成可组合证据。
