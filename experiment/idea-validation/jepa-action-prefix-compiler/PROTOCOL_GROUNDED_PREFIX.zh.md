# Grounded Prefix 三臂实验冻结协议

## 研究问题

RankDistill 已证明 planner-aware signal 可以改善 candidate ranking，但同时把 treatment/control 的 native-latent relative-MSE ratio 推到 `1.4228`。本实验只改变训练 target 的来源，验证真实 future-observation targets 是否能减少这项 trade-off；student architecture、native DINO-WM interface、planner objective 和 predictor timing boundary 均保持不变。

本实验不称为全量数据训练：`WIDE-T` 与 `WIDE-GT` 只消费 CPU prep manifest 中固定的 128 个 train examples 和 32 个 held-out examples。

## 三个 frozen arms

| Arm | 训练输入 | dense target | 作用 |
|---|---|---|---|
| `NARROW-T` | manifest train split 的前两个 train episodes 的真实 segment/actions | frozen teacher autoregressive rollout | 保留窄 episode coverage，隔离 coverage effect |
| `WIDE-T` | manifest 的 128 个 train examples | 同一 frozen teacher 的 rollout target | 区分 anchor coverage 与 target 来源 |
| `WIDE-GT` | 与 `WIDE-T` 完全相同的 manifest examples/actions | frozen `encode_obs` 对真实 future frames 的 latent | 测试 grounded dense supervision |

三个 student 从同一初始 state dictionary 开始，hidden dimension 为 `128`；goal 不进入 student。`NARROW-T` 固定消费 manifest 中按 episode 排序的前两个 train episodes（默认 8 examples），不使用 official anchors 的 synthetic action prefixes。两条 WIDE arm 必须逐 batch 使用相同的 manifest segment、action segment 和 update schedule。只允许 episode coverage 与 target 来源按 arm 定义不同。

## 固定数据与 seeds

- `H=5`，`frame_skip=5`；每个 anchor 的 future frame indices 为 `start + [5, 10, 15, 20, 25]`。
- manifest schema 必须为 `jepa-action-prefix-compiler.grounded-prefix-manifest`。
- train manifest 必须正好有 `128` examples，held-out manifest 必须正好有 `32` examples；train 与 held-out episode 不重叠。
- training seed：`20260921`。
- planner held-out action-prefix seeds：`20263920`、`20263921`；这两个 seed 不得复用 RankDistill 的 `20262920/20262921`。
- timing action-prefix seed：`20270921`。
- 每个 held-out seed 与两个 official anchors 组成四个 paired blocks；每 block 使用 300 candidates，top-k 为 30。

## 训练 call path

每个 update 从固定 manifest train examples 中 seeded-sample 32 个 segment；`NARROW-T` 的 sampling pool 限定为前两个 train episodes，`WIDE-T/WIDE-GT` 使用全部 128 个 train examples。每个 segment 读取起点帧、五个 future frames、对应 25 个 primitive actions，并按 `frame_skip=5` pack 成五个 action tokens。`NARROW-T/WIDE-T` 使用 frozen teacher rollout 得到五个 target latents；`WIDE-GT` 使用同一个 frozen `encode_obs` 得到五个真实 future latents。

三个 arm 的 loss 均为 dense native-latent MSE；本轮不加入 listwise KL、goal loss、self-consistency loss 或额外 retuning。每个 arm 运行 500 updates，batch `32`，AdamW learning rate `3e-4`；偶数 update 顺序为 `NARROW-T → WIDE-T → WIDE-GT`，奇数 update 反向，以抵消同一 GPU 上的 update-order 偏差。

## Held-out evaluation

1. 在两个 official anchors 上，以 fresh action-prefix seeds 生成四个 paired blocks；使用 frozen teacher terminal objective 计算各 student 的 Spearman 和 top-30 overlap。
2. 在 32 个 held-out manifest examples 上，以 8 个 seeded batches 评估真实 future latent 的 per-horizon relative-MSE 与 cosine；该项用于 grounded non-inferiority，不可被 planner ranking 结果替代。
3. 使用 cached native observation latent、normalized action prefix、CUDA synchronization 的 predictor-only timing；不包含 `encode_obs`、CEM、environment interaction 或 closed-loop。
4. 每 30 秒写 GPU utilization/VRAM telemetry，并在 summary 中保存 arm labels `NARROW-T`、`WIDE-T`、`WIDE-GT`。

## Frozen gates

### Capacity

- 三个 arm 的 outputs/training 必须 finite。
- future-action leakage 的 max absolute difference `<= 1e-6`。
- 每个 arm 的 last-10/first training latent-MSE ratio `<= 0.8`。

### Absolute planner fidelity

以 `WIDE-GT` 对 frozen teacher 的四个 held-out blocks 为主判据：

- median Spearman `>= 0.99`；
- median top-30 overlap `>= 0.95`；
- 最弱 block Spearman `>= 0.95`；
- 最弱 block top-30 overlap `>= 0.80`。

### Grounded non-inferiority

在真实 future frames 上，`WIDE-GT/WIDE-T` 的五个-horizon mean relative-MSE ratio 必须 `<= 1.25`。这不是与 NARROW-T 的比较，也不允许用 synthetic teacher target 的 MSE 代替。

### Grounding effect

primary effect 为 `WIDE-GT − WIDE-T`，要求：

- median paired Spearman delta `>= +0.05`；
- median paired top-30 delta `>= +0.10`；
- 两项各自至少 `3/4` blocks 为正。

### Coverage effect

`WIDE-T − NARROW-T` 使用同样的 `+0.05/+0.10/3-of-4` 描述性阈值，但只作为 coverage diagnostic，不改变 GO/NO-GO。

### Latency

`WIDE-GT` predictor-only median reduction 相对 frozen teacher 必须 `>=20%`。这不是 full-plan speedup，也不授权 CEM integration。

Overall 只有在 capacity、absolute fidelity、grounded non-inferiority、grounding effect 和 latency 这些 primary gates 均通过时才为 `GO`；`coverage effect` 只用于归因，不单独阻断 `GO`。任一 primary gate 失败即 `NO-GO`。结果后不得增加 updates、替换 seeds、放宽阈值或进入 CEM。

## Evidence boundary

本协议最多支持一个 predictor-level、DINO-WM PushT、固定 manifest 的 grounded-supervision 结论。它不能支持 closed-loop success、full planner speedup、LeWM transfer、all-JEPA universality、population-level statistical inference 或全量数据 claim。

## 文件与运行约束

- Freeze：`GROUNDED_PREFIX_FREEZE.json`
- GPU runner：`run_dino_pusht_grounded_prefix.py`
- CPU manifest：`prepare_grounded_prefix_assets.py`
- GPU wrapper：`dino_pusht_grounded_prefix.pbs`
- GPU job 必须只读取已 staged runtime/deps/checkpoint/data/manifest；禁止在 login node 下载、安装、解压、编译、模型加载、推理或重 I/O。
- 不取回 checkpoint 作为结果 artifact；取回 summary、job log、job status、GPU info/usage/telemetry 等小型文件即可。
