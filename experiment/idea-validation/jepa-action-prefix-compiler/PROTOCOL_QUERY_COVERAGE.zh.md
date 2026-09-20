# Query-Coverage 2×2 实验冻结协议

## 一句话目标

这轮实验把两个容易混在一起的因素拆开：student 看过多少不同的 observation context，以及 student 是否看过 planner 实际可能查询的 counterfactual action prefixes。四个 arm 只在这两个因素上不同；student architecture、teacher、训练步数和 evaluation block 保持固定。QUERY 的第三分量是训练前冻结的 one-step CEM elite-resample。

本协议允许 training-only 的 one-step CEM proposal generation；它不做 planner integration、不执行环境，也不宣称 closed-loop 成功。

## 研究问题与分析单位

主要问题是：在 context coverage 相同的情况下，加入 planner-query-like action prefixes 是否能提升 held-out candidate ranking？主要比较是：

```text
WIDE-QUERY − WIDE-LOGGED
```

真正的独立分析单位是一个 `heldout episode context × fresh action-prefix seed` block。每个 block 有 300 个 candidates；这 300 个 candidates 是同一 block 内的重复测量，不能当作 300 个独立样本。

## 四个 frozen arms

| Arm | Context coverage | Training action distribution | Dense target |
|---|---|---|---|
| `NARROW-LOGGED` | manifest 前两个 train episodes，共 8 contexts | 100% 真实 logged action prefixes | frozen teacher rollout |
| `WIDE-LOGGED` | manifest 全部 32 train episodes，共 128 contexts | 100% 真实 logged action prefixes | frozen teacher rollout |
| `NARROW-QUERY` | 同一个前两个 train episodes，共 8 contexts | 50% logged + 25% Gaussian planner-init + 25% one-step CEM elite-resample | frozen teacher rollout |
| `WIDE-QUERY` | 同一个全部 32 train episodes，共 128 contexts | 50% logged + 25% Gaussian planner-init + 25% one-step CEM elite-resample | frozen teacher rollout |

这里的 `NARROW/WIDE` 来自已经准备好的 CPU manifest，不重新生成数据。manifest 必须是 `jepa-action-prefix-compiler.query-coverage-manifest`，由 `prepare_query_coverage_assets.py` 生成，并且正好包含 128 个 train contexts、8 个 held-out contexts；train 与 held-out episode 不重叠。

NARROW 固定取 manifest 中按原顺序排列的前两个 train episode，并保留每个 episode 的四个 contexts。WIDE 使用全部 128 个 train examples。不能根据结果更换 episode、删 context 或补跑新的 manifest。

## Student、teacher 和训练固定项

四个 arm 都使用同一个 `NativeDinoPrefixStudent`：

- hidden dimension `128`；
- `H=5`，`frameskip=5`；
- primitive action dimension `2`，每个 action token 打包 `5 × 2 = 10` 个数；
- 输入是 cached native DINO observation latent 与五个 action tokens；
- goal 不进入 student；
- 输出是五个 future native observation latents；
- 不调用 `encode_obs`，不改变 frozen DINO-WM encoder/predictor。

四个 student 从同一个初始 state dictionary clone。每个 arm 训练 `500` steps，batch size `32`，AdamW，learning rate `3e-4`。loss 只有 dense native-latent MSE；不加入 rank loss、listwise KL、goal loss、self-consistency loss 或真实 future-frame target。

每个 update 的 context index schedule 在同一 coverage level 的两个 arm 之间完全相同：`NARROW-LOGGED` 与 `NARROW-QUERY` 共用一份 8-context schedule，`WIDE-LOGGED` 与 `WIDE-QUERY` 共用一份 128-context schedule。这样 action effect 不会被 context batch 改变混淆。

update order 也预注册：偶数 step 为

```text
NARROW-LOGGED → WIDE-LOGGED → NARROW-QUERY → WIDE-QUERY
```

奇数 step 为反向顺序：

```text
WIDE-QUERY → NARROW-QUERY → WIDE-LOGGED → NARROW-LOGGED
```

## QUERY action mixture 的可实现定义

所有 action 都在现有 DINO-WM 使用的 normalized primitive-action coordinate system 中生成，shape 为 `[batch, 5, 2]`。每个 QUERY batch 固定有 32 行：16 行 logged、8 行 Gaussian、8 行 one-step CEM elite-resample。role permutation 在 CPU 上由固定 seed 产生，并在 NARROW-QUERY/WIDE-QUERY 之间复用。

### 1. Logged（50%）

使用被选中 train context 所配对的真实五-token action prefix，数值原样保留，不做 held-out 或 official-anchor 替换。

### 2. Gaussian planner-init（25%）

每个 primitive action coordinate 独立采样：

```text
a ~ Normal(0, 1), shape = [8, 5, 2]
```

这是现有 predictor-level sampler 使用的 standard-normal normalized proposal。它不读取 goal，不读取 teacher score，也不读取 held-out 数据。由 CPU `torch.Generator` 产生后再传到 GPU，避免 CUDA RNG 状态影响复现。

### 3. One-step CEM elite-resample（25%）

这是预注册的、真正可执行的 one-step CEM proposal generation。它只在 training 开始前执行一次，不引入 multi-step planner loop。对每个 train context，使用该 context 的 frozen native latent、terminal goal objective 和固定 context seed：

```text
1. sample M=64 candidate prefixes from Normal(0, 1), shape = [64, 5, 2]
2. frozen teacher rollout every candidate
3. score terminal prediction with this context's frozen terminal goal objective
4. select the K=8 lowest-cost candidates
5. compute per-coordinate elite mean and variance over the [8, 5, 2] elites
6. variance = max(variance, 0.05) per coordinate
7. sample exactly one prefix from the resulting diagonal Normal
```

每个 context 的这一个 resampled prefix 在第一个 student update 前就预计算并冻结；训练中只做 lookup。M=64、K=8、Normal proposal、terminal objective、variance floor `0.05`、context seed formula 都在运行前固定。CEM candidate scoring 只读取 train contexts，不读取 held-out，不读取 student 结果，也不做 post-hoc elite 筛选。

三种 action 的每一条 prefix 都要通过同一个 frozen teacher rollout 生成 dense target。one-step CEM 的 M=64 scoring rollouts 只负责冻结 proposal bank；Gaussian 与 CEM-resampled prefixes 都是 counterfactual prefixes，不能写成 ground-truth target。

## Seeds 与 held-out 隔离

固定使用：

| 用途 | Seed |
|---|---:|
| training/context schedule | `20260922` |
| QUERY action generation / CEM base | `20260926` |
| role permutation | `20260927` |
| held-out ranking candidates | `20264925`, `20264926` |
| timing candidates | `20270925` |

所有 schedule、Gaussian proposals、每个 context 的 64 个 CEM candidates 和最终 resample 都由 CPU generator 产生；context `i` 的 CEM seed 固定为 `20260926 + 1009 × i`，其中 `i` 从 0 开始。不能根据中间结果换 seed、增加 training step 或改变 mixture 比例。

训练 action generation 只能读取 manifest train examples。held-out episode、held-out actions、held-out observations、held-out seeds、official evaluation anchors 和 evaluation candidate tensors 均不得参与 QUERY 生成。

## Held-out evaluation：16 个 block

held-out manifest 冻结提供 8 个 logged contexts。按 manifest 顺序使用全部 8 个 held-out contexts，分别使用两个 fresh action-prefix seeds，形成：

```text
8 held-out episode contexts × 2 fresh seeds = 16 blocks
```

每个 block 有 300 candidates、`top-k=30`。四个 arm 和 frozen teacher 在同一个 block 中使用完全相同的 candidate action tensor 与 goal。candidate 使用现有 predictor-level 的 `Normal(0,1)` normalized action sampler。主要统计量按 16 个 block 计算：median、minimum、paired delta 和 positive-block count。

另在这 8 个 held-out manifest contexts 上，用各自真实 logged prefixes 做 latent non-inferiority evaluation：target 仍是 frozen teacher rollout，比较 `WIDE-QUERY / WIDE-LOGGED` 的 mean relative native-latent MSE ratio。ranking 结果不能替代该项。

Timing 只测 cached native observation latent + normalized action prefix 到 predictor output 的边界，包含 CUDA synchronize；不包含 observation encoder、CEM、环境交互或 closed-loop。GPU job 每 30 秒写入 utilization/VRAM telemetry。

## Frozen gates

### Capacity 与 causal control

- 四个 arm 的 training/output 必须 finite，不能 OOM、NaN 或 silent fallback；
- future-action leakage 的 max absolute difference `≤ 1e-6`；
- 每个 arm 的 last-10/first training latent-MSE ratio `≤ 0.8`；
- shape、dtype、device 必须一致，student 不得隐藏调用 `encode_obs`。

### WIDE-QUERY 的 absolute fidelity

以 `WIDE-QUERY` 对 frozen teacher 的 16 个 held-out blocks 为主：

- median Spearman `≥ 0.99`；
- minimum Spearman `≥ 0.95`；
- median top-30 overlap `≥ 0.95`；
- minimum top-30 overlap `≥ 0.80`。

### Primary action-query effect

在相同 16 个 blocks 上计算 `WIDE-QUERY − WIDE-LOGGED`，要求：

- median ΔSpearman `≥ +0.05`；
- median Δtop-30 `≥ +0.10`；
- Spearman 为正的 blocks `≥ 12/16`；
- top-30 为正的 blocks `≥ 12/16`。

### Latent non-inferiority 与 latency

- `WIDE-QUERY / WIDE-LOGGED` 在全部 8 个 held-out logged contexts 上的 mean relative-MSE ratio `≤ 1.25`；
- `WIDE-QUERY` predictor-only median latency reduction 相对 frozen teacher `≥ 20%`。

只有 capacity、absolute fidelity、primary action-query effect、latent non-inferiority 和 latency 全部通过才可记为 `GO`。不得因 diagnostic 结果好而绕过 primary gate，也不得在结果后 retune。

## Diagnostic：context effect 与 interaction

以下只报告，不阻断 GO/NO-GO：

1. Context effect：`WIDE-LOGGED − NARROW-LOGGED`，报告 median ΔSpearman、median Δtop-30、positive-block count 和逐 block paired delta；
2. Query effect at narrow coverage：`NARROW-QUERY − NARROW-LOGGED`；
3. Interaction：

   ```text
   (WIDE-QUERY − NARROW-QUERY)
   − (WIDE-LOGGED − NARROW-LOGGED)
   ```

   分别报告 Spearman 与 top-30 的 median interaction 和逐 block interaction。

这三个 diagnostic 用于判断收益究竟主要来自 context coverage、action-query coverage，还是二者组合；它们不能替代 primary `WIDE-QUERY − WIDE-LOGGED` gate。

## Claim boundary

若 GO，本实验最多支持以下结论：在 frozen DINO-WM PushT、固定 query-coverage CPU manifest、固定 NativeDinoPrefixStudent 和固定 16-block predictor-level evaluation 下，预注册的 `50% logged + 25% Gaussian planner-init + 25% one-step CEM (M=64,K=8) elite-resample` action mixture 相对于 logged-only mixture 改善了 candidate ranking，并满足规定的 native-latent 与 latency 约束。

不能据此声称：

- 已验证 full multi-step 或 exact deployment-time goal-conditioned CEM proposal coverage；
- 已完成 planner integration、full-plan speedup 或 closed-loop task success；
- 已完成 LeWM transfer 或 all-JEPA generalization；
- 已获得 population-level statistical inference；
- 已训练全量数据；
- 结果对冻结 2×2 之外的设置具有普遍因果效应。

## 文件与执行边界

- Freeze：`QUERY_COVERAGE_FREEZE.json`
- 本协议：`PROTOCOL_QUERY_COVERAGE.zh.md`
- 相关 CPU manifest：由 `prepare_query_coverage_assets.py` 生成的 frozen manifest
- 执行环境：只允许在获批 PBS compute allocation 上进行；login node 不得下载、安装、解压、编译、加载模型、推理或 heavy I/O。
- 本次冻结动作只写入上述两个文件，不运行实验、不提交 Git、不修改其他文件。
