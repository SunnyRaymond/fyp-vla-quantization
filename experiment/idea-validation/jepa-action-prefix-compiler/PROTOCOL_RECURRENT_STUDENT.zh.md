# DINO-WM PushT：Shared Recurrent Latent-Transition Student 冻结协议

状态：`frozen / protocol only / 未提交作业`

对应冻结文件：[RECURRENT_STUDENT_FREEZE.json](RECURRENT_STUDENT_FREEZE.json)

## 1. 这项实验要回答什么

前面的实验已经把几个较直接的解释压低了优先级：增加训练 updates、扩大 hidden width、增加同一批 episode 的 context support，以及加入固定的 planner-aware rank loss，都没有在冻结 ranking gate 上成立。下一步只改变 predictor 的 temporal architecture：把现有的并行 direct prefix student 换成一个最小的 shared recurrent latent-transition student。

这项实验只关注 `action-conditioned predictor`。`encode_obs`、observation encoder、CEM 执行和闭环控制都不在范围内。

核心问题是：

> 在完全相同的 256-context、四-query slate、teacher targets、training schedule 和 held-out candidate bank 下，一个逐步反馈的 shared recurrent predictor，能否改善 candidate ranking，同时以更少的参数和更低的 predictor-only latency 保持 latent fidelity？

这里的 control 是已经完成的 query-slate 实验中的 `control`，不是重新训练的 baseline。

## 2. Baseline：只读的 direct student control

Baseline 来自 job `24494756.pbs101`：

- 本地 summary：`artifacts/24494756.pbs101/query_slate_rank_summary.json`；
- arm：`control`；
- snapshot：`step_1500`；
- architecture：`NativeDinoPrefixStudent`；
- loss：dense native-latent MSE；
- 训练 context：256；
- 每个 context：4 个 action queries；
- held-out：16 个 paired blocks。

其 step-1500 的只读参考值为：

| 指标 | control |
|---|---:|
| median Spearman | `0.9332448` |
| minimum Spearman | `0.8302790` |
| median top-30 overlap | `0.7166667` |
| minimum top-30 overlap | `0.5333333` |

这个 control 的 per-block metrics 和 summary fields 是权威参考。实验不得重新训练 control、加载 control checkpoint、修改 control summary，或用新的随机运行替换它。

Baseline 的 direct student 结构是：

1. 将当前 context 的 visual/proprio anchor 拼成 context summary；
2. 用一个两层 causal action-prefix Transformer 编码完整 action prefix；
3. 用独立的 visual/proprio residual heads 并行生成五个未来 native latent。

本实验的 architecture contrast 不要求两种不同结构拥有相同的初始 parameter tensor；两者使用相同的 initialization seed 和数据/调度 seeds，但 `initial_parameter_state_equal` 记录为 `not applicable`。

## 3. Treatment：一个 shared recurrent latent-transition student

Treatment 类名固定为 `RecurrentNativeLatentTransitionStudent`，hidden size 固定为 `256`。它没有 goal 输入，也不包含 teacher、source encoder 或 source predictor 的 parameter。

### 3.1 初始状态

从 cached native observation latent 取最后一个 context state：

```text
visual_0  = context["visual"][:, -1]    # [B, patches, 384]
proprio_0 = context["proprio"][:, -1]   # [B, 10]
```

这一步只消费已经缓存好的 native latent，不调用 `encode_obs`。之后的五步都使用 student 自己上一部的预测结果。

### 3.2 每一步的 cell

在第 `k` 步，输入仅为：

```text
(visual_k, proprio_k, packed_action_k)
```

其中 `packed_action_k` 的维度为 `10`，对应 official DINO-WM 的 packed action token，而不是未经 packing 的二维 primitive action。

固定 projections：

```text
v_h = visual_in(visual_k)       # Linear(384, 256)，逐 patch
p_h = proprio_in(proprio_k)     # Linear(10, 256)
a_h = action_proj(action_k)     # Linear(10, 256)
```

唯一的 spatial mixing 是 projected visual patches 的 mean pooling：

```text
s_k = mean(v_h, dim=patch) + p_h + a_h
r_k = Linear(1024, 256)(GELU(Linear(256, 1024)(LayerNorm(s_k))))
```

`r_k` 是同一个 shared residual transition 在每一个 `k=0..4` 重复使用的结果。输出更新为：

```text
visual_{k+1}  = visual_k  + visual_out(LayerNorm(v_h + r_k[:, None, :]))
proprio_{k+1} = proprio_k + proprio_out(LayerNorm(p_h + r_k))
```

其中：

- `visual_out = Linear(256, 384)`；
- `proprio_out = Linear(256, 10)`；
- activation 是 `GELU`；
- dropout 为 `0`；
- 没有 attention、Transformer layer、per-horizon cell copy 或 horizon-specific head；
- `visual_{k+1}` 与 `proprio_{k+1}` 是下一步的真实输入，不能 detach，也不能换成 teacher target。

因此它表达的是一个明确的 latent transition：

```text
当前 native latent + 当前 action token
        ↓ shared transition cell
下一 native latent
        ↓ feedback
下一步 transition
```

### 3.3 训练时的语义

训练目标仍然是 frozen official teacher rollout 产生的 dense native-latent target，但 recurrent student 的 rollout 始终是 free-running：

- 不使用 teacher forcing；
- 不把 ground-truth future frame 或 teacher future latent 喂回 cell；
- 不加入 rank loss、listwise KL、goal loss 或 self-consistency loss；
- 只计算所有 `32 query rows × 5 steps × native latent dimensions` 的 dense MSE。

这样实验比较的是 architecture，而不是 architecture 加额外 supervision。

## 4. 两边固定不变的训练资源

Treatment 必须复用 `QUERY_SLATE_RANK_FREEZE.json` 的以下内容：

| 项目 | 固定值 |
|---|---|
| train contexts | 256 |
| train episodes | 32 |
| 每 context queries | 4 |
| query rows | 1024 |
| 每 update contexts | 8 |
| 每 update query rows | 32 |
| training updates | 1500 |
| snapshots | 500 / 1000 / 1500 |
| optimizer | AdamW |
| learning rate | `3e-4` |
| weight decay | `0.01` |
| rank loss | `0` |

四个 query slot 的顺序和内容不变：

1. `primary_logged`；
2. `gaussian_planner_init_a`；
3. `gaussian_planner_init_b`；
4. `one_step_cem_resample`。

每个 context 的四个 action prefix 必须沿用现成 slate bank，且 pairwise distinct。CEM 仍然只是 training-only 的 `M=64, K=8` 一步 proposal；student 输出不参与 proposal 生成。

### 4.1 Schedule 与 seeds

| 用途 | seed |
|---|---:|
| initialization | `99` |
| training | `20260930` |
| context schedule | `20260931` |
| action slate | `20260932` |
| CEM proposal | `20260933` |
| held-out action prefixes | `20264925`, `20264926` |
| timing candidates | `20270925` |

Context schedule 仍然是：每 32 个 updates 对 `0..255` 做一次 CPU permutation，每个 update 取连续 8 个 ordinal。Treatment 必须使用完全相同的预计算 schedule；不得因为 recurrent cell 改变 batch 语义。

Teacher targets、slate bank、context schedule 和 held-out candidate bank 都在训练前固定并复用。禁止看结果后重新采样、换 seed、调整 schedule 或改变 loss。

## 5. Held-out paired evaluation

Held-out 部分完全继承 query-slate protocol：

- 8 个 held-out contexts；
- 两个 fresh action-prefix seeds：`20264925`、`20264926`；
- 每个 block 300 candidates；
- 总计 16 个 paired blocks；
- pairing key：`(action_prefix_seed, heldout_context_index)`；
- block 是统计单位，300 candidates 不是 300 个独立样本。

每个 block 使用相同的 context、candidate actions、goal 和 frozen teacher costs。最终主比较是：

```text
recurrent treatment − read-only slate-MSE control
```

不能把 recurrent treatment 与旧的 h128、Dhigh context-density treatment 或 rank treatment 混作本实验的 control。

### 5.1 Ranking

报告每个 paired block 的：

- objective Spearman；
- top-30 overlap；
- teacher-relative native-latent MSE；
- per-horizon cosine。

同时保留 step `500/1000/1500` 的 recurrent snapshot 结果，但 architecture effect 的正式判定使用 step `1500`。

### 5.2 Latent fidelity

对每个 held-out context 的 logged action prefix，比较 recurrent 输出与 frozen teacher rollout 的 dense relative native-latent MSE。报告：

```text
recurrent_logged_mse / baseline_control_logged_mse
```

这个 ratio 的上限固定为 `1.25`。

### 5.3 Causality

对 unchanged prefix length `1, 2, 3, 4`，只改变后续 action tokens，比较输出到 unchanged prefix 为止的差异。所有 case 的最大绝对差必须不超过 `1e-6`。

这个检查尤其重要，因为 recurrent feedback 可能把未来 action 的错误依赖通过状态传播；若 causality 失败，不能解释 ranking delta。

### 5.4 Predictor-only latency

Latency 边界固定为：

```text
cached native observation latent + normalized action prefix → predictor rollout
```

不包括 `encode_obs`、CEM、environment 或 closed-loop control。固定 batch size `300`、warmup `3` 次、technical repeats `10` 次，并在 CUDA 前后同步。

报告三组数：

1. recurrent student 相对 frozen teacher 的 reduction；
2. recurrent student 与 read-only direct control 的 latency ratio；
3. recurrent student 与 control 的 trainable parameter ratio。

第 2、3 项是 engineering tradeoff diagnostic，不单独构成 scientific win；主要 latency gate 仍是 recurrent 相对 teacher 的 predictor-only reduction `>=20%`。

## 6. 冻结 gates

先检查 integrity。若 baseline artifact、shared banks、held-out pairing、finite/convergence、causality 或 student purity 无效，不解释任何 ranking delta。

| gate | 要求 |
|---|---|
| integrity | 复用同一 slate bank、teacher targets、context schedule、held-out bank；无 NaN/OOM/fallback；student 不含 teacher/source params；baseline summary 只读 |
| convergence | recurrent latent-MSE 的 `last10 / first <= 0.8` |
| architecture effect | median ΔSpearman `>= +0.05`，median Δtop-30 `>= +0.10`，且两者各自至少 `12/16` blocks 为正 |
| absolute fidelity | recurrent median Spearman `>=.99`、minimum `>=.95`；median top-30 `>=.95`、minimum `>=.80` |
| logged MSE non-inferiority | recurrent/control ratio `<=1.25` |
| causality | future-action leakage `<=1e-6` |
| predictor latency | recurrent 相对 teacher reduction `>=20%` |

三个 decision 独立报告：

1. `architecture_effect`：完整 integrity 且四个相对 ranking effect 条件都通过；
2. `full_replacement`：recurrent 自身通过 absolute fidelity、MSE non-inferiority、causality 和 predictor latency；这一级不自动等于相对 baseline 有收益；
3. `recurrent_supported_replacement`：`architecture_effect=PASS` 且 `full_replacement=GO`。

任一 gate 失败都保留失败结果，不调 hidden size、cell、seeds、slate、loss、steps 或阈值，也不通过重跑同一个 cell 包装结果。

## 7. 结果应如何解释

如果 `architecture_effect=PASS`，最多可以说：

> 在冻结的 DINO-WM PushT predictor-level contract 下，指定的 shared recurrent native-latent transition student 相比 read-only slate-trained direct student，在 held-out action-candidate ranking 上达到预先指定的 paired improvement。

如果 `full_replacement=GO`，还可以说 recurrent student 在本协议的 absolute fidelity、MSE、causality 和 predictor-only latency 边界内可作为 predictor candidate。

只有两者同时成立，才使用 `recurrent_supported_replacement=GO`。

参数更少或 latency 更低只能说明 engineering tradeoff。若 ranking 没有提升，不能将更小模型称为整体更优；若 ranking 提升，也不能把收益归因于“参数更少”，因为本实验只比较了一个冻结 architecture contrast。

本协议不能支持：

- `encode_obs` 或 observation encoder 加速；
- full CEM、multi-step planner 或 closed-loop task success；
- LeWM/Fast-LeWM 的实证 transfer；
- all-JEPA universality；
- population-level inference；
- native low-bit deployment；
- 超出本 paired architecture cell 的 causal claim。

## 8. 执行与 artifact 约束

本文件只冻结 protocol，**没有提交 PBS 作业**。后续若授权执行：

- 所有模型加载、teacher rollout、训练、评估和 GPU benchmark 必须在 PBS compute allocation 内；
- login node 只允许提交、轻量状态查询和小型控制操作；
- 作业必须每 30 秒将 GPU utilization/VRAM 写入 job log；
- 主要回收 summary JSON、job log、job status 和 GPU telemetry，不要求回收 checkpoints；
- 不做额外 hash，也不重新跑 read-only baseline。

