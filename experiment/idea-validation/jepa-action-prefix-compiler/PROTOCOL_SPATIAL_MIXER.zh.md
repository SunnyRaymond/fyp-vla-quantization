# DINO-WM PushT：Shared Spatial Token-Mixer Recurrent Predictor 冻结协议

状态：`frozen / protocol only / 未提交作业`

对应冻结文件：[SPATIAL_MIXER_FREEZE.json](SPATIAL_MIXER_FREEZE.json)

## 1. 这项实验要回答什么

当前最好的 predictor candidate 是 `horizon-weighted recurrent student`。它已经在保持严格 action-prefix causality、logged-action MSE non-inferiority 和 predictor-only latency reduction 的同时，给出稳定但不足以过 gate 的 ranking 改善。更关键的是，它对 visual patch 只做 projected-token mean pooling，没有 token-to-token interaction。

本实验只检验一个剩余的结构假设：

> 在相同的 recurrent feedback 和 later-horizon weighting 下，一个共享的轻量 visual-token mixer 是否能补上 predictor 与 frozen teacher 之间的 patch-interaction gap，从而改善 held-out candidate ranking？

唯一新变量是一个 shared、4-head、pre-norm `MultiheadAttention` residual。所有 observation encoder、cached latent、teacher targets、action slate、schedule、seed、loss、training budget 和 held-out protocol 都保持不变。

实验边界严格限定为 `action-conditioned predictor`。`encode_obs`、observation encoder、CEM execution、environment interaction、closed-loop success 和 LeWM/Fast-LeWM transfer 不在范围内。

## 2. Control：只读的 horizon-weighted recurrent baseline

Control 使用已经完成的 job `24510395.pbs101`：

- 本地 summary：`artifacts/24510395.pbs101/horizon_weighted_summary.json`；
- arm：`horizon_weighted_recurrent_control`；
- snapshot：`step1500`；
- architecture：`RecurrentNativeLatentTransitionStudent`，hidden `256`；
- loss：五个 horizon 的 dense native-latent MSE，权重 `[1/3, 2/3, 1, 4/3, 5/3]`；
- held-out：2 个 action-prefix seeds × 8 个 contexts，共 16 个 paired blocks。

Control 的 summary、per-block metrics、candidate bank 和结果文档都是 read-only reference。不得重新训练 control、加载 control checkpoint、修改 summary，或用新的随机运行替代它。Treatment 使用相同 initialization seed `99` 和完全相同的 data/schedule seeds，但由于新增 attention，初始 parameter tensor 不要求逐元素相同；报告为 architecture contrast。

Control 的 step-1500 参考值：

| 指标 | horizon-weighted recurrent control |
|---|---:|
| median Spearman | `0.970596` |
| minimum Spearman | `0.894798` |
| median top-30 overlap | `0.833333` |
| minimum top-30 overlap | `0.600000` |

## 3. Treatment：一个 shared visual-token mixer

Treatment 类名固定为 `SpatialMixerRecurrentNativeLatentTransitionStudent`，hidden size 固定为 `256`。它继承 horizon-weighted recurrent student 的 recurrent cell、predicted-latent feedback 和 horizon weights，只增加一个 shared token-mixing module。

### 3.1 每一步的计算

第 `k` 步只接收当前 predicted state 和当前 action：

```text
(visual_k, proprio_k, packed_action_k)
```

其中 `visual_k` 形状为 `[B, patches, 384]`，`proprio_k` 为 `[B,10]`，packed action token 为 `[B,10]`。先逐 patch 投影：

```text
v_k = visual_in(visual_k)       # [B, patches, 256]
p_k = proprio_in(proprio_k)     # [B, 256]
a_k = action_proj(action_k)     # [B, 256]
```

唯一新增模块是一个在五个 recurrent steps 之间共享的 attention：

```text
m_k = v_k + MHA(LN(v_k), LN(v_k), LN(v_k))
```

固定配置为：4 heads、embedding size 256、dropout 0、`batch_first=True`、不返回 attention weights、无第二个 attention layer、无 FFN subblock。它只在 visual patch 轴上混合 token，不把 proprio 或 action 当作 attention token。

随后仍使用原来的 summary 和 transition：

```text
s_k = mean(m_k, dim=patch) + p_k + a_k
r_k = Linear(1024,256)(GELU(Linear(256,1024)(LayerNorm(s_k))))

visual_{k+1}  = visual_k + visual_out(LayerNorm(m_k + r_k[:,None,:]))
proprio_{k+1} = proprio_k + proprio_out(LayerNorm(p_k + r_k))
```

`m_k`、transition、visual/proprio output heads 都是 shared across `k=0..4`。`visual_{k+1}` 和 `proprio_{k+1}` 是 student 自己的预测，不能 detach，也不能在下一步替换为 teacher latent。

对于只有一个 latent vector 的 compact-vector model，attention 退化为近似 identity-like 的单 token mixer；这只是机制上的 portability 预期，本实验没有验证 compact-vector、LeWM、Fast-LeWM 或 all-JEPA transfer。

### 3.2 训练语义

Treatment 使用 frozen official teacher 产生的 detached dense native visual/proprio targets，但 rollout 始终 free-running：

- 不使用 teacher forcing；
- 不把 ground-truth future frame 或 teacher future latent 喂回 recurrent cell；
- 不加入 rank loss、listwise KL、goal loss 或 self-consistency loss；
- 只使用 horizon-weighted dense latent MSE。

每个 horizon 的基础项仍为：

```text
mse_h = 0.5 * (visual_MSE_h + proprio_MSE_h)
loss = (1/5) * sum_h ((h+1)/3) * mse_h
```

因此本实验比较的是一个 spatial/token architecture change，而不是 architecture 加额外 planner supervision。

## 4. 两边固定不变的训练资源

Treatment 必须复用 horizon-weighted recurrent recipe：

| 项目 | 固定值 |
|---|---|
| train contexts | 256（32 episodes × 8 contexts） |
| 每 context queries | 4 |
| query rows | 1024 |
| 每 update contexts | 8 |
| 每 update query rows | 32 |
| training updates | 1500 |
| snapshots | 500 / 1000 / 1500 |
| optimizer | AdamW |
| learning rate | `3e-4` |
| weight decay | `0.01` |
| horizon weights | `[1/3, 2/3, 1, 4/3, 5/3]` |
| rank loss | `0` |

四个 query slots、`M=64,K=8` 的 one-step CEM proposal、teacher target rows、context schedule 和 candidate bank 全部复用现有 frozen artifacts。student output 不能参与生成 action bank。

### 4.1 Seeds 与 schedule

| 用途 | seed |
|---|---:|
| initialization | `99` |
| training | `20260930` |
| context schedule | `20260931` |
| action slate | `20260932` |
| CEM proposal | `20260933` |
| held-out action prefixes | `20264925`, `20264926` |
| timing candidates | `20270925` |

Context schedule 仍为每 32 updates 对 `0..255` 做 CPU permutation，每个 update 取连续 8 个 ordinal。所有 schedule、slate、held-out candidates 和 timing candidates 在 GPU 计算前固定生成。

## 5. Held-out paired evaluation

Held-out protocol 完全继承现有 recurrent/query-slate protocol：

- 8 个 held-out contexts；
- fresh action-prefix seeds `20264925`、`20264926`；
- 每个 block 300 candidates；
- 共 16 个 paired blocks；
- pairing key：`(action_prefix_seed, heldout_context_index)`；
- block 是统计单位，300 candidates 不是 300 个独立 replicate。

正式比较固定为：

```text
spatial-mixer treatment − read-only horizon-weighted recurrent control
```

formal snapshot 固定为 `step1500`；step500/1000/1500 只作为预先规定的 descriptive snapshots，不得结果依赖地改选。

需要报告 objective Spearman、top-30 overlap、teacher-relative native-latent MSE、per-horizon relative MSE 和 per-horizon cosine。

### 5.1 Latent fidelity 与 causality

对 held-out logged action prefix，报告 treatment/control 的 teacher-relative dense native-latent MSE ratio，固定上限为 `1.25`。

对 unchanged prefix lengths `1,2,3,4`，只改变之后的 action tokens，比较 unchanged prefix 内的输出。最大 future-action leakage 必须不超过 `1e-6`；任一 prefix case 失败都使 integrity/causality contract 失败，不能解释 ranking delta。

### 5.2 Predictor-only latency 与参数量

Latency 边界固定为：

```text
cached native observation latent + normalized action prefix → predictor rollout
```

不包括 `encode_obs`、CEM、environment 或 closed-loop control。固定 batch size `300`、warmup `3` 次、technical repeats `10` 次，并在 CUDA 前后同步。

报告：

1. treatment 相对 frozen teacher 的 predictor-only reduction；
2. treatment / read-only horizon-weighted recurrent 的 latency ratio；
3. treatment / baseline 的 trainable parameter ratio。

后两项是 engineering diagnostics。attention 增加的成本必须单独测量，不能因为参数少/多就推断 ranking 原因。主要 latency gate 仍是 treatment 相对 teacher reduction `>=20%`。

## 6. 冻结 gates 与 decision levels

先检查 integrity、finite training、convergence 和 causality。若 baseline、shared banks、held-out pairing 或 student purity 无效，不解释 ranking delta。

| gate | 要求 |
|---|---|
| integrity | 复用同一 context schedule、slate、teacher targets、held-out bank；student 不调用 `encode_obs`、不接 goal、不含 teacher/source params；只有一个 shared mixer；无 NaN/OOM/fallback |
| convergence | treatment horizon-weighted loss 的 `last10 / first <= 0.8` |
| architecture effect | median ΔSpearman `>= +0.05`，median Δtop-30 `>= +0.10`，且两者各自至少 `12/16` blocks 为正 |
| absolute fidelity | treatment median Spearman `>=.99`、minimum `>=.95`；median top-30 `>=.95`、minimum `>=.80` |
| logged MSE non-inferiority | treatment/control ratio `<=1.25` |
| causality | future-action leakage `<=1e-6` |
| predictor latency | treatment 相对 teacher reduction `>=20%` |

三个层次独立报告：

1. `architecture_effect`：integrity、convergence 和四个相对 ranking effect 条件全部通过；
2. `full_replacement`：treatment 自身通过 absolute fidelity、MSE、causality 和 predictor-only latency；
3. `spatial_mixer_supported_replacement`：前两者同时成立。

任一 gate 失败都保留失败结果，不调 mixer、hidden size、horizon weights、seed、schedule、loss、steps 或阈值，也不改用 step1000 包装结果。

## 7. Claim boundary

若 `architecture_effect=PASS`，最多可以说：

> 在冻结的 DINO-WM PushT predictor-level contract 下，指定的 shared 4-head pre-norm visual-token mixer 相比 read-only horizon-weighted recurrent predictor，在 held-out action-candidate ranking 上达到预先指定的 paired improvement。

若 `full_replacement=GO`，还可以说该 spatial-mixer recurrent student 在本协议的 fidelity、MSE、causality 和 predictor-only latency 边界内是一个 predictor candidate。

本实验不能支持：

- `encode_obs` 或 observation encoder 加速；
- full planner、multi-step CEM 或 closed-loop task success；
- LeWM/Fast-LeWM 的实证 transfer；
- compact-vector latent 已经验证；
- all-JEPA universality；
- 由 16 个 nested blocks 得出的 population-level inference；
- “attention 是唯一因果机制”或“参数量直接造成 ranking gain”的结论；
- native low-bit deployment。

## 8. 执行与 artifact 约束

本文件只冻结 protocol，**没有提交 PBS 作业**。若后续授权执行：

- 所有模型加载、teacher rollout、训练、评估和 GPU benchmark 必须在 PBS compute allocation 内；
- login node 只允许提交、轻量状态查询和小型控制操作；
- 作业每 30 秒将 GPU utilization/VRAM 写入 job log；
- 只回收 summary JSON、job log、job status 和 GPU telemetry，不回收 checkpoints；
- 不做额外 hash，也不重跑 read-only baseline。
