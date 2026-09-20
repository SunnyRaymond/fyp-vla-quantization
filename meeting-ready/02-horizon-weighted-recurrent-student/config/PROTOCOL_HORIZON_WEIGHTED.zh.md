# DINO-WM PushT：Horizon-Weighted Recurrent Predictor 冻结协议

状态：`frozen / protocol only / 未提交作业`

对应冻结文件：[HORIZON_WEIGHTED_FREEZE.json](HORIZON_WEIGHTED_FREEZE.json)

## 1. 这项实验要回答什么

前一项 `uniform recurrent` 实验已经显示出一个值得继续验证、但尚未过 gate 的信号：它相对 direct student 的 ranking 改善方向基本一致，却没有达到预先冻结的幅度。更具体地说，16 个 paired blocks 上，Spearman 的 median Δ 为约 `+0.037`，正向 `15/16`；top-30 的 median Δ 为约 `+0.067`，正向 `12/16`。同时，held-out 的 per-horizon relative MSE 从约 `0.012` 向终端 `h5` 的约 `0.038` 上升，而 planner ranking 使用 terminal `h5`。

本实验只验证一个机制假设：uniform recurrent loss 是否把训练资源平均分给了较容易的早期 horizon，以致 terminal rollout 的误差相对更大。唯一新变量是 dense latent-MSE 的 horizon reduction；recurrent cell、student 参数规模、训练数据、teacher target、slate、schedule、seeds 和 inference interface 全部不变。

这不是对旧 recurrent 结果的 retune，也不是挑选 step1000 的重新汇报。旧 uniform recurrent 在 step1000 到 step1500 大致平台期、略有回落，因此正式结果固定使用新的 treatment 在 step1500 的输出；step500/1000 只作预先规定的 descriptive snapshots。

本协议只关注 `action-conditioned predictor`。`encode_obs`、observation encoder、CEM 执行、environment interaction 和 closed-loop control 都不在范围内。

## 2. Control：只读的 uniform recurrent reference

Control 使用已经完成的 job `24503839.pbs101`：

- 本地 summary：`artifacts/24503839.pbs101/recurrent_student_summary.json`；
- architecture：`RecurrentNativeLatentTransitionStudent`，hidden size `256`；
- shared recurrent transition 与 treatment 完全相同；
- loss：所有五个 horizon 等权的 dense native-latent MSE；
- snapshot：`step1500`；
- held-out：2 个 action-prefix seeds × 8 个 contexts，共 16 个 paired blocks。

Control 是 read-only reference：不得重新训练、加载 checkpoint、修改 summary，或用新的随机运行替代它。新 treatment 使用相同的 initialization seed `99`、data/schedule seeds、action slate、teacher target rows 和 held-out bank；这建立的是同一 frozen recipe 下的 paired reference，control 本身仍不重跑。

## 3. Treatment：唯一改变是 horizon weight

Treatment 仍然是 `RecurrentNativeLatentTransitionStudent(h256)`：

```text
(visual_k, proprio_k, packed_action_k)
              ↓ shared residual transition
(visual_{k+1}, proprio_{k+1})
              ↓ feedback into k+1
```

以下均保持与 uniform recurrent control 完全一致：

- visual projection、proprio projection、action projection；
- projected visual patch 的 mean pooling；
- shared LayerNorm–Linear–GELU–Linear transition；
- visual/proprio residual output heads；
- 5-step free-running feedback；
- hidden size `256`、dropout `0`、无 attention、无 per-horizon cell copy；
- 不接收 goal，不调用 `encode_obs`，不包含 teacher/source predictor 参数；
- 不使用 teacher forcing、future ground-truth latent、rank loss 或额外 self-consistency loss。

初始状态仍来自 cached native observation latent 的最后一个 context state。每一步的下一状态都是 student 自己的预测，不能 detach 或替换成 teacher target。

### 3.1 冻结 loss

令 `m_h` 表示第 `h` 个预测 horizon 上的 `0.5 × (visual MSE + proprio MSE)`，与 uniform recurrent baseline 的 modality reduction 完全一致。uniform control 为：

```text
L_uniform = (m_0 + m_1 + m_2 + m_3 + m_4) / 5
```

新 treatment 的权重固定为：

```text
w = [1/3, 2/3, 1, 4/3, 5/3]
L_weighted = (1/5) * (1/3*m_0 + 2/3*m_1 + 1*m_2 + 4/3*m_3 + 5/3*m_4)
```

权重的平均值为 `1`，所以 loss 的整体尺度与 uniform reduction 保持同量级；这里只改变 horizon 间的相对梯度分配。该向量在训练开始前冻结，不能做 weight sweep 或在看结果后调整。

Teacher target 仍是 frozen official DINO-WM predictor 的 detached dense future native visual/proprio latent。训练 rollout 始终 free-running。

## 4. 固定的 data、slate、schedule 和 training budget

Treatment 必须复用 recurrent freeze 的全部共享资源：

| 项目 | 固定值 |
|---|---|
| train contexts | 256 |
| train episodes | 32，每 episode 8 contexts |
| action queries/context | 4 |
| query rows | 1024 |
| contexts/update | 8 |
| effective query rows/update | 32 |
| updates | 1500 |
| snapshots | 500 / 1000 / 1500 |
| optimizer | AdamW |
| learning rate | `3e-4` |
| weight decay | `0.01` |
| initialization seed | `99` |
| training seed | `20260930` |
| context schedule seed | `20260931` |
| action slate seed | `20260932` |
| CEM seed | `20260933` |
| held-out action-prefix seeds | `20264925`, `20264926` |
| timing seed | `20270925` |

四个 query slots 保持原顺序：`primary_logged`、`gaussian_planner_init_a`、`gaussian_planner_init_b`、`one_step_cem_resample`。每个 context 的 slate bank、teacher target rows 和 candidate order 都必须复用既有 artifact；不得重新采样、重排或根据结果替换。

Context schedule 是每 32 updates 对 `0..255` 做一次 CPU permutation，每个 update 取连续 8 个 ordinal。所有 schedule、slate、held-out candidate 和 timing candidate 都在 GPU 计算前用显式 CPU generator 生成。

## 5. Held-out paired evaluation

Held-out 评估沿用 recurrent protocol：

- 8 个 held-out contexts；
- action-prefix seeds `20264925` 和 `20264926`；
- 每个 block 300 candidates；
- 16 个 paired blocks；
- pairing key：`(action_prefix_seed, heldout_context_index)`；
- 统计单位是 block，不能把 300 candidates 当作 300 个独立 replicates。

每个 block 使用相同 context、candidate actions、goal 和 frozen teacher costs。主比较为：

```text
horizon-weighted treatment − read-only uniform recurrent control
```

正式 effect 只看 treatment step1500；step500/1000/1500 都可以记录，但不能挑选最有利 snapshot 作为结论。

### 5.1 Ranking 与 per-horizon diagnostics

每个 paired block 报告 objective Spearman、top-30 overlap、teacher-relative latent MSE、per-horizon relative MSE 和 per-horizon cosine。重点观察 weighted treatment 是否特别改善 terminal `h5`，但只有预先冻结的 aggregate effect gate 能决定 `horizon_weight_effect`。

### 5.2 Logged latent fidelity

在相同 held-out logged action prefixes 上报告：

```text
weighted treatment logged teacher-relative MSE
------------------------------------------------
uniform recurrent control logged teacher-relative MSE
```

ratio 上限固定为 `1.25`。如果该 ratio 通过，只表示 weighted treatment 没有明显损害 logged latent fidelity；它不等价于 ranking effect 通过。

### 5.3 Causality

对 unchanged prefix length `1/2/3/4`，只改变后续 action tokens，比较输出到 unchanged prefix 为止的差异。最大绝对差必须不超过 `1e-6`。如果失败，不能解释任何 ranking delta，因为 horizon weighting 不应破坏 action-prefix causality。

### 5.4 Predictor-only latency

Latency boundary 固定为：

```text
cached native observation latent + normalized action prefix → predictor rollout
```

固定 batch `300`、warmup `3` 次、technical repeats `10` 次并执行 CUDA synchronization。不包括 `encode_obs`、CEM、environment 或 closed-loop control。treatment 与 uniform control 架构和参数完全相同，因此 latency ratio 主要是运行诊断；teacher-relative predictor reduction 的 gate 固定为 `>=20%`。

GPU 作业运行期间每 30 秒将 utilization 与 VRAM 写入 job log。只回收 summary JSON、job log、job status 和 telemetry 等小型 artifacts，不要求回收 checkpoints。

## 6. Frozen gates 与 decision levels

先检查 integrity 和 convergence；任一失败都不解释 ranking delta。

| gate | 要求 |
|---|---|
| integrity | recurrent architecture、初始化 seed、context schedule、slate bank、teacher targets、held-out bank、seeds 和 horizon weights 与 freeze 一致；输出/训练 finite；无 OOM、NaN 或 silent fallback；student 无 teacher/source 参数且不调用 `encode_obs` |
| convergence | weighted training loss 的 `last10 / first <= 0.8` |
| horizon-weight effect | median ΔSpearman `>= +0.05`、median Δtop-30 `>= +0.10`，且两者各自至少 `12/16` blocks 为正 |
| absolute fidelity | weighted treatment median Spearman `>=.99`、minimum `>=.95`；median top-30 `>=.95`、minimum `>=.80` |
| logged MSE non-inferiority | weighted treatment / uniform control `<=1.25` |
| causality | future-action leakage `<=1e-6` |
| predictor latency | weighted treatment 相对 frozen teacher 的 reduction `>=20%` |

三个主要 decision 必须分开报告：

1. `horizon_weight_effect`：integrity、convergence 与四个相对 ranking effect 条件全部通过；
2. `full_replacement`：integrity、convergence、absolute fidelity、logged MSE、causality 和 predictor latency 通过；该 decision 独立于相对 effect；
3. `weighted_recurrent_supported_replacement`：前两个 decision 同时成立。

所有 gate、loss、threshold、seeds 和 snapshots 在观察结果后保持不变。特别不能把 step1000 的结果替代预先指定的 step1500。

## 7. 结果解释边界

若 `horizon_weight_effect=PASS`，最多可以说：

> 在冻结的 DINO-WM PushT action-conditioned predictor contract 下，固定的线性 later-horizon weighting 相比 read-only uniform recurrent MSE control，达到预先指定的 held-out candidate-ranking paired improvement。

若 `full_replacement=GO`，还可以说 weighted recurrent treatment 在本协议的 absolute fidelity、logged MSE、causality 和 predictor-only latency 边界内具备 predictor replacement 条件。只有两者同时成立，才能使用 `weighted_recurrent_supported_replacement=GO`。

若 gate 失败，应描述为该 frozen loss cell 未建立对应证据；不能通过选择 step1000、追加 retune 或重命名旧 uniform 结果来规避。

本协议不能支持：

- `encode_obs` 或 observation encoder 加速；
- full CEM、multi-step planner 或 closed-loop task success；
- LeWM、Fast-LeWM 或所有 JEPA-style world model 的 transfer；
- population-level inference（16 blocks 是本 cell 的 paired evaluation units）；
- 将 loss weighting 的结果外推为超出本 recurrent cell 的 h5 因果机制；
- native low-bit deployment 或端到端 latency claim。

## 8. 执行约束

本文件只冻结 protocol，**没有提交 PBS 作业**。后续若授权执行：

- 模型加载、teacher rollout、训练、评估和 GPU benchmark 必须在 PBS compute allocation 内；
- login node 只允许提交、轻量状态查询和小型控制文件操作；
- 不在 login node 下载、传输、解压、编译、安装依赖或加载模型；
- 不做额外 hash；
- 不重跑 uniform recurrent baseline，也不取回 checkpoints 作为主要结果。
