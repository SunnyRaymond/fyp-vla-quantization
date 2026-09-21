# LeWM PushT：Horizon-Weighted Recurrent Student Transfer 冻结协议

状态：`frozen / protocol only / 未提交作业`

对应冻结文件：[LEWM_RECURRENT_STUDENT_FREEZE.json](LEWM_RECURRENT_STUDENT_FREEZE.json)

## 1. 实验要回答什么

当前最好的 predictor 机制是一个 compact-latent 上的 shared residual recurrent
transition：hidden size 固定为 `256`，每一步都把 student 自己预测的 latent
反馈到下一步，并用

```text
[1/3, 2/3, 1, 4/3, 5/3]
```

对 later horizons 加权。DINO-WM PushT 上的证据表明，这个结构比 direct student
更稳定，但仍未通过严格的 full-replacement gate。本实验把同一机制迁移到 LeWM
的 `192-D` CLS latent 和 official `10-D` packed action token，只回答
predictor-level transfer 是否可行，并在该 gate 后停止。

这里的 transfer 不是复用 DINO-WM 权重。LeWM 的 encoder、autoregressive teacher、
checkpoint、action preprocessing 和 CEM 都保持自己的 official backend；重新训练的
只有轻量 student。`Fast-LeWM` 没有在本协议中运行，也不作为已验证的 comparison。

本协议不做 environment closed-loop，也不运行 official CEM benchmark。

## 2. Backend 与绝对参考

Backend 固定为已有 LeWM PushT iteration-cache case 使用的 official source：

- LeWM source commit：`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`；
- stable-worldmodel CEM commit：`10c26dbd5677083fa31dba69eb738b973845e9a4`；
- dataset：`pusht_expert_train.h5`；
- checkpoint：external `$STABLEWM_HOME/pusht/lewm` asset；
- solver：`horizon=5`、`receding_horizon=5`、`action_block=5`。

Primary control 是 frozen official LeWM teacher，而不是另一个重新训练的 student：

```text
same latent history + same packed action prefix
    ├── official LeWM autoregressive predictor → teacher latent/cost
    └── recurrent student                  → student latent/cost
```

teacher 负责生成 detached dense future latent targets，并在 evaluation 中提供绝对
planner objective。teacher 不会进入 student 的参数或 inference graph。

可以额外运行一个、且仅一个 `uniform recurrent` loss ablation，用来判断 later-horizon
weighting 是否在 LeWM 上仍有独立信号；它不是 primary control，也不能扩展成 width、
step、loss、seed 的多臂 sweep。

## 3. Student 架构：只做 LeWM 必要的最小 adapter

LeWM predictor 的最大 causal history 是 `3`，但权威 interface probe
`24554356.pbs101` 证明 official PushT policy 从 `H=1` 当前 latent 开始，随后 history
随预测增长到 `2/3`。当前 action 是 official raw packed `10-D` token。Student 使用固定
latest-3 adapter：前两步对最早可用 latent 做确定性的 repeat-left-padding，之后正常滚动，
但不复制 teacher 的 6-layer causal Transformer。

```text
[z(t-2), z(t-1), z(t)]  ── shared Linear(576,256) history adapter ─┐
                                                                    +── condition
z(current) ─────────── Linear(192,256) ────────────────────────────┤
a(k) ∈ R^10 ─────────── Linear(10,256) ────────────────────────────┘
                                      ↓
             LayerNorm → Linear(256,1024) → GELU → Linear(1024,256)
                                      ↓ shared residual transition
                 shared LayerNorm → Linear(256,192) residual update
                                      ↓
                         ẑ(k+1) ∈ R^192
                                      ↓
                   append ẑ(k+1), roll history, repeat
```

更具体地，令 `H_k=[z_{k-2},z_{k-1},z_k]`，则每个 horizon 使用同一个 cell：

```text
c_k = history_adapter(flatten(H_k))
u_k = latent_projection(z_k) + action_projection(a_k) + c_k
d_k = shared_transition(u_k)
ẑ(k+1) = z_k + latent_update(LN(latent_projection(z_k) + d_k))
```

`ẑ(k+1)` 会进入下一步的 rolling history；不 detach、不换回 teacher latent，
也不使用 ground-truth future frame。history adapter、latent projection、action
projection、transition 和 output head 都在五个 horizon 间共享。预期 trainable
parameter count 约 `0.78M`，最终以作业 summary 的精确值为准。

Student 不包含：

- LeWM encoder 或 source predictor 参数；
- goal embedding、CEM solver、criterion；
- teacher forcing、rank loss、spatial/token attention；
- 每个 horizon 的独立 cell；
- `encode_obs` 调用。

Goal 只在 student 外部的 official criterion 中使用，因此“goal-free student”并不
意味着 CEM 评价时改变 LeWM 的 goal objective。

## 4. 固定数据与训练 schedule

### 4.1 Context manifest

从 pinned official `pusht_expert_train.h5` 预先生成 episode-disjoint manifest：

- train target：`256` 个 context anchors；
- held-out：`8` 个 context anchors；
- 每个 anchor 保存 episode ID、timestep、当前 frame ID，以及其后连续 25 个 primitive
  actions 的范围；
- manifest 由显式 CPU generator 生成，并同时用于 target generation、训练、held-out
  和 latency timing；
- 如果 dataset 不能提供所需的 episode-disjoint anchors，preflight 失败，不复制
  context、不静默改变 split。

这一步只准备当前 latent 的索引与小型 manifest；encoder 运行和 teacher
target 生成必须在 PBS compute allocation 内完成。

### 4.2 Query slate

每个 train context 使用四个 query slots，共 `1024` rows：

1. trajectory 中的 `primary_logged` action prefix；
2. 以 logged prefix 为中心、按 frozen seed 采样并 clip 到 official action range 的
   `gaussian_planner_init_a`；
3. 同分布但不同 seed 的 `gaussian_planner_init_b`；
4. 用 frozen official teacher 做一次 `M=64,K=8` CEM resampling 得到的
   `one_step_cem_resample`。

Action range 在首个训练作业前固定为 official PushT `[-1,1]^2`。Gaussian perturbation
standard deviation 不做 sweep，固定为 `sqrt(0.05)=0.22360679775`，直接对应已冻结的
`variance_floor=0.05`。

每个 action prefix 是 `25×2` primitive actions；每连续 5 个二维 actions 打成一个
`10-D` token，因此 planner/predictor 输入为 `5×10` action tokens。slate 在训练前
一次性生成；student 不能参与 slate
   生成、重排或筛选。

### 4.3 Teacher targets 与 loss

对每个 query row，official LeWM teacher 按自己的 autoregressive history interface
   rollout 五步，得到 detached dense targets：

```text
z_teacher(t+1), ..., z_teacher(t+5) ∈ R^192
```

student 同样 rollout 五步，但使用自己的 predicted-latent feedback。令 `m_h` 表示
第 `h` 个 horizon 的 `192-D` latent MSE，则 treatment loss 固定为：

```text
L_weighted = (1/5) * Σ_h w_h m_h
w = [1/3, 2/3, 1, 4/3, 5/3]
```

权重均值为 `1`，因此没有通过增大整体 loss scale 来制造收益。固定训练预算为：

| 项目 | 冻结值 |
|---|---:|
| updates | `1500` |
| snapshots | `500 / 1000 / 1500` |
| contexts/update | `8` |
| effective query rows/update | `32` |
| optimizer | `AdamW` |
| learning rate | `3e-4` |
| weight decay | `0.01` |
| initialization seed | `20300901` |
| training seed | `20300902` |

formal result 固定使用 `step1500`；`500/1000` 只能作为预注册 descriptive snapshots，
不能看完结果后挑选。

## 5. Stage A：predictor-level paired evaluation

### 5.1 Held-out pairing

- `8` 个 held-out contexts；
- 两个 fresh action-prefix seeds：`20300907`、`20300908`；
- 共 `16` 个 paired blocks；
- 每个 block `300` 个 candidate prefixes；
- `top-30` 与 objective ranking 使用同一个 fixed candidate order；
- 统计单位是 block，不能把 300 candidates 当作 300 个独立 replicates。

teacher 与 student 对同一 context、同一 latent history、同一 action candidates、
同一 goal 和同一 official criterion 做评价。student 的 predictor 本身不接 goal，
goal 只给外部 criterion。

### 5.2 必须报告的 metrics

每个 block、每个 snapshot 报告：

- official LeWM terminal objective Spearman；
- top-30 overlap；
- teacher-relative latent MSE；
- 五个 horizon 的 MSE 与 cosine；
- held-out logged-action latent fidelity；
- predictor latency。

另外执行 prefix causality test：保持 prefix 长度 `1/2/3/4` 不变，只改后续 action
tokens，比较 unchanged prefix 输出。最大 future-action leakage 必须不超过 `1e-6`。

Latency boundary 固定为：

```text
cached official H=1 current latent + five normalized packed action tokens → five-step predictor rollout
```

batch `300`，warmup `3`，technical repeats `10`，前后 CUDA synchronize；不包含
encoder、goal encoding、CEM 或 environment。

### 5.3 Predictor gates

先检查 integrity 和 convergence；任意一项失败时，不解释 ranking delta。primary
`predictor_feasibility` 要求：

| Gate | 预注册要求 |
|---|---:|
| training `last10 / first` | `≤ 0.80` |
| median Spearman | `≥ 0.95` |
| minimum Spearman | `≥ 0.80` |
| median top-30 overlap | `≥ 0.75` |
| minimum top-30 overlap | `≥ 0.50` |
| median relative latent MSE | `≤ 0.25` |
| positive Spearman blocks | `≥ 12/16` |
| positive top-30 blocks | `≥ 12/16` |
| predictor-only latency reduction | `≥ 20%` |

更严格的 replacement diagnostic 另外报告：Spearman median/minimum `≥.99/.95`，
top-30 median/minimum `≥.95/.80`。严格 gate 不通过不影响是否进入 Stage B；但它决定
能否称为 stronger predictor replacement。

本轮实验到 primary predictor feasibility gate 即停止。按用户指令，不提交、
不执行 Stage B official CEM；因此本轮也不作 planner-level viability claim。

## 6. Stage B：接入真实 official LeWM PushT CEM

**本轮禁用。** 以下内容仅保留为未来可恢复的 protocol reference，不属于本轮执行或
验收范围。

Stage B 不重新设计 planner。使用 pinned official stable-worldmodel CEM solver，
只把 candidate-cost predictor call 替换为训练好的 student wrapper；以下完全保留：

- `horizon=5`、`receding_horizon=5`、`action_block=5`；
- `M=300`、`30` iterations、`topk=30`、`var_scale=1.0`；
- action sampling、candidate RNG、criterion、top-k、mean/variance updates；
- official PushT preprocessing 和 fixed observation/goal。

固定 observation IDs 为 `pusht_obs_00`、`pusht_obs_01`。behavior 使用 seeds
`20300910/20300911`；timing 使用 `20300912...20300916`。每次 paired run 同时执行
teacher 与 student，并保存 30 个 CEM iterations 的 trace。

### 6.1 Stage B 必须汇报

- teacher/student returned first action 与 final action；
- 每个 iteration 的 candidate-cost Spearman；
- 每个 iteration 的 elite-set Jaccard/overlap；
- 每个 iteration 的 mean drift 与 variance drift；
- 完整 CEM solve latency；
- peak memory、NaN/OOM/fallback 状态。

Full planner latency boundary 是从 official `solve` 进入到返回 action，包含 30 次
predictor、sampling、criterion、top-k/update 和 solver overhead；不包含 environment
interaction。

### 6.2 Full CEM viability gates

`full_cem_viability=GO` 需要同时满足：

- teacher/student 使用相同 observations、goal、RNG、solver config 和 trace contract；
- first-action median normalized L2 `≤ 0.15`；
- first-action maximum absolute difference `≤ 0.25`；
- terminal elite Jaccard median `≥ 0.50`；
- full-solve latency reduction `≥ 20%`；
- peak-memory ratio `≤ 1.10`；
- 所有输出 finite，无 silent fallback。

candidate/elite/mean-variance drift 即使 gate 失败也必须报告。该阶段只建立固定
official CEM 边界上的 planner viability，不建立 closed-loop task success。

## 7. 结果解释边界

如果 Stage A 通过，最多可以说：

> 在 pinned official LeWM PushT 的 frozen predictor-level protocol 下，compact
> `192-D` latent 上的 h256 horizon-weighted recurrent student 达到预先注册的
> ranking、latent-fidelity、causality 和 predictor-latency feasibility gates。

如果 Stage B 也通过，才可以进一步说：

> 在固定 official LeWM PushT CEM observations、seeds 和 solver budget 下，student
> 满足预先注册的 first-action、terminal-elite 和 full-solve latency viability gates。

本协议不能支持：

- `encode_obs` 加速或 end-to-end environment success；
- LeWM 与 DINO-WM student 权重可直接复用；
- Fast-LeWM 已经被复现或优于/劣于本方案；
- 所有 JEPA-style world models 的普适性；
- 用 16 个 nested blocks 做 population-level inference；
- 看完结果后选择 step500/1000、调整 weights、hidden size、steps、seeds 或 candidate bank。

## 8. 执行约束

本文件只冻结设计，**不要在创建 freeze 时提交作业**。正式执行时：

- checkpoint、dataset、encoder、teacher rollout、训练、评估和 GPU benchmark 必须在
  PBS compute allocation 内完成；
- login node 只做提交、qstat 和轻量控制；
- 不在 login node 下载、传输、解压、编译、安装依赖、加载模型或 benchmark；
- GPU 作业每 30 秒写 utilization/VRAM telemetry 到 job log；
- 只回收 summary、job log、job status 和 telemetry 等小型 artifacts，不回收 checkpoint；
- 环境 closed-loop 不在本 protocol 中执行。
