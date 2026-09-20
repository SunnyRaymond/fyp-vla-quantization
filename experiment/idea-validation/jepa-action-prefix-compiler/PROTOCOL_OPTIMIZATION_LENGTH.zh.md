# DINO-WM PushT：WIDE+QUERY Optimization-Length Diagnostic 冻结协议

## 一句话目标

本轮只回答一个问题：在完全相同的 `WIDE+QUERY` student、128 个 training contexts、50/25/25 action-query bank、初始化与 optimizer 下，把连续训练从 `500` steps 延长到 `1500` steps，是否会 materially improve held-out planner-candidate ranking？

这是一个 optimization-length diagnostic，不是新的 model、data、query mixture 或 planner 方案。student 仍然是 `hidden=128` 的 `NativeDinoPrefixStudent`，只训练一个 `WIDE-QUERY` arm。实验不做 closed-loop、LeWM transfer 或 multi-step CEM integration。

## 与上一轮的关系

本协议继承 [QUERY_COVERAGE_FREEZE.json](./QUERY_COVERAGE_FREEZE.json) 的 frozen DINO-WM PushT、manifest、student architecture、action bank、seed、optimizer、held-out block 与 latency boundary。上一轮 `WIDE-QUERY` 的 500-step 结果只作为 non-gated historical sanity reference；它不能被用来 early stop、retune、换 seed 或放宽门槛。

上一轮 runner 的 context RNG 有一个必须保留的调用顺序：它先生成 500 steps 的 NARROW schedule，再生成 500 steps 的 WIDE schedule。若新 run 直接为 1500 steps 先生成 NARROW schedule，会改变旧 WIDE schedule 的 RNG state，因此不能复现旧的前 500 个 WIDE context indices。

## Frozen treatment

| 项目 | 固定定义 |
|---|---|
| Student | 一个 `NativeDinoPrefixStudent`，`hidden_dim=128`，`H=5`，`frameskip=5`；输入 action tensor 为 `[B,5,10]`，其中 `10=5×2`，即每个 token 打包 5 个 primitive 2-D actions |
| Training contexts | frozen CPU manifest 的全部 `128` 个 train contexts，32 episodes × 4 contexts |
| Action mixture | `50% logged + 25% Gaussian planner-init + 25% one-step CEM elite-resample` |
| 每 batch rows | `16 logged + 8 Gaussian + 8 CEM = 32` |
| One-step CEM | 每个 training context `M=64`、`K=8`、variance floor `0.05`，训练前每 context 只预计算 1 个 resample |
| Target | frozen teacher rollout 的 detached dense native-latent MSE |
| Optimizer | AdamW，learning rate `3e-4`，batch size `32` |
| 训练长度 | 连续 `1500` 次 optimizer updates |
| Student 输入 | cached native DINO observation latent + 五个 normalized action tokens；goal 不进入 student |
| 禁止项 | rank loss、listwise KL、goal loss、self-consistency loss、真实 future-frame target |

训练只消费 train manifest。held-out episode、observation、action、goal、seed 和候选 tensor 不得进入 context schedule、CEM bank 或任何 training target。

## RNG 与前 500 steps 的严格复现规则

### Context schedule

固定使用 `Python random.Random(20260925)`。新 runner 必须按下列逻辑生成 WIDE context indices：

```text
rng = random.Random(20260925)

# compatibility burn-in：模拟旧 runner 已经生成的 500-step NARROW schedule
for step in range(500):
    for batch_position in range(32):
        rng.randrange(8)       # 对应前两个 train episodes 的 8 个 NARROW contexts

# 新 run 的唯一 training pool 是 WIDE；从 burn-in 后继续生成 1500 steps
for step in range(1500):
    for batch_position in range(32):
        wide_index = rng.randrange(128)
```

这里的 `8` 必须对应 manifest 中按原顺序排列的前两个 train episodes 的 8 个 contexts，`128` 必须对应全部 train examples。burn-in 的 draws 可以即时丢弃，但不能省略；burn-in 与第一个 WIDE draw 之间不能插入任何额外 RNG 调用。

因此，新 run 的 `step 1..500` WIDE context-index matrix 必须等于旧 `run_dino_pusht_query_coverage.py` 在生成完旧 500-step NARROW schedule 后生成的 500-step WIDE schedule。这是 schedule compatibility contract，不是结果相似性的描述。

禁止以下做法：

- 为 WIDE 单独重新 seed；
- 按 1500 steps 先生成一份 NARROW schedule 再开始 WIDE；
- 使用 CUDA RNG 生成 context index；
- 根据 training loss 或 held-out 结果改变 schedule。

### Action-query role schedule

用 CPU `torch.Generator.manual_seed(20260927)`，固定 role multiset：

```text
[0] * 16 + [1] * 8 + [2] * 8
```

其中 `0=logged`、`1=Gaussian planner-init`、`2=one-step CEM`。每个 training step 对这 32 个 role 做一次 CPU `randperm(32)`；共生成 1500 个 permutation。前 500 个 permutation 必须是上一轮 500-step role schedule 的 prefix。

`action_query_seed=20260926`，第 `i` 个 training context 使用 `20260926 + 1009 × i` 的独立 CPU generator：先采 1 个 Gaussian row，再采 `M=64` 个 CEM candidates，再按 frozen teacher terminal objective 选 `K=8`，最后采 1 个 elite-resample。bank 在 step 1 前完成并冻结；1500-step training loop 不重新采样、不重新打分、不读取 student 结果。

## Continuous training 与 snapshot

先用 frozen training seed 创建一个 student template，再把其 initial `state_dict` clone 给唯一的 `WIDE-QUERY` student。训练从 step 1 一直运行到 step 1500：

- 不在 step 500 或 step 1000 重启 optimizer；
- 不重置 AdamW 的 moment、step counter 或 learning-rate state；
- 不重新初始化 student；
- 每个 step 只做一次 `WIDE-QUERY` AdamW update；
- teacher targets detached，student 不调用 `encode_obs`，不读取 goal。

保存三个 training snapshots：

| Step | 文件名 | 训练期间是否评估 held-out |
|---:|---|---|
| 500 | `wide_query_step0500.pt` | 否 |
| 1000 | `wide_query_step1000.pt` | 否 |
| 1500 | `wide_query_step1500.pt` | 否，保存后才开始统一评估 |

每个 snapshot 至少保存 student `state_dict`、当时的 AdamW `optimizer state_dict`、`global_step`、frozen seed/schedule metadata、CPU/CUDA RNG state 与 manifest/parent-freeze identifiers。snapshot 是 audit artifact；训练继续使用同一个 live optimizer state，不从 snapshot reload 继续训练。

训练期间只记录 training loss、finite/shape/device 状态和 GPU telemetry，不计算 held-out ranking、held-out latent MSE、causality 或 latency。这样 step 500、1000、1500 的 held-out comparison 不会改变 training trajectory。

## Held-out：训练结束后一次性统一评估

step 1500 update 完成并保存 final snapshot 后，才生成 held-out candidates。固定使用 frozen manifest 中按原顺序的 8 个 held-out contexts、两个 fresh seeds `20264925` 与 `20264926`，形成：

```text
8 contexts × 2 seeds = 16 paired blocks
```

每 block 有 `300` 个 candidates，`top-k=30`。teacher、step 500、step 1000 和 step 1500 必须使用同一 block 的 identical candidate action tensor 与 goal。candidate tensor 只生成一次，在三个 snapshot evaluation 之间复用。

统一评估顺序可以是：在训练结束后把三个 snapshot 依次载入 evaluation-only model copy，或保留等价的 in-memory state；不得在 step 500/1000 中途做 held-out forward。step 500/1000 的 snapshot replay 不做 optimizer update，也不构成 training restart。

此外，使用 8 个 held-out contexts 的真实 logged action prefixes，计算每个 snapshot 相对 frozen teacher 的 mean native-latent MSE。primary ratio 定义为：

```text
step1500 teacher-relative MSE / step500 teacher-relative MSE
```

该 ratio 不能被 ranking metrics、training loss 或 step1000 结果替代。

## 两个层级的判定

### Level 1：`optimization_materially_helps`

primary effect 是同一 16 个 paired blocks 上的：

```text
step1500 − step500
```

对每个 block 先算 paired delta，再对 16 个 block 取 median，并计算 delta 严格大于 0 的 block 数。只有以下四项全部通过，才标记 `optimization_materially_helps=PASS`：

- median ΔSpearman `≥ +0.05`；
- median Δtop-30 overlap `≥ +0.10`；
- Spearman 正 delta 至少 `12/16` blocks；
- top-30 正 delta 至少 `12/16` blocks。

step 1000 是诊断点，报告 `step1000 − step500` 的同样指标、逐 block delta，以及 step 1500 相对 step 1000 的 trajectory；它不阻断、不提前结束训练，也不触发 retune。不得用 delta of medians 替代 median of paired deltas。

### Level 2：`full_replacement GO/NO-GO`

Level 2 只看 final step 1500 是否满足 predictor-level replacement contract。`GO` 必须同时满足：

1. Capacity/integrity：所有 snapshots/output finite；没有 OOM、NaN、silent fallback；shape/dtype/device 一致；student 没有隐藏调用 `encode_obs`；training last-10/first latent-MSE ratio `≤0.8`。
2. Final absolute ranking fidelity（step 1500 对 frozen teacher）：
   - median Spearman `≥0.99`；
   - minimum Spearman `≥0.95`；
   - median top-30 overlap `≥0.95`；
   - minimum top-30 overlap `≥0.80`。
3. Final teacher-relative latent-MSE ratio：`step1500 / step500 ≤ 1.25`。
4. Final step-1500 causality：future-action leakage maximum absolute difference `≤1e-6`。
5. Final step-1500 predictor-only latency reduction relative to frozen teacher `≥20%`。

两个 level 必须分开报告。`optimization_materially_helps=PASS` 不等于 `full_replacement=GO`；例如 ranking 有明显提升但绝对门槛仍不够时，正确结论是“optimization 有效，但 replacement NO-GO”。反过来，若 final replacement gates 全部通过，即使 attribution level 未通过，也必须如实报告 attribution 未被建立，不能把它写成 optimization effect 已证实。

任何 step 500/1000 held-out 结果都不能 early stop、换 seed、增加/减少 updates、改变 mixture、重算 CEM bank 或放宽 final gates。1500 steps 必须完整执行。

## Causality 与 latency final checks

只对 step 1500 做 final causality 与 latency：

- Causality：固定 `z_t` 和前 `k` 个 action tokens，只改变 `k` 之后的 tokens，检查第 `k` 个输出的 maximum absolute difference；门槛 `≤1e-6`。
- Latency boundary：cached native observation latent + normalized action prefix 到五步 predictor output；包含 CUDA synchronize，batch `300`、warmup `3` 次、technical repeats `10` 次；不包含 `encode_obs`、CEM、environment 或 closed-loop。使用 `1 - median(student_ms)/median(teacher_ms)`，门槛 `≥0.20`。

GPU job 运行期间每 `30` 秒把 utilization 与 VRAM append 到 job artifact log，并保留 job status、GPU info 与 summary。PBS wrapper 必须在重操作前确认 `PBS_JOBID` 非空且 hostname 是获批 compute node；login node 只允许提交、查询和轻量控制。

## Evidence boundary

若 Level 1 通过，本轮最多支持：在 frozen DINO-WM PushT、固定 128-context WIDE+QUERY bank 与固定 student 下，从 step 500 连续优化到 step 1500 对这 16 个 held-out blocks 有预注册的 ranking 改善。

若 Level 2 为 GO，最多支持：final step-1500 student 在该 frozen predictor-level contract 下满足规定的 absolute ranking、teacher-relative MSE、causality 与 predictor-only latency gates。

本协议不能支持：closed-loop task success、multi-step CEM integration、full-plan speedup、LeWM transfer、all-JEPA universality、population-level inference、full-dataset training claim，或“仅凭延长训练已证明模型容量足够”的结论。

## 文件与运行边界

- Freeze：`OPTIMIZATION_LENGTH_FREEZE.json`
- Parent freeze：`QUERY_COVERAGE_FREEZE.json`
- Schedule/action-bank authority：`run_dino_pusht_query_coverage.py`
- 本冻结动作只创建上述 freeze 与 protocol；不运行实验、不提交 Git、不修改其他文件。
- 实验执行前仍须由新的 runner/wrapper 实现并静态核对本协议的 burn-in、连续 optimizer state、snapshot 和 post-training-only evaluation contract。
