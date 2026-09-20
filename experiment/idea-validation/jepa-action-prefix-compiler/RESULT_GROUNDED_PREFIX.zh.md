# DINO-WM PushT Grounded Prefix 三臂实验结果

## 结论

**冻结协议判定为 NO-GO。** 扩大真实 trajectory coverage 明显有效，但将 dense target 从 frozen teacher rollout 换成真实 future-observation latent，并没有带来足够的 planner-ranking 改善，也没有接近 planner-safe absolute fidelity。

本实验支持以下较窄结论：

> 在固定 DINO-WM PushT predictor contract 下，从 2 个 train episodes 扩展到 32 个 episodes 是主要收益来源；Fast-LeWM 式 grounded future-latent supervision 相比同 coverage 的 teacher-rollout supervision 只带来很小的平均 latent-MSE 改善，未带来稳定的 top-k ranking 改善。

因此，当前症结不是 teacher target 与真实 future latent 之间存在很大的 grounding gap。后续若继续，优先问题应是更大的 trajectory/action coverage、student capacity 或 decision-aware objective，而不是单独替换 target source。

## 运行与数据

- CPU preparation job：`24352614.pbs101`，正常完成。
- GPU experiment job：`24359622.pbs101`，runner/final exit 均为 `0`。
- GPU walltime：`00:10:13`。
- GPU：NVIDIA A100-SXM4-40GB。
- frozen manifest：128 个 train segments、32 个 held-out segments，`H=5`、`frame_skip=5`。
- `NARROW-T`：manifest 中前两个 train episodes，共 8 个真实 trajectory segments。
- `WIDE-T`：全部 128 个 train segments，teacher-rollout dense targets。
- `WIDE-GT`：与 `WIDE-T` 完全相同的 segments/actions，真实 future-observation latents。
- 三个 students 使用相同初始化、architecture、optimizer、500 updates 和 batch size 32。

GPU telemetry 共 20 个 30 秒样本；包含启动阶段在内 utilization 范围为 `0–100%`、median 为 `100%`，显存峰值 `25,355 MiB`。未记录 OOM、NaN 或 fallback。

## 冻结 primary gates

| Gate | 冻结要求 | 实测 | 判定 |
|---|---:|---:|---|
| Capacity / finite / causality | finite，leakage `<=1e-6`，loss ratio `<=0.8` | 三臂 leakage 均为 `0`；loss ratios `0.1502 / 0.2317 / 0.2582` | PASS |
| WIDE-GT median Spearman | `>=0.99` | `0.6642` | **FAIL** |
| WIDE-GT minimum Spearman | `>=0.95` | `0.5038` | **FAIL** |
| WIDE-GT median top-30 | `>=0.95` | `0.3167` | **FAIL** |
| WIDE-GT minimum top-30 | `>=0.80` | `0.1333` | **FAIL** |
| Grounding median delta Spearman | `>=+0.05` | `+0.0176` | **FAIL** |
| Grounding median delta top-30 | `>=+0.10` | `0.0000` | **FAIL** |
| Grounding positive blocks | 两项各 `>=3/4` | Spearman `3/4`；top-30 `1/4` | **FAIL** |
| Grounded real-future non-inferiority | WIDE-GT/WIDE-T MSE ratio `<=1.25` | `0.9710` | PASS |
| Predictor latency reduction | `>=20%` | `99.7159%` | PASS |

Frozen overall：**FAIL / NO-GO**。这里的 `FAIL` 是科学判据未通过，不是作业故障。

## Planner-ranking 结果

汇总结果：

| Arm | Median Spearman | Minimum Spearman | Median top-30 | Minimum top-30 |
|---|---:|---:|---:|---:|
| NARROW-T | 0.3904 | 0.3016 | 0.1333 | 0.1000 |
| WIDE-T | 0.6397 | 0.5067 | 0.3167 | 0.1000 |
| WIDE-GT | 0.6642 | 0.5038 | 0.3167 | 0.1333 |

四个 paired blocks：

| Seed | Anchor | NARROW-T Spearman / top-30 | WIDE-T Spearman / top-30 | WIDE-GT Spearman / top-30 |
|---:|---:|---:|---:|---:|
| 20263920 | 0 | 0.3813 / 0.1000 | 0.5209 / 0.2000 | 0.5407 / 0.2000 |
| 20263920 | 1 | 0.3995 / 0.1333 | 0.7585 / 0.5333 | 0.7902 / 0.5000 |
| 20263921 | 0 | 0.4156 / 0.1333 | 0.5067 / 0.1000 | 0.5038 / 0.1333 |
| 20263921 | 1 | 0.3016 / 0.2667 | 0.7722 / 0.4333 | 0.7877 / 0.4333 |

### Coverage effect：WIDE-T − NARROW-T

- median Spearman delta：`+0.2493`，超过 `+0.05`。
- median top-30 delta：`+0.1333`，超过 `+0.10`。
- positive blocks：Spearman `4/4`，top-30 `3/4`。
- held-out real-future MSE ratio：`0.3927`，即 WIDE-T 的平均误差比 NARROW-T 低约 `60.7%`。

该 diagnostic **通过**。它说明旧的 2-anchor/2-episode post-hoc training coverage 确实过窄，而且这是本轮最大的可观测改进来源。

### Grounding effect：WIDE-GT − WIDE-T

- median Spearman delta：`+0.0176`，未达到 `+0.05`。
- median top-30 delta：`0.0000`，未达到 `+0.10`。
- positive blocks：Spearman `3/4`，top-30 仅 `1/4`。
- 一个 block 的 Spearman 下降 `-0.00285`；一个 block 的 top-30 下降 `-0.0333`。

该 primary effect **失败**。Grounded supervision 有轻微的平均 Spearman 方向性收益，但没有转化为稳定的 candidate top-k 改善。

## 真实 future-latent fidelity 与 teacher bias

五个 horizons 的平均结果：

| Predictor | Mean relative MSE | Mean cosine | Terminal MSE | Terminal cosine |
|---|---:|---:|---:|---:|
| Frozen teacher rollout | 0.00993 | 0.99510 | 0.01439 | 0.99299 |
| NARROW-T | 0.21791 | 0.89160 | 0.31311 | 0.85556 |
| WIDE-T | 0.08556 | 0.96040 | 0.12223 | 0.95232 |
| WIDE-GT | 0.08308 | 0.95725 | 0.12573 | 0.94291 |

WIDE-GT/WIDE-T mean-MSE ratio 为 `0.9710`，只改善约 `2.9%`。而且该改善主要来自较早 horizons；terminal MSE 从 `0.12223` 变为 `0.12573`，terminal cosine 从 `0.95232` 变为 `0.94291`，均略差。

更关键的是，frozen teacher rollout 对真实 future latent 本身已经非常接近：mean relative MSE 只有 `0.00993`，terminal cosine 为 `0.99299`。因此没有证据表明 teacher-vs-ground-truth target gap 是当前 compiler 的主要误差来源。

这个 teacher-bias 结论只针对 dataset 中已观测到的 trajectory actions。对随机或 CEM candidate 这类 counterfactual actions，没有真实 future observation 可作为 target，因此本实验不能排除 counterfactual teacher bias。

## Predictor latency

在 frozen predictor-only boundary 下：

- frozen teacher median：`3502.33 ms`；
- WIDE-GT student median：`9.949 ms`；
- speedup：`352.03x`；
- reduction：`99.7159%`。

该结果只覆盖 `cached native z_t + normalized action prefix -> predictor`，不包含 `encode_obs`、CEM、environment interaction 或 closed-loop execution，不能表述为 full-plan speedup。

## 对 Fast-LeWM 借鉴的回答

Fast-LeWM 的真实 future-latent supervision 在本实验中不是无效，但它不是决定性因素：

1. 它保持了 real-future MSE non-inferiority，并有约 `2.9%` 的平均改善；
2. 但 frozen DINO-WM teacher rollout 本身已经非常接近真实 future latent；
3. 因此替换 target source 无法补上 student 相对 teacher 的主要 approximation gap；
4. coverage 扩展带来的 ranking/MSE 改善远大于 grounding target 的收益。

这意味着“直接照搬 Fast-LeWM 的 target grounding”不足以构成当前 post-hoc compiler 的解决方案。Fast-LeWM 的效果很可能依赖更大规模的 trajectory coverage、端到端训练预算及其 architecture/training recipe 的共同作用，而不是仅由 ground-truth target 定义产生。

## 后续决策

本次冻结 recipe 不得通过追加 steps、替换 seeds 或放宽 gates 继续包装为同一实验。若开启新假设，优先级应是：

1. **coverage scaling curve**：在保持 teacher targets 的前提下扩大 episode/segment coverage，确认 ranking 是否继续单调提升；并单独纳入 multi-anchor random/CEM-like action-prefix replay，而不把 observed-trajectory coverage 与 candidate-action coverage 混为一个变量；
2. **capacity/optimization scaling**：只有 coverage curve 尚未饱和时，再比较 student width/depth 或训练预算；
3. **decision-aware objective**：在足够 coverage 的基础上，研究 latent fidelity 与 candidate ranking 的联合约束。

单独继续增加 grounded-target 比例的优先级较低，因为当前 teacher bias 已经很小。

## Claim boundary

可以宣称：

> 在 frozen DINO-WM PushT predictor-level 实验中，扩大真实 trajectory coverage 显著改善 post-hoc action-prefix student；将同 coverage 的 teacher-rollout targets 换成真实 future-observation latents 仅带来很小的平均 latent-MSE 改善，未通过冻结的 planner-ranking effect 与 absolute-fidelity gates。

不能宣称：

- planner-safe replacement；
- CEM integration 或 closed-loop success；
- full planner speedup；
- Fast-LeWM 已被完整复现；
- LeWM transfer；
- observation-prefix composition；
- all-JEPA framework；
- SOTA。

## 证据文件

- `artifacts/24359622.pbs101/grounded_prefix_summary.json`
- `artifacts/24359622.pbs101/job.log`
- `artifacts/24359622.pbs101/job_status.txt`
- `artifacts/24359622.pbs101/gpu_info.csv`
- `artifacts/24359622.pbs101/gpu_usage.csv`
- `artifacts/24359622.pbs101/runner_exit_status.txt`
- `artifacts/24359622.pbs101/final_exit_status.txt`
