# DINO-WM PushT：Context Coverage × Action-Query Coverage 实验结果

## 结论

**冻结的 full compiler gate：NO-GO。**

但实验明确支持一个更窄、也更重要的机制结论：

> 在相同的 WIDE context coverage 下，把训练 action prefix 从纯 logged actions 扩展为 `50% logged + 25% Gaussian planner-init + 25% one-step CEM elite-resample`，会稳定提升 held-out planner candidate ranking。

这不是“多看一些 context”造成的假象，因为 primary comparison 是配对的 `WIDE-QUERY − WIDE-LOGGED`：两者使用相同的 128 个 training contexts、相同初始化、optimizer、500 steps、batch size 与 held-out candidates，只改变 action-prefix distribution。

然而，`WIDE-QUERY` 仍没有达到预注册的极严格 absolute fidelity 门槛。因此当前结果支持“方向正确、query coverage 是有效杠杆”，但不支持“现有 student 已经可以无损替代 teacher predictor”。

## 实验身份与执行状态

- PBS job：`24380917.pbs101`
- 状态：正常完成，`Exit_status=0`，`final_exit_code=0`
- GPU：A100-SXM4-40GB
- walltime：约 9 分 48 秒
- telemetry：20 个样本；median GPU utilization `99.5%`（约 100%）；峰值显存 `26,369 MiB`
- 训练：四臂各 500 steps，batch 32，AdamW，learning rate `3e-4`
- held-out：8 contexts × 2 fresh seeds = 16 paired blocks；每 block 300 candidates
- QUERY mixture：16 logged + 8 Gaussian + 8 one-step CEM rows / batch
- one-step CEM：每个 training context 固定 `M=64`、`K=8`、variance floor `0.05`，训练前预计算一次

## 四臂结果

| Arm | Median Spearman | Minimum Spearman | Median top-30 overlap | Minimum top-30 | Teacher-relative MSE |
|---|---:|---:|---:|---:|---:|
| NARROW-LOGGED | 0.4484 | 0.2416 | 0.3000 | 0.1333 | 0.24668 |
| WIDE-LOGGED | 0.7116 | 0.5902 | 0.4500 | 0.3333 | 0.06715 |
| NARROW-QUERY | 0.5618 | 0.4331 | 0.3333 | 0.2333 | 0.20736 |
| WIDE-QUERY | **0.8810** | **0.8004** | **0.6167** | **0.4667** | **0.06624** |

从 arm-level 中位数看，完整的 `NARROW-LOGGED → WIDE-QUERY` 路径把 Spearman 从 `0.448` 提高到 `0.881`，top-30 overlap 从 `0.300` 提高到 `0.617`。不过机制归因应使用下面的 paired effects，而不是简单相减 arm medians。

## 收益来自哪里

### 1. Context coverage 是第一层大收益

`WIDE-LOGGED − NARROW-LOGGED`：

- median ΔSpearman：`+0.1891`
- median Δtop-30 overlap：`+0.1667`
- Spearman 正向：`16/16` blocks
- top-30 正向：`16/16` blocks

这说明之前 grounded-prefix 实验中观察到的 coverage effect 是可重复的：仅靠 8 个 narrow contexts 很难学到可泛化的 action-conditioned dynamics；扩大到 128 contexts 后，ranking 明显改善。

### 2. Action-query coverage 在 WIDE context 上带来稳定的额外收益

Primary comparison `WIDE-QUERY − WIDE-LOGGED`：

- median ΔSpearman：`+0.1544`，门槛 `≥ +0.05`
- median Δtop-30 overlap：`+0.1333`，门槛 `≥ +0.10`
- Spearman 正向：`16/16`，门槛 `≥ 12/16`
- top-30 正向：`14/16`，门槛 `≥ 12/16`
- **primary action-query effect：PASS**

这是本实验最关键的结果。student 只在 logged trajectories 上蒸馏时，即使 context 已经足够宽，仍会面对 planner query distribution shift：CEM 会询问大量数据集中没有出现过的 counterfactual action prefixes。加入 planner-init Gaussian 和 one-step CEM elite-resample 后，student 在这些 query 上得到 teacher 的 dense latent targets，因此候选排序显著改善。

### 3. QUERY 与 WIDE context 有互补趋势，但本实验只把它当 diagnostic

交互项 `(WIDE-QUERY − NARROW-QUERY) − (WIDE-LOGGED − NARROW-LOGGED)`：

- median Spearman interaction：`+0.0989`
- median top-30 interaction：`+0.0500`
- 正向 blocks：Spearman `11/16`，top-30 `13/16`

这表明 query coverage 在更宽的 state/context support 上更有效，符合“先覆盖世界状态，再覆盖 planner 会问的 action”的直觉。但 interaction 未预注册为 primary gate，不能把它表述成已经建立的统计交互结论。

## 为什么仍然是 full-gate NO-GO

`WIDE-QUERY` 的绝对结果虽然显著优于 `WIDE-LOGGED`，但仍低于冻结门槛：

| Absolute gate | 实际 | 门槛 | 判定 |
|---|---:|---:|---|
| Median Spearman | 0.8810 | ≥ 0.99 | FAIL |
| Minimum Spearman | 0.8004 | ≥ 0.95 | FAIL |
| Median top-30 | 0.6167 | ≥ 0.95 | FAIL |
| Minimum top-30 | 0.4667 | ≥ 0.80 | FAIL |

所以 QUERY 解决了明显的一部分 distribution shift，却没有把当前 `hidden_dim=128`、500-step student 推到“近乎无损替代 teacher”的程度。下一步不应否定 query-coverage 方向，也不能把这次结果包装为完整成功；更准确的判断是：

- **机制 hypothesis：GO**——planner-query-like coverage 确实改善 candidate ranking。
- **当前 frozen compiler recipe：NO-GO**——absolute ranking fidelity 仍不足。

## 其他 gates

### Latent non-inferiority：PASS

- WIDE-LOGGED teacher-relative MSE：`0.0671545`
- WIDE-QUERY teacher-relative MSE：`0.0662387`
- ratio：`0.98636`，门槛 `≤ 1.25`

QUERY 没有通过牺牲 logged-action latent fidelity 来换 ranking；相反，teacher-relative MSE 略低约 `1.36%`。

### Causality：PASS

四个 arms 对所有 prefix horizon 的 future-action leakage maximum absolute delta 都为 `0`，低于 `1e-6` 门槛。

### Training convergence：PASS

四臂 last-10/first loss ratios 分别为：

- NARROW-LOGGED：`0.1359`
- WIDE-LOGGED：`0.2879`
- NARROW-QUERY：`0.1750`
- WIDE-QUERY：`0.3746`

全部低于 `0.8`。

### Predictor latency：PASS，但只属于 predictor-level claim

- teacher rollout median：`3502.89 ms / 300 candidates`
- WIDE-QUERY student median：`8.77 ms / 300 candidates`
- speedup：`399.3×`
- reduction：`99.75%`

该 timing boundary 从 cached native observation latent 和 normalized action prefix 开始，排除 `encode_obs`、CEM orchestration、environment interaction 和 closed-loop control。因此不能直接解释为 399× control-loop speedup。

## Summary 中的 capacity 误报

原始 `query_coverage_summary.json` 把 `finite_outputs_and_training` 写成 `false`，从而将 capacity 标为 FAIL。这个标签是 runner 的报告实现错误，不是数值异常：runner 误用了一个只接受“tensor mapping”的 `_all_finite` helper 来检查嵌套 scalar result tree，因此只要顶层值是 dict 就会返回 false。

核验结果：

- summary 中所有 serialized numeric values 都是 finite；没有 `NaN` 或 `Infinity`
- 四臂 training convergence 均通过
- 四臂 causality 均通过
- job 正常退出，无 OOM 或 silent fallback

因此修正后的 capacity 是 **PASS**。runner 已改为递归检查嵌套结果；没有重新训练、没有修改任何 metric、seed、loss、threshold 或 scientific decision。即使修正 capacity，overall 仍因 absolute fidelity 失败而保持 **FAIL**。

原始 summary 的 `query_mixture.fractions` 还同时保留了 `elite_perturbation` 与 `one_step_cem_resample` 两个别名；它们指向同一组 25% CEM rows，不是两个独立的 25%。实际执行契约与 batch counts 均为 `16/8/8 = 50%/25%/25%`。runner 的后续输出已只保留 canonical `one_step_cem_resample` 名称。

机器可读修正记录位于 `artifacts/24380917.pbs101/query_coverage_decision_corrected.json`。

## 可以与不可以声称什么

当前可以声称：

> 在冻结的 DINO-WM PushT predictor-level protocol 上，扩大 context coverage 与扩大 planner-query action coverage 都能改善 post-hoc NativeDinoPrefixStudent 的 held-out candidate ranking；在固定 WIDE context coverage 后，QUERY mixture 仍带来稳定、通过预注册 effect gates 的额外收益。

当前不可以声称：

- student 已达到 near-lossless teacher replacement
- closed-loop task success 已保持
- full multi-step CEM latency 或 control frequency 获得 399× 提升
- 结果已经迁移到 LeWM 或所有 JEPA-style world models
- one-step CEM mixture 就是最终最佳训练分布

## 下一步建议

下一步应保留已经验证有效的 `WIDE + QUERY` 训练分布，把问题从“是否要 planner-query coverage”转成“如何提高 student capacity/optimization 而不丢掉 speedup”。最小且可证伪的下一阶段是：

1. 固定同一 WIDE+QUERY bank、held-out blocks 与全部 seeds；
2. 只比较 student capacity/parameterization，例如 hidden width 或更贴近 teacher token mixing 的轻量结构；
3. primary target 仍是 absolute Spearman/top-30，而不是继续放宽门槛；
4. 保留 WIDE-LOGGED 作为 distribution baseline，证明进一步收益不是单纯增加参数。

不建议下一步立即做 closed-loop，也不建议在看到本次结果后重调 QUERY mixture；当前主要短板已经从 action-distribution coverage 转移到 absolute approximation capacity。

## 产物

- 原始 summary：`artifacts/24380917.pbs101/query_coverage_summary.json`
- 修正 decision：`artifacts/24380917.pbs101/query_coverage_decision_corrected.json`
- job log/status：`artifacts/24380917.pbs101/job.log`、`job_status.txt`
- GPU telemetry：`artifacts/24380917.pbs101/gpu_info.csv`、`gpu_usage.csv`、`gpu_telemetry.jsonl`
- 未取回任何 student checkpoint
