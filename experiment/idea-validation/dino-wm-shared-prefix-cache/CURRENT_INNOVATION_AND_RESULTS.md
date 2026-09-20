# DINO-WM Shared-Prefix Cache：当前创新与实验结果记录

**记录日期：** 2026-09-18  
**当前状态：** `bounded no-go under the frozen ≥20% latency gate`  
**适用范围：** DINO-WM `wall_single`、CEM `K=300, H=5, topk=30, opt_steps=10`、单张 NVIDIA A100-SXM4-40GB  

> 本文是内部 research record，不是论文定稿或 exhaustive novelty proof。所有结论均限制在下述实现、checkpoint、任务、硬件与冻结门槛内。

## 1. Executive summary

本研究考察 DINO-WM CEM planner 中一个重复计算问题：同一次 planner call 的 300 条 counterfactual action sequences 共享相同 observation/history，但 baseline 会在每个 CEM iteration 对 candidate-batched observation 重复执行 observation encoder。我们实现了 factorized shared-prefix cache：只以 batch size 1 编码 observation prefix，一次缓存，并在每轮扩展到 300 candidates；action encoder、predictor、objective 与 rollout suffix 保持原路径。[E1, E2]

原始 **RankSafe Exact-Prefix Cache (RankSafe-EPC)** 的 bitwise-exact 主张未成立：batch-1 encode 后扩展与 baseline batch-300 encode 在 CUDA 上产生了非 bitwise-exact 的 prefix、rollout 与 objective。因此 exact factorized cache 是 no-go。[E3]

随后冻结的 **Decision-Safe Approximate Factorized Cache** 通过了 planner-decision gate：10/10 independent Wall observations、每个完整 10-round chained CEM 均保持 top-30 set，且每轮 `mu/sigma` 与最终 first action 的 max-absolute difference 均不超过 `1e-5`。[E4]

系统测试测得完整 `CEMPlanner.plan()` latency 从约 `12.345 s` 降至约 `10.573 s`，paired median reduction 为 **14.3485%**，peak-memory ratio 为 **1.00007**。它通过了 decision 与 memory gate，但低于预先冻结的 **20% latency gate**，因此 system gate 失败，未授权 closed-loop。[E4, E5]

当前最准确的结论是：

> Factorized shared-prefix cache 在受测 DINO-WM Wall planner calls 中提供了稳定约 14.35% 的 planner-only latency reduction，未检测到超过冻结阈值的 chained-CEM decision drift；但它不是 bitwise exact，未达到预设 20% 加速门槛，也没有 closed-loop 或完整 MPC wall-clock evidence。

## 2. 当前创新内容

### 2.1 Candidate-axis observation-prefix factorization

一次 CEM call 中，所有 candidates 共享同一 observation/history，只有 action sequence 不同。当前实现将 rollout 拆成：

1. action-independent observation prefix：`encode_obs(obs)`；
2. candidate-specific action-conditioned suffix：`encode_act(action)`、latent construction、predictor rollout 与 objective。

Factorized path 以 batch size 1 计算 observation prefix，并将 encoded tensor 作为 read-only view 扩展到 `K=300`。它不缓存 action-dependent suffix，也不跨 observation 或 planner call 复用 prefix。[E2]

### 2.2 Planner-level safety contract

验证目标不是单纯的 feature similarity 或 rollout reconstruction error，而是 CEM 真正消费的 decision quantities：

- 每轮 top-30 elite set；
- 每轮更新后的 `mu` 与 `sigma`；
- 完整 10-round chained CEM 后返回的 first action；
- 完整 planner-call latency 与 peak memory。

Baseline 与 cache path 从相同 noise schedule seed 开始，但分别递归使用自己上一轮实际产生的 `mu/sigma`。因此测试允许微小数值差异真实传播，而不是每轮强行重置到共同状态。[E4]

### 2.3 Evidence-driven claim revision

该工作的重要方法贡献之一，是把“结构上 action-independent”与“数值上 bitwise exact”分开：

- 结构边界成立，不代表不同 batch shape 的 GPU execution bitwise identical；
- exact gate 失败后，没有放宽原 gate 或把近似结果包装成 exact；
- 新建独立 approximate contract，直接检验 downstream CEM decision stability；
- latency gate 在运行前冻结为 `≥20%`，结果为 14.35% 时按 no-go 停止扩展。

因此，当前可保留的方法名称是 **Decision-Safe Approximate Factorized Cache**；原名称 **RankSafe Exact-Prefix Cache** 不应作为已验证方法主张。

## 3. 实验一：Exact mechanism gate

### 3.1 设置

- PBS job：`23418502.pbs101`
- GPU：NVIDIA A100-SXM4-40GB
- PyTorch：2.2.0+cu121
- 2 个 Wall observations × 2 个 fixed candidate populations，共 4 个 records
- 每个 population：`K=300, H=5, topk=30`
- paths：`baseline`、`iteration_cache`、`factorized_cache`
- equality contract：prefix、rollout、objective、ordering 与 planner outputs 必须 bitwise exact

### 3.2 结果

| 路径 | 结果 | 解释 |
|---|---|---|
| `iteration_cache` | verifier 未报告 mismatch | 对已经 candidate-batched observation 的 encoded prefix 做跨 iteration 复用，在本测试中保持 exact |
| `factorized_cache` | **FAIL** | 4/4 records 的 prefix、rollout、objective 非 bitwise exact |
| `factorized_cache` full ordering | 2/4 mismatch | 极小 objective difference 足以改变部分完整排序 |
| `factorized_cache` top-30 / elite update / selected first action | 4/4 保持 | exact claim 失败，但 planner-relevant outputs 在该小样本中仍稳定 |

最可能的机制解释是 batch-300 与 batch-1 encoder execution 使用了 batch-shape-dependent CUDA numerical paths。该解释与观察一致，但本实验没有对 CUDA kernel 做 component-level profiling，因此应保留为机制推断，而不是已证明因果结论。[E3]

### 3.3 Gate decision

`MECHANISM_PASS` 未产生。Exact factorized-cache claim 被拒绝，原 frozen plan 的 system/closed-loop expansion 停止。[E3]

## 4. 实验二：Approximate decision and system screen

### 4.1 设置

- PBS job：`23435258.pbs101`
- 10 个 independent Wall observations
- 每个 observation：完整 chained CEM，`K=300, H=5, topk=30, opt_steps=10`
- baseline/cache 使用相同 planner-call noise schedule seed
- decision thresholds：
  - 每轮 top-30 set overlap = 1.0；
  - 每轮 `mu` max-abs difference ≤ `1e-5`；
  - 每轮 `sigma` max-abs difference ≤ `1e-5`；
  - 10/10 final first-action max-abs difference ≤ `1e-5`。
- latency subset：前 2 个 observations
- 每条 path：5 warm-ups + 10 technical repeats
- timing boundary：完整 `CEMPlanner.plan()`，包含 preprocessing 与 CUDA synchronization，不包含 environment interaction
- frozen system thresholds：latency reduction ≥20%，peak-memory ratio ≤1.10

### 4.2 Decision result

| 指标 | 结果 |
|---|---:|
| Independent observations | 10 |
| Passed observations | 10/10 |
| Chained CEM rounds per observation | 10 |
| Per-round top-30 set gate | PASS |
| Per-round `mu/sigma` gate | PASS |
| Final first-action gate | PASS |

该结果说明，在当前 10-observation engineering screen 中，非 bitwise-exact prefix 没有产生超过冻结阈值的 chained-CEM decision drift。它不是 statistical non-inferiority proof，也不能推出 population-level equivalence 或 closed-loop equivalence。[E4]

### 4.3 Latency and memory result

每个 observation 分别取 10 次 technical repeats 的 baseline median 与 cache median，然后计算：

\[
\text{reduction}=1-\frac{\operatorname{median}(T_{cache})}{\operatorname{median}(T_{baseline})}.
\]

| Observation | Baseline median | Cache median | Median saving | Reduction |
|---|---:|---:|---:|---:|
| `wall_case_00` | 12,344.794 ms | 10,573.368 ms | 1,771.426 ms | 14.3496% |
| `wall_case_01` | 12,344.499 ms | 10,573.390 ms | 1,771.108 ms | 14.3474% |
| **Paired median** | — | — | ≈1,771 ms | **14.3485%** |

Peak-memory ratio 为 `1.0000696`，低于 1.10 上限。Latency improvement 很稳定，但低于 20% frozen threshold。[E5]

### 4.4 Gate decision

| Gate | Result |
|---|---|
| Decision gate | **PASS** |
| Memory gate | **PASS** |
| Latency gate | **FAIL**：14.3485% < 20% |
| Joint system gate | **FAIL** |
| Shadow closed-loop | **NOT AUTHORIZED / NOT RUN** |

该 no-go 是对当前实现与冻结门槛的结论，不是“shared-prefix caching 完全无效”。观测到的约 1.77 秒 planner-call saving 是真实的工程信号，但不足以满足本研究预先定义的 expansion condition。[E4, E5]

## 5. Latency 的证据边界

当前 14.35% 的分母是 baseline 完整 planner-call latency，而不是整个 PBS job、完整 MPC iteration 或机器人 control step：

\[
\text{planner improvement}
=\frac{T_{planner,baseline}-T_{planner,cache}}{T_{planner,baseline}}.
\]

计时包含 observation/goal preprocessing、goal encoding、10-round CEM candidate sampling、world-model rollout、objective、stable sorting、top-30 update 与 CUDA synchronization。它不包含 environment rollout、action denormalization/execution、feedback acquisition、success evaluation、video、logging、checkpoint loading 或 PBS/runtime setup。[E2, E5]

若完整 step 的其他成本不变，则：

\[
\text{full-step improvement}
=14.35\%\times\frac{T_{planner}}{T_{full-step}}.
\]

因此完整 MPC wall-clock improvement 不会高于 14.35%，通常会更低。当前没有 component-level profiler 或 full-step timer，不能把剩余 latency 严谨分配给 predictor、objective、environment 或 logging。

## 6. 当前可以与不可以主张的结论

### 可以主张

1. DINO-WM Wall CEM 中存在可显式隔离的 observation-prefix 与 action-conditioned suffix。[E1, E2]
2. Batch-1 factorized prefix reuse 在当前实现上不是 bitwise exact。[E3]
3. 在 10 个受测 observations 的完整 chained CEM 中，factorized path 通过了冻结的 elite-set、`mu/sigma` 与 first-action thresholds。[E4]
4. 在单张 A100 上，完整 planner call 的 paired median latency reduction 为 14.3485%，peak-memory ratio 为 1.00007。[E5]
5. 当前实现没有达到预先冻结的 20% system gate，因此按计划停止，没有运行 closed-loop。[E4]

### 不可以主张

1. 不可以称该 factorized cache 为 exact 或 bitwise preserving。
2. 不可以声称 closed-loop success、trajectory 或 population-level decision equivalence。
3. 不可以把 14.35% 称为完整 MPC、simulator 或 robot wall-clock improvement。
4. 不可以声称达到 frozen system objective，或在失败后把门槛下调来报告 PASS。
5. 不可以将该 bounded result 泛化到其他 checkpoints、tasks、GPU architectures、candidate counts 或 horizons。
6. 当前 IdeaSpark prior-art record 只能支持 search-bounded positioning，不能替代一次面向投稿日期的完整 novelty audit。[E1]

## 7. 当前停止点与可能的重启条件

本轮实验在 joint system gate 结束，保留 negative result，不继续 shadow closed-loop。只有出现以下之一时才值得新建独立、重新冻结的研究阶段：

- 有明确的 profiler evidence 表明新的 implementation change 可去除当前未缓存的主要 planner bottleneck；
- 研究目标改为接受约 14% planner-only gain，并在运行前重新定义 full MPC wall-clock 与 closed-loop endpoints；
- 新硬件、checkpoint 或 planner configuration 形成新的目标 setting，且重新冻结 decision 与 performance gates。

不得把同一批结果通过事后修改阈值重新分类为 PASS。

已提出一个独立的后续 hypothesis：[`Elite-Band Multi-Fidelity CEM`](NEXT_IDEA_ELITE_BAND_MULTIFIDELITY_CEM.md)。它不修改本轮 no-go，而是尝试减少当前 cache 未覆盖的 candidate-specific predictor compute；必须另建 frozen campaign，通过 held-out elite-coverage、planner-decision 与 native-latency gates 后才能进入 closed-loop。

## 8. Evidence map and artifacts

| ID | Evidence | 用途 |
|---|---|---|
| E1 | [`idea.std.zh.md`](../../../ideaspark_run/dino-wm-shared-prefix-cache/phase4/idea.std.zh.md) | 原始动机、方法与 search-bounded novelty framing |
| E2 | [`APPROX_README.md`](APPROX_README.md), [`run_approx_screen.py`](run_approx_screen.py), [`cache_core.py`](cache_core.py) | 实际实现与 timing boundary |
| E3 | [`exact summary`](artifacts/23418502.pbs101/summary.json), [`exact verifier`](artifacts/23418502.pbs101/verifier.json) | Exact mechanism gate 结果 |
| E4 | [`approx verifier`](artifacts/23435258.pbs101/verifier.json), [`APPROX_FREEZE.json`](APPROX_FREEZE.json) | Decision/system gate 与 claim boundary |
| E5 | [`system_summary.json`](artifacts/23435258.pbs101/system_summary.json) | Paired latency、repeat records 与 memory result |

## 9. Drafting provenance

本文由 AI 协助整理本地实验记录；数值来自上述 frozen verifier 与 result artifacts。对外发表前，应由研究者逐项核对代码、artifacts、prior-art coverage、作者责任与最终表述。

用于组织 evidence-bound scientific record 的程序性参考：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
