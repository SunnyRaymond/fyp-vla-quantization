# DINO-WM 下一候选方向：Elite-Band Multi-Fidelity CEM

**记录日期：** 2026-09-18  
**状态：** `unvalidated research hypothesis`  
**建议方法名：** Elite-Band Multi-Fidelity CEM（EBMF-CEM）  
**目标 setting：** DINO-WM `wall_single`、现有 checkpoint、CEM `K=300, H=5, topk=30, opt_steps=10`

> 本文是下一轮实验提案，不是当前实验结果、novelty proof 或性能主张。旧的 RankSafe-EPC exact claim 已经 no-go；现有约 14.35% planner-only gain 仍按原冻结结论保留，不能由本提案重新分类。

## 1. 为什么转向这个方向

现有 factorized shared-prefix cache 已经去除了大量 observation-prefix 重复计算，但完整 `CEMPlanner.plan()` 的 latency 只下降约 14.35%。剩余主要计算仍位于 action-conditioned suffix：每个 CEM round 都要为全部 300 个 candidates 执行多步 rollout，而每个 rollout step 都调用完整的 6-layer ViT predictor。

CEM 实际只用 top-30 elite candidates 更新 `mu/sigma`，最终只返回 first action。当前实现却给所有 candidates 分配相同的完整 predictor compute。因此，下一问题不是继续扩大 prefix cache，而是：

> 能否先用廉价 evaluator 排除明确不可能进入 elite set 的 candidates，只对 elite cutoff 附近的 ambiguous candidates 执行完整 predictor，同时保持 CEM update 与 first action？

## 2. 核心 hypothesis

存在一个比完整 6-layer predictor 更便宜的 coarse evaluator。对每个 candidate，它产生 approximate objective `J_tilde` 和经独立 calibration 得到的误差区间 `[L, U]`。如果某个 candidate 的下界已经高于第 `topk` 个最小上界，则在区间同时覆盖真实 objective 的前提下，该 candidate 不可能进入真实 top-30，可以跳过完整 predictor。

令 candidate `i` 的区间为：

```text
L_i = J_tilde_i - epsilon_i
U_i = J_tilde_i + epsilon_i
tau = kth_smallest(U_1, ..., U_K),  k = 30
A = {i | L_i <= tau}
```

只对 ambiguous set `A` 执行完整 predictor，并仅使用这些 full objectives 选择 elite set。若 calibration coverage 不成立、`A` 过大而无法达到 break-even，或 runtime 检查异常，则当前 CEM round fallback 到原始 full evaluation。

这不是“用 approximate cost 直接替换真实 cost”。Approximation 只负责决定哪些 candidates 必须接受完整计算；最终 elite selection 仍由 full objectives 决定。

## 3. 首轮只验证一个 coarse evaluator

为避免无边界 sweep，首轮固定使用：

- **`cheap_L4`：** 使用现有 6-layer ViT predictor 的前 4 个 Transformer blocks，加现有 final LayerNorm；
- rollout horizon、action sequence、objective、CEM sampling 与 full path 相同；
- 不训练新模型，不改变 checkpoint；
- 不同时测试 BF16、token pruning、蒸馏或 learned router。

`cheap_L4` 只是最小 mechanism probe。若它不能形成较小且稳定的 ambiguous set，本轮停止，不继续搜索 layer count。

## 4. 分阶段实验

### Stage A：Coverage calibration，不主张加速

在 calibration observations 上，对完全相同的 candidate populations 同时运行 `cheap_L4` 与 full predictor，保存每轮所有 candidates 的：

- `J_tilde_i` 与 `J_full_i`；
- absolute error；
- cheap/full top-30；
- full elite cutoff；
- 按 CEM round 分组的误差分布。

从 calibration split 冻结 `epsilon`。首轮优先使用每个 CEM round 的 calibration maximum absolute error；不要在 validation observations 上重新选择或放宽 `epsilon`。

### Stage B：Held-out mechanism screen

在未参与 calibration 的 observations 上：

1. 用冻结的 `epsilon` 构造每个 candidate 的 interval；
2. 生成 ambiguous set `A`；
3. 检查真实 top-30 是否全部落入 `A`；
4. 用 `A` 内的 full objectives 重建 CEM elite update；
5. 分别递归运行 baseline 与 EBMF-CEM 的完整 10-round chained CEM，不能每轮强行重置到共同 `mu/sigma`；
6. 保存 ambiguity ratio、top-30、`mu/sigma`、最终 first action 和 fallback 原因。

Stage B 的 independent unit 是 observation/planner call；同一 observation 内的 candidates 或 technical repeats 不是独立样本。

### Stage C：Native system screen

只有 Stage B 通过后才运行。比较三个 paths：

1. `baseline_full`：原始完整 planner；
2. `factorized_full`：当前 Decision-Safe Approximate Factorized Cache；
3. `factorized_ebmf`：当前 cache 加 EBMF-CEM。

使用相同 observation、checkpoint、CEM noise schedule、warm-up、CUDA synchronization 和 interleaved seeded path order。计时边界仍是完整 `CEMPlanner.plan()` entry 到返回 first action；不得只计 predictor kernel。

系统是否可能加速由以下 break-even 条件决定：

```text
T_cheap(K) + T_full(|A|) + T_interval_and_routing
    < T_full(K)
```

必须同时记录 coarse path、full reevaluation、interval/routing 和其他 planner components 的耗时，避免把调度开销遗漏在 latency claim 外。

## 5. 冻结 gates

### Mechanism gate

全部 held-out observations 和全部 chained CEM rounds 必须满足：

- interval coverage 没有漏掉任何真实 top-30 candidate；
- reconstructed top-30 set 与 `factorized_full` 完全相同；
- 每轮 `mu/sigma` max-absolute difference `<= 1e-5`；
- 最终 first-action max-absolute difference `<= 1e-5`；
- 任一 coverage violation 必须触发该 round 的 full fallback，不能静默继续 approximate selection。

### System gate

- 相对 `baseline_full` 的 paired median planner-latency reduction `>= 20%`；
- 同时报告相对当前 `factorized_full` 的 incremental reduction，不能只与较慢 baseline 比；
- maximum peak-memory ratio `<= 1.10`；
- latency 必须包含 cheap evaluation、full reevaluation、routing、synchronization 和 fallback 成本。

### Closed-loop authorization

只有 mechanism gate 与 system gate 同时通过，才可新建独立 frozen shadow closed-loop stage。首轮 EBMF-CEM 实验不自动授权 closed-loop。

## 6. 最小 controls

- **Full evaluator：** 定义真实 elite set 与 planner output；
- **Cheap-only control：** 直接用 `cheap_L4` objectives 更新 CEM，不做 ambiguous-band full reevaluation。它只用于检验 full reevaluation 是否必要，不能作为有效方法；
- **Factorized full baseline：** 隔离 EBMF-CEM 相对当前 cache 的新增收益。

不在首轮加入更多 layer counts、不同 precision、token selectors、learned routers 或多个 interval estimators。

## 7. Kill switches

出现任一情况即停止当前 `cheap_L4` 分支：

1. validation 中冻结 interval 漏掉真实 elite candidate，且没有按约定 fallback；
2. top-30、`mu/sigma` 或 first action 超过 mechanism threshold；
3. ambiguous set 长期过大，native timing 无法达到 break-even；
4. 相对当前 `factorized_full` 没有稳定 incremental latency gain；
5. 为得到通过结果必须在 validation 上重新调 `epsilon` 或事后放宽 gate。

失败结论应保留为：`cheap_L4 cannot support useful elite-band separation under the frozen setting`。它不自动否定其他 multi-fidelity evaluator，但也不授权继续做无边界 layer/threshold sweep。

## 8. 当前可以期待的研究贡献

若通过，本方向可以主张的不是一般性的 candidate pruning，而是：

1. 在 DINO-WM latent world-model planning 中，将 predictor compute 按“是否可能改变 CEM elite selection”分配；
2. approximate evaluator 只用于界定 ambiguous band，最终 elite candidates 仍由 full predictor 决定；
3. planner-level decision contract、native latency 与 fallback cost 被联合验证。

当前不能主张：

- formal deterministic guarantee；`epsilon` 首轮是 calibration-bounded empirical interval；
- 对其他 tasks、checkpoints、candidate counts、horizons 或 GPUs 的泛化；
- closed-loop equivalence；
- novelty 已确认。Multi-fidelity MPC、candidate pruning、adaptive CEM 与 recent cache systems 都需要在投稿前做针对性的 scoop check。

## 9. 资源与执行边界

- CPU/local：实现、unit tests、结果 schema 与 verifier；
- ASPIRE2A：所有模型加载、candidate rollout、profiling 与 timing 必须在真实 PBS compute allocation；
- 首轮只需单张 A100 的短 allocation；不要一开始申请 2–4 GPUs；
- PBS 脚本在任何模型/data I/O 前检查 `PBS_JOBID` 与非 login hostname；
- 不在 login node 下载、解压、安装、模型推理、benchmark 或执行重 I/O；
- 先做小型 smoke，再运行 calibration/validation/system stages。

## 10. 与当前结果的关系

本提案继承当前 factorized cache 作为 baseline，但不改变已有结论：

- RankSafe exact factorized cache 仍是 no-go；
- 当前约 14.35% planner-only latency gain 仍低于旧的 20% gate；
- EBMF-CEM 是新的 research campaign，必须使用新的 frozen artifacts 与独立结果目录；
- 不得用未来 EBMF-CEM 的结果回写或重分类旧实验。
