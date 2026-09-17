# Campaign conclusion/status audit

日期：2026-09-13。范围：只读核对 `INDEX.zh.md` 的 29 个 evidence links 及其对应的 `RESULT`、`ROOT_DECISION`、`PRIOR_GATE`、`PROTOCOL` 或 `IDEA_BRIEF_AND_GATE`。未读取 raw、未计算指标、未连接 cluster、未新增 paper search，也未修改 `INDEX.zh.md`。

## 结论与状态规则

链接检查全部通过。主要问题是少数行把“某个数值子结果”“工程完成”或“协议准备好”误读成 campaign-level scientific go。建议下游汇总优先使用来源文件的完整状态；`*_preliminary_go` 只有在该 locus 的数值 gate、provenance 和工程 gate 都完整时才可作为整体结论。`implementation_inconclusive`、`inconclusive_binding`、`inconclusive_budget`、`resource_blocked` 和“尚无 job”都不能转成 scientific go 或 scientific no-go。

## 需要直接修正的条目

1. **Flow Geometry under PTQ。** 保留双层事实，但 campaign roll-up 应写为 `inconclusive_provenance`；括号内才写 `numeric_scope_limited_preliminary_go`（仅 language-backbone W4）及 expert-W4 的 `mechanism_no_go`。不能把该行简写为 `preliminary_go`，因为 GPU64763 的最终报告序列化/完整 provenance 缺口仍在，CPU64764 只恢复 raw 数值。来源：[RESULT](../flow-geometry-drift/RESULT.zh.md)。
2. **Value-head gauge。** canonical status 是 `implementation_inconclusive`。W4 的 62.38% median improvement 不构成 research go；FP original/centered no-op 未通过冻结容差，且该工程门是解释 W4 数值的前提。不要改成 `preliminary_go` 或 `mechanism_go`。来源：[RESULT](../value-head-gauge/RESULT.zh.md)。
3. **Existing-Q stratified rounding。** canonical status 是 `inconclusive_binding`，不是 mechanism go 或 mechanism no-go。GPU64799/CPU64801 的 member-level fairness/binding gate 没有给出足够可比误差，median gain 不能作为方法结论；保留 seed1 mismatch、seed3 strict-load 与后续修复历史，但不得用它们代替 scientific evidence。来源：[RESULT](../q-ensemble-rounding/RESULT.zh.md)。
4. **Denoising-call rounding persistence。** `INDEX` 中的“GPU64815运行中”已经过时。按现有交接和 artifact：64815 在 540 s 外部 timeout，完成 11/12 condition，**没有 scientific metrics**；当前应记为 `inconclusive_budget / no_science`，network blocked 时不应声称有可用 rerun。若未来允许的 batch2 是新 allocation，必须另列 job/result，不能覆盖 64815，也不能把 partial raw 改写成 go/no-go。来源：[PROTOCOL](../rounding-persistence/PROTOCOL.zh.md)、[BATCH2 gate](../rounding-persistence/BATCH2_REPAIR_GATE.zh.md)。
5. **Recorded-future error cancellation / teacher-bias。** 当前只有 prior gate 与冻结 protocol，尚无 teacher GPU/CPU scientific job 或新样本输出。建议 canonical status 为 `conditional_prior_go / experiment_pending`（或 `no_job_no_science`），不能写成 `preliminary_go`、`mechanism_go` 或 empirical result。来源：[PROTOCOL](../teacher-bias/PROTOCOL.zh.md)。
6. **Flow step refinement。** `INDEX` 的“prior/identifiability审查保留”过于含混。来源文件实际区分 `novelty_no_go` 与窄 diagnostic `conditional_go`；建议下游统一写 `novelty_no_go / diagnostic_conditional_go`，明确没有冻结实验和经验结论。来源：[PRIOR_GATE](../../../../idea/v100-new-angles/flow-step-refinement/PRIOR_GATE.zh.md)。
7. **Batch Scale Coupling。** 建议把 `novelty/applicability no-go` 标准化为 `novelty_no_go / applicability_unverified`：来源文件否定的是人为引入的当前不存在路径及其适用性，不是实验失败，也不是所有 candidate-batch scale 机制的 empirical no-go。来源：[RESULT](../batch-scale-coupling/RESULT.zh.md)。
8. **Temporal residual correction initial retrieval。** `INDEX` 的“不推进”必须保留为 `not_pursued / incomplete_prior_review`，不能被读成 empirical no-go；它只是与 FRT/PRR 的初步重叠记录。来源：[STATUS](../../../../idea/wm-trajectory-residual-ptq/STATUS.zh.md)。

## 八个已完成 screen 的防误读表

| Screen | 来源文件 canonical reading | 可以写什么 | 禁止写什么 |
|---|---|---|---|
| Antithetic Rounding Pairs | `mechanism_preliminary_go / practical_no_go` | 机制信号在本 screen 有初步支持，当前两-W4成本/保真 recipe 停止 | 不能写成可部署方法 go；也不能把 practical no-go 改成 mechanism no-go |
| Semantic Input Scales | `mechanism_no_go` | 固定 partition/recipe 的机制 gate 未过 | 不能因 GPU 的 2/6 报告把它写成改善；CPU FP64 最终为 1/6 |
| Flow Geometry under PTQ | `inconclusive_provenance`，含 language 数值子结果 `scope_limited_preliminary_go`、expert `mechanism_no_go` | 只保留带 provenance 限制的 language-locus 数值观察 | 不能写整体 preliminary go、deployment claim 或完整复现已证实 |
| Action-gradient geometry | `mechanism_no_go` | 冻结联合假设在 4/6 可识别样本中 0/6 通过 | 不能说 local gradient 一般无效；也不能把首个工程失败当科学证据 |
| Padded-coordinate feedback | `method_no_go` | 当前 fixed zero-target analytic recipe 的经验 gate 失败 | 不能外推所有 unused-coordinate 方法或真实 task success |
| Conditional action distribution | `statistical_inconclusive` | sensitivity/dissociation gate 不足，停止补样本 | 不能改写为 mechanism no-go，也不能把 0/8 positive 当全局无效证明 |
| Existing-Q stratified rounding | `inconclusive_binding` | fairness/member-level comparison 未绑定 | 不能以 −29.02% median 或 partial engineering pass 宣称方法失败/成功 |
| Value-head gauge | `implementation_inconclusive` | FP no-op 前提失败，停止 | 不能以 W4 median improvement 宣称 research go |

## 其余条目的术语对照

以下与来源文件一致，建议仅作规范化拼写，不需要追加实验：

- **Action-gradient、Recency-Split Latent-History Cache、TD-MPC2 ensemble uncertainty、Instruction contrast、Gripper decision margin、Bellman consistency、Residual branch interaction、Action-chunk suffix feedback、Prefix KV reuse vs suffix A8**：分别保留 `mechanism_no_go`、`model_structure_no_go`、`structural_no_go`、`identifiability_no_go`、`insufficient_mechanism_no_go`、`identifiability_no_go`、`identifiability_no_go`、`structural_no_go / identifiability_no_go`、`identifiability_no_go`。这些是 proposal/source/identifiability 或结构性判断，不是经验数值 no-go。
- **OpenVLA-OFT continuous/discrete interface**：统一为 `objective_mismatch_no_go (structural)`；它否定比较对象与目标的匹配，不否定 discrete VLA 或其 quantization。
- **MOPO dynamics uncertainty**：保持 `resource_blocked`；机制审查通过不等于资产、checkpoint 和最小可复现输入已通过。
- **Q ensemble common-U 初始案**：保持 `prior_method_no_go`；min 不是 planner 主路径，未运行实验。
- **Simplex latent quantization**：保持 `novelty_no_go / mechanism_claim_structural_no_go`；窄 diagnostic 可 conditional，但未执行。
- **Hidden-basis permutation invariance**：保持 `engineering_only_no_go`；它是实现 null/self-check，不是独立科学候选。
- **Q decoder-tail rounding**：保持 `insufficiently_differentiated`；未证明超出 data-aware rounding 邻域，也不是 empirical no-go。
- **Timestep constant folding**：保持 `novelty_no_go`；先例重叠导致不追加 GPU。
- **Camera redundancy prior gate**：本 audit 后新增文档的结论为 `identifiability_no_go / GPU=0`，但它尚未列入旧 `INDEX`；若登记，使用完整双层解释，不能只写 mask screen go。

## 可执行收尾清单

1. 在下一次 `INDEX` 状态更新中，将 Flow 的整体状态置于 `inconclusive_provenance`，把 numeric/locus 结果作为子字段；给 Value-head、TDQ stratified、Persistence、Teacher-bias 分别写入上面的 canonical token。
2. 对 persistence 64815 只保留 timeout/11-of-12/no-science receipt；network blocked 时不重试、不从 partial raw 计算指标。未来 batch2 以新 job identity 单独登记。
3. 对所有汇总表采用“科学 gate、工程/provenance gate、资源状态”三列，禁止单列 `go` 覆盖 `implementation_inconclusive`、`inconclusive_*` 或 `resource_blocked`。
4. 其他条目按“其余条目的术语对照”通过；无需重跑、读取 raw 或新增文献检索。

本审查只纠正状态解释和汇总用语，不改变任何冻结 gate、历史结果或停止决定。
