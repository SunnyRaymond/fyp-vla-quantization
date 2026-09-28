# LeWM PushT CEM student：2026-09-24 实验证据

## 当前结论

在与 pinned upstream PushT dataset evaluation 对齐的同一批 50 个任务上，official teacher 为 49/50；当前 `treatment_step1000.pt` student-only 为 11/50。每个 CEM batch 的最后 7 轮使用 teacher 可提高到 25/50（另一有效运行 26/50），但仍未达到预先冻结的 teacher 成功率差距与总规划耗时门槛。前 7 轮介入为 11/50，单独在第 30 轮介入也为 11/50。因此，当前证据支持连续后段 teacher 反馈带来部分收益，不支持只在最后一次评分或只在前段评分就能解决闭环误差。

| 对照 | 成功任务 | 解释 |
|---|---:|---|
| Official teacher | 49/50 | 同任务逐项复现 baseline |
| 当前 treatment student-only | 11/50 | 两个后续有效作业逐项复现 |
| Early teacher rounds 1–7 | 11/50 | 与 student 总数相同；late7 优于 early7，配对 `p=0.0025768` |
| Late teacher rounds 24–30 | 25/50 | 同作业较 student +14，配对 `p=0.0013123`；另一有效作业为 26/50 |
| Final-only teacher round 30 | 11/50 | 较 student 净差 0，配对 `p=1.0`；成功向量并不相同 |
| Reconstructed `balanced_base` proxy | 7/50 | 较当前 treatment −4，配对 `p=0.42395`；不是不可加载的原 Phase 5/6 checkpoint |

Late7 在首次有效四臂作业中较 student +15/50，`p=0.0007286`，但较 teacher 少 23/50，planner solve time 比值为 0.7583，未通过冻结的差距 ≤5 与耗时比 ≤0.70。第二次时机诊断中 late7 为 25/50，较 teacher 少 24/50，耗时比 0.7720；这两个作业不应被说成逐任务完全重现。

上述结果只属于 dataset-protocol-aligned evaluation：沿用 dataset 起点、全数据 scaler、goal offset 25、eval budget 50 和 pinned CEM 设置，使用已有 serialized teacher object loader，关闭视频；并非逐字运行 upstream `eval.py`，也不是先前 random-reset simulator benchmark。固定候选 bank 的 Spearman、top-30 或 screening 数字不能替代这里的闭环任务成功率。

## 决策与后续 gate

当前 checkpoint 来自此前 `cem-distribution-distill` treatment，而其固定起点 candidate-tail distillation 已在原 predictor gate 失败；另有 K-screening 与 elite-boundary pairwise ranking 失败。不要把这些既有 NO-GO 配方简单重跑或把 11/50 归因于最初 `balanced_base`。同协议 checkpoint 对照显示，重建 `balanced_base` proxy 也只有 7/50，且对当前 treatment 的差异不显著；这不是统计等价的证据。

若继续训练，下一步先单独冻结一个真正由 student planner 在 simulator 中访问的状态收集与 teacher-label aggregation pilot，区分它与既有固定 observation candidate-tail 数据。先验证任务/状态/候选的来源、teacher 标签、配对与有限性，再设定训练和 closed-loop 提升门槛；目前尚未提交新训练作业。

### Student-induced shadow pilot 的有效性诊断

新 pilot 已预选 16 个 collection episode 和 16 个 reserved holdout。首次 GPU 尝试 `25537018` 在 HDF5 身份预检因乱序行索引失败，没有产生实验指标。修复元数据读取顺序后，`25537036` 的原定 shadow-off/on 精确 gate 在前四组中有两组环境轨迹不一致，按冻结规则停止，没有进入 16-episode regret 分析。

后续两任务诊断 `25537175` 显示同一 episode 的 gate 结果随独立复跑变化。更关键的是，无 teacher shadow 的 control-control 作业 `25537296` 在 episode `12704` 上发现：dataset 不含 seed 列，`World.reset(None)` 两次形成不同隐藏 block 物理状态；尽管 t0 的 prepared observation、首轮 CEM 和首个动作完全一致，第一步后状态已分叉。固定 reset seed 42 的 control-control 作业 `25537334` 则在同一 episode 上使隐藏状态、CEM、动作与第一步后状态全部精确一致。这个结果只支持**受控 seeded protocol** 的复现性，不能回写为官方未固定 reset 的闭环结果，也还不能证明 teacher shadow 非干扰或 t25 regret 增长。

已冻结 seeded 16-episode shadow pilot 变体：任务、CEM、shadow rounds、四对 gate 和 GO 门槛与原 pilot 相同，仅 dataset reset seed 改为 42。2026-09-24 23 时段 NTU 跳板连续超时；经原 pinned NSCC host key 验证后，改用可达的 ASPIRE2A 官方直连。首次 seeded 作业 `25537594` 的四对 gate 全部通过，但在第九个 episode 恰好提交 25 步、尚未记录 t25 solve 时按原检查 fail-closed，无正式聚合。Pinned World 源码确认第 25 步终止可合法地跳过下一次 replan；保留首次提交 freeze 后，仅在有该步显式 `terminated/truncated` 信号时允许分类为非 matched，原任务、指标和 GO 门槛不变。

修订作业 `25537667` 正常完成：四对 gate 全通过，15/16 个 episode 有 t25 配对；剩余 1 个在第 25 步成功终止。主指标 round30 regret 中位增量 `+0.0861` 达到 `+0.05`，但仅 `9/15=60%` 为正，低于 `75%`，冻结判决 **NO-GO**。Round10/20 中位差分别为 `−0.4177/−0.1384`，提示错误不是一致的 t25 状态退化，而可能集中于晚期 CEM 与特定 episode。该结果只属于受控 seeded variant，不等于原官方 dataset reset 成功率；不能将已失败的 blanket on-policy aggregation 直接推进到训练。

复用现有 teacher shadow 分数的 CPU-only 后验分析 `25537945` 补充：15 个 matched episode 在 t0/t25 的 round10/20/30 绝对 standardized elite regret 中位数均约 `1.6–1.8`，六组均为 15/15 episode 超过 `0.5`；round30 recall@120 中位仅约 `0.23–0.27`。因此严重的 planner-facing 排序缺口在 t0 已存在，而不只在后续状态出现。这个描述性分析不改变原 NO-GO，但将下一步方向从“无条件追加 t25 状态训练”改为“在真实 planner context 上修复候选 elite 排序”，并须避免复刻已失败的固定 observation candidate-tail 蒸馏和 K-screening。

按此方向冻结并运行的 on-policy full-bank residual ranker `25538135` 完成了 4/4 精确配对 gate、1000 updates 和独立 8-episode validation。93 个 train bank、48 个 validation bank，8/8 达 t25。其 episode-level standardized teacher-elite regret 中位差为 `+0.08561`（要求 `≤−0.05`），仅 4/8 episode 改善，round20 中位差 `+0.09793`，判决 **predictor-level NO-GO**。因此不能把该 ranker 推进 CEM。此结果否定当前单候选 residual MLP/loss 配方的稳定修复能力；不排除 teacher shortlist rerank，但应先测其可恢复的候选质量与 K 预算曲线，且已用过的 8 个 validation episode 只能作后续描述性机制证据。

随后只读 CPU 后验诊断 `25538259` 使用同一批 on-policy banks：student Top30 的 teacher-elite recall 在 train/已用 validation 中位仅 `3.33%/5.00%`；Top120 提升到 `23.33%/25.83%`，teacher 在 Top120 内 oracle 精选的 standardized regret 中位仍为 `0.715/0.641`，相对 student Top30 有明显理论收益但远非全 teacher。它没有测当前 native latency，也没有将 teacher 精选接入 CEM；已用 validation 不能作为新的独立 gate。历史不同 checkpoint 的 K-screening native hybrid timing 曾慢于 teacher，故不能仅凭该上限进入 teacher-in-loop。

用户指出 teacher K-screening 与 native hybrid timing 此前已经做过，因此不再重做这一路线。新的 `25542215` 是另一项两任务机制 probe：保持原 student-only CEM 和 seed42 reset，对实际执行的 width-10 action tokens 与每5次 PushT env.step 的真实 observation 做逐 horizon latent 对齐。2/2 tasks 在 t0/t25 的计划动作与执行动作最大差 `≤2.39e-7`；student 对真实观测的误差均高于 teacher，其中 episode12704 的 t25 horizon2–5 relative MSE 约 `0.70–0.93`，teacher约 `0.05–0.08`。这支持“真实环境 latent target 与旧 teacher rollout target 不同”的机制，但只有两个 train episodes，不支持普遍化或直接训练部署。

不重跑这两项的 `25542467` 扩展覆盖了其余 14 个预选 train tasks。两批合并后，horizon 5 的 student−teacher 真实 observation relative MSE 差在 t0 为 15/16 tasks ≥0.05，在 t25 为 15/16 总 tasks ≥0.05（15/15 可测，另 1 项 step25 终止）。这达到扩展前冻结的继续研究阈值，说明此 checkpoint 的真实后续 latent 误差并非仅发生在最初两个例子；仍未证明真实目标训练可改善 planner-facing 排序、速度或闭环成功。此为探索性机制证据，reserved holdout 未触碰。

当前 HDF5 虽有连续 pixels/actions，但现有本地训练代码没有证明 action 与未来 poststep pixel 的索引语义；也没有证据证明这份 HDF5 由所查的 pinned `World.collect()` 实现生成。因此不直接用它构造真实目标，且取消了只能重复检查行索引、不能证明时间语义的预检。后续拟从新的 student-driven episodes 同步采集实际 actions 与 poststep observations，保留旧16-task诊断不重跑；这一步仍是数据准备，不是训练收益证据。

新采集的 CPU-only selection `25543404` 已固定 80 个不同任务（64 train/16 validation），与旧任务排除集交叠 0。首次 GPU collector `25543494` 在首个 episode 前的未排序 HDF5 身份核对失败，0 样本、无 NPZ；只修复身份读取排序，保持冻结任务顺序和清单不变，随后运行修复版 `25543623`。

`25543623` 完成全部80个新 episode、写出141条真实后续观测样本（111 train/30 validation），但最后一条进度打印错误使 PBS exit1 并覆写 summary 状态。没有重跑轨迹；独立 CPU recovery `25543843` 核验 NPZ/episode/window/action 对齐后另写恢复汇总，PBS exit0，保留原失败记录。64个预选train episode中58个有完整窗口，16个validation均有t0、14个有t25完整窗口。这批数据仅用于一次固定配方训练和独立 predictor 判据。

`25543873` 已完成，PBS exit0；训练 weighted MSE 显著下降，但按冻结公式只读重算的 episode-level validation 总体改善中位为 `−0.61%`、8/16 episode 改善，t0/t25 中位分别 `−4.65%/−6.55%`，判 **predictor NO-GO**。已提交 runner 的逐窗口改善平均公式与冻结公式不同，原作业也判 NO-GO；重算依据和两者差别已在结果报告列明。这个真实目标单配方不能稳定泛化到未训练的 student-driven episodes，故不推进其 planner-facing 排序或 CEM，也不在这 16 个 validation episodes 上调参重训。

CPU-only、read-only proposal-dispersion posthoc `25544225` 显示 train 与已用 validation 的 t0/t25 中，round10→30 full-bank 和 student Top30 action dispersion 均收缩；但按 round10 teacher-cost 标准化的 teacher-best access gap 也在四组里下降，说明 Top30 的 teacher-best 可达性改善。冻结的“dispersion 收缩且 access gap 恶化”联合模式不成立，故不提交 mixture-CEM GPU。Validation 已用于 residual-ranker gate；此分析仅为描述性机制证据，不能证明收缩原因或 planner/closed-loop 收益。详见[结果报告](student-induced-shadow-pilot/proposal-dispersion-posthoc/RESULT_25544225.zh.md)。

CPU-only read-only episode-level action-error coupling posthoc `25544993` 在 frozen train split 上得到 t0 `n=16`、t25 `n=15`；Spearman 中，latent H5−H1 error increment 对 first-action mean ΔL2 / normalized ΔL2 / teacher-elite regret 的相关系数分别为 t0 `0.221/0.282/0.441`、t25 `−0.057/0.221/0.136`。t0 仅弱到中等，t25 接近零或偏弱，且指标方向强度随 replan 状态变化，不支持直接以真实 latent error 触发 action correction。First-token elite mean 是同一 bank 内的排序反事实，不是实际执行 action；结果不能证明因果或排除其他误差机制，也没有新增 CEM/closed-loop 证据。t25 唯一缺项是 episode `15946` 在 replan 前终止，按 freeze 不插补；前两次工程失败已保留并排除在指标之外。详见[结果报告](student-induced-shadow-pilot/action-error-coupling-posthoc/RESULT_25544993.zh.md)及[首次尝试](student-induced-shadow-pilot/action-error-coupling-posthoc/ATTEMPT_25544874.zh.md)、[第二次尝试](student-induced-shadow-pilot/action-error-coupling-posthoc/ATTEMPT_25544910.zh.md)。

Frozen inference-time partial-horizon teacher hybrid `25545242` 在已用的 93 个 train banks 上，k0/k5 saved-cost control 186/186 全通过（最大绝对误差 0）；k1、k2 均达到冻结 predictor exploratory GO gate，regret median Δ 分别 `−0.662/−0.888`、15/16 episode 改善，且相对 full-teacher k5 的 latency reduction 为 `71.47%/53.62%`。这是 teacher-prefix/student-continuation 的 inference-time hybrid，不是训练结果；GO **仅许可另行冻结 fresh-episode、independent-bank gate**，不构成 CEM 或 closed-loop 证据，也不授权部署。t0/t25 分层及 native timing 口径见[结果报告](student-induced-shadow-pilot/onpolicy-fullbank-ranker/partial-horizon-teacher-hybrid/RESULT_25545242.zh.md)。

拟进行的同一 k1/k2 方法的 independent-bank 复验 `25545529` 在 8 个预留 episode 的第 29 号任务行身份预检处退出，原因是新 FREEZE 将原选择清单的 `row_index=1799776` 误抄为 `1799976`。前四项非干扰配对通过，但没有产生完整 bank 或 hybrid 质量/耗时结果。用户已明确要求不要重复此前做过的实验，因此**不修正后重提这项 k1/k2 复验**；保留已有 train-bank 探索信号及其未独立验证的边界，不推进该方法的 adaptive CEM 或 closed-loop。详见[失败尝试](student-induced-shadow-pilot/partial-horizon-independent-gate/ATTEMPT_25545529.zh.md)。

Read-only latent-score sensitivity posthoc `25545897` 在同一批 93 个 train banks 上通过全部 186 项 k0/k5 cost controls。episode-median normalized latent MSE 从 t0 的 h1 `0.0989` 增至 h5 `0.6407`，t25 从 h1 `0.0749` 增至 h5 `0.7044`；单 horizon teacher-latent swap 在 h1–h4 的 score/ranking 变化为零，而 h5 在 t0/t25 分别带来 median regret Δ `−1.758/−1.707`、Top30 recall Δ `+0.967/+0.967`，teacher-cost-SD-normalized score perturbation RMS `4.841/7.738`，teacher-reference regret 达 0、recall 达 1.0。h5 结果是 teacher 替换形成的机械上界式敏感度信号，不是可部署修复；teacher 不是 ground truth，单步 latent swap 不构成物理一致 rollout，也不能推出 criterion correctness、GO、CEM/planner 或 closed-loop 结论。详见[结果报告](student-induced-shadow-pilot/latent-score-sensitivity-posthoc/RESULT_25545897.zh.md)。

## 证据索引

- [Official teacher baseline](official-lewm-dataset-teacher-baseline/BASELINE_RESULT_25534994.zh.md)
- [有效四臂 late7 结果与无效尝试边界](official-lewm-dataset-teacher-baseline/STAGE2_RESULT_25535873.zh.md)
- [Early7 vs late7 时机诊断](phase-timing-diagnostic/RESULT_25536049.zh.md)
- [Reconstructed balanced-base checkpoint 对照](checkpoint-attribution/RESULT_25536308.zh.md)
- [Final-only round30 诊断](final-only-teacher/RESULT_25536606.zh.md)
- [既有 CEM candidate-tail 蒸馏 NO-GO](cem-distribution-distill/RESULT.zh.md)
- [既有 elite-boundary ranking NO-GO](cem-boundary-ranking/RESULT.zh.md)
- [既有 teacher K-screening NO-GO](teacher-screening/RESULT.zh.md)
- [Student-induced 选择与原 pilot freeze](student-induced-shadow-pilot/PILOT_FREEZE.json)
- [首次无效尝试：HDF5 索引](student-induced-shadow-pilot/ATTEMPT_25537018.zh.md)
- [原 pilot 配对 gate 失败](student-induced-shadow-pilot/RESULT_25537036.zh.md)
- [两任务动作/状态拆分诊断](student-induced-shadow-pilot/DIAGNOSTIC_RESULT_25537175.zh.md)
- [未播种 control-control 不可复现](student-induced-shadow-pilot/REPRO_RESULT_25537296.zh.md)
- [固定 seed control-control 精确复现](student-induced-shadow-pilot/SEEDED_REPRO_RESULT_25537334.zh.md)
- [修订后的 seeded pilot freeze](student-induced-shadow-pilot/SEEDED_PILOT_FREEZE.json)
- [首次 seeded pilot 边界停止](student-induced-shadow-pilot/SEEDED_PILOT_ATTEMPT_25537594.zh.md)
- [有效 seeded pilot NO-GO 结果](student-induced-shadow-pilot/SEEDED_PILOT_RESULT_25537667.zh.md)
- [t0/t25 绝对 elite regret 后验分析](student-induced-shadow-pilot/POSTHOC_ABSOLUTE_REGRET_RESULT_25537945.zh.md)
- [On-policy full-bank residual ranker NO-GO](student-induced-shadow-pilot/onpolicy-fullbank-ranker/RESULT_25538135.zh.md)
- [On-policy oracle shortlist 后验上限](student-induced-shadow-pilot/onpolicy-fullbank-ranker/ORACLE_SHORTLIST_RESULT_25538259.zh.md)
- [真实 observation latent 两任务对齐 probe](student-induced-shadow-pilot/real-observation-rollout-probe/RESULT_25542215.zh.md)
- [真实 observation latent 14-task 扩展及合并判据](student-induced-shadow-pilot/real-observation-rollout-probe-14-task/RESULT_25542467.zh.md)
- [新 student-driven 80-task 选样](student-induced-shadow-pilot/real-observation-training-data/SELECTION_RESULT_25543404.zh.md)
- [首次采集的零样本读取失败](student-induced-shadow-pilot/real-observation-training-data/ATTEMPT_25543494.zh.md)
- [完整采集后的打印失败](student-induced-shadow-pilot/real-observation-training-data/ATTEMPT_25543623.zh.md)
- [原数据的独立CPU恢复核验](student-induced-shadow-pilot/real-observation-training-data/RECOVERY_RESULT_25543843.zh.md)
- [真实 observation target finetune predictor NO-GO](student-induced-shadow-pilot/real-observation-finetune/RESULT_25543873.zh.md)
- [Proposal dispersion 与 teacher-best 可达性后验分析](student-induced-shadow-pilot/proposal-dispersion-posthoc/RESULT_25544225.zh.md)
- [Real-observation latent error 与 first-action elite coupling 描述性结果](student-induced-shadow-pilot/action-error-coupling-posthoc/RESULT_25544993.zh.md)
- [Action-error coupling 首次工程失败](student-induced-shadow-pilot/action-error-coupling-posthoc/ATTEMPT_25544874.zh.md)
- [Action-error coupling 第二次工程失败](student-induced-shadow-pilot/action-error-coupling-posthoc/ATTEMPT_25544910.zh.md)
- [Partial-horizon teacher hybrid predictor exploratory GO](student-induced-shadow-pilot/onpolicy-fullbank-ranker/partial-horizon-teacher-hybrid/RESULT_25545242.zh.md)
