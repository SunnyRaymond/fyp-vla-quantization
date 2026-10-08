# Training-free Efficient WAM 候选比较

资料截止 2026-10-02。本页从已提交的 Phase2 generation 提取，并更新唯一主候选的审查状态。主候选已完成 coherence、collision、critique、一次限定 revision 和独立 post-revision review；其余仅做机制与 prior 初筛。所有候选均没有模型实验验证。

## Source-group action-conditioning edge mask

- 机制：按每层 output-projected action contribution 排序，在后续 native action updates 中压紧 observation keys/values，并以真实 sparse/varlen kernel 删除低贡献 QK/AV 边。
- 关键前提：首个 solver update 的 source-group ranking 能代表同 chunk 后续 action updates；该路径有足够全模型 headroom，且 kernel 真正跳过计算。
- Prior 风险：ToPi 已有 attention×value norm、累计贡献选集和跨 step 复用；CAPA 已有 output-projected contribution，不能把 value/W_O 当作新意。Sparse-WAM 的 action-guided future selection 与 GeoBoN 的 action-future consistency gate 是相邻干预。当前只保留待验证的 WAM-specific action-query→observation source-group/task effect。
- 当前结论：条件保留唯一 canonical candidate；独立 post-revision review=pass 仅说明规则与修订符合审查要求。checkpoint、kernel、headroom、排序的任务优势、两 baseline 与真机条件都未验证。不建议立即投入大实验；先解决公开入口并检验 Amdahl 上限。
- 检查深度：Canonical coherence/collision/critique/revision/post-review completed; empirical validation absent. critique 的唯一 R1 增加同边预算的 attention-mass、ToPi/VATP 和 CAPA-style score 对照，原 falsifier 与预算不变。实现与成本报告保持 future cost=unknown。

## Exact observation K/V reuse

- 机制：若模型图证明 observation hidden/K/V 对 solver timestep 与变化中的 action/future state 严格不变，则在 action queries 间或 solver updates 间复用投影。
- 关键前提：存在由 mask 与 timestep modulation 保证的精确不变层；latent 固定本身不够。
- Prior 风险：Efficient-WAM video K/V cache 与 WAMachine intermediate-state reuse；FastWAM-Joint API/不变性未确认。
- 当前结论：淘汰为当前主机制：条件未证且可能退化成已有 cache。
- 检查深度：Preliminary prior screen only; no implementation or novelty gauntlet.

## Within-chunk action residual reuse

- 机制：按相邻 solver evaluations 的 action-branch residual 相似性跳过部分 transformer layer computation。
- 关键前提：动作 residual 跨 active flow steps 稳定，且 probe/refresh 成本低于重算。
- Prior 风险：WAMachine Residual Rescaling、C3ache residual reuse 与本地 CREC risk-envelope cache。
- 当前结论：淘汰：核心对象已与 state/residual reuse priors 邻近，改用 action branch 或风险分数不足以形成新机制。
- 检查深度：Preliminary prior screen only; no implementation or novelty gauntlet.

## Low-rank action-context operator

- 机制：用 low-rank 或 random-feature surrogate 替换 action-query 对 context keys 的 dense softmax interaction。
- 关键前提：跨 layer、timestep 与 embodiment 存在可压缩谱，且 approximation error 能联系到 action/task outcome。
- Prior 风险：Performer-style random-feature linear attention 与既有 approximate-attention families；无已知 WAM 控制保真界。
- 当前结论：淘汰：目前只有通用 operator approximation，没有已知 WAM-specific invariant。
- 检查深度：Preliminary prior screen only; no implementation or novelty gauntlet.

## Async lookahead/action separation

- 机制：将 future predictor 放到慢时钟，让 action policy 在快时钟读取 held lookahead latent。
- 关键前提：stale future latent 仍支持操作质量且该调度改善同步推理成本。
- Prior 风险：GlanceWAM asynchronous lookahead/co-training、WAMachine observation rebinding/execution slack、FBFM asynchronous feedback。
- 当前结论：淘汰：机制轴高度邻近，所需 co-training 也不符合 frozen training-free 条件。
- 检查深度：Preliminary prior screen only; no implementation or novelty gauntlet.
