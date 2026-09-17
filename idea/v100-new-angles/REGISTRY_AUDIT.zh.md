# SCREEN_REGISTRY audit

审查日期：2026-09-13。仅读取各目录的 `RESULT.zh.md`、协议/冻结文档和已引用的小型 verification 入口；未读取 raw、未运行模型/数值、未连接集群。

## 计数与链路

- registry 的 `count=13` 且实际 entries=13，与 INDEX 中从 Antithetic Rounding、Semantic Input Scales、Flow Geometry、Action-gradient、Padded-coordinate、Conditional Marginal、Existing-Q Stratified、Value-head Gauge、Rounding Persistence、Recorded-future、Policy-prior、Euler Jacobian 到 Broadcast 的经验 screen 集合一致。
- 13 个 entry 都有实际存在的 `primary_protocol`、`RESULT.zh.md`、producer receipt 与 CPU receipt 相对路径；GPU job 与 CPU verifier job 均按 RESULT 中的 producer/verifier 记录登记，Flow 的 scheduler failure 已由 `inconclusive_provenance` 明确保留，所有 `stop` 都为 `true`。
- Flow 的“backbone 数值 preliminary go / overall inconclusive_provenance”、Gauge 的 `implementation_inconclusive`、TDQ 的 `inconclusive_binding`、Persistence 的“数值 go + generic novelty no-go”以及 Broadcast 的“数值 go + application novelty unverified”均被保留，没有把它们压成单一的 success/no-go。

## 保留失败与边界

- 失败、兼容性探测与修复 lineage 统一放在 `retained_lineage`，每项明确 `job_ids/status/role`，不再把 64795 或 64798 等成功诊断统一标为 failed；Persistence 的 64815/64823/64829 与 Q-ensemble 的早期 source/binding/alias 路径没有和最终结果拼接。
- Q-ensemble 的 `64780-64790` 保留为 `mixed_or_not_individually_verified`，不猜测区间内每个 job 的状态，也不把它解读为额外 scientific run；其余 lineage 均有相应 RESULT 的 failure、repair 或 nonfinal 描述。
- Registry 只登记本 campaign 的 13 个 empirical screens；历史 CEM/FRT/PRR、GPU=0 prior-only 候选和 standalone preparation 不计入 `count`。各远端 raw/verification 的完整性仍以 RESULT 引用的 remote artifact 为准，导航文件本身不复制大数组。

未发现计数、协议路径、GPU/CPU 对应或冻结结论的断链；本文件与 registry 仅作为导航，不改变任何科学结果或 stop 决定。


Final closure 2026-09-13: GPU64840 COMPLETED26s and CPU64841 COMPLETED8s, all engineering and independent scientific replay checks passed; 6/6 binding, 0/6 joint positives => reference-branch mechanism_no_go STOP. Registry now14 screens, allSTOP. Actual account primary used99%/remaining1% reached; no new work, no pending jobs, no reset credit used. Previous13-entry or pending statements above are historical.
