# 已有 proposal 与本次差异约束

现有本地 proposal：`D:/Downloads/Final Year Project/idea/fast-wam-temporal-feature-cache/phase4/idea.std.zh.md`，方法名 Causal Risk-Envelope Cache (CREC)。

它提出在 Fast-WAM Optional-IDM 中逐缓存单位干预，测量 stale feature 引起的动作与预测视频联合偏差，按观测变化分层，估计风险包络，再用预算选择刷新单位，并失效所有后继。它目前是 proposal 文本；不能把渲染卡片或 gauntlet verdict 当作已完成实验证据。

现有 audit 要求和 DriveCache 区分：action-aware cache allocation、calibrated response budget、causal refresh/replanning 已有相关 prior。新 proposal 不应仅把 drift threshold 改成另一种下游风险/attention评分，或重新包装这个 CREC。

本次允许探索新的 load-bearing hypothesis；优先寻找能在 FastWAM-Joint 和 Cosmos3 Edge 的现有 frozen checkpoints 上共同实现的计算省略机制。每个候选须说明其与 Sparse-WAM action-guided future-token selection、C³ache cross-chunk residual reuse，以及 CREC 单位风险校准的具体区别。若只属 engineering/application-grade，应明确标注。
