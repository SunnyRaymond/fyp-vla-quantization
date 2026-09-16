# Root gate：conditional diagnostic go

独立 [prior复审](../new-cut-shortlist/BROADCAST_ROOT_REVIEW.zh.md) 允许一轮。Root接受空间activation coupling与已停止的weight-member/temporal-weight两个切面不同；不接受原审查“非native干预必不可识别”的规则，也不把generic novelty未验证改写成novelty成立。

最终只采用FP、RTN-before/after、Shared3/Spatial3；不做独立patch采样第三arm。三个固定draw在每个patch处的multiset严格匹配，解决有限样本扰动预算不等的问题。Per-state 20D共同scale在两arm完全相同；不采用每个scalar独立maxabs scale（会把单scalar变为±7而退化成无量化误差）。六个已冻结 observation 明确复用；不把它们称为fresh samples。

Spatial是固定balanced offsets的量化误差空间重排，非独立随机patch。其不同patch的code来自同一三draw，不能从本screen推断一般独立SR性能。Same-patch marginal与total-MSE相同，weights/predictor/visual相同，变化仅是空间joint assignment；因此output MSE差异能识别这项具体干预效应，无需另把“是否native kernel”当作识别门槛。训练输入曾exact repeated是重要边界，但quantization造成副本不同正是自变量，不作为自动弃验理由。

维持PROTOCOL冻结的6state、5/6至少10% primary gate，全部结果STOP。未训练、未跑完整验证。原encoder-versus-predictor、CA-versus-SA的module-confounded版本继续保留GPU0；本案不是对它们的经验结果做事后修复。
