# ROUND3 root gate

保留 ROUND3_PRIOR_GATE 原始独立建议，下列判断作为本轮实际执行决定。

**Prefix KV reuse vs suffix output quantization：identifiability_no_go，GPU0。** 两臂同时改变 intervention locus、tensor形状、数量及下游 Jacobian，same A8 不能保证同budget，更不能把 autocorrelation/endpoint差异归因于持久复用。即使各自common-FP-path回放，也没有隔离 reuse 本身。没有必要为此先跑 cache-shape GPU preflight；原建议conditional-go被root否决。

**Support-aware Q decoder-tail rounding：insufficiently_differentiated，GPU0。** 保留不推进的决定，但修正原理由：FP learned-value reference不是ground truth，确实不能认证真实return；然而FP保真作为quantization diagnostic本身有效，不能据此否定所有机制screen，否则与gauge/TDQ相矛盾。目前方案主要是给既有data-aware rounding换decoded-Q reconstruction objective，未给出超过AdaRound/GPTQ邻域的独立可识别机制，因此不为它启动新实验。这不是已实证的不可行，也未认证被特定prior完全scoop。

**Denoising-call rounding persistence：转入 matched design 审查，尚无输出。** 原方案独立抽frozen/redraw可能因有限draw造成marginal field误差不同。root提出固定3套SR weights，3条frozen与3条cyclic schedule；每个step都使用同一draw multiset，从设计上匹配common-FP-path逐步marginal，再分开测integrated forcing与free-running endpoint。它只改变时间coupling，没有平均不同trajectory的actions。代价是需要多份weight和切换，不作低bit部署加速承诺。数学、先例、资源gate审查后才冻结；不是对旧经验no-go的事后救援。
