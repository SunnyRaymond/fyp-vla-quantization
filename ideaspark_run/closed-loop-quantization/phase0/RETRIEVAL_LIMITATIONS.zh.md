# Phase 0 检索限制

本次 Phase 0 使用真实 bundled connectors，检索时间为 2026-09-12，采用小 pool：`arxiv=8, ss_recent=6, openalex=6, semanticscholar=6, openreview=4, oa_recent=0`。实际成功的 connector 只有 OpenAlex 与 Semantic Scholar；arXiv 因当前环境缺少 `feedparser` 跳过，OpenReview 因缺少 `openreview` 跳过。Semantic Scholar 多个 query 返回 HTTP 429，OpenAlex 有一次 HTTP 504；原始终端记录和结构化结果保留在本目录。

随后执行了有限的 Phase 0.5 coverage check。8 个候选中 3 个经 connector 标题核验后纳入：`Feedback World Model Enables Precise Guidance of Diffusion Policy`、`QuantWAMs: Calibrating at the Right Granularity for World Action Models`、`Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control`。DA-PTQ 已在初始 pool 中；DyQ-VLA、Omega-QVLA、QuantVLA 及 `Calibrate Where You Deploy` 因 connector 429 未解析，不能当作已纳入全文证据。该结果是有界、降级的文献地图，不是 exhaustive novelty claim；补充近邻说明见项目根目录 `ideaspark_run/PRIOR_SCOUT_2026-09-12.zh.md`。
