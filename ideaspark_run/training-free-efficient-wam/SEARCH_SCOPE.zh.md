# 检索范围与判据

四个 connector queries：
1. world action model action chunk inference — 以 joint video/action policy 的 action chunk 推理为领域入口。
2. world action models sparse future tokens — 以未来 video tokens 的计算省略为方法入口。
3. diffusion robot policy denoising acceleration — 以 robot policy 的 denoising 计算为相近问题入口。
4. action chunk caching imagination early exit — 以 action chunk、缓存、imagination 省略和 early exit 为 escape-mechanism 探针。

每个 query 均含可指认的计算对象（action chunk / future tokens / denoising robot policy）。第 4 个探针可能因组合术语过严而低 yield；必须检查 query_yield，不把未检索到视为 novelty。Phase 0.5 用 connector-verified host nominations 补 C³ache、Sparse-WAM、Fast-WAM 等 load-bearing omissions。

Anchor titles：Sparse-WAM: Accelerating World Action Models via Action-Guided Sparse Imagination；Fast-WAM: Do World Action Models Need Test-time Future Imagination?；Cosmos 3: Omnimodal World Models for Physical AI。

先做 structural grounding 与 code feasibility，再做 candidate-specific signature/alias collision search。历史 pattern corpus 只用于机制构思，不用于证明 2026 年 novelty。
