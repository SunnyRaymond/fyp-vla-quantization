# Scope distinctions to preserve through all pipeline phases

Latest user steering: Fast-WAM is the initial study only; the proposal must have a defensible extension to other WAM baselines.

Do not conflate three meanings of action-only:

1. Quantization parameter scope: only independently parameterized action expert weights are quantized; video expert remains BF16. This is potentially valid for Fast-WAM Base. Shared-backbone models do not necessarily expose such a subset.
2. Training objective scope: only action supervision/distillation is optimized. Gradients may still traverse video/multimodal paths and update shared weights; this is not parameter action-only.
3. Deployment query scope: only action outputs are requested. A shared transformer may still process video tokens internally, and quantizing its weights can affect those tokens; this is not token-only quantization of weights.

The proposal may formulate an architecture-independent action-generation mechanism, but must give explicitly different quantization/training contracts for independent-expert and shared-backbone instances. If a baseline requires full shared-backbone adaptation, label it as such, do not claim every extension fits the 1B action-only budget. Do not duplicate BF16/quantized shared weights to silently bypass shared-weight coupling or report an uncounted high-precision path as strict W4A8.

Core evidence hierarchy: mathematical derivation/assumptions; fixed-input solver fidelity; closed-loop success; native-kernel efficiency. None substitutes for the next. QAT fake quantization does not establish deployed W4A8 runtime. Full action-query latency includes video/context prefill; changing denoising steps, action chunks, controller, precision exclusions, and normalization must be held fixed per comparison.

Budget reality: 4 separate 40GB GPU spaces, not an automatically pooled 160GB. Total GPU-hours unspecified, so feasibility statements remain bounded estimates with a measured memory/throughput gate, not a promised multi-model campaign. Formulation does not authorize experiment submission. Current date for novelty search: 2026-10-03 Asia/Singapore.

Problem-necessity check: SteerQuant (arXiv2609.39056v1, Table1 and Section5.1) reports near-BF16 LIBERO success at W4A8 for Cosmos-Policy/FastWAMJoint; it is not honest to assume all W4A8 WAMs currently fail badly. Treat those as paper-reported results on their own checkpoints/scopes. A new QAT proposal needs a concrete unresolved mechanism and equal-cost benefit (e.g., the condition where existing treatment is inadequate), not just a claim that action quantization needs training. Do not silently change precision, denoising steps, or deploy scope to manufacture the gap.
