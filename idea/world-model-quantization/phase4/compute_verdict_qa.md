# Phase 4 compute verdict QA

The previous automatic skeleton verdict is preserved for audit history:

- verdict: `feasible`
- rationale: `GPU line ≈ 12 vs 150 GPU-day (8%) → feasible; API line ≈ $0 vs $10000 (0%) → feasible. Overall (worse line) → feasible.`
- source behavior: `compute_verdict_from_budget` applied factory defaults `DEFAULT_GPU_DAYS=150` and `DEFAULT_API_DOLLARS=10000` through an OR rule.

This run does not have a user supplied GPU-day or API-dollar budget. The single A100 40GB intake is a hardware constraint, not a recognized campaign quota. The stated 12 GPU-day (and any 1.2 GPU-day estimate) is unmeasured; any 80GB-to-40GB scaling is also unmeasured. Therefore the corrected compute verdict is `unverified`, and the overall feasibility label is `unverified`; the engineering verdict remains `tight`. The guarded `compute_budget` prose is unchanged.
