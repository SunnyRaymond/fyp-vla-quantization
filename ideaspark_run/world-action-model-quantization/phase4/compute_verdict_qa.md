# Compute verdict QA

Before repair, the Phase 4 skeleton and assembled expansion contained the automatic verdict `feasible` with rationale:

> GPU line ≈ 1.2 vs 150 GPU-day (1%) → feasible; API line ≈ $0 vs $10000 (0%) → feasible. Overall (worse line) → feasible.

This was a factory fallback comparison from `phase4_skeleton.py` (`DEFAULT_GPU_DAYS=150`, `DEFAULT_API_DOLLARS=10000`), not a user-supplied budget. The intake specifies a single A100-40GB setting but no GPU-day or API-dollar allowance. The 1.2 (or 12) estimate is unmeasured, and no 80GB-class-to-A100-40GB time conversion has been measured. The original `compute_budget` text is retained verbatim.

The repaired compute verdict is `unverified`: actual GPU-day/API-dollar limits are not provided, so feasibility cannot be inferred from factory caps. Engineering remains `tight`; this is a budget-evidence correction, not a scientific rerun.

Original structured value preserved for audit:

{
  "verdict": "feasible",
  "rationale": "GPU line ≈ 1.2 vs 150 GPU-day (1%) → feasible; API line ≈ $0 vs $10000 (0%) → feasible. Overall (worse line) → feasible."
}
