# DINO-WM PushT Action-Prefix Compiler — Stage A Result

## Decision

Job `24050748.pbs101` completed successfully, but the frozen scientific gate is **NO-GO**:

- capacity/control: **PASS**;
- predictor latency: **PASS**;
- planner-relevant fidelity: **FAIL**;
- overall: **FAIL**.

This recipe does not proceed to CEM integration, closed loop, LeWM transfer, or a full-framework claim. Thresholds and repeat counts are not changed after the result.

## Measurements

| Metric | Result | Frozen gate |
|---|---:|---:|
| last-10 / first training loss | `0.2248` | `<=0.8` |
| future-action leakage | PASS | max-abs `<=1e-6` |
| objective Spearman | `0.8691` | `>=0.99` |
| top-30 overlap | `0.5333` | `>=0.95` |
| frozen teacher median | `3499.99 ms` | diagnostic |
| compiled student median | `9.64 ms` | diagnostic |
| predictor-level reduction | `99.7246%` (`363.1x`) | `>=20%` |

The latency is predictor-only on one cached native anchor latent and batch 300. It excludes `encode_obs`, CEM, objective/ranking, MPC, and environment interaction; it is not a full planner speedup or a fair same-capacity SOTA result.

Per-horizon relative MSE rose monotonically from `0.0087` to `0.0333`, while cosine fell from `0.9962` to `0.9827`. Despite apparently strong latent similarity, ranking fidelity was inadequate. The failure was also anchor-dependent: held-out Spearman was about `0.77` on anchor 0 and `0.97` on anchor 1; top-30 overlap ranged from `0.33` to `0.83`.

## Interpretation boundary

The experiment supports three narrow statements:

1. the causal action-prefix interface is implementable on native DINO patch/proprio geometry without future-action leakage;
2. one parallel student call can be much cheaper than five frozen-teacher calls at predictor level;
3. latent MSE/cosine alone is insufficient to preserve planner candidate ranking.

It does not show that post-hoc compilation is planner-safe, that it transfers to LeWM, or that it combines with observation-prefix reuse. The independent PushT observation-cache arm also failed its frozen gate.

The next distinct hypothesis, if pursued, should be planner-aware teacher distillation rather than more steps or looser thresholds for this latent-MSE-only recipe. Any such run needs a new frozen protocol and fresh held-out action seeds, because the current held-out results have already informed that design.

## Evidence

- `artifacts/24050748.pbs101/stage_a_summary.json`
- `artifacts/24050748.pbs101/job.log`
- `artifacts/24050748.pbs101/job_status.txt`
- `artifacts/24050748.pbs101/gpu_usage.csv`
- `artifacts/24050748.pbs101/gpu_telemetry.jsonl`

The A100 training phase sustained approximately `99–100%` GPU utilization at about `3.3 GiB`; held-out batch-300 evaluation reached about `25.1 GiB`.
