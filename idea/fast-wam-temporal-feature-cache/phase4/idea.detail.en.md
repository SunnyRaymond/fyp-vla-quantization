# Causal Risk-Envelope Caching for Fast-WAM Optional-IDM

**Method.** Causal Risk-Envelope Cache (CREC)

## Motivation
**Problem framing.** The Phase 1 bottleneck is not simply redundant computation. In Fast-WAM Optional-IDM, a changed receding observation can alter the influence of a cached unit through action and predicted-video branches, while units with similar hidden-state drift can have very different downstream consequences. A global local-drift threshold therefore treats the cache as exchangeable even when a small high-risk tail dominates paired error.

The proposed setting keeps the existing model and graph fixed and turns that asymmetry into a measurable object. C3ache supplies a useful periodic reuse reference, EfficientVLA supplies a feature-similarity reuse reference, and WorldCache and X-Cache show that heterogeneous or structure-aware caching can preserve rollout quality; none of these references, as represented in the scaffold, establishes a unit-level stale-value intervention whose paired action/video risk determines dependency-complete refresh. The question is consequently whether causal risk allocation, rather than another similarity score or cadence, is the missing operation.

**Why now.** Recent caching work makes the required measurement stack timely: C3ache provides a World-Action caching baseline, WorldCache exposes heterogeneous temporal behavior, and X-Cache provides structure- and action-aware reuse guards. DriveCache is an important boundary case because it already combines a scene-level dynamic-programming schedule, an action-aware response budget, and causal drift-triggered refresh/replan; CREC therefore makes no broad claim for action-aware budgeted caching. The remaining bounded question is whether Fast-WAM Optional-IDM has a distinct per-unit paired action/video risk envelope, a testable heterogeneity gate, and descendant-complete invalidation that can be evaluated with the existing checkpoint and native GPU timing.

**Why prior work stopped.**
- `arxiv:2606.08962v1` (arXiv 2026-06): C3ache reuses same-denoising-step action-expert residuals on a periodic cross-chunk cadence to accelerate World-Action inference.
  - _Did not do_: It did not estimate a per-unit paired action/video downstream-risk envelope from stale-value interventions or use that envelope to allocate refreshes.
  - _Structural reason_: Its fixed cadence and aggregate task-success evaluation omit the causal intervention, heterogeneity test, and error-budgeted descendant recomputation needed for unit-level risk allocation.
- `semanticscholar:b5be12ccbb037be57a1dbc03fbd43d379170055f` (arXiv 2026-03): Fast-WAM studies whether test-time future imagination is needed for reliable world-action control with an Optional-IDM configuration.
  - _Did not do_: It did not turn the fixed Fast-WAM graph into a cache-unit intervention platform with paired action/video risk estimates.
  - _Structural reason_: Its question concerns the necessity of future imagination, not the missing measurement of stale intermediate influence under receding observations.
- `arxiv:2506.10100v1` (arXiv 2025): EfficientVLA reuses attention and MLP intermediates in a CogACT action head using feature-similarity criteria without retraining.
  - _Did not do_: It did not condition a paired action/video downstream-risk envelope on Fast-WAM receding-observation changes or validate a unit-label heterogeneity null.
  - _Structural reason_: Feature similarity is an input-side proxy, whereas the candidate requires intervention-based downstream measurement and dependency-aware refresh.
- `arxiv:2603.06331v2` (arXiv 2026): The heterogeneous-token WorldCache uses curvature and accumulated drift to preserve video rollout quality while reusing non-uniform tokens.
  - _Did not do_: It did not establish a paired action/video risk envelope for Fast-WAM units or test its tail against a unit-permutation control.
  - _Structural reason_: Its quality-preserving object is heterogeneous video-token dynamics, not causal paired-output influence with an action branch and descendant invalidation.
- `arxiv:2603.22286v1` (arXiv 2026-03): WorldCache uses content-aware reuse to accelerate video world-model rollouts while accounting for temporal variation.
  - _Did not do_: It did not solve a Fast-WAM paired action/video error budget whose refresh decisions are indexed by intervened cache units.
  - _Structural reason_: The video-rollout objective does not supply the action-coupled downstream measurement and dependency-complete recomputation required here.
- `arxiv:2604.20289v2` (arXiv 2026-04): X-Cache gates cross-chunk block reuse with structure- and action-aware input fingerprints and KV-update guards.
  - _Did not do_: It did not calibrate a causal per-unit risk envelope or allocate refreshes under a paired downstream error budget with descendant invalidation.
  - _Structural reason_: Its guard is a fingerprint and update rule, not an intervention that distinguishes heterogeneous causal influence from exchangeable drift.

**What changes when the gap closes.** Fast-WAM caching would gain an explicit allocation object: refresh decisions could protect the empirically demonstrated high-risk tail while reusing the bulk at a fixed skip rate, instead of treating local drift as a sufficient proxy. This would make paired action/video error and closed-loop task success part of the cache validity boundary, complementing the reuse mechanisms of C3ache, EfficientVLA, and WorldCache without claiming to subsume DriveCache's scene-level controller. A successful gate would also provide a reusable causal diagnostic for deciding when a latency gain is mechanistically meaningful.

## Method
**Pipeline.** CREC starts from held-out receding-observation replays of the existing Fast-WAM Optional-IDM graph and replaces one cached unit-timestep with its stale value while holding the remaining graph fixed. It aggregates the paired action and predicted-video deviations into a unit-specific risk envelope, then requires the envelope distribution to reject a unit-label permutation null before using it for allocation. A binary refresh decision protects the high-risk tail under a downstream error budget and reuses the bulk at the requested skip rate. Every refresh invalidates graph descendants before the paired outputs are evaluated for error, native latency, and closed-loop task success.

### Background
*Set up the existing Fast-WAM Optional-IDM graph and obtain per-unit intervention measurements.*

1. **Replay stale units** (`S1`)
   - Start from held-out receding-observation replay records keyed by episode, time, and observation. Use the cache-unit registry to load each unit's stable $`unit_{id}`$, producer node, tensor shape and dtype, cached tensor, and uncached reference tensor. For one unit at one observation change, replace only that cached tensor and keep the same receding observation, model parameters, random-number state, and all other parts of the Fast-WAM Optional Inverse Dynamics Model (IDM) graph fixed. 【author decision: choose whether a cache unit is a whole intermediate tensor, a graph-node output, or a finer tensor block, and freeze its registry and stable identifier before replay】 Run the action and predicted-video heads and save both uncached and stale outputs together with the episode, timestep, unit, and intervention metadata. Compute paired action and predicted-video deviations over matching output elements, then group the records by observation-change level `d`. 【author decision: define the representation, normalization, distance or norm, and fixed strata for `d` before calibration】 Use the S1 equation-defined estimator as an empirical conditional mean of the weighted paired deviations to produce $`r_{u}(d)`$, retaining $`unit_{id}`$, `d`, sample count, both component deviations, and the envelope estimate in the per-unit sample table. 【author decision: set and provenance the action and predicted-video weights, including whether they are equal, task-unit normalized, or selected on a separate calibration split】

*The unit-specific envelope combines paired action and predicted-video downstream deviations at an observation-change level.*
$$ r_u(d)=\mathbb{E}\left[\lambda_a\Delta_a+\lambda_v\Delta_v\mid u,d\right] \tag{1} $$

   - _Why:_ This closes the missing causal intervention: local hidden-state change is held separate from the downstream effect of one stale unit.

### M1_risk_envelope
*Test heterogeneous paired downstream risk and turn the calibrated risk envelope into a refresh allocation.*

2. **Gate on heterogeneity** (`S2`)
   - Group the S1 rows by $(`unit_{id}`,$ `d`) and recompute the unit-specific risk envelope with the same paired-deviation estimator. Construct the unit-label permutation null by shuffling complete unit labels while keeping each risk value attached to its change level, recomputing the predeclared heterogeneity statistic for every permutation. Compare the observed statistic with that null and emit a pass/fail gate, calibrated envelopes, the observed and null summaries, and the decision record. 【author decision: predeclare the envelope calibration rule, heterogeneity statistic, decision threshold, and permutation count】
   - _Why:_ The proposal needs evidence that risk is genuinely heterogeneous; without this gate, a global drift rule and the proposed tail are not mechanistically distinguishable.
3. **Allocate the error budget** (`S3`)
   - Load the passed envelope table, the cache-unit dependency graph, and a per-unit recomputation-weight table keyed by $`unit_{id}`$. Represent each candidate schedule with a refresh flag $`z_{u}`$, where 1 means refresh and 0 means reuse, and pass the selected observation-change scope from S2 to the binary selector. Solve the S3 equation-defined selection so the envelope risk left in reused units stays below the downstream error limit while total recomputation weight is minimized. 【author decision: choose per-level constraints, a distribution-weighted aggregate, or a worst-case aggregate across `d`, and specify how the error limit and matched skip rate are fixed】 Emit one schedule row per unit containing $`unit_{id}`, `z_{u}`$, the selected change-level scope, the applied envelope value, and the recomputation weight, together with the matched skip fraction and selector diagnostics.

*The refresh indicator minimizes recomputation cost while keeping the risk of reused units below the downstream budget.*
$$ \min_{\mathbf{z}}\sum_{u=1}^{U}z_uc_u\quad\text{s.t.}\quad\sum_{u=1}^{U}(1-z_u)r_u(d)\leq B,\quad z_u\in\{0,1\} \tag{2} $$

   - _Why:_ This converts the heterogeneous measurement into the candidate's load-bearing operation rather than another descriptive cache score.

### M2_validation
*Apply dependency-complete refresh and evaluate paired error, native latency, task success, and the permutation control.*

4. **Invalidate descendants** (`S4`)
   - Build the refresh set from the S3 rows whose refresh flag is 1. Using the dependency graph's producer-to-consumer adjacency, compute the S4 equation-defined descendant invalidation set, clear every selected and descendant cache entry, refresh selected units, and recompute invalidated nodes in topological order before evaluating either output head. For every intervention schedule, run matched uncached and control rollouts from identical episode, observation, and random-number states. Record paired downstream action mean squared error, predicted-video mean squared error, the joint downstream error, synchronized native graphics processing unit (GPU) latency, and per-episode task success. Repeat the same selector and matched skip fraction after applying a bijective permutation of complete risk envelopes across unit IDs, and store the paired aggregates, raw assignments, and control result. 【author decision: define the joint error composition, output horizon, episode aggregation, task-success predicate, and exact synchronized native GPU timing boundary】

*The invalidation set contains every graph descendant of a refreshed unit so stale dependencies are not silently reused.*
$$ \mathcal{I}(\mathcal{R})=\{v\mid \exists u\in\mathcal{R}:u\prec v\} \tag{3} $$

   - _Why:_ Dependency-complete recomputation tests whether the allocation survives the actual Fast-WAM graph instead of only improving an isolated intermediate metric.

## Reviewer concerns
- **Concern [non_blocking]:** Paper-pointed threat: openalex:W7203663435 (collision_hits). DriveCache reports that cache tolerance varies with ego translation and rotation, denoising progress, and consecutive reuse length, then uses a training-free action-aware controller with dynamic programming under a calibrated response budget and a causal drift-triggered refresh and replan. This directly threatens any broad novelty claim for action-aware budgeted caching with causal refresh. The supplied abstract does not, however, establish the candidate's narrower per-unit stale-value paired action/video envelope, unit-label heterogeneity null, or dependency-complete descendant invalidation, so it does not establish exact-mechanism overlap; the candidate must explicitly contrast these objects before claiming novelty.
  - **Response:** DriveCache is explicitly treated as the closest collision: its scene-level dynamic-programming controller already combines an action-aware response budget with causal drift-triggered refresh and replan. The bounded claim in `hook`, `falsification_prediction`, and `differentiation_from_lit` is narrower—stale-value, per-unit paired action/video risk envelopes, a unit-label heterogeneity gate, and dependency-complete descendant invalidation—so the Phase 3.2 rationale blocks broad novelty for action-aware budgeted caching but does not establish exact-mechanism overlap.
  - *Fields changed to address:* `differentiation_from_lit`

