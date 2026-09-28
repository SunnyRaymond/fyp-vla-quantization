# Causal Risk-Envelope Caching for Fast-WAM Optional-IDM

**Method.** Causal Risk-Envelope Cache (CREC)

## Motivation
Fast-WAM Optional-IDM repeatedly processes new, receding observations. The problem is not merely repeated computation: changing an observation can make a cached unit affect both the action output and the predicted-video output, and two units with similar hidden-state drift can have very different downstream effects. A single local-drift threshold treats all units as interchangeable even when a small high-risk tail accounts for most paired error.

The proposed setting keeps the existing Fast-WAM model and graph fixed and measures this asymmetry directly. C3ache provides a periodic World-Action caching reference, EfficientVLA provides a feature-similarity reuse reference, and WorldCache and X-Cache show that heterogeneous or structure-aware reuse can preserve rollout quality. As represented here, none measures the downstream paired action/video risk of making one cache unit stale and uses that measurement to decide which dependencies must be refreshed. The question is therefore whether causal risk allocation, rather than another similarity score or fixed cadence, is the missing operation.

Recent caching work makes this measurement timely. C3ache, WorldCache, and X-Cache supply relevant reuse references, while DriveCache is a boundary case that already combines scene-level scheduling, an action-aware response budget, and causal drift-triggered refresh or replanning. CREC therefore makes no broad claim about action-aware budgeted caching. The bounded question is whether Fast-WAM Optional-IDM has a distinct per-unit paired action/video risk envelope, whether its heterogeneity can be tested, and whether descendant-complete invalidation can be evaluated with the existing checkpoint and native GPU timing.

If the gap is closed, Fast-WAM caching gains an explicit allocation object: at a fixed skip rate, it can refresh the empirically high-risk tail and reuse the bulk instead of treating local drift as sufficient. Paired action/video error and closed-loop task success become part of the cache validity boundary. A successful gate would also provide a reusable causal diagnostic for deciding when a latency gain is mechanistically meaningful, while complementing rather than claiming to subsume the cited reuse mechanisms.

## Method
### Background
*Set up the existing Fast-WAM Optional-IDM graph and obtain per-unit intervention measurements.*

1. Start with held-out replay records keyed by episode, time, and observation. Use the cache-unit registry to load each unit's stable $`unit_{id}`$, producer node, tensor shape and type, cached tensor, and uncached reference tensor. For one unit at one observation change, replace only that cached tensor and keep the same receding observation, model parameters, random-number state, and all other parts of the Fast-WAM Optional Inverse Dynamics Model (IDM) graph fixed. 【author decision: choose whether a cache unit is a whole intermediate tensor, a graph-node output, or a finer tensor block, and freeze its registry and stable identifier before replay】 Run the action and predicted-video heads and save both outputs together with the episode, time, unit, and intervention metadata. Compute paired action and predicted-video deviations over matching output elements, then group the records by observation-change level `d`. 【author decision: define the representation, normalization, distance or norm, and fixed strata for `d` before calibration】 Use the estimator defined for step 1 (Start with held-out replay $records\dots )$ as an empirical conditional mean of the weighted paired deviations to produce $`r_{u}(d)`$. Save one row per unit and change level with the unit ID, change level, sample count, both component deviations, and the envelope estimate. 【author decision: set and provenance the action and predicted-video weights, including whether they are equal, task-unit normalized, or selected on a separate calibration split】

*The unit-specific envelope combines paired action and predicted-video downstream deviations at an observation-change level.*
$$ r_u(d)=\mathbb{E}\left[\lambda_a\Delta_a+\lambda_v\Delta_v\mid u,d\right] \tag{1} $$

   - _Why:_ This supplies the missing causal intervention: it separates local hidden-state change from the downstream effect of one stale unit.

### M1_risk_envelope
*Test heterogeneous paired downstream risk and turn the calibrated risk envelope into a refresh allocation.*

2. Group the step 1 (Start with held-out replay $records\dots )$ rows by unit ID and change level, and recompute each unit's envelope with the same paired-deviation estimator. Build the unit-label permutation null by shuffling whole unit labels while keeping each risk value attached to its change level; recompute the chosen heterogeneity statistic for every shuffle. Compare the observed statistic with the null distribution and save the pass/fail gate, calibrated envelopes, the observed statistic, the null summary, and the decision record. 【author decision: predeclare how envelopes are calibrated, which heterogeneity statistic is used, the decision threshold, and the number of permutations】
   - _Why:_ The method needs evidence that risk is genuinely heterogeneous. Without this gate, a global drift rule and the proposed high-risk tail cannot be distinguished by mechanism.
3. Load the passed envelope table, the cache-unit dependency graph, and a per-unit recomputation-weight table keyed by $`unit_{id}`$. Use a refresh flag `z_u`: 1 means refresh and 0 means reuse. Pass the step 2 (Group the S1 rows by unit ID $and\dots )$ change-level scope to the binary selector. Solve the stated selector so the risk left in reused units stays below the chosen downstream error limit while total recomputation weight is minimized. 【author decision: choose per-level constraints, a distribution-weighted aggregate, or a worst-case aggregate across `d`, and specify how the error limit and matched skip rate are fixed】 Save one row for every unit with its unit ID, refresh flag, selected change-level scope, applied envelope value, and recomputation weight, plus the matched skip fraction and selector diagnostics.

*The refresh indicator minimizes recomputation cost while keeping the risk of reused units below the downstream budget.*
$$ \min_{\mathbf{z}}\sum_{u=1}^{U}z_uc_u\quad\text{s.t.}\quad\sum_{u=1}^{U}(1-z_u)r_u(d)\leq B,\quad z_u\in\{0,1\} \tag{2} $$

   - _Why:_ This turns the heterogeneous measurement into the load-bearing cache operation rather than another descriptive cache score.

### M2_validation
*Apply dependency-complete refresh and evaluate paired error, native latency, task success, and the permutation control.*

4. Take the units whose step 3 (Load the passed envelope table) refresh flag is 1 as the refresh set. Follow the producer-to-consumer edges in the dependency graph to compute the descendant set defined for step 4 (Take the units whose $S3 refresh\dots )$; clear every selected and descendant cache entry, refresh the selected units, and recompute invalidated nodes in topological order before running either output head. For each schedule, use matched uncached and control rollouts with the same episode, observation, and random-number states. Measure paired downstream action mean squared error, predicted-video mean squared error, the joint downstream error, synchronized native graphics processing unit (GPU) latency, and task success for each episode. Then apply a bijective permutation of the complete risk envelopes across unit IDs, run the same selector at the same skip fraction, and save the paired aggregates, raw assignments, and permutation-control result. 【author decision: define the joint error composition, output horizon, episode aggregation, task-success predicate, and exact synchronized native GPU timing boundary】

*The invalidation set contains every graph descendant of a refreshed unit so stale dependencies are not silently reused.*
$$ \mathcal{I}(\mathcal{R})=\{v\mid \exists u\in\mathcal{R}:u\prec v\} \tag{3} $$

   - _Why:_ Recomputing all descendants tests the allocation in the actual Fast-WAM graph instead of relying on an isolated intermediate metric.

