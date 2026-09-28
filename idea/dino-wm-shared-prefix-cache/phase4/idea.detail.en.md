# RankSafe Prefix Cache for DINO-WM CEM Planning

**Method.** RankSafe Exact-Prefix Cache (RankSafe-EPC)

## Motivation
**Problem framing.** DINO-WM's test-time CEM planner evaluates multiple counterfactual action sequences from one observed history. The visual/history prefix before the action-conditioned dynamics suffix is shared across those candidates, but a naive implementation recomputes it for every branch. The bottleneck is therefore not merely model inference time: any reuse rule must preserve the ordering of candidate costs that determines the selected first action. A cache that saves FLOPs while silently changing that ordering can alter closed-loop behavior even when individual predictions look close.

$C^{3}ache$ and X-Cache make reuse decisions from residual similarity or action-aware fingerprints, and WorldCache uses curvature and drift to trade rollout fidelity for speed. These works establish that computation reuse is useful, but their validity signals are not an exact graph-level statement about which computation is independent of counterfactual actions. RankSafe-EPC treats the DINO-WM Wall planner as the primary instance of this gap: it isolates an action-independent prefix, leaves the suffix untouched, and tests the resulting candidate-order and first-action invariance directly.

**Why now.** Recent cache systems such as $C^{3}ache$, X-Cache, and WorldCache make reuse practical but stop short of planner-selection invariance. The existing DINO-WM Wall checkpoint and local trajectories provide a fixed, reproducible setting in which graph instrumentation, exact-prefix reuse, native latency, and first-action agreement can be measured without retraining; 2–4 A100s are sufficient for the bounded implementation and matched evaluation. This makes the missing contract testable now, while keeping the claim explicitly provisional until native end-to-end and closed-loop evidence is obtained.

**Why prior work stopped.**
- `arxiv:2606.08962v1` (arXiv 2026): $C^{3}ache$ reuses same-denoising-step residuals across smooth consecutive WAM chunks and reports task-success degradation as its main safety signal.
  - _Did not do_: It did not identify an exact observation/history prefix shared by counterfactual CEM branches or test preservation of their candidate-cost ordering.
  - _Structural reason_: Its residual-similarity construction is tied to smooth executed chunks, not an action-independence analysis of the planner's forward graph.
- `arxiv:2604.20289v2` (arXiv 2026-04): X-Cache gates cross-chunk residual reuse with structure- and action-aware fingerprints while protecting persistent KV updates.
  - _Did not do_: It did not certify equality of the planner-relevant prefix across a CEM candidate set or selected-action preservation under exact reuse.
  - _Structural reason_: Fingerprint gating supplies an approximate similarity test, whereas the missing mechanism is a graph-level action-independence contract with a planner-level consequence.
- `dblp:journals/corr/abs-2603-06331` (arXiv 2026): WorldCache adapts token skipping from curvature and drift to maintain video-rollout fidelity.
  - _Did not do_: It did not retain all action-conditioned suffix computation while targeting CEM candidate ranking and first-action selection.
  - _Structural reason_: Its fidelity proxy is defined for rollout content, not for the ordering statistic that a sampling-based planner uses to select an action.
- `arxiv:2506.10100v1` (arXiv 2025): EfficientVLA combines layer pruning, visual-token selection, and temporal action-head feature caching in CogACT.
  - _Did not do_: It did not define an exact shared-prefix object or validity condition for DINO-WM's counterfactual CEM branches.
  - _Structural reason_: Its compression changes the represented computation, so it does not isolate planner-preserving reuse by an action-dependency boundary.
- `arxiv:2411.04983v2` (arXiv 2024): DINO-WM uses pretrained visual features and test-time action-sequence optimization for zero-shot planning.
  - _Did not do_: It did not expose a reusable prefix object or a cache-validity test tied to candidate-cost and selected-action invariance.
  - _Structural reason_: The original planner evaluates each counterfactual branch directly and provides no graph instrumentation for separating action-independent from action-dependent computation.

**What changes when the gap closes.** A successful result would turn shared-prefix reuse into a planner-level operation with an explicit condition for preserving CEM candidate ordering and the selected first action, rather than a similarity heuristic whose downstream effect is unknown. It would also give systems such as $C^{3}ache$ and X-Cache a concrete diagnostic boundary for separating safe exact reuse from approximate cross-chunk reuse, while making native latency and closed-loop agreement jointly measurable.

## Method
**Pipeline.** Given one DINO-WM Wall observation and a CEM candidate population, instrument the forward graph to locate the boundary before action-dependent computation and record its binary mask. Compute the observation/history prefix once and reuse it only when the mask certifies candidate independence; evaluate every action-conditioned suffix normally. Compare the cached and full candidate costs, ordering, first action, closed-loop outcome, and native end-to-end latency, with a canary mismatch fallback for the valid cache path. A matched randomized-mask control tests whether any downstream change is caused by violating the proposed invariance rather than by caching alone.

### M1_mechanism
*Identify the exact action-independent boundary and reuse the shared prefix while preserving each action-conditioned suffix.*

1. **Trace the action boundary** (`S1`)
   - Treat the action vector as an explicit input node and enumerate tensor-valued graph boundaries in topological order from the observation/history encoder toward the rollout head. For each boundary tensor, run the existing forward graph in evaluation mode for every action vector in the current CEM population and use automatic differentiation to obtain the Jacobian of each boundary element with respect to every action component; set its binary mask entry to 1 iff any derivative is nonzero, then combine the masks over candidates. Select the latest topologically ordered boundary whose combined mask is all zero, and record its module name, tensor shape, data type, device, and binary mask. If no such boundary exists, record `no_certified_boundary` and use the full path for this population. Rebuild the mask whenever the observation/history or the CEM population changes.

*The binary mask marks graph nodes at boundary b whose representation depends on the candidate action.*
$$ M_b = \mathbf{1}\{\partial h_b / \partial a \neq 0\} \tag{1} $$

   - _Why:_ The bottleneck is repeated shared-prefix work, and an explicit dependency mask supplies the missing exact validity condition.
2. **Cache the invariant prefix** (`S2`)
   - For a certified boundary, run the shared prefix once from the current observation/history using the same evaluation mode, data type, device, model weights, and preprocessing as the full path. Store the boundary tensor with a cache key containing the observation/history identifier, checkpoint revision, boundary identifier, shape, data type, and device. On a cache hit, expose that tensor as read-only to every candidate-specific suffix, pairing each suffix call with its own candidate action and leaving all suffix parameters unchanged. Invalidate the entry when any key field changes, and reject in-place writes so one suffix cannot alter the representation seen by another.

*When the boundary is action-independent, one cached prefix representation is valid for every candidate j.*
$$ M_b=0 \Longrightarrow h_b^{(j)} = h_b^{\mathrm{cache}} \quad \forall j \in \{1,\ldots,K\} \tag{2} $$

   - _Why:_ Exact reuse removes duplicated computation without altering the action-conditioned part that determines counterfactual outcomes.

### M2_validation
*Guard candidate evaluation and test planner invariance, downstream decisions, and native latency.*

3. **Guard candidate evaluation** (`S3`)
   - Run every action-conditioned suffix and retain its predicted observation sequence and per-horizon loss values. Use the planner-supplied rollout loss `ell` against the matching reference observation at each horizon step, sum those losses into one scalar objective per candidate, and retain the candidate order. 【author decision: specify `ell`, the reference-observation source, normalization, and the horizon convention; these choices determine candidate ordering.】 Use the first candidate in the fixed population order as a deterministic canary: run it through the uncached full graph, compare its boundary tensor and final objective with the cached result using bitwise tensor equality (`torch.equal`), and on any mismatch invalidate the current cache entry and recompute every candidate through the full graph. For the separate negative control, flatten $`M_{b}`$, apply a seeded uniform permutation that preserves the number of one entries, redraw the seed if the permutation is unchanged, disable fallback, and run the same suffix and objective pipeline; record the seed, corrupted mask, per-candidate objectives, and mismatch flag.

*CEM ranks candidates by their rollout cost after the shared prefix and candidate-specific suffix are evaluated.*
$$ J_j = \sum_{\tau=1}^{H} \ell(\hat{o}_{j,\tau}, o_{t+\tau}), \qquad \pi = \operatorname{argsort}(J_{1:K}) \tag{3} $$

   - _Why:_ The provenance threat is that a self-contaminated gate can hide errors; an explicit canary and a matched corrupted-mask control expose that failure mode.
4. **Test planner invariance** (`S4`)
   - Run the same CEM candidate population, random seed, initial history, checkpoint, preprocessing, rollout horizon, and action bounds through three paths: full recomputation, the valid-mask cached path (RankSafe), and the randomized-mask control. For each planner population, stable-sort candidates by their scalar objectives, breaking ties by candidate index; mark rank disagreement as 1 iff the two complete orderings differ, and mark first-action agreement as 1 iff the first action vectors selected by the lowest-objective candidates are elementwise equal. Aggregate each binary metric over all evaluated populations by arithmetic mean while retaining per-population records. Execute closed-loop episodes from identical initial states and seeds, and apply the existing task evaluator to obtain success outcomes. 【author decision: define the task success predicate, episode termination conditions, and outcome aggregation rule; without them closed-loop success is not reproducible.】 Measure native end-to-end latency from planner-call entry to returned first action with the same warm-up and device-synchronization policy for every path, retain every call duration, and report the arithmetic mean and median. 【author decision: fix whether input preprocessing, synchronization, and environment interaction are included in that timing boundary; this choice defines native latency.】

*The mechanism check records rank disagreement and requires the selected first action to match the full recomputation.*
$$ \Delta_{\mathrm{rank}} = \mathbf{1}[\pi_{\mathrm{cache}} \neq \pi_{\mathrm{full}}], \qquad a^*_{\mathrm{cache}} = a^*_{\mathrm{full}} \tag{4} $$

   - _Why:_ The proposal is about preserving the planner's selection quantity, so speed without ordering and action agreement is insufficient evidence.

## Reviewer concerns
- **Concern [non_blocking]:** Paper-pointed threat: openalex:W7172558534 (collision_hits). The Gate, Not the Cache reports that at skip ratio 0.9 on LIBERO-Object, reuse and deletion fall to 0.68 and 0.31 respectively versus dense 1.00 when the next gate is harvested from the model's own accelerated forward; the collapse is invisible to the evaluated action-level detectors. This threatens any cache whose validity signal is self-contaminated across control steps. RankSafe instead computes a full candidate canary under the deterministic contract, falls back on mismatch, and directly measures first-action agreement and closed-loop success, so the hit is a provenance threat already covered by the candidate's guard rather than exact-mechanism overlap.
  - **Response:** The provenance concern is addressed in `method_flow.steps` S3: the valid cache path uses a deterministic canary and falls back on mismatch, while the randomized-mask control disables fallback so gate contamination cannot be hidden. The `falsification_prediction` and `sub_claims.c3` fields make the required evidence downstream—rank inversions, first-action agreement, and closed-loop success—not merely a change in the gate itself, matching the Phase 3.2 verdict_rationale.

