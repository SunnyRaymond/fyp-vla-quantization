# Action-Context Source Masking for Frozen Joint WAMs

**Method.** Action-Context Source Masking (ACSM)

## Motivation
In frozen joint World Action Models (WAMs), reducing future-side computation does not tell us whether each action update can stop reading the full observation. WAMs repeatedly update action chunks through coupled transformer paths while conditioning on observations and retained future tokens; each active native action solver update may still read all observation tokens. Existing evidence measures neither how action quality depends on those source interactions nor what share of full-inference time they consume, so token retention and end-to-end latency alone cannot answer whether this path is a useful target.

Recent work makes a bounded inference-time test possible: Sparse-WAM (2026-09) reduces future-side computation, WAMachine (2026-09) studies state and residual reuse, and Efficient-WAM (2026-06) separates video and action schedules. The remaining question is action-query-to-observation sensitivity and cost. This proposal profiles that path and compares masks with equal query-key/attention-value (QK/AV) edge counts on frozen models. It does not establish that the required checkpoints, source-provenance interface, or compact kernels are available.

Prior methods stop at neighboring mechanisms. Sparse-WAM selects future visual tokens and reuses selected or omitted-region predictions, while retaining all observation/action tokens and recomputing actions. WAMachine remaps trajectories, rebinds observations during execution, and reuses intermediate residuals after probes, but does not isolate current observation-source edges used by each action update. Efficient-WAM reduces video updates and caches video keys and values while continuing action updates, but leaves their observation sensitivity and cost unmeasured. C3ache reuses action-expert denoising residuals across chunks; FBFM injects new-state constraints and committed-action overlap through feedback; GlanceWAM overlaps sparse future imagination with execution and trains for staleness. These works do not rank and remove source-specific observation edges during frozen inference.

A positive result would show, for each tested frozen checkpoint, whether observation-source edges can be reduced while meeting the original task-success gate and whether measured headroom remains to lower full-inference latency. It would add a WAM-specific action-conditioning test and a same-edge-budget comparison with earlier scoring recipes. A negative result would bound only the tested opportunity, not prove a general impossibility.

## Method
### M0_background
*Fix each baseline and check measured headroom before applying the proposed mask.*

1. Freeze the named checkpoint, native solver schedule, task configuration, and backend separately for FastWAM-Joint/LIBERO and Cosmos3-Edge-Policy-DROID/RoboLab. Record that the matching FastWAM-Joint checkpoint is unconfirmed. Treat Motus/RoboTwin 2.0 only as a conditional simulation fallback, not as the matching primary baseline.
   - _Why:_ The comparison is defined for a particular frozen checkpoint and native evaluation path; changing its weights, schedule, task, or backend changes what is being measured.
2. Profile the named native paths: FastWAM's `fastwam_joint.py`, `fastwam.py`, and `mot.py`; Cosmos' `omni_mot_model.py` and the RoboLab policy-server entry point. Measure time and FLOPs separately for action-query-to-observation query-key/attention-value (QK/AV), other attention, QKV, feed-forward network (FFN), and mask/packing. Compute $`\phi _{obs}`$, the measured full-inference time share of action-query-to-observation QK/AV, and its ideal Amdahl upper bound. Stop if attainable savings cannot meet the original latency gate.

*phi_obs is the measured full-inference time share of action-query-to-observation QK/AV; this ideal Amdahl upper bound excludes scoring, packing, kernel, and other model-path costs.*
$$ \mathrm{speedup}_{\max}=\frac{1}{1-\phi_{\mathrm{obs}}} \tag{1} $$

   - _Why:_ Action-update count does not reveal the targeted path's share of complete inference time; the profile checks whether deleting its edges could affect full-path latency.

### M1_source_scoring
*Rank sources by output-projected action contribution and freeze a mask using separate held-out episodes.*

3. Use only existing camera/view or other source provenance from the token packer; stop if none is exposed. At each chunk's dense first native action evaluation, $`k_{ref}`$, read native attention weights $\alpha $, values V, and output-projection head slices $`W_{O}`$. For each layer and source group, sum the norms of that group's action-query contribution after the native output projection to obtain $`r_{lg}`$, then normalize within the layer to obtain $`\rho _{lg}`$. If the normalization denominator is zero, retain every group.

*For each layer and observation-source group, $r_{lg}$ sums the norm of its action-query attention contribution after the native output projection.*
$$ r_{lg}=\sum_{q\in Q_l}\left\|\sum_{h\in H_l}W^{l}_{O,h}\left(\sum_{j\in g}\alpha^{l}_{h}(q,j)V^{l}_{h}(j)\right)\right\|_2 \tag{2} $$


*$rho_{lg}$ normalizes each group contribution within its layer; if the denominator is zero, the rule retains every group.*
$$ \rho_{lg}=\frac{r_{lg}}{\sum_{g'\in G_o}r_{lg'}} \tag{3} $$

   - _Why:_ The score measures each observation source's output-projected contribution while keeping the dense reference inputs, weights, image, instruction, future/action queries, and training objective fixed.
4. On held-out selection episodes $`D_{select}`$, order groups by descending $`r_{lg}`$ and break ties by native source order. Choose $\epsilon $ in [0,1) so $`S_{l}(\epsilon )`$ is the shortest prefix whose cumulative $`\rho _{lg}`$ reaches $1- \epsilon $. Among masks that pass the original frozen task-success gate, select the one with the lowest measured full-path latency. If no pre-existing gate applies, report the success-latency curve without declaring a pass.

*The $g_{i}$ are ordered by descending $r_{lg}$ with native source order breaking ties; $S_{l}$ is the shortest prefix meeting the cumulative-share target, with epsilon selected in [0,1).*
$$ S_l(\epsilon)=\{g_1,\ldots,g_m\},\quad m=\min\left\{t:\sum_{i=1}^{t}\rho_{l g_i}\geq 1-\epsilon\right\} \tag{4} $$

   - _Why:_ The frozen success gate constrains how much context can be removed, and keeping $`D_{select}`$ separate from paired evaluation prevents choosing $\epsilon $ from the outcomes used to judge the candidate.

### M2_sparse_update
*Apply the frozen mask through a kernel that actually skips omitted action-to-observation computation.*

5. For later native action-solver evaluations in the same chunk, send action queries and retained observation keys/values to a true variable-length or block-sparse kernel, and verify omitted QK/AV work is not executed. Recompute softmax over retained keys, pass the attention output through the original residual/transformer blocks, and let the changed action hidden state and velocity update the original flow state. Leave video-query paths, future/action interactions, FFN, and solver schedule native.
   - _Why:_ This implements edge skipping in the action branch and distinguishes actual saved work from a dense mask. Recomputed attention may change the action trajectory; the method gives no fidelity or task-success guarantee.

### M3_validation
*Test the mask against dense inference and matched-cost controls; any robot phase remains conditional.*

6. On paired evaluation episodes $`D_{eval}`$ not used to choose $\epsilon $, compare dense inference, $`\rho _{lg}`-ranked$ masks, and same-retained-edge-cost source-group-permutation masks. Also compare native attention-mass, ToPi/VATP value-norm, and CAPA output-projected-score controls. Use the same frozen checkpoint, $`k_{ref}`$ dense forward, exact retained-edge budget $`B_{l}`$, and compact kernel. For score controls, sum token scores within existing groups, rank groups with native source order breaking ties, keep whole ranked groups, then add a boundary-group token prefix ordered by the same token score until exactly $`B_{l}`$ edges are retained; token ties use native token order. Run a permutation only if it is a distinct legal equal-cost mask; otherwise report it unavailable. Record task success, per-path FLOPs, full latency including scoring, packing, kernel launch and remaining model paths, action-chunk deviation, and first action divergence; then apply the original frozen gates.

*$B_{l}$ is the retained action-query-to-observation edge budget used to match ranked masks and scoring controls.*
$$ B_l=|Q_l||H_l|\sum_{g\in S_l(\epsilon)}|g| \tag{5} $$

   - _Why:_ The permutation tests source assignment at fixed cost; score controls test whether $`\rho _{lg}`$ adds task-action value beyond prior ranking recipes. Quality and cost remain separate evidence. A WAM-specific score advantage requires $`\rho _{lg}`$ to beat every matched-cost score arm under the original gates.
7. Consider a physical-robot phase only after simulation passes the frozen gate and the matching AgileX Cobot Magic policy/checkpoint, demonstrations/data, source implementation, and same-type hardware are confirmed. Account for baseline post-training separately; the mask itself remains training-free and does not update weights.
   - _Why:_ Physical prerequisites are unconfirmed, so simulation or recorded-data results cannot stand in for a closed-loop robot result.

