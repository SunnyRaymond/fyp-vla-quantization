# Action-Context Source Masking for Frozen Joint WAMs

**Method.** Action-Context Source Masking (ACSM)

## Motivation
**Problem framing.** Across frozen joint World Action Models (WAMs), iterative inference updates future visual latents and action chunks through coupled transformer paths. The Phase 1 bottleneck is specific to this model class: even after future-side tokens or video updates are reduced, each active native action solver update may still read the full observation and retained context. Existing evidence does not establish either how much action quality depends on those observation-source interactions or what share of full inference cost they consume, so token retention and total latency cannot answer whether action conditioning is a useful target.

Sparse-WAM is the anchor instance: it selects future tokens and reuses omitted-region predictions while retaining all observation/action tokens and recomputing action predictions. Efficient-WAM separately demonstrates an inference-only asymmetric video schedule and video K/V reuse while action updates continue. These works motivate a bounded test of the still-dense action-query-to-observation path. ACSM leaves video-query paths, action-future interactions, the native solver schedule, and frozen weights intact; it tests source-group observation edges only and does not infer that fixed observation latents imply fixed hidden states or K/V.

**Why now.** Recent WAM work makes the residual question concrete: Sparse-WAM (2026-09) reduces future-side computation, WAMachine (2026-09) studies state and residual reuse, and Efficient-WAM (2026-06) separates video and action schedules, yet the Phase 0 summaries leave action-query-to-observation sensitivity and cost unmeasured. This creates a bounded inference-time question that can be addressed without new training by profiling that path and comparing matched-edge masks on frozen models; it does not establish that every required checkpoint, source-provenance interface, or compact kernel is currently available, so those remain start conditions.

**Why prior work stopped.**
- `arxiv:2609.38984v1` (arXiv 2026-09): Sparse-WAM uses action-guided selection of future visual tokens and reuses selected or omitted-region predictions across denoising steps while recomputing action predictions.
  - _Did not do_: It retains all observation/action tokens and does not test whether action queries can drop observation-source edges while preserving task success and reducing full-path latency.
  - _Structural reason_: Its intervention targets future tokens, leaving the action-query-to-observation edge set and its task-cost contribution unmeasured.
- `arxiv:2609.34608v1` (arXiv 2026): WAMachine combines trajectory remapping, execution-period observation rebinding, and probe-gated intermediate residual reuse.
  - _Did not do_: It does not isolate the contribution of current observation-source edges read by each native action update.
  - _Structural reason_: Its mechanism reuses or rebinds intermediate state, rather than ranking and deleting action-query-to-observation edges.
- `openalex:W7164153965` (arXiv (Cornell University) 2026): Efficient-WAM uses an inference-only asymmetric schedule with fewer video updates and cached video K/V while continuing action updates.
  - _Did not do_: It does not establish the task sensitivity or full-path cost share of observation conditioning in those continuing action updates.
  - _Structural reason_: Its compute allocation distinguishes video from action updates, not observation-source edges within the action path.
- `arxiv:2606.08962v1` (arXiv 2026-06): C3ache reuses action-expert denoising residuals across inference chunks using cache-step and refresh choices.
  - _Did not do_: It does not test which current observation groups a joint FastWAM-Joint action update needs.
  - _Structural reason_: Its operator is cross-chunk residual reuse, not within-chunk action-query-to-observation context masking.
- `openalex:W7172338867` (arXiv (Cornell University) 2026): FBFM injects newly observed state constraints and committed action overlap into active flow updates through Jacobian/pseudoinverse feedback.
  - _Did not do_: It does not measure the action-query observation-context cost share or remove source-specific conditioning edges.
  - _Structural reason_: Its intervention corrects the vector field using feedback rather than changing the current action query's observation key set.
- `arxiv:2608.23927v2` (arXiv 2026-08): GlanceWAM overlaps sparse future imagination with execution through asynchronous lookahead and staleness-robust training.
  - _Did not do_: It does not isolate or sparsify the observation sources read by native action updates in a frozen joint WAM.
  - _Structural reason_: Its mechanism changes execution timing and is co-trained for staleness, rather than masking action-conditioning edges at frozen inference.

**What changes when the gap closes.** A positive result would identify, for each tested frozen checkpoint, whether observation-source edges in the action branch can be reduced under the original success gate and whether that path has enough measured headroom to improve full inference latency. This would give Sparse-WAM- and Efficient-WAM-style systems a WAM-specific action-conditioning test and a same-edge-budget comparison against prior scoring recipes; a negative result would bound the tested opportunity rather than establish a general impossibility.

## Method
**Pipeline.** Freeze the native checkpoint, solver, task configuration, and backend separately for FastWAM-Joint/LIBERO and Cosmos3-Edge-Policy-DROID/RoboLab. Profile the complete inference path and stop early if action-query-to-observation QK/AV has insufficient Amdahl headroom. On a dense first action evaluation, group observation tokens by existing provenance and compute output-projected contribution scores; choose epsilon on held-out selection episodes under the original task-success gate and full-path latency. Run later action updates with the resulting compact key/value sets, then compare paired evaluation episodes against dense inference, equal-edge controls, and the same-cost score baselines.

### M0_background
*Fix each baseline and check measured headroom before applying the proposed mask.*

1. **Freeze each baseline** (`S1`)
   - Lock each checkpoint, native solver schedule, task configuration, and backend. Keep FastWAM-Joint/LIBERO and Cosmos3-Edge-Policy-DROID/RoboLab as separate primary pairings; record that the FastWAM-Joint matching checkpoint is unconfirmed, and treat Motus/RoboTwin 2.0 only as a conditional simulation fallback.
   - _Why:_ The estimate is defined per frozen checkpoint and native evaluation path, so changing weights, schedule, task, or backend would change the comparison.
2. **Measure action-path headroom** (`S2`)
   - Profile the native paths in FastWAM's fastwam_joint.py/fastwam.py/mot.py and Cosmos' omni_mot_model.py plus the RoboLab policy-server entry point. Separate action-query-to-observation QK/AV, other attention, QKV, FFN, and mask/packing cost; compute phi_obs and its Amdahl upper bound, and stop if measured savings cannot meet the original latency gate.

*phi_obs is the measured full-inference time share of action-query-to-observation QK/AV; this ideal Amdahl upper bound excludes scoring, packing, kernel, and other model-path costs.*
$$ \mathrm{speedup}_{\max}=\frac{1}{1-\phi_{\mathrm{obs}}} \tag{1} $$

   - _Why:_ The action sequence is not enough to infer cost share; this measurement tests whether the proposed edge deletion can affect complete inference latency at all.

### M1_source_scoring
*Rank sources by output-projected action contribution and freeze a mask using separate held-out episodes.*

3. **Score source groups** (`S3`)
   - Use only the token packer's existing camera/view or other source provenance; if it does not expose provenance, stop rather than infer groups. On each chunk's dense first native action evaluation $k_{ref}$, read native attention alpha, values V, and output-projection head slices $W_{O}$, then compute $r_{lg}$ and $rho_{lg}$; retain all groups if the normalization denominator is zero.

*For each layer and observation-source group, $r_{lg}$ sums the norm of its action-query attention contribution after the native output projection.*
$$ r_{lg}=\sum_{q\in Q_l}\left\|\sum_{h\in H_l}W^{l}_{O,h}\left(\sum_{j\in g}\alpha^{l}_{h}(q,j)V^{l}_{h}(j)\right)\right\|_2 \tag{2} $$


*$rho_{lg}$ normalizes each group contribution within its layer; if the denominator is zero, the rule retains every group.*
$$ \rho_{lg}=\frac{r_{lg}}{\sum_{g'\in G_o}r_{lg'}} \tag{3} $$

   - _Why:_ The score measures the candidate's proposed output-projected source-group contribution while keeping token values, weights, image, instruction, future/action queries, and training objective fixed.
4. **Select and freeze the mask** (`S4`)
   - On held-out selection episodes $D_{select}$, sort groups by descending $r_{lg}$ with native source order breaking ties, and choose epsilon in [0,1) so $S_{l}(epsilon)$ is the shortest prefix whose cumulative $rho_{lg}$ reaches 1-epsilon. Subject to the original frozen task-success gate, select the epsilon with the lowest measured full-path latency; if no pre-existing gate applies, report the success-latency curve without declaring a pass.

*The $g_{i}$ are ordered by descending $r_{lg}$ with native source order breaking ties; $S_{l}$ is the shortest prefix meeting the cumulative-share target, with epsilon selected in [0,1).*
$$ S_l(\epsilon)=\{g_1,\ldots,g_m\},\quad m=\min\left\{t:\sum_{i=1}^{t}\rho_{l g_i}\geq 1-\epsilon\right\} \tag{4} $$

   - _Why:_ A declared task gate constrains the retained context; separating $D_{select}$ from later paired evaluation avoids choosing epsilon from the outcomes used to judge the candidate.

### M2_sparse_update
*Apply the frozen mask through a kernel that actually skips omitted action-to-observation computation.*

5. **Skip omitted observation edges** (`S5`)
   - For later native action solver evaluations in the chunk, send action queries and retained observation keys/values to a true variable-length or block-sparse kernel and verify omitted QK/AV is not executed. Recompute softmax over retained keys, feed the resulting attention output through the original residual/transformer blocks, and let the changed action hidden state/velocity update the original flow state; leave video-query paths, future/action interactions, FFN, and solver schedule native.
   - _Why:_ This implements the proposed graph change in the action branch and distinguishes real edge skipping from a dense mask; the recomputed output may change the action trajectory, with no fidelity or control-sensitivity guarantee.

### M3_validation
*Test the mask against dense inference and matched-cost controls; any robot phase remains conditional.*

6. **Run matched falsification comparisons** (`S6`)
   - On paired evaluation episodes $D_{eval}$ not used to select epsilon, compare dense, $rho_{lg}-ranked$, and same-retained-edge-cost source-group-permutation masks, alongside native attention-mass, ToPi/VATP value-norm, and CAPA output-projected score controls. Use the same frozen checkpoint, $k_{ref}$ dense forward, exact retained edge budget, and compact kernel; record task success, per-path FLOPs, full latency including scoring/packing/kernel launch, action-chunk deviation, and first action divergence, then apply the original frozen gates. Require a legally matched-cost source-group permutation; if existing provenance groups do not permit one, report that control as unavailable rather than substitute an unequal-cost arm.

*$B_{l}$ is the retained action-query-to-observation edge budget used to match ranked masks and scoring controls.*
$$ B_l=|Q_l||H_l|\sum_{g\in S_l(\epsilon)}|g| \tag{5} $$

   - _Why:_ The permutation tests whether source assignment matters at fixed cost, while the prior-score arms test whether $rho_{lg}$ adds WAM-specific task-action value beyond known contribution-ranking recipes; profiling and task outcomes remain separate evidence.
7. **Keep robot evaluation conditional** (`S7`)
   - Discuss a physical-robot phase only after simulation passes the frozen gate and a matching AgileX Cobot Magic policy/checkpoint, demonstrations/data, source implementation, and same-type hardware are confirmed. Account for any baseline post-training separately; the mask itself remains training-free and does not update weights.
   - _Why:_ The stated physical prerequisites are unconfirmed, so simulation or recorded evidence cannot stand in for a closed-loop robot result.

## Reviewer concerns
- **Concern [non_blocking]:** Paper-pointed threat: arxiv:2602.01609v1 (n/a). ToPi（supplemental_primary，§4.2–4.3；https://arxiv.org/html/2602.01609v1）在冻结 diffusion transformer 上按 target-to-context 的 attention/value contribution 排序，保留累计贡献达到 $\tau $ 的最小 context 子集，在 anchor denoising steps 重评分并在其间复用、压紧该子集；这已覆盖本候选“读出贡献分数$\to $累计阈值选 $context\to $跨 solver steps 复用”的通用 recipe。ToPi 的对象是图像编辑中的 target latents 与 reference tokens，并未检验 WAM $action-query\to observation-source$ edges、机器人 task success 或这一 action 路径的完整时延，故它不是同一 WAM 机制的 exact overlap。CAPA（supplemental_primary，§4.1 Eq.(2)；https://arxiv.org/html/2602.00247v1）另已使用经 $W_{O}$ 投影、取 residual contribution norm 的视觉 token 分数，因此投影本身也不能支撑新颖性；当前候选仍可主张 WAM-specific 边选择，但必须在相同保留边数和任务结果下显示其分数优于这些先前分数。
  - **Response:** Phase 3.2's verdict_rationale says the cumulative selection recipe and output projection are prior art but finds no exact WAM overlap, so the candidate does not claim novelty from those ingredients alone. Its revised falsification_prediction adds same-checkpoint, $same-k_{ref}$, same-edge-budget controls for native attention mass, ToPi/VATP value-norm, and CAPA-style projected scores using the same compact kernel. A WAM-specific score advantage remains conditional on paired task outcomes beating those controls while passing the original gates; no such result has been measured.
  - *Fields changed to address:* `falsification_prediction`

