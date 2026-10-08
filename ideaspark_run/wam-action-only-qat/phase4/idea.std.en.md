> **FAILED VALIDATION — not experimentally validated.** The Phase3 post-revision gate remains `needs_work` for F1. The original audit found no full-prefix time reduction or fixed estimator provenance for V_visit. Phase4 adds a supplemental scalar mean over all native executed-prefix timesteps and selected action channels, using one fixed pre-QAT matched-PTQ reference; this does not change or pass the old gate.
> The publication validator also records `kill_switch_integrity` for Phase2-to-Phase4 falsification text drift. Direct comparison shows Phase3 `final_candidate.json` and Phase4 match; the validator comparison does not reflect the authorized Phase3 control-arm revision. The finding is retained.

# Student-Path Suffix QAT for World-Action Models

**Method.** Student-Path Suffix QAT (SPS-QAT)

## Motivation
Strong low-bit results leave it unclear whether training a world-action model (WAM) adds value beyond strong frozen-weight quantization and a matched teacher target. Quantization-aware training (QAT) updates weights while simulating low-bit arithmetic; post-training quantization (PTQ) keeps pretrained weights fixed. Recent WAM work reports strong low-bit control results, so this proposal does not assume W4A8 already fails.

Fast-WAM Base is the initial platform: its roughly 1B ActionDiT can be the weight-scope action-only target while its roughly 5B video expert stays BF16 (16-bit brain floating point). Transfer is only a scoped hypothesis for shared-backbone LingBot-VA and Cosmos Policy. Updating their shared denoiser weights affects video and action computation, so this is action-targeted shared-backbone QAT, not action-only weight quantization.

The proposed operator pairs a BF16 teacher and fake-quantized student from the same detached state the student actually reached, then runs both through the same complete remaining native solver suffix with the same context and remaining noise updates. The teacher endpoint is a model prediction, not original-noise-endpoint KD or a physical oracle. After native action denormalization and executed-prefix selection, training uses a per-channel-scaled continuous command residual before clipping or gripper thresholding; exact clipped or thresholded commands are measured separately.

The testable difference from vanilla QAT plus matched teacher knowledge distillation (KD) is the full remaining-suffix endpoint target from a student-visited state. This is a hypothesis about the target, not a proven novelty or a claim of first WAM QAT. The prediction that completion gains over matched teacher-state KD rise with $`V_{visit}`$ is empirical. $`V_{visit}`$ is a per-task diagnostic on a held-out context subset, estimated using one fixed pre-QAT matched-PTQ reference for every arm, with no post-treatment per-arm ranking. The Phase4 clarification does not pass the earlier gate: status remains post_revision FAILED VALIDATION.

## Method
### M1_background
*Record each model's contract and generate the fake-quant student's own solver path.*

1. For Fast-WAM Base, shared-backbone LingBot-VA, and Cosmos Policy, record each BF16 checkpoint, native solver schedule, action maps and executed-prefix selector, matched branch-free W4A8 modules, excluded precision, and deployment kernel. Fast-WAM's independent ActionDiT is weight-scope action-only; the other two use action-targeted shared-backbone scope.
   - _Why:_ A comparison only isolates weight adaptation when scope, schedule, precision, action interface, and deployed arithmetic stay matched.
2. Split each LIBERO dataset by episode. From training episodes, draw contexts and native initial noise and solver updates, run the fake-quant student through its native schedule, and save the states it visits. These are action latents for Fast-WAM and coupled video/action states for shared-backbone models.
   - _Why:_ The target uses states the quantized student itself reaches; the fixed context and noise protocol supports paired comparisons and does not model observations after physical actions.

### M2_suffix_target
*Train from a student-visited state using paired full-suffix endpoints.*

3. Sample a native solver index, stop gradients through the student-visited state, and restart both BF16 teacher and fake-quant student from that identical state. Give both the same context and remaining native noise updates, run each unchanged full remaining suffix, and keep both endpoint action chunks.

*The visit-response difference is measured after the native action map and exact executed-prefix selector, using the same remaining innovations from a state reached by a fixed matched W4A8 PTQ reference student.*
$$ \delta^{\mathrm{PTQ-ref}}_{m,k,u,j}=\left[E_m\!\left(D_m\!\left(S[Q_{\mathrm{ref}}|Q_{\mathrm{ref}},m,k](x_{Q_{\mathrm{ref}},m,k},c,\eta_{>k})\right)\right)-E_m\!\left(D_m\!\left(S[\mathrm{BF16}|Q_{\mathrm{ref}},m,k](x_{Q_{\mathrm{ref}},m,k},c,\eta_{>k})\right)\right)\right]_{u,j} \tag{1} $$

   - _Why:_ This tests suffix response from a student-visited state, rather than only from the original noise endpoint or a teacher-visited state.
4. Convert both endpoints with each model's native action denormalization and executed-prefix selection. Train on the continuously valued command before clipping or gripper thresholding, scaled by each selected channel's fixed positive controller scale and averaged over every native prefix step and selected channel. Backpropagate through the student suffix only. Separately report exact clipped or thresholded command differences in native units.

*The student suffix is trained on the dimensionless mean squared continuous pre-threshold command residual across every native executed-prefix time step and selected action channel; only the student suffix receives gradient.*
$$ \mathcal{L}_{\mathrm{train},m}=\mathbb{E}_{c,z,\eta,k}\!\left[\frac{1}{T_{\mathrm{exec},m}|J_m|}\sum_{u=1}^{T_{\mathrm{exec},m}}\sum_{j\in J_m}\left(\frac{\tilde y_{Q|Q,m,k,u,j}-\tilde y_{\mathrm{BF16}|Q,m,k,u,j}}{s_{m,j}}\right)^2\right] \tag{2} $$

   - _Why:_ The continuous, dimensionless training signal is differentiable, while the exact command the controller executes is a separate outcome; the two must not be treated as identical.

### M3_validation
*Compare matched training controls, separate outcomes, and test scoped transfer.*

5. Compare with matched branch-free W4A8 PTQ, vanilla action-only QAT plus teacher KD, original-noise-endpoint KD, teacher-state suffix KD, and student-state per-step OPD. Deduplicate teacher-state arms with identical targets. Match either measured training GPU-time or total teacher and student NFE, and report both work totals, examples, context exposure, updates, and teacher-suffix work. In shared-backbone OPD, an action-output loss mask selects action channels and prefix steps; it does not make shared weights action-only.
   - _Why:_ These controls test whether the full student-state suffix endpoint adds value beyond ordinary QAT, PTQ, endpoint or teacher-state distillation, and per-step student supervision, while making teacher cost visible.
6. On held-out fixed contexts and paired noise, report exact executed-action deviation, LIBERO closed-loop completion, and full native action-query latency as separate results; latency includes video/context prefill and no teacher. Estimate scalar $`V_{visit}`$ over the full native prefix time-by-channel response, on a held-out context subset and from the same fixed pre-QAT matched-PTQ reference in every arm. Report measured action-time share before using Amdahl's law for whole-query speed.

*Phase4 supplement: reduce the exact executed-command response over the full native time-by-channel prefix to one scalar, estimated only from a fixed pre-QAT matched-PTQ checkpoint for every arm; this definition is not a passed post_revision gate.*
$$ \mathcal{V}^{\mathrm{PTQ-ref}}_{\mathrm{visit},m}=\mathbb{E}_{c\sim\mathcal{D}^{\mathrm{eval}}_m,z,\eta,k\sim\operatorname{Unif}(K_m)}\!\left[\frac{1}{T_{\mathrm{exec},m}|J_m|}\sum_{u=1}^{T_{\mathrm{exec},m}}\sum_{j\in J_m}\left(\frac{\delta^{\mathrm{PTQ-ref}}_{m,k,u,j}}{s_{m,j}}\right)^2\right] \tag{3} $$

   - _Why:_ Command fidelity, task completion, and deployed latency answer different questions. $`V_{visit}`$ is a fixed-reference diagnostic and an empirical moderator hypothesis, not proof of benefit. The Phase4 clarification does not pass the earlier gate; post_revision status remains FAILED VALIDATION.
7. Run separate transfer checks on shared-backbone LingBot-VA and Cosmos Policy under each model's native scope. Label shared denoiser updates as affecting both video and action computation, and keep their results separate from Fast-WAM.
   - _Why:_ This tests the scoped transfer hypothesis without treating shared weights as weight-scope action-only or pooling distinct model contracts.

## Falsification and measurement

Test whether closed-loop completion gains over matched teacher-state KD rise with task-level `V_visit`. Measure `V_visit` over the full native prefix time-by-channel response on a held-out context subset, using one fixed pre-QAT matched-PTQ reference for every arm; do not rank arms with post-treatment references. Compare matched PTQ, vanilla QAT plus teacher KD, original-noise-endpoint KD, teacher-state suffix KD, and student-state per-step OPD. Also report exact executed-command deviation and full native action-query latency separately. No numeric pass threshold is specified. No completion gain over matched teacher-state KD, or no predicted association with `V_visit`, would leave the proposed prediction unsupported; fidelity or action-block speed alone cannot establish control benefit. The estimator clarification does not pass the earlier gate: status remains post_revision FAILED VALIDATION.

## Resources and feasibility

Original and current campaign cost: unknown GPU-days, because campaign hours, samples per second, and suffix-training throughput were not measured; no GPU experiment was run. The ceiling is at most four separate A100-SXM4-40GB devices, not pooled 160GB. Fast-WAM's roughly 5B BF16 video plus 1B BF16 action core is about 12GB of weights. Converting only the action weights to W4 leaves about 10.5GB total, an idealized 1.14x whole-core weight reduction; it is not a 4x whole-model reduction. These are weight-storage figures, not a training-memory or throughput result. Gradients, optimizer state, teacher and student suffix evaluations, activations, cache, and workspaces add cost; shared-backbone adaptation can add activation and teacher-memory demands while changing video and action computation. LingBot-VA and Cosmos Policy are scoped transfer hypotheses; no LingBot checkpoint size is inferred from separate paper modules, and an action-output mask does not mean action-only weights. Whole-query speed requires a measured action-time share and Amdahl's law. Feasibility remains conditional and no SOTA or GPU-day result is claimed.
