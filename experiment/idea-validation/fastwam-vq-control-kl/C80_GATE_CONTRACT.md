# Full C80 reversible optimization check

This is a calibration-only diagnostic. It does not train or select checkpoints on heldout data.

Parent `protocol.json` and child `c80_gate_protocol.json` are frozen. Units use `control_common.guard`, the existing allocation deadline and 15-second GPU telemetry. GPU jobs remain `01:59:00`. `parameter_gate` and CPU-only `gate_selfcheck` must pass before any C80 GPU unit starts. The CPU unit runs the changed parameter, gradient/reducer and finite-evaluator toy checks; it does not validate GPU residency or real-model parity.

## Gradient units (root implementation)

`c80_grad_response_0/1`, `c80_grad_latent_0/1` each process exactly the existing40 cases of the corresponding calibration-pair shard, sorted by case_id, at the original per_channel_clip deployment parameters. Objective keys map through the child protocol. Use existing cached MSE/BF16 clean/noisy endpoints, validate their input bindings, and exact native/dynamic gradient-enabled action/future parity. No parameters change.

The final tensor file is `results/<unit>/gradient.pt`, a weights-only-safe dictionary:

- `protocol`: child protocol string; `parent_protocol`: parent string; `unit`, `pbs_jobid`, `objective`, `shard`, `case_ids` (40 IDs), `complete: true`.
- `parameter_keys`: ordered list of `ratio::<canonical module name>` then `channel::<canonical module name>`; 614 of each kind.
- `originals`: list of CPU FP32 original parameter tensors matching parameter_keys.
- `gradient_sum`: list of CPU FP32 tensors or None, sum of raw per-pair surrogate gradients, same order. None means unused in all shard cases.
- `loss_sum`: sum of the40 unweighted pair objectives; latent pair uses mean(clean,noisy).
- `case_metrics`: list of40 dicts with `case_id`, `loss`.

Receipt `summary.json` also contains parent protocol, unit, pbs_jobid, complete, expected_cases40, completed_cases40, objective, shard, case_ids, native_parity_all_zero, original_parameters_restored, frozen_reader_and_model_unchanged, training_updates0, heldout_inputs_used=false, sensor_evaluation_inputs_used=false. A partial `gradient_state.pt` may be reused after a verified terminal repair only if its full frozen config and originals match. No live job is restarted.

## CPU direction units (root implementation)

`c80_direction_response/latent` combine both corresponding gradient shards, require disjoint full80 IDs and exact same originals, compute mean gradient, apply original norm clip, derive fresh Adam delta, then project fixed fractions1.0 and0.1. These are candidate proposals, never trained deployments.

Each direction receipt `results/<unit>/summary.json` has parent protocol, child `gate_protocol`, unit, matching pbs_jobid, complete, objective, case_ids80, baseline_loss (full80 mean), parameter_keys, source_gradient_jobs, and `candidates` list. Each candidate has `fraction`, `path` (absolute .pt), `surrogate_g_dot_delta`, `projected_step_l2`.

Each candidate .pt has `protocol` (child string), `parent_protocol`, `direction_unit`, `direction_pbs_jobid`, `objective`, `fraction`, `case_ids`80, `ratios` (dict canonical name -> CPU FP32 tensor), `log2_scales` (same). It has no learned model weights or optimizer step history.

## Finite evaluation units (delegated implementation)

`c80_eval_response_0/1`, `c80_eval_latent_0/1` require the matching completed direction receipt and original completed calibration shard. Each uses its original40 clean/noisy pairs and seeds and evaluates both frozen candidate fractions. Use `SwitchingRuntime`, activate per_channel_clip once, restore candidate ratios/log2_scales, call install_compensated_weights from immutable decoded base, set cached=True, then use native `query`. Native initial action noise must match corresponding BF16 calibration cache. All614 linears, ten video and ten action steps remain required by runtime.query.

Response objective is `action_response_kl` of candidate clean/noisy outputs and the same-input original BF16 outputs, with original frozen stats. Latent objective is mean over clean/noisy `latent_control_kl(BF16Reader(candidate future), BF16 native action)`; reader is no-grad, uses retained original GPU BF16 weights, separate KV prefill for each future and same seed. Verify BF16Reader of the cached BF16 future exactly reproduces cached BF16 action for both variants of the first case. `SwitchingRuntime.query` returns `(action, future, counts, action_initial_noise)`; inspect the implementation rather than relying on this note if it differs.

Save per-case candidate results (no large endpoint tensors needed) with full child/config/case/input/seed/proposal binding. Resume only fully matching records. Keep the two fractions distinct. Deadline checkpoints remain incomplete; new repaired jobs may reuse completed matching case records.

Final summary must have parent `protocol`, child `gate_protocol`, unit, matching pbs_jobid, complete, objective, shard, expected_cases40, completed_cases40, case_ids40, `candidates` (fraction, case_ids, cases40, loss_sum, mean_loss), native_and_reader_initial_noise_parity_valid, bf16_reader_native_parity_all_zero (true or not-required), original_parameters_restored, frozen_vq, training_updates0, heldout_inputs_used=false, sensor_evaluation_inputs_used=false. Restore original MSE D/r in finally and verify equality; no trained deployment is saved.

## Final CPU aggregation

Combine both evaluation shards per objective and fraction into actual full80 mean. Compare only to its matching direction baseline and report actual delta/descent next to projected g_dot_delta. An implementation pass does not imply descent; descent does not imply heldout or closedloop improvement. All original heldout80 and paired280 protocols remain unchanged.
