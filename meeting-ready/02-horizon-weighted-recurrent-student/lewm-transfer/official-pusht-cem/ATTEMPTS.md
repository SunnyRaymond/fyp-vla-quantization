# Submission attempts

## 25460673.pbs101

- PBS state: `F`; `Exit_status=1`; walltime `00:00:02`.
- No Stage 1 summary or call-record file was created. The 900-byte `job.log` and PBS status files remain in the remote artifact directory for this job.
- The runner stopped at its output-directory guard before loading models or starting inference. The PBS wrapper had already created `job.log`, `gpu_info.csv`, and `execution_identity.txt`, which the original guard incorrectly treated as prior experiment results.
- Corrective change: Stage 1 and Stage 2 now reject only their own pre-existing result files (`stage1_summary.json` / `stage1_call_records.json`, or `stage2_episodes.jsonl` / `stage2_summary.json`). PBS-created log and identity sidecars are allowed. Scientific settings, seeds, schedules, gates, and checkpoints are unchanged.

The next Stage 1 attempt will use a fresh PBS job ID and output directory. The first attempt is a runner startup failure, not an experimental result.

## Retry 25461873.pbs101

- PBS state: `F`; `Exit_status=1`; walltime `00:02:43`.
- The runner loaded the models and completed the first 30-round CEM solve, but wrote no Stage 1 summary or call records. The 6.4KB `job.log` and PBS status files remain in this job's remote artifact directory.
- The failure was in the post-solve final-plan quality audit: it re-prepared raw `world.infos`, then called `JEPA.get_cost` without CEM's candidate-sample axis. That sliced the goal image to rank 3, while the ViT encoder requires batched `[B,C,H,W]` pixels.
- Corrective change: capture the actual expanded `info_dict` at the first solver `get_cost`, select its repeated sample 0, score the final plan as one candidate `[B,1,T,10]`, and persist all relevant prepared-info/candidate/cost input shapes. A capture-only proxy on the native path delegates the original inputs and return value unchanged, so native and routed fidelity paths both use the same actual-solver snapshot. The pinned `JEPA.get_cost` and CEM source confirm these dimensions.

This partial solver execution does not satisfy or fail the frozen Stage 1 gate and is not included in any quality or latency estimate.

## Retry 25469445.pbs101

- PBS state: `F`; `Exit_status=0`; walltime `00:03:21`. The GPU telemetry log contains 41 samples. Artifacts remain under `artifacts/25469445.pbs101/`.
- Submitted once after fixing the startup sentinel and final-plan audit input path. The native teacher reference uses a read-only `get_cost` capture proxy that delegates the exact objects and return value to the original teacher implementation; all four strategy arms and both fidelity paths audit the final plan from their captured first solver cost input.
- Frozen checks: native/routed teacher equivalence, schedule, quality, and latency passed. Validity failed because exact equality of policy-preprocessed observations was false in all eight paired blocks, although the saved raw initial state and goal values match exactly across paths. The summary does not persist per-key observation equality or tensor values, so the differing field cannot be identified from this run.
- The overall result is `FAIL_CLOSED`; it does not satisfy the Stage 1 entry gate. The reported quality and latency values are descriptive only because the paired observation gate failed. Stage 2 was not submitted. The two earlier failed jobs remain startup/audit diagnostics, not experimental results.
- At the time, the source audit was incomplete: it identified the wrapper sample but did not follow the next assignment that overwrites it with NaNs. Freeze revision 2 therefore seeded the action space and kept all keys, but the action comparison still used `torch.equal` on intentional NaNs.

## Retry 25476081.pbs101

- Submitted once after synchronizing freeze revision 2, the action-space pairing correction, and the archived `25469445.pbs101` FAIL_CLOSED evidence. SSH host keys were verified; pre-submit `qstat -u yguo017` was empty. It queued briefly, then ran on `x1000c0s3b0n1` and finished `F`, `Exit_status=0`, walltime `00:03:18`.
- Frozen Stage 1 gates were unchanged. Native/routed teacher equivalence, exact 30-round schedules, quality, and latency passed. Validity failed: for all eight seeds and all six compared paths, `pixels`, `goal`, `state`, `goal_state`, and `proprio` matched exactly, while prepared `action` remained unequal, despite calling `world.envs.envs[0].action_space.seed(seed)` before every reset.
- Descriptive fixed-observation results: median standardized late-minus-student final teacher cost `-2.5027053`; late was strictly lower on 8/8 seeds. Mean solve time was `0.494719 s` for late7 and `2.177604 s` for teacher-only, a `77.28%` reduction under the frozen calculation. These are not a valid paired quality/latency conclusion because the action observation-pair gate failed.
- PBS `job_status` and runner status are both zero; `job.log` contains 40 GPU telemetry samples. Stage 2 was not submitted. Full remote summary/call-record artifacts remain under `artifacts/25476081.pbs101/`.
- Follow-up: preserve this FAIL_CLOSED result and run the frozen, CPU-only action-reset diagnostic under `diagnostics/action-space-reset/`; do not exclude the action key or run CEM.

## CPU action-reset diagnostic 25481553.pbs101

- Submitted once after synchronizing the frozen diagnostic manifest, runner, and PBS script. Pre-submit `qstat -u yguo017` was empty; host-key verification succeeded.
- Initial status: `R` in `qdev`, on `x1001c6s4b1n1` with 2 CPUs, walltime `00:20:00`; started at 13:46:14 on 2026-09-23.
- The bounded job performs at most 12 official simulator resets (two already-used Stage 1 seeds across six path labels), with no model loading, GPU, CEM, or episode evaluation. It records wrapper/action-space identities, seed/sample events, and reset action summaries. No Stage 1 or Stage 2 settings are changed.
- Terminal status: `F`; `Exit_status=1`; walltime `00:01:57`. The runner stopped before constructing a World or resetting the simulator because the diagnostic manifest lacked `arm_order_seed_base`. No diagnostic samples were produced; the action-space cause remains unknown. The original log and manifest are archived under `artifacts/25481553.pbs101/`.
- Corrective freeze revision 2 adds only the existing Stage 1 order-seed bases `2609222000` and `2609223000`. After static checks, one fresh diagnostic job may be submitted; this startup failure is not a Stage 1 result.

## CPU action-reset diagnostic retry 25484059.pbs101

- Submitted once after archiving the prior manifest and startup failure. The revised manifest adds only the two existing Stage 1 order-seed bases; local JSON/AST/resource-bound checks and `bash -n` passed. Pre-submit `qstat -u yguo017` was empty.
- Terminal status: `F`; `Exit_status=1`; walltime `00:02:01`; host `x1001c2s6b0n1`. The bounded runner completed its 12 planned resets (2 seeds × 6 paths), but final JSON serialization failed because `allow_nan=False` rejected an observed NaN. Only the first partial row was flushed to `diagnostic.json.tmp`; the remaining in-memory rows were lost when the process exited. No diagnostic rerun is planned.

## Source resolution and Stage 1 measurement revision 3

- The two prior Stage 1 jobs, `25469445.pbs101` and `25476081.pbs101`, remain recorded as `FAIL_CLOSED`; neither result is retroactively upgraded. The first summary lacks per-key flags. In the second, action was the sole unequal prepared key across all 8 seeds and 6 paths; its saved numerical quality/latency outputs remain descriptive only.
- Pinned source review resolves the apparent finite-sample/NaN-info discrepancy. In `EverythingToInfoWrapper.reset`, the wrapper calls `action_space.sample()` at line 227 and then replaces that result with `np.full_like(..., np.nan)` at line 258. `EnvPool._stack_fresh` preserves the resulting NaN action array, `World.reset` stores the stacked info, and `BasePolicy._prepare_info` converts the numeric array using `torch.from_numpy`. Thus the diagnostic's finite sample is expected instrumentation of an intermediate value; the prepared action is an all-NaN reset placeholder. The wrapper also sets reset reward to NaN, but reward is not a compared prepared key.
- Measurement correction: retain the `action` key and compare matching shape/dtype, NaN/+Inf/-Inf masks, and all finite values with `torch.equal`; all other prepared keys require matching shape/dtype and `torch.equal`. This captures the source-defined placeholder while still rejecting any finite action mismatch or signed-infinity mismatch. Stage 1/2 arms, seeds, solver, schedule, thresholds, estimands, sample size, and PBS bounds are unchanged. Revision 2 is snapshotted at `FREEZE.stage1-attempt-25476081.json`; revision 3 is the only freeze for the next authorized Stage 1 run.
- Runtime action reset diagnostic is not repeated: source behavior explains the NaN field directly, while its failed JSON serialization prevented recovery of every row. The partial evidence did recover a finite reset `sample()` value followed by a nonmatching NaN `world.infos['action']`, consistent with the source assignment.

## Stage 1 retry 25489648.pbs101

- Submitted once after freezing revision 3, which preserves the action key and compares its NaN/+Inf/-Inf masks plus exact finite entries; the two earlier FAIL_CLOSED jobs remain unchanged. Pre-submit `qstat -u yguo017` was empty; SSH host keys were verified.
- PBS terminal state: `F`, `Exit_status=0`, walltime `00:03:23`; compute host `x1000c0s0b0n1`. Runner status is zero, and `job.log` contains 41 five-second GPU telemetry records.
- All Stage 1 gates passed: native/routed teacher traces match bitwise on all 8 seeds × 30 rounds; all expected schedules/call counts are exact; every prepared observation key pairs across all six paths; outputs are finite; quality median is `-2.5027053` with 8/8 strict improvements; late7 mean solve time is `0.495418 s` vs teacher-only `2.175061 s` (77.22% reduction).
- The first teacher-only solve took `11.074841 s` and was the randomized first arm on the first seed; the other seven teacher solves took `0.902216–0.906247 s`. The frozen latency estimate keeps all observations. A descriptive sensitivity excluding only that first-call outlier still yields a 45.18% reduction, above the 30% gate; no gate or result is altered.
- Stage 2 is now eligible under the frozen plan; it has not yet been submitted. Full result and call records are in `artifacts/25489648.pbs101/`, with the Chinese result note in `STAGE1_ATTEMPT_25489648.zh.md`.

## Stage 2 25491774.pbs101

- Stage 2 completed with PBS/runner exit status 0 on `x1000c0s3b0n1`; the retained local status/summary and episode records confirm completion. A fresh PBS history lookup now returns `Unknown Job Id`, and `tracejob` no longer finds logs, while the prior live PBS query had recorded `F`, `Exit_status=0`. `job.log` contains 85 five-second GPU telemetry samples.
- Full frozen sample: 50 seeds × 4 arms = 200 episodes; exact seed range `260923001–260923050`; no missing or duplicate seed-arm pairs, crashes, non-finite outputs, schedule/innovation mismatches, or paired reset/key mismatches. All 200 episodes reached the 50-step truncation limit; none terminated successfully.
- Primary result: all arms were 0/50 successes. Late7 vs student had 0 late-only and 0 student-only successes; exact two-sided McNemar `p=1.0`; the required ≥5 additional late7 successes and `p<0.05` gate failed. Late7 vs teacher was 0/50 vs 0/50, meeting the frozen practical −5-success margin only; this is not formal non-inferiority. Overall frozen Stage 2 result is `FAIL`, not `INVALID`.
- Descriptive secondary means per episode: teacher calls student 0, late7 14, uniform7 14, teacher 60; planner walltime student `0.943 s`, late7 `0.998 s`, uniform7 `1.012 s`, teacher `1.835 s`. Late7 is about 45.6% faster than teacher-only on this secondary measure, but about 5.8% slower than student-only, without observed success improvement. All arms at zero success indicate a floor effect; do not interpret `p=1` as equivalence.
- See `STAGE2_ATTEMPT_25491774.zh.md` and `artifacts/25491774.pbs101/`. Recommended next action is post-hoc diagnosis of the shared closed-loop floor, especially the unmodified teacher-only arm and the upstream official-eval versus frozen-protocol boundary, plus action/goal semantics, before freezing any new experiment. Do not extend this run or claim success-rate non-inferiority.
- Benchmark boundary: this is not the upstream official LeWM PushT benchmark. The frozen run uses random simulator resets and a 50-step cap. Pinned upstream config is `num_eval=50`, `seed=42`, `goal_offset_steps=25`, `eval_budget=50`, dataset `pusht_expert_train`, and `max_episode_steps=100`; `eval_budget=50` means the effective rollout budget also remains 50 steps. Differences include expert-dataset sampled starts/goals with state/goal callables versus this run's random reset and `dataset=None`, and preprocessing: this run's policy used `process={}`, while upstream fits `StandardScaler` on full-dataset action/proprio/state (excluding NaN rows) and reuses the corresponding scaler for goal fields. The shared 0/50 result cannot be attributed to one difference alone. The result applies only to this frozen protocol; teacher-only is confirmed at 0/50.
