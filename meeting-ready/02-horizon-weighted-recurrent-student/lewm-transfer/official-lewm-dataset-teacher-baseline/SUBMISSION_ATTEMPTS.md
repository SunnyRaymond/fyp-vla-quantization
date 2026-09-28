# Submission attempts

## Direct gdev attempt, 2026-09-24

- `qsub` returned exit status 174 with `qsub: Access to queue is denied` for the direct `gdev` target.
- No job ID was returned; the immediate read-only `qstat -u yguo017` showed no jobs. No output directory or result artifacts were created.
- Read-only queue configuration showed `gdev` has `from_route_only=True`; `normal` is a route queue whose destinations include `gdev`. The frozen resource request (1 GPU, 16 CPUs, 110 GB, 2 hours) fits the `gdev` limits.
- Corrective control-flow change only: PBS submission target is `normal`, the authorized routing entry point. Experiment settings, checkpoint, dataset, resources, and gates are unchanged. One PBS job submission remains to be attempted; do not duplicate it if a job ID is returned.

## Routed teacher baseline 25531848.pbs101

- `qstat -x -f` terminal state: `F`; `Exit_status=1`; walltime `00:02:36`; host `x1000c0s6b0n1`.
- The runner failed closed at the imported-root check: `stable_pretraining` resolved to the staging virtual environment's `lib/python3.11/site-packages`, while revision 1 incorrectly required it to be under the `le-wm` source directory. The Python process stopped before config composition, task selection, HDF5 opening, checkpoint loading, or evaluation. There is no `selected_tasks.json` or `summary.json`; success count and the 5/50 gate are not measured.
- `job.log` retained five-second GPU telemetry at 0% utilization and 1 MiB used for the recorded samples; PBS and runner status files record exit 1.
- Revision 2 corrects only this dependency-root assumption: require `Path(sys.prefix).resolve()` to equal the staging `$STAGE_ROOT/venv`, require `stable_pretraining` under that venv, and continue requiring `stable_worldmodel` under its staged source root. The revision-1 freeze is preserved. All experiment settings remain unchanged.
- This is a startup failure, not an empirical 0/50 result and not a quality NO-GO. The task-selection/success gate is unassessed. One bounded teacher-only retry is permitted after static checks; no four-arm job is eligible without a valid baseline at or above 5/50.

## Revision-2 teacher-only retry 25534994.pbs101

- Submitted once through the authorized `normal` route after JSON/AST/contract checks and PBS `bash -n` passed. The import guard now requires exact staging `sys.prefix` and the installed `stable_pretraining` package under that venv; the staged `stable_worldmodel` root check is unchanged.
- Authoritative `qstat -x -f` terminal state: `F`, `Exit_status=0`, `exec_host=x1000c3s7b0n1`, walltime `00:06:36`.
- The runner completed 50 frozen rows, 50 unique source episodes, and 49 successes; validity is `PASS`, metrics are finite, and the frozen `>=5/50` engineering floor is met. This is eligibility for the already frozen paired follow-up, not a statistical claim.
- Five-second GPU telemetry was written to the job log (79 samples); observed utilization reached 75% and recorded memory reached 639 MiB. See `BASELINE_RESULT_25534994.zh.md` and `artifacts/25534994.pbs101/`.
- No baseline retry is needed. The follow-up must compare its teacher-only per-task vector exactly to this baseline and fail closed on any mismatch.

## Conditional Stage 2 attempt 25535684.pbs101

- Submitted once via the authorized `normal` route and assigned to `gdev`, requesting 1 GPU, 16 CPUs, 110 GB RAM, and 2 hours.
- Authoritative `qstat -x -f` terminal state: `F`, `Exit_status=1`, 10 seconds on `x1000c0s1b0n0`.
- The frozen arm order was generated correctly, but the startup guard compared a Python list to a tuple, rejecting the correct value. This occurred before HDF5/dataset/model/planner work. The local failure evidence is preserved in `artifacts/25535684.pbs101/`; result status is `NO_RESULT`, not a scientific outcome.
- The only fix is a container-type correction in the arm-order assertion. The frozen order, task file, seeds, solver, arms, statistics, and gates are unchanged. A new independent PBS attempt is authorized once after static validation; do not retry job ID 25535684.

## Corrected Stage 2 attempt 25535692.pbs101

- After the tuple/list assertion fix, static AST/CLI/frozen-order checks and PBS shell syntax passed. Updated the remote control runner and recorded the failed first attempt before resubmitting.
- Submitted once through the authorized `normal` route; routed to `gdev` with 1 GPU, 16 CPUs, 110 GB RAM, and 2 hours. First authoritative `qstat -x -f`: `R` on `x1000c0s1b0n0`; terminal state `F`, `Exit_status=1`, walltime `00:03:24`.
- The run passed the arm-order guard, opened the HDF5 dataset, cached action/proprio/state, and verified row/task identity. It then failed before loading checkpoints or running a planner because `load_modules` was called on `run_adaptive_teacher_schedule`; the function is exposed by `run_official_pusht_cem.load_modules`.
- This is `NO_RESULT`, not an arm outcome. Preserve `artifacts/25535692.pbs101/`; the minimal call-site fix is statically checked before one new bounded attempt under the continuing authorization.

## Stage 2 callback-interface attempt 25535711.pbs101

- The runner loaded the frozen dataset/scalers and both model checkpoints, then entered pinned `CEMSolver.solve`; PBS terminated `F`, `Exit_status=1`, walltime `00:05:02`.
- The solver's callback lifecycle additionally reads `callback.history` and `callback.output_key` when assembling solver outputs. The custom lightweight callback lacked `history`, so the first solve aborted before action execution or episode outcomes. This is `NO_RESULT`; artifacts are in `artifacts/25535711.pbs101/`.
- Static review of pinned `cem.py` confirmed the full callback interface (`reset`, `start_batch`, `__call__`, `end_solve`, `output_key`, `history`). Fix by reusing the already validated `old_router.TraceCallback(retain_tensors=False)`; no solver semantics or freeze settings change. One new bounded attempt is authorized after static validation.

## Stage 2 batch-boundary attempt 25535774.pbs101

- PBS terminal state: `F`, `Exit_status=0`, walltime `00:03:16`. The result was `FAIL_CLOSED`, not a scientific outcome. Pinned `CEMSolver.solve` calls callback `start_batch()` once per environment chunk before running the 30 CEM steps. The routed round index had been reset only per outer `solver.solve`; late7 and uniform7 therefore made only 14 teacher calls across the two solve calls instead of applying seven calls in every batch. Preserve this attempt's artifacts; do not use its outcomes for performance claims.

## Corrected Stage 2 attempt 25535873.pbs101

- Updated only the Stage2 batch-boundary routing callback and schedule validator; exact tasks, policies, seeds, preprocessing, arm order, statistics, and performance gates remain frozen. Static AST/CLI and schedule-chunk checks passed before submission.
- Submitted once through the authorized `normal` route; routed to `gdev` with 1 GPU, 16 CPUs, 110 GB RAM, and a 2-hour walltime. Authoritative PBS state: `F`, `Exit_status=0`, walltime `00:08:09`.
- The experiment is valid and complete (50 tasks, 200 outcomes; teacher-only exactly matches the 49/50 baseline). Results are student 11/50, late7 26/50, teacher 49/50, uniform7 18/50; late7 gain +15 with exact cluster sign-flip `p=0.0007286`. The frozen teacher-gap threshold and latency ratio threshold both fail, so the decision is `FAIL`, not `FAIL_CLOSED`. Full report: `STAGE2_RESULT_25535873.zh.md`.
