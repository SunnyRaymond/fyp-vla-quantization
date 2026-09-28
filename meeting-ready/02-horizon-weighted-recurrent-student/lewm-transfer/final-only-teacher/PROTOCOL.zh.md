# LeWM PushT final-only teacher re-score protocol

## 问题与设计

previous same-task results show current `student_only` 11/50 and `late_teacher7` 25/50, while teacher-only is 49/50. This one-arm bounded diagnostic changes only the cost source at CEM round 30: each actual `start_batch` uses treatment student cost for rounds 1–29 and official teacher cost for round 30. It does not train, alter checkpoints, or repeat K-screening, candidate-tail distillation, or pairwise ranking.

The task list is exactly the 50 rows in baseline `25534994.pbs101`, with one unique source episode per row. The compute runner checks these row identities against the HDF5 file and reruns only `final_only_round30`. It uses the same treatment checkpoint, full-data `StandardScaler` fits and NaN-row exclusions, goal scaler reuse, ImageNet/Resize224 transforms, pinned dataset callables, `World.evaluate` dataset mode, `goal_offset=25`, `eval_budget=50`, `max_episode_steps=100`, and `video=None`. This is upstream-protocol-aligned dataset evaluation, not a verbatim `eval.py` execution or random-reset benchmark.

## 冻结参照与判定

The primary paired comparison uses the exact 50-task `treatment_step1000` outcome vector from valid `25536308.pbs101` (11/50). That vector was compared task by task and found identical to `student_only` in valid `25536049.pbs101`. The current teacher-only vector (49/50) also matches the baseline `25534994.pbs101` and the prior phase-timing job. These source IDs and per-task vectors are frozen in `REFERENCE_OUTCOMES.json`.

Primary test: exact two-sided McNemar for `final_only_round30` versus the frozen student-only vector. A practical gain requires at least +5 successes and `p<0.05`. The one-time late7 reference is `25536049.pbs101` at 25/50, but another valid run, `25535873.pbs101`, recorded 26/50. Treat late7 only as a descriptive reference, not a stable per-task oracle or an equivalence target. If final-only passes the primary gain gate and has at least 20/50 successes (within five of the frozen 25/50 late7 point estimate), label it `FINAL_ONLY_ENGINEERING_SUFFICIENCY_GATE`; this is not statistical equivalence. A primary gain that remains more than five below that reference is useful but does not meet the sufficiency gate. If primary gain is not demonstrated while the same-task late7 reference remains +14 over student-only, report that a single final score did not demonstrate the gain and multi-round late feedback remains a hypothesis, not a causal finding. All other patterns are inconclusive.

The exact paired comparison against the one-time late7 vector may be shown descriptively; its p-value is secondary and cannot establish repeatability. Do not change tasks, rounds, seeds, sample size, or thresholds after seeing outcomes.

## Validity and resource boundary

Require all 50 rows and outcomes, exact HDF5 `episode_idx`/`step_idx` identity, 50 unique episodes, frozen treatment provenance, seed 42, candidate shape and pre-solve generator records, all six prepared observation keys, finite outcomes/timing, and a complete per-batch route trace with exactly one teacher call at round 30 and student cost on all other rounds. Because earlier runs retained observation keys/equality checks but not raw tensors, claim semantic task/preprocessing reuse and key/shape capture; do not claim bytewise cross-job tensor equality.

Record synchronized planner solve time and full evaluation time for this arm only. Do not claim direct speedup against the earlier multi-arm jobs; run composition differs and this gate has no latency criterion. PBS must pass the compute-node guard, return runner/job status 0, and record GPU utilization/memory every 5 seconds in `job.log`. Use one 1-GPU, 16-CPU, 110-GB, 2-hour job with a 6,900-second internal timeout. Any validity failure is `FAIL_CLOSED`; preserve artifacts and do not retry under this freeze.

Reading: de Boer et al., [“A Tutorial on the Cross-Entropy Method”](https://doi.org/10.1007/s10479-005-5724-z) for how elite selection updates the sampling distribution over iterations. If a later project separately tests student-on-policy expert aggregation, Ross et al., [“A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning”](https://proceedings.mlr.press/v15/ross11a.html) is the original DAgger paper; applying it to LeWM planner-cost labels would be a new method transfer, not an established result here.
