# Seeded pilot absolute-regret post-hoc analysis

This is a descriptive, exploratory follow-up to seeded pilot `25537667.pbs101`, whose frozen decision remains NO-GO. It does not change that decision, define a new gate, estimate a new threshold, or authorize training.

The compute job reads only the pilot's `candidate_scores.jsonl`, `seeded_pilot_summary.json`, and `SEEDED_PILOT_FREEZE.json`. It does not open HDF5, model, checkpoint, episode trace, or paired-gate files. PBS metadata is read only to verify the allocation guard.

The analysis unit is the source episode. It uses the 15 episodes that the existing summary marks `matched_t0_t25`; the episode that terminated after 25 transitions before its next replan is excluded from both sides of paired descriptive summaries. For each frozen CEM round 10, 20, and 30, report t0 and t25 standardized teacher-elite regret median, linearly interpolated Q1/Q3 (Type 7), IQR width, counts above 0 and above 0.5, and median teacher-top30 recall@120. Emit one paired row per matched episode with both states' regret/recall and the t25-minus-t0 regret difference.

These absolute values can describe whether ranking regret is already present at t0 and how it differs at t25. The t25-minus-t0 contrast still combines planner context, simulator state, action history, and native CEM candidate/RNG changes; it cannot isolate a simulator-state cause. No result from this post-hoc table is a GO claim.
