# DINO-WM PushT CEM iteration cache

This bounded experiment compares the official DINO-WM PushT `CEMPlanner.plan`
with a planner-call-local `iteration_cache`. It does not use the earlier
candidate-axis `factorized_cache` path.

The cached path encodes the already candidate-batched `transformed_obs_0`
once per fixed observation and reuses that encoded observation across all CEM
iterations. The candidate batch remains `num_samples`; `encode_act`, predictor
rollout, action replacement, objective, ranking, elite selection, `mu/sigma`
updates and seeded candidate generation remain on the official path. The cache
is discarded after each `plan()` call.

Two synchronized timing boundaries are recorded for both paths:

* `full_planner_latency_ms`: public `plan()` entry through returned actions,
  including preprocessing, goal encoding, cache setup, all CEM iterations and
  trace logging.
* `plan_section_latency_ms`: after both observation transforms and before goal
  encoding through action return; it includes goal encoding, cache setup, the
  entire CEM loop and logging, but excludes the two input transforms.

`cem_loop_latency_ms` and `cache_setup_latency_ms` are diagnostic submetrics.
Each observation has five warmups and ten paired technical repeats. The runner
also records full decision traces, full/section paired reductions and peak
memory. `verify_iteration_cache.py` derives per-call `n_evals` from that call's
own returned action batch, avoiding the old aggregate-observation mismatch.

The PBS script reuses the already prepared official PushT assets and dependency
overlay from `pusht_transfer`; it does not download, extract, install, compile,
or hash anything. It must be submitted only after checking the existing remote
asset markers and with one bounded A100 allocation.
