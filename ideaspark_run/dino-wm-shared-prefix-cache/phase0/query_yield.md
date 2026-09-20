# Per-query yield (Phase 0.4 labels joined onto retrieval provenance)

A query whose share is mostly `off_topic` is spending guaranteed round-robin slots on noise — drop or rephrase it next run (see the VOCABULARY-OWNERSHIP TEST in references/intent-recognition.md). Papers reachable from several queries are credited to each.

| query | core | adjacent | off_topic | on-topic share |
|---|---|---|---|---|
| world model CEM candidate feature caching | 2 | 18 | 20 | 50% |
| shared observation prefix reuse candidate rollouts | 0 | 20 | 20 | 50% |
| candidate rollout encoder cache invalidation | 1 | 14 | 18 | 45% |
| model predictive control trajectory prefix caching | 0 | 14 | 13 | 52% |
