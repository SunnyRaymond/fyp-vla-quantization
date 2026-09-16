# Source-to-card fidelity correction

- Source of truth: `phase3_revise/final_candidate.json`; canonical candidate and guarded fields were not modified.
- Updated only Phase 4 source maps and the active implementation audit: `fill_map.json`, `derive_map.json`, and `phase4_implementability.json`.
- Restored the canonical d_a/d_o denominators, sigma definitions, epsilon_a=epsilon_o=1e-3 floors, lambda_a=lambda_o=1 default weights and ratio sweep, episode-mean C_r aggregation with complete-episode bootstrap intervals, local L_jr/L_r definition, and B=B_site site-count relation.
- Kept unresolved items open: checkpoint revision/digest and exact invocation manifest, low-bit operator contract, simulator restoration/integration and row inclusion/preprocessing, numeric B_site, bytes/latency protocol, held-out split/contact labels, and permutation/oracle details.
- Preserved the pre-repair implementation report at `phase4_implementability_original.json`; active open-point count after this repair is 10.
- Literal math backticks and the malformed step-reference trigger were removed from active source prose; generated cards are regenerated after assembly.
