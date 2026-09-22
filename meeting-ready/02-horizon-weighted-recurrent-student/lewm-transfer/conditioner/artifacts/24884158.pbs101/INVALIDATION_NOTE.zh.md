# Pilot invalidation note

`24884158.pbs101` completed with `Exit_status=0`, but it is **not** the formal
conditioner result. Post-run fairness review found that this runner version replaced
the baseline affine transition `LayerNorm` with a no-affine normalization, so the
treatment did not preserve the baseline shared transition exactly. Keep its summary
only as a debugging/pilot artifact; the corrected implementation first ran as
`24884222.pbs101`, and the final primary comparison uses `24884287.pbs101` with
training-period GPU telemetry after the zero-init base-path equivalence smoke test
returned `max_abs=0.0`.
