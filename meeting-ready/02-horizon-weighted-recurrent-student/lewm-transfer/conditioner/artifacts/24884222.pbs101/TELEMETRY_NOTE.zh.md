# Telemetry note

`24884222.pbs101` completed with `Exit_status=0` and produced the same deterministic
predictor metrics as the final run, but its 30-second sampler recorded only the
startup GPU sample because the job finished in about 25 seconds. It is therefore
retained as a metrics-valid, telemetry-insufficient artifact. The final primary run
is `24884287.pbs101`, which used a 5-second sampler and recorded three training-period
samples at `33%` GPU utilization and `633 MiB` memory use.
