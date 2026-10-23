# Fast-WAM R128 C80 loss-pair closed-loop evaluation

This directory freezes the paired LIBERO-Plus evaluation for `az` and `reader_kl` after the 8-update, 80-exposure C80 calibration. Both arms share the same frozen R0 and VQ initialization. Each arm runs all 280 variants from the existing paired manifest; four workers preserve the even/odd index split, with workers 0–1 assigned to `az` and 2–3 to `reader_kl`.

The paired calibration source is job `25921795.pbs101`, with its protocol and artifacts under `/scratch/users/ntu/yguo017/fastwam-vq-r128-c80-loss-pair-20261022/artifacts/25921795.pbs101`. Each arm binds its own `summary.json`, `deployment.pt`, `anchor.json`, and `anchor.pt`; its shared VQ bank is `fastwam-vq-r128-c80-dual-v1`. The fixed rotation source is `C80_once_lr1e4` from `fastwam-r128-improvements-20261021`, job `25905203.pbs101`. The dual-bank job `25912619.pbs101` supplies the shared code bank and reusable code only; it is not the R0 source.

Before episodes run, each worker validates the saved source anchor on case 2000 / seed index 0 / sampler seed 910004000, exact action and future tensor parity, actual inference action parity at `1e-6`, 614 observed modules, and the frozen installation metadata. The rotation must remain orthogonal within `5e-5` and show nonzero movement in both action and video streams. Startup performs two full queries; no new training updates are permitted.

The installation uses 4-bit VQ weights decoded to dense BF16, dynamic A4 activations, BF16 KV, and the signed 128-block Cayley MXVQ transform. This experiment makes no native low-bit kernel claim. Timings describe this decoded-BF16 execution only; utilization and VRAM are sampled every 15 seconds in the PBS job log.

Aggregation requires all 560 new episodes plus 4,200 frozen reference slots: the 13 original reference arms and the two completed R128 W16A4 arms from job `25910102.pbs101`. Completion additionally requires all four workers to exit successfully and the PBS job to complete with exit code 0. The single allocation requests four GPUs for 12 hours through the normal queue and must land on `g3`.

`control.py` retains durable submit intent and exact-name recovery, bounded status reads, and checkpoint-only resume. `run.pbs` stages the frozen evaluator/runtime dependencies inside the PBS allocation and records GPU utilization and VRAM samples. Local standard-library tests cover the protocol gates, paired shards, reference denominator, aggregation markers, controller recovery, and shell-script syntax; they do not load Torch or run model inference.
