# Host supplemental constraints

This file is a host-side constraint for the Idea Spark run. It is not a
pipeline retrieval record, citation, full-text gate, collision result, or
novelty certificate. Claims about papers must still be supported by this run's
connector records and full-text artifacts.

## Direct prior-art boundaries

- QuantWM, *An Empirical Study of World Model Quantization*, arXiv:2602.02110v1,
  already covers encoder/predictor asymmetric sensitivity, rollout/planning
  mismatch, and the fact that more planning optimization need not repair
  aggressive low-bit failures.
- *Where Bits Matter in World-Model Planning*, arXiv:2602.11882v1, already
  covers paired mixed-bit/planner-budget comparisons and module-aware allocation
  questions for latent world-model planning.
- QuantWAMs, *QuantWAMs: Calibrating at the Right Granularity for World Action
  Models*, arXiv:2607.28405v1, already contains the joint video/action
  empirical-Fisher cross term in Eq. 12--16 and fixed-count closed-loop
  protection-schedule repair in Eq. 22. Renaming those mechanisms is not novel.

## Fast-WAM feasibility boundary

- The official FastWAM README documents an Optional IDM mode on the same
  checkpoint, with `idm` versus `first_frame` modes. The original checkpoint
  uses `sigma_shift=5.0`; the Optional IDM setting uses `sigma_shift=1.0`.
- `infer_action` no longer defaults to VAE decoding in that README.
- If Fast-WAM is used as a feasibility baseline, the candidate must lock the
  exact checkpoint, mode, sigma-shift, and inference path. A `first_frame` run
  that completely skips the future branch cannot be treated as a baseline that
  tests protection of future-conditioning signals.

## LingBot-VA feasibility boundary

- The current official LingBot-VA README reports roughly 24 GB for RoboTwin
  single-GPU evaluation with VAE/text-encoder CPU offload, and roughly 18 GB
  for i2av with the same offload.
- These figures establish only the reported inference scope. They do not imply
  that backward-based calibration or empirical-Fisher profiling fits in 40 GB.
  Any candidate using LingBot-VA must budget calibration separately from
  inference and state the offload/checkpoint/inference configuration.

## Interpretation boundaries

- Keep this branch focused on manipulation WAMs that couple future prediction
  with action generation and on numerical weight/activation quantization.
- Keep it distinct from the main branch's latent rollout/planning route, from
  ordinary VLA-only quantization, and from latent/action vector quantization.
- CPU traces can test implementation or arithmetic consistency only; they do not
  establish robot success, GPU memory, or speedup. Fake-quant/dequantized GEMM
  must remain separate from real low-bit kernels.

## Earlier task-consequence quantization lineage

- Host review also identifies VAML (AISTATS 2017), Model Advantage / Value-Aware
  Model Learning (arXiv:2106.14080), and Deep Task-Based Quantization
  (arXiv:1908.06845) in the pre-48-month mechanism lineage.
- These records do not automatically subsume a manipulation WAM method, but the
  broad principle of defining a quantization target by downstream task
  consequence is already present in the lineage. It cannot be presented as
  novelty. Later audits must compare the concrete intervention, estimand, and
  numerical mechanism against these records.

## Direct adjacent robot-control prior

- SQIL, *Saliency-Aware Quantized Imitation Learning for Efficient Robotic
  Control*, ICCV 2025, arXiv:2505.15304, with the official project page
  `https://aiha-lab.github.io/sqil/` and arXiv page
  `https://arxiv.org/abs/2505.15304`, is a direct adjacent prior identified by
  host review.
- Its mission-critical-state protection and saliency-weighted QAT/action
  distillation mean that protecting contact or otherwise critical states is not
  by itself novel. The later audit must explicitly compare SQIL's concrete
  intervention, estimand, and numerical mechanism with any WAM paired-simulator
  site intervention; it must not treat SQIL as automatically identical to a
  coupled future-video/action WAM method.
