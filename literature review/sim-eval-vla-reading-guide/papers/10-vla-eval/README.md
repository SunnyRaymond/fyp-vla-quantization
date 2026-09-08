# vla-eval

**Paper:** *vla-eval: A Unified Evaluation Harness for Vision-Language-Action Models*  
**Version:** arXiv:2603.13966v2, 2026-04-17  
**Local PDF:** [paper-arxiv.pdf](paper-arxiv.pdf)  
**Priority:** Practical core; 15–25 min

## Background

vla-eval is an evaluation **harness**, not a new environment. It addresses the engineering and protocol errors that make cross-benchmark VLA results hard to reproduce.

## Problem

Each simulator requires incompatible dependencies, while seeds, preprocessing, action representation and termination details are often undocumented. A seemingly valid setting can shift success by tens of percentage points.

## Method

The system decouples model inference from benchmark execution using a network protocol and containerized environments. The paper integrates 14 simulation benchmarks and six model servers, defines canonical protocols and shards episodes for batch parallel evaluation. It audits six VLA codebases across three benchmarks.

## Key Innovation

Each model and benchmark integrates once, while versioned environments and complete run configurations make multi-benchmark evaluation more reproducible and scalable.

## Main Results

- Up to 47x wall-clock speedup; 2,000 LIBERO episodes complete in about 18 minutes on the reported H100 setup.
- Reproduced published scores closely across LIBERO, CALVIN and SimplerEnv.
- One wrong proprioceptive-state source drops X-VLA on LIBERO from 97.8% to 42%; a quaternion convention mismatch drops LIBERO-Long from 95% to 56%.

## Claim vs LIBERO

**Infrastructure claim:** vla-eval makes LIBERO and other benchmarks faster and less protocol-sensitive. It does not make LIBERO more diverse, harder or more real-world-valid.

## Limitations

- The audit covers six codebases and three benchmarks.
- Leaderboard values taken from papers are not independently verified.
- Current metrics are limited mainly to task success; progress, efficiency and safety are not yet supported.
- Public compatible checkpoints are still required.

## Why It Matters

Any FYP comparing several VLA models can lose more validity from preprocessing/action-mode mismatch than from statistical noise. This paper supplies a concrete reproducibility checklist even if the harness does not yet support every model.

## Reading Questions

1. Which configuration fields are sufficient to reproduce your exact run?
2. Does parallelism alter simulator determinism or model batching behavior?
3. How should partial-progress metrics be added without breaking comparability?

## Links

- [Paper](https://arxiv.org/abs/2603.13966)
- [Code](https://github.com/allenai/vla-evaluation-harness)
- [Leaderboard](https://allenai.github.io/vla-evaluation-harness/leaderboard/)
