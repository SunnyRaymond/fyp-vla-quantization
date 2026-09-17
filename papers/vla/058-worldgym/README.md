# WorldGym

**Paper:** *WorldGym: World Model as An Environment for Policy Evaluation*  
**Version:** arXiv:2506.00613v3, 2025-09-30; preprint  
**Local PDF:** [paper-arxiv.pdf](paper-arxiv.pdf)  
**Priority:** Core learned-environment reading; 25–40 min

## Background

WorldGym treats an action-conditioned video model as an interactive environment. A policy repeatedly receives generated observations, outputs actions and is scored by a VLM reward model.

## Problem

Can a single learned environment estimate real VLA performance across policies while allowing OOD image and language tests to be created from one initial frame?

## Method

The model uses autoregressive video generation with horizons aligned to policy action chunks, Monte Carlo rollouts and a VLM success judge. It evaluates RT-1-X, Octo and OpenVLA on 17 Bridge evaluation tasks and also explores Google Robot and synthetic OOD image/language conditions.

## Key Innovation

Unlike a static video evaluator, it closes the loop between policy and generated environment and makes new OOD conditions easy to specify via image editing or changed instructions.

## Main Results

Per-task WorldGym success correlates with real success at Pearson r=0.78. Mean success differs from real by 3.3 points on average: RT-1-X 18.5% real vs 15.5% generated, Octo 20.0% vs 23.82%, OpenVLA 70.6% vs 67.4%. Relative rankings across model versions and checkpoints are preserved in reported cases. Added distractors reduce WorldGym success for RT-1-X by 51%, Octo by 83% and OpenVLA by 41.5% relative.

## Claim vs LIBERO

**Synthesis:** WorldGym is more flexible than a handcrafted LIBERO scene for rapidly creating OOD image/language conditions and for approximating real-policy ranking. It is less standardized and physically grounded, so it is not a clean replacement for LIBERO's deterministic task execution.

## Limitations

- Authors state that not all generated interactions are realistic.
- Validation centers on three policies and 17 Bridge tasks.
- VLM reward errors and image-editing artifacts can confound policy errors.
- Learned dynamics can underestimate in-distribution actions or overestimate OOD actions; ranking evidence is stronger than exact rollout fidelity.

## Why It Matters

WorldGym shows how a World Action Model might become an evaluation environment, directly linking to research on WAM/VLA interfaces.

## Reading Questions

1. Is a 0.78 task-level correlation enough for model selection?
2. How does uncertainty accumulate over hundreds of generated control steps?
3. Which OOD findings survive real execution?

## Links

- [Paper](https://arxiv.org/abs/2506.00613)
- [Project](https://world-model-eval.github.io/)

