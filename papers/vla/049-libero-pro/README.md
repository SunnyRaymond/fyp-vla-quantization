# LIBERO-PRO

**Paper:** *LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization*  
**Version:** arXiv:2510.03827v2, 2026-05-25  
**Local PDF:** [paper-arxiv.pdf](paper-arxiv.pdf)  
**Priority:** Core for the overfitting claim; 20–30 min

## Background

LIBERO-PRO directly targets concern that strong standard LIBERO results may reflect memorization because training and evaluation preserve almost the same task, scene and instruction aside from small initial-state perturbations.

## Problem

Can a model that scores above 90% on standard LIBERO still act correctly when objects, positions, language, task structure or environment change?

## Method

The benchmark applies controlled variations to manipulated objects, initial positions, instructions, task composition and environments while preserving compatibility with LIBERO. It evaluates OpenVLA, π0 and π0.5 using 50 episodes per task.

## Key Innovation

It converts the qualitative “LIBERO may be saturated” concern into a plug-and-play OOD stress test where each perturbation family has an interpretable source.

## Main Results

The abstract reports that models above 90% on standard LIBERO can collapse to 0.0% in generalized settings. Position and task changes are especially damaging; in one LIBERO-Goal position-shift setting, π0.5 achieves 0.38 while OpenVLA and π0 achieve 0. The paper also shows policies acting on replaced or absent objects and continuing under disrupted instructions.

## Authors' Claim vs LIBERO

**Direct extension:** LIBERO-PRO claims a fairer measure of generalization because evaluation is no longer near-identical to training across object, position, language, task and environment dimensions.

## Limitations

- “Memorization” is the authors' interpretation of the failure pattern; the experiment shows brittleness but does not uniquely establish the internal causal mechanism.
- Only three model families are evaluated.
- The benchmark inherits LIBERO's physics, assets and tabletop domain.
- The 0.0% headline is setting-specific and should not be presented as every model's overall score.

## Why It Matters

This is the most direct paper to cite when motivating why a standard LIBERO number alone is insufficient.

## Reading Questions

1. Are the OOD changes task-preserving or do some create a genuinely different task?
2. Which perturbations isolate memorization rather than general perception/control difficulty?
3. Should results be averaged or reported per shift family?

## Links

- [Paper](https://arxiv.org/abs/2510.03827)
- [Code](https://github.com/Zxy-MLlab/LIBERO-PRO)

