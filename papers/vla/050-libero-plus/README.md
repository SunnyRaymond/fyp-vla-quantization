# LIBERO-Plus

**Paper:** *LIBERO-Plus: A Progressive Robustness Benchmark for Visual-Language-Action Models*  
**Version:** CVPR 2026 final  
**Local PDF:** [paper-cvpr2026.pdf](paper-cvpr2026.pdf)  
**Priority:** Core; 20–30 min

## Background

LIBERO-Plus treats robustness as a multidimensional, graded property rather than one success score on a narrow initial-state distribution.

## Problem

How brittle are high-scoring VLA models to camera, robot state, scene, language, lighting and sensor changes, and can this be measured at progressive difficulty levels?

## Method

It automatically produces 10,030 task instances and more than 56k scenarios across seven factors and 21 subdimensions, with five levels L1–L5. Factors cover object layout, camera viewpoint, robot initial state, language, lighting, background texture and sensor noise. Ten VLA models are evaluated.

## Key Innovation

The benchmark combines large-scale automated generation, controlled single-factor perturbations and progressive severity, yielding fine-grained robustness profiles.

## Main Results

The abstract reports a fall from 95% to below 30% under modest perturbations. Table 2 gives examples: OpenVLA 76.5→15.6 overall, OpenVLA-OFT 97.1→69.6 and OpenVLA-OFTw 95.3→55.8. Camera viewpoint and robot initial state are among the largest weaknesses; relatively small language effects motivate a further diagnostic that some policies may ignore language. Training on a 20k-trajectory generalized set raises the authors' overall robustness result to 79.6 within this framework.

## Authors' Claim vs LIBERO

**Direct extension:** Standard LIBERO hides sensitivity to many deployment-relevant perturbations; LIBERO-Plus adds scale, factor isolation and progressive difficulty while retaining ecosystem compatibility.

## Limitations

- It remains LIBERO-derived and does not solve physics or sim-to-real validity.
- Automated perturbations may not match the frequency or joint structure of real deployment shifts.
- Aggregate robustness can hide catastrophic weakness on one factor; per-axis results should be retained.

## Why It Matters

For an FYP needing one practical add-on to standard LIBERO, this is the strongest general-purpose choice because it is final, broad and graded.

## Reading Questions

1. Are difficulty levels calibrated equally across axes?
2. Should robustness be averaged, worst-case, or risk-weighted?
3. Does robustness training improve unseen factors or only benchmark-defined ones?

## Links

- [Paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Fei_LIBERO-Plus_A_Progressive_Robustness_Benchmark_for_Visual-Language-Action_Models_CVPR_2026_paper.pdf)
- [Code](https://github.com/sylvestf/LIBERO-plus)

