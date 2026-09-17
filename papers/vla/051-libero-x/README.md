# LIBERO-X

**Paper:** *LIBERO-X: Robustness Litmus for Vision-Language-Action Models*  
**Version:** arXiv:2602.06556v1; RSS 2026 accepted per official repository  
**Local PDF:** [paper-arxiv.pdf](paper-arxiv.pdf)  
**Priority:** Core when both training and evaluation diversity matter; 25–40 min

## Background

LIBERO-X argues that changing only evaluation perturbations is insufficient when training demonstrations themselves are homogeneous and each scene supports very few tasks.

## Problem

How can a LIBERO-compatible suite evaluate cumulative spatial, topology, visual and semantic shifts while also supplying diverse training data?

## Method

It collects 2,520 human demonstrations over 600 tasks in 100 scenes. Evaluation has five cumulative levels: local spatial perturbation, extended spatial perturbation, scene-topology reconstruction, visual-attribute/unseen/confounding objects and semantically equivalent instruction reformulation. Multi-label annotations support failure analysis, and stricter success predicates reduce false positives.

## Key Innovation

Unlike test-only perturbation suites, LIBERO-X expands both training diversity and evaluation composition, with progressive cumulative shifts.

## Main Results

Across five evaluated VLA models, mean success falls from 39.4 at Level 1 to roughly 8.2 at Level 5, a 31.2-point mean drop. π0.5 is strongest among reported models but falls 65.2→18.0; GR00T N1.5 falls 43.3→9.7.

## Authors' Claim vs LIBERO

**Direct design critique:** LIBERO-X claims standard LIBERO has high train-test similarity, few tasks per scene, homogeneous demonstrations and sometimes loose success predicates. Its response is diverse data, multi-task scenes, cumulative shifts and more exact evaluation.

## Limitations

- Models are finetuned on different LIBERO-X data, so scores are not directly comparable with standard LIBERO leaderboards.
- Episode time limits materially affect results.
- It still inherits much of the LIBERO/MuJoCo lineage.
- The local file is arXiv v1 even though the repository states RSS 2026 acceptance.

## Why It Matters

Use it if the research claim is about compositional generalization and you can afford retraining, not just checkpoint-only stress testing.

## Reading Questions

1. How much of the improvement comes from more tasks versus more scenes?
2. Are cumulative levels interpretable when multiple shifts interact?
3. Do stricter predicates change historical model rankings?

## Links

- [Paper](https://arxiv.org/abs/2602.06556)
- [Project](https://zackhxn.github.io/LIBERO-X/)
- [Code](https://github.com/meituan/LIBERO-X)

