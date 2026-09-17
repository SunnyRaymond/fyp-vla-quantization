# LIBERO-Para

**Paper:** *LIBERO-Para: A Diagnostic Benchmark and Metrics for Paraphrase Robustness in VLA Models*  
**Version:** arXiv:2603.28301v1, 2026-03-30  
**Local PDF:** [paper-arxiv.pdf](paper-arxiv.pdf)  
**Priority:** Focused companion; 20–30 min

## Background

Standard LIBERO commonly evaluates the same instruction wording seen during training. LIBERO-Para isolates whether VLA policies preserve task execution under semantically equivalent language changes.

## Problem

Does a policy understand an instruction compositionally, or has it coupled actions to familiar lexical templates?

## Method

The benchmark combines action-expression and object-reference variations, covering 43 linguistic variation types. Seven VLA configurations from 0.6B to 7.5B parameters are tested on LIBERO-Goal. It proposes PRIDE, which combines task success with paraphrase difficulty, and uses DTW-based trajectory analysis to classify planning-level divergence.

## Key Innovation

It isolates language robustness at fine granularity and avoids treating an easy paraphrase success as equivalent to a hard structural paraphrase success.

## Main Results

The authors report 22–52 percentage-point success drops under paraphrasing. They classify 80–96% of failures as Far-GT trajectory divergence and find object-level lexical changes especially damaging. Binary success overestimates robustness by 8.4–22.0% relative to PRIDE in their experiments.

## Authors' Claim vs LIBERO

**Direct extension:** LIBERO-Para claims standard LIBERO cannot measure paraphrase robustness because train and evaluation language are identical. It adds controlled language shifts and a difficulty-aware metric.

## Limitations

- It covers only LIBERO-Goal and only language variation.
- PRIDE, paraphrase categories and DTW-based failure labels are author-designed and need external validation.
- It does not address visual diversity, physics or sim-to-real transfer.

## Why It Matters

This is the right diagnostic if your VLA improvement involves language encoder compression, instruction tuning or semantic generalization.

## Reading Questions

1. Do paraphrases preserve all referential assumptions of the original scene?
2. Is PRIDE stable across different paraphrase generators?
3. Can language robustness be predicted from VLM benchmarks without robot rollout?

## Links

- [Paper](https://arxiv.org/abs/2603.28301)
- [Code](https://github.com/cau-hai-lab/LIBERO-Para)
