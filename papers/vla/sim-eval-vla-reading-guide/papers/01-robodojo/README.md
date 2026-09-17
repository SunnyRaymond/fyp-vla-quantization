# RoboDojo

**Paper:** *RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies*  
**Author identity:** Chen Tianxing is first author  
**Version:** arXiv:2607.04434v3, 2026-07-08  
**Local PDF:** [paper-arxiv.pdf](paper-arxiv.pdf)  
**Priority:** Core; 35–50 min

## Background

RoboDojo argues that modern generalist robot policies are evaluated with fragmented suites, inconsistent interfaces and limited real-robot coverage. It is the work most likely matching the user's description of Chen Tianxing's updated VLA simulation evaluation platform.

## Problem

How can one compare many VLA policies across diverse capabilities, bimanual manipulation, simulation and real hardware while reducing hand-tuned benchmark gaming?

## Method

RoboDojo contains 42 simulation tasks and 18 real-world tasks organized into five dimensions: Generalization, Memory, Precision, Long-Horizon and Open. Simulation uses Isaac Sim and a bimanual ARX X5 setup. The protocol runs 50 episodes per simulation task, reports binary success plus partial-progress score, and uses hidden verification layouts. Real evaluation is remotely standardized across ARX X5, Piper and Piper X hardware. The paper integrates 30 policy baselines; its comparison table describes the ecosystem as 35+ policies.

## Key Innovation

The strongest contribution is not raw task count but the combined protocol: capability-oriented taxonomy, bimanual tasks, partial progress, sim-and-real evaluation, one policy interface and hidden anti-overfitting verification.

## Main Results

- Best evaluated policy averages 8.80% success and 13.07 progress score, versus human experts at 76.03% and 80.42.
- Table 3 reports severe Standard-to-Random degradation; for example, π0.5 drops from 20.92 to 5.82 score, a 72.2% relative decrease.
- The benchmark comparison table attributes 24 skills, heterogeneous parallel execution and sim-and-real evaluation to RoboDojo, versus 6 skills and simulation-only for LIBERO.

## Authors' Claim vs LIBERO

**Direct author comparison:** RoboDojo claims broader operation primitives, bimanual manipulation, explicit real-world tasks, heterogeneous parallelism, a much larger integrated policy set and hidden verification. It does **not** claim a larger raw task count: its own table lists LIBERO at 130 tasks and RoboDojo at 60 total tasks.

## Limitations

- Very recent preprint; independent reproduction is limited.
- Sim and real tasks are not necessarily one-to-one paired, so sim-to-real predictiveness is not established as cleanly as in SIMPLER.
- Remote real evaluation depends on specific hardware and organizer infrastructure.
- Cross-policy comparison may still reflect different data, fine-tuning and implementation budgets.

## Why It Matters

For an FYP, RoboDojo is a strong candidate when the research question concerns bimanual capability coverage, hidden-test robustness or unified sim-and-real protocol. It complements rather than automatically replaces LIBERO.

## Reading Questions

1. Do hidden layouts sufficiently prevent leakage if model training data are not auditable?
2. How should partial progress be aggregated across tasks of different lengths?
3. Does the real benchmark preserve policy ranking seen in simulation?

## Links

- [Paper](https://arxiv.org/abs/2607.04434)
- [Project](https://robodojo-benchmark.com/)
- [Code](https://github.com/RoboDojo-Benchmark/RoboDojo)
- [Chen Tianxing](https://tianxingchen.github.io/index.html)
