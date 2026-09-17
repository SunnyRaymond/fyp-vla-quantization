# RoboTwin

**Paper:** *RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins*  
**Author identity:** Chen Tianxing is a co-first author  
**Version:** arXiv:2504.13059v1, 2025-04-17  
**Local PDF:** [paper-arxiv.pdf](paper-arxiv.pdf)  
**Priority:** Companion; 20–30 min

## Background

RoboTwin is the predecessor of RoboTwin 2.0 and introduces the Generative Digital Twin idea for dual-arm manipulation.

## Problem

Dual-arm training lacks diverse demonstrations and evaluation settings that align simulated objects/tasks with real deployments.

## Method

The framework uses 3D generative foundation models to reconstruct digital twins from object images and uses LLM-assisted spatial-relation-aware code generation for expert trajectories. The original benchmark contains 15 dual-arm tasks on ManiSkill3, with simulation and real-world data on the COBOT Magic platform.

## Key Innovation

It couples object-level digital-twin generation with task-code generation, providing both data and a sim/real bimanual benchmark.

## Main Results

The paper reports that pretraining on generated simulation data and finetuning on 20 real samples improves success by more than 70% for selected single-arm tasks and more than 40% for selected dual-arm tasks relative to training only on the 20 real samples.

## Claim vs LIBERO

**Synthesis:** RoboTwin addresses dual-arm coordination, generated object twins and sim-to-real data alignment—dimensions outside LIBERO's original lifelong single-arm tabletop focus. It does not present a controlled LIBERO head-to-head, and its 15-task scale is smaller.

## Limitations

- v1 benchmark is small and superseded in scope by RoboTwin 2.0.
- 3D reconstruction, code generation and physics quality can each become a hidden source of bias.
- Reported gains are task-specific and depend on the COBOT Magic setup.

## Why It Matters

Read it to understand the design lineage behind RoboTwin 2.0 and Chen Tianxing's move from generated digital twins toward scalable randomized evaluation.

## Reading Questions

1. Which parts of the pipeline require manual correction?
2. Are generated twins visually realistic, physically accurate, or both?
3. Does simulation pretraining preserve gains outside the demonstrated hardware and tasks?

## Links

- [Paper](https://arxiv.org/abs/2504.13059)
- [Project](https://robotwin-benchmark.github.io/)
- [Code](https://github.com/robotwin-Platform/RoboTwin)
