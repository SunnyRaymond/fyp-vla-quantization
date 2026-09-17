# VLABench

**Paper:** *VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks*  
**Version:** arXiv:2412.18194v1; ICCV 2025 accepted per official repository  
**Local PDF:** [paper-arxiv.pdf](paper-arxiv.pdf)  
**Priority:** Core for semantic/long-horizon evaluation; 30–45 min

## Background

VLABench targets the gap between low-level manipulation success and the world knowledge, language understanding and multi-step reasoning expected from a VLA.

## Problem

Can a policy interpret implicit instructions, use commonsense and semantic knowledge, plan long-horizon composite tasks and generalize across diverse assets and embodiments?

## Method

The benchmark defines 100 task categories: 60 primitive and 40 composite. It includes more than 2,000 objects, strong domain randomization, multiple observations and embodiments, automatic trajectory generation and separate tests of VLM reasoning components. Composite tasks can exceed 500 time steps on average versus about 120 for primitive tasks.

## Key Innovation

It combines semantically rich language, logic, knowledge, long-horizon composition, large asset diversity and VLA-compatible control in one benchmark.

## Main Results

The benchmark reports generally low performance for OpenVLA, Octo and RDT-1B, especially on composite tasks, indicating substantial headroom. Its comparison table lists VLABench with 100 tasks, 163 object categories and 2,164 objects, versus LIBERO with 130 tasks, 51 categories and 75 objects, while attributing stronger semantic/logic/knowledge/randomization features to VLABench.

## Authors' Claim vs LIBERO

**Author table plus synthesis:** VLABench claims richer semantics, reasoning, object diversity, cross-embodiment support and automated data generation. It has fewer named tasks than LIBERO's 130 and does not provide a controlled same-training head-to-head.

## Limitations

- High difficulty can conflate language, perception, planning and control failures.
- Automatic expert trajectories and assets may introduce bias.
- Local file is arXiv v1; later conference materials may differ.

## Why It Matters

Use VLABench when a VLA paper claims knowledge transfer, compositional language or long-horizon reasoning beyond familiar tabletop routines.

## Reading Questions

1. Can failure be localized to VLM reasoning versus action execution?
2. Are implicit instructions objectively solvable from the scene and common knowledge?
3. Do composite tasks reward reusable skills or mainly longer error accumulation?

## Links

- [Paper](https://arxiv.org/abs/2412.18194)
- [Project](https://vlabench.github.io/)
- [Code](https://github.com/OpenMOSS/VLABench)

