# LIBERO

**Paper:** *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*  
**Version:** arXiv:2306.03310v2; NeurIPS 2023 Datasets and Benchmarks  
**Local PDF:** [paper-arxiv.pdf](paper-arxiv.pdf)  
**Reading time:** 15–25 min for baseline context

## Background

LIBERO was designed for **lifelong robot learning**, especially the transfer of declarative and procedural knowledge. It was not originally designed as a modern VLA robustness benchmark, even though later VLA papers adopted it as a standard evaluation suite.

## Problem

The paper asks how a robot can learn a sequence of tasks without forgetting and how knowledge transfers across object, spatial and goal changes.

## Method

LIBERO builds on robosuite/MuJoCo and uses a procedural task-generation pipeline. It defines 130 language-conditioned tasks across LIBERO-Spatial, LIBERO-Object, LIBERO-Goal and LIBERO-100, with human-teleoperated demonstrations. In current VLA practice, evaluation commonly uses four 10-task suites: Spatial, Object, Goal and the 10 long-horizon tasks from LIBERO-100.

## Key Innovation

The benchmark disentangles knowledge-transfer factors and provides standardized demonstrations, task definitions and sparse goal predicates.

## Main Results

The original study focuses on lifelong learning algorithms, architectures, task ordering and pretraining. It finds, among other things, that sequential finetuning can outperform tested lifelong-learning methods in forward transfer and that no visual encoder is best for every transfer type.

## Limitations for modern VLA evaluation

- The original scientific target is lifelong learning, not broad open-world VLA robustness.
- Standard VLA train/eval protocols often reuse nearly identical task semantics, scenes and language, making high success compatible with memorization or narrow interpolation.
- Success predicates are binary and may not expose partial progress, unsafe behavior or trajectory quality.
- Tabletop MuJoCo simulation does not by itself establish real-world validity.

## Why It Matters

LIBERO remains essential because it provides continuity with a large body of VLA results. The right response to saturation is usually to keep LIBERO as an in-distribution anchor and add OOD, cross-domain or real-alignment evaluation—not to silently replace it.

## Reading Questions

1. Which part of your claim needs lifelong-transfer evidence, and which part needs OOD robustness?
2. Does the evaluation split differ meaningfully from training in scene, language, object and task structure?
3. Is binary success sufficient for your model's expected failure modes?

## Links

- [Paper](https://arxiv.org/abs/2306.03310)
- [Project](https://libero-project.github.io/)
- [Code](https://github.com/Lifelong-Robot-Learning/LIBERO)
