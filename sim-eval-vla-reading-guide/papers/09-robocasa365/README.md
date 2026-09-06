# RoboCasa365

**Paper:** *RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots*  
**Version:** ICLR 2026 conference paper  
**Local PDF:** [paper-iclr2026.pdf](paper-iclr2026.pdf)  
**Priority:** Extended ecosystem reading; 30–45 min

## Background

RoboCasa365 extends household manipulation from narrow tabletop tasks toward large multi-task kitchen environments and mobile manipulation.

## Problem

How can robot learning scale to hundreds of household tasks, thousands of scenes and enough human/synthetic data for foundation-model training and continual skill acquisition?

## Method

The ecosystem contains 365 kitchen tasks—65 atomic and 300 composite—across 2,500 scenes. It provides more than 500k demonstrations, around 612 hours of human data plus 1,615 hours of synthetic data. Tasks include 220 mobile-manipulation and 145 stationary settings, with long sequences of subtasks.

## Key Innovation

It integrates large-scale scene/task generation, a very large demonstration corpus, mobile manipulation and evaluation for multi-task, foundation-model and lifelong-learning settings.

## Main Results

The paper reports that adding simulation data to real training raises average success on four real tasks from 61.8 to 79.8, about +18.0 percentage points. Large pretraining generally improves downstream simulated and real performance.

## Claim vs LIBERO

**Synthesis:** RoboCasa365 is stronger for room-scale household diversity, mobile manipulation, training-data scale and long composite tasks. It is not primarily a controlled robustness audit and does not show a same-policy LIBERO head-to-head.

## Limitations

- Domain is still kitchen-only.
- Simulation sensing and physics do not cover all real-world errors.
- Large data volume makes fair comparison expensive.
- Evaluation subsets may cover much less than all 365 tasks and 2,500 scenes.

## Why It Matters

Read it when your FYP needs a scaling reference or household/mobile manipulation direction, not as the first replacement for a lightweight LIBERO protocol.

## Reading Questions

1. What fraction of performance comes from scene diversity versus task diversity?
2. Are mobile and stationary results reported separately enough for diagnosis?
3. How much real validation supports the large simulated space?

## Links

- [Paper](https://robocasa.ai/assets/robocasa365_iclr26.pdf)
- [Project](https://robocasa.ai/)
- [Code](https://github.com/robocasa/robocasa)
