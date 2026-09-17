# SIMPLER

**Paper:** *Evaluating Real-World Robot Manipulation Policies in Simulation*  
**System:** SIMPLER / SimplerEnv  
**Version:** CoRL 2024 paper  
**Local PDF:** [paper-corl2024.pdf](paper-corl2024.pdf)  
**Priority:** Core for sim-to-real validity; 30–45 min

## Background

SIMPLER asks a different question from LIBERO: not whether a policy succeeds on a simulated training task, but whether simulation can predict the relative real-world performance of policies trained on real robot data.

## Problem

Real-robot evaluation is slow and costly, while a simulator is useful only if its conclusions track real deployments.

## Method

The authors build visually matched SAPIEN environments for Google Robot and WidowX/BridgeData V2 settings, align control and physical parameters through system identification, and compare roughly 1,500 paired simulation/real episodes across eight tasks. Metrics include Pearson correlation and Mean Maximum Rank Violation (MMRV).

## Key Innovation

Evaluation quality is validated externally using paired real execution and relative policy ranking, rather than assumed from simulator realism.

## Main Results

Across major configurations, the authors report high aggregate correlation, around r=0.92, and low MMRV around 0.06–0.08. Task-level correlations vary; not every task is equally predictive. The paper also demonstrates sensitivity to controlled distribution shifts.

## Claim vs LIBERO

**Synthesis:** SIMPLER is stronger than standard LIBERO for claims about real-world ranking because it empirically checks sim-real agreement for real-trained policies. It is not broader in task count, semantic reasoning or embodiment coverage.

## Limitations

- Original scope is eight mostly rigid-object tasks and two embodiments.
- Good relative ranking does not guarantee accurate absolute success.
- Environment construction still requires substantial visual and control matching.
- Authors explicitly frame SIMPLER as a complement, not a replacement, for real evaluation.

## Why It Matters

If an FYP claims deployment relevance, SIMPLER's evaluation philosophy is more important than another in-simulation leaderboard score.

## Reading Questions

1. Which simulator parameters are tuned with knowledge of real test results?
2. Does ranking remain stable for policies outside the calibration set?
3. What minimum paired real evaluation is needed to validate a new task?

## Links

- [Paper](https://arxiv.org/abs/2405.05941)
- [Project](https://simpler-env.github.io/)
- [Code](https://github.com/simpler-env/SimplerEnv)
