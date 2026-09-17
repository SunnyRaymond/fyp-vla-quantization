# WorldEval

**Paper:** *WorldEval: World Model as Real-World Robot Policies Evaluator*  
**Version:** arXiv:2505.19017v1, 2025-05-25  
**Local PDF:** [paper-arxiv.pdf](paper-arxiv.pdf)  
**Priority:** Core for learned evaluator lineage; 25–40 min

## Background

WorldEval uses a learned video world model to evaluate policies trained for real robots, aiming to scale policy ranking and safety screening without manually constructing a simulator for every embodiment and task.

## Problem

Can action-conditioned generated videos preserve real-world policy rankings and expose unsafe or collapsed actions well enough to guide policy development?

## Method

Policy2Vec uses latent representations from the policy itself as the action encoding for a finetuned WAN 2.1 video model. A multimodal judge evaluates generated rollouts, while FID is explored as a cheap proxy for simple tasks. The study evaluates Diffusion Policy, OpenVLA, DexVLA and π0 using more than 1,000 real trials; the world model is trained on 1,400 robot trajectories.

## Key Innovation

It avoids training a separate high-dimensional action encoder by exploiting each policy's internal representation and makes relative ranking, checkpoint selection and safety screening the evaluation objective.

## Main Results

Against a real-to-sim implementation using SIMPLER-style visual matching, WorldEval reports average Pearson r=0.942 versus 0.411 and MMRV=0.044 versus 0.261 on three tasks. Policy2Vec outperforms VQVAE and one-hot action encodings. OOD scene tests report r=0.927 and MMRV=0.047.

## Claim vs LIBERO

**Synthesis:** WorldEval's advantage is scalable real-policy evaluation without hand-building a physics task for every setting. This does not establish superiority to LIBERO on standardized simulated capability testing. The paper could not directly run SIMPLER because of equipment constraints; it only applied related real-to-sim techniques.

## Limitations

- Generated videos show object deformation, disappearance, ghosting and hallucination, especially for poor policies.
- Actions are not always faithfully reflected, and the world model may require retraining for novel domains.
- The success judge adds another model-dependent error source.
- The evaluation is small relative to claims of thousands of tasks.

## Why It Matters

WorldEval is the conceptual starting point for current learned world-model evaluators and helps frame evaluation as a policy-ranking problem rather than photorealistic video generation.

## Reading Questions

1. Does using a policy's own latent representation bias comparison across architectures?
2. How often does the world model turn a failed action into a plausible success?
3. Can calibration on a few real trials detect ranking inversion?

## Links

- [Paper](https://arxiv.org/abs/2505.19017)
- [Project](https://worldeval.github.io/)

