# WorldCache: Content-Aware Caching for Accelerated Video World Models

paper_id: arxiv:2603.22286v1
tier: U
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

World models predict future visual states that are physically consistent and use-
ful for downstream decision-making, enabling agents to plan and act within simu-
lated environments [50]. Large-scale Diffusion Transformers (DiTs) have become
the dominant backbone for such models [9,47, 48], because spatio-temporal at-
tention over latent tokens captures the long-range dependencies central to world
consistency (e.g., object permanence and causal motion). However, this expres-
siveness comes at a steep computational cost: world-model rollouts require many
frames, and each frame is produced by sequentially invoking deep transformer
blocks across dozens of denoising steps [10,35]. The resulting latency is the pri-
mary obstacle to interactive world simulation and closed-loop deployment.
A natural remedy is to exploit redundancy along the denoising trajectory.
Consecutive steps often produce only small changes in intermediate features [13],
⋆Corresponding author: umair.nawaz@mbzuai.ac.ae
arXiv:2603.22286v1  [cs.CV]  23 Mar 2026

2
U. Nawaz et al.
Fig. 1: Qualitative and quantitative comparison of acceleration methods on
video world model generation using Cosmos-Predict2.5-2B. Left: Visual com-
parison of Baseline (no acceleration), FasterCache, DiCache, and WorldCache (ours)
across three representative timesteps (T1, . . . , TN) of a driving scene from the City
Street domain. FasterCache achieves 1.6× speedup but introduces severe visual ar-
tifacts and scene hallucinations. DiCache (1.3×) better preserves scene fidelity but
exhibits noticeable spatial artifacts at later timesteps (red dashed boxes). WorldCache
(ours) achieves 2.3× speedup while faithfully reproducing scene content, motion, and

## Method

dynamic rollouts.
2. We introduce WorldCache, a unified framework that improves both when to
skip (motion and saliency-aware decisions) and how to approximate (optimal
blending and motion compensation), while adapting to the denoising phase.
3. We demonstrate state-of-the-art training-free acceleration on multiple DiT
backbones, achieving up to 2.3× speedup with 99.4% quality retention on
Cosmos-Predict2.5, and show that the approach transfers across model scales
and conditioning modalities.
2
