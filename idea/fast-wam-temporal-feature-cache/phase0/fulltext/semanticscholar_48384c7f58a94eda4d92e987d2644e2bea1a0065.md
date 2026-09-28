# AdaCorrection: Adaptive Offset Cache Correction for Accurate Diffusion Transformers

paper_id: semanticscholar:48384c7f58a94eda4d92e987d2644e2bea1a0065
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Transformer-based diffusion models [1], [2] have emerged
as a leading class of generative models for image and video
synthesis. These models iteratively refine a noisy input over
hundreds of denoising steps, requiring full recomputation
of token embeddings at every layer and timestep. Despite
their impressive performance, the inference cost of Diffusion
Transformers (DiTs) remains prohibitive, especially in long
sequences and high-resolution settings [3].
To reduce this computational burden, recent works propose
caching intermediate features during sampling [4], [5]. These

## Method

computation, but rely on fixed policies or block-level reuse
schedules. As a result, they suffer from cache misalignment —
where outdated activations are incorrectly reused — degrading
generation quality or limiting reuse opportunities.
We propose AdaCorrection, a lightweight and training-
free module that introduces adaptive offset correction for
cached activations in DiTs. Unlike static reuse strategies,
AdaCorrection evaluates cache validity per layer using spatio-
temporal misalignment signals and adaptively blends cached
and fresh activations to restore alignment.
Our design integrates seamlessly with existing diffusion
pipelines and does not require model retraining or architecture
modification. The system operates in an on-the-fly fashion,
correcting cache entries in real time based on layer-wise de-
viation metrics such as temporal change and spatial variation.
Key contributions. We introduce offset cache correction for
diffusion transformers, enabling layer-wise validation and cor-
rection of reused activations during sampling to maintain gen-
eration fidelity. Building on this idea, we design a lightweight
alignment module that detects spatio–temporal drift using
sensitivity-aware offset estimators, thereby improving gener-
ation quality while preserving efficient cache reuse. Finally,
we demonstrate that AdaCorrection maintains near-original
FID (4.37 vs 4.42 Full Recompute, only 0.05 difference)
with competitive speed, showing that quality can be preserved
without sacrificing efficiency.
Figure 1 illustrates how static cache reuse causes feature
misalignment across layers and timesteps.
