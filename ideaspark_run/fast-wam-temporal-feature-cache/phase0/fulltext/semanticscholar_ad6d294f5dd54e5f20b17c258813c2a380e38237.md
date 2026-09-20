# SenCache: Accelerating Diffusion Model Inference via Sensitivity-Aware Caching

paper_id: semanticscholar:ad6d294f5dd54e5f20b17c258813c2a380e38237
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Diffusion models [11, 35] and flow matching models [1,
20] have reshaped generative modeling by setting the state
of the art in image and video synthesis. Despite their suc-
cess, diffusion inference remains computationally expen-
sive: sample generation requires multiple denoising iter-
ations, and each iteration incurs a full forward pass of a
large network. This cost is especially prohibitive for mod-
ern video diffusion transformers, which contain billions of
parameters and can require minutes of computation even
for a few seconds of video [38, 40]. Reducing inference
latency—without retraining the model or degrading output
quality—has therefore become a key challenge for practical
deployment.
Among acceleration strategies, caching-based meth-
ods [15, 21, 28] are particularly appealing because they
reduce inference cost by reusing previously computed de-
noiser outputs, without retraining (as in distillation-based
methods
[23, 31, 36]) or modifying the model architec-
ture. The underlying premise is that denoiser outputs at
consecutive timesteps can be sufficiently similar, allowing
cached outputs to replace expensive forward evaluations.
Existing methods, however, determine these reuse timesteps
using empirical heuristics.
For instance, TeaCache [21]
builds cache-reuse rules through output residual modeling
with time embedding difference or modulated input differ-
ence, while MagCache [28] selects reuse timesteps based
on the magnitude of the residual (the difference between
the model’s prediction and its input). While effective in
favorable regimes, these heuristics have two fundamental
limitations: (1) they lack theoretical justification and re-
quire extensive hyperparameter tuning, and (2) they pro-
duce static caching schedules that cannot adapt to the vary-
ing difficulty of each sample. As a consequence, caching
may over-cache challenging samples or under-cache easy
ones, because reuse decisions are not adapted to sample-
specific dynamics.
In this work, we propose a sensitivity-based criterion for
cache/reuse decisions in diffusion inference. Our key idea
is to use the denoiser’s local sensitivity—i.e., the variation
of its output with respect to perturbations in the noisy latent
and timestep—as a proxy for output change between neigh-
boring denoising steps. We show that this variation is well
characterized by the denoiser’s derivatives with respect to
the noisy latent and the timestep. These sensitivities quan-
tify the effect of latent drift and timestep spacing on the
denoiser output, allowing us to predict when the induced
output change is sufficiently small for cache reuse.
Through analysis and empirical study, we show that lo-
cal sensitivity is a strong predictor of caching error, and
that both latent and timestep sensitivities contribute signifi-
cantly. This reveals a core limitation of prior heuristic poli-
cies, which do not explicitly model both sources of varia-
tion.
Motivated by this insight, we introduce Sensitivity-
Aware Caching (SenCache),
a principled,
dynamic
caching framework that adapts cache/reuse decisions to
each sample. At every denoising step, SenCache predicts
the denoiser output change using a first-order sensitivity ap-
proximation and reuses the cached output only when the
predicted deviation is below a target tolerance. Our experi-
ments on three state of the art video diffusion models, Wan
2.1 [38], CogVideoX [40], and LTX-Video [10], show that
SenCache outperforms existing caching strategies in visual
quality under similar computational budgets.
SenCache framework offers several advantages:
1. It provides a theoretically motivated decision rule for
caching with an explicit tolerance for controlling the
speed–quality trade-off.
2. It provides a sensitivity-based interpretation of why prior
heuristics succeed in some regions and fail in others.
3. It adapts cache/reuse decisions per sample, unlike prior
methods that use fixed timesteps for all samples.
4. It requires no additional training and no model modifi-
cation, and is agnostic to architecture and sampler.
5. While our experiments focus on visual domain, the un-
derlying principle of using network sensitivity as a proxy
for cache/reuse decisions is general and can be extended
to other domains, such as audio and human motion.

## Method

timesteps and require extensive tuning. We address this lim-
itation with a principled sensitivity-aware caching frame-
work. Specifically, we formalize the caching error through
an analysis of the model output sensitivity to perturba-
tions in the denoising inputs, i.e., the noisy latent and the
timestep, and show that this sensitivity is a key predic-
tor of caching error. Based on this analysis, we propose
Sensitivity-Aware Caching (SenCache), a dynamic caching
policy that adaptively selects caching timesteps on a per-
sample basis. Our framework provides a theoretical basis
for adaptive caching, explains why prior empirical heuris-
tics can be partially effective, and extends them to a dy-
namic, sample-specific approach. Experiments on Wan 2.1,
CogVideoX, and LTX-Video show that SenCache achieves
better visual quality than existing caching methods under
similar computational budgets. The code is available at
https://github.com/vita-epfl/SenCache.git
arXiv:2602.24208v1  [cs.CV]  27 Feb 2026
