# BeCARE: Budgeted Cache Refresh for Diffusion Transformer Acceleration

paper_id: semanticscholar:d68249a8f670dcf5b435537b62c8a2e06ed8e05d
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Diffusion models have become a dominant paradigm for
high-fidelity image and video generation (Rombach et al.
2022; Blattmann et al. 2023), with diffusion transformers
(DiTs) emerging as a scalable backbone (Peebles and Xie
2023). However, their strong capability comes with substan-
tial inference cost. Each denoising timestep repeatedly exe-
cutes a deep stack of Transformer blocks, and the sampling
process typically requires dozens of such steps. This heavy
sequential computation leads to high latency and FLOPs,
posing a major obstacle to the practical deployment of DiTs.
0
10
20
30
40
(a) Denoising step
0.00
0.02
0.04
0.06
0.08
0.10
Relative latent deviation
Early
Mid
Late
0
10
20
30
40
(b) Denoising step
0.00
0.25
0.50
0.75
1.00
1.25
1.50
Observer drift mt
10--90% band
200-prompt median
low
mid
high
Figure 1: Temporal and prompt-dependent structure of
cache-refresh signals. (a) Relative latent deviation from the
uncorrupted trajectory after a single order-2 Taylor interven-
tion early (step 5), mid (25), or late (45). (b) Online drift
signal under the deployed policy (200 prompts): per-step
median, 10–90% band, and three example prompts.
Therefore, accelerating DiT inference while preserving gen-
eration quality has become an important research problem.
To alleviate this computational bottleneck, existing DiT
acceleration methods mainly follow two directions. The
first reduces the number of denoising steps using non-
Markovian samplers (Song, Meng, and Ermon 2021), high-
order solvers (Lu et al. 2022), or model distillation (Qiu, Lu,
and Wang 2025). Although effective, these methods typi-
cally modify the original sampling process or require addi-
tional training. The second reduces the computational cost
of each denoising step through feature caching. This strategy
performs a full Transformer computation at selected steps,
stores the resulting intermediate features, and reuses or fore-
casts them at subsequent steps. Existing studies have mainly
focused on improving how skipped computations are ap-
proximated, ranging from direct feature reuse (Ma, Fang,
and Wang 2024; Selvaraju et al. 2024; Liu et al. 2025a) to
feature forecasting, error correction, and speculative veri-
fication (Liu et al. 2025c; Qiu et al. 2025a; Zheng et al.
2026b; Liu et al. 2025d). A complementary line refines which
components to approximate, at token (Zou et al. 2025; Zhu
et al. 2026), cluster (Zheng et al. 2025), frequency (Liu et al.
2025b), or subspace (Chen et al. 2026; Zheng et al. 2026a)
granularity. This paradigm requires no additional training,
preserves the pretrained model and sampler configuration,
and can be readily applied to existing diffusion transformers.
Despite this rapid progress, existing caching methods still
arXiv:2605.27075v2  [cs.CV]  29 Jul 2026

determine cache refresh timing using fixed intervals (Sel-
varaju et al. 2024; Liu et al. 2025c) or hand-tuned error
thresholds (Liu et al. 2025a,d). The former makes compute
predictable but ignores temporal and prompt-dependent risk;
the latter adapts locally but is budget-blind: its realized com-
pute is an emergent outcome of the threshold, so meeting a
deployment target requires searching thresholds by trial and
error. Moreover, these timestep-wise triggers implicitly as-
sume that approximation errors have similar consequences
regardless of when they occur. This assumption fails in se-
quential denoising. We probe this directly: on three fixed
prompts, we replace the full computation with a single order-
2 Taylor approximation at step 5, 25, or 45 of a 50-step trajec-
tory, keep the remaining 49 steps exact, and track the relative
latent deviation from the uncorrupted run (Figure 1 (a)). We
find that the early intervention never recovers, because every
subsequent exact step evolves from a shifted state: its ter-
minal deviation (0.100, red) is roughly 11× that of the late
one (0.009), with the mid intervention in between. Part of
this asymmetry is predictable in advance: the sampler fixes
its noise schedule before sampling begins, and this schedule
already indicates where along the trajectory errors are most
consequential. An offline schedule alone is not sufficient,
however. Figure 1 (b) tracks the drift signal our controller
observes online (the deviation of each step’s first-block em-
bedding from its Taylor prediction, under the deployed pol-
icy) across all 200 benchmark prompts: the per-step median
is far from flat, and at a single denoising step the 10–90%
band spans up to 1.07, so a prediction that is still reliable for
one prompt has already drifted badly for another. No fixed re-
fresh pattern serves all prompts equally well. Consequently,
under the same budget of full computations and identical
FLOPs, changing only the placement of these computations
can shift output fidelity by several decibels. At matched com-
pute, the bottleneck is where full computations are placed,
not how many there are.
We therefore formulate cache refresh scheduling as a
feedback-control problem and present Budgeted Cache
Refresh (BeCARE), a training-free control layer built on
top of a forecasting-based caching engine. BeCARE first
extracts an amplification profile from the sampler’s noise
schedule, which provides a prior on the impact of approxi-
mation errors at different denoising stages. During sampling,
an observer weights the engine’s own extrapolation residuals
by this profile and by the current cache age, and accumulates
them into a risk score that triggers a refresh once contin-
ued caching becomes harmful. Because the residuals are
measured online for each prompt, refresh positions are not
fixed in advance: prompts with fast-moving features receive
their refreshes earlier and closer together, while for easier
prompts the controller holds refreshes back for later steps.
This addresses the prompt-to-prompt variability shown in
Figure 1 (b). However, risk-based triggering alone remains
budget-blind. To address this issue, BeCARE accepts a user-
specified cap on the number of full computations and closes
a feedback loop around the budget: the cumulative form of
the same amplification profile serves as an analytic reference
for how much of the cap should have been spent by each step,
and a feedback law compares realized spending with this ref-
erence and adjusts the refresh threshold, so the cap is neither
exhausted early nor left unused late. Finally, a small set of
structural guards completes the loop: a fixed initial warmup,
plus a reserve for the final steps and a limit on how long a
cache may be reused, both following from the cap in closed
form. With these pieces the realized compute matches the
specification, and one calibration at a single operating point
transfers across budget tiers with no per-tier tuning.
The contributions of our method are threefold:
• We show that the harm of a cache approximation depends
strongly on when it occurs, that this asymmetry is antic-
ipated by an amplification profile read directly from the
sampler’s noise schedule at almost zero cost, and that at
fixed FLOPs the placement of full evaluations, not their
number, dominates output quality.
• We turn the when-to-refresh decision from hand-tuned
thresholds into feedback control, in which a single am-
plification profile drives both a drift-risk observer and
a reference-tracking budget controller, governed by one
user-facing knob and a small set of structural guards.
• On
FLUX.1-dev,
BeCARE
attains
the
best
full-
referenced fidelity among training-free caching baselines
at every tested ratio from 3× to 6×, while meeting tight
budgets exactly. On SD3.5 Large, it substantially im-
proves the same Taylor cache engine over fixed-interval
scheduling at matched compute, demonstrating the value
of prompt-adaptive refresh placement.
2

## Method

ate features across adjacent timesteps. Reuse-based methods
replay stored module outputs (Ma, Fang, and Wang 2024;
Selvaraju et al. 2024), later work forecasts or corrects them in-
stead (Taylor extrapolation (Liu et al. 2025c), gradient-based
error compensation (Qiu et al. 2025a,b), linear-multistep and
ODE-solver forecasters (Cui et al. 2026; Zheng et al. 2026b)),
and a parallel line refines the granularity of caching (to-
ken (Zou et al. 2025; Zhu et al. 2026), cluster (Zheng et al.
2025), frequency (Liu et al. 2025b), subspace (Chen et al.

(A) Caching engine
(B) Amplification-weighted drift observer
causal: proxy updated only after real Fulls · warm-start seed after warmup
(C) Budget controller & guard-first decision
“A black colored dog.”
𝝐
· · ·
warmup 𝐾
denoising steps 𝑡= 0, 1, . . . (one realized Full/Cache trace)
· · ·
latest anchor 𝑡𝑎: Full
all blocks computed
current step 𝑡: decide
blocks skipped if Cache
· · ·
x0
realized Full count
𝐹𝑇≤𝑁cap
(met exactly under scarcity)
bF: forecast block outputs
store F(x𝑡𝑎)
apply Full/Cache decision
anchor
features
F(x𝑡𝑎), . . .
at true Fulls
divided
differences
ΔF,
Δ2 F
order-2
forecast
bF,
bz𝑡
residuals
𝑒(1)
𝑡
, 𝑒(2)
𝑡
,
𝑒(𝑐)
𝑡
bz𝑡
weighted
drift 𝑚𝑡
amplification 𝐴𝑡(from ˜𝜎𝑡)
cache age
𝑔(𝑎𝑡)
×
risk mass
R𝑡+=
𝐴𝑡𝑚𝑡𝑔(𝑎𝑡)
