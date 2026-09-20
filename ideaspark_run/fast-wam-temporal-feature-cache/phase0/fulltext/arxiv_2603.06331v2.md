# WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching

paper_id: arxiv:2603.06331v2
tier: U
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

World models (Bar et al., 2025; Liu et al., 2024; Bruce
et al., 2024; Agarwal et al., 2025; Hafner et al., 2025; An
et al., 2025) have recently emerged as a compelling founda-
tion for building more general-purpose intelligence. Rather
than merely generating observed contents, world models
aim to capture spatiotemporal dynamics of the environment,
enabling long-horizon imagination for planning, decision
making, and interactive agents. With the rapid progress of
large-scale generative models (Croitoru et al., 2023; Naveed
et al., 2025; Feng et al., 2026b), generation-driven world
models (Russell et al., 2025; Bar et al., 2025; Huang et al.,
2025; Zhu et al., 2025; Li et al., 2026) built upon diffusion
models have gained increasing attention for synthesizing im-
mersive, coherent, and even interactive virtual environments
from large-scale data.
However, modern diffusion-based world models remain
costly at inference time since they require many denoising
steps with repeated backbone evaluations (Ho et al., 2020;
Lipman et al., 2022). Recently, various techniques (Feng
et al., 2025c; Xi et al., 2025; Feng et al., 2025b; Zhang
et al., 2024; Feng et al., 2025a; 2026c; 2025d) have been
developed to enable efficient diffusion inference. Among
which, feature caching (Selvaraju et al., 2024; Ma et al.,
2024b; Liu et al., 2025a) is particularly attractive due to its
training-free nature: it reduces sampling cost by reusing
or cheaply forecasting intermediate representations across
timesteps.
While feature caching has achieved strong speedups in
single-modal image or video diffusion, we identify that di-
rectly transferring existing policies to diffusion world mod-
els often leads to rapid error accumulation and unstable
rollouts. In particular, world-model simulation exhibits two
distinctive properties that fundamentally challenge conven-
tional caching:
❶Heterogeneous token evolution with a long-tailed dif-
ficulty profile. Unlike single-modal diffusion where to-
ken dynamics are relatively uniform, world models jointly
evolve tokens that correspond to different physical factors
(e.g., appearance vs. geometry) and different spatial deriva-
tion. Consequently, the predictability of token trajectories
is highly non-uniform: most tokens evolve smoothly and
are easy to reuse or extrapolate, but a small fraction ex-
hibit sharp, non-linear changes tied to physically critical
structures (e.g., motion boundaries or depth discontinuities).
This long-tailed difficulty makes uniform caching inherently
inefficient: a global conservative rule wastes computation
on the easy majority, whereas a global aggressive rule is
bottlenecked by the hard minority and causes overall drift.
❷Non-stationary temporal regimes where a few bot-
tleneck tokens dominate failure. World-model denoising
is also regime-dependent: the model may traverse long in-
tervals where trajectories are smooth and caching is reli-
able, followed by short intervals where dynamics become
abruptly non-linear. Importantly, caching failure is typically
triggered not by average feature change, but by the same
hard-to-cache token subset becoming unpredictable in these
difficult regimes. As a result, fixed skipping schedules may
miss critical updates, and global-threshold heuristics that
treat all tokens equally either (i) react too late when bottle-
neck tokens drift, or (ii) over-trigger due to benign changes
in easy tokens, yielding poor speed–quality trade-offs.
To address these challenges, we present WorldCache, a
training-free acceleration framework tailored for diffusion
world models through heterogeneous token caching. Our ap-
proach introduces Curvature-guided Heterogeneous Token
Prediction (CHTP), which uses a physics-grounded curva-
ture score to estimate token-wise predictability and assigns
different approximation rules: 0th-order reuse for stable
tokens, 1st-order extrapolation for near-linear tokens, and
a curvature-aware damped predictor for chaotic tokens. To
regulate when expensive backbone evaluations are neces-
sary, we further propose Chaotic-prioritized Adaptive Skip-
ping (CAS), which constructs a dimensionless drift indicator
by combining curvature with feature deviations. This yields
a unified, scale-normalized uncertainty score whose accu-
mulation triggers FULL computation precisely when the
bottleneck token subset begins to drift, enabling aggressive
skipping without destabilizing multi-modal rollouts.
Our contributions can be summarized as:
• We identify two world-model-specific challenges that
hinder existing diffusion caching methods: long-tailed
token predictability induced by multi-modal hetero-
geneity, and non-stationary temporal regimes where
bottleneck tokens dominate caching failure.
• We propose curvature-guided heterogeneous token pre-
diction that allocates different caching rules to to-
kens based on trajectory nonlinearity, with a dedicated
damped predictor for chaotic tokens.
• We introduce a chaotic-prioritized adaptive skip-
ping strategy with a curvature-induced dimensionless
2

WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching
w
h
w
h
w
h
Multi-modal
Latent Ȝt
RGB
Depth
Transformer 
Backbone ư޿
Curvature Analysis         Token Partioning
Stable
Linear
Chaotic
Masks
Curvature ض
Ư(ǘ) = ضٍǝ֩
If Ưacc ≥س
Else 
Heterogeneous Token Prediction
Direct Reuse
ǝ֩ = ǝ֩’
Linear Extrap.
ǝ֩ = ǝ֩’ + Ǐǚ֩
Damped Update
ǝ֩ = ǝ֩’ + Ǐǚ֩
adapt
ǚ֩
ǝ֩
ǚ֩’
ǚ֩
ǝ֩
ǝ֩
Linear
Chaotic
Stable
+Ư(ǘ)
Accumulated 
Drift   Ư՝եե
Drift
High
Scheduler
੢
Real Output ǝ֩∗
Slow
Surrogate
Output ǝ֩
…
Multi-modal
Content
w
h
RGB
Depth
Camera Guide
FULL
 Cache
Figure 2. Overview of the proposed WorldCache framework. The pipeline alternates between FULL backbone evaluation and CACHE
approximation. (Top) In each full computation step, tokens are partitioned into Stable, Linear, and Chaotic groups based on their curvature
κ. (Bottom) During caching steps, heterogeneous predictors (Reuse, Linear Extrapolation, or Damped Update) are applied accordingly.
(Left) The Chaotic-prioritized Adaptive Skipping (CAS) mechanism accumulates a curvature-normalized drift score Eacc specifically
from chaotic tokens, triggering a full computation only when critical drift is detected.
drift score, enabling a unified threshold for stable
caching decisions across heterogeneous token scales
and timesteps.
• Extensive experiments on diffusion world models
demonstrate that WorldCache substantially reduces
sampling cost while preserving multi-modal rollout
quality.
2. Related Works
2.1. Data-Driven World Models
Data-driven world models (Ha & Schmidhuber, 2018; Le-
Cun, 2022) learn predictive internal representations of the
environment to simulate futures for control and planning.
Classical methods build compact latent dynamics with re-
current state-space models, enabling imagination and policy
optimization (Hafner et al., 2019; 2020; 2023; 2025). More
recently, scaling laws in generative modeling have moti-
vated tokenized world models that generate high-fidelity,
long-horizon rollouts, including interactive environments
learned from large-scale video data (Bruce et al., 2024;
Agarwal et al., 2025) and large video generators discussed
as emergent “world simulators” (Liu et al., 2024; Rus-
sell et al., 2025; Bar et al., 2025). Building upon gener-
ative models, diffusion-based world models further adopt
DiT (Peebles & Xie, 2023) backbones to jointly model
coupled modalities (e.g., RGB and geometry/depth, option-
ally action-conditioned) for unified world representation
and downstream simulation (Huang et al., 2025; Zhu et al.,
2025). However, such unified multi-modal generation sub-
stantially amplifies inference cost, motivating acceleration
techniques tailored to world-model dynamics.
2.2. Feature Caching for Diffusion Models
Feature caching is a training-free paradigm that acceler-
ates diffusion sampling by exploiting temporal redundancy
across denoising steps. Existing methods can be broadly
grouped into: (i) reuse-based caching that skips computa-
tion by reusing intermediate representations across nearby
steps, often at block/layer granularity (Selvaraju et al., 2024;
Ma et al., 2024b;a; Kahatapitiya et al., 2025; Chen et al.,
2025); (ii) token-adaptive caching that applies selective
reuse to subset tokens while preserving other tokens for
full computation (Zou et al., 2024a;b; Zheng et al., 2025);
(iii) forecasting-based caching that predicts future features
via local trajectory approximation (e.g., Taylor expansion)
or trajectory integration, reducing long-interval drift (Liu
et al., 2025b); and (iv) runtime-adaptive scheduling that
decides when to cache using lightweight proxies or online
uncertainty signals (Liu et al., 2025a; Zhou et al., 2025).
However, most prior caching strategies are developed for
single-modal image/video diffusion and implicitly assume
relatively homogeneous feature dynamics. This assump-
tion becomes fragile in world models, where coupled
multi-modal tokens exhibit distinct physical evolution pat-
terns. This motivates us to introduce a token-heterogeneous
3

WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching
caching mechanism tailored to world models.
3. Preliminaries
Diffusion World Models with Transformer Backbones.
We consider a diffusion-based world model that generates a
multi-modal world state through T denoising steps, follow-
ing recent transformer-based diffusion world models such as
Voyager (Huang et al., 2025) and Aether (Zhu et al., 2025)
(we use Voyager for illustration). Let zr
t ∈Rc×f×h×w de-
note the RGB latents for 2D video generation and zd
t ∈
Rc×f×h×w denote the corresponding depth latents for 3D
estimation at timestep t. f, h, w denote the frame, height,
and width, respectively. We form the multi-modal latent by
spatial concatenation
zt = concat

zr
t, zd
t

∈Rc×f×2h×w.
(1)
The transformer backbone Fθ takes tokenized multi-modal
inputs and predicts the denoising direction in token space

## Method

4. WorldCache
4.1. Curvature-guided Heterogeneous Token Prediction
Observation ❶. World models exhibit strong token het-
erogeneity. As shown in Fig. 3, world models mix hetero-
geneous modalities (e.g., RGB video vs. depth) and exhibit
large spatial variance, yielding markedly different token tra-
jectories across denoising steps. As a result, a single global
caching rule (e.g., always reuse or always apply the same
linear predictor) is mismatched: conservative rules waste
computation on stable tokens, while aggressive rules fail on
a small subset of chaotic tokens and cause global drift.
Frame
Timestep 0
Depth
RGB
Timestep 49
(a) Visualization of curvature maps for RGB and Depth modalities.
(b) PCA trajectory visualization of different type tokens.
Figure 3. An illustration of token heterogeneity. (a) Modality
and Spatial Variance: Distinct patterns between modalities and
across spatial regions. (b) Trajectory Dynamics: Three trajectory
trends: static, predictable, and sharp, non-linear direction shifts
that defy simple extrapolation. More analysis in Appendix Sec. E
This motivates a token-adaptive caching strategy: estimate
how difficult each token is to predict from cached history,
then apply different approximations accordingly, allocating
compute only to tokens that need it.
Curvature as a physics-grounded predictability cue.
To
quantify how predictable each token is under caching, we
measure the local nonlinearity of its temporal trajectory
via a curvature-like score. Let yt ∈RN×d be the FULL
computation output in token space at timestep t, and let
yt,i ∈Rd denote token i (the i-th row) of yt. Given the last
three FULL outputs at timesteps t2 > t1 > t0 (following
the denoising order), we define discrete velocities
vt0,i = yt0,i −yt1,i
t0 −t1
,
vt1,i = yt1,i −yt2,i
t1 −t2
,
(5)
and a discrete acceleration
at0,i = vt0,i −vt1,i
t0 −t1
.
(6)
We compute the curvature score
κi =
∥at0,i∥2
∥vt0,i∥2
2 + ε,
(7)
where ε is a small constant (e.g., ε = 1e−8). This formula-
tion is physically motivated (Federer, 1959): v captures the
local drift of token features along denoising time, while a
captures how quickly this drift changes. More importantly,
the relevant skipped-step error is the deviation between the
4

WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching
true future token and its local first-order surrogate. For a
smooth local trajectory yi(τ), Taylor expansion gives
∥yi(τ + ∆) −yi(τ) −∆y′
i(τ)∥2 ≤∆2
2
sup
ξ∈[τ,τ+∆]
∥y′′
i (ξ)∥2.
(8)
Hence, the cache difficulty of first-order prediction is gov-
erned by second-order departure from local linearity rather
than raw displacement magnitude. In the smooth limit, cur-
vature, namely second-order variation normalized by local
speed squared, upper-bounds this local error under a local
speed bound. Eq. (7) is its finite-difference counterpart esti-
mated from the last three FULL outputs. Appendix Sec. A
provides the formal statements and proofs. Thus, κi acts
as a normalized “turning rate.” Small curvature indicates
nearly linear evolution that is amenable to reuse or extrapo-
lation. By contrast, large curvature indicates fast direction
changes, where naive caching becomes more drift-prone.
Token grouping by curvature.
We partition tokens by
curvature percentiles (computed from {κi}N
i=1):
Istable = {i : κi < Qps(κ)},
Ichaotic = {i : κi ≥Qpc(κ)},
Ilinear = [N] \ (Istable ∪Ichaotic),
(9)
where Qp(κ) denotes the p-quantile and (ps, pc) are fixed
percentiles. The masks are refreshed whenever a new FULL
output is obtained and three FULL outputs are available.
Ground Truth
Trajectory
ǘ*-1
ǚ֩∗−1
ǘ*
ǚ֩∗
Linear Extrapolation
(Overshoot/Drift)
Naive Reuse
(Lag Error)
WorldCache
Damped Update
(Stabilized)
 حօ3ǚ֩∗−1
ǘ*+1
ǘ*+2
ǘ*+3
 حօ1
 حօ2
(a) Visualization of different update direction.
(b) Quantitative analysis of cache error under different updates.
Figure 4. Mechanism and effectiveness of the Damped Update.
(a) Trajectory Illustration: Damped update stabilizes predic-
tion through historical vt⋆−1. (b) Quantitative Error Analysis:
Damped update reduces chaotic tokens cache error as the predic-
tion window k increases.
Heterogeneous prediction: easy tokens vs. chaotic to-
kens.
Let t⋆denote the most recent FULL computation
timestep, and let k be the number of consecutive CACHE
steps since t⋆. We denote the most recent FULL output as
yt⋆, the corresponding velocity vt⋆, and its token as yt⋆,i.
In CACHE, we construct a surrogate token-space output ˜yt
token-wise:
˜yt,i =





yt⋆,i,
i ∈Istable,
yt⋆,i + k · vt⋆,i,
i ∈Ilinear,
yt⋆,i + k · vadapt
i
(k),
i ∈Ichaotic.
(10)
Here vt⋆,i is the latest cached velocity (computed from the
two most recent FULL outputs).
Chaotic group. Tokens in Ichaotic exhibit high curvature
with abrupt direction shifts; naive 1st-order extrapolation
quickly accumulates errors and causes drift. To stabilize
long cached streaks, we adopt a curvature-aware damped up-
date by blending two recent velocities with a cubic Hermite
(smoothstep) schedule (Weisstein, 2002):
vadapt
i
(k) = (1 −αk)vt⋆,i + αkvt⋆−1,i,
αk = 3x2
k −2x3
k,
xk = min

k
nmax
, 1

,
(11)
where nmax is the maximum cached streak. As shown in
Fig. 4, this design reduces reliance on a single-step tangent
direction. As k grows, αk increases and the update becomes
more conservative, mitigating drift under high-curvature
dynamics while retaining caching efficiency.
4.2. Chaotic-prioritized Adaptive Skipping
Figure 5. An illustration of non-uniform temporal dynamics.
We plot the feature difference magnitude across denoising steps
for different token percentiles (p25 to p100). The global drift is
dominated by a small subset of ”hard” tokens (top percentile,
red line), while the majority remain stable. More analysis in
