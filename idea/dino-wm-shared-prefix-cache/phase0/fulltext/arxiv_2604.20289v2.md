# X-Cache: Cross-Chunk Block Caching for Few-Step Autoregressive World Models Inference

paper_id: arxiv:2604.20289v2
tier: T3
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Generative world models are emerging as a key infrastructure for autonomous driving. By condi-
tioning on ego actions and sensor history, these models generate photorealistic future observations,
enabling two capabilities that are difficult to achieve through real-world testing alone: scalable
closed-loop evaluation of end-to-end driving policies, and online reinforcement learning where a
policy explores counterfactual maneuvers in a controlled, repeatable environment [21, 15, 10]. In
both settings, the world model must operate as an interactive simulator: generating observations,
receiving actions, and responding in real time over long-horizon rollouts.
Autoregressive video diffusion [19, 9, 1, 3] has become a compelling backbone for such simulators.
Unlike bidirectional video diffusion that generates an entire clip jointly, autoregressive formulations
produce video chunk by chunk with causal attention, conditioning each new chunk solely on previ-
ously generated content. This causal, streaming structure naturally supports interactive simulation,
where the model must respond to each newly issued action without waiting for a full clip to complete.
Combined with a persistent key-value (KV) cache [8, 9, 21] that bounds memory regardless of
sequence length, these models enable stable, unbounded-length generation. To meet throughput
demands, they are further distilled to few-step denoising schedules [18, 19, 9]. This work focuses on
accelerating inference for few-step autoregressive video diffusion deployed in interactive, closed-loop
world simulation.
A growing line of training-free caching methods [12, 4, 11, 5] accelerates diffusion inference by
reusing block outputs in Diffusion Transformers (DiTs) [14] across adjacent denoising steps, where
structural changes between steps are often small enough [5, 11] to permit substantial reuse. Recent
extensions to autoregressive video follow the same cross-step design: FlowCache [13] applies chunk-
wise caching policies, while SCOPE [6] combines tri-modal scheduling with predictive extrapolation
along the denoising trajectory. However, this strategy is poorly suited to the few-step regime required
for real-time interactive world simulation. With only four denoising steps, the cross-step similarity
that these methods rely on is greatly reduced. Each step contributes substantial and non-redundant
structural updates, leaving little room for safe reuse. FlowCache and SCOPE are subject to the same

## Method

We first establish notation for the AR video diffusion inference loop, then describe each component
of the caching mechanism.
2.1
Preliminaries: AR Video Diffusion Inference
Let n index the generation step (i.e., the n-th chunk of video), t ∈{0, . . . , S−1} the denoising step
within a chunk, and b ∈{0, . . . , B−1} the DiT block index. At generation step n, chunk latents are
initialized from Gaussian noise and iteratively refined through S denoising steps, where each step
passes through all B DiT blocks:
x(n)
t,b = x(n)
t,b−1 + fb

x(n)
t,b−1; c(n)
t

,
(1)
where x(n)
t,0 is the latent at the start of block 0 at denoising step t, fb(·; c) denotes the b-th DiT block
parameterized by conditioning c (timestep embedding, action embedding, text embedding, etc.), and
the operation is a standard residual connection. We define the block residual as:
r(n)
t,b
= fb

x(n)
t,b−1; c(n)
t

= x(n)
t,b −x(n)
t,b−1.
(2)
After all S denoising steps complete for chunk n, the model performs a KV update pass: it runs
one additional forward pass through all blocks using the fully denoised (clean) latent, computing
key-value projections that are appended to the persistent KV cache for conditioning future chunks via
cross-attention. In rolling KV cache implementations, oldest entries are evicted via FIFO when the
cache reaches capacity.
3

AI INFRA TEAM
X-Cache
TECHNICAL REPORT
Figure 2: Overall architecture of X-Cache
2.2
Cross-Chunk Residual Caching
2.2.1
Key observation
In AR video generation of physically continuous environments (especially in autonomous driving),
consecutive chunks depict scenes that change smoothly relative to the generation rate. The block
input x(n)
t,b−1 at position (t, b) is therefore highly similar to x(n−1)
t,b−1 at the same position in the previous
generation step. This cross-chunk redundancy is independent of the number of denoising steps S
and persists even under aggressive few-step distillation (e.g., S = 4), unlike cross-step redundancy,
which diminishes as S decreases.
2.2.2
Caching mechanism
When block b is fully computed at generation step n and denoising step t, we cache its residual:
ˆrt,b ←r(n)
t,b = x(n)
t,b −x(n)
t,b−1.
(3)
At the next generation step n+1, if the gating mechanism (Section 2.3) determines that block b at
denoising step t can be skipped, we approximate its output by additive reuse:
˜x(n+1)
t,b
= x(n+1)
t,b−1 + ˆrt,b.
(4)
The cached residual is indexed by the pair (t, b), ensuring reuse occurs between matching positions
in the denoising trajectory.
2.2.3
Initialization
No cached residuals exist for the first generation step (n = 0). During this warmup phase (config-
urable to W ≥1 steps), all blocks compute fully, populating the cache for every (t, b) pair.
4

AI INFRA TEAM
X-Cache
TECHNICAL REPORT
2.3
Dual-Metric Gating
Beyond the warmup phase, X-Cache evaluates whether each block can be safely skipped using a
dual-metric similarity test between the current input fingerprint and the cached fingerprint from the
same (t, b) position at the previous generation.
2.3.1
Fingerprint
Each block input x(n)
t,b−1 has shape B×V ×L×C, where V is the number of cameras in a view group
and the token axis L is a flattened (Fg×Hg×Wg) spatio-temporal grid. Comparing the full tensor at
every (t, b) is too expensive, and uniform 1D subsampling along L would cover frames and spatial
locations unevenly. We instead subsample on the 3D grid: given a target budget of K tokens, we
allocate the three axes in proportion to the grid’s aspect ratio (kF : kH : kW ≈Fg : Hg : Wg with
kF kHkW ≈K), take uniformly spaced indices along each axis, and index x with their Cartesian
product, yielding a compact fingerprint ϕ(x) of shape B×V ×(kF kHkW )×C (we use K=32). For
blocks with L≤K we keep the full x; if the grid shape is unavailable at runtime we fall back to 1D
linspace along L. The fingerprint is computed independently per view group: in the multi-camera
setting, the seven cameras form three view groups (front, side, and rear) by shared grid shape, with
cameras within a group stacked along V and sharing one per-group fingerprint.
2.3.2
Auxiliary channels
Because ϕ(x) is sparse and depends only on the block input, we attach two cheap auxiliary signals to
close blind spots:
• Global channel: The per-view-group sequence-mean of the block input, obtained by
averaging along the token axis. It captures bulk latent drift that the sparse spatial sample
may miss.
• Condition channel: The per-chunk action vector consumed by adaLN-Zero is flattened
and appended as an additional fingerprint entry. Its shape is stable (fixed by the action
dimensionality and the frames-per-chunk) and it changes once per generation, so the over-
head is negligible. Including it lets the input-similarity metric react directly to per-step
control maneuvers that would otherwise only propagate through the block-0 cascade. Other
conditioning signals (dynamic-object and lane embeddings injected via cross-attention, text
via cross-attention) have variable padding across chunks and slower change timescales; we
leave them to the cascade and anchor-block mechanisms described in Section 2.5.
All fingerprint entries are flattened before metric computation.
2.3.3
Metric 1: Cosine similarity (global direction)
scos =
ϕ(x(n)
t,b−1) · ϕ(x(n−1)
t,b−1 )
∥ϕ(x(n)
t,b−1)∥· ∥ϕ(x(n−1)
t,b−1 )∥
,
(5)
computed per fingerprint entry (each spatial view group, plus the global and action-condition channels)
and aggregated by taking the minimum across all entries: scos = mink s(k)
cos.
2.3.4
Metric 2: Maximum token deviation (local outlier)
dmax =
max
ϕ(x(n)
t,b−1) −ϕ(x(n−1)
t,b−1 )

mean
ϕ(x(n−1)
t,b−1 )
 + ϵ
,
(6)
computed only over the per-group spatial fingerprints (the global and action-condition channels use
cosine alone) and aggregated by taking the maximum across view groups: dmax = maxg d(g)
max.
2.3.5
Skip decision
Block b at denoising step t is skipped if and only if both metrics pass:
skip(t, b) = (scos ≥τcos(t, b)) ∧(dmax < τdev) ,
(7)
5

AI INFRA TEAM
X-Cache
TECHNICAL REPORT
where τcos(t, b) is an adaptive threshold (Section 2.4) and τdev is a fixed deviation threshold. The
conservative aggregation (min-cosine over all fingerprint entries, max-deviation over spatial view
groups) ensures that anomalous change in any single view group, in the global summary, or in the
action-condition channel triggers recomputation.
2.4
Adaptive Threshold
Rather than a fixed cosine threshold, X-Cache learns a per-position threshold from the block’s own
history. For each (t, b) position, we maintain an exponential moving average of the observed cosine
similarity:
¯st,b ←α · s(n)
cos + (1 −α) · ¯st,b,
(8)
updated each time the test at (t, b) is evaluated (α = 0.3). The adaptive threshold is:
τcos(t, b) = max(τfloor, ¯st,b −m) ,
(9)
where m is a configurable margin and τfloor is a hard quality floor. In our implementation, τfloor = 0.95
and m = 0.02.
Blocks with consistently high cross-chunk similarity accumulate high EMAs and their thresholds
settle just below their typical similarity, maximizing skips. Blocks with volatile similarity remain
conservative. The quality floor provides an absolute safety guarantee independent of history.
2.5
Safety Mechanisms
Condition visibility gap
External conditioning signals are injected inside each DiT block, not into
the block input x. The spatial fingerprint ϕ(x) therefore does not directly reflect condition changes.
The action-condition channel from Section 2.3 lifts the action vector into fingerprint space, closing
the gap for per-step controls. Dynamic-object and lane embeddings have variable padding across
chunks and change on slower timescales, so including them directly in the fingerprint is neither
shape-stable nor cost-effective; together with the text condition they are handled instead by the
following mechanisms:
1. Denoising step 0 protection: At t = 0, the input latent is dominated by high-level noise,
and conditioning signals have maximal relative influence on the block output. Noise is
also resampled at each KV update cycle and new context from the previous chunk’s output
is embedded in the noisy input, so consecutive generations naturally exhibit low cosine
similarity at t = 0. By default, X-Cache forces full computation at t = 0. An optional
relaxation mode applies a strict threshold τ strict
0
= 0.999, which is near-unity: even minor
latent changes drop the similarity below 0.999, keeping step 0 effectively protected. Once
step 0 computes with updated conditions, its output cascades through subsequent steps: each
block’s input is derived from the prior step’s output, so fingerprints diverge from the cached
versions and trigger recomputation where needed.
2. Anchor blocks (Fn): The first Fn blocks are unconditionally computed at all denoising
steps (default Fn = 1). With Fn = 1, block 0 always processes the current conditioning
via adaLN-Zero, and its changed output cascades through subsequent blocks’ fingerprints.
This provides a per-step guarantee independent of step 0 protection. The last Bn blocks can
similarly be designated as tail anchors (default Bn = 0).
3. KV update frame protection: The generation step whose clean latent will be used for the
KV update pass is critical: its KV projections are attended to by all future chunks. During
this generation, X-Cache enters force-compute mode: all blocks compute fully, but the cache
remains attached so that fingerprints and residuals are refreshed for the next generation. This
ensures both (a) precise KV entries and (b) that the subsequent generation starts with fresh
cache data, avoiding a double-heavy-frame penalty.
4. Maximum staleness: A counter tracks consecutive skips per (t, b) position. If it exceeds
threshold M, the block is forced to recompute.
6

AI INFRA TEAM
X-Cache
TECHNICAL REPORT
3
Experiments
3.1
Settings
3.1.1
Hardware
All of our experiments run on the Zhenwu (真武) 810E, an AI accelerator developed by Alibaba
T-Head. In the remainder of this paper we will refer to the device as a Parallel Processing Unit (PPU).
Each PPU integrates 96 GB of HBM2e on-chip memory and natively supports FP16, BF16, and INT8
with hardware acceleration. In the following experiments, all DiT forward passes are executed in
BF16 on a single PPU.
3.1.2
