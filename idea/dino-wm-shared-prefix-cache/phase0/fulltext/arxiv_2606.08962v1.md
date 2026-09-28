# C$^3$ache: Accelerating World Action Models with Cross Inference Chunk Cache

paper_id: arxiv:2606.08962v1
tier: H
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Vision-Language-Action (VLA) models have become the dominant recipe for generalist robot poli-
cies. VLAs such as RT-2, OpenVLA, and π0 inherit the semantic priors of internet-scale image-text
data and carry them into control [1, 2, 3, 4]. However, the behaviors VLAs produce remain bounded
by the demonstrations they were trained on [5]. World Action Models (WAMs) come at this from
the other side, pairing action prediction with a video-modeling objective that learns to predict how
the scene itself will evolve [6, 7, 8, 5]. This changes what the model can learn from: video is a
dense supervisory signal, and unlike action-labeled trajectories it is available in enormous quantity.
The resulting representations are shaped by how objects move and interact rather than by a fixed set
of demonstrated trajectories, giving WAMs markedly stronger generalization to novel motions and
unseen environments than VLAs of comparable scale [5].
Until recently, turning this objective into a policy means following the imagine-then-execute
paradigm: at test time, the model synthesizes future frames through a video-diffusion process and
reads off actions from the imagined rollout. Generating frames this way is expensive. Each rollout
is a long sequence of denoising steps through a diffusion transformer [9], as shown in the top part
of Figure 1a, and it fits poorly with the latency budget of closed-loop control [10, 11]. To complete
a task, a WAM runs over multiple inference chunks, and each chunk requires this expensive denois-
ing process. This is a sequence of forward passes that each refine a noisy estimate toward a clean
one [12, 3].
Most acceleration work targets redundancy in denoising, building on one premise: a diffusion

## Method

† Corresponding authors.
arXiv:2606.08962v1  [cs.LG]  8 Jun 2026

full
cached
h0
DiT ×L
run all L blocks
hL
˜R = hL −h0
h0
DiT ×L
skipped
hL = h0 + ˜R
bypass: skip all L blocks
residual
cache
˜R
store
reuse ˜R
inference schedule (each square = one inference chunk)
· · ·
refresh every τ chunks: 1 full + (τ−1) cached
(a) C3ache method overview.
LIBERO
RoboTwin
0
1
2
3
Inference speed up
1.0 x
1.0 x
2.5 x
1.8 x
Fast-WAM
C3ache
(b) Inference speed-up.
Figure 1: (a) Overview of C3ache. In full-computation chunks, our method runs the full DiT blocks
and computes the residual. In cached chunks, it reuses the pre-computed residual to skip all DiT
blocks. The residual is refreshed according to the inference schedule. (b) Inference speed compari-
son between C3ache and Fast-WAM (no cache) on two benchmarks: LIBERO and RoboTwin.
repeated. They differ mainly in the granularity at which they cache. To accelerate visual gener-
ation diffusion model, DeepCache and Block Caching reuse whole blocks of the network across
steps [13, 14]; Learning-to-Cache instead treats each layer as the unit of reuse and learns which
layers can be skipped [15]; ToCa pushes the granularity down to individual tokens, caching only
those whose features are stable [16]; and methods such as PAB adapt what is reused and when [17].
Diverse as these methods are, they share a common scope: they exploit redundancy within a single
chunk’s denoising trajectory (across its steps, layers, or tokens) and treat each chunk as an indepen-
dent unit of computation. We therefore ask two questions: (1) Does any redundancy exist in WAM
across chunks? (2) Can we exploit this redundancy to accelerate WAM’s inference?
We find that the answer to the first question is yes: our empirical analysis reveals that this scope
leaves a substantial source of redundancy untouched. When a robot executes a smooth, temporally
coherent behavior, the observations at successive inference chunks change only gradually, and the
action velocities predicted from them differ only slightly. We observe that this surface-level similar-
ity propagates inward: at a fixed denoising step, the intermediate residuals computed by the action
expert for one chunk are strongly correlated with those computed for the chunk before it. Caching
strategies that operate one chunk at a time cannot capture this, because the quantities they would
reuse never persist across the chunk.
More importantly, we find that the answer to the second question is also yes. We introduce C3ache
(Cross Inference Chunk Cache), a training free method that exploits precisely this cross-chunk re-
dundancy. C3ache caches the residuals produced at each denoising step and reuses them at the
corresponding step of subsequent chunks as shown in Figure 1a, eliminating a large fraction of the
denoising computation while leaving the model weights and training procedure untouched.
In summary, our contributions are as follows:
• We empirically characterize cross-chunk redundancy in a world action model, showing that the
residuals computed at the same denoising step are strongly correlated across consecutive action
chunks—a form of redundancy that current chunk-local caching methods do not exploit.
• We propose C3ache, a training-free acceleration method that caches and reuses these residuals
across chunks, and which composes with existing within-chunk step-wise caching.
• We show on LIBERO and RoboTwin [18, 19], with a Fast-WAM[20] backbone, that C3ache
achieves up to a 2.5× wall clock inference speedup with negligible degradation of task success
rate as shown in Figure 1b.
2

2
