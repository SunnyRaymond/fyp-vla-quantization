# MoECa: Aligning Feature Reuse with Expert Decomposition in Diffusion Transformers

paper_id: arxiv:2606.15615v2
tier: T3
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Diffusion models, especially Diffusion Transformers (DiTs) [12],
have become a dominant paradigm for visual generation due to their
strong scalability and generation quality. As model sizes continue
to grow, researchers [1, 3, 8, 16, 17] have begun to introduce sparse
Mixture-of-Experts (MoE) [15] into DiT, as illustrated in Fig. 1,
to improve model capacity while keeping the number of active
parameters manageable. More fundamentally, the effect of MoE is
not limited to the capacity gain from sparse activation. By replacing
the original dense transformation with multiple expert branches,
MoE reorganizes the feature evolution process of each token into
the joint progression of multiple expert branch features.
Beyond model capacity, the inference efficiency of diffusion mod-
els is still fundamentally constrained by the iterative denoising pro-
cess. In addition to reducing the number of sampling steps through
arXiv:2606.15615v2  [cs.LG]  4 Aug 2026

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil
Maoliang Li, Haojing Chen, Jiayu Chen, Zihao Zheng, Xinhao Sun, Hailong Zou, and Xiang Chen
(b) Expert Affinity Distribution
(a) Temporal Feature Evolution
DSMoE-S-E16  Pos=0, Class 0
DSMoE-S-E16  Pos=64 , Class 0
DSMoE-L-E16  Pos=0 , Class 100
DSMoE-L-E16  Pos=64 , Class 100
Expert ID
Denoising Progress
0
16
8
0
16
8
0
16
8
0
16
8
0
250
200
150
100
50
1.0
0.8
0.6
0.4
0.2
0.0
Expert Weight
Async Progression & Transition
Sub Feature Similarity
Generation Stage Affinity
Content Position Affinity
Expert ID
0
4
8
Timestep
0
250
125
0
4
8 0
4
8 0
4
8
0
250
125
Patch ID
0
4
8 0
4
8 0
4
8 0
4
8
0
250
125
0
250
125
L1
L2
L3
L4
L5
L6
L7
L8
L1
L2
L3
L4
L5
L6
L7
L8
×
Feature- and Structure-Awared Caching Mechanism
Figure 2: Preliminary Observation: (a) Temporal progression
of MoE feature for example tokens; (b) Expert activation
across Image Patch and Timestep.
improved samplers [18] or distillation [20], feature caching has
become one of the most widely adopted acceleration paradigms for
diffusion inference. The underlying motivation is that features at
adjacent timesteps often exhibit considerable similarity, making it
possible to cache and reuse previously computed features to avoid
redundant computation. Existing studies further show that tempo-
ral variation is heterogeneous across different granularities, such as
layers [9], blocks [21], and tokens [14, 22], and that selective reuse
strategies can therefore achieve favorable speed-quality trade-offs.
In dense DiTs, token-level reuse is a natural design choice because
token heterogeneity is one of the primary sources of temporal
redundancy in visual representations. However, in MoE models, to-
ken features are further decomposed into multiple expert branches.
With token as the atomic unit for selective update, the branch-level
heterogeneity introduced by expert decomposition is ignored.
To this end, the temporal evolution of noisy features over experts
and expert branches becomes the key to understanding why token-
level reuse is suboptimal in DiT-MoE. We first present two prelimi-
nary observations in Fig. 2. From the perspective of cross-timestep
feature evolution, Fig. 2(a) shows that the branch-wise feature evo-
lution within the same token becomes highly heterogeneous, while
each individual branch still exhibits noticeable temporal similarity.
From the perspective of expert specialization, Fig. 2(b)1 shows that
experts exhibit various response affinities over content positions
and denoising stages. Consequently, the information carried by dif-
ferent experts is not identical, which further suggests that different
experts should not be treated as equally sensitive to caching error.
Taken together, the internal representation evolution in DiT-MoE
simultaneously exhibits cross-timestep similarity and cross-expert
/ cross-branch heterogeneity. Therefore, as illustrated in Fig. 1, ex-
isting token-level caching leads to suboptimal reuse decisions in
DiT-MoE. On one hand, recomputing the entire token to avoid
error accumulation forces stable branches to be unnecessarily re-
computed, introducing redundancy. On the other, reusing the entire
token for more acceleration causes rapidly changing branches to
be excessively reused, resulting in larger error.
To address this problem, we propose MoECa, a feature- and
structure-aware caching mechanism for MoE-based DiTs. Instead
1Illustration style and sampling scripts are adapted from DiT-MoE[8]
of treating each token as the atomic caching unit, MoECa pushes
the caching granularity down to the expert-branch level, so that
caching decisions are aligned with the internal computation de-
composition structure of MoE and its feature evolution pattern. On
top of this design, MoECa builds a branch-wise caching and update
framework, introduces an expert-aware adaptive control mecha-
nism, and further coordinates the MoE path and the attention path
to reduce inference overhead while preserving generation quality
as much as possible. Our contributions are summarized as follows:
(1) We analyze the cross-timestep evolution of features in DiT-MoE
from the perspective of expert decomposition and specialization,
and reveal the structural mismatch between caching granularity
and feature representations of MoE. (2) We propose MoECa, a
fine-grained caching framework for DiT-MoE, which enables more
efficient feature reuse through expert-branch-level caching and up-
date. (3) We evaluate MoECa on multiple popular DiT-MoE models,
demonstrating a favorable speed–quality trade-off across model
families and generation settings.
2
Backgrounds
2.1
Diffusion Models and Transformers
Diffusion models are built upon two complementary stochas-
tic processes: a forward noising process and a reverse denoising
process. In the forward process, Gaussian perturbations are progres-
sively injected into a clean sample until it approaches an isotropic
Gaussian distribution. In the reverse process, a learnable network
iteratively removes noise and maps a noisy sample back to the data
manifold. Let 𝑡denote the diffusion timestep and 𝛽𝑡the variance
schedule. The reverse transition can be written as
𝑝𝜃(𝑥𝑡−1 | 𝑥𝑡) = N (𝑥𝑡−1;
1
√𝛼𝑡
(𝑥𝑡−1 −𝛼𝑡
√1 −¯𝛼𝑡
𝜖𝜃(𝑥𝑡,𝑡)), 𝛽𝑡I),
(1)
where 𝛼𝑡= 1 −𝛽𝑡and ¯𝛼𝑡= Î𝑡
𝑖=1 𝛼𝑖. Here, 𝜖𝜃(·) is the denoiser
parameterized by 𝜃, which predicts the noise component from
(𝑥𝑡,𝑡). The denoiser is repeatedly invoked over 𝑇timesteps during
sampling, dominating both generation quality and inference cost.
Recent diffusion systems commonly instantiate 𝜖𝜃with a Trans-
former, yielding Diffusion Transformers (DiT). A DiT stacks 𝐿
blocks, each typically containing self-attention and MLP sublayers.
G = 𝑔1 ◦𝑔2 ◦· · · ◦𝑔𝐿,
𝑔𝑙= F 𝑙
SA ◦F 𝑙
MLP.
(2)
The input at timestep 𝑡is represented as a token sequence x𝑡=
{𝑥𝑖}𝐻×𝑊
𝑖=1
, where each token corresponds to an image (or latent)
patch. For a generic sublayer 𝑓, the block-level update is usually
implemented in residual form, F (x) = x + AdaLN ◦𝑓(x), where
AdaLN modulates normalized activations according to diffusion
timestep (and optional condition signals), enabling the model to
adapt computation across denoising stages.
2.2
Feature Caching for Diffusion Acceleration
Feature caching accelerates DiT models by reusing intermediate
features across nearby timesteps. Let {𝑡,𝑡+ 1, . . . ,𝑡+ 𝑁−1} be a
window of 𝑁adjacent denoising steps. At the first step 𝑡(refresh
step), the model executes normally and writes intermediate features
into cache memory. Denoting the cache at layer index 𝑙as C[𝑙],

MoECa: Aligning Feature Reuse with Expert Decomposition in Diffusion Transformers
MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil
(b) Routing Distance Comparison
DSMoE-S-E48 Token128 Output Distance
DSMoE-S-E48 Output Distance
-0.1
0.0
0.1
0.2
0.3
0.4
0.5
0.6
-0.1
0.0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0
20
40
60
80
100
0
20
40
60
80
100
(a) Feature Distance Comparison
0
20
40
60
80
100
0
20
40
60
80
100
0.00
0.01
0.02
0.03
0.04
0.05
0.06
0.00
0.01
0.02
0.03
0.04
0.05
DSMoE-S-E16 MoE Routing L1 Per Layer Distance
DSMoE-S-E48 MoE Routing L1 Per Layer Distance
Sampling Progress (%)
Sampling Progress (%)
Figure 3: Temporal Feature Evolution Analysis: (a) Distance
between adjacent timesteps of token- and branch-level fea-
tures. The upper figure sets distance to 0 for branch transition.
(b) Distance between adjacent timesteps of routing output.
cache is assigned as:
C[𝑙] := F𝑙

x𝑙−1
𝑡

,
𝑙= 1, . . . , 𝐿,
(3)
For each following step 𝑡+ 𝑖with 𝑖∈{1, . . . , 𝑁−1}, naive caching
directly reuses the stored intermediate feature:
F𝑙

x𝑙−1
𝑡+𝑖

:= C[𝑙],
𝑙= 1, . . . , 𝐿.
(4)
In practice, one refresh step is amortized over several reuse steps to
reduce attention/MLP computation, but overly long reuse windows
can cause cache staleness and quality degradation.
From the perspective of feature granularity, caching methods
can be divided into step-, block-, layer-, and token-level, by hi-
erarchy of re-use decision in DiT architecture. Finer reuse better
preserves quality but requires higher control overhead. ToCa is a
representative token-level method that selectively recomputes only
high-variation tokens, which effectively exploits feature hetero-
geneity across various tokens.
2.3
MoE and DiT-MoE
Mixture-of-Experts (MoE) replaces the dense FFN in Transformer
structure with multiple routed experts and a router. For token input
𝑢𝑡, the standard MoE FFN output (without shared experts) is
ℎ𝑡= 𝑢𝑡+
∑︁𝑁𝑟
𝑖=1 𝑔𝑖,𝑡FFN(𝑟)
𝑖
(𝑢𝑡),
(5)
where 𝑁𝑟is the number of routed experts. Since such structure
is inherently compatible with the DiT paradigm, DSMoE [10], a
practical DiT-MoE training recipe with sparse routed and shared
experts, and DiT-MoE [8] introduce MoE to DiT models without
significant modification to model structure. Following DeepSeek
MoE [7], shared experts are further introduced in addition to routed
experts, so the output of original FFN becomes:
ℎ𝑡= 𝑢𝑡+
∑︁𝑁𝑠
𝑖=1 FFN(𝑠)
𝑖
(𝑢𝑡) +
∑︁𝑁𝑟
𝑖=1 𝑔𝑖,𝑡FFN(𝑟)
𝑖
(𝑢𝑡),
(6)
where 𝑁𝑠and 𝑁𝑟are the numbers of shared exp

## Method

ter characterized at the expert-branch level than at the whole-token
level. Based on this observation, we propose MoECa, a fine-grained
caching framework that performs branch-level feature reuse across
timesteps. MoECa further introduces expert-aware adaptive control
and synchronized cache updates across MoE and attention paths
to maintain stable intermediate states. Experiments on multiple
DiT-MoE models show a favorable speed–quality trade-off, with
up to 2.93× speedups while preserving generation quality.
CCS Concepts
• Information systems →Multimedia content creation.
Keywords
Diffusion, Feature Cache, Acceleration, Mixture-of-Experts
ACM Reference Format:
Maoliang Li, Haojing Chen, Jiayu Chen, Zihao Zheng, Xinhao Sun, Hailong
Zou, and Xiang Chen. 2026. MoECa: Aligning Feature Reuse with Expert
Decomposition in Diffusion Transformers. In Proceedings of the 34th
ACM International Conference on Multimedia (MM ’26), November 10–14,
∗Both authors contributed equally to this research.
†Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
MM ’26, Rio de Janeiro, Brazil
© 2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2213-4/2026/11
https://doi.org/10.1145/3767308.3835789
𝑿𝒕-#
Self-Attn
Cond
AdaLN
AdaLN
MLP
AdaLN
MoE Layer
DiT Block
𝑿𝒕
𝑿𝒕-#
Self-Attn
Router
Output
ES
E1
E2
E3
𝑿𝒕$%
E4
ToCa-like
Router
Output
ES
E1
E2
E3
E4
Cache
Compute
MoECa
Token 
Heterogeneity
Expert 
Heterogeneity
Figure 1: Conceptualization of: (a) expert heterogeneity in
DiT-MoE; and (2) misalignment between token-level cache
and MoE models.
2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 9 pages. https://doi.
org/10.1145/3767308.3835789
1
