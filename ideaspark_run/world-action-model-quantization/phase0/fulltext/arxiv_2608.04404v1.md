# Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models

paper_id: arxiv:2608.04404v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

General-purpose robot intelligence requires not only recognizing the current environment but also anticipating how the world will evolve after interaction.
While recent Vision-Language-Action (VLA) models
(
Black et al. 2024
;
Intelligence et al. 2025
;
Kim et al. 2024
)
have achieved impressive progress in robotic manipulation, most existing approaches predict actions primarily from current visual observations and language instructions, without explicitly modeling future scene dynamics.
World Action Models (WAMs)
(
Tian et al. 2025
;
Ye et al. 2026
)
address this limitation by augmenting action prediction with future visual modeling.
By learning how objects move and scenes evolve under robot interaction, WAMs provide policies with temporal representations beyond the current observation.
However, an important question remains unresolved:
Are future predictions merely useful as a training signal, or do future representations provide essential information during inference?
Existing WAMs provide two different answers.
Joint-WAMs
(
Li et al. 2026b
;
Bi et al. 2025
)
couple future video generation and action prediction through shared denoising, allowing the action branch to access evolving future representations during inference.
However, repeatedly running the video branch and performing dense video–action interaction introduces substantial computational overhead.
Fast-WAM
(
Yuan et al. 2026
)
explores the opposite direction by using future modeling only during training and removing future representations during inference.
Although this design significantly improves efficiency and achieves competitive in-distribution performance
(
Liu et al. 2023
;
Chen et al. 2025
)
, it raises a fundamental limitation: without inference-time future conditioning, the policy may lose temporal information required to handle unseen environments.
Figure 1:
Inference-time future conditioning is critical for robust WAMs.
Compared with Joint-WAM (a controlled implementation representing Joint-WAMs), Fast-WAM improves efficiency by removing future conditioning at inference, yet suffers a marked performance drop under distribution shift.
Motivated by this finding, Faster-WAM efficiently preserves inference-time future conditioning, achieving strong OOD robustness at lower latency.
To investigate this question, we evaluate WAMs under distribution shift
(
Fei et al. 2025
)
and observe that removing future representations at inference substantially harms robustness.
Specifically, as shown in Fig.
1
, Fast-WAM exhibits a significant performance degradation in out-of-distribution (OOD) settings compared with Joint-WAM, suggesting that future representations are not merely an auxiliary training objective but an important source of generalizable temporal knowledge.
This finding leads to a new design principle for WAMs:
Future representations should be preserved at inference, but their interaction with action prediction must become selective and efficient.
Based on this principle, we propose Faster-WAM, an efficient future-conditioning World Action Model that maintains inference-time future conditioning while redesigning video–action interaction.
Instead of repeatedly executing the video branch, Faster-WAM computes future representations once and reuses them through cached intermediate representations during action denoising.
To achieve efficient future conditioning, Faster-WAM introduces two complementary mechanisms.
First, SparseMoT reduces unnecessary computation by concentrating video–action interaction at a compact subset of stages while performing lightweight action-only refinement between successive interactions.
Second, Interval KV-Fusion aggregates future representations from multiple video depths within each interaction interval, providing richer temporal information without increasing attention complexity.
Extensive experiments demonstrate that Faster-WAM achieves a superior balance between robustness and efficiency.
On the OOD LIBERO-Plus benchmark
(
Fei et al. 2025
)
, Faster-WAM achieves a 73.57% success rate compared with 49.14% for Fast-WAM, while achieving a
2.21
×
2.21\times
inference speedup over Joint-WAM.
It also achieves state-of-the-art performance on LIBERO and RoboTwin 2.0, while demonstrating strong robustness in real-world manipulation.
Our contributions are summarized as follows:
•
We identify inference-time future conditioning as an important factor for WAM generalization, showing that future representations provide robustness beyond their role as a training objective.
•
We propose Faster-WAM, a future-conditioning framework that preserves inference-time temporal representations through sparse and efficient video–action interaction.
•
We introduce SparseMoT and Interval KV-Fusion, enabling selective access to multi-level future representations without the computational cost of dense interaction.
•
Extensive experiments demonstrate state-of-the-art performance on in- and out-of-distribution benchmarks with improved inference efficiency.

## Method

Figure 2:
Overview of Faster-WAM. (a) Overall framework of Faster-WAM, featuring SparseMoT for selective video–action interaction and Interval KV-Fusion for aggregating multi-level future representations. (b) Inference comparison between Joint-WAM and Faster-WAM, contrasting iterative dense coupling with one-pass future-context caching and sparse interaction.
Problem Formulation
We consider language-conditioned visuomotor control from demonstrations.
At control step
t
t
, the policy observes an image
o
t
o_{t}
, a language instruction
l
l
, and a proprioceptive state
s
t
s_{t}
, and predicts an action chunk
A
t
=
a
t
+
1
:
t
+
H
A_{t}=a_{t+1:t+H}
of horizon
H
H
.
A WAM parameterized by
θ
\theta
defines its action policy through an internal visual interface:
π
θ
=
p
θ
​
(
A
t
∣
s
t
,
l
,
ℛ
t
v
)
,
\pi_{\theta}=p_{\theta}\!\left(A_{t}\mid s_{t},l,\mathcal{R}_{t}^{v}\right),
(1)
where
ℛ
t
v
\mathcal{R}_{t}^{v}
is the internal visual representation derived from the current observation
o
t
o_{t}
by the video branch.
Let
ℰ
v
\mathcal{E}_{v}
denote the visual encoder,
z
t
0
=
ℰ
v
​
(
o
t
)
z_{t}^{0}=\mathcal{E}_{v}(o_{t})
the current-observation latent, and
Z
t
Z_{t}
the future-video latents.
Joint-WAMs
(
Ye et al. 2026
;
Bi et al. 2025
)
jointly update future-video and action states, yielding an evolving video representation at step
k
k
:
ℛ
t
,
k
v
,
joint
=
G
v
​
(
z
t
0
,
Z
t
(
k
)
,
l
)
.
\mathcal{R}_{t,k}^{v,\mathrm{joint}}=G_{v}\!\left(z_{t}^{0},Z_{t}^{(k)},l\right).
(2)
Here,
G
v
G_{v}
denotes the video-side representation map.
As
Z
t
(
k
)
Z_{t}^{(k)}
evolves, the dense Joint-WAM used for comparison recomputes this representation and repeats cross-branch interaction at every step.
Fast-WAM
(
Yuan et al. 2026
)
instead removes future slots at inference:
ℛ
t
v
,
fast
=
G
v
​
(
z
t
0
,
l
)
.
\mathcal{R}_{t}^{v,\mathrm{fast}}=G_{v}\!\left(z_{t}^{0},l\right).
(3)
The resulting interface can be reused throughout action generation, but is constructed without explicit future temporal slots.
Faster-WAM instead constructs a fixed future-aware interface
ℛ
¯
t
v
\overline{\mathcal{R}}_{t}^{v}
in one video-expert pass and reuses it through sparse interaction during action generation (Fig.
2
(b)).
Faster-WAM
Overview.
As illustrated in Fig.
2
(a), Faster-WAM couples a video expert initialized from a pretrained video generator
(
Wan et al. 2025
)
with an action expert through a Mixture-of-Transformers (MoT) architecture
(
Liang et al. 2024
)
.
The architecture comprises
L
L
aligned stages, each pairing the corresponding video and action layers.
Language and proprioception are supplied as shared conditioning signals to both the video and action experts.
A single video pass produces a layer-wise attention key/value (K/V) hierarchy.
Interval KV-Fusion turns it into compact action-facing summaries for SparseMoT to expose at selected stages, leaving the remaining stages for action-only refinement.
One-pass Future Conditioning.
Prior work
(
Pai et al. 2025
;
Ma et al. 2026
)
shows that control-relevant states can be extracted from high-noise video latents without completing denoising.
We formulate both branches with flow matching, taking
τ
=
0
\tau=0
as clean data and
τ
=
1
\tau=1
as Gaussian noise, and denote the interpolated future latent at video flow time
τ
v
\tau_{v}
by
Z
t
,
τ
v
Z_{t,\tau_{v}}
(Eq.
9
).
At the noisy endpoint, the video expert processes Gaussian future slots together with the clean current-frame anchor
z
t
0
z_{t}^{0}
and language.
Estimating the flow-matching direction for these future-video latents requires reasoning about plausible scene dynamics, so the resulting hidden states can encode future-aware cues without reconstructing a rollout.
To expose these cues to the action expert, we retain from each video attention block at flow time
τ
v
\tau_{v}
the key/value projections
K
t
,
τ
v
,
j
v
K_{t,\tau_{v},j}^{v}
and
V
t
,
τ
v
,
j
v
V_{t,\tau_{v},j}^{v}
, whose token layout and shape are shared across depth.
Collecting them gives
𝒞
t
,
τ
v
v
=
{
(
K
t
,
τ
v
,
j
v
,
V
t
,
τ
v
,
j
v
)
}
j
=
1
L
.
\mathcal{C}_{t,\tau_{v}}^{v}=\left\{\left(K_{t,\tau_{v},j}^{v},V_{t,\tau_{v},j}^{v}\right)\right\}_{j=1}^{L}.
(4)
This raw K/V hierarchy is the source from which the action-facing interface is constructed.
To make it reusable across action flow steps, we use asymmetric attention within the video stream: future slots may attend to the clean anchor and one another, while the anchor cannot attend to them.
Across experts, action queries may read video features, whereas video queries cannot read action tokens.
Consequently, the video hierarchy is independent of the evolving action trajectory and can be constructed before action integration.
At inference, setting
Z
t
,
1
Z_{t,1}
to Gaussian noise
ϵ
t
v
\epsilon_{t}^{v}
yields the fixed raw hierarchy
𝒞
t
,
1
v
\mathcal{C}_{t,1}^{v}
in one video pass.
SparseMoT.
In a conventional dense MoT, video and action features interact at every aligned stage.
Even with a precomputed video hierarchy, this cross-branch attention repeats across all
L
L
stages at every action flow step.
SparseMoT reduces this repeated cost by restricting video access to the interaction set
𝒥
=
{
j
1
,
…
,
j
M
}
⊆
{
1
,
…
,
L
}
,
1
≤
j
1
<
⋯
<
j
M
≤
L
.
\begin{array}[]{c}\mathcal{J}=\{j_{1},\ldots,j_{M}\}\subseteq\{1,\ldots,L\},\\
1\leq j_{1}<\cdots<j_{M}\leq L.\end{array}
(5)
We select
𝒥
\mathcal{J}
at a fixed layer stride and reuse it at every action flow step;
M
M
therefore counts the video-reading stages per action evaluation, with
M
=
L
M=L
recovering dense MoT.
For notational simplicity, we consider a fixed control step
t
t
, video flow time
τ
v
\tau_{v}
, and action flow time
τ
a
\tau_{a}
, and omit these indices below.
Let
Q
j
a
Q_{j}^{a}
,
K
j
a
K_{j}^{a}
, and
V
j
a
V_{j}^{a}
denote the action-token query, key, and value projections at stage
j
j
.
At
j
m
∈
𝒥
j_{m}\in\mathcal{J}
,
(
K
^
j
m
v
,
V
^
j
m
v
)
(\widehat{K}_{j_{m}}^{v},\widehat{V}_{j_{m}}^{v})
is the fused video pair summarizing its preceding depth interval (Eq.
7
).
Its head dimensions match those of the action K/V projections, enabling the update
X
~
j
m
a
=
Attn
⁡
(
Q
j
m
a
,
[
K
^
j
m
v
;
K
j
m
a
]
,
[
V
^
j
m
v
;
V
j
m
a
]
)
,
\widetilde{X}_{j_{m}}^{a}=\mathrm{Attn}\!\left(Q_{j_{m}}^{a},[\widehat{K}_{j_{m}}^{v};K_{j_{m}}^{a}],[\widehat{V}_{j_{m}}^{v};V_{j_{m}}^{a}]\right),
(6)
where
Attn
\mathrm{Attn}
is standard attention,
[
;
]
[\,;\,]
concatenates tokens, and
X
~
j
m
a
\widetilde{X}_{j_{m}}^{a}
is the action output combining the future summary with the current action state.
For
j
∉
𝒥
j\notin\mathcal{J}
, action-only self-attention and residual/feed-forward updates carry previously injected future information through the action state without reading video K/V again.
Thus all
L
L
action stages remain active, and only cross-branch communication is sparse.
Interval KV-Fusion.
SparseMoT reduces how often the action pathway reads video context, while the video expert continues to transform its representation at the intervening depths.
If interaction stage
j
m
j_{m}
consumed only its own K/V pair, intermediate representations would not be directly exposed to the action pathway.
Alternatively, concatenating them would lengthen the action-attention context.
To address this, we propose Interval KV-Fusion, which aggregates the video K/V pairs accumulated within each interaction interval.
Specifically, we assign each selected stage
j
m
j_{m}
exactly one preceding interval
ℐ
m
=
{
j
m
−
1
+
1
,
…
,
j
m
}
\mathcal{I}_{m}=\{j_{m-1}+1,\ldots,j_{m}\}
.
For each assigned interval
ℐ
m
\mathcal{I}_{m}
, we introduce softmax-normalized fusion weights
W
m
,
j
fuse
W^{\mathrm{fuse}}_{m,j}
to aggregate the video representations across its stages without lengthening action attention.
Because the stage-wise K/V pairs share a common token layout and dimensionality, the fused pair for
j
m
j_{m}
is
(
K
^
j
m
v
,
V
^
j
m
v
)
=
∑
j
∈
ℐ
m
W
m
,
j
fuse
​
(
K
j
v
,
V
j
v
)
.
\left(\widehat{K}_{j_{m}}^{v},\widehat{V}_{j_{m}}^{v}\right)=\sum_{j\in\mathcal{I}_{m}}W^{\mathrm{fuse}}_{m,j}\left(K_{j}^{v},V_{j}^{v}\right).
(7)
This weighted sum combines the K/V information from all stages in
ℐ
m
\mathcal{I}_{m}
into a single pair for
j
m
j_{m}
, while preserving key–value correspondence and the sequence length of one video stage.
Restoring
t
t
and
τ
v
\tau_{v}
, the
M
M
fused pairs form the action-facing interface
𝒞
^
t
,
τ
v
v
=
{
(
K
^
t
,
τ
v
,
j
m
v
,
V
^
t
,
τ
v
,
j
m
v
)
}
m
=
1
M
.
\widehat{\mathcal{C}}_{t,\tau_{v}}^{v}=\left\{\left(\widehat{K}_{t,\tau_{v},j_{m}}^{v},\widehat{V}_{t,\tau_{v},j_{m}}^{v}\right)\right\}_{m=1}^{M}.
(8)
At inference, setting
τ
v
=
1
\tau_{v}=1
produces the fixed interface
𝒞
^
t
,
1
v
\widehat{\mathcal{C}}_{t,1}^{v}
, which is supplied to the action expert as reusable future-aware context throughout action integration.
Overall, Interval KV-Fusion preserves multi-depth future context under SparseMoT without increasing attention complexity.
Joint Training.
Faster-WAM jointly learns the video and action flow fields through flow matching
(
Lipman et al. 2022
)
.
For each training example, the video and action flow times
τ
v
\tau_{v}
and
τ
a
\tau_{a}
are sampled independently, together with Gaussian noise samples
ϵ
t
v
\epsilon_{t}^{v}
and
ϵ
t
a
\epsilon_{t}^{a}
:
Z
t
,
τ
v
=
(
1
−
τ
v
)
​
Z
t
+
τ
v
​
ϵ
t
v
,
A
t
,
τ
a
=
(
1
−
τ
a
)
​
A
t
+
τ
a
​
ϵ
t
a
.
\begin{array}[]{rcl}Z_{t,\tau_{v}}&=&(1-\tau_{v})Z_{t}+\tau_{v}\epsilon_{t}^{v},\\
A_{t,\tau_{a}}&=&(1-\tau_{a})A_{t}+\tau_{a}\epsilon_{t}^{a}.\end{array}
(9)
Given
(
Z
t
,
τ
v
,
τ
v
,
z
t
0
,
l
)
(Z_{t,\tau_{v}},\tau_{v},z_{t}^{0},l)
, the video expert predicts the video flow
u
^
t
v
\widehat{u}_{t}^{v}
for the future slots while producing the raw K/V hierarchy
𝒞
t
,
τ
v
v
\mathcal{C}_{t,\tau_{v}}^{v}
.
Interval KV-Fusion converts this hierarchy into
𝒞
^
t
,
τ
v
v
\widehat{\mathcal{C}}_{t,\tau_{v}}^{v}
, which conditions the action-flow predictor
F
a
F_{a}
through SparseMoT at the selected stages:
u
^
t
a
=
F
a
(
A
t
,
τ
a
,
τ
a
∣
s
t
,
l
,
𝒞
^
t
,
τ
v
v
)
.
\widehat{u}_{t}^{a}=F_{a}\!\left(A_{t,\tau_{a}},\tau_{a}\mid s_{t},l,\widehat{\mathcal{C}}_{t,\tau_{v}}^{v}\right).
(10)
With targets
u
t
v
=
ϵ
t
v
−
Z
t
u_{t}^{v}=\epsilon_{t}^{v}-Z_{t}
and
u
t
a
=
ϵ
t
a
−
A
t
u_{t}^{a}=\epsilon_{t}^{a}-A_{t}
, the joint objective is
ℒ
=
λ
v
​
E
​
[
W
v
flow
​
(
τ
v
)
​
‖
u
^
t
v
−
u
t
v
‖
2
2
]
+
λ
a
​
E
​
[
W
a
flow
​
(
τ
a
)
​
‖
u
^
t
a
−
u
t
a
‖
2
2
]
.
\begin{array}[]{rcl}\mathcal{L}&=&\lambda_{v}\mathrm{E}\left[W_{v}^{\mathrm{flow}}(\tau_{v})\left\|\widehat{u}_{t}^{v}-u_{t}^{v}\right\|_{2}^{2}\right]\\[2.0pt]
&&+\lambda_{a}\mathrm{E}\left[W_{a}^{\mathrm{flow}}(\tau_{a})\left\|\widehat{u}_{t}^{a}-u_{t}^{a}\right\|_{2}^{2}\right].\end{array}
(11)
The video and action flow-matching MSE losses are weighted separately by the flow-time-dependent factors
W
v
flow
​
(
τ
v
)
W_{v}^{\mathrm{flow}}(\tau_{v})
and
W
a
flow
​
(
τ
a
)
W_{a}^{\mathrm{flow}}(\tau_{a})
, respectively, while
λ
v
\lambda_{v}
and
λ
a
\lambda_{a}
balance the overall contributions of the two branches.
The video term supervises future-video dynamics, whereas the action term encourages the fused hierarchy to retain control-relevant information.
Independently sampling
τ
v
\tau_{v}
and
τ
a
\tau_{a}
exposes the action pathway to diverse combinations of video and action noise levels.
Efficient Inference.
At deployment, Faster-WAM initializes the future slots and action state from Gaussian noise.
It evaluates the video expert once at
τ
v
=
1
\tau_{v}=1
and applies Interval KV-Fusion to the resulting raw hierarchy, yielding the cached action-facing interface
𝒞
^
t
,
1
v
\widehat{\mathcal{C}}_{t,1}^{v}
.
At each action-flow step, the action expert reuses this cache through SparseMoT, without updating or decoding the future-video latents.
As illustrated in Fig.
2
(b), this replaces
N
N
dense joint video–action evaluations with one video-side pass followed by
N
N
sparse, cache-conditioned action evaluations.
\captionbox
Success rates (%) on LIBERO. P.T. denotes embodied pretraining. The best and second-best average results are shown in bold and underlined, respectively.[0.48][c]
Method
P.T.
Spa.
Obj.
Goa.
Lon.
Avg.
π
0.5
\pi_{0.5}
(
Intelligence et al. 2025
)
✓
98.8
98.2
98.0
92.4
96.9
LingBot-VA
(
Li et al. 2026b
)
✓
98.5
99.6
97.2
98.5
98.5
Motus
(
Bi et al. 2025
)
✓
96.8
99.8
96.6
97.6
97.7
Fast-WAM
(
Yuan et al. 2026
)
✗
98.2
100.0
97.0
95.2
97.6
Joint-WAM
✗
99.6
99.4
98.2
96.8
98.5
Faster-WAM (Ours)
✗
99.6
99.8
98.2
98.2
99.0
\captionbox
Success rates (%) on RoboTwin 2.0. P.T. denotes embodied pretraining. The best and second-best average results are shown in bold and underlined, respectively.[0.48][c]
Method
P.T.
Clean
Rand.
Avg.
π
0.5
\pi_{0.5}
(
Intelligence et al. 2025
)
✓
82.7
76.8
79.8
Motus
(
Bi et al. 2025
)
✓
88.7
87.0
87.9
LingBot-VA
(
Li et al. 2026b
)
✓
92.9
91.5
92.2
Fast-WAM
(
Yuan et al. 2026
)
✗
91.9
91.8
91.9
Joint-WAM
✗
90.8
90.3
90.6
Faster-WAM (Ours)
✗
92.8
92.3
92.6
\captionbox
Success rates (%) on LIBERO-Plus across seven distribution shifts. P.T. denotes embodied pretraining. The best and second-best average results are shown in bold and underlined, respectively.[][c]
Method
P.T.
Camera
Robot
Lang.
Light
Backg.
Noise
Layout
Avg.
UniVLA
(
Bu et al. 2025b
)
✓
1.8
46.2
69.6
69.0
81.0
21.2
31.9
42.9
OpenVLA-OFT
(
Kim et al. 2025
)
✓
56.4
31.9
79.5
88.7
93.3
75.8
74.2
69.6
π
0
\pi_{0}
(
Black et al. 2024
)
✓
13.8
6.0
58.8
85.0
81.4
79.0
68.9
53.6
π
0
\pi_{0}
-Fast
(
Pertsch et al. 2025
)
✓
65.1
21.6
61.0
73.2
73.2
74.4
68.8
61.6
WorldVLA
(
Cen et al. 2025
)
✓
0.1
27.9
41.6
43.7
17.1
10.9
38.0
25.0
Fast-WAM
(
Yuan et al. 2026
)
✗
18.8
45.7
70.1
83.2
45.7
29.8
62.7
49.1
Joint-WAM
✗
37.5
64.5
93.0
95.0
55.9
47.3
79.6
66.3
Faster-WAM (Ours)
✗
53.8
71.6
94.7
96.3
61.3
63.6
79.1
73.6
