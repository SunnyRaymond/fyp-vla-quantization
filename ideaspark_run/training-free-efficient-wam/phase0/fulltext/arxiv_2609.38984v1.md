# Sparse-WAM: Accelerating World Action Models via Action-Guided Sparse Imagination

paper_id: arxiv:2609.38984v1
tier: U
source_used: html_arxiv
warning: none

## Intro

World action models (WAMs) have emerged as a promising paradigm for robotic control. Recent WAMs such as DreamZero
(
Ye et al., 2026b
)
and Cosmos 3
(
NVIDIA, 2026
)
jointly denoise future visual states and actions, leveraging spatiotemporal priors to improve generalization and robustness
(
Zhang et al., 2026
)
.
However, explicitly generating high-resolution future frames introduces a large number of visual tokens into each denoising step. Since the cost of each denoising step grows significantly
with the number of processed tokens
(
Anagnostidis et al., 2025
)
,
repeatedly processing these future-frame tokens increases
inference latency and limits control frequency
(
Guo et al., 2024
)
.
Figure 1:
Representative architectures of VLA and WAM. In WAM, noisy imagination tokens constitute the majority of the visual-action sequence and evolve throughout denoising.
To reduce this repeated computation, existing methods either
remove imagination at inference or compress future-frame tokens.
The former uses future imagination during training but omits
explicit future prediction during action
generation
(
Yuan et al., 2026
;
Li et al., 2026b
;
Zhao et al., 2026a
)
.
The latter retains future prediction with fewer or lower-fidelity
visual tokens
(
Chen et al., 2026
;
Li et al., 2026a
)
,
but still devotes computation to regions with limited
relevance to actions.
Token pruning offers a fine-grained means of reducing the cost of
processing dense imagination tokens by selectively retaining
a subset for computation.
This paradigm has been explored in VLAs to retain useful information
from observed inputs
(
Xu et al., 2025
;
Wang et al., 2026a
)
and in video generation to preserve visual
quality
(
Zou et al., 2025
;
Feng et al., 2026
)
.
In WAMs, however, future-frame tokens are initialized from noise
and jointly denoised with action tokens, making action-relevant
regions difficult to determine in advance.
To understand how action-relevant regions evolve during denoising,
we examine attention from action tokens to future-frame tokens
(see
subsection 3.2
).
We observe that, despite continued changes in future representations,
the regions attended to by action tokens exhibit substantial
spatial overlap between consecutive denoising steps.
This consistency offers an opportunity to reuse token selections
across steps, but does not imply that the relevant regions remain
unchanged throughout denoising.
These observations motivate using action-to-future attention to select future regions for sparse computation and reusing the selections across denoising steps. However, attention scoring and token reorganization introduce additional overhead that can offset the computational savings. The core challenge is therefore to identify useful future regions while keeping the cost of selection and sparse execution low.
To address this challenge, we introduce
Sparse-WAM
, a training-free framework for action-guided sparse imagination in WAMs. It uses action-to-future attention to retain frame-specific regions together with shared spatial context. An efficient execution engine combines lightweight scoring with cross-step selection reuse to reduce the cost of joint visual–action inference.
We summarize our contributions as follows:
•
We reveal substantial spatial overlap in action-to-future
attention between consecutive denoising steps, despite continued
changes in future representations.
This finding motivates reusing action-guided token selections
during joint denoising.
•
We propose
Sparse-WAM
, which uses temporally aligned
action-to-future attention to select frame-specific regions
together with shared spatial context.
It concentrates Transformer computation on selected future
tokens while retaining all observation and action tokens.
•
We develop
Pilot
, an efficient engine for online
token selection and sparse execution.
It combines lightweight attention profiling with cross-step
reuse of selected positions and packing metadata,
reducing the overhead of action-guided sparse inference.
•
We evaluate Sparse-WAM on three WAMs across
LIBERO
(
Liu et al., 2023
)
,
RoboLab-120
(
Yang et al., 2026
)
,
LIBERO-Plus
(
Fei et al., 2025
)
,
and real-world robotic tasks.
On LIBERO, Sparse-WAM achieves a
1.98
×
1.98\times
speedup
over dense eager inference with a
0.30
0.30
percentage-point
decrease in average task success.

## Method

As illustrated in
Figure 3
, Sparse-WAM uses
action-to-future attention to allocate computation within
imagined futures.
In
subsection 4.1
, we combine frame-specific
core tokens with shared spatial anchors to retain changing
attention hotspots and persistent context.
In
subsection 4.2
, we introduce
Pilot
, an engine that combines lightweight attention
profiling with cross-step reuse to execute these selections
efficiently.
Figure 3:
Overview of Sparse-WAM.
Future-frame token positions are selected using action-to-future
attention and reused during sparse denoising.
4.1
Online Action-Guided Token Selection
As discussed in
Figure 2
,
the token selection needs to follow the hotspots
of each future frame while retaining contextual regions
shared across frames.
We address these complementary needs with
K
c
K_{c}
frame-specific
core tokens and
K
s
K_{s}
shared spatial anchors per frame,
where
K
c
+
K
s
≤
N
s
K_{c}+K_{s}\leq N_{s}
.
Selection uses attention from the dense conditional forward
at a full denoising step
τ
d
\tau_{d}
.
For brevity, we write
U
ℓ
​
(
i
,
f
,
j
)
=
U
ℓ
(
τ
d
)
​
(
i
,
f
,
j
)
U_{\ell}(i,f,j)=U_{\ell}^{(\tau_{d})}(i,f,j)
below.
Scoring Action-Relevant Regions.
The temporal alignment in
Figure 2
(Insight 1(a)) suggests that each future frame should be
scored using its associated action queries.
Assuming that
H
H
is divisible by
F
F
, we partition the
H
H
queries into
F
F
consecutive groups of size
H
/
F
H/F
,
denoting the group for frame
f
f
by
ℐ
f
\mathcal{I}_{f}
.
The spatial score at layer
ℓ
\ell
is
S
ℓ
​
(
f
,
j
)
=
∑
i
∈
ℐ
f
α
f
,
i
​
U
ℓ
​
(
i
,
f
,
j
)
,
S_{\ell}(f,j)=\sum_{i\in\mathcal{I}_{f}}\alpha_{f,i}U_{\ell}(i,f,j),
(2)
where the nonnegative weights
α
f
,
i
\alpha_{f,i}
sum to one.
Central queries receive larger weights because queries near
group boundaries also attend to neighboring frames.
To identify layers whose attention provides localized signals for token selection, we assign each layer a score
Q
ℓ
=
R
ℓ
​
(
1
−
E
ℓ
)
Q_{\ell}=R_{\ell}(1-E_{\ell})
.
Here,
R
ℓ
R_{\ell}
measures frame-aligned attention mass,
and
E
ℓ
E_{\ell}
is the normalized spatial entropy averaged
across frames
(
Zhang et al., 2025b
)
.
We select the
K
layer
K_{\mathrm{layer}}
highest-scoring layers,
forming
ℒ
core
\mathcal{L}_{\mathrm{core}}
, and aggregate their
spatial scores:
V
⁡
(
f
,
j
)
=
∑
ℓ
∈
ℒ
core
Q
ℓ
​
S
ℓ
​
(
f
,
j
)
∑
ℓ
∈
ℒ
core
Q
ℓ
.
V(f,j)=\frac{\sum_{\ell\in\mathcal{L}_{\mathrm{core}}}Q_{\ell}S_{\ell}(f,j)}{\sum_{\ell\in\mathcal{L}_{\mathrm{core}}}Q_{\ell}}.
(3)
Appendix
A.1
provides the query-group
definition and layer-quality calculations.
Frame-Specific Core Tokens.
Since attention hotspots shift across future frames,
each frame selects its own core positions using
V
⁡
(
f
,
j
)
V(f,j)
.
We first identify positions with high core scores in at least
one future frame, then select frame-specific tokens from
this common candidate pool.
Let
𝒥
=
{
1
,
…
,
N
s
}
\mathcal{J}=\{1,\ldots,N_{s}\}
denote all spatial positions.
The candidate pool contains
N
s
−
K
s
N_{s}-K_{s}
positions, preserving
capacity for the shared anchors.
We construct the pool and select core positions as
𝒫
\displaystyle\mathcal{P}
=
TopK
j
∈
𝒥
⁡
(
max
f
⁡
V
⁡
(
f
,
j
)
,
N
s
−
K
s
)
,
\displaystyle=\operatorname{TopK}_{j\in\mathcal{J}}\left(\max_{f}V(f,j),\,N_{s}-K_{s}\right),
(4)
𝒞
f
\displaystyle\mathcal{C}_{f}
=
TopK
j
∈
𝒫
⁡
(
V
⁡
(
f
,
j
)
,
K
c
)
,
\displaystyle=\operatorname{TopK}_{j\in\mathcal{P}}\left(V(f,j),K_{c}\right),
(5)
where
TopK
\operatorname{TopK}
returns the indices of the
largest scores.
The maximum across frames determines the common candidate
pool, while each frame’s own scores determine its core
selection, allowing the retained positions to follow
spatially shifting hotspots.
Shared Spatial Anchor Tokens.
The increased cross-frame attention overlap after hotspot
exclusion (
Figure 2
, Insight 2) suggests
that contextual regions receive more consistent attention
across future frames.
We therefore complement frame-specific core tokens
with shared spatial anchors.
To avoid duplicating core positions, anchors are selected
from
ℰ
\mathcal{E}
, the positions not selected as core
tokens in any frame.
Because all core selections lie within the common pool
𝒫
\mathcal{P}
of size
N
s
−
K
s
N_{s}-K_{s}
, at least
K
s
K_{s}
positions
remain available for anchors.
Appendix
A.1
provides the formal
candidate-set definition and availability guarantee.
To rank these candidates, we aggregate
S
ℓ
​
(
f
,
j
)
S_{\ell}(f,j)
over all layers using the quality weights
Q
ℓ
Q_{\ell}
,
including signals beyond the layers selected for
core localization.
Let
μ
j
\mu_{j}
and
CV
j
\mathrm{CV}_{j}
denote the cross-frame
mean and coefficient of variation of this aggregated
score at position
j
j
.
We select positions with strong and consistent attention:
𝒮
=
TopK
j
∈
ℰ
⁡
(
μ
j
1
+
CV
j
,
K
s
)
.
\mathcal{S}=\operatorname{TopK}_{j\in\mathcal{E}}\left(\frac{\mu_{j}}{1+\mathrm{CV}_{j}},K_{s}\right).
(6)
The numerator rewards attention strength, while
the denominator penalizes cross-frame variation.
Each frame retains the positions
𝒞
f
∪
𝒮
\mathcal{C}_{f}\cup\mathcal{S}
.
The two sets are disjoint, giving exactly
K
c
+
K
s
K_{c}+K_{s}
tokens
per frame and a fixed sequence length for sparse execution.
Only anchor positions are shared: their representations
remain frame-specific and are updated separately.
4.2
Pilot: Online Selection and Sparse Execution
In this section, we implement sparse execution through lightweight
attention profiling, cross-step selection reuse, and cached
prediction updates.
Together, these mechanisms reduce selection and execution
overhead while maintaining the sampling updates required
by joint visual–action denoising.
Low-Overhead Attention Profiling.
At each full step, Pilot obtains the required scores from
the dense conditional forward without an additional
network forward.
The scorer computes only the query–key products between
each future frame and its associated action-query group.
It reuses the original log-sum-exp normalizers computed over
all visible keys, preserving the attention mass used
in layer-quality scoring.
The resulting probabilities are aggregated directly into
S
ℓ
​
(
f
,
j
)
S_{\ell}(f,j)
, avoiding materialization of the full attention
matrix or an additional value-weighted attention output.
Appendix
A.2
provides the extraction
formula and implementation details.
Cross-Step Reuse and Compact Execution.
The observed cross-step attention consistency motivates
reusing selections within each action chunk.
A full step constructs the selection and caches the visual
velocity predictions; subsequent sparse steps reuse both.
Given the short denoising schedules of the evaluated WAMs,
our default configuration uses only the initial step for
full computation.
Selections and cached predictions are recomputed for each
new action chunk.
Alternative refresh schedules are evaluated in
Appendix
C.2
.
During sparse steps, the selected future tokens are processed
as a compact sequence together with all observation and action
tokens.
Pilot reuses their original-position indices and compatible
packing metadata across Transformer layers and denoising steps.
The fixed per-frame token budget maintains a constant sequence
length despite different spatial selections across frames.
Sampling Updates for Omitted Regions.
Future tokens omitted from Transformer computation still
participate in the sampling process.
Let
τ
d
\tau_{d}
denote the most recent full step and
𝐌
v
\mathbf{M}_{\mathrm{v}}
the retained-position mask mapped
to the full future-latent grid.
Pilot combines current predictions at retained positions
with cached predictions elsewhere:
𝐯
~
v
(
τ
)
=
𝐌
v
⊙
𝐯
v
(
τ
)
+
(
𝟏
−
𝐌
v
)
⊙
𝐯
v
(
τ
d
)
,
τ
>
τ
d
,
\widetilde{\mathbf{v}}_{\mathrm{v}}^{(\tau)}=\mathbf{M}_{\mathrm{v}}\odot\mathbf{v}_{\mathrm{v}}^{(\tau)}+(\mathbf{1}-\mathbf{M}_{\mathrm{v}})\odot\mathbf{v}_{\mathrm{v}}^{(\tau_{d})},\qquad\tau>\tau_{d},
(7)
where
⊙
\odot
denotes element-wise multiplication,
𝐯
v
(
τ
)
\mathbf{v}_{\mathrm{v}}^{(\tau)}
contains current predictions
scattered to their retained positions in the full grid,
and
𝐯
v
(
τ
d
)
\mathbf{v}_{\mathrm{v}}^{(\tau_{d})}
is the cached prediction
from the full step.
The original sampler uses
𝐯
~
v
(
τ
)
\widetilde{\mathbf{v}}_{\mathrm{v}}^{(\tau)}
to update all future latents, including omitted regions,
while action predictions are recomputed at every step.
