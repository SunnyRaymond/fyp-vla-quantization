# Latent Goal Prediction from Language for Model-Based Planning

paper_id: arxiv:2606.20627v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Learning to act from high-dimensional observations remains a central open problem in artificial intelligence. World models (WMs) have emerged as a promising paradigm by learning latent representations of environment dynamics that enable planning through imagination
Ha and Schmidhuber (2018)
;
Hafner et al. (2020a)
;
Bruce et al. (2024)
;
Zhou et al. (2025)
;
Maes et al. (2026b)
. By simulating future trajectories in a compact latent space, these models can achieve sample-efficient control and scale across diverse domains
Hafner et al. (2025)
. However, their effectiveness is fundamentally limited by two challenges: how goals are specified and how planning is performed over long horizons.
A common approach is to define goals directly in observation space, for instance using target images
Bar et al. (2025)
;
Assran et al. (2025)
;
Zhou et al. (2025)
;
Daniel et al. (2026)
. Such image goals provide precise supervision and induce well-shaped objectives near the target, but they offer little guidance when the goal lies beyond the planning horizon and are often impractical to obtain for complex tasks. Alternatively, language provides a flexible and human-aligned interface for specifying objectives. Yet, incorporating language into model-based planning remains difficult: contrastive vision-language models (VLMs) induce noisy and highly non-convex alignment objectives
Rocamonde et al. (2024a)
;
Roy et al. (2025)
, while large generative models are computationally ill-suited for the high-frequency evaluations required by model-based planning. As a result, existing methods struggle to combine the precision of image-based objectives with the flexibility of language-based specification.
In parallel, planning with world models is hindered by compounding prediction errors over long horizons
Xiao et al. (2019)
;
Lambert et al. (2022)
;
Quevedo et al. (2025)
. Even small inaccuracies in latent dynamics accumulate over rollouts, degrading the reliability of imagined trajectories and limiting effective planning depth. Consequently, many approaches sidestep this issue by increasing temporal abstraction during training through large frame skips, evaluating on tasks where goals remain within the effective planning horizon, or a combination of both
Zhou et al. (2025)
;
Wang et al. (2026)
;
Maes et al. (2026b)
.
In this work, we address both challenges by using language to predict intermediate latent targets that guide planning toward distant objectives. We introduce
LAGO
(
Latent Goal Prediction from Language
), a framework that generates structured latent trajectories directly within the world model’s representation space. Rather than optimizing toward a single distant objective, our approach decomposes tasks into language-conditioned latent subgoals, which are re-predicted online at each planning step. This effectively transforms long-horizon planning into a sequence of locally tractable objectives, mitigating compounding errors while retaining alignment with the overall task. By grounding language in the latent space of the world model, LAGO bridges the gap between image-based and language-based goal specification, combining their complementary strengths within a single planning framework.
We summarize our contributions as follows:
•
To bridge the gap between image-based and language-based goal specification, we introduce LAGO, a world model trained to map natural language instructions to target states within its own latent space, combining the precision of visual goals with the flexibility of language.
•
Building on this formulation, we extend goal prediction to generate sequences of intermediate latent subgoals. We further introduce a planning objective that softly aligns imagined trajectories with these subgoals across time, enabling robust long-horizon planning through continual online re-prediction rather than explicit subgoal completion detection.
•
We demonstrate that our approach enables robust long-horizon planning across multiple environments. To systematically evaluate this setting, we introduce a distance-based protocol that varies task difficulty, showing that prior methods degrade sharply beyond their effective planning horizon while our method remains robust.

## Method

3.1
LAGO
LAGO is a Joint Embedding Predictive Architecture
(
Assran et al., 2023
)
supporting two operating modes: one-step transition prediction conditioned on actions, and subgoal prediction conditioned on language instructions.
Figure 2
:
The LAGO architecture.
LAGO is a Joint Embedding Predictive Architecture operating in two modes over a shared latent space. Given a current latent observation
z
t
z_{t}
, a conditioned latent predictor
π
⁡
(
⋅
)
\pi(\cdot)
forecasts a target latent state: either the next state
z
~
t
+
1
\tilde{z}_{t+1}
when conditioned on a low-level action, or a subgoal state
z
~
ρ
\tilde{z}_{\rho}
when conditioned on a language instruction and a completion scalar
ρ
\rho
. Both modes share a single predictor and is agnostic to the choice of latent space.
Vision Encoder.
Given an RGB observation
o
t
∈
ℝ
H
×
W
×
3
o_{t}\in\mathbb{R}^{H\times W\times 3}
, the encoder produces a sequence of patch tokens:
z
t
=
ψ
⁡
(
o
t
)
∈
ℝ
N
×
D
z
z_{t}=\psi(o_{t})\in\mathbb{R}^{N\times D_{z}}
(1)
where
N
N
is the number of patches and
D
z
D_{z}
is the embedding dimension. LAGO is agnostic to the choice of vision encoder, and can leverage either pre-trained or jointly learned representations.
Multi-Modal Conditioning.
All conditioning signals are projected into a shared condition space of dimension
D
c
D_{c}
. A learned
condition-type embedding
τ
∈
ℝ
D
c
\tau\in\mathbb{R}^{D_{c}}
is added to each conditioning vector to inform
the predictor of the operating mode.
Action conditioning.
For next-step transition prediction, continuous low-level actions
a
t
∈
ℝ
D
a
a_{t}\in\mathbb{R}^{D_{a}}
are encoded via a two-layer MLP
ϕ
a
​
c
​
t
:
ℝ
D
a
→
ℝ
D
c
\phi_{act}:\mathbb{R}^{D_{a}}\to\mathbb{R}^{D_{c}}
, while discrete actions use a learned embedding layer mapping to the same space.
The resulting conditioning vector is:
e
t
a
​
c
​
t
=
ϕ
a
​
c
​
t
​
(
a
t
)
e_{t}^{act}=\phi_{act}(a_{t})
(2)
c
t
a
​
c
​
t
=
e
t
a
​
c
​
t
+
τ
a
​
c
​
t
∈
ℝ
D
c
c_{t}^{act}=e_{t}^{act}+\tau_{act}\in\mathbb{R}^{D_{c}}
(3)
Subgoal conditioning.
For subgoal prediction, a language instruction is first embedded by
a language embedding model
ω
⁡
(
⋅
)
\omega(\cdot)
to obtain
g
∈
ℝ
D
L
g\in\mathbb{R}^{D_{L}}
, then projected to
ℝ
D
c
\mathbb{R}^{D_{c}}
via a two-layer MLP
ϕ
l
​
a
​
n
​
g
:
ℝ
D
L
→
ℝ
D
c
\phi_{lang}:\mathbb{R}^{D_{L}}\to\mathbb{R}^{D_{c}}
. A completion scalar
ρ
t
∈
[
0
,
1
]
\rho_{t}\in[0,1]
estimating task progress is projected independently to
ℝ
D
c
\mathbb{R}^{D_{c}}
via a separate two-layer MLP
ϕ
c
​
o
​
m
​
p
:
ℝ
→
ℝ
D
c
\phi_{comp}:\mathbb{R}\to\mathbb{R}^{D_{c}}
. The subgoal conditioning vector combines both signals:
e
t
l
​
a
​
n
​
g
=
ϕ
l
​
a
​
n
​
g
​
(
g
)
,
e
t
c
​
o
​
m
​
p
=
ϕ
c
​
o
​
m
​
p
​
(
ρ
t
)
e_{t}^{lang}=\phi_{lang}(g),\quad e_{t}^{comp}=\phi_{comp}(\rho_{t})
(4)
c
t
s
​
g
=
e
t
l
​
a
​
n
​
g
+
e
t
c
​
o
​
m
​
p
+
τ
s
​
g
∈
ℝ
D
c
c_{t}^{sg}=e_{t}^{lang}+e_{t}^{comp}+\tau_{sg}\in\mathbb{R}^{D_{c}}
(5)
Conditioning injection.
The final condition vector
c
t
∈
ℝ
D
c
c_{t}\in\mathbb{R}^{D_{c}}
(either
c
t
a
​
c
​
t
c_{t}^{act}
or
c
t
s
​
g
c_{t}^{sg}
depending on the operating mode) is broadcast across all
N
N
patch token
positions and concatenated with the latent
z
t
∈
ℝ
N
×
D
z
z_{t}\in\mathbb{R}^{N\times D_{z}}
, yielding the
conditioned input
z
t
⊕
c
t
∈
ℝ
N
×
(
D
z
+
D
c
)
z_{t}\oplus c_{t}\in\mathbb{R}^{N\times(D_{z}+D_{c})}
.
Latent Predictor.
The predictor
π
⁡
(
⋅
)
\pi(\cdot)
operates over the conditioned token sequence
z
t
⊕
c
t
∈
ℝ
N
×
(
D
z
+
D
c
)
z_{t}\oplus c_{t}\in\mathbb{R}^{N\times(D_{z}+D_{c})}
. Before returning the processed tokens, we remove the broadcasted condition tokens from the patch tokens to recover the original embedding dimension of the visual encoder
D
z
D_{z}
:
z
^
=
π
⁡
(
z
t
⊕
c
t
)
∈
ℝ
N
×
D
z
\hat{z}=\pi(z_{t}\oplus c_{t})\in\mathbb{R}^{N\times D_{z}}
(6)
Training Objectives.
The model is trained with a unified mean-squared error objective.
Next-step transition prediction.
Transition samples are drawn from an action-labeled interaction dataset. Each sample consists of a state observation
o
t
o_{t}
, the executed action
a
t
a_{t}
and the subsequent observation
o
t
+
1
o_{t+1}
. Using
a
t
a_{t}
we create
c
t
a
​
c
​
t
c_{t}^{act}
as described in equations (
2
) and (
3
), and train the predictor to minimize:
ℒ
act
=
‖
z
^
t
+
1
−
z
t
+
1
‖
2
2
,
z
^
t
+
1
=
π
⁡
(
z
t
⊕
c
t
a
​
c
​
t
)
\mathcal{L}_{\text{act}}=\|\hat{z}_{t+1}-z_{t+1}\|_{2}^{2},\quad\hat{z}_{t+1}=\pi(z_{t}\oplus c_{t}^{act})
(7)
Subgoal prediction.
Subgoal samples are drawn from an action-less demonstration dataset. Given an demonstration episode of length
G
G
, we sample an initial state observation
o
j
o_{j}
with index
j
∈
{
0
,
…
,
G
−
2
}
j\in\{0,\dots,G-2\}
, and a subgoal state observation
o
k
o_{k}
with index
k
∈
{
j
+
1
,
…
,
G
−
1
}
k\in\{j+1,\dots,G-1\}
. From these, we set the completion scalar as
ρ
=
k
−
j
G
−
1
−
j
∈
(
0
,
1
]
.
\rho=\frac{k-j}{G-1-j}\in(0,1].
(8)
Intuitively, this encodes how far along the episode the subgoal lies relative to
o
j
o_{j}
. With this completion scalar we create
c
s
​
g
c^{sg}
as described in equations (
4
) and (
5
), and train the predictor to minimize:
ℒ
subgoal
=
‖
z
^
ρ
−
z
ρ
‖
2
2
,
z
^
ρ
=
π
⁡
(
z
j
⊕
c
s
​
g
)
\mathcal{L}_{\text{subgoal}}=\|\hat{z}_{\rho}-z_{\rho}\|_{2}^{2},\quad\hat{z}_{\rho}=\pi(z_{j}\oplus c^{sg})
(9)
Because both objectives optimise the same latent-space prediction objective under different conditioning inputs, transition and subgoal samples can be trained jointly within a single unified framework.
3.2
Planning with LAGO
Figure 3
:
Planning with LAGO.
At each step, LAGO generates a sequence of
K
K
latent subgoals from an arbitrary list of completion scalars
{
ρ
k
}
k
=
1
K
\{\rho_{k}\}_{k=1}^{K}
, thereby connecting the current state to the language-specified goal. A CEM planner then optimizes action sequences by evaluating candidate rollouts against the subgoals via
J
LAGO
J_{\text{LAGO}}
, a soft minimum cost that measures the best alignment achieved between each subgoal and any timestep along the rollout. Subgoals are re-predicted at every planning step, allowing the agent to continuously adapt to its evolving state.
Subgoal Generation.
Given the current observation
o
t
o_{t}
and a language-specified goal, we generate
K
K
latent subgoals
{
z
1
s
​
g
,
…
,
z
K
s
​
g
}
\{z^{sg}_{1},\dots,z^{sg}_{K}\}
at uniformly spaced completion scalars
ρ
k
=
k
K
\rho_{k}=\frac{k}{K}
,
k
∈
{
1
,
…
,
K
}
k\in\{1,\dots,K\}
using the subgoal conditioning described in
3.1
. All subgoals are predicted independently from the current
z
t
z_{t}
rather than autoregressively, avoiding compounding prediction errors. Subgoals are re-predicted at every MPC step, so the sequence continuously adapts to the agent’s evolving state.
Action Optimization and Cost Formulation.
At each planning step, the CEM solver samples candidate action sequences over a fixed horizon
H
H
, generating latent state rollouts
z
^
1
:
H
\hat{z}_{1:H}
. To evaluate these rollouts against the predicted subgoals, we introduce a soft minimum trajectory cost that measures, for each subgoal, the best alignment achieved at any point along the rollout:
J
LAGO
=
∑
k
=
1
|
S
|
λ
k
⋅
(
−
τ
log
∑
h
=
1
H
exp
(
−
1
τ
MSE
(
z
^
h
,
z
k
s
​
g
)
)
)
J_{\text{LAGO}}=\sum_{k=1}^{|S|}\lambda^{k}\cdot\left(-\tau\log\sum_{h=1}^{H}\exp\!\left(-\frac{1}{\tau}\text{MSE}\big(\hat{z}_{h},\,z^{sg}_{k}\big)\right)\right)
(10)
where
λ
∈
(
0
,
1
]
\lambda\in(0,1]
is a subgoal decay factor and
τ
>
0
\tau\,{>}\,0
is a temperature parameter controlling the hardness of the temporal aggregation. At
τ
→
0
\tau\,{\to}\,0
, the soft-min approaches a hard minimum, selecting only the closest timestep to each subgoal. As
τ
\tau
increases, timesteps near the minimum contribute proportionally more, favoring trajectories that converge to and remain near a subgoal over those that only transiently visit it. The decay
λ
k
\lambda^{k}
prioritizes earlier subgoals over distant ones.
