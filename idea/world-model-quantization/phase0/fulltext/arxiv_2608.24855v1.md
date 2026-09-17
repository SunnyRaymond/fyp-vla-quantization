# LeFlow: Generative Latent Flow Planning for World Models

paper_id: arxiv:2608.24855v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

World models let agents plan by simulating the consequences of their actions
before acting
[
8
]
. Joint-Embedding Predictive
Architectures (JEPAs)
[
14
]
make this practical by modeling
dynamics in a compact latent space: observations are encoded into
low-dimensional latents and a predictor rolls the latent state forward
conditioned on actions. LeWorldModel (LeWM)
[
18
]
is a recent
JEPA that trains stably end-to-end from raw pixels with a single regularization
hyperparameter and is competitive across diverse 2D and 3D control tasks. Latent
world models, in short, have become strong
predictors
.
Figure 1
:
LeFlow plans an order of magnitude faster than CEM at
better success.
End-to-end evaluation time (left) and success rate
(right) for LeWM+CEM and LeFlow under the same protocol (
H
=
5
H{=}5
,
50
50
episodes). Replacing online black-box search with an amortized latent
trajectory prior removes the per-step optimization loop while preserving task
performance.
Planning on top of them, however, has not advanced at the same pace. Even with a
well-trained latent world model in hand, planning is still posed as online
trajectory optimization: for every state–goal pair, an optimizer searches for a
raw action sequence whose autoregressive latent rollout terminates near the
encoded goal. LeWM, for instance, solves this finite-horizon optimal-control
problem at inference with the Cross-Entropy Method
(CEM)
[
24
]
. CEM is a practical optimizer for
low-dimensional, smooth objectives. Our focus is its repeated online cost: CEM
treats the world model as a black-box simulator and solves every state–goal
query
from scratch
: the world model is queried thousands of times per
replanning step yet never contributes structural knowledge about which
trajectories are worth considering, and no effort spent planning one query is
reused for the next.
We propose to amortize planning itself. Rather than repeatedly solving an
optimization problem online, we learn – once, offline – a reusable
latent trajectory prior
directly in the frozen world model’s latent
space. A frozen world model already contains the structure needed: its encoder
organizes observations geometrically and its predictor defines which latent
transitions are dynamically reachable. Planning should exploit that structure,
not rediscover it through black-box search.
We recast planning as
conditional latent trajectory generation
: rather
than searching for actions, we generate a latent path and decode the actions that
realize it afterwards. Action trajectories are low-level and highly multimodal –
many action sequences realize nearly the same state trajectory – whereas latent
state trajectories are smoother and geometrically structured, so the reusable
planning structure lives in how the latent state should evolve.
We instantiate this in
LeFlow
, a lightweight planner on top of a
frozen
LeWM. LeFlow has three modular components: (1) a
rectified-flow latent planner
[
17
,
16
]
that generates the interior of a latent path conditioned on the start and goal
embeddings – the amortized prior; (2) an
inverse dynamics
decoder that
converts each latent transition into an executable action chunk, separating
planning from control; and (3) a
rollout verification
stage that scores
each candidate by its
actual
frozen-LeWM rollout distance to the goal
rather than the generated (clamped) endpoint, projecting generative proposals
back onto the manifold of trajectories the world model can actually control. On LeWM, it yields an order-of-magnitude
reduction in planning time at better success across four benchmarks, as shown
in Figure
1
. Our contributions are:
•
We recast latent planning as conditional latent trajectory generation,
and show that planning knowledge can be
amortized
– learned
once from offline trajectories into a reusable latent trajectory prior,
rather than re-solved online for every state–goal pair.
•
We show that latent world models contain sufficient structure for learned planning: LeFlow combines generative latent trajectory proposals with rollout-based verification
inside
the
original world model, without modifying it.
•
Across four benchmarks, amortized latent
planning replaces iterative action-space optimization with fixed-budget
rollout selection while
improving
success, cutting end-to-end
planning time by roughly an order of
magnitude and generalizing to held-out episodes; ablations show that
latent-path generation beats direct action generation and that rollout
verification is essential for dynamically feasible plans.

## Method

The central design question in LeFlow is
what space to plan in
.
A naïve generative planner generates action sequences directly, but
actions are the wrong abstraction for a reusable prior: they are
high-dimensional and highly multimodal (many kinematically distinct
trajectories realize the same sub-goal), and they are decoupled from the
world model’s internal geometry – the encoder has already organized
task-relevant structure into a compact latent space, yet an action-space
generator must re-derive this structure from scratch.
Latent state trajectories avoid both problems. The encoder collapses
action-level multiplicity: many action sequences that move
o
t
o_{t}
to
o
t
+
1
o_{t+1}
share a single latent transition
z
t
→
z
t
+
1
z_{t}\to z_{t+1}
. Planning in
this space is 1) lower-dimensional and smoother, 2) already organized
around the predictor’s dynamics, and 3) action-agnostic – the planner
reasons about
where the agent should be
, delegating
how it
gets there
to a lightweight inverse dynamics decoder. This separation
decouples multi-step goal-directed reasoning from low-level control,
allowing each module to be trained and improved independently.
LeFlow realizes this design with three learned components on a
frozen
LeWM backbone: (1) a
rectified-flow latent-path
planner
that generates goal-conditioned latent trajectories; (2) an
inverse dynamics decoder
that converts latent transitions into
executable action chunks; and (3) a
rollout reranking
stage that
validates generative proposals against the frozen world model’s dynamics.
We describe each in turn, followed by training and inference.
3.1
Preliminary
LeWM
[
18
]
provides two components that LeFlow uses
unchanged. The
encoder
enc
θ
\mathrm{enc}_{\theta}
maps a pixel
observation
o
o
to a compact latent embedding
z
=
enc
θ
​
(
o
)
z=\mathrm{enc}_{\theta}(o)
.
The
predictor
pred
ϕ
\mathrm{pred}_{\phi}
models latent dynamics
autoregressively,
z
^
t
+
1
=
pred
ϕ
​
(
z
^
t
,
a
t
)
,
\hat{z}_{t+1}=\mathrm{pred}_{\phi}(\hat{z}_{t},a_{t}),
(1)
so given a start latent and an action sequence, LeWM rolls out a predicted
latent trajectory. LeWM’s own planner solves the finite-horizon
optimal-control problem
a
1
:
H
⋆
=
arg
min
a
1
:
H
∥
z
^
H
−
z
g
∥
2
2
a^{\star}_{1:H}=\arg\min_{a_{1:H}}\lVert\hat{z}_{H}-z_{g}\rVert_{2}^{2}
online at every replanning step using
CEM
[
24
]
. We keep the backbone frozen and replace
only this online search. LeWM serves three roles in LeFlow: a
representation backbone that supplies the latent geometry, a dynamics
prior whose structure we distill into the trajectory generator, and a
rollout-based feasibility verifier that grounds generative proposals in
realizable dynamics.
3.2
Latent Flow planner
Let
z
start
=
enc
θ
​
(
o
cur
)
z_{\text{start}}=\mathrm{enc}_{\theta}(o_{\text{cur}})
and
z
goal
=
enc
θ
​
(
o
goal
)
z_{\text{goal}}=\mathrm{enc}_{\theta}(o_{\text{goal}})
be the encoded
current and goal observations. A
latent path
of horizon
H
H
is a
sequence
z
0
:
H
=
(
z
0
,
z
1
,
…
,
z
H
)
z_{0:H}=(z_{0},z_{1},\dots,z_{H})
whose endpoints are clamped
by construction to
z
0
=
z
start
z_{0}=z_{\text{start}}
and
z
H
=
z
goal
z_{H}=z_{\text{goal}}
.
The planner generates only the
H
−
1
H{-}1
interior
latent states.
Fixing the endpoints is a deliberate design choice: it injects the
goal-conditioning directly into the geometric structure of the generated
path, rather than relying on the model to output a path that
happens
to end
near the goal. The model’s capacity is thus fully spent on how
the latent state should evolve between two known anchors – the
qualitative shape
of the trajectory – which is precisely where
task-level planning knowledge lives.
We model the interior path with a conditional rectified
flow
[
17
,
16
]
. The choice of generative
model family matters for online efficiency. Diffusion models denoise
through many small steps, so generating
N
N
candidates at every replanning
step incurs a cost that scales with both
N
N
and the number of denoising
iterations. Rectified flow instead learns a velocity field
v
ψ
v_{\psi}
that
transports a Gaussian noise sample
u
0
∼
𝒩
⁡
(
0
,
I
)
u_{0}\sim\mathcal{N}(0,I)
to a data
sample
u
1
u_{1}
along the
straight
interpolant
u
τ
=
(
1
−
τ
)
​
u
0
+
τ
​
u
1
u_{\tau}=(1{-}\tau)u_{0}+\tau u_{1}
by matching the constant target velocity
u
1
−
u
0
u_{1}-u_{0}
. Because the learned trajectories in function space are approximately
linear, the ODE can be integrated accurately with very few Euler steps –
typically 16 in our setting – regardless of
N
N
. This is well suited to
latent paths: the encoder produces a smooth, low-dimensional space where
straight-line interpolations between nearby embeddings are geometrically
meaningful, making the rectified-flow assumption a natural fit.
The training objective is
ℒ
flow
=
𝔼
τ
,
u
0
,
u
1
∥
v
ψ
(
u
τ
,
τ
∣
z
start
,
z
goal
)
−
(
u
1
−
u
0
)
∥
2
2
,
\mathcal{L}_{\text{flow}}=\mathbb{E}_{\tau,\,u_{0},\,u_{1}}\big\lVert v_{\psi}\!\left(u_{\tau},\tau\mid z_{\text{start}},z_{\text{goal}}\right)-(u_{1}-u_{0})\big\rVert_{2}^{2},
(2)
where
u
1
u_{1}
is the flattened interior of a ground-truth latent path from
the offline dataset. At inference,
N
N
diverse interiors are sampled in
parallel by integrating the learned ODE from
N
N
independent noise draws.
3.3
Inverse dynamics decoder
A latent path specifies
where
the agent should be at each step but
says nothing about
which actions get it there
. Rather than trying
to jointly generate actions alongside latent states – which would
re-introduce the multimodality problem in the generative model – we
delegate action recovery to a dedicated
inverse dynamics decoder
g
ω
g_{\omega}
that operates locally on each latent transition:
a
t
=
g
ω
​
(
[
z
t
,
z
t
+
1
,
z
t
+
1
−
z
t
]
)
.
a_{t}=g_{\omega}\!\big([\,z_{t},\;z_{t+1},\;z_{t+1}-z_{t}\,]\big).
(3)
The decoder takes the current and next latent states together with their
displacement
z
t
+
1
−
z
t
z_{t+1}-z_{t}
as an explicit feature. The
displacement matters: it provides directional information about the
transition that is not separately recoverable from
z
t
z_{t}
or
z
t
+
1
z_{t+1}
alone, and empirically improves the conditioning of the regression
without adding parameters. The inverse dynamics problem is
much better posed in latent space than in observation space: because the
encoder has collapsed many distinct observations to the same latent, the
mapping from a latent transition to the corresponding action chunk is
far less ambiguous than a mapping from raw pixel pairs.
Critically, this separation means the planner and the decoder solve
genuinely different problems at different levels of abstraction. The flow
model reasons about multi-step goal-directed structure – the
shape
of the latent trajectory – while the decoder answers the purely local
question of which action realizes a given latent step. Neither module
needs to solve the other’s problem, and both can be improved or replaced
independently.
The decoder is trained with a mean-squared error loss against the
dataset’s normalized action chunks:
ℒ
inv
=
𝔼
​
∥
g
ω
​
(
[
z
t
,
z
t
+
1
,
z
t
+
1
−
z
t
]
)
−
a
t
∥
2
2
.
\mathcal{L}_{\text{inv}}=\mathbb{E}\,\big\lVert g_{\omega}\!\left([z_{t},\,z_{t+1},\,z_{t+1}-z_{t}]\right)-a_{t}\big\rVert_{2}^{2}.
(4)
3.4
Rollout reranking for controllable manifold
Figure 3
:
Rollout reranking projects generative proposals onto the
controllable latent manifold.
The flow model samples
N
N
candidate latent paths (dashed) with endpoints
clamped to
z
start
z_{\text{start}}
and
z
goal
z_{\text{goal}}
, but generated interiors
may pass through latent states unreachable by any admissible action sequence.
Each candidate is decoded into actions and rolled out autoregressively through
the frozen LeWM predictor (solid), producing an
actual
terminal latent
z
^
H
(
i
)
\hat{z}_{H}^{(i)}
. Candidates are ranked by rollout distance to the goal
(Eq.
5
); the best is executed. This step grounds amortized
proposals in the predictor’s dynamics without modifying the flow model.
Generative models do not respect the world model’s dynamics by
construction: the flow model can interpolate through latent states that
no admissible action sequence actually reaches. Even a geometrically
reasonable path – one that curves smoothly from
z
start
z_{\text{start}}
to
z
goal
z_{\text{goal}}
– may pass through regions of latent space that the
predictor’s learned dynamics cannot follow. We call the set of latent
trajectories that
are
reachable by some action sequence the
controllable latent manifold
, and we use the frozen LeWM predictor
to project our generative proposals back onto it (Figure
3
).
At inference we sample
N
N
candidate paths from the flow model, decode
each into an action sequence, and roll the actions
autoregressively
through the frozen predictor to obtain the true
predicted terminal latent
z
^
H
(
i
)
\hat{z}_{H}^{(i)}
. Candidates are scored by
their rollout distance to the goal:
score
(
i
)
=
∥
z
^
H
(
i
)
−
z
goal
∥
2
2
.
\mathrm{score}^{(i)}=\big\lVert\hat{z}_{H}^{(i)}-z_{\text{goal}}\big\rVert_{2}^{2}.
(5)
Note that lower score indicates a better candidate – the rollout landed
closer to the goal. We therefore select
arg
⁡
min
i
⁡
score
(
i
)
\arg\min_{i}\,\mathrm{score}^{(i)}
and execute its decoded action sequence. This reranking step is
inference-time and selection-only – it does not alter the flow model’s
distribution, only chooses among its samples.
3.5
Training
LeWM is frozen throughout; LeFlow trains only the flow planner
v
ψ
v_{\psi}
and the inverse dynamics decoder
g
ω
g_{\omega}
, jointly, on offline
trajectory data. The total objective is
ℒ
=
ℒ
flow
+
ℒ
inv
+
λ
cons
​
ℒ
cons
.
\mathcal{L}=\mathcal{L}_{\text{flow}}+\mathcal{L}_{\text{inv}}+\lambda_{\text{cons}}\,\mathcal{L}_{\text{cons}}.
(6)
Table 1
:
Main benchmark results: success rate (%) on four
goal-conditioned pixel-control tasks.
Baseline numbers (goal-conditioned BC
and offline RL – GCBC, GCIVL, GCIQL; the JEPA world model PLDM; and the
foundation-encoder world model DINO-WM) are as reported by
Maes et al. [18]
; “–” marks benchmarks for which a baseline is not
reported. LeFlow is the mean
±
\pm
std over
five
independent evaluation runs (seeds
42
42
–
46
46
) under the LeWM
codebase-default protocol (
H
=
5
H{=}5
,
50
50
episodes). Best per column in
bold
.
Method
TwoRoom
PushT
Reacher
OGBench-Cube
Random
0.0
2.0
10.0
48.0
GCBC
[
7
]
100.0
75.0
–
84.0
GCIVL
[
19
]
100.0
33.0
–
56.0
GCIQL
[
13
]
100.0
20.0
–
64.0
PLDM
[
25
]
97.0
78.0
78.0
65.0
DINO-WM
[
28
]
100.0
74.0
79.0
86.0
LeWM-based Method
CEM
[
24
]
82.0
±
\,\pm\,
2.0
89.3
±
\,\pm\,
6.4
68.0
±
\,\pm\,
9.2
73.3
±
\,\pm\,
9.0
iCEM
[
23
]
88.0
±
\,\pm\,
2.0
84.7
±
\,\pm\,
3.1
67.3
±
\,\pm\,
11.4
76.0
±
\,\pm\,
7.2
MPPI
[
26
]
71.3
±
\,\pm\,
6.4
60.7
±
\,\pm\,
4.2
42.7
±
\,\pm\,
4.6
49.3
±
\,\pm\,
9.5
LeFlow
100.0
±
\,\pm\,
0.0
95.2
±
\,\pm\,
3.0
86.8
±
\,\pm\,
4.8
100.0
±
\,\pm\,
0.0
Flow matching loss
ℒ
flow
\mathcal{L}_{\text{flow}}
.
Trains the flow planner to match the distribution of latent-path
interiors from the offline dataset. This is the primary learning signal
for
what trajectories look like
in the world model’s latent space.
Inverse dynamics loss
ℒ
inv
\mathcal{L}_{\text{inv}}
.
Trains the decoder to map latent transitions to action chunks. Because the
decoder is trained on
dataset
transitions, it is naturally aligned
with the regions of latent space that the offline data covers – the same
regions the flow model learns to generate in.
Consistency loss
ℒ
cons
\mathcal{L}_{\text{cons}}
.
The two losses above do not guarantee that generated latent paths are
dynamically realizable:
ℒ
flow
\mathcal{L}_{\text{flow}}
teaches the planner
to mimic the distribution of
recorded
transitions, and
ℒ
inv
\mathcal{L}_{\text{inv}}
teaches the decoder to invert them, but neither
explicitly links the planner’s output to the predictor’s learned dynamics.
The consistency loss closes this loop. For each generated transition
z
t
→
z
t
+
1
z_{t}\to z_{t+1}
, we decode the action
a
^
t
=
g
ω
​
(
[
z
t
,
z
t
+
1
,
z
t
+
1
−
z
t
]
)
\hat{a}_{t}=g_{\omega}([z_{t},z_{t+1},z_{t+1}-z_{t}])
, roll it one step through the frozen predictor
to obtain
z
^
t
+
1
=
pred
ϕ
​
(
z
t
,
a
^
t
)
\hat{z}_{t+1}=\mathrm{pred}_{\phi}(z_{t},\hat{a}_{t})
, and
penalize the gap:
ℒ
cons
=
𝔼
​
∥
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
∥
2
2
.
\mathcal{L}_{\text{cons}}=\mathbb{E}\,\big\lVert\hat{z}_{t+1}-z_{t+1}\big\rVert_{2}^{2}.
(7)
This one-step constraint
distills the predictor’s dynamics into the
planner at training time
: it steers the flow model’s distribution toward
latent transitions that the predictor can actually follow, progressively
reducing the fraction of generated paths that reranking must discard.
The consistency loss and rollout reranking are therefore complementary
rather than redundant – the former shapes the
training distribution
toward the controllable manifold, while the latter selects the best sample
at inference. A small weight (
λ
cons
=
0.1
\lambda_{\text{cons}}=0.1
) is sufficient
because the flow model already learns from in-distribution transitions;
too large a weight would collapse the planner’s distribution onto
the predictor’s myopic one-step rollout, forfeiting the multi-step
planning structure that makes the flow model useful. We ablate this
weight in Section
4.5
.
3.6
Inference
Online planning reduces to a single batched computation: encode the
current and goal observations, sample
N
N
latent paths from the flow model
with a small number of Euler integration steps, decode each into an action
sequence, rerank by frozen-LeWM rollout distance (Eq.
5
),
and execute the best action chunk. Planning runs inside a receding-horizon
MPC loop following LeWM’s action normalization and rollout conventions.
Crucially, there is no online optimization loop: the iterative
search of CEM – hundreds of candidate evaluations over many refinement
rounds, restarted from scratch at each replanning step – is entirely
replaced by a single forward pass through the flow model followed by one
batch of rollout evaluations. The planning computation is thus dominated
by the
N
N
parallel rollouts of the frozen predictor, which are
embarrassingly parallelizable on a GPU, explaining the order-of-magnitude
speedup over CEM reported in Section
4.2
.
Table 2
:
Planning efficiency: LeWM+CEM vs. LeFlow.
Both methods use
the same frozen LeWM backbone and the same evaluation protocol (
H
=
5
H{=}5
,
50
50
episodes). Success rates are reported as in Table
1
(LeFlow:
mean over five runs, seeds
42
42
–
46
46
; LeWM+CEM: mean
±
\pm
std over three
seeds under
stable_worldmodel
defaults); eval time for CEM and LeFlow are both five-run mean. LeFlow exceeds CEM success on
every benchmark while reducing end-to-end planning time by roughly an order
of magnitude. Speedup is the ratio of CEM to mean LeFlow eval time.
Benchmark
Method
Success (%)
↑
\uparrow
Eval Time (s)
↓
\downarrow
Speedup
↑
\uparrow
TwoRoom
LeWM + CEM
82.0
±
\,\pm\,
2.0
224.78
1.0
×
\times
LeFlow
100.0
±
\,\pm\,
0.0
15.58
14.4
×
\times
PushT
LeWM + CEM
89.3
±
\,\pm\,
6.4
198.92
1.0
×
\times
LeFlow
95.2
±
\,\pm\,
3.0
17.42
11.4
×
\times
Reacher
LeWM + CEM
68.0
±
\,\pm\,
9.2
326.01
1.0
×
\times
LeFlow
86.8
±
\,\pm\,
4.8
27.67
11.8
×
\times
OGBench-Cube
LeWM + CEM
73.3
±
\,\pm\,
9.0
224.62
1.0
×
\times
LeFlow
100.0
±
\,\pm\,
0.0
50.23
4.5
×
\times
