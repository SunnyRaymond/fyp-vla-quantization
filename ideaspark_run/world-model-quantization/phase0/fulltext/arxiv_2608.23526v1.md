# Correcting a learned physical invariant improves world-model rollouts

paper_id: arxiv:2608.23526v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

World models learn to predict future observations and can use those predictions for planning. Good
video prediction, however, does not by itself show that a model has learned the underlying dynamics
of the system it observes. That distinction matters most when we ask the model to predict far beyond
the trajectories it saw during training.
A common way to look for physical structure is to fit a probe from the hidden state to a physical
variable. Probe accuracy tells us what information a latent representation retains, but not whether
the model’s transition uses that information. Our own baseline makes this problem concrete. Six
randomly initialized DreamerV3 models contain polynomial functions of their latent state that
correlate with pendulum energy at up to
0.908
0.908
. A strong correlation can therefore appear even
before learning.
Figure 1:
Each point represents one model.
(a)
The three trained conservative models recover
energy correlations between
0.967
0.967
and
0.975
0.975
, while six untrained models span
0.17
0.17
–
0.91
0.91
.
(b)
The recovered scalar changes very little within observation-conditioned trajectories
for conservative models and much more for damped models.
(c)
Rollout error as a function of
projection strength
α
\alpha
, relative to no edit. Enforcing the recovered constraint lowers error;
enforcing a matched random polynomial constraint usually raises it; applying the recovered
constraint to a damped model changes little.
We therefore ask whether a trained world model contains a scalar that its own transition
approximately preserves, and whether failures to preserve that scalar contribute to prediction
error. We study a frozen DreamerV3
(
Hafner et al., 2023
)
trained only on
64
×
64
64\times 64
video of
a Gymnasium pendulum, with no physical labels and no actor or critic.
Independently trained models recover nearly identical energy-like invariants. Decodability alone
does not explain this: randomly initialized Dreamers also produce strongly energy-correlated
readouts, while matched models trained on damped dynamics contain no comparably conserved scalar.
The recovered invariant then begins to drift during autonomous imagination. Correcting that drift
improves 50-step predictions in every conservative model, whereas matched random corrections usually
make them worse. Dreamer has learned a physical constraint that its own rollout dynamics fail to
preserve.

## Method

Model and data.
We analyze the published DreamerV3 architecture
(
Hafner et al., 2023
)
: 13.5M parameters, categorical
32
×
32
32\times 32
stochastic latents, a
convolutional encoder and decoder, Kullback–Leibler balancing, and unimix. We train only the world
model, with no actor or critic.
The data come from Gymnasium’s
Pendulum-v1
(
Towers et al., 2024
)
. Each simulator step
renders a
500
×
500
500\times 500
frame, which we crop to
448
×
448
448\times 448
and block-average to
64
×
64
64\times 64
. We
set the simulator state directly, sampling
θ
∼
U
⁡
[
−
1.8
,
1.8
]
\theta\sim U[-1.8,1.8]
and
θ
˙
∼
U
⁡
[
−
2.2
,
2.2
]
\dot{\theta}\sim U[-2.2,2.2]
, and reject any trajectory that reaches the simulator’s
|
θ
˙
|
=
8
|\dot{\theta}|=8
speed clip, since clipping would break conservation. Actions are always zero. A
single frame does not reveal velocity, so the model must infer
θ
˙
\dot{\theta}
from the sequence.
We train on 204 trajectories of 120 frames using Adam at
10
−
4
10^{-4}
, batch size 16, and sequence
length 64, capped at 30 minutes of wall-clock time, which gives about 6,500 gradient steps. The
remaining 52 trajectories are reserved as a post-training
analysis set
. Three independently
trained seeds pass checks fixed before analysis: KL divergence above 1 nat, one-step decoding at
least
4
×
4\times
better than predicting the dataset mean, and finite rollouts.
Appendix
A
gives the remaining training details. Our claims concern a DreamerV3
world model trained on pendulum video, not DreamerV3 as a full reinforcement-learning agent.
Three kinds of latent trajectories.
We use three latent trajectories and keep them
separate throughout.
An
observation-conditioned
trajectory is the sequence of deterministic recurrent states
produced when the model receives the next real frame at every step. This is what the encoder returns
on the analysis set.
The
one-step transition
T
T
is the model’s autonomous recurrent update with zero action and
no new observation.
An
imagination rollout
starts from one encoded state and repeatedly applies
T
T
without
supplying any later frames.
We estimate the latent flow by applying
T
T
once at each state of an observation-conditioned
trajectory, so the transition is autonomous even though the states come from real video.
Sections
3
and
4
measure conservation along observation-conditioned
trajectories. Section
5
tests what happens to the same scalar under autonomous
imagination.
Latent coordinates.
After training we freeze the model. Let
h
h
denote the deterministic
recurrent state on the analysis set and
h
¯
\bar{h}
its mean. We take the top 12 principal directions
of
h
h
, remove any direction with no support in the data, and let
P
P
denote the resulting
orthonormal map. All extraction, projection, and perturbation operate on
z
=
P
⊤
​
(
h
−
h
¯
)
.
z=P^{\top}(h-\bar{h}).
(1)
Let
F
⁡
(
z
)
F(z)
be the projection through
P
P
of the model’s one-step displacement
T
⁡
(
h
)
−
h
T(h)-h
. Using the
model’s own displacement avoids fitting a separate vector field and then analyzing that surrogate.
The candidate family.
We look for the invariant among degree-4 polynomials in
z
z
.
Writing
φ
⁡
(
z
)
\varphi(z)
for the vector of monomials up to degree 4 in the 12 latent coordinates, a
candidate is
C
⁡
(
z
)
=
a
⊤
​
φ
​
(
z
)
C(z)=a^{\top}\varphi(z)
, and the search is over the coefficient vector
a
a
. There are
1819 such monomials, from
z
1
z_{1}
through
z
1
​
z
2
​
z
3
​
z
4
z_{1}z_{2}z_{3}z_{4}
, excluding the constant term, which is
conserved trivially. The pendulum’s energy is quadratic in
θ
˙
\dot{\theta}
and needs polynomial terms
to approximate
cos
⁡
θ
\cos\theta
, and the latent is an unknown nonlinear encoding of
(
θ
,
θ
˙
)
(\theta,\dot{\theta})
rather than those coordinates themselves, so the family must be rich enough
to express energy through that distortion while staying small enough to fit.
The invariance criterion.
An invariant is constant along any one trajectory and differs
between trajectories. We score a candidate by
ratio
​
(
C
)
=
mean within-trajectory variance of
​
C
total variance of
​
C
,
\text{ratio}(C)\;=\;\frac{\text{mean within-trajectory variance of }C}{\text{total variance of }C},
(2)
which is
0
0
for a perfectly conserved scalar and near
1
1
for one that wanders as much within a
trajectory as across the dataset. The denominator keeps the criterion non-trivial: without it,
C
=
0
C=0
would score perfectly while distinguishing nothing.
Both variances are quadratic forms in
a
a
. With
W
W
the mean within-trajectory covariance of the
features and
T
T
their total covariance, Equation
2
is
a
⊤
​
W
​
a
/
a
⊤
​
T
​
a
a^{\top}Wa/a^{\top}Ta
, a
Rayleigh quotient. Its minimiser is the eigenvector of the generalized problem
W
​
a
=
λ
​
T
​
a
Wa=\lambda\,Ta
with the smallest eigenvalue, and
λ
\lambda
is the invariance ratio itself. One
eigendecomposition returns the whole family, ranked from most to least conserved.
Selecting among conserved candidates.
Conservation alone does not pick out a unique
scalar. If
C
C
is conserved then so is any function of it, so
E
E
,
E
2
E^{2}
and mixtures of
E
E
with
other conserved quantities all sit near the top of the ranking, and the leading eigenvector is
generally some combination of them. We therefore use flow alignment as a secondary criterion among
the leading candidates, fitting
C
C
jointly with an antisymmetric operator
B
B
such that
F
≈
B
∇
C
,
B
⊤
=
−
B
.
F\approx B\nabla C,\qquad B^{\top}=-B.
(3)
This mirrors the Hamiltonian relation between a conserved quantity and the flow it generates: an
antisymmetric operator maps the gradient of
C
C
to a direction tangent to its level set
(
Arnold, 1989
)
. Because
∇
C
⊤
B
∇
C
=
0
\nabla C^{\top}B\nabla C=0
for any antisymmetric
B
B
, a
scalar satisfying Equation
3
is constant along the flow by construction. The criterion
also separates candidates the invariance ratio cannot:
E
2
E^{2}
is exactly as conserved as
E
E
but does
not pair with the same
B
B
.
The fit is bilinear, so fixing either
a
a
or
B
B
makes the other a least-squares problem and we
alternate. It searches within the top eight eigenvectors from Equation
2
, so the
recovered
C
C
has eight effective degrees of freedom rather than 1819. That matters for a search run
on 52 trajectories, where a free fit over the full basis could drive the in-sample ratio to zero by
overfitting. Appendix
A
isolates the contribution of the flow criterion:
conservation drives most of the recovery and intervention effect, while flow alignment mainly
reduces variation across seeds.
Reference energy.
We compare
C
C
with the textbook pendulum energy
E
=
1
6
​
θ
˙
2
+
5
​
cos
⁡
θ
E=\tfrac{1}{6}\dot{\theta}^{2}+5\cos\theta
. Gymnasium uses a semi-implicit integrator, so it
conserves a nearby shadow Hamiltonian
H
~
=
H
+
O
⁡
(
Δ
​
t
)
\tilde{H}=H+O(\Delta t)
rather than
E
E
itself
(
Hairer et al., 2006
)
. In our trajectories this produces about 12% relative oscillation in
E
E
without secular drift, which we treat as a noise floor.
The search sees only latent states and the model’s own one-step transition. It never sees
θ
\theta
,
θ
˙
\dot{\theta}
or
E
E
, and nothing in Equations
2
and
3
refers to them. Ground
truth enters afterward, when we score the recovered
C
C
against
E
E
. We selected the extraction
dimension on three development models and fixed it before evaluating the three models reported here.
Code and data.
The extraction code, the run logs behind every figure and table, and
the source of this paper are at
https://github.com/Zarand3r/world-model-invariants
. The figures
regenerate from the committed logs without a GPU.
Preliminary experiments.
We developed the extraction procedure in preliminary
experiments on smaller recurrent models. Those experiments motivated the untrained, dissipative, and
intervention controls used here, and we do not use them as evidence for the DreamerV3 results.
