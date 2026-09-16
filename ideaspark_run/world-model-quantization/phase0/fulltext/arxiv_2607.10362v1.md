# A Control Theory of Predictability in Latent World Models

paper_id: arxiv:2607.10362v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

A latent world model supports planning by predicting the future of a compact latent state, which
allows an agent to act without modeling every detail of its observations. This places two competing
requirements on the representation: it must retain enough of the signal to be useful, and it must
remain
predictable
from its own past by a simple predictor acting in the latent space.
Joint-Embedding Predictive Architectures
(
Assran et al., 2023
;
LeCun and others, 2022
)
make this trade-off the
explicit training objective, mapping observations to a latent space with a shared encoder and
operating a predictor inside that space rather than reconstructing the input. Once trained, the
latent model is deployed inside a planner, such as model-predictive control or trajectory
optimization in latent space
(
Hafner et al., 2020
;
Garcia and Ross, 2013
;
Assran et al., 2025
)
, that rolls
candidate action sequences forward and executes the one of lowest predicted cost.
A world model is evaluated not by its prediction error but by the quality of the actions it induces,
and here common practice rests on an assumption that we argue is unsound: that lowering the
prediction error, the single- or multi-step rollout loss on held-out data, is the correct objective
for control. A planner does not query the model on the training distribution. It queries it on the
states that candidate actions reach, and those states generally leave the data manifold, where the
model extrapolates. The prediction loss is an average over the data distribution, whereas control
success is a functional of the states the planner visits; the two need not move together, and we show
that they need not.
This paper develops a control theory for latent world models organized around this objective. We
prove in Section
3
that the planner’s suboptimality is controlled by a single
scalar, the largest discrepancy between the model’s
predicted
plan-cost and the
true
plan-cost over the plans the planner considers, so that the target for control is to make the
predicted cost track the true cost rather than to lower the averaged prediction error. We then
establish why the averaged error is the wrong instrument. It cannot bound this discrepancy once the
query leaves the data support, which is the decoupling; the harmful part of the discrepancy is
one-sided, arising from a
hole
on which the predicted cost underestimates the true; and the
discrepancy is seeded only when the predictor carries realizable nonlinearity. To make the objective
quantitative, Section
4
prices the predicted-versus-true gap. Under a linear-control
premise it separates into a small on-manifold residual, on which the predicted and true dynamics
agree, and an off-manifold divergence, on which an action carries the state off the data manifold and
the two dynamics diverge. The off-manifold divergence is the binding term and the object of our main
claim; it is priced by a
rollout tax
that compounds the residual along the action-selected
trajectory and is bounded by no data-average. The on-manifold residual is priced by a
spectral
tax
set by the non-normality of the latent transition operator, whose closed-form frontier vanishes
on the self-adjoint, Gaussian boundary where prior identifiability guarantees live and thereby places
those guarantees at the zero-cost corner of a broader surface; it does not by itself govern control.
Contributions.
(1)
The correct control objective, and one core theorem
(Section
3
).
We prove that a latent planner’s suboptimality is bounded
by twice the supremum gap between predicted and true plan-cost over the candidate set. The
control objective is therefore to make predicted cost track true cost at the committed plan;
we then prove three supporting lemmas showing why the data-averaged prediction error cannot
serve this objective: it fails to bound the gap once the query leaves the data support, it is
two-sided where the gap is one-sided, and it is blind to the realizable nonlinearity that seeds
the gap.
(2)
Pricing the gap, with the off-manifold divergence as the binding term
(Section
4
).
Under a linear-control premise the gap separates into an on-manifold
residual and an off-manifold divergence. We price the divergence—an off-manifold rollout tax
that no data-average bounds—and combine it with the residual into a single control-error theorem
with an optimal exploration width. The on-manifold residual is priced by a spectral tax that
vanishes on the self-adjoint boundary of prior work but, as the experiments show, does not by
itself govern control.
(3)
A theory-guided method and empirical validation
(Section
5
).
The analysis prescribes a minimal intervention, a linear state
readout, together with three falsifiable predictions. Latent model-predictive control experiments
confirm the decoupling, in that the single-step validation error does not separate a
fourteen-point spread in success across seeds, and identify the off-manifold amplification as an
informative training signal; synthetic-operator experiments corroborate the spectral-tax pricing
formulas and the off-manifold suboptimality bound.
Full derivations and proofs of every result, together with the numerical-verification
protocols, are collected in the appendices.

## Method

Standard practice measures a world model by its prediction error and treats that error as a proxy
for control quality. We open with the setup that separates the two quantities, then state the one
theorem that identifies the correct control objective, and finally prove three lemmas that
explain why the prediction error cannot serve it.
3.1
Setup
Let
{
X
t
}
\{X_{t}\}
be a stationary, ergodic, first-order Markov process with transition (Koopman)
operator
P
P
on
L
2
​
(
μ
)
L^{2}(\mu)
; restricting to the mean-zero subspace
H
0
=
𝟏
⟂
H_{0}=\mathbf{1}^{\perp}
gives
P
0
=
P
|
H
0
P_{0}=P|_{H_{0}}
, real with
‖
P
0
‖
≤
1
\|P_{0}\|\leq 1
. A whitened encoder corresponds to an isometry
V
:
ℝ
d
→
H
0
V:\mathbb{R}^{d}\to H_{0}
; the
content
it retains is
𝒞
=
‖
P
0
​
V
‖
HS
2
\mathcal{C}=\|P_{0}V\|_{\mathrm{HS}}^{2}
and
the single-step
gap
is
δ
2
=
‖
(
I
−
Π
)
​
P
0
​
V
‖
2
\delta^{2}=\|(I-\Pi)P_{0}V\|^{2}
with
Π
=
V
​
V
∗
\Pi=VV^{*}
. For control we take the
deployed latent dynamics to be the learned linear model
z
t
+
1
=
A
^
​
z
t
+
B
^
​
u
t
+
ε
t
,
z_{t+1}=\hat{A}z_{t}+\hat{B}u_{t}+\varepsilon_{t},
(1)
with
(
A
^
,
B
^
)
(\hat{A},\hat{B})
the least-squares fit and residual
ε
\varepsilon
. Let
ℳ
=
supp
​
μ
\mathcal{M}=\mathrm{supp}\,\mu
be the data manifold,
r
⁡
(
z
)
=
dist
⁡
(
z
,
ℳ
)
r(z)=\mathrm{dist}(z,\mathcal{M})
the
off-manifold distance, and assume the residual grows linearly off the manifold,
‖
ε
⁡
(
z
,
u
)
‖
≤
δ
0
+
L
​
r
​
(
z
)
\|\varepsilon(z,u)\|\leq\delta_{0}+L\,r(z)
, with
δ
0
\delta_{0}
the on-manifold residual and
L
L
a
nonlinearity scale (
L
=
0
L=0
for a genuinely linear system).
A planner ranks candidate action sequences
u
∈
𝒰
u\in\mathcal{U}
by a predicted terminal cost and
executes the best. Write the predicted and true terminal costs to a goal
z
g
z_{g}
as
D
^
​
(
u
)
=
‖
z
^
H
​
(
u
)
−
z
g
‖
,
D
⁡
(
u
)
=
‖
z
H
​
(
u
)
−
z
g
‖
,
\hat{D}(u)=\|\hat{z}_{H}(u)-z_{g}\|,\qquad D(u)=\|z_{H}(u)-z_{g}\|,
(2)
where
z
^
H
​
(
u
)
\hat{z}_{H}(u)
is the rollout of (
1
) under
u
u
and
z
H
​
(
u
)
z_{H}(u)
the true terminal
state. The planner selects
u
^
=
arg
⁡
min
u
​
D
^
​
(
u
)
\hat{u}=\arg\min_{u}\hat{D}(u)
, while the true optimum is
u
⋆
=
arg
⁡
min
u
⁡
D
⁡
(
u
)
u^{\star}=\arg\min_{u}D(u)
. Because candidate actions reach states off
ℳ
\mathcal{M}
, the planner
queries the predictor on a
reachable measure
ν
\nu
its candidates generate, generally with
supp
​
ν
⊋
supp
​
μ
\mathrm{supp}\,\nu\supsetneq\mathrm{supp}\,\mu
. Reading the same residual under the two measures
gives the training gap
δ
μ
2
=
𝔼
μ
​
‖
ε
‖
2
\delta_{\mu}^{2}=\mathbb{E}_{\mu}\|\varepsilon\|^{2}
(the monitored MSE) and the
deployment gap
δ
ν
2
=
𝔼
ν
​
‖
ε
‖
2
\delta_{\nu}^{2}=\mathbb{E}_{\nu}\|\varepsilon\|^{2}
(what success rests on).
3.2
The control objective: predicted cost must track true cost
Everything the planner does is choose
u
^
\hat{u}
to minimize
D
^
\hat{D}
. Its regret is therefore
controlled by how faithfully
D
^
\hat{D}
ranks candidates against
D
D
—nothing else. This is the
content of the core theorem.
Theorem 1
(Control objective: suboptimality is priced by the predicted-versus-true gap)
.
For a planner that executes
u
^
=
arg
⁡
min
u
∈
𝒰
​
D
^
​
(
u
)
\hat{u}=\arg\min_{u\in\mathcal{U}}\hat{D}(u)
,
D
⁡
(
u
^
)
−
D
⁡
(
u
⋆
)
≤
2
​
sup
u
∈
𝒰
|
D
^
​
(
u
)
−
D
⁡
(
u
)
|
.
D(\hat{u})-D(u^{\star})\ \leq\ 2\sup_{u\in\mathcal{U}}\big|\hat{D}(u)-D(u)\big|.
(3)
Consequently the sole objective for control is to make the predicted plan-cost
D
^
\hat{D}
agree with
the true plan-cost
D
D
on the candidate set—in particular at the plan
u
^
\hat{u}
the planner
commits to. Lowering the data-averaged prediction error is neither necessary nor sufficient.
Proof.
Let
g
=
sup
u
|
D
^
​
(
u
)
−
D
⁡
(
u
)
|
g=\sup_{u}|\hat{D}(u)-D(u)|
. Since
u
^
\hat{u}
minimizes
D
^
\hat{D}
,
D
^
​
(
u
^
)
≤
D
^
​
(
u
⋆
)
\hat{D}(\hat{u})\leq\hat{D}(u^{\star})
,
so
D
⁡
(
u
^
)
≤
D
^
​
(
u
^
)
+
g
≤
D
^
​
(
u
⋆
)
+
g
≤
D
⁡
(
u
⋆
)
+
2
​
g
,
D(\hat{u})\leq\hat{D}(\hat{u})+g\leq\hat{D}(u^{\star})+g\leq D(u^{\star})+2g,
which is (
3
). The bound is attained, so no smaller quantity governs regret; and
g
g
is
a supremum over the candidate set, which by construction contains off-manifold plans, whereas the
monitored error is an average over
ℳ
\mathcal{M}
—the two are compared next.
∎
The gap
g
=
sup
u
|
D
^
−
D
|
g=\sup_{u}|\hat{D}-D|
is the
control tax
: the price the planner pays for
predicting a plan-cost that differs from the truth at the plan it most wants to believe. The three
lemmas that follow establish that the monitored prediction error is structurally unable to bound
g
g
, along three independent axes—distribution, sign, and nonlinearity.
Lemma 1
(Decoupling: the averaged error does not bound the gap)
.
If
ν
≪
μ
\nu\ll\mu
with density
w
=
d
​
ν
/
d
​
μ
w=d\nu/d\mu
, then
δ
ν
2
≤
‖
w
‖
∞
​
δ
μ
2
\delta_{\nu}^{2}\leq\|w\|_{\infty}\,\delta_{\mu}^{2}
. But if
ν
\nu
places any mass off
supp
​
μ
\mathrm{supp}\,\mu
, then
δ
μ
\delta_{\mu}
gives
no
upper bound on
δ
ν
\delta_{\nu}
: there is a residual with
δ
μ
=
0
\delta_{\mu}=0
and
δ
ν
\delta_{\nu}
—and hence the gap
g
g
of
Theorem
1
—arbitrarily large. Since control success is monotone in
δ
ν
\delta_{\nu}
,
every in-manifold quantity (
δ
μ
\delta_{\mu}
and hence the monitored MSE, the content frontier, and the
spectral tax of Section
4
) places no bound on success once actions carry the query
off support, and
ρ
⁡
(
MSE
,
success
)
⟶
0
.
\rho\big(\mathrm{MSE},\ \mathrm{success}\big)\ \longrightarrow\ 0.
(4)
Proof sketch.
In-support this is change of measure,
∫
‖
ε
‖
2
​
𝑑
ν
=
∫
‖
ε
‖
2
​
w
​
𝑑
μ
≤
‖
w
‖
∞
​
δ
μ
2
\int\!\|\varepsilon\|^{2}d\nu=\int\!\|\varepsilon\|^{2}w\,d\mu\leq\|w\|_{\infty}\delta_{\mu}^{2}
. Off-support the density is infinite and the two integrals are
independent: set
ε
≡
0
\varepsilon\equiv 0
on
supp
​
μ
\mathrm{supp}\,\mu
(so
δ
μ
=
0
\delta_{\mu}=0
) and free
elsewhere. With
δ
μ
\delta_{\mu}
unable to control
δ
ν
\delta_{\nu}
and success a function of
δ
ν
\delta_{\nu}
,
no functional relation ties
δ
μ
\delta_{\mu}
to success. Full proof:
Appendix
C
, Theorems
C.2.1
–
C.2.2
.
∎
Lemma 2
(The consequential part of the gap is one-sided (holes))
.
With signed error
Δ
​
D
=
D
^
−
D
\Delta D=\hat{D}-D
, a
hole
is a candidate that looks best in prediction
yet is truly bad, of depth
Hole
=
sup
u
(
−
Δ
​
D
)
+
\mathrm{Hole}=\sup_{u}(-\Delta D)_{+}
. The suboptimality obeys
D
⁡
(
u
^
)
−
D
⁡
(
u
⋆
)
≤
Δ
​
D
​
(
u
⋆
)
+
Hole
,
D(\hat{u})-D(u^{\star})\ \leq\ \Delta D(u^{\star})+\mathrm{Hole},
(5)
and the rollout MSE—a two-sided, in-manifold average—cannot bound the hole depth, a one-sided,
off-manifold supremum. Only underestimates—assigning a low predicted cost to a truly high-cost
action—affect the suboptimality, which is why in-manifold fixes such as autoregressive rollout
training, which reduce exposure bias on the support, leave success unchanged.
Proof sketch.
Since
u
^
\hat{u}
minimizes
D
^
\hat{D}
,
D
⁡
(
u
^
)
=
D
^
​
(
u
^
)
−
Δ
​
D
​
(
u
^
)
≤
D
^
​
(
u
⋆
)
−
Δ
​
D
​
(
u
^
)
=
D
⁡
(
u
⋆
)
+
Δ
​
D
​
(
u
⋆
)
−
Δ
​
D
​
(
u
^
)
D(\hat{u})=\hat{D}(\hat{u})-\Delta D(\hat{u})\leq\hat{D}(u^{\star})-\Delta D(\hat{u})=D(u^{\star})+\Delta D(u^{\star})-\Delta D(\hat{u})
with
−
Δ
​
D
​
(
u
^
)
≤
Hole
-\Delta D(\hat{u})\leq\mathrm{Hole}
.
A model exact on
μ
\mu
(MSE
=
0
=0
) can still assign a low predicted cost to a
μ
\mu
-null
counterfactual whose true cost is high, making the hole depth unbounded; MSE and hole depth differ in
distribution, aggregation (mean vs. supremum), and sign. Appendix
C
,
Theorems
C.3.1
–
C.3.3
.
∎
Lemma 3
(The gap is seeded only by realizable nonlinearity)
.
A deployed MLP predictor
g
g
has a best linear fit
g
=
A
∗
​
z
+
B
∗
​
u
+
n
g=A^{*}z+B^{*}u+n
, and the gap splits orthogonally
as
δ
μ
2
=
ι
2
+
‖
n
‖
L
2
​
(
μ
)
2
\delta_{\mu}^{2}=\iota^{2}+\|n\|_{L^{2}(\mu)}^{2}
into an insufficiency term
ι
\iota
and a
realizable-nonlinearity term
‖
n
‖
\|n\|
. If
‖
n
‖
=
0
\|n\|=0
the predictor is globally linear, the predicted
cost is a single quadratic bowl, and no hole exists. Realizable nonlinearity is thus necessary for a
hole; we do not claim the hole depth is monotone in
‖
n
‖
\|n\|
, and Section
5
finds
that reducing it is not the operative lever in practice.
Proof sketch.
m
z
=
𝔼
⁡
[
m
x
∣
z
t
]
m_{z}=\mathbb{E}[m_{x}\mid z_{t}]
is an orthogonal projection of the full-state conditional mean, giving
the Pythagorean split; if
n
≡
0
n\equiv 0
then
g
g
is affine and
‖
A
∗
​
z
+
B
∗
​
u
−
z
g
‖
2
\|A^{*}z+B^{*}u-z_{g}\|^{2}
has a unique
minimizer without folding. Deployment failure therefore materializes
only
when the
representation genuinely carries nonlinear content. Appendix
C
,
Theorems
C.4.1
–
C.4.2
.
∎
Geometry of the gap.
The bound isolates a specific failure geometry. On
supp
​
μ
\mathrm{supp}\,\mu
the predicted and true costs
agree up to the monitored residual, so the ranking induced by
D
^
\hat{D}
is reliable there. A candidate
action can map the latent state outside
supp
​
μ
\mathrm{supp}\,\mu
, where
D
^
\hat{D}
is an extrapolation of a
predictor fitted on
μ
\mu
; when such an extrapolation underestimates the true cost it realizes the
configuration termed a
hole
in Lemma
2
, a candidate of low predicted cost whose
true terminal state is distant from the goal. Since the planner selects
u
^
=
arg
⁡
min
u
⁡
D
^
\hat{u}=\arg\min_{u}\hat{D}
, an underestimate at
u
^
\hat{u}
enters the suboptimality while an overestimate
does not, which is the one-sidedness of Lemma
2
. The three lemmas describe this object
along three axes: it is not detectable by a
μ
\mu
-average because its support is disjoint from
μ
\mu
(Lemma
1
), it contributes only through underestimates (Lemma
2
), and
it does not arise for a globally affine predictor, for which
D
^
\hat{D}
is a single quadratic with a
unique minimizer (Lemma
3
). Reducing it therefore requires either observations
in the affected region (counterfactual data) or a penalty that raises
D
^
\hat{D}
off the manifold
(
D
^
+
λ
​
r
^
\hat{D}+\lambda\hat{r}
), rather than a lower average residual on
supp
​
μ
\mathrm{supp}\,\mu
.
Reading.
Theorem
1
names the target—predicted cost tracking true cost at the committed
plan—and the three lemmas locate why the monitored loss misses it: the gap lives off the data
manifold (Lemma
1
), it is one-sided in the consequential direction
(Lemma
2
), and it is nonlinear-seeded (Lemma
3
). The operative
levers are therefore to fill holes by manifold-aware pessimism (
D
^
+
λ
​
r
^
\hat{D}+\lambda\hat{r}
) or to shrink
the gap’s nonlinear component—not to lower MSE. To turn the objective into computable quantities
we now price the gap.
