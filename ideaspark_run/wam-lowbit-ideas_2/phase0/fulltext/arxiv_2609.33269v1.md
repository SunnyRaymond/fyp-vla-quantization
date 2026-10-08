# Q-WAM: 4-Bit Quantization of World Action Models with Action-Subspace Protection

paper_id: arxiv:2609.33269v1
tier: H
source_used: html_arxiv
warning: none

## Intro

World Action Models (WAMs) couple action generation with prediction of future visual states to learn manipulation policies
(
Li et al., 2026
;
Ye et al., 2026b
;
Yuan et al., 2026
)
. Despite strong task performance, their large backbones and iterative denoising make deployment costly. Recent efforts accelerate inference by removing test-time future generation
(
Yuan et al., 2026
)
, simplifying action modules
(
Ma et al., 2026
)
, and distilling denoising steps
(
Akbari et al., 2026b
;
Lyu et al., 2026
)
. Memory demand remains a deployment bottleneck; for example, running ImageWAM
(
Zhang et al., 2026b
)
in bf16 requires 19.9 GB of GPU memory. Post-training quantization complements these advances by reducing weight and activation precision without retraining the full model. Achieving these savings requires keeping the generated actions accurate for reliable closed-loop control.
Mixed-precision quantization balances compression and accuracy by keeping sensitive components at higher precision and quantizing the rest more aggressively. Existing VLA methods allocate precision across weight tensors or individual channels
(
Akbari et al., 2026a
;
Xu et al., 2026
)
. SVDQuant
(
Li et al., 2025
)
instead keeps a small low-rank part of each layer in 16 bits. This moves activation outliers into the weights and stores the dominant components of each weight in a 16-bit low-rank branch, so that only the residual is quantized to 4 bits. This works well for text-to-image diffusion models, but like most post-training methods, SVDQuant decides what to protect by reducing the quantization error of each linear layer’s output on its own.
In a WAM, however, what matters is the generated action, and a layer’s own output error says little about how much the action changes once that error passes through the rest of the network and the remaining denoising steps. Because visual world modeling and action generation are tightly coupled through iterative denoising, choosing what to keep in higher precision requires tracing how quantization errors propagate through the full action-generation process. Our analysis shows that this downstream action sensitivity concentrates in compact activation subspaces, motivating the central question:
which activation directions should retain higher precision to keep the generated action accurate?
A natural first question is whether reducing activation outliers suffices to preserve action fidelity. Smoothing
(
Xiao et al., 2023
)
and rotation
(
Ashkboos et al., 2024
)
reduce these outliers, but rounding the resulting activations to 4 bits still introduces considerable error, leaving existing quantization methods unsuitable for WAMs. Figure
1
illustrates the distinction between activation magnitude and action sensitivity: smoothing and rotation make activation magnitudes more uniform (Figure
1
b
), yet the resulting quantization errors can still distort the final action along a few sensitive directions (Figure
1
c
). In the examined layers, a compact subspace accounts for much of the final-action sensitivity, motivating selective protection of these directions (Figure
1
d
). This suggests keeping the action-sensitive subspace in higher precision and the complementary computation in W4A4.
Figure 1:
Removing outliers is not enough.
(a)
The activation entering a layer has a
few outliers.
(b)
Smoothing and rotation remove them.
(c)
Quantizing
the layer still damages the action, through a few AOG directions.
(d)
Our proposed ASP protects those
directions and the damage is gone.
Building on these observations, we propose
Q-WAM
, a post-training quantization framework for W4A4 world-action models. Q-WAM estimates an Action Observability Gramian (AOG), a per-layer sensitivity matrix measuring how activation errors affect the final action, using label-free randomized probes through the unrolled denoising process during calibration. Furthermore, we introduce Action-Subspace Protection (ASP), which splits each protected linear layer into two parallel computations whose outputs are summed. A low-dimensional branch projects activations onto the AOG’s most action-sensitive directions and applies the corresponding projected weights in 16-bit precision, while the complementary computation uses 4-bit weights and activations. Since the selected subspace is fixed during inference, the 16-bit branch is small and adds negligible computational overhead. For a fixed protection rank, we show that the selected subspace minimizes a local approximation to action distortion when activation rounding errors have equal variance in all directions. For WAMs with Mixture-of-Transformers architecture, Q-WAM aggregates AOG sensitivity scores to concentrate protection on experts with high estimated action sensitivity.
We evaluate Q-WAM on three WAMs across 50 RoboTwin 2.0 tasks
(
Chen et al., 2026
)
and five real-world tasks on bimanual UR3 arms and a Unitree G1 humanoid. In simulation, Q-WAM limits the success rate drop from bf16 to at most 1.03 points while reducing targeted-block memory by 68–71%, and on physical robots it improves success over SVDQuant, the strongest prior method, by 12.8–17.6 points. Our contributions are as follows:
•
We introduce the Action Observability Gramian (AOG), a per-layer matrix predicting how much a rounding error in each weighted combination of the layer’s input channels changes the final action. We estimate it label-free with random probes.
•
We propose Action-Subspace Protection (ASP), which keeps the most action-sensitive channel combinations of each layer in a small 16-bit branch and quantizes the rest to W4A4. Moreover, we introduced a method to use the AOG to decide which experts to protect.
•
Across Fast-WAM, ImageWAM, and LingBot-VA, Q-WAM preserves near-bf16 success on RoboTwin 2.0 with substantially reduced memory. Evaluation on Unitree G1 and bimanual UR3 demonstrates improved physical task success over SVDQuant, while ablations examine subspace protection, weight quantization, and expert selection.

## Method

We propose Q-WAM, a post-training W4A4 quantization method for World Action Models that allocates
activation precision by the sensitivity of the generated action. Section
4.1
defines and estimates this sensitivity, the Action Observability Gramian (AOG).
Section
4.2
introduces Action-Subspace Protection (ASP), which keeps the activation
directions that the AOG marks as most sensitive in 16 bits and quantizes the remaining weights and
activations to 4 bits. Section
4.3
aggregates the AOG over the layers of each expert to
select the experts that receive this protection.
Figure 2:
Overview of Q-WAM.
(a)
The AOG
G
ℓ
G_{\ell}
measures how far a rounding error in each weighted combination of layer
ℓ
\ell
’s input channels moves the action in later denoising steps.
(b)
ASP keeps the most sensitive of these combinations in a 16-bit branch and quantizes the deflated remainder to 4 bits; the action mass decides which experts receive it.
4.1
The Action Observability Gramian
Within one action prediction, layer
ℓ
\ell
runs once for each of the
h
h
action tokens at each of the
T
T
steps of Equation
1
. We index these token–step pairs by
i
∈
ℐ
ℓ
i\in\mathcal{I}_{\ell}
,
write
x
ℓ
(
i
)
x^{(i)}_{\ell}
for the input activation at pair
i
i
, and write
δ
ℓ
(
i
)
=
Q
4
​
(
x
ℓ
(
i
)
)
−
x
ℓ
(
i
)
\delta^{(i)}_{\ell}=Q_{4}(x^{(i)}_{\ell})-x^{(i)}_{\ell}
for its quantization residual. The residual changes
with the input, so we treat it as random vector over calibration inputs
(
o
t
,
l
)
(o_{t},l)
. Let
𝒜
\mathcal{A}
be the action chunk
a
a
generated by the full-precision
model and
𝒜
δ
\mathcal{A}_{\delta}
the chunk generated when the activations carry the residuals
δ
=
(
δ
ℓ
(
i
)
)
ℓ
,
i
\delta=(\delta^{(i)}_{\ell})_{\ell,i}
. A first-order Taylor expansion gives
𝒜
δ
−
𝒜
=
∑
ℓ
∑
i
∈
ℐ
ℓ
J
ℓ
(
i
)
​
δ
ℓ
(
i
)
+
O
⁡
(
‖
δ
‖
2
)
,
J
ℓ
(
i
)
=
∂
a
∂
x
ℓ
(
i
)
∈
ℝ
m
×
d
ℓ
,
\mathcal{A}_{\delta}-\mathcal{A}=\sum_{\ell}\sum_{i\in\mathcal{I}_{\ell}}J^{(i)}_{\ell}\,\delta^{(i)}_{\ell}+O(\|\delta\|^{2}),\qquad J^{(i)}_{\ell}=\frac{\partial a}{\partial x^{(i)}_{\ell}}\in\mathbb{R}^{m\times d_{\ell}},
(4)
The Jacobian
J
ℓ
(
i
)
J^{(i)}_{\ell}
says how the final action responds to a small change in the input of
layer
ℓ
\ell
at pair
i
i
. It accounts for everything that happens after that point: the rest of the
block, and all remaining denoising steps. Our goal is that the
quantized model generates the same action as the full-precision model, so we minimize the expected
squared change of the action caused by the rounding errors. Inserting Equation
4
into
this squared change gives one term for every layer and token–step pair, plus cross terms between
different pairs. We treat the rounding errors as zero-mean, uncorrelated with each other, and
uncorrelated with the Jacobians (Appendix
A.2
states these approximations
precisely). The cross terms then vanish, and since all token–step pairs of a layer share one error
distribution, what remains is a sum with one term per layer:
𝔼
​
‖
𝒜
δ
−
𝒜
‖
2
≈
∑
ℓ
𝔼
⁡
[
δ
ℓ
⊤
​
G
ℓ
​
δ
ℓ
]
,
G
ℓ
=
𝔼
⁡
[
∑
i
∈
ℐ
ℓ
J
ℓ
(
i
)
⊤
​
J
ℓ
(
i
)
]
∈
ℝ
d
ℓ
×
d
ℓ
.
\mathbb{E}\big\|\mathcal{A}_{\delta}-\mathcal{A}\big\|^{2}\;\approx\;\sum_{\ell}\mathbb{E}\big[\delta_{\ell}^{\top}G_{\ell}\,\delta_{\ell}\big],\qquad G_{\ell}=\mathbb{E}\Big[\sum_{i\in\mathcal{I}_{\ell}}J^{(i)\top}_{\ell}J^{(i)}_{\ell}\Big]\in\mathbb{R}^{d_{\ell}\times d_{\ell}}.
(5)
We call
G
ℓ
G_{\ell}
the Action Observability Gramian (AOG) of layer
ℓ
\ell
. Equation
5
says
what the AOG measures: if the input of layer
ℓ
\ell
carries a rounding error
δ
\delta
, then
δ
⊤
​
G
ℓ
​
δ
\delta^{\top}G_{\ell}\,\delta
approximates the expected squared change this error causes in the final
action. This
is useful for two reasons. First, it turns a question about the whole model, how far quantization
moves the robot’s action, into one matrix per layer that we compute once during calibration.
Second, because
G
ℓ
G_{\ell}
is a matrix and not a single score, it tells how sensitive the action is to
errors in every direction of the layer’s input, that is, in every weighted combination of its
channels. For each
layer, this quadratic form is also exact to second order: at the full-precision model,
G
ℓ
G_{\ell}
is the
Hessian of
1
2
​
𝔼
​
‖
𝒜
δ
−
𝒜
‖
2
\frac{1}{2}\mathbb{E}\|\mathcal{A}_{\delta}-\mathcal{A}\|^{2}
with respect to the layer’s input
(Appendix
A.1
).
Estimating the AOG
G
ℓ
G_{\ell}
.
Computing
G
ℓ
G_{\ell}
exactly is expensive:
J
ℓ
(
i
)
J^{(i)}_{\ell}
has one row per action coordinate,
h
​
d
a
hd_{a}
in
total, and each row needs its own backward pass through all denoising steps. We avoid this with random probes
u
∼
𝒩
⁡
(
0
,
I
m
)
u\sim\mathcal{N}(0,I_{m})
. A single
backward pass of
⟨
a
,
u
⟩
\langle a,u\rangle
returns
J
ℓ
(
i
)
⊤
​
u
J^{(i)\top}_{\ell}u
for every layer and token–step pair
at once
(
Baydin et al., 2018
)
, and since
𝔼
⁡
[
u
​
u
⊤
]
=
I
m
\mathbb{E}[uu^{\top}]=I_{m}
, the expectation of
(
J
⊤
​
u
)
​
(
J
⊤
​
u
)
⊤
(J^{\top}u)(J^{\top}u)^{\top}
over probes equals
J
⊤
​
J
J^{\top}J
for every Jacobian
J
J
. Averaging over
N
N
calibration inputs with
P
P
probes each therefore gives the unbiased estimate
G
^
ℓ
=
1
N
​
P
​
∑
n
=
1
N
∑
p
=
1
P
∑
i
∈
ℐ
ℓ
(
J
ℓ
,
n
(
i
)
⊤
​
u
n
​
p
)
​
(
J
ℓ
,
n
(
i
)
⊤
​
u
n
​
p
)
⊤
,
\widehat{G}_{\ell}=\frac{1}{NP}\sum_{n=1}^{N}\sum_{p=1}^{P}\sum_{i\in\mathcal{I}_{\ell}}\big(J^{(i)\top}_{\ell,n}u_{np}\big)\big(J^{(i)\top}_{\ell,n}u_{np}\big)^{\top},
(6)
where
J
ℓ
,
n
(
i
)
J^{(i)}_{\ell,n}
is the Jacobian at input
n
n
. Each input then costs
P
P
backward passes
instead of
h
​
d
a
hd_{a}
; we use
P
=
12
P{=}12
, about
37
×
37\times
fewer than the exact computation on Fast-WAM. Appendix
A.4
gives the procedure and bounds the error of the
trace and of the leading subspace of
G
^
ℓ
\widehat{G}_{\ell}
.
The AOG predicts quantization damage.
We check this prediction on Fast-WAM. For each of the linear layers in its action expert, we
quantize that layer alone to W4A4 and measure the normalized root-mean-square error (NRMSE) of the
generated action, which is the damage from quantizing the layer. Figure
3
a
compares
this damage with two per-layer criteria. Each point is one layer, placed by its damage and by its
score under a criterion, and each score is divided by its median over the layers so that both
criteria share one axis. The first is the local error
‖
δ
ℓ
‖
2
\|\delta_{\ell}\|^{2}
, the
squared quantization error of the layer input. Round-to-nearest minimizes it by construction, and
widely used post-training methods
(
Lin et al., 2024
;
Frantar et al., 2023
;
Li et al., 2025
)
calibrate
each layer or block from local quantities of this kind, the error or statistics of its own input and
output, without looking at the generated action. The second is the AOG prediction
𝔼
⁡
[
δ
ℓ
⊤
​
G
ℓ
​
δ
ℓ
]
\mathbb{E}[\delta_{\ell}^{\top}G_{\ell}\delta_{\ell}]
. The AOG prediction tracks the damage with a
Pearson correlation
ρ
=
0.95
\rho{=}0.95
, while the local error reaches
ρ
=
0.04
\rho{=}0.04
.
Figure 3:
Deciding what to protect with the AOG.
(a)
Action error from quantizing a
single layer, against the layer’s local error and its AOG score.
(b)
The first 128 eigenvalues of the rotated AOG for several action-expert layers, which fall
quickly; the dashed line marks the rank we keep.
(c)
The action expert’s share of the
action mass and of the parameters in each model.
4.2
Action-Subspace Protection
The AOG weighs each direction of the activation by its effect on the action. A residual
δ
ℓ
\delta_{\ell}
costs
𝔼
⁡
[
δ
ℓ
⊤
​
G
ℓ
​
δ
ℓ
]
\mathbb{E}[\delta_{\ell}^{\top}G_{\ell}\,\delta_{\ell}]
, and we keep the few most costly
directions in 16 bits. We find these directions in three steps. First, we apply the smoothing and
rotation of Section
3
, so the layer quantizes
x
~
=
H
​
diag
⁡
(
γ
)
−
1
​
x
\tilde{x}=H\diag(\gamma)^{-1}x
instead
of
x
x
. Second, we express the AOG in these coordinates. Since
x
=
diag
⁡
(
γ
)
​
H
​
x
~
x=\diag(\gamma)H\tilde{x}
, a rounding
error
δ
\delta
of
x
~
\tilde{x}
shifts
x
x
by
diag
⁡
(
γ
)
​
H
​
δ
\diag(\gamma)H\delta
and therefore costs
δ
⊤
​
G
~
ℓ
​
δ
\delta^{\top}\widetilde{G}_{\ell}\,\delta
, with
G
~
ℓ
=
H
​
diag
⁡
(
γ
)
​
G
ℓ
​
diag
⁡
(
γ
)
​
H
\widetilde{G}_{\ell}=H\diag(\gamma)\,G_{\ell}\,\diag(\gamma)H
. Third, since
G
~
ℓ
\widetilde{G}_{\ell}
is symmetric and positive semidefinite, it has orthonormal eigenvectors
v
j
v_{j}
,
G
~
ℓ
=
∑
j
λ
j
​
v
j
​
v
j
⊤
\widetilde{G}_{\ell}=\sum_{j}\lambda_{j}v_{j}v_{j}^{\top}
, and
δ
⊤
​
G
~
ℓ
​
δ
=
∑
j
λ
j
​
(
v
j
⊤
​
δ
)
2
,
λ
1
≥
λ
2
≥
⋯
≥
0
.
\delta^{\top}\widetilde{G}_{\ell}\,\delta=\sum_{j}\lambda_{j}\big(v_{j}^{\top}\delta\big)^{2},\qquad\lambda_{1}\geq\lambda_{2}\geq\dots\geq 0.
(7)
The cost splits into one term per eigenvector: the component
v
j
⊤
​
δ
v_{j}^{\top}\delta
of the residual contributes
λ
j
​
(
v
j
⊤
​
δ
)
2
\lambda_{j}(v_{j}^{\top}\delta)^{2}
. A large eigenvalue marks a direction in which rounding changes the
action strongly, and a small eigenvalue marks a direction to which the action barely reacts. The
eigenvalues fall by orders of magnitude within a few dozen directions
(Figure
3
b
), so only a few directions of each activation are sensitive.
Each direction mixes many channels and is set by the action’s response to errors, not by
activation values, so we protect directions rather than channels. We call the span of the top
r
r
eigenvectors the layer’s
action subspace, keep the component of
x
~
\tilde{x}
inside it in 16 bits, and quantize the rest to 4 bits.
Splitting the layer.
Let
V
∈
ℝ
d
ℓ
×
r
V\in\mathbb{R}^{d_{\ell}\times r}
hold the top-
r
r
eigenvectors of
G
~
ℓ
\widetilde{G}_{\ell}
, so that
V
⊤
​
V
=
I
r
V^{\top}V=I_{r}
, and let
Π
=
V
​
V
⊤
\Pi=VV^{\top}
project onto the action subspace. We split every activation into
a protected component inside the action subspace and a remainder orthogonal to it,
x
~
=
Π
​
x
~
+
(
I
−
Π
)
​
x
~
=
V
⁡
(
V
⊤
​
x
~
)
⏟
protected component
+
x
~
⟂
⏟
remainder
,
x
~
⟂
=
(
I
−
Π
)
​
x
~
.
\tilde{x}=\Pi\tilde{x}+(I-\Pi)\tilde{x}=\underbrace{V\big(V^{\top}\tilde{x}\big)}_{\text{protected component}}+\underbrace{\tilde{x}_{\perp}}_{\text{remainder}},\qquad\tilde{x}_{\perp}=(I-\Pi)\tilde{x}.
(8)
Since
W
⊤
​
x
=
W
~
⊤
​
x
~
W^{\top}x=\widetilde{W}^{\top}\tilde{x}
, multiplying Equation
8
by
W
~
⊤
\widetilde{W}^{\top}
splits the output of
the layer into two products. The first,
W
~
⊤
​
V
​
(
V
⊤
​
x
~
)
=
(
V
⊤
​
W
~
)
⊤
​
(
V
⊤
​
x
~
)
\widetilde{W}^{\top}V(V^{\top}\tilde{x})=(V^{\top}\widetilde{W})^{\top}(V^{\top}\tilde{x})
, runs
over only
r
≪
d
ℓ
r\ll d_{\ell}
dimensions. In the second,
I
−
Π
I-\Pi
is a symmetric projection, so it leaves
x
~
⟂
\tilde{x}_{\perp}
unchanged and can be moved onto the weight,
W
~
⊤
​
x
~
⟂
=
W
~
⊤
​
(
I
−
Π
)
​
x
~
⟂
=
(
(
I
−
Π
)
​
W
~
)
⊤
​
x
~
⟂
=
W
~
⟂
⊤
​
x
~
⟂
\widetilde{W}^{\top}\tilde{x}_{\perp}=\widetilde{W}^{\top}(I-\Pi)\tilde{x}_{\perp}=\big((I-\Pi)\widetilde{W}\big)^{\top}\tilde{x}_{\perp}=\widetilde{W}_{\perp}^{\top}\tilde{x}_{\perp}
,
where
W
~
⟂
=
(
I
−
Π
)
​
W
~
\widetilde{W}_{\perp}=(I-\Pi)\widetilde{W}
is the weight with its protected component removed. We keep the first
product in 16 bits and quantize both factors of the second to 4 bits,
W
⊤
​
x
=
W
~
⊤
​
x
~
=
(
V
⊤
​
W
~
)
⊤
​
(
V
⊤
​
x
~
)
+
W
~
⟂
⊤
​
x
~
⟂
≈
(
V
⊤
​
W
~
)
⊤
​
(
V
⊤
​
x
~
)
⏟
low-rank 16-bit branch
+
Q
4
​
(
W
~
⟂
)
⊤
​
Q
4
​
(
x
~
⟂
)
⏟
4-bit deflated path
.
W^{\top}x=\widetilde{W}^{\top}\tilde{x}=\big(V^{\top}\widetilde{W}\big)^{\top}\big(V^{\top}\tilde{x}\big)+\widetilde{W}_{\perp}^{\top}\tilde{x}_{\perp}\approx\underbrace{\big(V^{\top}\widetilde{W}\big)^{\top}\big(V^{\top}\tilde{x}\big)}_{\text{low-rank 16-bit branch}}+\underbrace{Q_{4}(\widetilde{W}_{\perp})^{\top}Q_{4}(\tilde{x}_{\perp})}_{\text{4-bit deflated path}}.
(9)
We call this construction Action-Subspace Protection (ASP). Projecting
x
~
\tilde{x}
before
Q
4
Q_{4}
keeps the
protected component out of the 4-bit grid, and deflating the weight to
W
~
⟂
\widetilde{W}_{\perp}
lets the rounding
error
δ
\delta
reach the output only as
(
I
−
Π
)
​
δ
(I-\Pi)\delta
, up to the product of the weight and
activation rounding errors. Both paths are dense GEMMs that run efficiently on GPUs, and the thin rank-
r
r
branch adds
little overhead (Appendix
B.2
).
By Equation
7
, the damage along
v
j
v_{j}
depends on how strongly the action
reacts to that direction (
λ
j
\lambda_{j}
) and how much rounding error lands on it. After the
rotation, the rounding error has roughly the same mean square
σ
2
\sigma^{2}
in every direction, i.e., it
is
isotropic
, so the expected damage along
v
j
v_{j}
is
σ
2
​
λ
j
\sigma^{2}\lambda_{j}
. The top-
r
r
eigenvectors
are therefore the best
r
r
directions to protect, which Appendix
A.3
proves.
4.3
Where to Protect: The action mass of an expert
Each protected layer carries a 16-bit branch, so in a Mixture-of-Transformers we protect only the
experts whose rounding errors damage the action most. We denote an expert by
E
E
, the set of layers
that process one modality, and score it by the expected damage its 4-bit rounding does to the
action. With the isotropic rounding error of Section
4.2
, every direction of a layer
receives error
σ
2
\sigma^{2}
, so the layer’s term in Equation
5
is
σ
2
\sigma^{2}
times the sum
of its eigenvalues,
𝔼
⁡
[
δ
ℓ
⊤
​
G
~
ℓ
​
δ
ℓ
]
=
σ
2
​
tr
⁡
G
~
ℓ
\mathbb{E}[\delta_{\ell}^{\top}\widetilde{G}_{\ell}\delta_{\ell}]=\sigma^{2}\tr\widetilde{G}_{\ell}
. Taking
the same
σ
2
\sigma^{2}
for all layers and summing over the layers of
E
E
, the expert contributes
σ
2
​
μ
E
\sigma^{2}\mu_{E}
, where
μ
E
=
∑
ℓ
∈
E
tr
⁡
(
G
~
ℓ
)
\mu_{E}=\sum_{\ell\in E}\tr\big(\widetilde{G}_{\ell}\big)
(10)
is the action mass of
E
E
. Protecting a layer removes the damage along its top-
r
r
eigenvectors,
which is the most that any rank-
r
r
branch can remove; Appendix
A.3
proves this. As
depicted in Figure
3
b
, the eigenvalues fall steeply, so the protected
directions carry nearly all of a layer’s damage, and the gain from protecting an expert grows with
its action mass. ASP therefore protects the experts with the largest action mass per added 16-bit
value. As depicted in Figure
3
c
, on both Mixture-of-Transformers models the
action expert holds under a fifth of the parameters yet carries
89.3
%
89.3\%
of the action mass on
Fast-WAM (Appendix
C.2
visualize it per layer) and
99.99
%
99.99\%
on ImageWAM, so ASP protects
the action expert alone. LingBot-VA shares one backbone, so ASP covers
all of its layers. Furthermore, table
4
shows that protecting the other expert
changes the success rate by less than the run-to-run RoboTwin variation.
