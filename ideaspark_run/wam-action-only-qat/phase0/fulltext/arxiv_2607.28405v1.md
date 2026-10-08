# QuantWAMs: Calibrating at the Right Granularity for World Action Models

paper_id: arxiv:2607.28405v1
tier: U
source_used: html_arxiv
warning: none

## Intro

World Action Models integrate a video diffusion world model with a diffusion action expert
to jointly predict future observations and the actions that realize them, a promising
paradigm for general purpose robotic manipulation that is costly to deploy
Yuan et al. (2026)
;
Li et al. (2026)
;
Ye et al. (2026)
;
Bi et al. (2025)
;
Cen et al. (2025)
;
Liang et al. (2025)
.
Each control
cycle requires iterative denoising, and proceeds in closed loop, where every generated
action changes the observations used by later predictions
Chi et al. (2025)
;
Peebles and Xie (2023)
;
Li et al. (2026)
;
Ye et al. (2026)
.
Post-training quantization is
therefore a natural route to real-robot deployment.
Yet most PTQ methods rely on two assumptions that fail for WAMs. They optimize open loop
objectives such as perplexity or single image reconstruction, where each forward pass is
scored independently, and they assume a homogeneous transformer
Frantar et al. (2022)
;
Xiao et al. (2023)
;
Lin et al. (2025)
;
Chen et al. (2025a)
;
He et al. (2023)
;
So et al. (2023)
;
Shang et al. (2023)
;
Li et al. (2023)
.
WAMs instead propagate
early quantization error into later states through closed loop interaction
Ross et al. (2011)
, and distribute
computation across coupled pathways, either dual stream Mixture of Transformers or a shared
diffusion backbone
Yuan et al. (2026)
;
Li et al. (2026)
;
Ye et al. (2026)
;
Bi et al. (2025)
.
A static single stream proxy may mis-allocate precision even
when its calibration loss remains low.
During our experiments to transfer quantization algorithms from classic large models and DiT models to WAMs,
we found that every PTQ decision is a finite-sample estimate fixed before deployment under a
calibration context
: the scope
𝒢
\mathcal{G}
over which evidence is pooled, the state
distribution
𝒟
^
\widehat{\mathcal{D}}
on which it is measured, and the objective
ℒ
\mathcal{L}
under which it is scored. A decision is useful only if all three remain
appropriate at deployment. A
structural
mismatch pools evidence across members without a
common coordinate system, or splits it until sampling noise dominates. A
distributional
mismatch profiles sensitivity on states unreachable in closed loop,
producing phantom denoising-step peaks that misdirect the precision budget. An
objective
mismatch scores a single stream, or fuses streams after squaring,
discarding a coordinatewise interaction present in the joint video–action gradient. The three are not separate
heuristics but one requirement applied along structural, distributional, and objective axes.
We screen each decision at the granularity supported by the available calibration data.
Based on this principle, we propose QuantWAMs, a PTQ algorithm for WAMs.
We evaluate it on RoboTwin 2.0
Chen et al. (2025b)
, LIBERO
Liu et al. (2023)
and real-robot manipulation with
AgiBot G2 using two released WAM implementations
Yuan et al. (2026)
;
Li et al. (2026)
.
The same calibration procedure applies to both implementations with
architecture-specific grouping rules. Under the W4A4-dominant setting, its
reported simulation means differ from FP16 by 0.2–0.7 percentage
points.
Our contributions are threefold:
•
We formulate post-training quantization for WAMs around the structural, distributional, and objective contexts that determine a deployment-time precision decision.
•
We derive when activation evidence may be pooled across modules or modes and characterize the sample-heterogeneity crossover governing Top-
K
K
mask recovery.
•
We jointly score video–action gradients at a calibration-stable layer granularity and use fixed-intervention replay on real rollouts to revise denoising-step schedules at a fixed precision budget.

## Method

Key PTQ decisions, including outlier preservation, layerwise weight-bit allocation,
and timestep protection, constitute finite-sample estimates established prior to deployment.
3.1
Shared-Basis Outlier Calibration
Outlier-aware activation quantization preserves
K
=
⌊
ρ
​
d
⌋
K=\lfloor\rho d\rfloor
input channels at high precision. We retain
Atom’s mixed-precision channel-offloading primitive and study
only how its static mask should be shared. A context
i
i
specifies a module,
execution mode, and depth. Let
A
i
(
n
)
∈
ℝ
T
i
,
n
×
d
A_{i}^{(n)}\in\mathbb{R}^{T_{i,n}\times d}
be its activation on trajectory
n
n
before the offline transforms. The
runtime quantizer receives
A
~
i
(
n
)
=
(
A
i
(
n
)
​
R
)
​
S
i
−
1
,
\widetilde{A}_{i}^{(n)}=\left(A_{i}^{(n)}R\right)S_{i}^{-1},
(1)
where the Hadamard rotation
R
R
is common within an admissible group and
S
i
S_{i}
is a context-specific diagonal smoothing matrix. All statistics below
are collected from this transformed input:
z
i
(
n
)
(
c
)
=
1
T
i
,
n
∑
t
=
1
T
i
,
n
(
A
~
i
(
n
)
)
t
,
c
2
,
e
i
(
c
)
=
𝔼
[
z
i
(
n
)
(
c
)
]
,
e
^
i
(
c
)
=
1
N
i
∑
n
=
1
N
i
z
i
(
n
)
(
c
)
,
e
g
(
c
)
=
∑
i
∈
g
π
i
e
i
(
c
)
,
e
^
g
​
(
c
)
=
∑
i
∈
g
π
i
​
e
^
i
​
(
c
)
.
\begin{gathered}z_{i}^{(n)}(c)=\frac{1}{T_{i,n}}\sum_{t=1}^{T_{i,n}}\left(\widetilde{A}_{i}^{(n)}\right)_{t,c}^{2},\qquad e_{i}(c)=\mathbb{E}[z_{i}^{(n)}(c)],\\[2.84526pt]
\widehat{e}_{i}(c)=\frac{1}{N_{i}}\sum_{n=1}^{N_{i}}z_{i}^{(n)}(c),\qquad e_{g}(c)=\sum_{i\in g}\pi_{i}e_{i}(c),\\[2.84526pt]
\widehat{e}_{g}(c)=\sum_{i\in g}\pi_{i}\widehat{e}_{i}(c).\end{gathered}
(2)
Here
∑
i
π
i
=
1
\sum_{i}\pi_{i}=1
, and
π
i
\pi_{i}
encodes the intended deployment exposure
rather than token count. Under an equal-cost energy surrogate, the
shared-constrained oracle and its estimator are
Ω
g
⋆
\displaystyle\Omega_{g}^{\star}
=
arg
⁡
max
⁡
∑
i
∈
g
|
Ω
|
=
K
⁡
π
i
​
∑
c
∈
Ω
e
i
​
(
c
)
,
\displaystyle=\arg\max_{|\Omega|=K}\sum_{i\in g}\pi_{i}\sum_{c\in\Omega}e_{i}(c),
(3)
=
TopK
K
⁡
(
e
g
)
,
\displaystyle=\operatorname{TopK}_{K}(e_{g}),
Ω
^
g
\displaystyle\widehat{\Omega}_{g}
=
TopK
K
⁡
(
e
^
g
)
.
\displaystyle=\operatorname{TopK}_{K}(\widehat{e}_{g}).
This optimality is restricted to the stated surrogate and does not imply
optimal closed-loop performance.
Coordinate admissibility.
A literal index can be shared only when it denotes the same quantizer-input
coordinate in every member.
Let a common representation
Z
∈
ℝ
n
×
d
Z\in\mathbb{R}^{n\times d}
in a fixed ordered basis and
A
i
=
P
i
​
Z
A_{i}=P_{i}Z
, where
P
i
P_{i}
acts only on token rows. For a column selector
E
Ω
E_{\Omega}
and the restricted diagonal matrix
S
i
,
Ω
S_{i,\Omega}
,
A
~
i
​
E
Ω
=
P
i
​
(
Z
​
R
)
​
E
Ω
​
S
i
,
Ω
−
1
.
\widetilde{A}_{i}E_{\Omega}=P_{i}(ZR)E_{\Omega}S_{i,\Omega}^{-1}.
(4)
Thus the common rotation fixes one ordered basis throughout the group, while
the context-specific diagonal scaling changes magnitudes but not coordinate
identity. A context-specific dense rotation would invalidate literal
index sharing unless an explicit map to a common basis were supplied.
In Fast-WAM, the shared value pathway aligns paired output projections, whereas
expert-private residual streams do not license cross-expert Q/K/V sharing.
In the released shared-backbone LingBot-VA, modes reuse the input columns of
the same physical Linear, making cross-mode sharing admissible when
post-fusion column transforms are shared or coordinate-preserving. None of
these statements implies distributional equality. Pooling across depth is a
separate exchangeability assumption.
For a coordinate-admissible group, model each channel statistic as
z
i
(
n
)
​
(
c
)
\displaystyle z_{i}^{(n)}(c)
=
μ
g
​
(
c
)
+
τ
c
​
η
i
​
(
c
)
+
σ
c
​
ξ
i
(
n
)
​
(
c
)
,
\displaystyle=\mu_{g}(c)+\tau_{c}\eta_{i}(c)+\sigma_{c}\xi_{i}^{(n)}(c),
(5)
e
i
​
(
c
)
\displaystyle e_{i}(c)
=
μ
g
​
(
c
)
+
τ
c
​
η
i
​
(
c
)
,
\displaystyle=\mu_{g}(c)+\tau_{c}\eta_{i}(c),
where
τ
c
2
\tau_{c}^{2}
is cross-member heterogeneity and
σ
c
2
\sigma_{c}^{2}
is
within-member sampling variation. Under the balanced working model with
m
=
|
g
|
m=|g|
, equal weights,
N
N
independent trajectory units per member, and
independent sampling errors across members, the expected member-fidelity
risks are
R
ind
,
c
\displaystyle R_{\mathrm{ind},c}
:
=
∑
i
∈
g
π
i
​
𝔼
​
[
(
e
^
i
​
(
c
)
−
e
i
​
(
c
)
)
2
]
=
σ
c
2
N
,
\displaystyle:=\sum_{i\in g}\pi_{i}\mathbb{E}\!\left[\bigl(\widehat{e}_{i}(c)-e_{i}(c)\bigr)^{2}\right]=\frac{\sigma_{c}^{2}}{N},
(6)
R
pool
,
c
\displaystyle R_{\mathrm{pool},c}
:
=
∑
i
∈
g
π
i
​
𝔼
​
[
(
e
^
g
​
(
c
)
−
e
i
​
(
c
)
)
2
]
,
\displaystyle:=\sum_{i\in g}\pi_{i}\mathbb{E}\!\left[\bigl(\widehat{e}_{g}(c)-e_{i}(c)\bigr)^{2}\right],
=
m
−
1
m
​
τ
c
2
+
σ
c
2
m
​
N
.
\displaystyle=\frac{m-1}{m}\tau_{c}^{2}+\frac{\sigma_{c}^{2}}{mN}.
Proposition 1
(Pooling crossover)
.
For
τ
c
2
>
0
\tau_{c}^{2}>0
, pooling lowers risk exactly when
N
<
N
c
⋆
=
σ
c
2
τ
c
2
,
N
eff
,
c
=
m
​
N
1
+
(
m
−
1
)
​
N
​
τ
c
2
/
σ
c
2
.
N<N_{c}^{\star}=\frac{\sigma_{c}^{2}}{\tau_{c}^{2}},\qquad N_{\mathrm{eff},c}=\frac{mN}{1+(m-1)N\tau_{c}^{2}/\sigma_{c}^{2}}.
(7)
If
τ
c
2
=
0
\tau_{c}^{2}=0
, pooling dominates for every finite
N
N
.
Lower energy risk need not recover a member’s preferred mask. Let
Ω
i
⋆
=
TopK
K
⁡
(
e
i
)
\Omega_{i}^{\star}=\operatorname{TopK}_{K}(e_{i})
,
Δ
i
=
e
i
,
(
K
)
−
e
i
,
(
K
+
1
)
\Delta_{i}=e_{i,(K)}-e_{i,(K+1)}
, and
β
i
=
‖
e
g
−
e
i
‖
∞
\beta_{i}=\|e_{g}-e_{i}\|_{\infty}
. If
β
i
<
Δ
i
/
2
\beta_{i}<\Delta_{i}/2
and the centered
pooled error is channel-wise sub-Gaussian with proxy
s
g
2
s_{g}^{2}
, then
Pr
[
Ω
^
g
≠
Ω
i
⋆
]
≤
2
d
exp
[
−
(
Δ
i
/
2
−
β
i
)
2
2
​
s
g
2
]
.
\Pr\!\left[\widehat{\Omega}_{g}\neq\Omega_{i}^{\star}\right]\leq 2d\exp\!\left[-\frac{(\Delta_{i}/2-\beta_{i})^{2}}{2s_{g}^{2}}\right].
(8)
The architecture first supplies candidate groups. For each benchmark, the
same fixed set of 32 training trajectories is used to estimate the group
statistics, fit the mask, and calibrate the remaining quantizer parameters.
We form a rank window
ℋ
^
g
\widehat{\mathcal{H}}_{g}
around
K
K
and compute
ℋ
^
g
\displaystyle\widehat{\mathcal{H}}_{g}
=
{
c
:
|
rank
↓
⁡
(
e
^
g
​
(
c
)
)
−
K
|
≤
h
}
,
\displaystyle=\left\{c:\left|\operatorname{rank}_{\downarrow}\bigl(\widehat{e}_{g}(c)\bigr)-K\right|\leq h\right\},
(9)
N
^
g
⋆
\displaystyle\widehat{N}_{g}^{\star}
=
median
c
∈
ℋ
^
g
σ
^
c
2
max
⁡
(
τ
^
c
2
,
ε
)
.
\displaystyle=\operatorname*{median}_{c\in\widehat{\mathcal{H}}_{g}}\frac{\widehat{\sigma}_{c}^{2}}{\max(\widehat{\tau}_{c}^{2},\varepsilon)}.
We pool only coordinate-admissible groups satisfying
N
cal
<
N
^
g
⋆
N_{\mathrm{cal}}<\widehat{N}_{g}^{\star}
without degrading bootstrap mask
stability. The bootstrap resamples complete trajectories and preserves the
paired observations across contexts; it is a stability analysis of the same
32-trajectory calibration set, not a second fitting split. The final mask is
TopK
K
⁡
(
e
^
g
)
\operatorname{TopK}_{K}(\widehat{e}_{g})
on all 32 trajectories.
Proofs, estimators, the covariance-aware bootstrap, and architecture-specific
grouping rules appear in Appendix A.
3.2
The Weight Axis: Co-Training-Objective Saliency
The preceding section determines where calibration evidence may be pooled;
here the same principle determines the granularity of weight-precision
decisions. For a Linear
y
L
=
W
L
​
x
L
y_{L}=W_{L}x_{L}
, we evaluate perturbations under the
video–action co-training objective used by the pretrained WAM,
ℓ
co
=
λ
v
​
ℓ
v
+
λ
a
​
ℓ
a
.
\ell_{\mathrm{co}}=\lambda_{\mathrm{v}}\ell_{\mathrm{v}}+\lambda_{\mathrm{a}}\ell_{\mathrm{a}}.
(10)
For each of the 32 paired calibration trajectories, let
g
m
,
L
=
∇
y
L
ℓ
m
g_{m,L}=\nabla_{y_{L}}\ell_{m}
,
m
∈
{
v
,
a
}
m\in\{\mathrm{v},\mathrm{a}\}
, and define
Σ
L
=
𝔼
⁡
[
x
L
​
x
L
⊤
]
,
G
L
m
=
𝔼
⁡
[
g
m
,
L
​
g
m
,
L
⊤
]
.
\Sigma_{L}=\mathbb{E}[x_{L}x_{L}^{\top}],\qquad G_{L}^{m}=\mathbb{E}[g_{m,L}g_{m,L}^{\top}].
(11)
Computing the empirical Fisher after combining the two losses gives
G
L
joint
\displaystyle G_{L}^{\mathrm{joint}}
=
𝔼
⁡
[
(
λ
v
​
g
v
,
L
+
λ
a
​
g
a
,
L
)
​
(
λ
v
​
g
v
,
L
+
λ
a
​
g
a
,
L
)
⊤
]
\displaystyle=\mathbb{E}\!\left[(\lambda_{\mathrm{v}}g_{\mathrm{v},L}+\lambda_{\mathrm{a}}g_{\mathrm{a},L})(\lambda_{\mathrm{v}}g_{\mathrm{v},L}+\lambda_{\mathrm{a}}g_{\mathrm{a},L})^{\top}\right]
(12)
=
λ
v
2
​
G
L
v
+
λ
a
2
​
G
L
a
+
λ
v
​
λ
a
​
(
Ξ
L
+
Ξ
L
⊤
)
,
\displaystyle=\lambda_{\mathrm{v}}^{2}G_{L}^{\mathrm{v}}+\lambda_{\mathrm{a}}^{2}G_{L}^{\mathrm{a}}+\lambda_{\mathrm{v}}\lambda_{\mathrm{a}}\left(\Xi_{L}+\Xi_{L}^{\top}\right),
G
L
fusion
\displaystyle G_{L}^{\mathrm{fusion}}
=
λ
v
2
G
L
v
+
λ
a
2
G
L
a
,
Ξ
L
=
𝔼
[
g
v
,
L
g
a
,
L
⊤
]
.
\displaystyle=\lambda_{\mathrm{v}}^{2}G_{L}^{\mathrm{v}}+\lambda_{\mathrm{a}}^{2}G_{L}^{\mathrm{a}},\qquad\Xi_{L}=\mathbb{E}[g_{\mathrm{v},L}g_{\mathrm{a},L}^{\top}].
Under the diagonal approximation used for scoring, the retained difference
has the explicit form
diag
⁡
(
G
L
joint
−
G
L
fusion
)
=
2
​
λ
v
​
λ
a
​
𝔼
​
[
g
v
,
L
⊙
g
a
,
L
]
.
\operatorname{diag}\left(G_{L}^{\mathrm{joint}}-G_{L}^{\mathrm{fusion}}\right)=2\lambda_{\mathrm{v}}\lambda_{\mathrm{a}}\mathbb{E}[g_{\mathrm{v},L}\odot g_{\mathrm{a},L}].
(13)
Thus, post-hoc Fusion uses both marginal objectives but omits their
coordinatewise alignment before the outer product. Our saliency uses
G
L
=
G
L
joint
G_{L}=G_{L}^{\mathrm{joint}}
, with loss weights and normalizers fixed to
their training-time values. This is a gradient-assisted PTQ step: it requires
the original co-training targets and a backward pass, but it does not update
the pretrained weights.
Let
ϵ
L
(
b
)
=
Q
b
​
(
W
L
)
−
W
L
\epsilon_{L}^{(b)}=Q_{b}(W_{L})-W_{L}
. A Kronecker-factored
empirical-Fisher approximation
Martens and Grosse (2015)
gives the bit-specific
distortion
D
L
​
(
b
)
=
1
2
​
tr
⁡
[
G
L
​
ϵ
L
(
b
)
​
Σ
L
​
(
ϵ
L
(
b
)
)
⊤
]
.
D_{L}(b)=\frac{1}{2}\operatorname{tr}\!\left[G_{L}\epsilon_{L}^{(b)}\Sigma_{L}(\epsilon_{L}^{(b)})^{\top}\right].
(14)
The benefit of upgrading layer
L
L
from
b
lo
b_{\mathrm{lo}}
to
b
hi
b_{\mathrm{hi}}
is
B
L
=
D
L
​
(
b
lo
)
−
D
L
​
(
b
hi
)
.
B_{L}=D_{L}(b_{\mathrm{lo}})-D_{L}(b_{\mathrm{hi}}).
(15)
Layerwise mixed precision is selected by
max
⁡
∑
L
z
L
∈
{
0
,
1
}
⁡
z
L
​
B
L
,
s
.
t
.
∑
L
z
L
​
c
L
≤
ℬ
.
\max_{z_{L}\in\{0,1\}}\sum_{L}z_{L}B_{L},\qquad\mathrm{s.t.}\quad\sum_{L}z_{L}c_{L}\leq\mathcal{B}.
(16)
The implementation uses a count budget over the candidate Linears:
c
L
=
1
c_{L}=1
and
ℬ
=
⌊
0.2
​
|
ℒ
|
⌋
\mathcal{B}=\lfloor 0.2|\mathcal{L}|\rfloor
. It therefore upgrades the
top 20% of candidate Linears by
B
L
B_{L}
; the budget is not weighted by the
number of parameters in a Linear. Under diagonal factors, each scalar
contributes
s
L
​
(
i
,
j
)
=
1
2
​
(
G
L
)
i
​
i
​
(
Σ
L
)
j
​
j
​
[
(
ϵ
L
,
i
​
j
(
lo
)
)
2
−
(
ϵ
L
,
i
​
j
(
hi
)
)
2
]
.
s_{L}(i,j)=\frac{1}{2}(G_{L})_{ii}(\Sigma_{L})_{jj}\left[(\epsilon_{L,ij}^{(\mathrm{lo})})^{2}-(\epsilon_{L,ij}^{(\mathrm{hi})})^{2}\right].
(17)
Element-, column-, and layer-level scores are obtained by summing
s
L
​
(
i
,
j
)
s_{L}(i,j)
over the scalars contained in the corresponding allocation unit.
In particular,
C
L
​
(
j
)
=
∑
i
s
L
​
(
i
,
j
)
C_{L}(j)=\sum_{i}s_{L}(i,j)
is the column score.
Finite calibration makes fine-grained rankings unstable when adjacent scores
are close. We therefore use layer totals for the precision assignment and
retain column totals only as an ordering heuristic inside GPTQ; element-level
scores are not allocation units. The corresponding top-budget ranking bound
and the distinction between full-ranking recovery and selection stability
are given in Appendix B. GPTQ compensation remains governed by
Σ
L
\Sigma_{L}
.
The common Hadamard rotation is fused into the weights, whereas smoothing
statistics and diagonal scales may differ by context.
3.3
Fixed-Intervention Real-Rollout Auditing
Diffusion PTQ is known to require timestep-aware calibration because its
activation distributions vary along the denoising trajectory.
In a WAM, however, the mismatch is
stronger: actions produced by earlier calls change later observations.
A sensitivity profile built from synthetic or open-loop inputs can
therefore measure the right local discrepancy on the wrong states.
Let
q
=
{
q
t
}
t
=
1
T
q=\{q_{t}\}_{t=1}^{T}
be a deployed precision schedule and let
q
0
q_{0}
denote the unprotected baseline, in which every target module uses the low
precision. Here
t
t
indexes the inner denoising step; the outer control-call
index is suppressed. For a recorded FP16 rollout snapshot
x
t
x_{t}
—including
its observation history, chunk position, and persistent cache—define
ℓ
t
​
(
x
t
,
a
)
=
‖
f
a
​
(
x
t
)
−
f
fp
​
(
x
t
)
‖
2
2
.
\ell_{t}(x_{t};a)=\left\|f_{a}(x_{t})-f_{\mathrm{fp}}(x_{t})\right\|_{2}^{2}.
(18)
The following profiles share this discrepancy but differ in the state
distribution and intervention:
S
syn
​
(
t
)
\displaystyle S^{\mathrm{syn}}(t)
=
𝔼
x
t
∼
𝒟
t
syn
​
[
ℓ
t
​
(
x
t
,
q
0
)
]
,
\displaystyle=\mathbb{E}_{x_{t}\sim\mathcal{D}_{t}^{\mathrm{syn}}}[\ell_{t}(x_{t};q_{0})],
(19)
S
obs
​
(
t
,
q
)
\displaystyle S^{\mathrm{obs}}(t;q)
=
𝔼
x
t
∼
𝒟
t
q
​
[
ℓ
t
​
(
x
t
,
q
t
)
]
,
\displaystyle=\mathbb{E}_{x_{t}\sim\mathcal{D}_{t}^{q}}[\ell_{t}(x_{t};q_{t})],
S
ref
replay
​
(
t
)
\displaystyle S_{\mathrm{ref}}^{\mathrm{replay}}(t)
=
𝔼
x
t
∼
𝒟
t
ref
​
[
ℓ
t
​
(
x
t
,
q
0
)
]
.
\displaystyle=\mathbb{E}_{x_{t}\sim\mathcal{D}_{t}^{\mathrm{ref}}}[\ell_{t}(x_{t};q_{0})].
S
syn
S^{\mathrm{syn}}
omits reachable rollout histories;
S
obs
S^{\mathrm{obs}}
uses real states but is self-masked by the active protection schedule: a
protected step may appear insensitive because
q
t
q_{t}
is already active. The
fixed-intervention profile instead replays every recorded state under the
same unprotected intervention
q
0
q_{0}
. We restore the complete immutable FP16
snapshot before each branch, deep-copy persistent state, and match stochastic
seeds between
f
q
0
f_{q_{0}}
and
f
fp
f_{\mathrm{fp}}
. Since the reference
trajectories are generated by the full-precision model, we write
𝒟
t
ref
=
𝒟
t
fp
\mathcal{D}_{t}^{\mathrm{ref}}=\mathcal{D}_{t}^{\mathrm{fp}}
rather than calling
it the quantized deployment distribution. A distribution-shift diagnostic
can compare this profile with replay on all-low-bit rollout states through
Δ
prof
​
(
t
)
=
|
S
𝒟
fp
replay
​
(
t
)
−
S
𝒟
q
0
replay
​
(
t
)
|
.
\Delta_{\mathrm{prof}}(t)=\left|S_{\mathcal{D}^{\mathrm{fp}}}^{\mathrm{replay}}(t)-S_{\mathcal{D}^{q_{0}}}^{\mathrm{replay}}(t)\right|.
(20)
This remains a one-call diagnostic. If
s
j
s_{j}
is the outer closed-loop
state and
ϵ
j
\epsilon_{j}
the local model error, then to first order
δ
​
s
j
+
1
=
A
j
​
δ
​
s
j
+
B
j
​
ϵ
j
,
\delta s_{j+1}=A_{j}\delta s_{j}+B_{j}\epsilon_{j},
(21)
so downstream task impact also depends on products of transition Jacobians,
not on
ℓ
t
\ell_{t}
alone. We therefore use profile values to propose a
controlled schedule repair, never as estimates of marginal task gain.
We use the profile only to
audit
an existing
K
K
-step schedule. Let
𝒯
q
\mathcal{T}_{q}
be its protected set and define
𝒯
replay
=
TopK
t
∈
[
T
]
⁡
S
ref
replay
​
(
t
)
,
|
𝒯
replay
|
=
|
𝒯
q
|
=
K
.
\mathcal{T}_{\mathrm{replay}}=\operatorname{TopK}_{t\in[T]}S_{\mathrm{ref}}^{\mathrm{replay}}(t),\qquad|\mathcal{T}_{\mathrm{replay}}|=|\mathcal{T}_{q}|=K.
(22)
The repaired schedule replaces
𝒯
q
\mathcal{T}_{q}
by
𝒯
replay
\mathcal{T}_{\mathrm{replay}}
without changing the precision levels or
their counts. For each benchmark, the profile is estimated from 32 FP16
closed-loop rollouts that are trajectory-disjoint from the 32-trajectory PTQ
calibration set. The resulting benchmark-level schedule is evaluated on a
separate schedule-validation set and frozen before final testing. Validation
and test trajectories, including their initial-state seeds, are disjoint from
the two 32-rollout sets; task identities may overlap because this is
benchmark-specific calibration rather than held-out-task transfer.
Profile magnitudes are not interpreted as marginal closed-loop gains.
Appendix C gives the complete data flow, and Appendix D specifies snapshot
replay and schedule freezing.
Table 1
:
RoboTwin 2.0 and LIBERO benchmark quantization results on Fast-WAM.
Method
Precision
RoboTwin 2.0
Speedup
↑
\uparrow
LIBERO
Speedup
↑
\uparrow
Mem. (GB)
↓
\downarrow
Clean
↑
\uparrow
Random
↑
\uparrow
Average
↑
\uparrow
Goal
↑
\uparrow
Spatial
↑
\uparrow
Object
↑
\uparrow
Long
↑
\uparrow
Average
↑
\uparrow
Full Precision
FP16
91.9
±
0.2
91.9{\scriptstyle\,\pm\,0.2}
91.8
±
0.6
91.8{\scriptstyle\,\pm\,0.6}
91.9
±
0.3
91.9{\scriptstyle\,\pm\,0.3}
1.0
×
\times
98.2
±
0.3
98.2{\scriptstyle\,\pm\,0.3}
99.8
±
0.2
99.8{\scriptstyle\,\pm\,0.2}
97.0
±
0.3
97.0{\scriptstyle\,\pm\,0.3}
95.2
±
0.4
95.2{\scriptstyle\,\pm\,0.4}
97.6
±
0.2
97.6{\scriptstyle\,\pm\,0.2}
1.0
×
\times
14.4
GPTQ
W4A16
91.2
±
0.3
91.2{\scriptstyle\,\pm\,0.3}
90.6
±
0.6
90.6{\scriptstyle\,\pm\,0.6}
90.9
±
0.4
90.9{\scriptstyle\,\pm\,0.4}
1.2
×
\times
95.6
±
0.5
95.6{\scriptstyle\,\pm\,0.5}
97.5
±
0.4
97.5{\scriptstyle\,\pm\,0.4}
95.8
±
0.5
95.8{\scriptstyle\,\pm\,0.5}
95.1
±
0.6
95.1{\scriptstyle\,\pm\,0.6}
96.0
±
0.4
96.0{\scriptstyle\,\pm\,0.4}
1.3
×
\times
5.5
SmoothQuant
W8A8
91.6
±
0.3
91.6{\scriptstyle\,\pm\,0.3}
91.0
±
0.4
91.0{\scriptstyle\,\pm\,0.4}
91.3
±
0.3
91.3{\scriptstyle\,\pm\,0.3}
1.4
×
\times
96.6
±
0.4
96.6{\scriptstyle\,\pm\,0.4}
97.9
±
0.3
97.9{\scriptstyle\,\pm\,0.3}
96.3
±
0.4
96.3{\scriptstyle\,\pm\,0.4}
94.8
±
0.5
94.8{\scriptstyle\,\pm\,0.5}
96.4
±
0.3
96.4{\scriptstyle\,\pm\,0.3}
1.5
×
\times
7.2
SVDQuant
W4A4
63.8
±
0.7
63.8{\scriptstyle\,\pm\,0.7}
58.4
±
1.0
58.4{\scriptstyle\,\pm\,1.0}
61.1
±
0.8
61.1{\scriptstyle\,\pm\,0.8}
1.6
×
\times
73.0
±
0.8
73.0{\scriptstyle\,\pm\,0.8}
75.1
±
0.9
75.1{\scriptstyle\,\pm\,0.9}
74.4
±
0.9
74.4{\scriptstyle\,\pm\,0.9}
72.3
±
1.0
72.3{\scriptstyle\,\pm\,1.0}
73.7
±
0.7
73.7{\scriptstyle\,\pm\,0.7}
1.7
×
\times
3.6
SVDQuant
∗
\textbf{SVDQuant}^{*}
W4A4
66.3
±
0.7
66.3{\scriptstyle\,\pm\,0.7}
65.5
±
0.8
65.5{\scriptstyle\,\pm\,0.8}
65.9
±
0.7
65.9{\scriptstyle\,\pm\,0.7}
1.5
×
\times
75.5
±
0.7
75.5{\scriptstyle\,\pm\,0.7}
76.6
±
0.8
76.6{\scriptstyle\,\pm\,0.8}
75.7
±
0.8
75.7{\scriptstyle\,\pm\,0.8}
73.2
±
0.9
73.2{\scriptstyle\,\pm\,0.9}
75.3
±
0.6
75.3{\scriptstyle\,\pm\,0.6}
1.5
×
\times
4.2
Atom
W4A4
71.5
±
0.7
71.5{\scriptstyle\,\pm\,0.7}
71.9
±
0.8
71.9{\scriptstyle\,\pm\,0.8}
71.7
±
0.6
71.7{\scriptstyle\,\pm\,0.6}
1.6
×
\times
75.5
±
0.7
75.5{\scriptstyle\,\pm\,0.7}
77.6
±
0.7
77.6{\scriptstyle\,\pm\,0.7}
76.3
±
0.8
76.3{\scriptstyle\,\pm\,0.8}
75.2
±
0.9
75.2{\scriptstyle\,\pm\,0.9}
76.2
±
0.7
76.2{\scriptstyle\,\pm\,0.7}
1.6
×
\times
3.8
Atom
∗
\textbf{Atom}^{*}
W4A4
78.0
±
0.6
78.0{\scriptstyle\,\pm\,0.6}
76.4
±
0.8
76.4{\scriptstyle\,\pm\,0.8}
77.2
±
0.6
77.2{\scriptstyle\,\pm\,0.6}
1.5
×
\times
81.7
±
0.6
81.7{\scriptstyle\,\pm\,0.6}
84.4
±
0.7
84.4{\scriptstyle\,\pm\,0.7}
82.0
±
0.7
82.0{\scriptstyle\,\pm\,0.7}
80.1
±
0.8
80.1{\scriptstyle\,\pm\,0.8}
82.1
±
0.5
82.1{\scriptstyle\,\pm\,0.5}
1.6
×
\times
4.2
QuantWAMs
W4A4
91.8
±
0.2
\mathbf{91.8}{\scriptstyle\,\pm\,\mathbf{0.2}}
91.6
±
0.5
\mathbf{91.6}{\scriptstyle\,\pm\,\mathbf{0.5}}
91.7
±
0.3
\mathbf{91.7}{\scriptstyle\,\pm\,\mathbf{0.3}}
1.4
×
\times
98.0
±
0.3
\mathbf{98.0}{\scriptstyle\,\pm\,\mathbf{0.3}}
99.6
±
0.2
\mathbf{99.6}{\scriptstyle\,\pm\,\mathbf{0.2}}
96.8
±
0.3
\mathbf{96.8}{\scriptstyle\,\pm\,\mathbf{0.3}}
95.0
±
0.4
\mathbf{95.0}{\scriptstyle\,\pm\,\mathbf{0.4}}
97.4
±
0.2
\mathbf{97.4}{\scriptstyle\,\pm\,\mathbf{0.2}}
1.6
×
\times
4.2
Table 2
:
RoboTwin 2.0 and LIBERO benchmark quantization results on LingBot-VA.
Since the officially released LingBot-VA weights support only
LIBERO-Long, results on the other LIBERO suites are not reported.
Method
Precision
RoboTwin 2.0
Speedup
↑
\uparrow
LIBERO
Speedup
↑
\uparrow
Mem. (GB)
↓
\downarrow
Clean
↑
\uparrow
Random
↑
\uparrow
Average
↑
\uparrow
Goal
↑
\uparrow
Spatial
↑
\uparrow
Object
↑
\uparrow
Long
↑
\uparrow
Average
↑
\uparrow
Full Precision
FP16
92.9
±
0.3
92.9{\scriptstyle\,\pm\,0.3}
91.6
±
0.5
91.6{\scriptstyle\,\pm\,0.5}
92.3
±
0.3
92.3{\scriptstyle\,\pm\,0.3}
1.0
×
\times
N/A
N/A
N/A
98.5
±
0.2
98.5{\scriptstyle\,\pm\,0.2}
98.5
±
0.2
98.5{\scriptstyle\,\pm\,0.2}
1.0
×
\times
13.5
GPTQ
W4A16
90.9
±
0.4
90.9{\scriptstyle\,\pm\,0.4}
90.3
±
0.6
90.3{\scriptstyle\,\pm\,0.6}
90.6
±
0.4
90.6{\scriptstyle\,\pm\,0.4}
1.3
×
\times
N/A
N/A
N/A
97.0
±
0.3
97.0{\scriptstyle\,\pm\,0.3}
97.0
±
0.3
97.0{\scriptstyle\,\pm\,0.3}
1.4
×
\times
5.6
SmoothQuant
W8A8
91.5
±
0.3
91.5{\scriptstyle\,\pm\,0.3}
90.9
±
0.5
90.9{\scriptstyle\,\pm\,0.5}
91.2
±
0.3
91.2{\scriptstyle\,\pm\,0.3}
1.5
×
\times
N/A
N/A
N/A
97.5
±
0.3
97.5{\scriptstyle\,\pm\,0.3}
97.5
±
0.3
97.5{\scriptstyle\,\pm\,0.3}
1.6
×
\times
6.8
SVDQuant
W4A4
65.8
±
0.8
65.8{\scriptstyle\,\pm\,0.8}
64.0
±
1.0
64.0{\scriptstyle\,\pm\,1.0}
64.9
±
0.8
64.9{\scriptstyle\,\pm\,0.8}
1.5
×
\times
N/A
N/A
N/A
73.8
±
0.9
73.8{\scriptstyle\,\pm\,0.9}
73.8
±
0.9
73.8{\scriptstyle\,\pm\,0.9}
1.6
×
\times
3.4
SVDQuant
∗
\textbf{SVDQuant}^{*}
W4A4
69.9
±
0.7
69.9{\scriptstyle\,\pm\,0.7}
68.5
±
0.9
68.5{\scriptstyle\,\pm\,0.9}
69.2
±
0.7
69.2{\scriptstyle\,\pm\,0.7}
1.4
×
\times
N/A
N/A
N/A
79.6
±
0.8
79.6{\scriptstyle\,\pm\,0.8}
79.6
±
0.8
79.6{\scriptstyle\,\pm\,0.8}
1.5
×
\times
3.9
Atom
W4A4
73.0
±
0.7
73.0{\scriptstyle\,\pm\,0.7}
72.8
±
0.9
72.8{\scriptstyle\,\pm\,0.9}
72.9
±
0.7
72.9{\scriptstyle\,\pm\,0.7}
1.6
×
\times
N/A
N/A
N/A
76.9
±
0.8
76.9{\scriptstyle\,\pm\,0.8}
76.9
±
0.8
76.9{\scriptstyle\,\pm\,0.8}
1.7
×
\times
3.6
Atom
∗
\textbf{Atom}^{*}
W4A4
77.0
±
0.6
77.0{\scriptstyle\,\pm\,0.6}
75.5
±
0.8
75.5{\scriptstyle\,\pm\,0.8}
76.3
±
0.6
76.3{\scriptstyle\,\pm\,0.6}
1.4
×
\times
N/A
N/A
N/A
81.5
±
0.7
81.5{\scriptstyle\,\pm\,0.7}
81.5
±
0.7
81.5{\scriptstyle\,\pm\,0.7}
1.5
×
\times
3.9
QuantWAMs
W4A4
92.3
±
0.3
\mathbf{92.3}{\scriptstyle\,\pm\,\mathbf{0.3}}
90.9
±
0.5
\mathbf{90.9}{\scriptstyle\,\pm\,\mathbf{0.5}}
91.6
±
0.3
\mathbf{91.6}{\scriptstyle\,\pm\,\mathbf{0.3}}
1.4
×
\times
N/A
N/A
N/A
98.0
±
0.3
\mathbf{98.0}{\scriptstyle\,\pm\,\mathbf{0.3}}
98.0
±
0.3
\mathbf{98.0}{\scriptstyle\,\pm\,\mathbf{0.3}}
1.6
×
\times
3.9
