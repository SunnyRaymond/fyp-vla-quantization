# DriveCache: Action-Aware Caching for Driving World Model Inference

paper_id: arxiv:2608.16354v1
tier: H
source_used: html_arxiv
warning: none

## Intro

Driving video generation models predict how traffic scenes evolve under different ego actions, providing controllable future observations for simulation, policy training, planning evaluation, and offline data generation
(
Hu et al. 2023
;
Wang et al. 2024a
;
Gao et al. 2024b
)
. These action-conditioned predictions let developers and planning systems examine possible futures under candidate plans. They also support diverse scenario generation for development. Recent video diffusion models improve visual fidelity and temporal coherence through larger spatiotemporal backbones and longer generation horizons
(
Ho et al. 2022
;
Blattmann et al. 2023
;
Yang et al. 2025
;
Kong et al. 2024
;
Lin et al. 2024
)
.
These advances increase inference cost. Iterative denoising evaluates the backbone across many sampling steps, and the cost grows with model capacity, resolution, and video length. Online prediction must fit within the planning cycle, while offline generation must scale to large scenario collections. Inference latency therefore constrains online prediction and offline scenario generation.
Diffusion acceleration methods reduce the number or cost of denoising evaluations. Distillation, quantization, pruning, and efficient operators can require new weights, retraining, calibration data, or specialized kernels
(
Salimans and Ho 2022
;
Song et al. 2023
)
. Feature caching leaves the generator unchanged and reuses intermediate features across neighboring denoising steps. Existing cache controllers derive reuse from fixed schedules or signals observed after denoising begins
(
Liu et al. 2025a
;
Kahatapitiya et al. 2025
;
Zhou et al. 2025
;
Ma et al. 2025
;
Bu et al. 2025
;
Liu et al. 2025b
;
Chung et al. 2026
)
. These general-purpose methods omit driving signals available before generation, including ego speed and planned trajectories.
Figure 1:
DriveCache achieves better video fidelity and consistency across evaluation metrics.
Figure 2:
Motivation of DriveCache. Planned ego motion reveals scene-dependent cache tolerance before denoising.
However, driving generation provides planned ego motion before denoising. Offline generation receives recorded or specified future ego poses, while planner-conditioned generation receives predicted poses from the planning stack. Stationary, straight, and turning plans induce different viewpoint changes. Figure
2
shows that planned ego motion predicts scene-dependent cache tolerance before the denoising pass begins.
Planned motion is available before the first denoising step, but it cannot determine a schedule by itself. Cache error also varies with denoising position, cache age, model architecture, and generated content. A driving-aware controller must therefore combine a scene-level motion prior with a denoising-level response model, while retaining a causal correction for states that depart from calibration.
DriveCache converts this prior into a complete schedule. One low-motion anchor and one moving-turn anchor measure terminal responses of consecutive reuse; planned translation and rotation interpolate them for each scene. Exact dynamic programming selects reuse quantity and positions under one budget. A pre-reuse drift veto rejects out-of-support decisions and replans the unexecuted suffix.
Our main contributions are threefold.
•
To the best of our knowledge, this work is the first to identify and validate planned ego motion as a pre-generation signal for diffusion caching in driving video generation, showing that ego translation and rotation predict scene-level cache tolerance across driving motions.
•
We propose DriveCache, a training-free cache controller that assigns scene-level reuse budgets from planned motion, models consecutive-reuse error, and uses exact dynamic programming with causal drift correction to place reuse across denoising steps.
•
Across multiple driving video generators, DriveCache improves quality–efficiency trade-offs over cache baselines. At approximately
2
×
2\times
speedup on Wan2.2 A14B, it improves PSNR by 2.036 dB over TeaCache.

## Method

DriveCache formulates caching as a causal decision made before each backbone evaluation. Consider a frozen video diffusion model with
K
K
denoising steps. At step
k
k
, Equation (
1
) separates the expensive reusable backbone from the inexpensive output interface:
r
k
=
F
k
​
(
u
k
)
,
ϵ
k
=
H
k
​
(
x
k
,
r
k
)
,
r_{k}=F_{k}(u_{k}),\qquad\epsilon_{k}=H_{k}(x_{k},r_{k}),
(1)
where
x
k
x_{k}
is the current latent,
u
k
u_{k}
is the backbone input, and
r
k
r_{k}
is the cached backbone output. A full decision evaluates
F
k
F_{k}
; a reuse decision replaces
r
k
r_{k}
with its most recently computed value. The first step always uses full computation. The controller observes
u
k
u_{k}
before executing
F
k
F_{k}
, so it can make the veto decision without first executing the backbone call.
DriveCache has four stages (Figure
4
). Two ego-motion anchors measure joint run responses; planned ego motion interpolates them for the current scene; exact dynamic programming jointly selects reuse quantity and placement; and a pre-reuse drift check can refresh and replan the unexecuted suffix while preserving the executed prefix.
The ablation study in Table
3
supports planned ego motion as a cache prior. Cache tolerance varies because planned ego motion changes the rendered viewpoint.
Scene-agnostic scheduling, shuffled trajectories, and translation-matched turns separate planned-motion allocation from dataset correlation and fixed step preference. Figure
3
shows that planned ego motion separates run tolerance while local input drift remains nearly unchanged. Planned ego motion remains a scene coordinate. Denoising position and cache age determine placement cost, while support checks and the causal veto handle departures from the calibrated regime.
Figure 3:
Planned ego motion changes joint run responses even when local input drift remains nearly unchanged.
The search enforces the mandatory first evaluation, maximum cache age, backbone boundaries, and native no-cache steps. It maximizes skipped evaluations under one cumulative response budget, with deterministic lower-cost tie breaking. Planned ego motion changes allocation before denoising, run costs place reuse jointly, and a veto corrects the next decision before a cached backbone output changes the latent. The executed prefix is never altered.
Figure 4:
DriveCache combines planned ego motion, two-anchor run calibration, exact DP, and a causal reuse guard.
Planned ego motion calibrates run costs
Let a scene provide planned ego poses
(
p
t
,
ψ
t
)
(p_{t},\psi_{t})
over the generated horizon. Equation (
2
) summarizes total translation and accumulated rotation:
L
s
=
∑
t
∥
p
t
−
p
t
−
1
∥
2
,
Θ
s
=
∑
t
|
wrap
⁡
(
ψ
t
−
ψ
t
−
1
)
|
.
L_{s}=\sum_{t}\lVert p_{t}-p_{t-1}\rVert_{2},\qquad\Theta_{s}=\sum_{t}\left|\operatorname{wrap}(\psi_{t}-\psi_{t-1})\right|.
(2)
The statistics remain separate because equal travel distance can induce different cached-output changes under straight and turning motion.
We z-score
L
L
and
Θ
\Theta
using their calibration means and standard deviations. The low-motion anchor
(
L
0
,
Θ
0
)
(L_{0},\Theta_{0})
minimizes the sum of these standardized values. Among clips whose raw
Θ
\Theta
exceeds its calibration median and whose raw
L
>
L
0
L>L_{0}
and
Θ
>
Θ
0
\Theta>\Theta_{0}
, the moving-turn anchor
(
L
1
,
Θ
1
)
(L_{1},\Theta_{1})
maximizes the product of the standardized values. Clip index breaks ties, and an empty candidate set triggers full-inference fallback. Equation (
3
) expresses a scene relative to this support:
z
s
L
=
L
s
−
L
0
max
⁡
(
L
1
−
L
0
,
ε
)
,
z
s
Θ
=
Θ
s
−
Θ
0
max
⁡
(
Θ
1
−
Θ
0
,
ε
)
.
z_{s}^{L}=\frac{L_{s}-L_{0}}{\max(L_{1}-L_{0},\varepsilon)},\qquad z_{s}^{\Theta}=\frac{\Theta_{s}-\Theta_{0}}{\max(\Theta_{1}-\Theta_{0},\varepsilon)}.
(3)
The anchor pair maps low motion toward the origin and the moving turn toward the upper corner without fitting scene-specific coefficients. A scene lies inside the calibration support only when
(
z
s
L
,
z
s
Θ
)
∈
[
0
,
1
]
2
(z_{s}^{L},z_{s}^{\Theta})\in[0,1]^{2}
; otherwise DriveCache uses full inference. Equation (
4
) then reduces the supported pair to one interpolation coordinate:
D
s
=
1
2
​
(
z
s
L
)
2
+
(
z
s
Θ
)
2
.
D_{s}=\frac{1}{\sqrt{2}}\sqrt{(z_{s}^{L})^{2}+(z_{s}^{\Theta})^{2}}.
(4)
The coordinatewise gate prevents radial compression from hiding an unsupported dimension.
The anchor traces also define how DriveCache scores a consecutive reuse run. For each anchor
i
∈
{
0
,
1
}
i\in\{0,1\}
, one full denoising trace stores
(
x
k
(
i
)
,
u
k
(
i
)
,
r
k
(
i
)
)
(x_{k}^{(i)},u_{k}^{(i)},r_{k}^{(i)})
. A candidate run starts after a full refresh at step
j
j
and reuses
r
j
(
i
)
r_{j}^{(i)}
for
h
h
steps. Equation (
5
) measures how far the current backbone input has moved from the refresh input at the
ℓ
\ell
-th reuse:
d
j
,
ℓ
(
i
)
=
∥
u
j
+
ℓ
(
i
)
−
u
j
(
i
)
∥
2
∥
u
j
+
ℓ
(
i
)
∥
2
+
ε
.
d_{j,\ell}^{(i)}=\frac{\lVert u_{j+\ell}^{(i)}-u_{j}^{(i)}\rVert_{2}}{\lVert u_{j+\ell}^{(i)}\rVert_{2}+\varepsilon}.
(5)
Input drift is available before the expensive backbone call and is therefore suitable for the runtime veto. It does not by itself measure the denoising error caused by substituting a cached backbone output. Equation (
6
) defines that signed output perturbation against the full trace:
e
j
,
ℓ
(
i
)
=
H
j
+
ℓ
​
(
x
j
+
ℓ
(
i
)
,
r
j
(
i
)
)
−
H
j
+
ℓ
​
(
x
j
+
ℓ
(
i
)
,
r
j
+
ℓ
(
i
)
)
.
e_{j,\ell}^{(i)}=H_{j+\ell}(x_{j+\ell}^{(i)},r_{j}^{(i)})-H_{j+\ell}(x_{j+\ell}^{(i)},r_{j+\ell}^{(i)}).
(6)
Frozen-sampler Jacobian-vector products propagate the whole run to the terminal latent. Let
P
t
(
i
)
P_{t}^{(i)}
be the Jacobian of terminal latent
x
K
x_{K}
with respect to denoiser output
ϵ
t
\epsilon_{t}
, evaluated along anchor
i
i
’s frozen full trace. Its product with
e
t
(
i
)
e_{t}^{(i)}
gives the first-order terminal perturbation. Equation (
7
) combines these signed perturbations into joint run response
q
j
,
h
(
i
)
q_{j,h}^{(i)}
:
G
j
,
h
(
i
)
=
∑
ℓ
=
1
h
P
j
+
ℓ
(
i
)
​
e
j
,
ℓ
(
i
)
,
q
j
,
h
(
i
)
=
∥
G
j
,
h
(
i
)
∥
2
∥
x
K
(
i
)
∥
2
+
ε
.
G_{j,h}^{(i)}=\sum_{\ell=1}^{h}P_{j+\ell}^{(i)}e_{j,\ell}^{(i)},\qquad q_{j,h}^{(i)}=\frac{\lVert G_{j,h}^{(i)}\rVert_{2}}{\lVert x_{K}^{(i)}\rVert_{2}+\varepsilon}.
(7)
Normalizing by terminal-latent energy makes responses comparable across steps without a learned evaluator. Replay isolates cache perturbation, while the signed vector preserves cross-step cancellation and amplification.
For scene
s
s
, Equation (
8
) interpolates only the nonnegative planned-motion-dependent difference:
q
~
s
,
j
,
h
=
q
j
,
h
(
0
)
+
D
s
​
max
⁡
(
q
j
,
h
(
1
)
−
q
j
,
h
(
0
)
,
0
)
.
\widetilde{q}_{s,j,h}=q_{j,h}^{(0)}+D_{s}\max\!\left(q_{j,h}^{(1)}-q_{j,h}^{(0)},0\right).
(8)
This one-sided interpolation preserves the low-motion response when the moving anchor is easier and enforces a monotone planned-motion risk relation. Equation (
9
) converts joint run response
q
~
\widetilde{q}
into monotone envelope
Q
Q
and incremental run cost
c
^
\widehat{c}
:
Q
s
,
j
,
h
\displaystyle Q_{s,j,h}
=
max
1
≤
ℓ
≤
h
⁡
q
~
s
,
j
,
ℓ
,
\displaystyle=\max_{1\leq\ell\leq h}\widetilde{q}_{s,j,\ell},
(9)
c
^
s
,
j
,
h
\displaystyle\widehat{c}_{s,j,h}
=
Q
s
,
j
,
h
−
Q
s
,
j
,
h
−
1
,
Q
s
,
j
,
0
=
0
.
\displaystyle=Q_{s,j,h}-Q_{s,j,h-1},\qquad Q_{s,j,0}=0.
Planned ego motion changes how much reuse a scene can tolerate, while
(
j
,
h
)
(j,h)
captures denoising position and cache age. We obtain drift threshold
d
¯
s
,
j
,
h
\bar{d}_{s,j,h}
with the same one-sided anchor interpolation as Equation (
8
), replacing
q
q
with
d
d
.
The envelope keeps longer-run increments nonnegative; refresh resets age but not response already propagated into the latent.
Exact scheduling supports causal correction
DriveCache converts the run costs into a legal schedule under response budget
τ
\tau
. For target reuse fraction
ρ
tar
\rho_{\mathrm{tar}}
, calibration chooses the smallest table-cost threshold whose median schedule reaches
⌈
ρ
tar
​
(
K
−
1
)
⌉
\lceil\rho_{\mathrm{tar}}(K-1)\rceil
reuses; neither PSNR nor evaluation clips enter this choice. Since legal reuses skip the same interface, maximizing their count minimizes backbone evaluations. Let
J
k
​
(
n
,
h
)
J_{k}(n,h)
be minimum cumulative response after
k
k
steps,
n
n
reuses, and cache age
h
h
. We set
J
1
​
(
0
,
0
)
=
0
J_{1}(0,0)=0
and all other states to
+
∞
+\infty
, with
0
≤
n
≤
k
−
1
0\leq n\leq k-1
and
0
≤
h
≤
H
max
0\leq h\leq H_{\max}
. Here,
ℒ
⁡
(
k
,
h
)
=
1
\mathcal{L}(k,h)=1
marks reuse that satisfies the model-specific legality constraints:
J
k
+
1
​
(
n
,
0
)
\displaystyle J_{k+1}(n,0)
=
min
h
⁡
J
k
​
(
n
,
h
)
,
\displaystyle=\min_{h}J_{k}(n,h),
(10)
J
k
+
1
​
(
n
+
1
,
h
+
1
)
\displaystyle J_{k+1}(n+1,h+1)
=
min
{
J
k
+
1
(
n
+
1
,
h
+
1
)
,
\displaystyle=\min\!\bigl\{J_{k+1}(n+1,h+1),
J
k
(
n
,
h
)
+
c
^
s
,
k
−
h
,
h
+
1
}
,
\displaystyle J_{k}(n,h)+\widehat{c}_{s,k-h,h+1}\bigr\},
ℒ
⁡
(
k
,
h
+
1
)
=
1
.
\displaystyle\mathcal{L}(k,h+1)=1.
A full evaluation pays no new reuse cost and resets cache age. A reuse extends the run whose last full evaluation occurred at
j
=
k
−
h
j=k-h
and adds the calibrated increment only when the next age is legal. Illegal transitions receive
+
∞
+\infty
. Equation (
11
) selects the maximum feasible reuse count:
B
s
∗
=
max
⁡
{
n
:
min
h
⁡
J
K
​
(
n
,
h
)
≤
τ
}
.
B_{s}^{*}=\max\left\{n:\min_{h}J_{K}(n,h)\leq\tau\right\}.
(11)
Backtracking recovers the minimum-response schedule at
B
s
∗
B_{s}^{*}
. The recurrence exactly optimizes the calibrated response objective in
O
⁡
(
K
3
)
O(K^{3})
time and
O
⁡
(
K
2
)
O(K^{2})
rolling memory.
The schedule remains causal during inference. We initialize accumulated charge as
C
1
=
0
C_{1}=0
. Before the
h
h
-th reuse in a run whose last full evaluation occurred at
j
j
, DriveCache measures the observed input drift in Equation (
12
):
d
j
,
h
obs
=
∥
u
j
+
h
−
u
j
∥
2
∥
u
j
+
h
∥
2
+
ε
.
d_{j,h}^{\mathrm{obs}}=\frac{\lVert u_{j+h}-u_{j}\rVert_{2}}{\lVert u_{j+h}\rVert_{2}+\varepsilon}.
(12)
It accepts reuse only when
d
j
,
h
obs
≤
d
¯
s
,
j
,
h
+
ε
d_{j,h}^{\mathrm{obs}}\leq\bar{d}_{s,j,h}+\varepsilon
and accumulated charge satisfies
C
k
+
c
^
s
,
j
,
h
≤
τ
C_{k}+\widehat{c}_{s,j,h}\leq\tau
. Acceptance sets
C
k
+
1
=
C
k
+
c
^
s
,
j
,
h
C_{k+1}=C_{k}+\widehat{c}_{s,j,h}
. A failed check evaluates the backbone, keeps
C
k
+
1
=
C
k
C_{k+1}=C_{k}
, and refreshes the cache. Replanning starts from age zero after this forced evaluation and maximizes additional suffix reuse under budget
τ
−
C
k
\tau-C_{k}
; the executed prefix remains fixed. Calibration uses no optimizer or parameter update, and inference leaves generator weights and sampling steps unchanged at runtime.
DriveCache applies model-specific legality to the same recurrence. Here,
u
k
u_{k}
,
r
k
r_{k}
, and
H
k
H_{k}
are the reusable-region input, cached backbone output, and remaining sampler interface. Wan2.2 separates guidance branches; A14B also forbids cross-expert reuse. Epona caches only within each 100-step visual denoising call, never across autoregressive segments. Cache tensors, legal steps, maximum age, norm axes, and ties are frozen before evaluation.
Two full anchor traces support all candidate runs. Building the
(
j
,
h
)
(j,h)
table costs
O
⁡
(
K
2
)
O(K^{2})
substitutions and
O
⁡
(
K
3
)
O(K^{3})
unbatched Jacobian-vector products once per configuration, without evaluation clips. On one H20, calibration takes 4 minutes for 5B, 9 minutes for A14B, and 16 minutes for Epona, with at most 6.1 GB additional memory and response tables below 2 MB. Initial DP solves take 0.8, 1.4, and 9.2 ms, respectively; drift checks and suffix replanning add less than 0.7% to end-to-end latency.
DriveCache updates no weights and trains no predictor; one model-level calibration applies across clips.
