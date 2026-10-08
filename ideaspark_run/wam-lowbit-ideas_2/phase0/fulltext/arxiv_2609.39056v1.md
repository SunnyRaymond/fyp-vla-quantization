# SteerQuant: Steering Quantization Error with Action-Guided Scaling in World-Action Models

paper_id: arxiv:2609.39056v1
tier: H
source_used: html_arxiv
warning: none

## Intro

World–action models (WAMs) build on pretrained video models
to jointly model future world states and actions, using
spatiotemporal priors learned from large-scale video
data
(
Kim et al., 2026
;
NVIDIA, 2026
;
Ye et al., 2026
;
Yuan et al., 2026
)
.
During inference, diffusion Transformers
(
Peebles and Xie, 2023
)
repeatedly process
heterogeneous semantic streams, including visual representations,
proprioceptive states, and actions, across denoising steps.
This repeated computation and data movement make inference costly.
Post-training quantization (PTQ) reduces these costs
(
Frantar et al., 2023
;
Li et al., 2023
;
Shang et al., 2023
;
Hu et al., 2026
)
by lowering
weight and activation precision, reducing memory traffic and
enabling hardware-accelerated low-precision arithmetic (e.g., INT4 and NVFP4).
To preserve accuracy, existing PTQ methods typically
control outlier magnitudes to reduce numerical
error
(
Xiao et al., 2023
;
Ashkboos et al., 2024
;
Li et al., 2025
;
Liu et al., 2025
)
.
However, WAMs use shared weights to process heterogeneous
semantic streams, so a single weight quantization choice
affects multiple streams at once.
The resulting errors propagate through stream interactions
and subsequent denoising steps, potentially producing
different final-action deviations even at similar magnitudes.
Effective WAM quantization must therefore balance errors
across streams according to their impact on final actions.
Achieving this balance requires identifying which streams’
errors most affect final actions.
We observe that comparable numerical errors across semantic
streams can lead to markedly different final-action deviations
and task success rates (see Sec.
3
).
A naive method is to calibrate weight–activation scaling
independently for each stream.
However, independently chosen scales generally require
different quantized weight matrices, sacrificing weight sharing.
Moreover, a stream whose errors strongly affect final actions
at one denoising step may have less influence at another,
limiting the effectiveness of fixed per-stream choices.
The core challenge is therefore to adapt error trade-offs
across streams as their action impact changes, while retaining
one static quantized weight matrix per layer and fixed bit-widths.
To address this challenge, we propose
SteerQuant
,
a 4-bit quantization framework for WAMs that steers errors
toward computations with less influence on final actions.
It coordinates shared channel scaling and stream-specific
activation scaling across denoising steps while retaining
shared weights and fixed 4-bit.
To reduce the extra kernel launches and memory traffic
introduced by separate scaling operations, we fuse these
transformations into low-bit kernels.
Our contributions are summarized as follows:
•
We reveal that heterogeneous semantic streams differ
in their tolerance to quantization: comparable numerical
errors can produce markedly different action deviations
and task success rates. This tolerance varies across layers
and denoising steps, motivating calibration guided by
downstream action impact.
•
We propose
SteerQuant
, which uses an Action-Impact
Stream Map to guide shared channel scaling and stream-specific
activation calibration across denoising steps. These two stages
prioritize computations with greater action impact while
retaining one static quantized weight matrix per layer
and fixed bit-widths.
•
We develop
Rudder
, a 4-bit inference engine
for WAMs that integrates calibrated scaling and output
compensation into fused kernels. By reducing extra kernel
launches and intermediate memory traffic, Rudder enables
efficient execution of SteerQuant while preserving
the computational and memory benefits of quantization.
•
We evaluate three WAMs under W4A8 and W4A4 on LIBERO
and RoboLab, preserving near-BF16 LIBERO success
and outperforming evaluated same-precision RoboLab baselines.
We achieve up to 2.23
×
\times
denoising speedup over BF16.
Real dual-arm deployment delivers 1.35
×
\times
end-to-end inference speedup while maintaining average
task success.

## Method

As illustrated in Figure
2
, SteerQuant first
constructs an
Action-Impact Stream Map
that estimates how
local quantization errors affect final actions.
Guided by this map,
Shared-Weight Error Routing
calibrates
shared channel scaling, followed by
Stream-Specific Activation Modulation
, which refines
activation quantization across streams and denoising steps
while keeping the quantized weights fixed.
The inference engine
Rudder
fuses scaling and output
compensation into quantization and GEMM kernels for efficient
execution.
Figure 2:
Overview of SteerQuant.
Γ
ℓ
,
τ
\Gamma_{\ell,\tau}
is a diagonal matrix of token-wise stream gains. The upper-right gray box depicts offline weight preparation; the other gray boxes denote fused inference kernels.
4.1
Action-Impact Stream Map
To identify where quantization errors most affect final actions,
we score each layer–step–stream region by its impact on the
action output.
Before calibrating the scaling factors, we compute the local
quantization error at the target bit-widths:
𝐄
ℓ
,
τ
,
s
=
𝐘
^
ℓ
,
τ
,
s
−
𝐘
ℓ
,
τ
,
s
.
\mathbf{E}_{\ell,\tau,s}=\widehat{\mathbf{Y}}_{\ell,\tau,s}-\mathbf{Y}_{\ell,\tau,s}.
(4)
Here,
𝐘
^
ℓ
,
τ
,
s
\widehat{\mathbf{Y}}_{\ell,\tau,s}
is the output for
stream
s
s
after quantizing the weights and input activations
of linear layer
ℓ
\ell
at step
τ
\tau
.
The input is taken from a full-precision forward pass
to exclude upstream quantization errors.
Injecting each stream’s error and rerunning the remaining
full-precision computation requires
𝒪
⁡
(
|
𝒟
|
​
R
)
\mathcal{O}(|\mathcal{D}|R)
downstream forward evaluations
for
R
R
regions and calibration set
𝒟
\mathcal{D}
.
Instead, we use derivatives of the full-precision model
to estimate the resulting action changes.
Let
𝐚
∈
ℝ
m
\mathbf{a}\in\mathbb{R}^{m}
denote the vectorized final action
chunk, standardized using fixed, stabilized per-dimension
standard deviations.
Its Jacobian
𝐉
ℓ
,
τ
,
s
\mathbf{J}_{\ell,\tau,s}
with respect to
vec
⁡
(
𝐘
ℓ
,
τ
,
s
)
\operatorname{vec}(\mathbf{Y}_{\ell,\tau,s})
gives the
first-order action change
𝐉
ℓ
,
τ
,
s
​
vec
⁡
(
𝐄
ℓ
,
τ
,
s
)
\mathbf{J}_{\ell,\tau,s}\operatorname{vec}(\mathbf{E}_{\ell,\tau,s})
.
We define the action-impact score as
S
ℓ
,
τ
,
s
=
[
1
m
​
𝔼
𝒟
​
[
‖
𝐉
ℓ
,
τ
,
s
​
vec
⁡
(
𝐄
ℓ
,
τ
,
s
)
‖
2
2
]
]
1
/
2
.
S_{\ell,\tau,s}=\left[\frac{1}{m}\mathbb{E}_{\mathcal{D}}\left[\left\|\mathbf{J}_{\ell,\tau,s}\operatorname{vec}(\mathbf{E}_{\ell,\tau,s})\right\|_{2}^{2}\right]\right]^{1/2}.
(5)
Here,
𝔼
𝒟
\mathbb{E}_{\mathcal{D}}
averages over calibration samples.
The score measures the RMS first-order action displacement;
larger values indicate greater estimated impact on final actions.
Forming these Jacobians requires
m
m
coordinate
vector–Jacobian products (VJPs) per sample and stores
m
m
entries for every local output element.
We instead estimate the squared scores using
P
P
Rademacher
projections
(
Hutchinson, 1989
)
.
Each projected reverse pass provides VJPs at all captured
linear outputs, reusable across their stream error matrices.
This yields scores for all
R
R
regions with
𝒪
⁡
(
|
𝒟
|
​
P
)
\mathcal{O}(|\mathcal{D}|P)
reverse passes, without storing
dense Jacobians.
The map remains fixed during subsequent calibration.
Appendix
B.2
derives the unbiased
squared-score estimator and analyzes its cost and approximation
error; Appendices
B.3
and
B.4
examine the map’s structure
and stability.
4.2
Shared-Weight Error Routing
Within each layer, semantic streams and denoising steps share
the same weights but differ in how their errors affect final
actions. A scaling that reduces reconstruction error in one
region may increase it in another.
We therefore use the Action-Impact Stream Map to choose a shared
channel scaling that prioritizes regions with greater action impact.
For each linear layer, a positive diagonal matrix
𝐃
ℓ
\mathbf{D}_{\ell}
scales the weights and inversely scales the input activations:
𝐘
^
ℓ
,
τ
,
s
​
(
𝐃
ℓ
)
=
𝒬
⁡
(
𝐗
ℓ
,
τ
,
s
​
𝐃
ℓ
−
1
)
​
𝒬
​
(
𝐃
ℓ
​
𝐖
ℓ
)
.
\widehat{\mathbf{Y}}_{\ell,\tau,s}(\mathbf{D}_{\ell})=\mathcal{Q}\!\left(\mathbf{X}_{\ell,\tau,s}\mathbf{D}_{\ell}^{-1}\right)\mathcal{Q}\!\left(\mathbf{D}_{\ell}\mathbf{W}_{\ell}\right).
(6)
The two scalings cancel in full precision but change the ranges
of weights and activations seen by the quantizer, allowing us
to reshape quantization error at fixed bit-widths.
Using the same
𝐃
ℓ
\mathbf{D}_{\ell}
across streams and denoising
steps retains one quantized weight matrix per layer.
We select this shared scaling by minimizing reconstruction
error weighted by the map-derived importance
ω
ℓ
,
τ
,
s
D
\omega^{D}_{\ell,\tau,s}
(Appendix
C
):
𝐃
ℓ
∗
=
arg
⁡
min
𝐃
ℓ
​
∑
τ
,
s
π
τ
​
ω
ℓ
,
τ
,
s
D
n
s
​
𝔼
𝒟
​
[
‖
𝐘
^
ℓ
,
τ
,
s
​
(
𝐃
ℓ
)
−
𝐘
ℓ
,
τ
,
s
‖
F
2
]
.
\mathbf{D}_{\ell}^{*}=\underset{\mathbf{D}_{\ell}}{\arg\min}\sum_{\tau,s}\frac{\pi_{\tau}\omega^{D}_{\ell,\tau,s}}{n_{s}}\mathbb{E}_{\mathcal{D}}\!\left[\left\|\widehat{\mathbf{Y}}_{\ell,\tau,s}(\mathbf{D}_{\ell})-\mathbf{Y}_{\ell,\tau,s}\right\|_{F}^{2}\right].
(7)
Here,
π
τ
\pi_{\tau}
is the normalized calibration frequency
of step
τ
\tau
, and
n
s
n_{s}
is the number of tokens in stream
s
s
,
so that
1
/
n
s
1/n_{s}
averages the loss over stream tokens.
The map weights penalize errors more strongly in high-impact
regions, guiding the trade-off when regions favor different scalings.
After calibration, we fold
𝐃
ℓ
∗
\mathbf{D}_{\ell}^{*}
into the weights,
yielding
𝐖
^
ℓ
=
𝒬
⁡
(
𝐃
ℓ
∗
​
𝐖
ℓ
)
\widehat{\mathbf{W}}_{\ell}=\mathcal{Q}(\mathbf{D}_{\ell}^{*}\mathbf{W}_{\ell})
.
These weights are stored as packed integers with corresponding
quantization scales and remain fixed during subsequent
activation calibration and inference.
4.3
Stream-Specific Activation Modulation
As discussed in Section
3.2
, the relative
action impact of semantic streams changes across denoising steps.
A shared channel scaling provides one compromise across all steps,
but cannot follow these changing priorities.
We therefore adjust activation quantization while keeping
the weights fixed.
At each layer and step, a shared clipping threshold establishes
a common reference range and base quantization scale.
Stream gains adjust each stream’s effective range relative to
this reference, allowing different streams to receive greater
protection at different steps.
We bound both the threshold and gains to limit extreme range
choices that can cause excessive clipping or coarse quantization.
For the transformed activation
𝐗
~
ℓ
,
τ
,
s
=
𝐗
ℓ
,
τ
,
s
​
(
𝐃
ℓ
∗
)
−
1
\widetilde{\mathbf{X}}_{\ell,\tau,s}=\mathbf{X}_{\ell,\tau,s}(\mathbf{D}_{\ell}^{*})^{-1}
,
the modulated output is
𝐘
^
ℓ
,
τ
,
s
​
(
γ
ℓ
,
τ
,
s
,
c
ℓ
,
τ
)
=
γ
ℓ
,
τ
,
s
−
1
​
𝒬
​
(
γ
ℓ
,
τ
,
s
​
𝐗
~
ℓ
,
τ
,
s
,
c
ℓ
,
τ
)
​
𝐖
^
ℓ
.
\displaystyle\widehat{\mathbf{Y}}_{\ell,\tau,s}(\gamma_{\ell,\tau,s},c_{\ell,\tau})=\gamma_{\ell,\tau,s}^{-1}\mathcal{Q}\!\left(\gamma_{\ell,\tau,s}\widetilde{\mathbf{X}}_{\ell,\tau,s};c_{\ell,\tau}\right)\widehat{\mathbf{W}}_{\ell}.
(8)
The inverse gain compensates for the input scaling at the output.
Using map-derived weights
ω
ℓ
,
τ
,
s
γ
\omega^{\gamma}_{\ell,\tau,s}
,
we jointly calibrate the gain vector and shared threshold:
(
𝜸
ℓ
,
τ
∗
,
c
ℓ
,
τ
∗
)
=
arg
⁡
min
𝜸
,
c
​
∑
s
ω
ℓ
,
τ
,
s
γ
n
s
​
𝔼
𝒟
​
[
‖
𝐘
^
ℓ
,
τ
,
s
​
(
γ
s
,
c
)
−
𝐗
~
ℓ
,
τ
,
s
​
𝐖
^
ℓ
‖
F
2
]
s.t.
c
min
≤
c
≤
c
max
,
γ
min
≤
γ
s
≤
γ
max
∀
s
,
∑
s
n
s
​
log
⁡
γ
s
=
0
.
\begin{gathered}(\bm{\gamma}_{\ell,\tau}^{*},c_{\ell,\tau}^{*})=\underset{\bm{\gamma},\;c}{\arg\min}\sum_{s}\frac{\omega^{\gamma}_{\ell,\tau,s}}{n_{s}}\mathbb{E}_{\mathcal{D}}\!\left[\left\|\widehat{\mathbf{Y}}_{\ell,\tau,s}(\gamma_{s},c)-\widetilde{\mathbf{X}}_{\ell,\tau,s}\widehat{\mathbf{W}}_{\ell}\right\|_{F}^{2}\right]\\[2.84526pt]
\text{s.t.}\quad c_{\min}\leq c\leq c_{\max},\qquad\gamma_{\min}\leq\gamma_{s}\leq\gamma_{\max}\quad\forall s,\\
\sum_{s}n_{s}\log\gamma_{s}=0.\end{gathered}
(9)
All streams share the same positive bounds.
When these constraints require a trade-off across streams,
the map weights favor lower reconstruction errors in those
with greater impact on final actions.
Appendix
C
details the weight construction
and effective quantization ranges.
The calibrated gains and thresholds are stored in a
layer–step lookup table, preserving uniform bit-widths
without online optimization.
4.4
Rudder
Executing channel scaling, stream gains, and output compensation
separately adds kernel launches and intermediate memory traffic
at every denoising step.
Rudder, our 4-bit inference engine for WAMs, reduces this overhead
through two fused kernels while preserving stream-specific
quantization (Figure
3
).
Below, layer and step indices are omitted, and all scaling
parameters denote their calibrated values.
The first kernel applies channel scaling
𝐃
−
1
\mathbf{D}^{-1}
and stream gains during activation quantization.
It writes only integer activations
𝐐
𝐗
\mathbf{Q}_{\mathbf{X}}
and row scales
r
i
=
Δ
𝐗
/
γ
s
⁡
(
i
)
r_{i}=\Delta_{\mathbf{X}}/\gamma_{s(i)}
,
where
s
⁡
(
i
)
s(i)
identifies the stream of row
i
i
and
Δ
𝐗
=
c
/
q
max
\Delta_{\mathbf{X}}=c/q_{\max}
.
The row scale incorporates inverse-gain compensation,
eliminating a separate compensation operation.
The second kernel reuses the packed INT4 weights
𝐐
𝐖
\mathbf{Q}_{\mathbf{W}}
across streams and steps.
Using CUTLASS
(
NVIDIA,
)
, it accumulates
𝐂
=
𝐐
𝐗
​
𝐐
𝐖
\mathbf{C}=\mathbf{Q}_{\mathbf{X}}\mathbf{Q}_{\mathbf{W}}
in INT32 and produces the output in its epilogue:
Y
^
i
​
j
=
cast
FP16
/
BF16
⁡
[
(
FP32
⁡
(
C
i
​
j
)
​
r
i
)
​
Δ
𝐖
,
j
+
b
j
]
.
\widehat{Y}_{ij}=\operatorname{cast}_{\mathrm{FP16/BF16}}\!\left[\bigl(\operatorname{FP32}(C_{ij})r_{i}\bigr)\Delta_{\mathbf{W},j}+b_{j}\right].
(10)
Here,
Δ
𝐖
,
j
\Delta_{\mathbf{W},j}
is the weight scale for output
channel
j
j
, and
b
j
b_{j}
is the optional bias.
This fusion avoids writing scaled floating-point activations
and INT32 accumulation results to global memory, retaining
only
𝐐
𝐗
\mathbf{Q}_{\mathbf{X}}
and
𝐫
\mathbf{r}
between kernels.
Figure 3:
Rudder’s fused execution.
The first kernel combines channel scaling, stream gains,
and activation quantization.
The second performs low-bit GEMM and applies dequantization,
inverse-gain compensation, and bias before writing the output.
Only integer activations and row scales pass between kernels.
Dashed boxes indicate kernel boundaries.
