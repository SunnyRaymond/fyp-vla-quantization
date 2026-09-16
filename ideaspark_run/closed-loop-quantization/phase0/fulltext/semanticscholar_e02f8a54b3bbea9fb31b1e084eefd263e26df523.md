# DA-PTQ: Drift-Aware Post-Training Quantization for Efficient Vision-Language-Action Models

paper_id: semanticscholar:e02f8a54b3bbea9fb31b1e084eefd263e26df523
tier: T2
source_used: html_arxiv
warning: none

## Intro

Vision-Language-Action models (VLAs)
(
Zhong et al., 2025
)
have emerged as a promising paradigm for embodied artificial intelligence, enabling robots to perform complex tasks based on visual observations and natural language instructions. By integrating large-scale vision encoders, language backbones, and action generation modules within a unified architecture, recent models such as RT-2
(
Zitkovich et al., 2023
)
and OpenVLA
(
Kim et al., 2024
)
demonstrate strong generalization across a wide range of manipulation tasks. However, these capabilities come at the cost of substantial memory and computational overhead, often involving billions of parameters. Such requirements fundamentally conflict with the constraints of onboard robotic systems, where low latency, limited memory, and power efficiency are critical. Consequently, enabling efficient deployment of VLAs remains a central challenge for real-world embodied AI.
Model compression offers a natural pathway to address this challenge. Among various techniques, quantization is particularly attractive due to its ability to simultaneously reduce memory footprint and inference cost. While Quantization-Aware Training (QAT)
(
Jacob et al., 2018
)
can recover accuracy through retraining, it is often impractical for large-scale VLA models due to the substantial computational cost and the limited availability of high-quality multimodal robotic data. In contrast, Post-Training Quantization (PTQ)
(
Liu et al., 2021
)
provides a lightweight alternative by calibrating models without retraining. Recent PTQ methods have achieved remarkable success in large language models and vision-language models. However, directly applying these techniques to VLAs often leads to severe performance degradation, and in some cases unstable control behaviors during sequential execution
(
Zhang et al., 2026
;
Xu et al., 2026
)
.
We attribute this limitation to a fundamental mismatch between conventional quantization objectives and the sequential, control-sensitive nature of embodied decision-making. Existing PTQ methods typically assume that quantization errors are independent and locally bounded, such that minimizing layer-wise reconstruction error is sufficient to preserve downstream functionality (e.g., AWQ
(
Lin et al., 2024
)
, GPTQ
(
Frantar et al., 2022
)
). While this assumption is reasonable for static generation tasks, it becomes inadequate in embodied control settings, where decisions are executed sequentially and coupled through system dynamics. In VLA models, quantization perturbs latent representations at the vision-language-to-action interface, and these perturbations become temporally coupled and progressively amplified over time
(
Chen et al., 2024
;
Liu et al., 2026
)
. As accumulated errors interact with robot dynamics and feedback control loops, they manifest as kinematic drift, i.e., the deviation between the executed trajectory and the nominal trajectory induced by quantization, ultimately resulting in significant performance degradation
(
Park et al., 2025
)
.
Recent works have begun to explore quantization strategies tailored for VLAs. For example, SQAP-VLA
(
Fang et al., 2025
)
jointly optimizes quantization and token pruning, QuantVLA
(
Zhang et al., 2026
)
stabilizes the perception-to-action interface via scale-calibrated adjustments, and QVLA
(
Xu et al., 2026
)
allocates bit-widths based on channel-wise action sensitivity. While these methods mitigate performance degradation to some extent, substantial gaps persist under aggressive compression. A key limitation is that existing approaches primarily rely on static or single-step approximations, failing to capture long-horizon error propagation and its interaction with robot dynamics. Consequently, they cannot effectively model or control trajectory-level error accumulation, leaving kinematic drift as a fundamental bottleneck in achieving an optimal efficiency-accuracy trade-off.
To address the above challenges, we propose Drift-Aware Post-Training Quantization (DA-PTQ), a training-free framework that explicitly models and mitigates kinematic drift, as conceptually illustrated in Figure
1
. Instead of treating quantization as a static reconstruction problem, DA-PTQ formulates it as a drift-aware optimization process that explicitly accounts for both temporal accumulation and physical amplification of errors. Specifically, DA-PTQ consists of two complementary components. Cross-Space Representation Compensation mitigates structured distortions between multimodal representations and the action space via lightweight affine and low-rank transformations, improving action consistency under quantization. Motion-Driven Mixed-Precision Allocation further reduces long-horizon drift by assigning bit-widths based on trajectory-level motion error under resource constraints. Together, these components enable stable and efficient low-precision deployment without introducing additional inference overhead.
Our main contributions are summarized as follows:
•
We present a systematic analysis of error accumulation in VLA quantization, identifying kinematic drift as a key bottleneck arising from the interplay between quantization perturbations and sequential embodied control.
•
We propose DA-PTQ, a training-free framework that reformulates quantization as a drift-aware optimization problem, integrating cross-space representation compensation and motion-driven precision allocation without incurring additional inference overhead.
•
Experimental results show that DA-PTQ reduces kinematic drift and achieves performance comparable to full-precision models under low-bit settings, enabling deployment on resource-constrained robotic platforms.

## Method

3.1.
Problem Formulation
Vision-Language-Action models (VLAs) define a policy
Π
θ
\Pi_{\theta}
that maps visual observations
𝐕
t
\mathbf{V}_{t}
and a language instruction
p
p
to continuous actions
𝐚
t
\mathbf{a}_{t}
:
(1)
Π
θ
​
(
𝐚
t
∣
𝐕
t
,
p
)
=
Ψ
θ
​
(
f
θ
​
(
𝐕
t
,
p
)
)
,
\Pi_{\theta}(\mathbf{a}_{t}\mid\mathbf{V}_{t},p)=\Psi_{\theta}\left(f_{\theta}(\mathbf{V}_{t},p)\right),
where
f
θ
f_{\theta}
denotes the vision-language backbone and
Ψ
θ
\Psi_{\theta}
is the action decoder. The backbone encodes multimodal inputs into a latent representation
𝐳
t
=
f
θ
​
(
𝐕
t
,
p
)
\mathbf{z}_{t}=f_{\theta}(\mathbf{V}_{t},p)
, which serves as the conditioning signal for action generation.
In modern VLA architectures,
Ψ
θ
\Psi_{\theta}
is often implemented as a diffusion-based policy, where actions are generated via iterative denoising conditioned on
𝐳
t
\mathbf{z}_{t}
. While effective, this design poses unique challenges under Post-Training Quantization (PTQ), as latent-space quantization errors are repeatedly injected during decoding and further accumulate over time in sequential control. These challenges stem from two coupled mechanisms:
Quantization Sensitivity at the Conditioning Interface.
The perception-to-action interface
𝐳
t
\mathbf{z}_{t}
acts as a tightly coupled information bottleneck through which all task-relevant signals are conveyed to the action decoder. Under quantization, perturbations in
𝐳
t
\mathbf{z}_{t}
directly distort the conditioning distribution. Since the decoder depends solely on
𝐳
t
\mathbf{z}_{t}
, these distortions propagate through all denoising steps without downstream correction, and are further amplified by the iterative diffusion process, resulting in structured distributional shifts even within a single action prediction.
Temporal Error Accumulation in Sequential Control.
Beyond single-step sensitivity, VLA policies operate in a closed-loop setting where actions affect future observations. Let
ϵ
t
∈
ℝ
7
\boldsymbol{\epsilon}_{t}\in\mathbb{R}^{7}
denote the action error at timestep
t
t
, primarily induced by quantization. The resulting end-effector deviation is:
(2)
δ
​
𝐞
t
=
𝐉
(
t
)
​
ϵ
t
,
\delta\mathbf{e}_{t}=\mathbf{J}^{(t)}\boldsymbol{\epsilon}_{t},
where
𝐉
(
t
)
\mathbf{J}^{(t)}
is the manipulator Jacobian. Over a horizon
T
T
, the accumulated deviation becomes:
(3)
𝐄
T
=
∑
t
=
1
T
𝐉
(
t
)
​
ϵ
t
.
\mathbf{E}_{T}=\sum_{t=1}^{T}\mathbf{J}^{(t)}\boldsymbol{\epsilon}_{t}.
This shows that quantization errors are both temporally accumulated and geometrically amplified via
∥
𝐉
(
t
)
:
,
j
∥
\|\mathbf{J}^{(t)}_{:,j}\|
, causing small per-step errors to induce significant long-horizon drift.
Taken together, quantization in VLAs induces both representation-level distortion and trajectory-level drift, which are tightly coupled yet not explicitly addressed by existing PTQ methods.
3.2.
Overview of DA-PTQ
To address the coupled challenges of representation distortion and trajectory-level drift, we propose DA-PTQ, a streamlined calibration pipeline with no retraining overhead. As shown in Figure
2
, it unifies cross-space representation compensation and drift-aware mixed-precision allocation into a three-stage process given a small calibration dataset.
First, we perform full-precision forward passes to collect calibration statistics. During this stage, we extract action trajectories to estimate the manipulator Jacobian and derive temporal drift propagation scores that characterize how errors accumulate over time. In parallel, we record activation statistics at the critical vision-language-to-action interfaces, capturing the reference distribution of conditioning representations.
Second, we apply cross-space representation compensation to correct distortion at the conditioning interface. After quantization, we perform a forward pass to measure activation shifts and solve for lightweight affine transformations that align quantized representations with their full-precision counterparts. These transformations are analytically merged into the quantized weights and biases, incurring no additional inference cost.
Finally, we conduct drift-aware mixed-precision allocation. Based on the compensated model, we perform a lightweight sensitivity analysis to evaluate the contribution of each layer to long-horizon error accumulation. This results in a layer-wise bit-width assignment that preserves higher precision for drift-sensitive components while aggressively compressing less critical layers.
The resulting DA-PTQ model achieves substantial compression while maintaining stable and accurate continuous control, effectively mitigating both conditioning distortion and long-horizon drift without incurring runtime overhead.
3.3.
Cross-Space Representation Compensation
Quantization induces structured distributional distortions at the vision-language-to-action interface, where activation shifts corrupt the conditioning signal for action generation. To mitigate this, we introduce Cross-Space Representation Compensation (CSRC), which aligns quantized activations with their full-precision counterparts through a hierarchical compensation scheme.
Let
𝐳
l
,
c
FP
\mathbf{z}_{l,c}^{\text{FP}}
and
𝐳
^
l
,
c
Q
\hat{\mathbf{z}}_{l,c}^{\text{Q}}
denote the full-precision and quantized activations of channel
c
c
at interface layer
l
l
. We first match their first- and second-order statistics computed on the calibration set. Specifically, we derive a scale factor to align the standard deviation:
(4)
g
l
,
c
=
clip
⁡
(
σ
l
,
c
FP
σ
l
,
c
Q
+
ε
,
g
min
,
g
max
)
,
g_{l,c}=\mathrm{clip}\!\left(\frac{\sigma_{l,c}^{\text{FP}}}{\sigma_{l,c}^{\text{Q}}+\varepsilon},\ g_{\min},g_{\max}\right),
and a bias term to restore the mean:
(5)
d
l
,
c
=
μ
l
,
c
FP
−
g
l
,
c
⋅
μ
l
,
c
Q
.
d_{l,c}=\mu_{l,c}^{\text{FP}}-g_{l,c}\cdot\mu_{l,c}^{\text{Q}}.
The corrected activation is:
(6)
z
~
l
,
c
=
g
l
,
c
​
z
^
l
,
c
Q
+
d
l
,
c
.
\tilde{z}_{l,c}=g_{l,c}\hat{z}_{l,c}^{\text{Q}}+d_{l,c}.
While per-channel scaling corrects diagonal shifts, it fails to capture structured cross-channel distortion. To address this, we introduce a dense affine transformation
𝐌
l
\mathbf{M}_{l}
that aligns second-order statistics between full-precision and quantized activations. Let
𝚺
l
FP
\boldsymbol{\Sigma}_{l}^{\text{FP}}
and
𝚺
l
Q
\boldsymbol{\Sigma}_{l}^{\text{Q}}
denote their empirical covariance matrices. We solve:
(7)
min
𝐌
l
⁡
λ
f
​
‖
𝐖
⊙
(
𝚺
l
FP
−
𝐌
l
​
𝚺
l
Q
​
𝐌
l
⊤
)
‖
F
2
+
λ
i
​
‖
𝐌
l
−
𝐈
‖
F
2
,
\min_{\mathbf{M}_{l}}\ \lambda_{f}\left\|\mathbf{W}\odot\left(\boldsymbol{\Sigma}_{l}^{\text{FP}}-\mathbf{M}_{l}\boldsymbol{\Sigma}_{l}^{\text{Q}}\mathbf{M}_{l}^{\top}\right)\right\|_{F}^{2}+\lambda_{i}\left\|\mathbf{M}_{l}-\mathbf{I}\right\|_{F}^{2},
where
𝐖
\mathbf{W}
is a diagonal weight matrix derived from per-channel variance ratios, and
λ
i
\lambda_{i}
regularizes the solution toward identity to preserve stability.
To ensure efficiency, we parameterize
𝐌
l
\mathbf{M}_{l}
as a low-rank update to the identity:
(8)
𝐌
l
=
𝐈
+
𝐔
l
​
𝐕
l
⊤
,
r
≪
d
,
\mathbf{M}_{l}=\mathbf{I}+\mathbf{U}_{l}\mathbf{V}_{l}^{\top},\quad r\ll d,
obtained via truncated SVD of the dense solution.
The fully corrected activation is given by:
(9)
𝐳
~
l
=
𝐌
l
​
𝐳
^
l
Q
+
𝐝
l
,
\tilde{\mathbf{z}}_{l}=\mathbf{M}_{l}\hat{\mathbf{z}}_{l}^{\text{Q}}+\mathbf{d}_{l},
where the bias restores the mean of the aligned distribution. All compensation parameters are analytically folded into the quantized weights during calibration, incurring zero inference overhead.
3.4.
Drift-Aware Mixed-Precision Allocation
To mitigate temporally accumulated errors in continuous control, we propose a Drift-Aware Mixed-Precision Allocation (DA-MPA) strategy that explicitly accounts for how quantization noise propagates and amplifies over long horizons.
In embodied control tasks, quantization errors introduced at each timestep accumulate through closed-loop interactions, resulting in trajectory drift and covariate shift. Directly modeling this process by unrolling environment dynamics is both computationally prohibitive and non-differentiable. To address this, we adopt an analytical surrogate that approximates temporal drift via single-step spatial error propagation. The action vector is defined as a 7-dimensional Cartesian increment:
(10)
𝐚
t
=
[
Δ
​
x
,
Δ
​
y
,
Δ
​
z
,
Δ
​
r
x
,
Δ
​
r
y
,
Δ
​
r
z
,
Δ
​
g
]
∈
ℝ
7
.
\mathbf{a}_{t}=[\Delta x,\Delta y,\Delta z,\Delta r_{x},\Delta r_{y},\Delta r_{z},\Delta g]\in\mathbb{R}^{7}.
Under the small-angle assumption (
‖
Δ
​
𝐫
‖
≪
1
\|\Delta\mathbf{r}\|\ll 1
), rotational components can be approximated as angular deviations:
(11)
Δ
​
r
i
≈
δ
​
θ
i
.
\Delta r_{i}\approx\delta\theta_{i}.
For typical tabletop manipulation, drift is dominated by motion in the horizontal plane. We therefore project the action into three principal components corresponding to planar translation and rotation.
To model error accumulation, we reinterpret the action dimensions as joint increments of a virtual planar serial chain,
𝐪
=
[
q
1
,
…
,
q
7
]
\mathbf{q}=[q_{1},\dots,q_{7}]
. The absolute orientation of the
j
j
-th segment accumulates upstream perturbations:
(12)
θ
j
=
∑
i
=
1
j
q
i
,
\theta_{j}=\sum_{i=1}^{j}q_{i},
which provides a differentiable proxy for how per-dimension errors propagate and accumulate, mimicking long-horizon drift behavior.
We quantify how perturbations in each action dimension affect the end-effector by deriving the structural Jacobian
𝐉
∈
ℝ
3
×
7
\mathbf{J}\in\mathbb{R}^{3\times 7}
of the virtual chain:
(13)
J
x
(
j
)
=
−
∑
k
=
j
6
sin
θ
k
,
J
y
(
j
)
=
∑
k
=
j
6
cos
θ
k
,
J
θ
(
j
)
=
1
.
J_{x}^{(j)}=-\sum_{k=j}^{6}\sin\theta_{k},\ \ \ J_{y}^{(j)}=\sum_{k=j}^{6}\cos\theta_{k},\ \ \ J_{\theta}^{(j)}=1.
Stacking these components yields:
(14)
𝐉
=
[
J
x
(
1
)
⋯
J
x
(
7
)
J
y
(
1
)
⋯
J
y
(
7
)
1
⋯
1
]
.
\mathbf{J}=\begin{bmatrix}J_{x}^{(1)}&\cdots&J_{x}^{(7)}\\
J_{y}^{(1)}&\cdots&J_{y}^{(7)}\\
1&\cdots&1\end{bmatrix}.
This structure naturally captures error amplification: earlier dimensions influence more downstream segments, resulting in larger column norms, i.e.,
∥
𝐉
:
,
j
∥
>
∥
𝐉
:
,
j
+
1
∥
\|\mathbf{J}_{:,j}\|>\|\mathbf{J}_{:,j+1}\|
. Thus, the Jacobian encodes the intrinsic topology of drift propagation without requiring task-specific supervision.
To isolate the contribution of each dimension to overall drift, we compute the damped least-squares pseudo-inverse:
(15)
𝐉
+
=
𝐉
⊤
​
(
𝐉𝐉
⊤
+
λ
​
𝐈
3
)
−
1
,
\mathbf{J}^{+}=\mathbf{J}^{\top}\left(\mathbf{J}\mathbf{J}^{\top}+\lambda\mathbf{I}_{3}\right)^{-1},
where
λ
>
0
\lambda>0
ensures numerical stability.
We further introduce axis-dependent weights to balance translational and rotational sensitivities. The drift propagation score for dimension
j
j
is defined as:
(16)
s
j
=
𝔼
𝐚
∼
𝒟
​
[
∑
c
∈
{
x
,
y
,
θ
}
w
c
​
|
J
j
,
c
+
|
]
.
s_{j}=\mathbb{E}_{\mathbf{a}\sim\mathcal{D}}\left[\sum_{c\in\{x,y,\theta\}}w_{c}\left|J^{+}_{j,c}\right|\right].
We normalize these scores to obtain drift sensitivity weights:
(17)
s
^
j
=
s
j
1
7
​
∑
i
=
1
7
s
i
.
\hat{s}_{j}=\frac{s_{j}}{\frac{1}{7}\sum_{i=1}^{7}s_{i}}.
These weights quantify how strongly errors in each action dimension contribute to long-horizon drift.
We incorporate the drift sensitivity weights into the calibration objective to penalize quantization noise that disproportionately amplifies drift:
(18)
ℒ
drift
=
𝔼
𝐱
,
ϵ
,
t
​
[
∑
j
=
1
7
s
^
j
⋅
(
ϵ
^
j
​
(
𝐱
t
,
t
,
𝐳
)
−
ϵ
j
)
2
]
.
\mathcal{L}_{\text{drift}}=\mathbb{E}_{\mathbf{x},\boldsymbol{\epsilon},t}\left[\sum_{j=1}^{7}\hat{s}_{j}\cdot\left(\hat{\epsilon}_{j}(\mathbf{x}_{t},t,\mathbf{z})-\epsilon_{j}\right)^{2}\right].
Under this objective, gradients are automatically reweighted to emphasize dimensions with high drift sensitivity. For each quantizable layer
l
l
, we compute its drift sensitivity score by averaging gradient magnitudes over
R
R
calibration steps:
(19)
ϕ
l
=
1
R
​
∑
r
=
1
R
1
d
out
​
∑
i
=
1
d
out
|
∂
ℒ
drift
∂
𝐖
l
|
i
(
r
)
.
\phi_{l}=\frac{1}{R}\sum_{r=1}^{R}\frac{1}{d_{\text{out}}}\sum_{i=1}^{d_{\text{out}}}\left|\frac{\partial\mathcal{L}_{\text{drift}}}{\partial\mathbf{W}_{l}}\right|_{i}^{(r)}.
Layers are then ranked according to
ϕ
l
\phi_{l}
. To tightly control temporal drift, the top
k
%
k\%
most sensitive layers are retained in high precision (BF16), while the remaining layers are quantized to low bit-width:
(20)
b
l
=
{
BF16
,
if
​
ϕ
l
≥
ϕ
(
k
)
,
W4
,
otherwise
.
b_{l}=\begin{cases}\text{BF16},&\text{if }\phi_{l}\geq\phi_{(k)},\\
\text{W4},&\text{otherwise}.\end{cases}
3.5.
Summary of the DA-PTQ Pipeline
We summarize DA-PTQ as a three-stage calibration procedure in Algorithm
1
. The pipeline is fully training-free and requires only forward passes and lightweight gradient accumulation. Given a pretrained VLA model and a small calibration dataset, the procedure first performs drift profiling by collecting full-precision activation statistics and estimating both per-dimension drift sensitivity and layer-wise impact on error accumulation. Next, a drift-aware mixed-precision configuration is determined by retaining the most sensitive layers in high precision while aggressively quantizing the rest. Finally, cross-space representation compensation is calibrated on the quantized model by aligning activation distributions and folding the resulting affine transformations into the weights, incurring zero inference overhead.
Algorithm 1
Drift-Aware Post-Training Quantization (DA-PTQ)
Input:
Pretrained VLA model with DiT action head
Ψ
θ
\Psi_{\theta}
, calibration dataset
𝒟
\mathcal{D}
, retained BF16 ratio
k
%
k\%
Output:
Quantized VLA model with W4/BF16 mixed precision and folded compensation
1:
# Stage 1: Drift Profiling
2:
for
each batch in
𝒟
\mathcal{D}
do
3:
Accumulate full-precision statistics
𝝁
l
FP
\boldsymbol{\mu}_{l}^{\text{FP}}
,
𝚺
l
FP
\boldsymbol{\Sigma}_{l}^{\text{FP}}
at the perception-action interface
4:
Compute structural Jacobian
𝐉
\mathbf{J}
and per-dimension drift sensitivities
s
^
j
\hat{s}_{j}
5:
Estimate layer-wise drift sensitivities
ϕ
l
\phi_{l}
under
ℒ
drift
\mathcal{L}_{\text{drift}}
6:
end
for
7:
# Stage 2: Cross-Space Representation Compensation
8:
Quantize the model with the initial calibration configuration
9:
for
each batch in
𝒟
\mathcal{D}
do
10:
Accumulate quantized statistics
𝝁
l
Q
\boldsymbol{\mu}_{l}^{\text{Q}}
,
𝚺
l
Q
\boldsymbol{\Sigma}_{l}^{\text{Q}}
11:
end
for
12:
for
each interface layer
l
l
do
13:
Solve for affine matrix
𝐌
l
\mathbf{M}_{l}
and bias
𝐝
l
\mathbf{d}_{l}
to align quantized statistics with full-precision statistics
14:
Fold
𝐌
l
\mathbf{M}_{l}
and
𝐝
l
\mathbf{d}_{l}
into the quantized weights of layer
l
l
15:
end
for
16:
# Stage 3: Drift-Aware Mixed-Precision Allocation
17:
for
each layer
l
∈
Ψ
θ
l\in\Psi_{\theta}
do
18:
b
l
←
BF16
b_{l}\leftarrow\text{BF16}
if
ϕ
l
≥
ϕ
(
k
)
\phi_{l}\geq\phi_{(k)}
else
W4
19:
end
for
20:
Apply the bit-width map
{
b
l
}
\{b_{l}\}
to obtain the final quantized model
21:
return
Compressed VLA model
