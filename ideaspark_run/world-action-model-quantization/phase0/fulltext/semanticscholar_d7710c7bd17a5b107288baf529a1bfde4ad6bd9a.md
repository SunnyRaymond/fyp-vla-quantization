# Tactile-WAM: Touch-Aware World Action Model with Tactile Asymmetric Attention

paper_id: semanticscholar:d7710c7bd17a5b107288baf529a1bfde4ad6bd9a
tier: T2
source_used: html_arxiv
warning: none

## Intro

World Action Models (WAMs) jointly predict future states and robot actions, enabling policies to anticipate the consequences of an action sequence
(
Ye et al. 2026
;
Guo et al. 2024
;
Hu et al. 2024
)
. Through large-scale video pretraining, they acquire strong visual appearance and motion priors. However, their predicted futures remain predominantly visual and cannot fully capture the physical states involved in contact-rich manipulation.
In tasks such as insertion, assembly, and object reorientation, success often depends on slip, jamming, contact-direction changes, and subtle misalignment. These events may be difficult to observe from RGB images but directly determine the next corrective action, as illustrated in Figure
. Vision-based tactile sensors capture local deformation and contact transitions
(
Yuan et al. 2017
;
Lambeta et al. 2020
;
Ward-Cherrier et al. 2018
)
, and learned tactile representations have shown clear benefits in contact-rich manipulation
(
Xu et al. 2024
;
Higuera et al. 2024
;
Xue et al. 2025
;
Luu et al. 2025
)
. A WAM for such tasks should therefore predict future tactile states and effectively use touch for action generation.
Directly incorporating touch, however, is not always beneficial. Tactile datasets are far smaller than video pretraining corpora, and tactile signals mainly describe sparse, local contact events. Unconstrained attention from video queries to tactile keys can therefore interfere with the pretrained visual representation and degrade both video and action prediction. We term this phenomenon
tactile pollution
. As shown in Figure
, naive tactile fusion produces blur and distortion in predicted visual futures.
We further observe that pixel-level changes in tactile images do not reliably reflect actual contact changes. Small pixel differences may correspond to critical events such as slip, compression, or slight in-gripper rotation, while large pixel differences may result from visually prominent but physically minor variations
(
Chen et al. 2026a
;
Chen et al. 2026b
)
. Our analysis shows only weak correlation between tactile pixel similarity and deformation-derived contact similarity. Thus, effective tactile modeling must determine when touch is important and preserve contact-relevant dynamics in predicted tactile representations.
Motivated by these observations, we introduce Tactile-WAM, a Wan2.2-based tactile-aware world action model. Its core component, the Tactile Asymmetric Attention Mechanism (
TAAM
), prevents video queries from attending to tactile keys while preserving tactile access for action queries. We further derive a six-dimensional touch-aware proxy from optical flow between consecutive tactile images. Observed proxy changes generate a causal attention bias that strengthens action attention to touch, while future-proxy supervision encourages predicted tactile representations to preserve action-relevant contact dynamics.
Our main contributions are as follows:
•
We identify and quantify
tactile pollution
in visually pretrained WAMs and introduce
TAAM
to protect video prediction while retaining tactile information for action generation;
•
We reveal the mismatch between tactile pixel changes and contact changes, and propose an optical-flow-based touch-aware proxy for adaptive attention and future tactile supervision;
•
We validate Tactile-WAM on nine simulation tasks and five real-robot tasks, demonstrating substantial improvements in contact-rich manipulation while preserving visual prediction quality.

## Method

Architecture Overview
Figure 3:
Overview of
Tactile-WAM
. RGB and tactile histories are encoded by a shared frozen VAE, and a Wan DiT jointly predicts future visual latents, tactile latents, and action chunks. A
Videoclean
mask prevents visual queries from accessing tactile keys, while a touch-aware bias routes action queries to tactile features within each causal block. A differentiable deformation proxy supervises tactile prediction and drives the bias computation.
Figure
3
illustrates the overall framework of
Tactile-WAM
. RGB and tactile history sequences are independently encoded into separate visual and tactile latent variables via the frozen Wan2.2 VAE, with corresponding modality embeddings added. The language-conditioned Wan DiT then jointly predicts future video frames, future tactile states, and action chunks within a unified framework.
The remainder of this section is organized as follows. We first describe the joint visual-tactile-action modeling based on Wan (Section 4.2). To address the tactile pollution and pixel–touch misalignment issues identified in Section 4.3, we introduce a tactile asymmetric attention mask (Section 4.4) and a touch-aware proxy (Section 4.5), respectively. Specifically, the mask module blocks visual queries from accessing tactile keys to mitigate tactile pollution, while simultaneously encouraging action queries to attend more to tactile information during contact changes via a touch-aware attention bias. To resolve pixel–touch misalignment, we employ a touch-aware proxy to derive the attention bias and additionally supervise the predicted tactile images with proxy signals during training. Further architectural details and training configurations are provided in Appendix B.
Joint Visual-Tactile-Action Modeling
To incorporate vision-based tactile sensing into the unified world-action-model framework, tactile observations are first encoded into a latent space shared with vision. Specifically, the left and right tactile images
o
t
τ
,
L
,
o
t
τ
,
R
o^{\tau,L}_{t},o^{\tau,R}_{t}
(Eq. (6)) are resized, horizontally mosaicked via
ℳ
\mathcal{M}
, and encoded by the causal VAE encoder
E
Wan
E_{\mathrm{Wan}}
of the Wan2.2 backbone:
z
τ
=
E
Wan
​
(
ℳ
⁡
(
o
τ
,
L
,
o
τ
,
R
)
)
,
z^{\tau}=E_{\mathrm{Wan}}\!\left(\mathcal{M}(o^{\tau,L},o^{\tau,R})\right),
(7)
where the future prediction target is denoted as
z
0
τ
z_{0}^{\tau}
, defined in parallel with the visual latent
z
0
v
z_{0}^{v}
in Eq. (2).
The conditional context is accordingly extended to explicitly incorporate the left and right tactile histories:
𝒞
t
τ
=
(
o
≤
t
v
,
{
o
≤
t
τ
,
r
}
r
=
L
,
R
,
s
≤
t
,
ℓ
)
,
\mathcal{C}_{t}^{\tau}=\left(o^{v}_{\leq t},\ \{o^{\tau,r}_{\leq t}\}_{r=L,R},\ s_{\leq t},\ \ell\right),
(8)
where historical tactile latents, after learnable linear projection and modality embedding, are injected into the denoising network as auxiliary conditional signals.
Training and inference for the tactile modality directly follow the flow-matching paradigm defined in Section
2
: substituting the conditional context in Eq. (4) with
𝒞
t
τ
\mathcal{C}_{t}^{\tau}
yields the tactile flow-matching loss
ℒ
FM
τ
\mathcal{L}_{\mathrm{FM}}^{\tau}
; during inference, the clean latent
z
^
0
τ
\widehat{z}_{0}^{\tau}
is reconstructed from the sampled noise
ϵ
τ
\epsilon^{\tau}
following Eq. (5), and then decoded by
D
Wan
D_{\mathrm{Wan}}
to produce the predicted tactile images. Each action chunk is temporally aligned with the corresponding tactile latent frames. Finally, the flow-matching losses for vision, action, and tactile modalities are jointly optimized.
Tactile Asymmetric Attention Mask
Naive symmetric routing allows visual queries to directly attend to sparse tactile keys. Since local tactile events do not necessarily predict global future appearance, this pathway can corrupt the pre-trained visual dynamics representation, leading to the tactile pollution failure mode. To address this, we propose a tactile asymmetric attention mask that protects visual prediction while retaining tactile information for action generation.
Videoclean
Mask.
Let
G
⁡
(
q
)
G(q)
and
G
⁡
(
k
)
G(k)
denote the token groups to which query
q
q
and key
k
k
belong. We block only the access from visual queries to tactile keys:
M
q
​
k
vc
=
{
−
∞
,
G
⁡
(
q
)
=
V
∧
G
⁡
(
k
)
=
T
,
0
,
otherwise
.
M^{\mathrm{vc}}_{qk}=\begin{cases}-\infty,&G(q)=V\ \wedge\ G(k)=T,\\
0,&\text{otherwise}.\end{cases}
(9)
Action queries retain full access to tactile keys, and tactile queries preserve their multimodal context. Thus, this mask protects the visual prediction pathway in a unidirectional manner, rather than completely isolating modalities.
Table 1:
UniVTAC success counts after 100K training steps, with 20 trials per task. Bold marks the best result per task.
Method
Grasp cls.
Hole ins.
Tube ins.
HDMI ins.
Lift bottle
Lift can
Pull key
Bottle shelf
Overall
Tactile-WAM
11
/
20
11/20
4
/
20
4/20
𝟏𝟕
/
𝟐𝟎
\mathbf{17/20}
0
/
20
0/20
1
/
20
1/20
2
/
20
2/20
4
/
20
4/20
9
/
20
9/20
48
/
160
48/160
(30.0%)
w/o
VideoClean
10
/
20
10/20
0
/
20
0/20
12
/
20
12/20
𝟒
/
𝟐𝟎
\mathbf{4/20}
6
/
20
6/20
4
/
20
4/20
4
/
20
4/20
2
/
20
2/20
42
/
160
42/160
(26.3%)
DreamZero
16
/
20
16/20
4
/
20
4/20
4
/
20
4/20
2
/
20
2/20
2
/
20
2/20
2
/
20
2/20
2
/
20
2/20
2
/
20
2/20
34
/
160
34/160
(21.3%)
π
0.5
\pi_{0.5}
𝟏𝟗
/
𝟐𝟎
\mathbf{19/20}
𝟖
/
𝟐𝟎
\mathbf{8/20}
4
/
20
4/20
0
/
20
0/20
𝟏𝟏
/
𝟐𝟎
\mathbf{11/20}
𝟖
/
𝟐𝟎
\mathbf{8/20}
𝟗
/
𝟐𝟎
\mathbf{9/20}
𝟏𝟎
/
𝟐𝟎
\mathbf{10/20}
𝟔𝟗
/
𝟏𝟔𝟎
\mathbf{69/160}
(43.1%)
Table 2:
ManiFeel success counts after 60K training steps, with 50 trials per task. Bold marks the best result per task.
Method
Peg ins.
USB ins.
Power ins.
Gear ins.
Bulb ins.
Bolt-nut
Object search
Peg reorient.
Ball sort
Overall
Tactile-WAM
𝟖
/
𝟓𝟎
\mathbf{8/50}
1
/
50
1/50
9
/
50
9/50
𝟏𝟗
/
𝟓𝟎
\mathbf{19/50}
𝟑𝟒
/
𝟓𝟎
\mathbf{34/50}
𝟐𝟎
/
𝟓𝟎
\mathbf{20/50}
2
/
50
2/50
17
/
50
17/50
37
/
50
37/50
147
/
450
147/450
(32.7%)
DreamZero
4
/
50
4/50
0
/
50
0/50
4
/
50
4/50
9
/
50
9/50
17
/
50
17/50
10
/
50
10/50
0
/
50
0/50
8
/
50
8/50
18
/
50
18/50
70
/
450
70/450
(15.6%)
π
0.5
\pi_{0.5}
5
/
50
5/50
𝟓
/
𝟓𝟎
\mathbf{5/50}
𝟏𝟒
/
𝟓𝟎
\mathbf{14/50}
11
/
50
11/50
18
/
50
18/50
15
/
50
15/50
𝟐𝟖
/
𝟓𝟎
\mathbf{28/50}
𝟐𝟏
/
𝟓𝟎
\mathbf{21/50}
𝟒𝟖
/
𝟓𝟎
\mathbf{48/50}
𝟏𝟔𝟓
/
𝟒𝟓𝟎
\mathbf{165/450}
(36.7%)
Touch-aware Attention Bias.
To enable action prediction to perceive the current contact state, we design a touch-aware attention bias that guides action queries to attend to tactile representations within the same temporal block. The tactile sequence is partitioned into causal blocks according to the action-chunk boundaries, where each block relies solely on historical observations up to its end time to ensure causality.
For each tactile causal block
c
c
, we extract a touch-aware perception vector from the tactile frames at the previous and current policy-calling instants:
F
c
obs
=
Φ
⁡
(
o
u
c
−
8
τ
,
o
u
c
τ
)
,
F^{\mathrm{obs}}_{c}=\Phi(o^{\tau}_{u_{c}-8},o^{\tau}_{u_{c}}),
(10)
where
Φ
⁡
(
⋅
)
\Phi(\cdot)
denotes the differentiable deformation estimator detailed in Section
4
, and
u
c
u_{c}
is the anchor timestamp of the block. To obtain a compact gating signal, we convert this high-dimensional vector into a contact intensity score:
s
c
=
tanh
⁡
(
ReLU
⁡
(
‖
F
c
obs
‖
2
−
θ
)
T
)
,
s_{c}=\tanh\!\left(\frac{\operatorname{ReLU}(\|F^{\mathrm{obs}}_{c}\|_{2}-\theta)}{T}\right),
(11)
which is normalized to
[
0
,
1
)
[0,1)
via threshold
θ
\theta
and temperature
T
T
, reflecting the presence and strength of contact. Each tactile patch key
k
k
within the block receives the following bias enhancement:
b
c
,
k
=
clip
⁡
(
α
​
s
c
,
0
,
b
max
)
,
b_{c,k}=\operatorname{clip}\!\left(\alpha s_{c},0,b_{\max}\right),
(12)
where
α
\alpha
controls the amplification factor and
b
max
b_{\max}
caps the upper bound. This bias acts as a block-level modality gate, applied between all action queries and tactile keys. Let
I
q
​
k
c
I^{c}_{qk}
indicate whether action query
q
q
and tactile key
k
k
belong to the same block (1 if yes, 0 otherwise). The final biased attention logit is:
A
q
​
k
′
=
A
q
​
k
+
M
q
​
k
vc
+
I
q
​
k
c
​
b
c
⁡
(
k
)
,
k
,
A^{\prime}_{qk}=A_{qk}+M^{\mathrm{vc}}_{qk}+I^{c}_{qk}b_{c(k),k},
(13)
where
A
A
is the original attention score and
M
vc
M^{\mathrm{vc}}
is the video–tactile masking matrix from Eq. (10). Note that the bias is computed solely from observed tactile forces, independent of future tactile targets or intermediate denoising predictions.
Touch-aware Proxy
Proxy Construction.
We employ a differentiable deformation estimator
Φ
\Phi
as a touch-aware proxy, mapping successive decoded tactile frames to a 3D deformation proxy signal for each sensor. Specifically, for sensor
s
∈
{
L
,
R
}
s\in\{L,R\}
, let
I
t
s
I_{t}^{s}
denote the grayscale tactile image and
Δ
​
I
t
s
=
I
t
s
−
I
t
−
1
s
\Delta I_{t}^{s}=I_{t}^{s}-I_{t-1}^{s}
its temporal difference (with zero-initialization for the first frame). Using the centered spatial gradients
(
g
x
s
,
g
y
s
)
=
∇
I
t
s
(g_{x}^{s},g_{y}^{s})=\nabla I_{t}^{s}
, we construct a differentiable gradient-aligned deformation field:
u
x
,
t
s
=
Δ
​
I
t
s
​
g
x
s
(
g
x
s
)
2
+
(
g
y
s
)
2
+
ϵ
,
u
y
,
t
s
=
Δ
​
I
t
s
​
g
y
s
(
g
x
s
)
2
+
(
g
y
s
)
2
+
ϵ
,
u_{x,t}^{s}=\frac{\Delta I_{t}^{s}g_{x}^{s}}{\sqrt{(g_{x}^{s})^{2}+(g_{y}^{s})^{2}+\epsilon}},\qquad u_{y,t}^{s}=\frac{\Delta I_{t}^{s}g_{y}^{s}}{\sqrt{(g_{x}^{s})^{2}+(g_{y}^{s})^{2}+\epsilon}},
(14)
where
ϵ
=
10
−
6
\epsilon=10^{-6}
. The proxy signal for each sensor is then defined as
f
t
s
=
[
⟨
u
x
,
t
s
⟩
,
⟨
u
y
,
t
s
⟩
,
⟨
∂
x
u
x
,
t
s
+
∂
y
u
y
,
t
s
⟩
]
f_{t}^{s}=[\langle u_{x,t}^{s}\rangle,\langle u_{y,t}^{s}\rangle,\langle\partial_{x}u_{x,t}^{s}+\partial_{y}u_{y,t}^{s}\rangle]
,
where
⟨
⋅
⟩
\langle\cdot\rangle
denotes spatial averaging. The first two components summarize tangential deformation, while the third measures the average divergence. Each action chunk corresponds to 8 frames of tactile transition; applying
Φ
\Phi
to the predicted tactile images yields the predicted proxy signal:
F
^
=
Φ
⁡
(
o
^
τ
,
L
)
⊕
Φ
⁡
(
o
^
τ
,
R
)
∈
ℝ
C
×
8
×
6
,
\widehat{F}=\Phi(\widehat{o}^{\tau,L})\oplus\Phi(\widehat{o}^{\tau,R})\in\mathbb{R}^{C\times 8\times 6},
(15)
where
C
C
is the number of action chunks.
Proxy Supervision.
Applying the same estimator to the ground-truth tactile trajectories yields the reference signal
F
∗
F^{*}
. We supervise the proxy signal with:
ℒ
F
=
0.05
​
SmoothL1
⁡
(
F
^
,
F
∗
)
.
\mathcal{L}_{F}=0.05\,\operatorname{SmoothL1}(\widehat{F},F^{*}).
(16)
The overall training objective is:
ℒ
=
ℒ
video
+
ℒ
action
+
λ
τ
​
ℒ
FM
τ
+
ℒ
F
.
\mathcal{L}=\mathcal{L}_{\mathrm{video}}+\mathcal{L}_{\mathrm{action}}+\lambda_{\tau}\mathcal{L}_{\mathrm{FM}}^{\tau}+\mathcal{L}_{F}.
(17)
During training, gradients are propagated back to the predicted tactile latents through the frozen decoder and the differentiable estimator
Φ
\Phi
; all Wan VAE parameters remain frozen.
