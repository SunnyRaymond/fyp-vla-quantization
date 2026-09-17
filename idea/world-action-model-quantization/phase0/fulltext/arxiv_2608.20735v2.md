# ForeTime-VLA: Causal Future-Token Distillation from a World Action Model for Conveyor-Belt Manipulation

paper_id: arxiv:2608.20735v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

Vision-language-action (VLA) models transfer semantic priors from large
vision-language backbones to robot control. RT-2
[
37
]
,
OpenVLA
[
18
]
, Octo
[
22
]
, and the
π
0
\pi_{0}
/
π
0.5
\pi_{0.5}
family
[
3
,
26
]
show
that a common policy can be adapted across tasks, scenes, and embodiments.
Expressive action decoders—including diffusion
[
6
]
and flow matching
[
19
]
—further support multimodal,
high-dimensional action chunks.
Dynamic conveyor-belt pick-and-place exposes a complementary weakness. The
correct motion depends not only on what is visible now, but also on when the
object will enter the reachable set, when the gripper should close, and how the
end effector should be oriented at contact. A policy trained only to reconstruct
demonstrated actions may learn these regularities indirectly, but receives no
explicit future-structure supervision.
World models provide such supervision through video prediction. Text-guided
video policies
[
8
]
, generative video-language-action
models
[
33
]
, and recent world action models (WAMs) use predicted
future observations to support control. Explicit imagine-then-act pipelines,
however, add video generation to the control loop. Fast-WAM
[
35
]
instead reports that video modeling is especially useful as a training signal
and can be removed from test-time inference. This raises a practical question:
can a pretrained VLA acquire a compact version of that predictive
structure without running the WAM at deployment?
We answer this question with ForeTime-VLA, whose teacher–student pipeline is
summarized in Fig.
1
. During offline preprocessing, a
Fast-WAM/Wan video latent pipeline sees the current observation and eight
future offsets. A non-collapsed adapter compresses these features into an
action-equivalent teacher code. During policy training, a small causal encoder
must predict that code from the preceding eight observations. The predicted
code conditions both the slow VLM path and the fast action-expert path of
π
0.5
\pi_{0.5}
, while the original flow-matching construction continues to use the
recorded action chunk. Ground-truth actions remain unchanged, avoiding the
target-alignment confound of pseudo-label replacement.
Our contributions are:
•
a deployable teacher–student formulation that transfers
future-conditioned WAM structure into an eight-frame causal history encoder,
with no future frame or teacher forward pass at inference;
•
dual-path conditioning of
π
0.5
\pi_{0.5}
with future, phase, and
time-to-transition variables, supervised by pointwise, relational, temporal, and
action-equivalence objectives; and
•
a paired offline evaluation and two quantitative real-robot studies
spanning stationary, moving, and speed-stress conditions, with outcome-level
analysis connecting future-aware supervision to fewer late and contact-pose
failures.
Fig. 1:
ForeTime-VLA overview.
Left:
the offline teacher encodes
the current observation and sampled future frames with a frozen video VAE, then
maps them to a compact action-equivalent target.
Center:
the deployable
student uses eight causal observations to predict future, phase, and transition
horizon variables. These predictions condition both the slow
π
0.5
\pi_{0.5}
VLM
prefix and the fast action-expert suffix, while the original noisy-action and
target-flow construction is retained.
Right:
training combines action
flow, future alignment, relational geometry, transition, phase, and auxiliary
action-reconstruction losses. Future frames and the WAM are absent at test
time.

## Method

III-A
Problem Formulation
At time
t
t
, the policy receives current RGB observations
o
t
o_{t}
, a language
instruction
ℓ
\ell
, robot state
s
t
s_{t}
, and an eight-step causal history
ℋ
t
\mathcal{H}_{t}
. It predicts an action chunk
a
t
:
t
+
H
−
1
∈
ℝ
H
×
D
a_{t:t+H-1}\in\mathbb{R}^{H\times D}
, with
H
=
20
H=20
and
D
=
16
D=16
. The base
π
0.5
\pi_{0.5}
action expert uses conditional flow matching. For actions
a
a
, noise
ϵ
∼
𝒩
⁡
(
0
,
I
)
\epsilon\sim\mathcal{N}(0,I)
, and
τ
∈
(
0
,
1
)
\tau\in(0,1)
,
x
τ
=
(
1
−
τ
)
​
a
+
τ
​
ϵ
,
u
τ
=
ϵ
−
a
,
x_{\tau}=(1-\tau)a+\tau\epsilon,\qquad u_{\tau}=\epsilon-a,
(1)
and the base loss is
ℒ
flow
=
𝔼
⁡
[
‖
v
θ
​
(
x
τ
,
τ
,
o
t
,
ℓ
)
−
u
τ
‖
2
2
]
.
\mathcal{L}_{\mathrm{flow}}=\mathbb{E}\left[\left\|v_{\theta}(x_{\tau},\tau,o_{t},\ell)-u_{\tau}\right\|_{2}^{2}\right].
(2)
This noisy-action/target-flow construction is shown in
Fig.
1
(C) and is preserved when the temporal conditioning is added.
Our goal is to add future-sensitive structure without changing
a
a
and without
using future observations at inference.
III-B
Offline Action-Equivalent Future Teacher
The left side of Fig.
1
depicts the privileged teacher branch.
For every valid training window, we sample the present and eight future frames
at offsets
{
2
,
4
,
7
,
9
,
12
,
14
,
17
,
19
}
\{2,4,7,9,12,14,17,19\}
. A frozen Wan2.2 video
VAE
[
32
]
—the visual latent pipeline used by
Fast-WAM
[
35
]
—encodes the sequence. Spatial-temporal means
and standard deviations produce 96-D current and 96-D future features.
The initially exported quotient codes exhibited insufficient variation.
We therefore train a non-collapsed action-equivalent adapter while keeping the
video features frozen. Its teacher encoder consumes current feature, future
feature, and the 26-D state; a causal auxiliary encoder consumes only current
feature and state. Both emit 64-D codes. A shared action decoder reconstructs
the normalized
20
×
16
20\times 16
action chunk, and a future decoder reconstructs the
future-minus-current feature. Training combines teacher and causal action MSE,
future smooth-L1, teacher–causal cosine, per-dimension variance, off-diagonal
covariance, and pairwise action-geometry losses with weights
1.0
1.0
,
0.5
0.5
,
0.25
0.25
,
0.10
0.10
,
0.20
0.20
,
0.01
0.01
, and
0.05
0.05
, respectively.
The resulting teacher codes are whitened per dimension and cached.
Thus the privileged future is used to define
z
t
T
z_{t}^{T}
, but never enters the
deployed policy.
Figure
2
summarizes the adapter from top to bottom. The
frozen latent pipeline extracts
c
t
c_{t}
from the current observation and
f
t
f_{t}
from
the sampled future frames, while the 26-D state
s
t
s_{t}
is supplied to the
adapter. The privileged teacher encoder uses all three inputs,
z
^
t
T
=
g
T
​
(
c
t
,
f
t
,
s
t
)
\hat{z}_{t}^{T}=g_{T}(c_{t},f_{t},s_{t})
, whereas the causal auxiliary encoder uses only
(
c
t
,
s
t
)
(c_{t},s_{t})
and produces
z
^
t
C
=
g
C
​
(
c
t
,
s
t
)
\hat{z}_{t}^{C}=g_{C}(c_{t},s_{t})
. Both 64-D codes are sent
through a shared action decoder, and the teacher code is additionally sent to a
future decoder that reconstructs
f
t
−
c
t
f_{t}-c_{t}
. Teacher–causal alignment transfers
future information to the causal branch; variance and covariance regularization
keep every latent dimension active; and pairwise action geometry preserves
relative distances between examples. After training, only the per-dimension
whitened teacher output
z
t
T
z_{t}^{T}
is cached for causal-student supervision.
Fig. 2:
Offline action-equivalent future teacher.
The frozen latent
pipeline extracts current and future features
(
c
t
,
f
t
)
(c_{t},f_{t})
, and the adapter also
receives robot state
s
t
s_{t}
. The privileged encoder uses
(
c
t
,
f
t
,
s
t
)
(c_{t},f_{t},s_{t})
,
whereas the causal branch uses
(
c
t
,
s
t
)
(c_{t},s_{t})
. Both codes share an action decoder;
the teacher additionally reconstructs the future feature difference.
Teacher–causal alignment, variance/covariance regularization, and pairwise
action geometry prevent collapse and retain action relevance. The selected
64-D teacher code is whitened and cached as
z
t
T
z_{t}^{T}
.
III-C
Causal Temporal Student
Figure
1
(A) shows the causal input and its three prediction heads.
The deployment history contains eight 26-D normalized states and 12-D visual
statistics (per-channel means and standard deviations from two auxiliary
cameras). The current full-resolution images still enter the standard
π
0.5
\pi_{0.5}
vision backbone. For history index
i
i
, we compute
e
i
=
swish
⁡
(
W
s
​
LN
⁡
(
s
i
)
)
+
swish
⁡
(
W
v
​
LN
⁡
(
r
i
)
)
+
p
i
,
e_{i}=\operatorname{swish}(W_{s}\operatorname{LN}(s_{i}))+\operatorname{swish}(W_{v}\operatorname{LN}(r_{i}))+p_{i},
(3)
where
p
i
p_{i}
is a learned temporal embedding and
e
i
∈
ℝ
256
e_{i}\in\mathbb{R}^{256}
.
After flattening, a two-layer residual MLP produces
q
=
swish
(
W
p
vec
(
e
1
:
8
)
)
,
h
t
=
LN
(
swish
(
W
o
q
)
+
q
)
.
q=\operatorname{swish}(W_{p}\operatorname{vec}(e_{1:8})),\quad h_{t}=\operatorname{LN}(\operatorname{swish}(W_{o}q)+q).
(4)
Linear heads predict the compressed future state
z
^
t
∈
ℝ
64
\hat{z}_{t}\in\mathbb{R}^{64}
, four manipulation-phase logits
p
^
t
\hat{p}_{t}
, and a
sigmoid normalized time-to-transition
d
^
t
∈
[
0
,
1
]
\hat{d}_{t}\in[0,1]
; the latter is the
temporal-horizon signal shown in the figure.
The phase target is obtained from gripper-state transitions: approach,
grasp/release transition, transport while holding, and place/retract. The
time-to-transition target is the clipped, horizon-normalized distance to the
next gripper-state transition.
These low-cost labels expose event structure that is otherwise implicit in the
action chunk.
III-D
Dual-Path Conditioning
As illustrated in Fig.
1
(B), the student conditions the policy
through two complementary routes. The slow path maps
z
^
t
\hat{z}_{t}
to four
2048-D future tokens. The expected
embedding under the predicted phase distribution supplies a fifth token. The
five tokens are appended to the
π
0.5
\pi_{0.5}
prefix. The fast path maps
z
^
t
\hat{z}_{t}
and
d
^
t
\hat{d}_{t}
to a 1024-D residual that is added to every
action-expert suffix token and its adaptive normalization condition. Learned
scalar gates initialize both injections at
0.05
0.05
, limiting disruption of the
pretrained policy early in training.
The added modules contain 1.251M parameters, small relative to the dense
Gemma-2B VLM and 311M-parameter action expert. The Fast-WAM teacher, video
VAE, and offline adapter are absent from deployment.
III-E
Training Objective
The right side of Fig.
1
groups the supervision into five
functional objectives: action flow, time to transition, future-state alignment,
auxiliary action reconstruction, and manipulation-phase classification. The
future-state term contains both pointwise and relational components. Let
row-normalized predicted and teacher codes in a batch be
Z
¯
\bar{Z}
and
Z
¯
T
\bar{Z}^{T}
. We use pointwise cosine and relational geometry losses,
ℒ
cos
\displaystyle\mathcal{L}_{\mathrm{cos}}
=
1
B
​
∑
i
(
1
−
z
¯
i
⊤
​
z
¯
i
T
)
,
\displaystyle=\frac{1}{B}\sum_{i}(1-\bar{z}_{i}^{\top}\bar{z}_{i}^{T}),
(5)
ℒ
geo
\displaystyle\mathcal{L}_{\mathrm{geo}}
=
1
B
2
​
‖
Z
¯
​
Z
¯
⊤
−
Z
¯
T
​
(
Z
¯
T
)
⊤
‖
F
2
.
\displaystyle=\frac{1}{B^{2}}\left\|\bar{Z}\bar{Z}^{\top}-\bar{Z}^{T}(\bar{Z}^{T})^{\top}\right\|_{F}^{2}.
(6)
Cross-entropy supervises phase, Huber loss with
δ
=
0.1
\delta=0.1
supervises
time to transition, and a linear decoder from
z
^
t
\hat{z}_{t}
reconstructs the normalized
action chunk for
ℒ
ae
\mathcal{L}_{\mathrm{ae}}
. The total objective is
ℒ
=
ℒ
flow
+
0.20
​
ℒ
cos
+
0.02
​
ℒ
geo
+
0.04
​
ℒ
phase
+
0.04
​
ℒ
transition
+
0.05
​
ℒ
ae
.
\begin{split}\mathcal{L}={}&\mathcal{L}_{\mathrm{flow}}+0.20\mathcal{L}_{\mathrm{cos}}+0.02\mathcal{L}_{\mathrm{geo}}+0.04\mathcal{L}_{\mathrm{phase}}\\
&+0.04\mathcal{L}_{\mathrm{transition}}+0.05\mathcal{L}_{\mathrm{ae}}.\end{split}
(7)
All
π
0.5
\pi_{0.5}
parameters and ForeTime-VLA modules are optimized jointly; the
recorded action target is never replaced by a teacher action.
III-F
Design Rationale and Distinction
ForeTime-VLA transfers predictive structure rather than generated pixels. Its
privileged branch compresses future video into a code constrained by the
demonstrated action, filtering appearance changes that are irrelevant to
control. The causal student recovers this code from history alone, retaining
the temporal bias of a video model without future synthesis in the control loop.
Future and phase tokens expose the forecast to the semantic VLM path, while the
fast residual gives the action expert direct access to the predicted code and
transition horizon. This action-equivalent, event-aware interface—together
with an unchanged action target—distinguishes ForeTime-VLA from attaching a video
generator to the policy and makes it complementary to flow matching.
