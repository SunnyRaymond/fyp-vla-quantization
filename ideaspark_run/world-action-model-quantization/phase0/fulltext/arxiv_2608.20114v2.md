# DECOWAM: Decoupled Whole-Body World-Action Model for Legged Mobile Manipulation

paper_id: arxiv:2608.20114v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

A robot that can both
move through
and
act on
the physical world is the long-standing target of embodied AI research. Recent vision–language–action (VLA) and world–action models—Aloha
[
1
]
, RT-1/2
[
26
,
2
]
, OpenVLA
[
3
]
, PaLM-E
[
27
]
, SayCan
[
28
]
,
π
0
\pi_{0}
[
4
]
, RDT-2
[
5
]
, and Motus—have produced impressive results on stationary bimanual platforms, where the camera geometry is fixed and the action space is restricted to arm trajectories. Legged mobile manipulators change the modeling problem: the camera is carried by a moving base, and the policy must coordinate high-rate arm motion with lower-rate base velocity commands.
Fig. 1:
DECOWAM architecture, training, and deployment. (A) A frozen WAN backbone is augmented with trainable residual adapters, a teacher–student future bottleneck, separated base/arm latents, and base-velocity ego-motion conditioning; ActionDiT produces future RGB clips and a 48-step, 14-D action chunk. (B) Deployment removes future-frame inputs and the privileged teacher path, yielding strictly causal 14-D action inference. (C) Staged training first aligns pretrained FastWAM to
ARMDOG
and then adapts the decoupled modules to obtain the deployable policy.
Problem formulation.
Given an instruction
ℓ
\ell
, a current RGB observation
x
0
x_{0}
, and robot state
s
s
, a whole-body world–action model predicts a future action chunk
𝐚
^
1
:
K
\hat{\mathbf{a}}_{1:K}
and a future video clip
x
^
1
:
T
\hat{x}_{1:T}
. In our setting,
𝐚
1
:
K
∈
ℝ
K
×
14
\mathbf{a}_{1:K}\in\mathbb{R}^{K\times 14}
contains arm joints, gripper state, base velocity, and loader-compatible padding. The difficulty is structural. First, the camera coordinate system changes with the legged base, so apparent image motion mixes scene dynamics, arm motion, and ego-motion. Second, the action vector is multi-factor: arm/gripper channels and base-velocity channels have different semantics and different control time scales. Third, base velocity has a dual role: it is a target to be predicted by the action expert and an observation that explains camera motion for the video expert.
Why is legged mobile manipulation harder?
These properties make legged-arm modeling different from fixed-base manipulation:
•
Dynamic viewpoint.
The on-board camera moves with the base. Hand–eye geometry varies continuously, and image streams contain a mixture of ego-motion and scene motion. A mobile-manipulation world model needs a route for representing camera motion rather than treating all pixel displacement as scene dynamics.
•
Multi-rate action coupling.
Arm joint trajectories require high-rate control (
∼
\sim
15–30 Hz), while base velocity commands are typically issued at a lower rate (
∼
\sim
3–5 Hz). Concatenating both into a single uniformly sampled action chunk asks one representation to cover navigation-scale velocity and manipulation-scale joint corrections at the same time.
•
Hierarchical intent.
Real tasks interleave
where to go
with
how to act
. A single monolithic latent struggles to represent navigation-scale decisions and manipulation-scale corrections simultaneously.
Our approach.
The central idea of this paper is a decoupled modeling paradigm for mobile manipulation: a world–action model should represent
where the base moves
,
how the arm acts
, and
how the camera ego-motion changes future pixels
as explicit factors. We implement this paradigm on top of FastWAM, a Wan-2.2-based world–action backbone with a paired ActionDiT branch for action chunks. The action-equivalent future bottleneck supplies a modest causal training signal from privileged future latents. Staged frozen adaptation turns the method into a parameter-efficient system by keeping the base FastWAM prior fixed in the final stage and learning only residual robot-specific pathways. The base/arm dual latent with GRL is the main action-factorization mechanism, separating navigation-scale base commands from manipulation-scale arm commands. The base-velocity token is the explicit ego-motion interface for the video branch: it conditions future visual rollout on the current normalized base velocity rather than asking the video branch to infer camera motion only from pixels.
Dataset.
Method development is only half of the story. Legged mobile manipulation requires data in which visual change, base ego-motion, arm motion, and language intent are synchronized rather than recorded as separate logs. We therefore build
ARMDOG
, a real-robot data resource for a quadrupedal platform with a 6-DoF arm. Its contribution is the embodiment-complete model interface: each converted episode aligns a 15 Hz RGB video stream, a
T
×
14
T{\times}14
whole-body state/action tensor with explicit base and arm channels, natural-language instruction text, and a precomputed language embedding. The current converted world–action snapshot contains 217 episodes from 27 task folders and 56,041 synchronized frames. This data organization is what makes it possible to train and evaluate base/arm factorization, ego-motion-aware video conditioning, and whole-body world–action prediction within one replay protocol.
Contributions.
1.
We formulate legged mobile manipulation as a decoupled world–action modeling problem in which base control, arm manipulation, and camera ego-motion enter through explicit, semantically aligned interfaces.
2.
We realize this formulation in DECOWAM through frozen residual adaptation, causal distillation from privileged future latents, adversarial base/arm factorization, and base-velocity-conditioned video prediction.
3.
We introduce the synchronized
ARMDOG
dataset and evaluate DECOWAM through controlled replay and real-robot experiments, demonstrating parameter-efficient prediction, improved whole-body coordination, and stronger perturbation tolerance.

## Method

IV-A
Problem Formulation
Let
ℓ
\ell
denote a language instruction,
x
0
x_{0}
the current RGB observation, and
s
0
∈
ℝ
14
s_{0}\in\mathbb{R}^{14}
the current whole-body state. We learn the conditional joint model
p
θ
(
x
1
:
T
,
𝐚
1
:
K
∣
x
0
,
s
0
,
ℓ
)
,
p_{\theta}\!\left(x_{1:T},\mathbf{a}_{1:K}\mid x_{0},s_{0},\ell\right),
(1)
where
x
1
:
T
x_{1:T}
is a future video and
𝐚
1
:
K
∈
ℝ
K
×
14
\mathbf{a}_{1:K}\in\mathbb{R}^{K\times 14}
is an action chunk. We use
T
=
8
T{=}8
and
K
=
48
K{=}48
. Each action has the semantic decomposition
𝐚
k
=
[
𝐚
k
arm
,
a
k
grip
,
𝐚
k
base
,
𝐚
k
pad
]
,
𝐚
k
base
∈
ℝ
3
.
\mathbf{a}_{k}=\left[\mathbf{a}^{\mathrm{arm}}_{k},a^{\mathrm{grip}}_{k},\mathbf{a}^{\mathrm{base}}_{k},\mathbf{a}^{\mathrm{pad}}_{k}\right],\qquad\mathbf{a}^{\mathrm{base}}_{k}\in\mathbb{R}^{3}.
(2)
The arm and gripper occupy channels
[
0
:
7
]
[0{:}7]
, base velocity occupies
[
7
:
10
]
[7{:}10]
, and loader padding occupies
[
10
:
14
]
[10{:}14]
. Base velocity is both a control target and a source of camera ego-motion. DECOWAM therefore separates arm control, base control, and visual ego-motion instead of encoding them in one undifferentiated context.
Our backbone is FastWAM, which pairs an ActionDiT action expert with a WAN video expert. Both experts receive language and proprioceptive context. The action expert predicts flow over action chunks, while the video expert predicts flow over future visual latents. DECOWAM preserves this joint interface and changes how embodiment-specific information conditions each expert. Future observations supervise the model only during training, so deployment remains causal in
(
x
0
,
s
0
,
ℓ
)
(x_{0},s_{0},\ell)
.
IV-B
Staged Parameter-Efficient Adaptation
Training separates domain alignment from structural adaptation. In Stage 1, all FastWAM parameters
Θ
\Theta
are adapted to
ARMDOG
for 50k steps:
Θ
(
1
)
=
arg
⁡
min
Θ
​
𝔼
𝒟
​
[
ℒ
video
​
(
Θ
)
+
ℒ
action
​
(
Θ
)
]
.
\Theta^{(1)}=\arg\min_{\Theta}\mathbb{E}_{\mathcal{D}}\left[\mathcal{L}_{\mathrm{video}}(\Theta)+\mathcal{L}_{\mathrm{action}}(\Theta)\right].
(3)
This stage aligns the video prior, action expert, and proprioceptive interface with the moving-camera observations and quadruped–arm action space.
Stage 2 freezes
Θ
(
1
)
\Theta^{(1)}
and optimizes only
Φ
=
{
ϕ
adp
,
ϕ
q
,
ϕ
ba
,
ϕ
ego
}
,
Φ
⋆
=
arg
⁡
min
Φ
⁡
ℒ
⁡
(
Θ
(
1
)
,
Φ
)
.
\Phi=\left\{\phi_{\mathrm{adp}},\phi_{q},\phi_{\mathrm{ba}},\phi_{\mathrm{ego}}\right\},\qquad\Phi^{\star}=\arg\min_{\Phi}\mathcal{L}\!\left(\Theta^{(1)},\Phi\right).
(4)
The four parameter groups represent residual adapters, an action-equivalent future bottleneck, base/arm factorization, and ego-motion conditioning. This restriction reduces the Stage-2 trainable footprint from 6020.75M to 25.95M parameters.
The frozen WAN backbone is adapted after each block through
h
l
+
=
h
l
+
α
l
​
W
up
(
l
)
​
σ
​
(
W
down
(
l
)
​
LN
​
(
h
l
)
)
,
h_{l}^{+}=h_{l}+\alpha_{l}W_{\mathrm{up}}^{(l)}\sigma\!\left(W_{\mathrm{down}}^{(l)}\mathrm{LN}(h_{l})\right),
(5)
where
W
down
(
l
)
W_{\mathrm{down}}^{(l)}
projects to a 128-D bottleneck,
W
up
(
l
)
W_{\mathrm{up}}^{(l)}
restores the hidden dimension, and
σ
\sigma
is SiLU. The residual branch learns a compact, robot-specific correction while preserving the pretrained video prior.
IV-C
Decoupled Conditional Interfaces
Action-equivalent future bottleneck.
Future frames contain information about action-equivalent outcomes that is unavailable from the current frame alone. We distill this privileged information from a teacher into a causal student.
Fig. 2:
Future-information bottleneck. A privileged teacher observes current and future visual summaries, whereas the causal student observes only the current summary and robot state. Only the student is retained during deployment.
Let
e
0
=
ψ
vae
​
(
x
0
)
e_{0}=\psi_{\mathrm{vae}}(x_{0})
and
e
1
:
T
=
ψ
vae
(
x
1
:
T
)
e_{1:T}=\psi_{\mathrm{vae}}(x_{1:T})
be WAN-VAE latents. We summarize each latent tensor using
c
=
ρ
(
e
0
)
,
f
=
ρ
(
e
1
:
T
)
,
ρ
(
e
)
=
[
mean
(
e
)
,
std
(
e
)
]
.
c=\rho(e_{0}),\qquad f=\rho(e_{1:T}),\qquad\rho(e)=\left[\operatorname{mean}(e),\operatorname{std}(e)\right].
(6)
The teacher and student embeddings are
z
t
=
q
t
(
[
c
,
f
,
s
0
]
)
,
z
s
=
q
s
(
[
c
,
s
0
]
)
,
z
t
,
z
s
∈
ℝ
d
q
.
z_{t}=q_{t}([c,f,s_{0}]),\qquad z_{s}=q_{s}([c,s_{0}]),\qquad z_{t},z_{s}\in\mathbb{R}^{d_{q}}.
(7)
Only
z
s
z_{s}
conditions the causal action expert:
u
~
a
=
u
a
+
η
q
​
B
q
​
z
s
,
\tilde{u}^{a}=u^{a}+\eta_{q}B_{q}z_{s},
(8)
where
u
a
u^{a}
denotes its context tokens and
η
q
∈
[
0
,
1
]
\eta_{q}\in[0,1]
controls the residual bias.
The bottleneck is trained with action reconstruction, teacher–student distillation, and geometry preservation:
ℒ
rec
q
=
\displaystyle\mathcal{L}_{\mathrm{rec}}^{q}={}
‖
r
s
(
z
s
)
−
𝐚
1
:
K
‖
2
2
\displaystyle\left\|r_{s}(z_{s})-\mathbf{a}_{1:K}\right\|_{2}^{2}
+
‖
r
t
(
z
t
)
−
𝐚
1
:
K
‖
2
2
,
\displaystyle+\left\|r_{t}(z_{t})-\mathbf{a}_{1:K}\right\|_{2}^{2},
ℒ
q
=
\displaystyle\mathcal{L}_{q}={}
λ
act
q
​
ℒ
rec
q
+
λ
dist
q
​
‖
z
s
−
sg
⁡
(
z
t
)
‖
2
2
\displaystyle\lambda_{\mathrm{act}}^{q}\mathcal{L}_{\mathrm{rec}}^{q}+\lambda_{\mathrm{dist}}^{q}\left\|z_{s}-\operatorname{sg}(z_{t})\right\|_{2}^{2}
+
λ
geom
q
​
ℒ
geom
,
\displaystyle+\lambda_{\mathrm{geom}}^{q}\mathcal{L}_{\mathrm{geom}},
(9)
where
sg
\operatorname{sg}
stops gradients. For a batch of size
B
B
, the geometry term is
d
z
i
​
j
=
‖
z
t
i
−
z
t
j
‖
2
τ
z
,
d
a
i
​
j
=
∥
𝐚
i
1
:
K
−
𝐚
j
1
:
K
∥
2
τ
a
,
d_{z}^{ij}=\frac{\|z_{t}^{i}-z_{t}^{j}\|_{2}}{\tau_{z}},\qquad d_{a}^{ij}=\frac{\|\mathbf{a}^{i}_{1:K}-\mathbf{a}^{j}_{1:K}\|_{2}}{\tau_{a}},
(10)
ℒ
geom
\displaystyle\mathcal{L}_{\mathrm{geom}}
=
(
B
⁡
(
B
−
1
)
)
−
1
\displaystyle={}\bigl(B(B-1)\bigr)^{-1}
(11)
×
∑
i
≠
j
SL1
⁡
(
d
z
i
​
j
,
d
a
i
​
j
)
.
\displaystyle\times\sum_{i\neq j}\operatorname{SL1}\!\left(d_{z}^{ij},d_{a}^{ij}\right).
with robust scales
τ
z
\tau_{z}
and
τ
a
\tau_{a}
obtained from batch medians. Thus, trajectories with similar normalized actions are encouraged to remain close in the privileged latent space. Distillation transfers this structure to the deployable student.
Base–arm factorization.
The action context contains navigation-scale and manipulation-scale information. We map its pooled representation into two 16-D factors:
z
base
=
b
ϕ
(
u
a
)
,
z
arm
=
m
ϕ
(
u
a
)
,
z
base
,
z
arm
∈
ℝ
16
.
z_{\mathrm{base}}=b_{\phi}(u^{a}),\qquad z_{\mathrm{arm}}=m_{\phi}(u^{a}),\qquad z_{\mathrm{base}},z_{\mathrm{arm}}\in\mathbb{R}^{16}.
(12)
Their concatenation conditions the action expert through
u
¯
a
=
u
~
a
+
η
ba
​
B
ba
​
[
z
base
,
z
arm
]
.
\bar{u}^{a}=\tilde{u}^{a}+\eta_{\mathrm{ba}}B_{\mathrm{ba}}[z_{\mathrm{base}},z_{\mathrm{arm}}].
(13)
Let
𝐚
b
1
:
K
=
𝐚
1
:
K
,
7
:
10
\mathbf{a}^{b}_{1:K}=\mathbf{a}_{1:K,7:10}
and
𝐚
m
1
:
K
=
𝐚
1
:
K
,
0
:
7
\mathbf{a}^{m}_{1:K}=\mathbf{a}_{1:K,0:7}
. Direct heads preserve the assigned factor, while gradient-reversal cross heads suppress information about the opposite factor:
ℒ
disent
=
\displaystyle\mathcal{L}_{\mathrm{disent}}={}
‖
g
b
(
z
base
)
−
𝐚
1
:
K
b
‖
2
2
+
‖
g
m
(
z
arm
)
−
𝐚
1
:
K
m
‖
2
2
\displaystyle\left\|g_{b}(z_{\mathrm{base}})-\mathbf{a}^{b}_{1:K}\right\|_{2}^{2}+\left\|g_{m}(z_{\mathrm{arm}})-\mathbf{a}^{m}_{1:K}\right\|_{2}^{2}
+
‖
g
~
b
(
GRL
(
z
arm
)
)
−
𝐚
1
:
K
b
‖
2
2
\displaystyle+\left\|\tilde{g}_{b}(\operatorname{GRL}(z_{\mathrm{arm}}))-\mathbf{a}^{b}_{1:K}\right\|_{2}^{2}
+
‖
g
~
m
(
GRL
(
z
base
)
)
−
𝐚
1
:
K
m
‖
2
2
.
\displaystyle+\left\|\tilde{g}_{m}(\operatorname{GRL}(z_{\mathrm{base}}))-\mathbf{a}^{m}_{1:K}\right\|_{2}^{2}.
(14)
The prediction heads minimize all reconstruction terms, whereas gradient reversal changes the sign of cross-task gradients entering the encoders. Each latent is therefore encouraged to retain its assigned control factor and discard the other.
Ego-motion-aware video conditioning.
For a body-mounted camera, apparent image motion combines scene dynamics, manipulator motion, and base-induced viewpoint change. We expose the last component using the normalized current base velocity
v
0
=
Π
base
​
(
s
0
)
=
(
v
x
,
v
y
,
ω
z
)
∈
ℝ
3
.
v_{0}=\Pi_{\mathrm{base}}(s_{0})=(v_{x},v_{y},\omega_{z})\in\mathbb{R}^{3}.
(15)
Each video token receives the same projected ego-motion condition:
h
~
i
v
=
h
i
v
+
β
B
v
v
0
,
i
=
1
,
…
,
N
v
,
\tilde{h}_{i}^{v}=h_{i}^{v}+\beta B_{v}v_{0},\qquad i=1,\ldots,N_{v},
(16)
where
B
v
B_{v}
maps velocity into the WAN hidden dimension. This token does not impose geometric warping. It supplies an explicit explanatory variable for camera-frame motion. Base velocity consequently acts as an action target in Eq. (
2
) and a visual condition in Eq. (
16
).
IV-D
Training Objective and Deployment
Both experts use conditional flow matching
[
48
]
. For a target
y
∈
{
𝐚
1
:
K
,
e
1
:
T
}
y\in\{\mathbf{a}_{1:K},e_{1:T}\}
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
, and time
τ
∼
𝒰
⁡
(
0
,
1
)
\tau\sim\mathcal{U}(0,1)
, define
y
τ
=
(
1
−
τ
)
​
ϵ
+
τ
​
y
,
v
⋆
​
(
y
τ
,
τ
)
=
y
−
ϵ
.
y_{\tau}=(1-\tau)\epsilon+\tau y,\qquad v^{\star}(y_{\tau},\tau)=y-\epsilon.
(17)
The corresponding objective is
ℒ
FM
​
(
F
θ
,
y
,
c
)
=
𝔼
τ
,
ϵ
​
[
‖
F
θ
​
(
y
τ
,
τ
,
c
)
−
v
⋆
​
(
y
τ
,
τ
)
‖
2
2
]
.
\mathcal{L}_{\mathrm{FM}}(F_{\theta};y,c)=\mathbb{E}_{\tau,\epsilon}\left[\left\|F_{\theta}(y_{\tau},\tau,c)-v^{\star}(y_{\tau},\tau)\right\|_{2}^{2}\right].
(18)
We instantiate this loss as
ℒ
action
\mathcal{L}_{\mathrm{action}}
with context
u
¯
a
\bar{u}^{a}
and as
ℒ
video
\mathcal{L}_{\mathrm{video}}
with context
h
~
v
\tilde{h}^{v}
. The complete Stage-2 objective is
ℒ
=
λ
v
​
ℒ
video
+
λ
a
​
ℒ
action
+
γ
q
​
λ
q
​
ℒ
q
+
γ
ba
​
λ
ba
​
ℒ
disent
.
\mathcal{L}=\lambda_{v}\mathcal{L}_{\mathrm{video}}+\lambda_{a}\mathcal{L}_{\mathrm{action}}+\gamma_{q}\lambda_{q}\mathcal{L}_{q}+\gamma_{\mathrm{ba}}\lambda_{\mathrm{ba}}\mathcal{L}_{\mathrm{disent}}.
(19)
We set
λ
v
=
λ
a
=
1.0
\lambda_{v}=\lambda_{a}=1.0
,
λ
q
=
0.2
\lambda_{q}=0.2
, and
λ
ba
=
0.1
\lambda_{\mathrm{ba}}=0.1
. The reported run sets
γ
q
\gamma_{q}
,
γ
ba
\gamma_{\mathrm{ba}}
,
η
q
\eta_{q}
, and
η
ba
\eta_{\mathrm{ba}}
to one throughout Stage 2.
At inference, the teacher
q
t
q_{t}
and all auxiliary prediction heads are removed. The model computes
z
s
z_{s}
,
(
z
base
,
z
arm
)
(z_{\mathrm{base}},z_{\mathrm{arm}})
, and
v
0
v_{0}
from current inputs, then samples both flows using only
(
x
0
,
s
0
,
ℓ
)
(x_{0},s_{0},\ell)
.
