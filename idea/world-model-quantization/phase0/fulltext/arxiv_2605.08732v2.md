# Latent Geometry Beyond Search: Amortizing Planning in World Models

paper_id: arxiv:2605.08732v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

The utility of world models in control ultimately depends on their ability to facilitate efficient decision-making.
Predicting future observations or latent states is useful, but it does not by itself solve the control problem: an agent must still convert those predictions into actions. In many current systems, this final step remains
computationally
expensive. Learned representations make forecasting accurate and fast, yet action selection still relies on substantial online optimization. As a result, there is often a gap between learning a good model of the world and using that model to act efficiently.
The LeWorldModel
(LeWM)
(
Maes et al., 2026
)
is an attractive starting point because it is already efficient as a world model: with
∼
15
{\sim}15
M parameters trainable on a single GPU in a few hours, it plans up to
48
×
48\times
faster than foundation-model-based world models while remaining competitive across diverse control tasks. Yet even in LeWM, control is still implemented through CEM, which requires
9,000
9{,}000
candidate rollouts (
45,000
45{,}000
predictor forward passes) for every planning step. This means that the dominant computational cost no longer comes from modeling the dynamics; it comes from searching over candidate action sequences. We refer to this mismatch between cheap prediction and expensive decision-making as the
planning tax
.
This raises a basic question: is
this
tax
intrinsic, or is it
primarily
an artifact of how control is formulated? If the latent space is poorly organized, then search may indeed remain necessary, because the mapping from desired future states to actions is unstable and nonlocal. But if the latent space is smooth and action-sensitive, properties that regularizers like SIGReg are designed to encourage, then action selection may admit a simpler description. In that case, planning can be viewed not as a generic combinatorial search problem, but as a local inverse problem induced by representation regularity.
We study this perspective in the setting of a pretrained JEPA world model. In this paper, we replace online search with a small goal-conditioned inverse dynamics model
(GC-IDM)
trained on frozen LeWM latents. The model takes the current latent state, the goal latent state, and the remaining horizon, and predicts the next action directly.
Our
central question is
whether inverse dynamics
serves as
a sufficiently faithful abstraction of control for well-regularized latent representations
.
Across four benchmark environments,
the proposed GC-IDM
matches or exceeds CEM in seven of
the
eight environment–protocol cells while reducing
per-decision
planning cost by
100
100
–
130
×
130\times
. The same conclusion holds under a strict episode-level holdout, under a broad CEM compute sweep across a
500
×
500\times
range, and across
a
broader family of test-time planners including CEM
(
Rubinstein, 1999
)
, iCEM
(
Pinneri et al., 2021
)
, MPPI
(
Williams et al., 2017
)
,
and
gradient-based trajectory optimization
(
Hansen et al., 2022
;
Hansen et al., 2024
)
. Together, these results suggest that, in this regime, the planner is recovering structure that the representation has already made locally accessible.
On the empirical side, we confirm that a small inverse dynamics model trained by supervised regression on frozen world model latents can replace test-time search
while achieving comparable or superior success rates
across a broad range of benchmark conditions
and maintaining
high planning efficiency.
Our main contributions are:
1)
We propose GC-IDM, a learned planning method for JEPA world models that recasts control as an inverse problem in latent space, replacing iterative test-time search with a single forward pass per environment step.
2)
We empirically demonstrate that a
lightweight
goal-conditioned inverse model can outperform the broader family of test-time planners (including CEM, MPPI, iCEM, and gradient-based methods) on the LeWM benchmarks, achieve strong performance relative to prior baselines, and reduce planning cost by
100
100
–
130
×
130\times
.
3)
Our experiments confirm that learning latent inverse dynamics on well-regularized latent representations enables fast
, effective
closed-loop planning.

## Method

Figure 1
:
Evolution of Push-T latent geometry across training.
Panels (a)–(f) show two-dimensional t-SNE embeddings of latent states from Push-T sequence at epochs 1, 2, 4, 6, 8, and 10, with points colored by frame index. Panel (g) shows subsampled observation frames from the same sequence. Panel (h) shows the standardized marginal latent distribution at
t
=
0
t{=}0
for epoch 10, together with a Gaussian reference curve. Across training, the embedding evolves from a diffuse cloud into a smooth ordered path aligned with task progress, providing qualitative support for the claim that the learned representation is structured enough for local inverse recovery.
Figure 2
:
Pipeline overview.
Left:
World model encoder training process, which follows LeWM
(
Maes et al., 2026
)
.
Center:
Goal-conditioned inverse dynamics model (GC-IDM) training. From a trajectory
τ
∼
𝒟
\tau\sim\mathcal{D}
, a tuple
(
𝒛
t
,
𝒛
g
,
h
,
𝒂
t
)
(\bm{z}_{t},\bm{z}_{g},h,\bm{a}_{t})
is sampled at random horizon
h
∈
[
1
,
H
max
]
h\in[1,H_{\max}]
using frozen LeWM embeddings; the IDM is trained by MSE regression with gradients flowing only into the inverse dynamics module, i.e.,
GC-IDM
ψ
\text{GC-IDM}_{\psi}
.
Right:
GC-IDM planning. The current observation
o
t
o_{t}
is re-encoded at every step; the goal embedding
z
g
z_{g}
is encoded once and cached. The concatenated pair
𝒛
t
|
𝒛
g
\bm{z}_{t}\|\bm{z}_{g}
passes through a
3
3
-layer MLP modulated by the remaining horizon
h
t
h_{t}
via AdaLN-Zero, producing estimated action
𝒂
^
t
\hat{\bm{a}}_{t}
in a single forward inference pass with no search or rollout.
4.1
Overview
Motivation.
Our motivation is to remove the planning tax without
altering
the pretrained world model. A useful failure case clarifies
our
design: our first attempt (Appendix
D
) was a
pairwise
IDM that decoded actions from consecutive latent pairs
(
𝒛
t
,
𝒛
t
+
1
)
→
𝒂
t
(\bm{z}_{t},\bm{z}_{t+1})\to\bm{a}_{t}
and then planned by linearly interpolating between start and goal embeddings. That model achieved near-perfect oracle reconstruction (
R
2
=
0.993
R^{2}=0.993
) but poor planning. This supports the idea that bottleneck is not decoding actions from well-posed local transitions; it is constructing a valid latent path between distant states.
Latent geometry intuition.
If the pretrained latent geometry is already smooth and task-aligned, then the controller does not need to invent an entire trajectory between distant states; it only needs to recover the locally appropriate action that moves the current state toward the goal under a finite budget. Figure
1
provides qualitative evidence for this picture on Push-T by showing how the t-SNE embedding of one sequence evolves across training for a pretrained LeWM, together with subsampled observations and a marginal latent histogram. Across epochs, the latent states organize from a diffuse cloud into a smooth ordered path aligned with task evolution, suggesting that nearby latent displacements correspond to locally coherent behavioral changes. That is, it supports the motivating hypothesis that sufficiently regular latent geometry can make local inverse recovery easy to amortize.
Controller.
Our replacement for search is, therefore, a goal-conditioned inverse dynamics operator that acts directly on the current latent state, the goal latent state, and the remaining horizon. Given observations
(
𝒐
t
,
𝒐
goal
)
(\bm{o}_{t},\bm{o}_{\text{goal}})
, we encode them with the frozen encoder and predict the next action by
𝒛
t
=
enc
θ
​
(
𝒐
t
)
,
𝒛
goal
=
enc
θ
​
(
𝒐
goal
)
,
𝒂
^
t
=
gc
​
-
​
idm
ψ
​
(
𝒛
t
,
𝒛
goal
,
h
t
)
.
\bm{z}_{t}=\mathrm{enc}_{\theta}(\bm{o}_{t}),\quad\bm{z}_{\text{goal}}=\mathrm{enc}_{\theta}(\bm{o}_{\text{goal}}),\quad\hat{\bm{a}}_{t}=\mathrm{gc\text{-}idm}_{\psi}\!\bigl(\bm{z}_{t},\,\bm{z}_{\text{goal}},\,h_{t}\bigr).
(5)
Equation (
5
) is the full inference-time model. Relative to pairwise interpolation, it avoids imagined latent trajectories and conditions only on real encoder outputs from actual observations. The resulting policy is still goal-conditioned, but it shifts the problem from online search over action sequences to offline amortization of the local inverse map induced by the latent world model.
4.2
Model Architecture and Details
Algorithm 1
Closed-loop goal-reaching control induced by GC-IDM
Input:
Environment
ℰ
\mathcal{E}
, encoder
enc
θ
\mathrm{enc}_{\theta}
, inverse policy
gc
​
-
​
idm
ψ
\mathrm{gc\text{-}idm}_{\psi}
, goal observation
𝒐
g
\bm{o}_{g}
, horizon budget
T
T
Output:
Closed-loop action sequence
(
𝒂
1
,
…
,
𝒂
τ
)
(\bm{a}_{1},\ldots,\bm{a}_{\tau})
, where
τ
≤
T
\tau\leq T
𝒛
g
←
enc
θ
​
(
𝒐
g
)
\bm{z}_{g}\leftarrow\mathrm{enc}_{\theta}(\bm{o}_{g})
reset or initialize
ℰ
\mathcal{E}
and observe initial observation
𝒐
1
\bm{o}_{1}
for
t
←
1
t\leftarrow 1
to
T
T
do
𝒛
t
←
enc
θ
​
(
𝒐
t
)
\bm{z}_{t}\leftarrow\mathrm{enc}_{\theta}(\bm{o}_{t})
h
t
←
T
−
t
+
1
h_{t}\leftarrow T-t+1
𝒂
t
←
gc
​
-
​
idm
ψ
​
(
𝒛
t
,
𝒛
g
,
h
t
)
\bm{a}_{t}\leftarrow\mathrm{gc\text{-}idm}_{\psi}(\bm{z}_{t},\bm{z}_{g},h_{t})
apply
𝒂
t
\bm{a}_{t}
to
ℰ
\mathcal{E}
and observe next observation
𝒐
t
+
1
\bm{o}_{t+1}
if
goal reached or episode terminated
then
return
(
𝒂
1
,
…
,
𝒂
t
)
(\bm{a}_{1},\ldots,\bm{a}_{t})
end if
end for
return
(
𝒂
1
,
…
,
𝒂
T
)
(\bm{a}_{1},\ldots,\bm{a}_{T})
We
purposefully
instantiate Equation (
5
) with a
lightweight
neural network (Figure
2
) so that any gain over CEM is attributable to latent geometry and control structure rather than raw function capacity. The backbone is a 3-layer MLP with hidden dimension
512
512
, LayerNorm, GELU activation, and
10
%
10\%
dropout, with the two embeddings concatenated as
𝒛
t
|
𝒛
goal
\bm{z}_{t}\,\|\,\bm{z}_{\text{goal}}
at the input. A final linear head maps the modulated representation to the action space. Total parameters:
∼
1.5
{\sim}1.5
M, roughly
10
%
10\%
of the LeWM backbone and orders of magnitude smaller than the
∼
10
{\sim}10
M-parameter predictor that CEM invokes
45,000
45{,}000
times per plan call. Given a pretrained LeWM checkpoint, GC-IDM can be trained from scratch in approximately 20 minutes per environment on a single GPU, making it a lightweight extension of an already efficient world model.
The remaining-horizon variable modulates the inverse map rather than being concatenated as an undifferentiated observation feature. Concretely, the remaining-step count is first normalized as
h
t
=
min
⁡
(
steps_remaining
,
H
max
)
/
H
max
∈
[
0
,
1
]
h_{t}=\min(\text{steps\_remaining},H_{\text{max}})/H_{\text{max}}\in[0,1]
, sinusoidally encoded (
64
64
dimensions), and passed through a 2-layer MLP to produce an embedding
𝒄
\bm{c}
. Two separate linear projections
γ
⁡
(
𝒄
)
,
β
⁡
(
𝒄
)
\gamma(\bm{c}),\beta(\bm{c})
then modulate the backbone output
𝒉
\bm{h}
via zero-initialized AdaLN-Zero modulation
(
Peebles and Xie, 2023
)
:
AdaLN
0
​
(
𝒉
,
𝒄
)
=
𝒉
⊙
(
𝟏
+
γ
⁡
(
𝒄
)
)
+
β
⁡
(
𝒄
)
,
\mathrm{AdaLN}_{0}(\bm{h},\bm{c})\;=\;\bm{h}\,\odot\,\bigl(\mathbf{1}+\gamma(\bm{c})\bigr)\;+\;\beta(\bm{c}),
(6)
applied immediately before the action head. The two projection layers
γ
⁡
(
⋅
)
,
β
⁡
(
⋅
)
\gamma(\cdot),\beta(\cdot)
have their weights and biases zero-initialized, so at the start of training
γ
=
β
=
0
\gamma{=}\beta{=}\textbf{0}
, ensuring the horizon signal is introduced only as the regression loss demands it.
Training on frozen embeddings.
Training uses the same offline demonstration dataset that trained LeWM; no additional environment interaction is required. Each training example is induced from a trajectory by sampling
(
t
,
h
)
(t,h)
, with
h
h
drawn uniformly from
[
1
,
H
max
]
[1,H_{\text{max}}]
, and forming the triple
(
𝒛
t
,
𝒛
t
+
h
,
𝒂
t
)
(\bm{z}_{t},\bm{z}_{t+h},\bm{a}_{t})
. The loss function is
ℒ
gc-idm
​
(
ψ
)
=
𝔼
(
t
,
h
)
∼
𝒟
​
[
‖
gc
​
-
​
idm
ψ
​
(
𝒛
t
,
𝒛
t
+
h
,
h
)
−
𝒂
t
‖
2
2
]
.
\mathcal{L}_{\text{gc-idm}}(\psi)=\mathbb{E}_{(t,h)\sim\mathcal{D}}\left[\,\bigl\|\mathrm{gc\text{-}idm}_{\psi}(\bm{z}_{t},\bm{z}_{t+h},h)-\bm{a}_{t}\bigr\|_{2}^{2}\,\right].
(7)
Closed-loop control at test time.
At test time, the trained inverse map replaces the entire iterative optimization loop. Control is implemented as a receding-horizon inverse policy: given a goal observation
𝒐
g
\bm{o}_{g}
and a budget
T
T
, the controller encodes the goal once, then repeatedly re-encodes the current observation, evaluates the inverse map at the current latent state and remaining horizon, and applies the resulting action to the environment. The remaining horizon is computed as
h
t
=
T
−
t
+
1
h_{t}=T-t+1
and clamped to
H
max
H_{\text{max}}
(matching the training distribution); in all experiments
H
max
=
T
=
50
H_{\text{max}}=T=50
. There is no inner optimization loop and no open-loop commitment to a predicted latent trajectory. Algorithm
1
states the control law formally.
Algorithm
1
is a closed-loop controller: after each action, it re-encodes the actual observation and recomputes the next move from the true current state. In our implementation, the added computation is a
∼
1.5
{\sim}1.5
M-parameter MLP on top of encoder calls already required by LeWM. Intuitively, the latent goal distance
‖
𝒛
t
−
𝒛
g
‖
\|\bm{z}_{t}-\bm{z}_{g}\|
acts as a potential field, and GC-IDM approximately follows its gradient at each step; per-step replanning accumulates these directionally correct predictions into successful trajectories.
