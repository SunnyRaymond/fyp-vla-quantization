# FF-JEPA: Long-Horizon Planning in World Models with Latent Planners

paper_id: semanticscholar:11c3d2fc3bd55e3d735e6f6034d94266fe530587
tier: T2
source_used: html_arxiv
warning: none

## Intro

Fig. 1
:
A conceptual visualization of planning with our approach.
Given the latent of the current observation or a history of observations, the latent planner
G
G
predicts the next subgoal latent for the world model. This subgoal is then used during the rollout of the predictor
P
P
to optimize the action sequence. This enables inference with world models without the need for a goal image.
Learning world models that enable agents to plan and act in complex environments has become a trend in modern reinforcement learning and robotics. Recent advances in visual world models, such as LeWorldModel
[
5
]
, DINO-based world models
[
13
]
, and related architectures, have demonstrated promising capabilities in predicting future observations and supporting model-based control. These approaches learn latent dynamics that allow agents to “imagine” trajectories and optimize actions via sampling-based planners such as the Cross-Entropy Method (CEM)
[
8
]
. Despite this progress, two key limitations hinder their deployment in real-world, long-horizon tasks. First, planning over long horizons remains computationally prohibitive due to the need for repeated rollouts and the compounding of prediction errors. Second, most existing approaches require an explicit goal specification in the form of a target image or state, which is often unavailable or impractical in real-world scenarios.
A growing body of work has attempted to address the long-horizon planning problem through hierarchical decomposition or more efficient imagination. For example, hierarchical foresight methods generate intermediate subgoals to break tasks into manageable segments
[
6
,
11
]
. More recent approaches improve planning efficiency by reducing the cost of latent rollouts, for instance via sparse imagination over subsets of visual tokens
[
3
]
. Generative approaches have also been proposed to guide planning. For example,
Ziakas et al. [14]
leverage video generation models to propose feasible trajectories. However, despite these advances, such methods still operate in pixel space or rely on externally specified goal images or trajectories, limiting their applicability in open-ended environments.
Closer to our approach, recent work has revisited the role of inverse dynamics models and their interaction with forward models. Predictive inverse dynamics models (PIDMs) like latent diffusion planning
[
10
]
train an action-free forward dynamics model and an inverse dynamics model separately, enabling training of the latent planner on unlabelled trajectories. PIDMs have also been shown to require fewer demonstrations to reach comparable performance to standard behavior cloning
[
9
]
. From a vision-language-action perspective,
Zhang et al. [12]
disentangle the pretraining of forward and inverse dynamics models to improve representation learning and downstream control. These perspectives suggest an alternative view in which the planner operates over predicted future states, rather than directly mapping observations to actions.
In this work, we propose forward-forward JEPA (FF-JEPA), a unified framework that bridges world models, forward prediction, and inverse dynamics inference to enable goal-directed behavior without requiring explicit goal images. Leveraging a pretrained world model
[
5
]
, we train an action-free forward model within the world model’s latent space, referred to as the latent planner. This planner forecasts trajectories toward implicitly defined objectives. Rather than learning a separate control policy, we repurpose the world model as an inference engine to extract action sequences through sampling-based optimization. In contrast to prior approaches, we reinterpret the world model as an inverse dynamics module operating over imagined latent trajectories, effectively unifying predictive modeling and control within a single, coherent latent space. Our framework addresses the challenges of long-horizon planning and the reliance on explicit goal observations in current world models. This positions our approach as an alternative towards policies that do not strictly require action-labeled demonstrations; if a pretrained world model is available, the latent planner can be trained on unlabeled data.

## Method

(a)
Latent deterministic planner
(b)
Latent diffusion planner
Fig. 2
:
Training schemes for the two architectures we evaluated.
Both models are trained on the latent space defined by the world model’s frozen encoder.
II-A
Preliminaries: JEPA-style world models
We build our method on top of the LeWM JEPA world model
[
5
]
, which consists of an encoder
E
E
and a forward dynamics predictor
P
P
. Given an observed frame
𝐨
t
\mathbf{o}_{t}
, the encoder produces a latent state
𝐳
t
=
E
⁡
(
𝐨
t
)
\mathbf{z}_{t}=E(\mathbf{o}_{t})
. The predictor takes a sliding window of at most
W
P
W_{P}
consecutive latent states and outputs the next predicted state:
𝐳
^
t
+
1
=
P
(
𝐳
t
−
W
P
+
1
:
t
,
𝐚
t
)
,
\hat{\mathbf{z}}_{t+1}=P\!\left(\mathbf{z}_{t-W_{P}+1:t},\,\mathbf{a}_{t}\right),
where
𝐚
t
\mathbf{a}_{t}
is the action taken at time
t
t
. In other words,
P
P
is an action-conditioned forward dynamics model that predicts how an action transforms the environment state in latent space.
The world model can in principle be used for goal-conditioned planning. Given an initial observation
𝐨
1
\mathbf{o}_{1}
and a goal observation
𝐨
g
\mathbf{o}_{g}
, we encode both as
𝐳
1
=
E
⁡
(
𝐨
1
)
\mathbf{z}_{1}=E(\mathbf{o}_{1})
and
𝐳
g
=
E
⁡
(
𝐨
g
)
\mathbf{z}_{g}=E(\mathbf{o}_{g})
, and search for a sequence of
H
H
actions
𝐚
^
1
:
H
\mathbf{\hat{a}}_{1:H}
that drives the predicted state towards the goal:
𝐚
^
1
:
H
=
arg
min
𝐚
1
:
H
‖
𝐳
g
−
P
AR
(
𝐳
1
,
𝐚
1
:
H
)
‖
2
2
,
\hat{\mathbf{a}}_{1:H}=\arg\min_{\mathbf{a}_{1:H}}\left\|\mathbf{z}_{g}-P_{\text{AR}}(\mathbf{z}_{1},\mathbf{a}_{1:H})\right\|_{2}^{2},
where
P
AR
(
𝐳
1
,
𝐚
1
:
t
)
P_{\text{AR}}(\mathbf{z}_{1},\mathbf{a}_{1:t})
denotes the process of autoregressively applying
P
P
to obtain the predicted state
t
t
steps in the future, and the optimization is carried out with CEM
[
8
]
.
II-B
Motivation
This flat planning scheme has three key limitations. First, the goal must be reachable within a fixed horizon
H
H
: longer trajectories require increasing
H
H
, which makes CEM optimization prohibitively expensive. Second, errors compound over many autoregressive steps, causing CEM to diverge for complex trajectories. Third, requiring a concrete goal image
𝐨
g
\mathbf{o}_{g}
upfront is often impractical in real-world tasks. We address all three issues with the hierarchical approach described next.
II-C
Forward-Forward JEPA (FF-JEPA)
Subgoal planner
We introduce a latent planner
G
G
that operates one level above the world model. Every
H
H
steps,
G
G
predicts the next
subgoal
state
𝐳
^
s
​
g
\hat{\mathbf{z}}_{sg}
directly in the encoder’s latent space:
𝐳
^
s
​
g
,
m
+
1
=
G
(
𝐳
s
​
g
,
m
−
W
G
+
1
:
m
)
,
\hat{\mathbf{z}}_{sg,\,m+1}=G\!\left(\mathbf{z}_{sg,\,m-W_{G}+1:m}\right),
where
m
m
indexes subgoals (in our experiments, each separated by
H
H
environment steps), and
W
G
W_{G}
is the planner’s context window. Crucially,
G
G
is
action-free
: it predicts future subgoals purely from latent observations, without requiring a known final goal or an additional layer of CEM search over the full trajectory like in
Zhang et al. [11]
. The world model
P
P
then uses CEM to find the actions that reach each subgoal within
H
H
steps (
section
II-A
, with
𝐳
g
\mathbf{z}_{g}
replaced by
𝐳
^
s
​
g
,
m
+
1
\hat{\mathbf{z}}_{sg,\,m+1}
). This inference scheme is summarized in
fig.
1
.
Training
The latent planner is trained on latent representations of successful demonstrations computed by the frozen, pre-trained world model’s encoder
E
E
. Subgoal states
𝐳
s
​
g
,
m
\mathbf{z}_{sg,m}
are obtained by subsampling each demonstration with stride
H
H
. We experiment with two architectures for
G
G
as illustrated in
fig.
2
.
•
Deterministic planner
G
Det
G_{\text{Det}}
.
This is a transformer with the same architecture as the LeWM’s predictor
P
P
but without action
conditioning. It is trained to minimize the mean-squared error between the predicted and true next subgoal at
H
H
steps in the future, given a sliding context window of size
W
G
W_{G}
past subgoals:
𝐳
^
s
​
g
,
m
+
1
\displaystyle\hat{\mathbf{z}}_{sg,\,m+1}
=
G
Det
(
𝐳
s
​
g
,
m
−
W
G
+
1
:
m
)
,
\displaystyle=G_{\text{Det}}\!\left(\mathbf{z}_{sg,\,m-W_{G}+1:m}\right),
ℒ
Det
\displaystyle\mathcal{L}_{\text{Det}}
=
‖
𝐳
^
s
​
g
,
m
+
1
−
𝐳
s
​
g
,
m
+
1
‖
2
2
.
\displaystyle=\left\|\hat{\mathbf{z}}_{sg,\,m+1}-\mathbf{z}_{sg,\,m+1}\right\|_{2}^{2}.
•
Diffusion planner
G
DM
G_{\text{DM}}
.
This architecture uses a DiT backbone
[
7
]
and is trained with a standard denoising score-matching objective over a predicted horizon of
N
N
future subgoals, similar to
Xie et al. [10]
. Let
k
∈
{
1
,
…
,
K
}
k\in\{1,\dots,K\}
denote the diffusion denoising step. The training objective is:
ℒ
DM
(
ψ
)
=
𝔼
k
,
ϵ
[
‖
ϵ
ψ
(
𝐳
s
​
g
,
m
+
1
:
m
+
N
(
k
)
;
𝐳
s
​
g
,
m
,
k
)
−
ϵ
‖
2
]
,
\mathcal{L}_{\text{DM}}(\psi)=\mathbb{E}_{k,\,\epsilon}\!\left[\left\|\,\epsilon_{\psi}\!\left(\mathbf{z}_{sg,\,m+1:m+N}^{(k)};\;\mathbf{z}_{sg,\,m},\;k\right)-\epsilon\,\right\|^{2}\right],
where
ϵ
\epsilon
denotes the added gaussian noise, and
ϵ
ψ
\epsilon_{\psi}
is the denoising model.
Summary
By introducing a latent planner
G
G
, we obtain a policy that decomposes long trajectories into a sequence of short subproblems that the world model can reliably solve with CEM, without requiring known goal images or a prohibitively large planning horizons.
Fig. 3
:
Example trajectories produced by FF-JEPA (DM).
Dashed red frames indicate subgoals predicted by the latent diffusion planner and decoded for visualization. The first row corresponds to a successful trajectory, while the second row is a failure case where the agent goes out of bounds at t=10 and never recovers.
