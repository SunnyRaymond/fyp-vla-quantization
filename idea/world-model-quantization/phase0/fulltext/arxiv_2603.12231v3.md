# Temporal Straightening for Latent Planning

paper_id: arxiv:2603.12231v3
tier: T3
source_used: html_arxiv
warning: none

## Intro

Latent world models offer a compelling solution for planning due to better efficiency and generalization
(
Nguyen and Widrow, 1989
;
Sutton, 1991
;
Ha and Schmidhuber, 2018
;
Hafner et al., 2020
;
Hafner et al., 2021
;
Hafner et al., 2023
;
Hansen et al., 2022
;
Hansen et al., 2024
)
. They compress high-dimensional observations into compact latent representations, learn predictive dynamics in that latent space, and enable imaginary rollouts for action optimization. Compared to operating directly in pixel or state space, the latent abstraction reduces dimensionality and ignores noise, making dynamics learning more efficient. At test time, planning is typically posed as optimizing an action sequence by rolling the model forward and minimizing a cost function between the goal and the predicted states in the latent space.
Figure 1
:
Latent trajectories encoded by a pretrained visual encoder are usually highly curved, increasing the difficulty of prediction and planning. We learn a representation space where feasible trajectories are straighter to facilitate latent planning.
In practice, however, optimization in the learned latent space remains challenging. The induced planning objective is typically highly non-convex, potentially causing gradient-based optimizers to struggle. As a result, many successful practices
(
Hafner et al., 2019
;
Hansen et al., 2024
;
Zhou et al., 2025
;
Sobal et al., 2025
;
Terver et al., 2025
)
rely on search-based methods such as CEM
(
Rubinstein, 1997
)
or MPPI
(
Williams et al., 2015
)
, which achieve competitive performance but introduce a substantial compute burden and latency. Moreover, commonly used goal cost metrics based on Euclidean distance can be misleading if the embedding space is not properly regularized. In particular, when latent trajectories are highly curved, straight-line distances in embedding space misrepresent the geodesic distance along feasible transitions. These challenges call for better representations that facilitate latent planning.
Figure 2
:
Latent trajectories before vs. after straightening. The upper PushT example is a rotation and the bottom UMaze example shows the agent traveling from the top-left to the top-right, with the star denoting the target. Straightening yields less curved and smoother trajectories, and makes Euclidean distance a more faithful proxy for geodesic progress towards the goal. More examples are in
Section
E.2
.
What is a “good” representation for latent planning? Although general-purpose visual pretraining provides powerful semantic-aware features, it is not tailored to the dynamics of the environment and often retains plenty of planning-irrelevant low-level details. We argue that planning could benefit from representations that are (i) sufficient for predicting dynamics but without task-irrelevant information and (ii) properly regularized so that embedding distances reflect the geodesic distance and gradient-based optimization is reliable. With such representations, we can exploit the differentiability of latent world models and enable efficient gradient-based planning, bypassing the need for computationally expensive search-based methods.
Inspired by the perceptual straightening hypothesis in human vision
(
Hénaff et al., 2019
)
, which posits that visual systems transform complex natural videos into straighter internal representations, we introduce a simple approach to straighten latent trajectories for planning. Concretely, we jointly learn an encoder and a predictor of a Joint-Embedding Predictive Architecture (JEPA) world model, while imposing regularization on the curvature of latent trajectories during training. We
find that the JEPA prediction objective alone induces
implicit straightening
to some extent, and introducing an
explicit
curvature regularizer further strengthens and
stabilizes this effect. The resulting encoded trajectories are significantly straighter, with Euclidean distances better aligned with geodesic distances (
Figure
2
). We prove that reducing curvature improves convergence of gradient-based planners, and observe superior empirical gains across a suite of goal-reaching tasks: open-loop planning success improves by 20–60% and MPC by 20–30% with a simple gradient-based planner.
Figure 3
:
During training, we minimize the prediction loss between the predicted embedding
z
^
t
\hat{z}_{t}
and the target embedding
z
t
{z}_{t}
with stop-grad in the target branch, and minimize the local curvature of embeddings. During planning, we roll out for the horizon T using the trained predictor and select optimal actions that minimize the cost between the predicted terminal state
z
^
T
\hat{z}_{T}
and the goal embedding
z
g
z_{g}
.

## Method

We consider control tasks with high-dimensional observations
o
t
∈
ℝ
n
o
o_{t}\in\mathbb{R}^{n_{o}}
of an agent interacting with its environment using actions
a
t
∈
ℝ
n
a
a_{t}\in\mathbb{R}^{n_{a}}
. Our goal is to learn a world model that maps observations to a latent space and models the dynamics in this space, which we use for latent planning.
In this section, we first outline the architecture of our world model, then define the training objectives with a novel geometric regularization that straightens latent trajectories.
3.1
World Model
Our world model predicts future states in a learned latent space and consists of three components: a sensory encoder, an action encoder, and a predictor.
Sensory encoder.
The sensory encoder
ℰ
ϕ
s
\mathcal{E}^{s}_{\phi}
maps raw observations
o
t
o_{t}
into latent representations
z
t
∈
ℝ
d
=
ℰ
ϕ
s
​
(
o
t
)
.
\displaystyle z_{t}\in\mathbb{R}^{d}=\mathcal{E}^{s}_{\phi}(o_{t}).
(1)
The sensory encoder can be any function that maps observations to latent representations. For visual observations, the encoder may preserve spatial structure or collapse it into a global vector representation.
Action encoder.
Each action
a
t
∈
ℝ
n
a
a_{t}\in\mathbb{R}^{n_{a}}
is mapped to a latent action embedding via
ℰ
ψ
a
:
ℝ
n
a
→
ℝ
d
a
.
\mathcal{E}^{a}_{\psi}:\mathbb{R}^{n_{a}}\to\mathbb{R}^{d_{a}}.
Predictor.
The predictor
f
θ
:
ℝ
K
×
d
×
ℝ
K
×
d
a
→
ℝ
d
f_{\theta}:\mathbb{R}^{K\times d}\times\mathbb{R}^{K\times d_{a}}\to\mathbb{R}^{d}
models transitions in the latent space. Given a history of
K
K
past latent states and actions, it predicts the next latent state
z
^
t
=
f
θ
​
(
{
z
i
}
i
=
t
−
K
t
−
1
,
{
ℰ
ψ
a
​
(
a
i
)
}
i
=
t
−
K
t
−
1
)
.
\hat{z}_{t}=f_{\theta}\left(\{z_{i}\}_{i=t-K}^{t-1},\{\mathcal{E}^{a}_{\psi}(a_{i})\}_{i=t-K}^{t-1}\right).
(2)
3.2
Straightening Latent Trajectories
We seek to straighten the latent space induced by the sensory encoder
ℰ
ϕ
s
\mathcal{E}^{s}_{\phi}
by penalizing the curvature along trajectories. Let
z
t
,
z
t
+
1
z_{t},z_{t+1}
, and
z
t
+
2
z_{t+2}
be three consecutive latent representations obtained by encoding observations
o
t
,
o
t
+
1
o_{t},o_{t+1}
, and
o
t
+
2
o_{t+2}
using
ℰ
ϕ
s
\mathcal{E}^{s}_{\phi}
. We define approximate latent velocity vectors
v
t
=
z
t
+
1
−
z
t
,
v
t
+
1
=
z
t
+
2
−
z
t
+
1
,
v_{t}=z_{t+1}-z_{t},\quad v_{t+1}=z_{t+2}-z_{t+1},
(3)
and seek to minimize the angle between them, or equivalently maximize their cosine similarity
𝒞
=
v
t
⋅
v
t
+
1
‖
v
t
‖
2
⋅
‖
v
t
+
1
‖
2
.
\mathcal{C}=\frac{v_{t}\cdot v_{t+1}}{||v_{t}||_{2}\cdot||v_{t+1}||_{2}}.
(4)
3.3
Training Objective.
The parameters
ϕ
,
ψ
\phi,\psi
and
θ
\theta
of the world model components
ℰ
ϕ
s
\mathcal{E}^{s}_{\phi}
,
ℰ
ψ
a
\mathcal{E}^{a}_{\psi}
and
f
θ
f_{\theta}
are trained jointly to minimize prediction error and enforce straightened trajectories.
Prediction objective.
We minimize the MSE between the predicted and target latent states
z
^
t
+
1
\hat{z}_{t+1}
and
z
t
+
1
z_{t+1}
:
ℒ
p
​
r
​
e
​
d
=
|
|
z
^
t
+
1
−
sg
(
z
t
+
1
)
|
|
2
2
,
\mathcal{L}_{pred}=||\hat{z}_{t+1}-\operatorname{sg}(z_{t+1})||_{2}^{2}\\
\text{ },
(5)
where
sg
\mathrm{sg}
denotes the stop-gradient operation to prevent collapse of the latent space.
Straightening objective.
We minimize trajectory curvatures by minimizing the negative cosine similarity
ℒ
c
​
u
​
r
​
v
=
1
−
𝒞
.
\mathcal{L}_{curv}=1-\mathcal{C}.
(6)
This straightening loss can be applied to any differentiable sensory encoder, either in isolation or jointly with the prediction objective.
Overall objective.
The total training objective combines prediction and straightening as
ℒ
t
​
o
​
t
​
a
​
l
=
ℒ
p
​
r
​
e
​
d
+
λ
​
ℒ
c
​
u
​
r
​
v
,
\mathcal{L}_{total}=\mathcal{L}_{pred}+\lambda\mathcal{L}_{curv},
(7)
where
λ
≥
0
\lambda\geq 0
controls the strength of the straightening.
Collapse prevention.
Since our encoder is trainable, the model is likely to produce degenerate solutions in which all latent representations collapse to a constant. Common anti-collapse strategies can be regularization-based
(
Bardes et al., 2022
;
Zhu et al., 2024
;
Balestriero and LeCun, 2025
;
Kuang et al., 2026
)
, contrastive-based
(
Chen et al., 2020
;
He et al., 2020
)
, and stop-gradient-based
(
Chen and He, 2021
;
Grill et al., 2020
)
. Our curvature regularizer is
orthogonal to
these anti-collapse methods
and can be combined with any of them. We use stop-grad for major experiments due to its simplicity and efficiency, as it does not require negative samples or introduce new hyperparameters. We apply stop-gradient to the target latent in the prediction loss (
5
) to prevent the gradients from
ℒ
p
​
r
​
e
​
d
\mathcal{L}_{pred}
from being backpropagated through the target branch.
Although a collapsing solution is still possible in theory, stop-grad has been shown to be effective in self-supervised vision learning
(
Chen and He, 2021
)
, and also effective in our experiments.
(a)
DINOv2
(b)
Straightened
Figure 4
:
Action-Space Loss Landscape. We pick one test sample from PushT with a planning horizon of 25 steps. For each coordinate
(
a
x
,
a
y
)
(a_{x},a_{y})
in the grid, we fix the first action and optimize the remaining actions in the planning horizon to minimize the terminal goal cost. The heatmap represents the minimum attainable loss for each initial action choice, with darker colors indicating lower loss. The loss landscape is closer to being convex after straightening.
