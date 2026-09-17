# Planning as Descent: Goal-Conditioned Latent Trajectory Synthesis in Learned Energy Landscapes

paper_id: semanticscholar:ea2b5f2f71e58f8cbea1502f6d95042a4a07ef6e
tier: T2
source_used: html_arxiv
warning: none

## Intro

Learning to act primarily from observation alone remains a central challenge in modern artificial intelligence
(
LeCun, 2022
)
. This challenge is particularly acute in real-world domains such as robotics, where interaction is expensive, unsafe, or impractical, and where available data mostly consist of offline, reward-free trajectories collected under unknown and potentially suboptimal policies. In such settings, agents must infer how to achieve user-specified goals purely from heterogeneous demonstrations, without access to online exploration or reward signals.
We study this problem in the setting of offline goal-conditioned reinforcement learning (GCRL), where the objective is to reach arbitrary target states using only a static dataset of reward-free trajectories. Offline GCRL poses several fundamental difficulties: (i) extracting meaningful structure from unstructured and suboptimal data; (ii) composing disjoint behavioral fragments that may not co-occur within a single trajectory; (iii) propagating sparse goal information over long horizons; and (iv) reasoning about multi-modal futures under stochastic dynamics. Recent benchmarks such as OGBench
(
Park et al., 2024
)
highlight the difficulty of these challenges and show that many existing methods struggle to generalize robustly to
unseen
goals.
A common strategy for addressing offline decision making is to separate modeling and planning. Model-based methods learn forward dynamics and then perform trajectory optimization or model predictive control (MPC) at inference time
(
Zhou et al., 2024
;
Hansen et al., 2023
;
Sobal et al., 2025
)
. While conceptually appealing, this separation often leads to train–test mismatches: powerful optimizers can exploit small inaccuracies in learned dynamics models, producing adversarial or physically implausible trajectories that fail at deployment time
(
Henaff et al., 2019
)
.
An alternative line of work reframes control as trajectory generation, using sequence models such as Decision Transformers
(
Chen et al., 2021
)
, masked trajectory models
(
Wu et al., 2023
;
Janner et al., 2021
;
Carroll et al., 2022
)
, or diffusion-based policies
(
Chi et al., 2023
;
Janner et al., 2022
)
. These models directly model the distribution of trajectories and can synthesize diverse, multimodal behaviors from offline datasets. However, their sampling-based nature often leads to reproducing undesirable behaviors when trained on noisy or suboptimal data, and they lack explicit mechanisms for enforcing long-horizon dynamical feasibility or goal satisfaction. More broadly, these approaches learn
how to generate trajectories
, but do not explicitly learn
how to evaluate or verify them
(
West et al., 2023
)
.
In this work, we propose Planning as Descent (PaD), a framework that rethinks offline goal-conditioned control through the lens of generation by verification. Rather than learning a policy, generator, or explicit planner, PaD learns a goal-conditioned energy landscape over entire future trajectories. This energy assigns low values to trajectories that are dynamically plausible and consistent with a desired goal, and high values to incompatible ones. Planning then arises implicitly as gradient descent in this learned energy landscape, iteratively refining candidate trajectories to minimize their energy.
Crucially, PaD explicitly enforces alignment between training and inference by using the same gradient-based refinement procedure in both phases. The energy landscape is shaped during training around the exact descent dynamics used at test time, ensuring that inference corresponds to optimization behavior the model has been trained to support. The forward pass of the model computes trajectory energies (verification), while the backward pass provides structured descent directions that refine trajectories toward feasible, goal-consistent futures (synthesis). This
training-as-inference
alignment helps mitigate the train–test discrepancies that arise in decoupled modeling and planning pipelines, and contrasts sharply with diffusion and masked trajectory models, which rely on stochastic sampling or reconstruction objectives. In PaD, planning is goal-directed and realized entirely through energy minimization, without autoregressive rollouts, learned noise schedules or value backups.
Moreover, our method is motivated by the most fundamental design bias in deep learning:
depth enables composition
. Just as deep networks compose pixels into objects and words into sentences, PaD allows primitive transitions to compose into subgoals and plans within the depth of a single-learned energy-based model (EBM). Hierarchical planning structure occurs implicitly through gradient-based refinement in the representation space, potentially eliminating the need for explicitly engineered handcrafted abstractions.
We demonstrate that PaD performs robustly across qualitatively different data regimes, including narrow expert demonstrations and broad, highly suboptimal datasets. On challenging tasks from the OGBench single-cube manipulation suite, PaD achieves state-of-the-art performance, exhibits strong robustness to distribution shift, and–perhaps counterintuitively–produces more efficient plans when trained on diverse but highly suboptimal data. These results suggest that learning to verify trajectories, rather than directly generating them, provides a powerful foundation for offline goal-conditioned planning. An implementation of the proposed method will be made publicly available at:
https://github.com/inescopresearch/pad
Contributions.
Our main contributions are:
•
We introduce an energy-based formulation of offline GCRL that unifies trajectory evaluation and synthesis, casting planning as energy minimization in latent trajectory space.
•
We propose a self-supervised training scheme based on hindsight goal relabeling that aligns training-time verification with inference-time planning.
•
We present a gradient-based planning procedure that iteratively refines noisy latent trajectories into coherent, goal-directed plans.
•
We demonstrate state-of-the-art performance on OGBench single-cube tasks and provide an analysis showing that the learned energy landscape supports effective planning and generalization to unseen goals.

## Method

We introduce
Planning as Descent
(PaD), a goal-conditioned latent-space planning framework in which future trajectories are synthesized by descending a learned conditional energy landscape. The core idea is to learn an energy function over latent trajectories that simultaneously serves as a
verifier
of dynamical plausibility and goal satisfaction, and as a
planner
whose gradient field specifies how candidate trajectories should be refined. Given past observations and a desired goal specification, the model assigns low energy to latent futures that are feasible and goal-consistent, while the gradient of this energy provides the descent direction required to synthesize such trajectories.
Figure 1:
Overview of the Planning as Descent (PaD) learning framework.
Given trajectory states and a hindsight-relabeled goal
(
s
g
,
λ
)
(s_{g},\lambda)
, states are independently encoded into latent representations using
f
θ
f_{\theta}
.
Future latents are corrupted to form an initial trajectory
z
future
0
z_{\mathrm{future}}^{0}
, which is iteratively refined by descending the conditional energy
E
θ
E_{\theta}
and projecting updates back onto the encoder-induced manifold through
p
θ
p_{\theta}
.
At each refinement step, a denoising loss compares the intermediate trajectory to the clean future latents while stop-gradient operations prevent (i) mode collapse and (ii) backpropagation through the refinement dynamics.
PaD consists of three main components trained jointly end-to-end: (i) a state encoder
f
θ
f_{\theta}
that maps individual observations into a latent space without modeling temporal structure; (ii) a conditional energy function
E
θ
E_{\theta}
that jointly evaluates and refines entire latent trajectories; and (iii) a projector network
p
θ
p_{\theta}
that ensures refinement remains confined to the encoder-induced manifold. A defining property of PaD is that the same refinement mechanism is used during both training (Algorithm
1
) and inference (Algorithm
2
), allowing the energy landscape to be shaped around the planning dynamics.
4.1
Latent State Representation
Given an observation sequence
(
s
0
,
…
,
s
T
)
(s_{0},\ldots,s_{T})
, each state
s
t
s_{t}
is encoded independently as a latent vector
z
t
=
f
θ
​
(
s
t
)
z_{t}=f_{\theta}(s_{t})
, where the encoder
f
θ
:
𝒮
→
ℝ
d
f_{\theta}:\mathcal{S}\rightarrow\mathbb{R}^{d}
captures only state-wise information and does not impose temporal dependencies. Given past latents
z
past
=
(
z
0
,
…
,
z
k
)
z_{\mathrm{past}}=(z_{0},\ldots,z_{k})
and a goal specification
(
s
g
,
λ
)
(s_{g},\lambda)
, where
λ
∈
[
0
,
1
]
\lambda\in[0,1]
is a normalized time-to-reach variable specifying the desired relative point within the planning horizon at which the goal should be reached, PaD introduces the future latent sequence
z
future
=
(
z
k
+
1
,
…
,
z
k
+
H
)
,
z_{\mathrm{future}}=(z_{k+1},\ldots,z_{k+H}),
which serves as a free optimization variable during planning. Importantly,
λ
\lambda
does not correspond to a physical time or fixed number of environment steps, but instead provides a relative temporal conditioning signal that allows the planner to reason jointly about trajectory feasibility and goal-reaching speed.
4.2
Conditional Energy Model and Gradient-Based Planning
We define a conditional energy
E
θ
​
(
z
future
∣
z
past
,
s
g
,
λ
)
,
E_{\theta}(z_{\mathrm{future}}\mid z_{\mathrm{past}},s_{g},\lambda),
which assigns a scalar value to a candidate future trajectory
z
future
z_{\mathrm{future}}
, conditioned on the latent past
z
past
z_{\mathrm{past}}
, the goal state
s
g
s_{g}
, and the continuous time-to-reach parameter
λ
∈
[
0
,
1
]
\lambda\in[0,1]
. Low energy indicates that the trajectory is dynamically plausible and consistent with reaching the goal within the temporal budget encoded by
λ
\lambda
.
PaD performs planning by iteratively refining the latent trajectory through gradient descent on this energy landscape. Given a current trajectory estimate
z
future
(
t
)
z_{\mathrm{future}}^{(t)}
, the raw refinement step is
z
future
−
raw
(
t
)
=
z
future
(
t
)
−
η
∇
z
future
E
θ
(
z
future
(
t
)
|
z
past
,
s
g
,
λ
)
,
z_{\mathrm{future-raw}}^{(t)}=z_{\mathrm{future}}^{(t)}-\eta\,\nabla_{z_{\mathrm{future}}}E_{\theta}\!\left(z_{\mathrm{future}}^{(t)}\,\middle|\,z_{\mathrm{past}},s_{g},\lambda\right),
where
η
\eta
denotes the refinement step size.
To ensure that refinement remains on or near the encoder-induced manifold, the updated trajectory is passed through a shallow learnable projector
p
θ
p_{\theta}
, producing the final update rule:
z
future
(
t
+
1
)
=
p
θ
(
z
future
(
t
)
−
η
∇
z
future
E
θ
(
z
future
(
t
)
|
z
past
,
s
g
,
λ
)
)
.
z_{\mathrm{future}}^{(t+1)}=p_{\theta}\!\left(z_{\mathrm{future}}^{(t)}-\eta\,\nabla_{z_{\mathrm{future}}}E_{\theta}\!\left(z_{\mathrm{future}}^{(t)}\,\middle|\,z_{\mathrm{past}},s_{g},\lambda\right)\right).
(1)
The projector matters.
We empirically find that the projector
p
θ
p_{\theta}
plays a critical role
in stabilizing refinement. Figure
2
shows that, in the absence of the projector, training becomes unstable,
as energy gradients alone may push latent states toward off-manifold
regions that do not correspond to valid encoded observations, leading
to degenerate latents and degraded planning performance.
Conversely, the projector cannot produce meaningful refinements in the
absence of the structured descent directions supplied by the energy model.
Effective refinement therefore arises from the interaction between the
two components: the energy function provides informed, goal-conditioned
descent directions, while the projector enforces representational validity
by mapping refined trajectories back onto the encoder-induced manifold.
We further ablate the projector component in Section
5.5
.
Figure 2:
Training loss on
single-cube-noisy-v0
when ablating the
manifold projector. The lightweight 130K-parameter projector substantially
stabilizes training and improves convergence with negligible computational
overhead.
4.3
Training as Inference
A defining property of PaD is that the refinement procedure used for planning is identical during training and inference.
This “training-as-inference” principle regularizes the energy landscape such that its gradient flow implements the desired planning behavior.
Hindsight goal relabeling with temporal targets.
Given a trajectory of length
L
L
, we first sample a scalar
r
∈
[
0
,
1
]
r\in[0,1]
from the truncated arccos distribution
p
(
r
)
=
2
π
(
1
−
r
2
)
−
1
/
2
p(r)=\frac{2}{\pi}(1-r^{2})^{-1/2}
and map it linearly to a past-window length
P
past
∈
[
1
,
P
max
]
P_{\mathrm{past}}\in[1,P_{\max}]
, where
P
max
P_{\max}
is the maximum allowed past context.
This biases training toward larger past windows while remaining aligned with inference-time conditions.
To specify the temporal target, we draw
λ
∼
𝒰
⁡
(
0
,
1
)
\lambda\sim\mathcal{U}(0,1)
and map it linearly to a future index
G
∈
[
P
max
,
H
]
G\in[P_{\max},H]
, where
H
H
is the planning horizon.
The state at index
G
G
becomes the goal
s
g
s_{g}
.
Conditioning on
λ
\lambda
is essential: without an explicit temporal target, the model would treat all trajectories that eventually reach the goal as equivalent, making denoising from heavily corrupted latents considerably more difficult.
Latent trajectory corruption.
Let
z
clean
z_{\mathrm{clean}}
denote the clean future latent trajectory
f
θ
​
(
s
future
)
f_{\theta}(s_{\mathrm{future}})
.
To mimic uncertainty over future predictions, we take a corruption scheme inspired by diffusion models
z
future
(
0
)
=
β
​
z
clean
+
1
−
β
​
ϵ
,
ϵ
∼
𝒩
⁡
(
0
,
I
)
z_{\mathrm{future}}^{(0)}=\sqrt{\beta}\,z_{\mathrm{clean}}+\sqrt{1-\beta}\,\epsilon,\qquad\epsilon\sim\mathcal{N}(0,I)
with the corruption level
β
\beta
uniformly sampled as
β
∼
𝒰
⁡
(
0
,
1
)
\beta\sim\mathcal{U}(0,1)
.
Denoising-based training objective.
Starting from this noisy initialization, the model performs
T
T
refinement steps using Equation
1
.
At each refinement step, the intermediate trajectory
z
future
(
t
)
z_{\mathrm{future}}^{(t)}
is compared to the clean target
z
clean
z_{\mathrm{clean}}
using a smooth-
L
1
L_{1}
distance.
The target is treated as a constant, and stop-gradient is applied to
z
future
(
t
)
z_{\mathrm{future}}^{(t)}
between refinement steps to prevent backpropagation through the refinement dynamics themselves, which empirically accelerates training and improves stability.
The overall training loss is therefore
ℒ
=
∑
t
=
1
T
ℓ
⁡
(
z
future
(
t
)
,
z
clean
)
.
\mathcal{L}=\sum_{t=1}^{T}\ell\!\left(z_{\mathrm{future}}^{(t)},\,z_{\mathrm{clean}}\right).
Importantly, this loss is backpropagated through the entire optimization process, which requires second-order derivatives–specifically, gradients of gradients with respect to model parameters arising from the refinement steps. These second-order terms are computed efficiently as Hessian-vector products, which increases training cost about
1.66
×
1.66\times
compared to standard first-order backpropagation in a feed-forward model, assuming a single refinement step and all other factors held constant
(
Gladstone et al., 2025
)
.
We summarize the training procedure of the proposed framework in Algorithm
1
.
Algorithm 1
Training
: Latent Trajectory Denoising with Hindsight Goal Relabeling.
Inputs:
clean trajectory
s
0
:
L
s_{0:L}
, encoder
f
θ
f_{\theta}
, planner
E
θ
E_{\theta}
, projector
p
θ
p_{\theta}
, refinement steps
T
T
1:
(
s
past
,
s
future
,
s
g
,
λ
)
←
(s_{\mathrm{past}},\,s_{\mathrm{future}},s_{g},\lambda)\leftarrow
Hindsight
(
s
0
:
L
)
(s_{0:L})
⊳
\triangleright
Hindsight goal relabeling (See
4.3
)
2:
z
past
←
f
θ
​
(
s
past
)
z_{\mathrm{past}}\leftarrow f_{\theta}(s_{\mathrm{past}})
⊳
\triangleright
Encode past states
3:
z
clean
←
StopGradient
​
(
f
θ
​
(
s
future
)
)
z_{\mathrm{clean}}\leftarrow\textsc{StopGradient}(f_{\theta}(s_{\mathrm{future}}))
⊳
\triangleright
Prevent representation collapse
4:
β
∼
𝒰
⁡
(
0
,
1
)
\beta\sim\mathcal{U}(0,1)
⊳
\triangleright
Sample corruption level
5:
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
⊳
\triangleright
Sample Gaussian noise
6:
z
future
←
β
​
z
clean
+
1
−
β
​
ϵ
z_{\mathrm{future}}\leftarrow\sqrt{\beta}\,z_{\mathrm{clean}}+\sqrt{1-\beta}\,\epsilon
⊳
\triangleright
Corrupt future latents
7:
ℒ
←
0
\mathcal{L}\leftarrow 0
8:
for
t
=
1
t=1
to
T
T
do
9:
E
←
E
θ
​
(
z
future
,
z
past
,
s
g
,
λ
)
E\leftarrow E_{\theta}(z_{\mathrm{future}},z_{\mathrm{past}},s_{g},\lambda)
⊳
\triangleright
Compute energy
10:
z
future
←
z
future
−
η
​
∇
z
future
E
z_{\mathrm{future}}\leftarrow z_{\mathrm{future}}-\eta\nabla_{z_{\mathrm{future}}}E
⊳
\triangleright
Gradient refinement
11:
z
future
←
p
θ
​
(
z
future
)
z_{\mathrm{future}}\leftarrow p_{\theta}(z_{\mathrm{future}})
⊳
\triangleright
Projection
12:
ℒ
←
ℒ
+
smooth-
L
1
​
(
z
future
,
z
clean
)
\mathcal{L}\leftarrow\mathcal{L}+\text{smooth-$L_{1}$}(z_{\mathrm{future}},z_{\mathrm{clean}})
⊳
\triangleright
Accumulate denoising loss
13:
z
future
←
StopGradient
​
(
z
future
)
z_{\mathrm{future}}\leftarrow\textsc{StopGradient}(z_{\mathrm{future}})
⊳
\triangleright
Prevent backprop-through-time
14:
end
for
15:
Update
θ
\theta
using
ℒ
\mathcal{L}
4.4
Inference with Multi-Hypothesis Temporal Targets
At test time, PaD only has access to the sequence of past observations
s
past
=
(
s
0
,
…
,
s
L
)
s_{\mathrm{past}}=(s_{0},\ldots,s_{L})
and a desired goal state
s
g
s_{g}
.
Inference proceeds by sampling and refining multiple candidate future trajectories under different temporal hypotheses as shown in Figure
3
.
At test time, PaD is provided with a sequence of past observations
s
past
=
(
s
0
,
…
,
s
k
)
s_{\mathrm{past}}=(s_{0},\ldots,s_{k})
and a desired goal state
s
g
s_{g}
. Inference proceeds by synthesizing and refining multiple candidate future trajectories under different normalized time-to-reach hypotheses
λ
\lambda
, allowing the model to jointly reason about
how
to reach the goal and
how quickly
it should be reached.
Given the encoded past
z
past
=
f
θ
​
(
s
past
)
z_{\mathrm{past}}=f_{\theta}(s_{\mathrm{past}})
, we initialize a batch of
B
B
candidate future trajectories
z
future
(
0
)
∼
𝒩
​
(
0
,
I
)
B
,
z_{\mathrm{future}}^{(0)}\sim\mathcal{N}(0,I)^{B},
each paired with an independently sampled temporal hypothesis
λ
b
∼
𝒰
⁡
(
0
,
1
)
\lambda_{b}\sim\mathcal{U}(0,1)
. As in training,
λ
\lambda
serves as a relative temporal conditioning signal rather than an absolute number of environment steps, and conditions the planner on the intended fraction of the planning horizon at which the goal should be achieved.
Every candidate is refined for
T
T
steps using the same update rule as in training (Eq.
1
).
After refinement, each trajectory is scored by its final energy
E
b
=
E
θ
(
z
future
(
T
,
b
)
|
z
past
,
s
g
,
λ
b
)
,
b
=
1
,
…
,
B
,
E_{b}=E_{\theta}\!\left(z_{\mathrm{future}}^{(T,b)}\,\middle|\,z_{\mathrm{past}},s_{g},\lambda_{b}\right),\qquad b=1,\ldots,B,
where superscript
(
T
,
b
)
(T,b)
denotes the
b
b
-th candidate after
T
T
refinement steps. Lower energies correspond to trajectories that are both dynamically plausible and consistent with reaching the goal at the rate specified by
λ
b
\lambda_{b}
.
PaD selects a final plan in two stages. First, it identifies the
K
K
candidates with the lowest energies. Second, it samples one trajectory from this top-
K
K
set using a categorical distribution with logits proportional to
−
λ
-\lambda
, introducing a mild bias toward plans that reach the goal sooner whenever doing so remains energetically feasible. This selection mechanism–summarized in Algorithm
2
–balances feasibility and efficiency within the learned energy landscape and mirrors the temporal conditioning used during training.
Algorithm 2
Inference
: Energy Planning with Multiple Time-to-Reach
λ
\lambda
Candidates.
Inputs:
past sequence
s
past
s_{\mathrm{past}}
, goal state
s
g
s_{g}
, encoder
f
θ
f_{\theta}
, planner
E
θ
E_{\theta}
,
projector
p
θ
p_{\theta}
, refinement steps
T
T
,
number of samples
B
B
, top-
k
k
set size
K
K
1:
z
past
←
tile
⁡
(
f
θ
​
(
s
past
)
,
B
)
z_{\mathrm{past}}\leftarrow\mathrm{tile}\left(f_{\theta}\left(s_{\mathrm{past}}\right),B\right)
⊳
\triangleright
Encode past states
2:
z
future
∼
𝒩
​
(
0
,
I
)
B
z_{\mathrm{future}}\sim\mathcal{N}(0,I)^{B}
⊳
\triangleright
Sample
B
B
future latents
3:
λ
∼
𝒰
​
(
0
,
1
)
B
\lambda\sim\mathcal{U}(0,1)^{B}
⊳
\triangleright
Sample
B
B
Steps-to-reach candidates
4:
for
t
=
1
t=1
to
T
T
do
5:
E
←
E
θ
​
(
z
future
,
z
past
,
s
g
,
λ
)
E\leftarrow E_{\theta}(z_{\mathrm{future}},z_{\mathrm{past}},s_{g},\lambda)
⊳
\triangleright
Compute energies
6:
z
future
←
z
future
−
η
​
∇
z
future
E
z_{\mathrm{future}}\leftarrow z_{\mathrm{future}}-\eta\nabla_{z_{\mathrm{future}}}E
⊳
\triangleright
Gradient refinement
7:
z
future
←
p
θ
​
(
z
future
)
z_{\mathrm{future}}\leftarrow p_{\theta}(z_{\mathrm{future}})
⊳
\triangleright
Projection
8:
end
for
9:
E
←
E
θ
​
(
z
future
,
z
past
,
s
g
,
λ
)
E\leftarrow E_{\theta}(z_{\mathrm{future}},z_{\mathrm{past}},s_{g},\lambda)
⊳
\triangleright
Compute final energies
10:
(
E
top
,
ℐ
K
)
←
TopK
​
(
−
E
,
K
)
(E_{\text{top}},\,\mathcal{I}_{K})\leftarrow\textsc{TopK}(-E,K)
⊳
\triangleright
Select
K
K
lowest-energy plans
11:
i
final
∼
Categorical
⁡
(
logits
=
−
λ
ℐ
K
)
i_{\mathrm{final}}\sim\mathrm{Categorical}(\mathrm{logits}=-\lambda_{\mathcal{I}_{K}})
⊳
\triangleright
Sample index biased by lower Steps-to-reach
12:
z
plan
←
z
future
​
[
i
final
]
z_{\mathrm{plan}}\leftarrow z_{\mathrm{future}}[i_{\mathrm{final}}]
13:
return
z
plan
z_{\mathrm{plan}}
More advanced inference schemes (e.g., proposal distributions conditioned on previous refinements) are possible but intentionally omitted to highlight the intrinsic capability of the learned energy landscape. We leave such extensions to future work.
Online Replanning.
At inference, PaD is deployed within an iterative replanning loop in which planning and execution alternate. After synthesizing a latent future trajectory using the refinement procedure described above, only the first
N
N
predicted transitions are decoded into actions via the inverse dynamics model
g
ψ
g_{\psi}
and executed open-loop in the environment. The agent then receives new observations, updates the past window, resamples temporal hypotheses, and invokes the same refinement mechanism to obtain a fresh plan. This plan–execute–replan structure enables PaD to continually correct plan inaccuracies and stochastic transitions while remaining computationally lightweight, as all refinement steps are fully parallelized across hypotheses. In Section
5.4
, we ablate the choice of the replanning interval
N
N
and analyze how it affects task performance, stability, and computational cost.
Figure 3:
Multi-hypothesis planning and temporal target selection in PaD.
Left:
As task execution progresses, PaD naturally selects time-to-reach hypotheses (
λ
\lambda
) that correspond to decreasing distances to the goal, reflecting adaptive planning as the agent approaches completion.
Right:
Example timesteps during the rollout. The top row displays the corresponding environment states at each step. The bottom row shows the distribution of sampled temporal targets and their associated energies, with lower energies indicating more plausible plans for reaching the goal.
4.5
Action Decoding via Inverse Dynamics
Once a latent future trajectory has been synthesized by PaD, it must be translated into executable control inputs. To this end, we employ a separate inverse dynamics model
a
t
=
g
ψ
​
(
z
t
,
z
t
+
1
)
,
a_{t}=g_{\psi}(z_{t},z_{t+1}),
which maps consecutive latent states to the action responsible for the corresponding transition.
The inverse dynamics model
g
ψ
g_{\psi}
is trained independently from the planner using supervised learning on action-labeled transitions drawn from the offline dataset. Importantly, gradients from
g
ψ
g_{\psi}
do
not
propagate to the encoder or energy model; planning is therefore learned entirely from state-only trajectories, and action labels are not required for shaping the energy landscape or the refinement dynamics.
In our experiments, for simplicity,
g
ψ
g_{\psi}
is trained using the full set of available action-labeled transitions. However, this choice is not fundamental to the method. Since inverse dynamics only needs to model local, single-step state transitions rather than long-horizon planning behavior, it can in principle be trained from a substantially smaller labeled subset without affecting the planner itself. We leave a systematic study of the trade-off between action-label availability and execution performance to future work.
Decoupling action decoding from latent planning provides two key advantages. First, it enables PaD to learn goal-conditioned planning entirely from action-free data, which is particularly relevant in settings where only state observations are available. Second, it prevents imperfections in the inverse dynamics model from distorting the learned planning representation, as planning quality is determined solely by the energy-based refinement in latent space.
