# RAY-TOLD: Ray-Based Latent Dynamics for Dense Dynamic Obstacle Avoidance with TDMPC

paper_id: semanticscholar:8a89c23591be3467eceda0b786ffb90998258136
tier: T2
source_used: html_arxiv
warning: none

## Intro

Navigating through dense crowds of dynamic obstacles (Fig.
1
) remains a formidable challenge for autonomous mobile robots. Real-world deployment in human-centric environments requires navigation systems capable of rapidly anticipating environmental changes, avoiding unpredictable dynamic agents, and strictly adhering to kinematic constraints
[
1
]
. While classical optimization-based methods provide reliable collision avoidance, the exponential growth of complexity in highly dynamic environments often necessitates a trade-off between computational efficiency and long-term foresight.
Model Predictive Path Integral (MPPI) control has emerged as a powerful sampling-based method for handling non-linear vehicle dynamics in uncertain environments
[
2
]
. By leveraging parallel trajectory rollouts, MPPI achieves robust short-term collision avoidance and kinematic feasibility. However, purely reactive planners like standard MPPI suffer from a critical limitation: the computational cost of simulating forward dynamics curtails the feasible prediction horizon
[
3
]
. Consequently, these methods lack long-term spatial awareness, frequently causing the robot to become trapped in local minima such as U-shaped obstacles
[
4
]
or dense crowd formations, where short-term safe paths fail to align with the global navigation goal.
Reinforcement learning (RL), on the other hand, excels in consolidating long-horizon reasoning directly from environmental interactions. By learning a value function that approximates the infinite-horizon cost-to-go, RL provides agents with an intuitive foresight that helps avoid myopic decisions and local minima
[
5
]
. Furthermore, recent advancements in world models and latent dynamics have enabled agents to acquire structured and predictive representations of the environment, facilitating planning through learned internal models rather than relying solely on high-dimensional raw observations
[
6
,
7
]
.
Fig. 1
:
(a) An environment with dense dynamic obstacles and (b) the corresponding bird’s-eye view.
Despite these advances in model-based and representation-driven RL, many practical robotic systems still rely on end-to-end model-free RL policies due to their simplicity and scalability. However, such approaches often struggle to enforce hard safety constraints during learning and tend to suffer from inefficient or unsafe exploration, particularly in environments with complex or dynamic obstacles. These limitations pose significant challenges for the reliable deployment of RL policies in safety-critical physical robotic systems
[
8
]
.
To bridge the gap between the constraint-satisfying robustness of physics-based MPC and the long-horizon foresight of learning-based models, this paper proposes
Ray-based Task-Oriented Latent Dynamics (RAY-TOLD)
. Built upon the principles of temporal-difference learning for model predictive control
[
9
,
10
]
, RAY-TOLD introduces a hybrid control architecture tailored for LiDAR-equipped autonomous robots navigating in dynamic crowds. Our approach encodes high-dimensional obstacle information from LiDAR scans into a low-dimensional latent representation. Within this latent space, a transition dynamics model, a terminal value function, and an action policy are jointly trained.
During online execution, RAY-TOLD resolves the local minima problem inherent to MPPI by introducing a policy mixture sampling strategy. Instead of relying entirely on unbiased Gaussian noise for trajectory generation, RAY-TOLD augments the MPPI candidate population with trajectories proposed by the learned policy prior. The sampled trajectories are then evaluated using the learned terminal value function, which encapsulates the long-term intent of the navigation task. This allows the MPPI planner to effectively look beyond its limited short-horizon rollout, securely guiding the robot around complex dynamic obstacles while guaranteeing kinematic feasibility.
The main contributions of this work are summarized as follows:
•
We present RAY-TOLD, a hybrid architecture that seamlessly integrates LiDAR-centric latent dynamics modeling with sampling-based MPPI to tackle navigation in dense dynamic environments.
•
We introduce a policy mixture sampling technique coupled with a learned terminal value function, effectively eliminating the vulnerability of MPPI to local minima without requiring explicitly handcrafted heuristics.
•
We conduct extensive tests in highly stochastic environments, demonstrating that blending short-horizon physics-based rollouts with long-horizon learned intent significantly reduces collision rates and outperforms standard MPPI baselines in safety and reliability.
The remainder of the paper is organized as follows. Section II reviews the related literature on MPC and RL-based collision avoidance. Section III presents the detailed methodology of RAL-TOLD, including the mathematical formulations. Section IV provides test validation of RAL-TOLD in environments with dense dynamic obstacles. Finally, Section V concludes the paper.

## Method

We present RAY-TOLD, a hybrid control framework that couples the robustness of physics-based model predictive control (MPC) with the long-horizon foresight of model-based reinforcement learning (MBRL). Our approach builds upon the conceptual foundation of task-oriented latent dynamics (TOLD)
[
10
]
, developing a LiDAR-centric latent model that incorporates physics-based rollout for short-horizon interactions while leveraging a learned value function and policy prior to guide long-term decision-making.
III-A
Problem Formulation
We consider a discrete-time formulation of the optimal control problem. Let
s
t
∈
𝒮
s_{t}\in\mathcal{S}
denote the state and
a
t
∈
𝒜
a_{t}\in\mathcal{A}
the action at time
t
t
, evolving according to the system dynamics
s
t
+
1
=
f
⁡
(
s
t
,
a
t
)
s_{t+1}=f(s_{t},a_{t})
. The objective is to maximize the expected cumulative reward:
J
⁡
(
π
)
=
𝔼
π
​
[
∑
t
=
0
∞
γ
t
​
r
​
(
s
t
,
a
t
)
]
,
J(\pi)=\mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty}\gamma^{t}r(s_{t},a_{t})\right],
(1)
where
r
⁡
(
s
t
,
a
t
)
r(s_{t},a_{t})
is the reward function and
γ
∈
[
0
,
1
)
\gamma\in[0,1)
is the discount factor. The
r
⁡
(
s
t
,
a
t
)
r(s_{t},a_{t})
is designed to encourage efficient navigation while ensuring safety. It is composed of six terms:
r
=
r
d
​
i
​
s
​
t
+
r
c
​
o
​
l
​
l
+
r
s
​
i
​
d
​
e
+
r
v
​
e
​
l
+
r
p
​
r
​
o
​
g
+
r
g
​
o
​
a
​
l
r=r_{dist}+r_{coll}+r_{side}+r_{vel}+r_{prog}+r_{goal}
(2)
•
r
d
​
i
​
s
​
t
=
−
‖
p
t
−
g
‖
2
r_{dist}=-\|\mathrm{p_{t}}-\mathrm{g}\|_{2}
: Negative Euclidean distance to the goal
g
\mathrm{g}
.
•
r
c
​
o
​
l
​
l
=
−
120
⋅
𝕀
(
d
o
​
b
​
s
<
0.1
)
r_{coll}=-120\cdot\mathbb{I}(d_{obs}<0.1)
: Heavy penalty for collision, where
d
o
​
b
​
s
d_{obs}
is the distance to the nearest obstacle surface.
•
r
s
​
i
​
d
​
e
=
−
15
⋅
exp
(
−
4
⋅
d
o
​
b
​
s
)
r_{side}=-15\cdot\exp(-4\cdot d_{obs})
: Soft clearance penalty to encourage keeping a safe distance from obstacles.
•
r
v
​
e
​
l
r_{vel}
: Velocity incentive (
0.5
​
v
0.5v
) when far from the goal, switching to a braking penalty, i.e.,
−
v
-v
when within
2.0
2.0
m.
•
r
p
​
r
​
o
​
g
=
5
⋅
(
𝐡
t
⋅
𝐝
g
​
o
​
a
​
l
)
⋅
v
r_{prog}=5\cdot(\mathbf{h}_{t}\cdot\mathbf{d}_{goal})\cdot v
: Progress reward based on alignment between heading
𝐡
t
\mathbf{h}_{t}
and goal direction
𝐝
g
​
o
​
a
​
l
\mathbf{d}_{goal}
.
•
r
g
​
o
​
a
​
l
=
300
⋅
𝕀
⁡
(
‖
p
t
−
g
‖
2
<
0.7
)
r_{goal}=300\cdot\mathbb{I}(\|p_{t}-g\|_{2}<0.7)
: Sparse terminal reward for reaching the goal region.
Notation:
p
t
∈
ℝ
2
\mathrm{p_{t}}\in\mathbb{R}^{2}
denotes the ego position,
g
∈
ℝ
2
\mathrm{g}\in\mathbb{R}^{2}
the goal position, and
𝕀
⁡
(
⋅
)
\mathbb{I}(\cdot)
the indicator function.
We define the goal-direction unit vector
𝐝
g
​
o
​
a
​
l
=
(
𝐠
−
𝐩
𝐭
)
​
‖
𝐠
−
𝐩
𝐭
‖
2
−
1
\mathbf{d}_{goal}=(\mathbf{g}-\mathbf{p_{t}})\|\mathbf{g}-\mathbf{p_{t}}\|_{2}^{-1}
and the heading unit vector
𝐡
t
=
[
cos
⁡
θ
t
,
sin
⁡
θ
t
]
⊤
\mathbf{h}_{t}=[\cos\theta_{t},\ \sin\theta_{t}]^{\top}
.
d
o
​
b
​
s
d_{obs}
is computed as the minimum LiDAR ray intersection distance to obstacle boundaries (i.e., truncated at the first hit), consistent with the occlusion-aware sensing model in Sec.
IV
.
Our proposed architecture integrates two complementary prediction modalities, which address the primary limitations of the standard MPPI:
1.
Policy-Guided Rollouts (
t
∼
t
+
H
−
1
t\sim t{+}H{-}1
):
Unlike standard MPPI which relies solely on random noise perturbations, we perform
H
H
-step rollouts using a differentiable kinematic bicycle model where a fraction of the sampled actions are guided by a learned latent policy
π
θ
\pi_{\theta}
. This mixture sampling explicitly biases the search space toward goal-directed, kinematically feasible trajectories while evaluating short-horizon collision outcomes.
2.
Latent Terminal Value Estimation (
≥
t
+
H
\geq t{+}H
):
Standard MPPI typically ignores the long-term consequences beyond the planning horizon
H
H
. Instead, we augment the finite-horizon objective with a learned terminal value. We encode the final state of the rollout into a compact latent representation
z
t
+
H
=
h
θ
​
(
𝐱
t
+
H
)
z_{t+H}=h_{\theta}(\mathbf{x}_{t+H})
and use the learned value function
Q
θ
Q_{\theta}
to approximate the expected return beyond the rollout horizon, effectively preventing the planner from being trapped in local minima.
Algorithm 1
Planning algorithms using RAY-TOLD model
1:
Input:
Augmented state
𝐱
t
\mathbf{x}_{t}
, Policy prior
π
ξ
\pi_{\xi}
, Value function
C
ω
C_{\omega}
, Encoder
h
θ
h_{\theta}
, Latent Dynamics
d
ϕ
d_{\phi}
2:
Hyperparameters:
Horizon
H
H
, Samples
K
K
, Mixture ratio
α
\alpha
, Temp
λ
\lambda
, Iterations
M
M
3:
Initialize mean action sequence
μ
0
:
H
−
1
←
shift
(
μ
t
−
1
)
\mu_{0:H-1}\leftarrow\text{shift}(\mu_{t-1})
4:
u
π
0
:
H
−
1
←
u^{\pi}_{0:H-1}\leftarrow
Rollout
π
ξ
\pi_{\xi}
in latent space using
d
ϕ
d_{\phi}
from
h
θ
​
(
𝐱
t
)
h_{\theta}(\mathbf{x}_{t})
5:
for
m
=
1
m=1
to
M
M
do
6:
Sample
ϵ
k
,
τ
∼
𝒩
⁡
(
0
,
Σ
)
\epsilon_{k,\tau}\sim\mathcal{N}(0,\Sigma)
for
k
=
1
​
…
​
K
,
τ
=
0
​
…
​
H
−
1
k=1\dots K,\tau=0\dots H-1
7:
for
k
=
1
k=1
to
K
K
do
8:
if
k
≤
α
​
K
k\leq\alpha K
then
9:
a
k
,
0
:
H
−
1
←
clip
(
u
0
:
H
−
1
π
+
ϵ
k
,
0
:
H
−
1
)
a_{k,0:H-1}\leftarrow\text{clip}(u^{\pi}_{0:H-1}+\epsilon_{k,0:H-1})
10:
else
11:
a
k
,
0
:
H
−
1
←
clip
(
μ
0
:
H
−
1
+
ϵ
k
,
0
:
H
−
1
)
a_{k,0:H-1}\leftarrow\text{clip}(\mu_{0:H-1}+\epsilon_{k,0:H-1})
12:
end
if
13:
s
k
,
0
←
s
t
,
R
k
←
0
s_{k,0}\leftarrow s_{t},R_{k}\leftarrow 0
14:
for
τ
=
0
\tau=0
to
H
−
1
H-1
do
15:
s
k
,
τ
+
1
←
f
⁡
(
s
k
,
τ
,
a
k
,
τ
)
s_{k,\tau+1}\leftarrow f(s_{k,\tau},a_{k,\tau})
16:
R
k
←
R
k
+
γ
τ
​
r
​
(
s
k
,
τ
,
a
k
,
τ
)
R_{k}\leftarrow R_{k}+\gamma^{\tau}r(s_{k,\tau},a_{k,\tau})
17:
end
for
18:
z
k
,
H
←
h
θ
​
(
𝐱
k
,
H
)
z_{k,H}\leftarrow h_{\theta}(\mathbf{x}_{k,H})
19:
a
term
←
π
ξ
​
(
z
k
,
H
)
a_{\text{term}}\leftarrow\pi_{\xi}(z_{k,H})
20:
Φ
term
←
C
ω
​
(
z
k
,
H
,
a
term
)
\Phi_{\text{term}}\leftarrow C_{\omega}(z_{k,H},a_{\text{term}})
21:
R
k
←
R
k
+
γ
H
​
Φ
term
R_{k}\leftarrow R_{k}+\gamma^{H}\Phi_{\text{term}}
22:
end
for
23:
Ω
←
∑
k
=
1
K
exp
⁡
(
1
λ
​
R
k
)
,
w
k
←
1
Ω
​
exp
⁡
(
1
λ
​
R
k
)
\Omega\leftarrow\sum_{k=1}^{K}\exp(\frac{1}{\lambda}R_{k}),w_{k}\leftarrow\frac{1}{\Omega}\exp(\frac{1}{\lambda}R_{k})
24:
μ
τ
←
∑
k
=
1
K
w
k
​
a
k
,
τ
\mu_{\tau}\leftarrow\sum_{k=1}^{K}w_{k}a_{k,\tau}
for
τ
=
0
​
…
​
H
−
1
\tau=0\dots H-1
25:
end
for
26:
Return:
mean action
μ
0
\mu_{0}
III-B
RAY-TOLD Model
Parallel to the physics-based planner, we train a RAY-TOLD model to learn a compact latent representation
z
t
z_{t}
and associated value dynamics, where the overall schematic is shown in Fig.
2
. We define an augmented state vector
𝐱
t
=
[
s
t
,
s
t
g
​
o
​
a
​
l
,
s
t
r
​
a
​
y
]
\mathbf{x}_{t}=[s_{t},s^{goal}_{t},s^{ray}_{t}]
, where
s
t
s_{t}
is the vehicle kinematics,
s
t
g
​
o
​
a
​
l
s^{goal}_{t}
is the relative goal position, and
s
t
r
​
a
​
y
s^{ray}_{t}
comprises the
N
r
​
a
​
y
​
s
N_{rays}
range measurements and obstacle velocities.
The encoder
h
θ
​
(
𝐱
t
)
h_{\theta}(\mathbf{x}_{t})
compresses this high-dimensional input into a low-dimensional latent state
z
t
z_{t}
. The individual components of the RAY-TOLD model are parameterized by separate neural networks:
z
t
\displaystyle z_{t}
=
h
θ
​
(
𝐱
t
)
\displaystyle=h_{\theta}(\mathbf{x}_{t})
(Encoder)
(3)
z
t
+
1
\displaystyle z_{t+1}
=
d
ϕ
​
(
z
t
,
a
t
)
\displaystyle=d_{\phi}(z_{t},a_{t})
(Latent Dynamics)
(4)
r
^
t
\displaystyle\hat{r}_{t}
=
R
ψ
​
(
z
t
,
a
t
)
\displaystyle=R_{\psi}(z_{t},a_{t})
(Reward Predictor)
(5)
Q
⁡
(
z
t
,
a
t
)
\displaystyle Q(z_{t},a_{t})
=
C
ω
​
(
z
t
,
a
t
)
\displaystyle=C_{\omega}(z_{t},a_{t})
(Value Function)
(6)
a
^
t
\displaystyle\hat{a}_{t}
∼
π
ξ
​
(
z
t
)
\displaystyle\sim\pi_{\xi}(z_{t})
(Policy Prior)
(7)
Here, the reward predictor
R
ψ
R_{\psi}
and the value function
C
ω
C_{\omega}
serve complementary but distinct roles. The reward predictor
R
ψ
​
(
z
t
,
a
t
)
≈
r
⁡
(
s
t
,
a
t
)
R_{\psi}(z_{t},a_{t})\approx r(s_{t},a_{t})
is a neural network that estimates the
immediate, single-step
reward directly from the latent state and action, without requiring access to the full physics engine. This is essential during training, where multi-step imagined rollouts are performed entirely within the learned latent space using the latent dynamics model
d
ϕ
d_{\phi}
: since these rollouts do not pass through the true environment, the ground-truth reward function
r
⁡
(
s
t
,
a
t
)
r(s_{t},a_{t})
is unavailable, and
R
ψ
R_{\psi}
serves as its differentiable surrogate. The reward predictor thus enables end-to-end gradient-based optimization of the latent representation by providing the reward signal needed to train the value function and policy prior through temporal difference (TD) learning.
In contrast, the value function
C
ω
​
(
z
t
,
a
t
)
≈
Q
π
​
(
s
t
,
a
t
)
C_{\omega}(z_{t},a_{t})\approx Q^{\pi}(s_{t},a_{t})
serves as the critic, approximating the
infinite-horizon discounted return
from a given latent state-action pair. During evaluation,
C
ω
C_{\omega}
is used exclusively at the terminal state of each MPPI rollout (i.e., at time
t
+
H
t{+}H
) to estimate the expected future return beyond the finite planning horizon. This terminal value estimation is the key mechanism by which RAY-TOLD overcomes the myopic limitation of standard MPPI.
III-C
Model Training
The RAY-TOLD model is trained end-to-end using trajectories collected in the environment. Following the TD-MPC2 framework
[
10
]
, the training objective minimizes a combined loss function:
ℒ
=
λ
1
​
ℒ
r
​
e
​
w
​
a
​
r
​
d
+
λ
2
​
ℒ
v
​
a
​
l
​
u
​
e
+
λ
3
​
ℒ
p
​
o
​
l
​
i
​
c
​
y
+
λ
4
​
ℒ
l
​
a
​
t
​
e
​
n
​
t
\mathcal{L}=\lambda_{1}\mathcal{L}_{reward}+\lambda_{2}\mathcal{L}_{value}+\lambda_{3}\mathcal{L}_{policy}+\lambda_{4}\mathcal{L}_{latent}
(8)
where
ℒ
r
​
e
​
w
​
a
​
r
​
d
\mathcal{L}_{reward}
trains
R
ψ
R_{\psi}
to predict ground-truth single-step rewards,
ℒ
v
​
a
​
l
​
u
​
e
\mathcal{L}_{value}
trains
C
ω
C_{\omega}
via Temporal Difference (TD) learning, and
ℒ
p
​
o
​
l
​
i
​
c
​
y
\mathcal{L}_{policy}
optimizes the actor network
π
ξ
\pi_{\xi}
to maximize the Q-value. Finally,
ℒ
l
​
a
​
t
​
e
​
n
​
t
\mathcal{L}_{latent}
enforces consistency between the state transitions predicted by the Latent Dynamics
d
ϕ
d_{\phi}
and the actual encoded future states, ensuring the latent space accurately captures the underlying transition physics.
Model Predictive Path Integral (MPPI) control is deployed for trajectory optimization. Standard MPPI draws action perturbations around the previous solution, which yields a unimodal proposal distribution. In dense, multimodal interaction scenarios, this often under-explores alternative homotopy classes and leads to local-minima.
Thus, we introduce a policy mixture sampling strategy, originally introduced in TD-MPC2
[
10
]
. The MPPI population is augmented by seeding a fraction
α
\alpha
of the samples with trajectories derived from the learned policy
π
ξ
\pi_{\xi}
. Specifically, let
u
π
t
:
t
+
H
−
1
u^{\pi}_{t:t+H-1}
be the action sequence generated by recursively unrolling the policy
π
ξ
\pi_{\xi}
using the Latent Dynamics network
d
ϕ
d_{\phi}
. This allows generating multi-step future actions in the compact latent space without repeatedly invoking the expensive physics engine. The sampling distribution for the
k
k
-th candidate trajectory is given by:
a
t
(
k
)
∼
{
𝒩
⁡
(
a
t
π
,
Σ
)
if
​
k
<
α
​
N
𝒩
⁡
(
a
t
M
​
P
​
P
​
I
,
Σ
)
otherwise
a^{(k)}_{t}\sim\begin{cases}\mathcal{N}(a^{\pi}_{t},\Sigma)&\text{if }k<\alpha N\\
\mathcal{N}(a^{MPPI}_{t},\Sigma)&\text{otherwise}\end{cases}
(9)
TABLE I
:
Hyperparameters for RAY-TOLD Training and Evaluation
Parameter
Value
Horizon (
H
H
)
30
Evaluation Steps
300
MPPI samples (
K
K
)
256
MPPI optimization iterations (
M
M
)
3
Softmax temperature for MPPI weighting (
λ
\lambda
)
1.0
Discount factor (
γ
\gamma
)
0.99
Lidar Rays (
N
rays
N_{\text{rays}}
)
60
Obstacle Radius
0.4 m
Map Size
20
​
m
×
10
​
m
20\;\mathrm{m}\times 10\;\mathrm{m}
Batch Size
256
Learning Rate
1
×
10
−
4
1\times 10^{-4}
Replay Buffer Size
50,000
This mechanism injects global knowledge from the learned policy into the local optimization process of MPPI, effectively guiding the planner towards long-term objectives while retaining the reactivity of the physics-based model. The corresponding objective function is formulated by
J
(
a
t
:
t
+
H
−
1
)
=
𝔼
[
∑
k
=
0
H
−
1
γ
k
r
(
s
t
+
k
,
a
t
+
k
)
+
γ
H
C
ω
(
z
t
+
H
,
π
ξ
(
z
t
+
H
)
)
]
,
J(a_{t:t+H-1})=\mathbb{E}\!\left[\sum_{k=0}^{H-1}\gamma^{k}r(s_{t+k},a_{t+k})+\gamma^{H}C_{\omega}\!\left(z_{t+H},\,\pi_{\xi}(z_{t+H})\right)\right],
(10)
where
z
t
+
H
=
h
θ
​
(
𝐱
t
+
H
)
z_{t+H}=h_{\theta}(\mathbf{x}_{t+H})
is the latent representation of the terminal state, and
C
ω
C_{\omega}
evaluates its expected return following the policy prior
π
ξ
\pi_{\xi}
.
TABLE II
:
Performance comparison. The proposed method achieves an improvement in success rate and reduction in collision rate compared to the MPPI baseline, demonstrating the benefit of RAY-TOLD model in complex scenarios.
Method
Success Rate
Improv.
Collision Rate
Improv.
Safety Margin
Improv.
MPPI (Baseline)
89.0%
-
11.0%
-
0.19m
-
RAY-TOLD (
α
=
0
\alpha{=}0
)
89.0%
0.0%
11.0%
0.0%
0.17m
10.53%
RAY-TOLD (
α
=
0.1
\alpha{=}0.1
)
94.0%
5.62%
6.0%
45.45%
0.18m
5.26%
RAY-TOLD (
α
=
0.2
\alpha{=}0.2
)
91.0%
2.25%
9.0%
18.18%
0.18m
5.26%
(a)
(b)
(c)
(d)
Fig. 3
:
Four representative results from 100 test scenarios comparing trajectories in challenging dynamic environments. (a) Baseline failure, where MPPI fails to predict converging obstacles. (b) RAY-TOLD (
α
=
0.2
\alpha{=}0.2
), where stronger policy guidance yields more efficient paths. (c) RAY-TOLD (
α
=
0.1
,
0.2
\alpha{=}0.1,\;0.2
), which successfully anticipates obstacle reflections. (d) Only RAY-TOLD (
α
=
0.1
\alpha{=}0.1
) avoids collision in this scenario.
