# HALO: Hybrid Auto-encoded Locomotion with Learned Latent Dynamics, Poincaré Maps, and Regions of Attraction

paper_id: semanticscholar:e306a1a1fcd7dc90d442bc6b84c093c03ed1199c
tier: T2
source_used: html_arxiv
warning: none

## Intro

Reduced-order model (ROM) techniques aim to approximate high-dimensional dynamical systems with lower-dimensional representations. Such reductions enable more efficient verification, control synthesis, and planning for systems whose full-order model (FOM) dynamics may be too complex to analyze directly. Classical approaches to model reduction in control include balanced truncation and related projection-based methods
Antoulas (2005)
;
Benner et al. (2015)
, moment-matching for linear and nonlinear systems
Astolfi (2010)
, and methods based on invariant manifolds
Haller and Ponsioen (2016)
. These approaches share conceptual ties with notions of abstraction in hybrid and symbolic control
Tabuada (2009)
;
Girard and Pappas (2009)
, where simpler surrogate systems are constructed to preserve relevant dynamical properties of the original one.
While these methods provide rigorous guarantees for structured or weakly nonlinear systems, extending them to high-dimensional robotic systems, particularly those with contact-rich dynamics, remains challenging. Motivated by these limitations, recent works
learn ROMs from data.
Approaches based on dynamic mode decomposition, PCA, and proper orthogonal decomposition
Brunton and Kutz (2022)
identify dominant low-dimensional structure in trajectories, while autoencoders learn nonlinear embeddings that map between full-order states and latent coordinates. A central challenge, however, is determining when properties such as stability, safety, or invariance in the latent space faithfully transfer back to the original full-order dynamics. Recent work has begun to establish sufficient conditions and approximation guarantees in this direction
Lutkus et al. (2025)
;
Nakamura et al. (2025)
, though practical instantiations for hybrid locomotion remain limited.
ROMs play a particularly influential role in legged robotics, where simple models such as the linear inverted pendulum (LIP)
Kajita et al. (2001)
and the spring-loaded inverted pendulum (SLIP)
Blickhan (1989)
capture the salient dynamics of center-of-mass motion
Raibert (1986)
;
Wensing and Revzen (2017)
. These templates are effective for control design and gait stabilization, but are inherently approximate: they do not directly arise from a systematic reduction of the FOM, and their accuracy can degrade outside of nominal operating regimes. Thus, recent work has learned reduced-order locomotion models from data
Castillo et al. (2024)
;
Chen et al. (2024a)
;
Chen et al. (2024b)
;
Li et al. (2024)
;
Starke et al. (2022)
, though questions remain regarding how to analyze their stability properties.
Overview and Contributions:
In this paper, we propose HALO (Hybrid Autoencoded LOcomotion), a framework for constructing and analyzing ROMs of hybrid robotic locomotion using autoencoder-based latent representations. Our approach represents the high-dimensional hybrid dynamics as a discrete-time system via a Poincaré return map
Westervelt et al. (2018)
. We then learn an autoencoder that embeds this return map into a low-dimensional latent space, yielding a latent Poincaré map that captures the step-to-step evolution of the locomotion gait cycle.
We first motivate this reduction in the linear setting, then extend to the nonlinear case by leveraging the existence of low-dimensional invariant surfaces associated with periodic orbits. This perspective provides practical guidance for selecting the latent dimension and clarifies when the autoencoder can represent the orbit geometry with negligible reconstruction error. Once the latent dynamics are identified, we perform Lyapunov stability analysis directly in the latent space and subsequently lift the corresponding region of attraction to the full-order state space via the decoder.
We evaluate HALO
across a suite of hybrid systems.
Across all systems, we show that stability properties inferred from the latent Poincaré dynamics predict stability in the full-order system, even when it is controlled via reinforcement learning policies.

## Method

Figure 3:
The three systems of study, their configurations, and their impacts.
We recall that the stability of a hybrid system can be characterized by its impact-to-impact dynamics. We consider a closed-loop hybrid system with state
𝐱
\mathbf{x}
and feedback controller
𝐤
⁡
(
𝐱
)
\mathbf{k}(\mathbf{x})
, learned through reinforcement learning (RL). To study stability of this system, we extract sequences of pre-impact states,
𝔇
=
{
{
𝐱
k
}
k
=
0
K
:
𝐱
0
∈
𝒳
0
⊂
ℝ
n
x
,
𝐱
k
+
1
=
𝐟
(
𝐱
k
)
}
\mathfrak{D}=\{\{\mathbf{x}_{k}\}_{k=0}^{K}:\mathbf{x}_{0}\in\mathcal{X}_{0}\subset\mathbb{R}^{n_{x}},\;\mathbf{x}_{k+1}=\mathbf{f}(\mathbf{x}_{k})\}
(7)
where
𝐟
:
ℝ
n
x
→
ℝ
n
x
\mathbf{f}:\mathbb{R}^{n_{x}}\rightarrow\mathbb{R}^{n_{x}}
is the Poincaré map of the closed-loop system and each
𝐱
k
≔
𝐱
−
\mathbf{x}_{k}\coloneq\mathbf{x}^{-}
is its state just before an impact. The configurations and pre-impact states of the systems considered here—a paddle-ball, a planar hopper, and a Unitree G1 humanoid robot—are illustrated in Figure
3
. Using the dataset
𝔇
\mathfrak{D}
, we train an autoencoder-based ROM that captures the Poincaré dynamics in a low-dimensional latent space. We then use the learned latent dynamics to predict trajectories and estimate regions of attraction of the FOM. The remainder of this section details each component of this pipeline
1
1
1
Full code is available at:
github.com/sesteban951/halo-latent-locomotion
.
.
Reinforcement Learning Controllers:
As described above, the feedback controller
𝐤
⁡
(
𝐱
)
\mathbf{k}(\mathbf{x})
is an RL policy
𝝅
ξ
​
(
𝐨
)
\bm{\pi}_{\xi}(\mathbf{o})
, where the observation
𝐨
\mathbf{o}
is a function of the state for the paddle-ball and hopper, and a function of the state together with auxiliary variables for the humanoid. The paddle-ball policy is trained to regulate the ball at a desired height, whereas the hopper and humanoid policies are trained to maintain a constant forward velocity.
We train controllers for the paddle-ball and hopper using the actor-critic PPO algorithm implemented in Brax
Freeman et al. (2021)
. The learned policy directly outputs bounded torque commands at 50 Hz. We train a locomotion controller for the G1 humanoid using the actor-critic PPO algorithm implemented in mjlab
Zakka et al. (2026)
with the
rsl_rl
library. The learned policy outputs joint position setpoints at 50 Hz, which are tracked by joint-level PD controllers to produce motor torques. Additional implementation details are provided in Appendix
C.1
.
Poincaré Data Collection:
To collect training data, we simulate the closed-loop system in MuJoCo MJX
Todorov et al. (2012)
for a fixed time horizon, using the JAX backend for the paddle-ball and hopper systems and the Warp backend for the G1. At each impact, we use MuJoCo sensor measurements to record the generalized position and velocity in a contact-relative frame—in the world frame for the paddle-ball, relative to the foot position for the hopper, and in a yaw-aligned foot frame for the humanoid. After pruning faulty Poincaré data, such as impacts with sliding contact or contact chatter, we obtain clean sequences of Poincaré returns.
Remark 7
.
In practice, impacts can be estimated without dedicated contact sensors. Kinematic methods use foot height and velocity; dynamic methods use changes in generalized velocities, accelerations, or momentum; force methods use joint torques and ground reaction force estimates.
Autoencoder Architecture:
We parameterize the autoencoder using three networks: an
encoder
that projects the FOM state onto a lower-dimensional latent state,
𝐳
k
=
𝐄
ϕ
​
(
𝐱
k
)
\mathbf{z}_{k}=\mathbf{E}_{\bm{\phi}}(\mathbf{x}_{k})
, a
decoder
that reconstructs the FOM state from the latent state,
𝐱
^
k
=
𝐃
𝝍
​
(
𝐳
k
)
\hat{\mathbf{x}}_{k}=\mathbf{D}_{\bm{\psi}}(\mathbf{z}_{k})
, and a
latent dynamics
network that propagates the latent state forward via learned residual ROM dynamics,
𝐳
k
+
1
=
𝐠
𝝆
​
(
𝐳
k
)
≔
𝐳
k
+
𝐠
¯
𝝆
​
(
𝐳
k
)
.
\mathbf{z}_{k+1}=\mathbf{g}_{\bm{\rho}}(\mathbf{z}_{k})\coloneq\mathbf{z}_{k}+\bar{\mathbf{g}}_{\bm{\rho}}(\mathbf{z}_{k}).
(8)
The encoder and decoder are mirrored MLPs with swish activations, sized
[
64
,
32
,
16
]
[64,32,16]
and
[
16
,
32
,
64
]
[16,32,64]
for the hopper and paddle-ball systems, and
[
256,128
,
64
]
[256,128,64]
and
[
64,128,256
]
[64,128,256]
for the humanoid. The latent dynamics network is an MLP with swish activations, sized
[
64
,
64
,
64
]
[64,64,64]
for the hopper and paddle-ball systems, and
[
128,128,128
]
[128,128,128]
for the humanoid. The latent dimension
n
z
n_{z}
is 2 for the paddle-ball, 4 for the hopper, and 12 for the humanoid. More details are provided in Section
C.2
.
Loss Function:
Figure 4:
A visual guide of the losses.
Using the insights from Section
3
, we seek parameters
ϕ
\bm{\phi}
,
𝝍
\bm{\psi}
, and
𝝆
\bm{\rho}
such that the encoder
𝐄
ϕ
\mathbf{E}_{\bm{\phi}}
, decoder
𝐃
𝝍
\mathbf{D}_{\bm{\psi}}
, and latent dynamics model
𝐠
𝝆
\mathbf{g}_{\bm{\rho}}
approximate the FOM Poincaré map:
inf
ϕ
,
𝝍
,
𝝆
∑
𝐱
k
∈
𝔇
‖
𝐟
⁡
(
𝐱
k
)
−
(
𝐃
𝝍
∘
𝐠
𝝆
∘
𝐄
ϕ
)
​
(
𝐱
k
)
‖
2
.
\displaystyle\inf_{\bm{\phi},\bm{\psi},\bm{\rho}}\sum_{\mathbf{x}_{k}\in\mathfrak{D}}\left\|\mathbf{f}(\mathbf{x}_{k})-(\mathbf{D}_{\bm{\psi}}\circ\mathbf{g}_{\bm{\rho}}\circ\mathbf{E}_{\bm{\phi}})(\mathbf{x}_{k})\right\|^{2}.
(9)
The encoder, decoder, and latent dynamics networks are trained jointly using a weighted sum of losses, each enforcing a different aspect of reconstruction accuracy and dynamical consistency. Inspired by
Lutkus et al. (2025)
, we organize these losses into reconstruction, conjugacy, prediction, whitening, and regularization terms. The loss functions are summarized in Table
1
and illustrated in Figure
4
.
The reconstruction losses
L
rec
,
x
L_{\mathrm{rec},x}
and
L
rec
,
z
L_{\mathrm{rec},z}
ensure that the encoder-decoder pair
(
𝐄
ϕ
,
𝐃
𝝍
)
(\mathbf{E}_{\bm{\phi}},\mathbf{D}_{\bm{\psi}})
accurately reconstructs states in both the full-order and latent spaces, respectively. The forward and backward conjugacy losses,
L
fwd
L_{\mathrm{fwd}}
and
L
bck
L_{\mathrm{bck}}
, encourage the learned latent dynamics to align with the FOM Poincaré dynamics. The prediction loss
L
pred
L_{\mathrm{pred}}
improves multi-step rollout accuracy, encouraging the learned ROM to capture long-horizon trajectory evolution rather than myopic one-step behavior. The whitening loss
L
iso
L_{\mathrm{iso}}
helps prevent latent space collapse by encouraging the latent covariance to remain close to identity. Finally, the weight regularization loss
L
reg
L_{\mathrm{reg}}
penalizes large network weights to reduce overfitting to data. The total loss function is given by:
L
=
L
rec
,
x
+
L
rec
,
z
+
L
fwd
+
L
bck
+
L
pred
+
L
iso
+
L
reg
.
L=L_{\mathrm{rec},x}+L_{\mathrm{rec},z}+L_{\mathrm{fwd}}+L_{\mathrm{bck}}+L_{\mathrm{pred}}+L_{\mathrm{iso}}+L_{\mathrm{reg}}.
(10)
All loss terms are scaled as shown in Table
1
, where all weights are set to 1.0 except for the L2 regularization term, which uses
λ
reg
=
10
−
6
\lambda_{\mathrm{reg}}=10^{-6}
.
Loss
Definition
Purpose
L
rec
,
x
​
(
ϕ
,
𝝍
)
L_{\mathrm{rec},x}(\bm{\phi},\bm{\psi})
λ
rec
,
x
B
​
K
​
∑
b
,
k
=
1
B
,
K
‖
𝐱
k
(
b
)
−
(
𝐃
𝝍
∘
𝐄
ϕ
)
​
(
𝐱
k
(
b
)
)
‖
2
\frac{\lambda_{\mathrm{rec},x}}{BK}\sum_{b,k=1}^{B,K}\left\|\mathbf{x}^{(b)}_{k}-(\mathbf{D}_{\bm{\psi}}\!\circ\!\mathbf{E}_{\bm{\phi}})(\mathbf{x}^{(b)}_{k})\right\|^{2}
Reconstruction
L
rec
,
z
​
(
ϕ
,
𝝍
)
L_{\mathrm{rec},z}(\bm{\phi},\bm{\psi})
λ
rec
,
z
B
​
K
​
∑
b
,
k
=
1
B
,
K
‖
𝐄
ϕ
​
(
𝐱
k
(
b
)
)
−
(
𝐄
ϕ
∘
𝐃
𝝍
∘
𝐄
ϕ
)
​
(
𝐱
k
(
b
)
)
‖
2
\frac{\lambda_{\mathrm{rec},z}}{BK}\sum_{b,k=1}^{B,K}\left\|\mathbf{E}_{\bm{\phi}}(\mathbf{x}^{(b)}_{k})-(\mathbf{E}_{\bm{\phi}}\!\circ\!\mathbf{D}_{\bm{\psi}}\!\circ\!\mathbf{E}_{\bm{\phi}})(\mathbf{x}^{(b)}_{k})\right\|^{2}
Reconstruction
L
fwd
​
(
ϕ
,
𝝆
)
L_{\mathrm{fwd}}(\bm{\phi},\bm{\rho})
λ
fwd
B
⁡
(
K
−
1
)
​
∑
b
,
k
=
1
B
,
K
−
1
‖
𝐄
ϕ
​
(
𝐱
k
+
1
(
b
)
)
−
(
𝐠
𝝆
∘
𝐄
ϕ
)
​
(
𝐱
k
(
b
)
)
‖
2
\frac{\lambda_{\mathrm{fwd}}}{B(K-1)}\sum_{b,k=1}^{B,K-1}\left\|\mathbf{E}_{\bm{\phi}}(\mathbf{x}^{(b)}_{k+1})-(\mathbf{g}_{\bm{\rho}}\!\circ\!\mathbf{E}_{\bm{\phi}})(\mathbf{x}^{(b)}_{k})\right\|^{2}
Conjugacy
L
bck
​
(
ϕ
,
𝝆
,
𝝍
)
L_{\mathrm{bck}}(\bm{\phi},\bm{\rho},\bm{\psi})
λ
bck
B
⁡
(
K
−
1
)
​
∑
b
,
k
=
1
B
,
K
−
1
‖
𝐱
k
+
1
(
b
)
−
(
𝐃
𝝍
∘
𝐠
𝝆
∘
𝐄
ϕ
)
​
(
𝐱
k
(
b
)
)
‖
2
\frac{\lambda_{\mathrm{bck}}}{B(K-1)}\sum_{b,k=1}^{B,K-1}\left\|\mathbf{x}^{(b)}_{k+1}-(\mathbf{D}_{\bm{\psi}}\!\circ\!\mathbf{g}_{\bm{\rho}}\!\circ\!\mathbf{E}_{\bm{\phi}})(\mathbf{x}^{(b)}_{k})\right\|^{2}
Conjugacy
L
pred
​
(
ϕ
,
𝝆
,
𝝍
)
L_{\mathrm{pred}}(\bm{\phi},\bm{\rho},\bm{\psi})
λ
pred
B
⁡
(
K
−
1
)
​
∑
b
=
1
,
k
=
2
B
,
K
‖
𝐱
k
(
b
)
−
(
𝐃
𝝍
∘
𝐠
𝝆
(
k
−
1
)
∘
𝐄
ϕ
)
​
(
𝐱
1
(
b
)
)
‖
2
\frac{\lambda_{\mathrm{pred}}}{B(K-1)}\sum_{b=1,k=2}^{B,K}\left\|\mathbf{x}^{(b)}_{k}-\big(\mathbf{D}_{\bm{\psi}}\!\circ\!\mathbf{g}_{\bm{\rho}}^{(k-1)}\!\circ\!\mathbf{E}_{\bm{\phi}}\big)(\mathbf{x}^{(b)}_{1})\right\|^{2}
Prediction
L
iso
​
(
ϕ
)
L_{\mathrm{iso}}(\bm{\phi})
λ
iso
​
‖
𝐈
−
1
B
​
K
​
∑
b
,
k
=
1
B
,
K
(
𝐄
ϕ
​
(
𝐱
k
(
b
)
)
−
𝝁
)
​
(
𝐄
ϕ
​
(
𝐱
k
(
b
)
)
−
𝝁
)
⊤
‖
F
2
\lambda_{\mathrm{iso}}\left\|\mathbf{I}-\frac{1}{BK}\sum_{b,k=1}^{B,K}\big(\mathbf{E}_{\bm{\phi}}(\mathbf{x}^{(b)}_{k})-\bm{\mu}\big)\big(\mathbf{E}_{\bm{\phi}}(\mathbf{x}^{(b)}_{k})-\bm{\mu}\big)^{\!\top}\right\|_{F}^{2}
Whitening
L
reg
​
(
ϕ
,
𝝆
,
𝝍
)
L_{\mathrm{reg}}(\bm{\phi},\bm{\rho},\bm{\psi})
λ
reg
​
∑
w
i
∈
𝐖
⁡
(
ϕ
,
𝝍
,
𝝆
)
‖
w
i
‖
2
\lambda_{\mathrm{reg}}\sum_{w_{i}\in\mathbf{W}(\bm{\phi},\bm{\psi},\bm{\rho})}\|w_{i}\|^{2}
L2 Regularization
Table 1:
Summary of the loss terms.
𝐖
\mathbf{W}
denotes trainable weights (excluding biases), and
𝐠
𝝆
(
k
−
1
)
\mathbf{g}_{\bm{\rho}}^{(k-1)}
denotes repeated application of
𝐠
𝝆
\mathbf{g}_{\bm{\rho}}
. In addition,
B
B
denotes the mini-batch size and
K
K
denotes the trajectory length.
Training:
We train autoencoders using the Adam optimizer
Kingma and Ba (2014)
with shuffled mini-batches of trajectory data. The model is implemented in JAX and Flax
Bradbury et al. (2018)
;
Heek et al. (2024)
. Before training, each state feature is normalized independently using training-set statistics aggregated over all trajectories and time steps. The dataset of trajectories is partitioned into disjoint training, validation, and test sets. More details are provided in Appendix
C.2
.
Latent Region of Attraction:
To approximate the region of attraction (ROA) for the FOM impact-to-impact dynamics, we linearize the latent Poincaré map around the latent equilibrium point
2
2
2
Linearization of
𝐠
𝝆
​
(
𝐳
)
\mathbf{g}_{\bm{\rho}}(\mathbf{z})
is obtained via automatic differentiation using
jax.jacfwd
, giving
𝐐
=
∂
𝐠
𝝆
∂
𝐳
|
𝐳
∗
\mathbf{Q}=\frac{\partial\mathbf{g}_{\bm{\rho}}}{\partial\mathbf{z}}|_{\mathbf{z}^{*}}
.
𝐳
∗
\mathbf{z}^{*}
, yielding the discrete-time linear system
𝐳
k
+
1
=
𝐠
𝝆
​
(
𝐳
k
)
≈
𝐐
⁡
(
𝐳
k
−
𝐳
∗
)
+
𝐠
𝝆
​
(
𝐳
∗
)
\mathbf{z}_{k+1}=\mathbf{g}_{\bm{\rho}}(\mathbf{z}_{k})\approx\mathbf{Q}(\mathbf{z}_{k}-\mathbf{z}^{*})+\mathbf{g}_{\bm{\rho}}(\mathbf{z}^{*})
. Without loss of generality, we shift coordinates so that
𝐳
∗
=
𝟎
\mathbf{z}^{*}=\mathbf{0}
. We then construct a discrete-time Lyapunov function
V
𝐳
​
(
𝐳
)
=
𝐳
⊤
​
𝐏𝐳
V_{\mathbf{z}}(\mathbf{z})=\mathbf{z}^{\top}\mathbf{P}\mathbf{z}
where
𝐏
≻
𝟎
\mathbf{P}\succ\mathbf{0}
satisfies the discrete-time Lyapunov equation
𝐐
⊤
​
𝐏𝐐
−
𝐏
+
𝐈
=
𝟎
\mathbf{Q}^{\top}\mathbf{P}\mathbf{Q}-\mathbf{P}+\mathbf{I}=\mathbf{0}
.
Next, we construct an estimated ROA around the latent equilibrium and map it through the decoder to obtain an approximation of the full-order ROA. Consider the following set:
𝒟
=
{
𝐳
∈
ℝ
n
z
:
Δ
​
V
𝐳
​
(
𝐳
)
≤
0
}
=
{
𝐳
∈
ℝ
n
z
:
−
𝐳
⊤
​
𝐳
+
𝐠
𝝆
​
(
𝐳
)
⊤
​
𝐏𝐠
𝝆
​
(
𝐳
)
−
𝐳
⊤
​
𝐐
⊤
​
𝐏𝐐𝐳
≤
0
}
.
\mathcal{D}=\{\mathbf{z}\in\mathbb{R}^{n_{z}}:\Delta V_{\mathbf{z}}(\mathbf{z})\leq 0\}=\Bigl\{\mathbf{z}\in\mathbb{R}^{n_{z}}:-\mathbf{z}^{\top}\mathbf{z}+\mathbf{g}_{\bm{\rho}}(\mathbf{z})^{\top}\mathbf{P}\mathbf{g}_{\bm{\rho}}(\mathbf{z})-\mathbf{z}^{\top}\mathbf{Q}^{\top}\mathbf{P}\mathbf{Q}\mathbf{z}\leq 0\Bigr\}.
(11)
This is the set of points for which the latent candidate Lyapunov function
V
𝐳
V_{\mathbf{z}}
does not increase under the nonlinear latent dynamics.
To obtain the largest Lyapunov sublevel set contained in
𝒟
\mathcal{D}
, we solve:
c
∗
=
inf
𝐳
∈
∂
𝒟
𝐳
⊤
​
𝐏𝐳
c^{\ast}=\inf_{\mathbf{z}\in\partial\mathcal{D}}\mathbf{z}^{\top}\mathbf{P}\mathbf{z}
(12)
using a Monte Carlo sampling strategy
3
3
3
Direct optimization of
c
∗
c^{*}
is challenging as
∂
𝒟
\partial\mathcal{D}
is non-convex and implicitly determined by a neural network. We thus use a simple but effective sampling-based sweep to obtain a conservative ROA estimate.
. The sublevel set
𝛀
c
∗
≔
{
𝐳
:
𝐳
⊤
​
𝐏𝐳
≤
c
∗
}
\bm{\Omega}_{c^{*}}\coloneq\{\mathbf{z}:\mathbf{z}^{\top}\mathbf{P}\mathbf{z}\leq c^{*}\}
is taken as the latent ROA estimate.
We then sample points in
𝛀
c
∗
\bm{\Omega}_{c^{\ast}}
, decode them through
𝐃
𝝍
\mathbf{D}_{\bm{\psi}}
to obtain full-order states, and simulate these initial conditions under the FOM dynamics to assess if they remain stable.
