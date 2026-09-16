# WAM-RL: World-Action Model Reinforcement Learning with Reconstruction Rewards and Online Video SFT

paper_id: semanticscholar:f3be325255c5fd6d53ae023730bdd0d542ee08fe
tier: T2
source_used: html_arxiv
warning: none

## Intro

Recent advances in World-Action (WA) models have demonstrated strong generalization ability and data efficiency for robot policy learning
[
7
,
12
,
10
]
. By jointly modeling future observations and actions through video-based generative models, WA frameworks enable implicit planning and have shown promising performance in both simulation and real-world settings. Compared to conventional vision-language-action (VLA) models, WA models leverage predictive structure in visual dynamics, which provides a more powerful inductive bias for long-horizon decision making
[
18
,
4
]
.
Despite these advantages, existing WA models are primarily trained with supervised learning from expert trajectories. This reliance on demonstration data imposes two fundamental limitations. First, the learned policy is constrained by the support of the training data and struggles to acquire fine-grained manipulation skills beyond the demonstration distribution. Second, the model lacks the ability to continuously improve through interaction with the environment, which is essential for adapting to new scenarios and correcting errors during execution.
A natural direction to address these limitations is to incorporate reinforcement learning (RL). However, applying RL to WA models is non-trivial. Unlike conventional policies, WA models consist of two tightly coupled components: a world model that generates future predictions and an action model that maps these predictions to executable actions. Existing RL approaches for VLA models mainly focus on optimizing the action policy while treating the visual representation as fixed. In contrast, in WA models, the action model is deeply dependent on the latent space of the world model. Naively applying RL or online fine-tuning can lead to distribution shifts in the latent representation, causing instability and performance degradation.
In this work, we propose
WAM-RL
, a reinforcement learning framework that enables joint optimization of the world model and the action model through online interaction. Our key observation is that the primary capability of WA models originates from the world model, while the action model mainly serves as a translator that converts latent predictions into actions. Based on this perspective, we design a two-part optimization scheme. The world model is refined via online video self-supervised fine-tuning using successful trajectories, with a KL regularization term to stabilize its latent space. The action model is optimized using reinforcement learning with a reconstruction-based reward that measures the consistency between imagined and executed outcomes.
Beyond the method itself, our experiments reveal an important insight about learning in WA models. We find that optimizing only the action model leads to improvements on short-horizon tasks but fails to yield significant gains on long-horizon tasks. This is because the actor is constrained by the accuracy of the world model, and cannot correct accumulated prediction errors over long horizons. In contrast, jointly optimizing both the world model and the action model is critical for achieving strong performance in complex tasks.
Empirically, our approach achieves consistent improvements on both LIBERO
[
13
]
and RLBench
[
6
]
benchmarks. Notably, we observe that online adaptation of the world model leads to more realistic predictions of recovery behaviors, such as re-attempting failed grasps, which further improves policy robustness.
In summary, our contributions are as follows:
•
We introduce WAM-RL, the first framework that incorporates reinforcement learning into the World-Action paradigm with joint optimization of the world model and the action model.
•
We propose a stable training strategy that combines online video SFT with KL regularization and reconstruction-based RL for the actor.
•
We provide empirical insights showing that actor-only optimization is insufficient for long-horizon tasks, highlighting the importance of jointly improving the world model.

## Method

3.1
Preliminaries
In this section, we briefly review flow matching for video generation and its extension to reinforcement learning via Flow-SDE, which serves as the foundation for optimizing flow-based action models.
Flow Matching for Video Generation.
Flow matching provides a continuous-time formulation for generative modeling by learning a vector field that transports a simple distribution to a data distribution. Let
x
0
∼
p
0
​
(
x
)
x_{0}\sim p_{0}(x)
denote a sample from a base distribution (e.g., Gaussian noise), and
x
1
∼
p
data
​
(
x
)
x_{1}\sim p_{\text{data}}(x)
denote a data sample (e.g., a video sequence). Flow matching defines a conditional trajectory
x
t
x_{t}
for
t
∈
[
0
,
1
]
t\in[0,1]
that interpolates between
x
0
x_{0}
and
x
1
x_{1}
, and learns a time-dependent vector field
v
θ
​
(
x
t
,
t
)
v_{\theta}(x_{t},t)
such that:
d
​
x
t
d
​
t
=
v
θ
​
(
x
t
,
t
)
.
\frac{dx_{t}}{dt}=v_{\theta}(x_{t},t).
(1)
The model is trained to match the target velocity field along the trajectory by minimizing:
ℒ
FM
=
𝔼
x
0
,
x
1
,
t
​
[
‖
v
θ
​
(
x
t
,
t
)
−
v
∗
​
(
x
t
,
t
)
‖
2
]
,
\mathcal{L}_{\text{FM}}=\mathbb{E}_{x_{0},x_{1},t}\left[\left\|v_{\theta}(x_{t},t)-v^{*}(x_{t},t)\right\|^{2}\right],
(2)
where
v
∗
​
(
x
t
,
t
)
v^{*}(x_{t},t)
denotes the ground-truth transport direction defined by the interpolation between
x
0
x_{0}
and
x
1
x_{1}
. In video generation,
x
1
x_{1}
corresponds to future visual observations, and the learned vector field enables the model to generate temporally coherent predictions by integrating the flow from noise to video frames.
Flow-SDE for Reinforcement Learning with Flow-Based Policies.
While flow matching enables expressive generative modeling, it results in a deterministic generation process, which makes it incompatible with reinforcement learning due to the lack of stochasticity and tractable action likelihoods. To address this, Flow-SDE introduces stochasticity into the flow dynamics by converting the deterministic ODE into a stochastic differential equation (SDE):
d
​
x
t
=
v
θ
​
(
x
t
,
t
)
​
d
​
t
+
σ
​
d
​
W
t
,
dx_{t}=v_{\theta}(x_{t},t)\,dt+\sigma\,dW_{t},
(3)
where
d
​
W
t
dW_{t}
denotes Brownian motion and
σ
\sigma
controls the noise scale.
This stochastic formulation induces a sequence of conditional transitions:
p
⁡
(
x
t
−
1
∣
x
t
)
=
𝒩
⁡
(
x
t
−
1
,
μ
θ
​
(
x
t
,
t
)
,
σ
2
​
I
)
,
p(x_{t-1}\mid x_{t})=\mathcal{N}\big(x_{t-1};\mu_{\theta}(x_{t},t),\sigma^{2}I\big),
(4)
which enables tractable likelihood estimation over the denoising trajectory. Consequently, the flow-based policy can be interpreted as a Markov decision process (MDP) in the latent space, where the state corresponds to
x
t
x_{t}
and the action corresponds to the denoising transition.
Given this formulation, the likelihood of an action sequence can be decomposed as:
log
⁡
π
θ
​
(
a
∣
s
)
=
∑
t
log
⁡
p
⁡
(
x
t
−
1
∣
x
t
)
,
\log\pi_{\theta}(a\mid s)=\sum_{t}\log p(x_{t-1}\mid x_{t}),
(5)
allowing standard policy gradient methods to be applied:
∇
θ
J
=
𝔼
⁡
[
∇
θ
​
log
​
π
θ
​
(
a
∣
s
)
​
A
​
(
s
,
a
)
]
,
\nabla_{\theta}J=\mathbb{E}\left[\nabla_{\theta}\log\pi_{\theta}(a\mid s)\,A(s,a)\right],
(6)
where
A
⁡
(
s
,
a
)
A(s,a)
denotes the advantage function.
This formulation enables reinforcement learning for flow-based action models by treating the denoising process as a stochastic trajectory, providing both exploration and a well-defined optimization objective.
3.2
Overall Framework
We build upon the World-Action (WA) paradigm, where a policy consists of a world model and an action model (actor). We observe that the core capability of WA models primarily comes from the world model, which captures predictive structure and supports implicit planning through video generation. In contrast, the actor mainly functions as a translator that maps the latent representations of the world model into executable actions.
Figure 1
:
Overview of WAM-RL. Our framework jointly optimizes a world model and an action model (actor) through online interaction. The world model generates imagined future observations, which are translated into actions by the actor and executed in the real environment. The resulting observations are then used to update both components: the world model is refined via online video self-supervised fine-tuning with KL regularization using successful trajectories, while the actor is optimized with a reconstruction-based dense reward that measures the consistency between imagined and executed outcomes. This design enables the world model and actor to co-evolve, leading to improved planning accuracy and robust long-horizon behavior.
As illustrated in Fig.
1
, improving WA models requires jointly addressing two aspects. First, the world model must be improved so that it can generate more accurate and task-relevant future predictions, which directly affects planning quality. Second, the actor must be improved so that it can faithfully translate the world model’s latent representations into real-world actions.
To this end, we propose WAM-RL, which optimizes the world model and the action model through two coordinated mechanisms. The world model is refined via online video self-supervised fine-tuning using successful trajectories collected during interaction, while a KL regularization term is introduced to stabilize its latent representation. The actor is optimized with reinforcement learning using a reconstruction-based dense reward that measures the consistency between imagined and executed outcomes.
In the overall pipeline shown in Fig.
1
, the world model generates imagined future observations in its latent space, which are translated into actions by the actor. The environment then produces real observations, which are used to refine both components. We next describe these two parts in detail.
3.3
Online Video SFT with KL Regularization
A natural way to improve the world model is to leverage trajectories collected during reinforcement learning and perform online fine-tuning. Given a sequence of observations
x
1
:
T
x_{1:T}
from successful rollouts, we train the world model to better predict future observations using a standard video modeling objective:
ℒ
video
=
𝔼
x
1
:
T
[
ℓ
(
f
θ
(
x
<
t
)
,
x
t
)
]
,
\mathcal{L}_{\text{video}}=\mathbb{E}_{x_{1:T}}\left[\ell\big(f_{\theta}(x_{<t}),x_{t}\big)\right],
(7)
where
f
θ
f_{\theta}
denotes the world model and
ℓ
\ell
is a prediction loss such as flow matching or reconstruction.
However, naively fine-tuning the world model together with the actor leads to severe instability. The actor depends critically on the latent feature distribution induced by the world model, and online updates can significantly shift this distribution. As a result, the actor quickly becomes ineffective due to the mismatch between its learned policy and the evolving latent space.
To stabilize training, we introduce a KL regularization term that constrains the latent feature distribution of the updated world model to remain close to that of the pretrained model. Since the intermediate features of a DiT-based world model are deterministic, we construct a Gaussian approximation over latent features to make the KL divergence well-defined.
Let
z
t
=
f
θ
​
(
x
<
t
)
z_{t}=f_{\theta}(x_{<t})
denote the latent feature at time step
t
t
, and let
z
t
old
=
f
old
​
(
x
<
t
)
z_{t}^{\text{old}}=f_{\text{old}}(x_{<t})
denote the corresponding feature from a frozen copy of the pretrained world model. We define the approximate latent distributions as:
p
θ
​
(
z
t
∣
x
<
t
)
=
𝒩
⁡
(
z
t
,
Σ
θ
)
,
p
old
​
(
z
t
∣
x
<
t
)
=
𝒩
⁡
(
z
t
old
,
Σ
old
)
,
p_{\theta}(z_{t}\mid x_{<t})=\mathcal{N}\big(z_{t},\Sigma_{\theta}\big),\quad p_{\text{old}}(z_{t}\mid x_{<t})=\mathcal{N}\big(z_{t}^{\text{old}},\Sigma_{\text{old}}\big),
(8)
where the mean is given by the deterministic feature, and
Σ
θ
\Sigma_{\theta}
and
Σ
old
\Sigma_{\text{old}}
are diagonal covariance matrices.
The covariance
Σ
θ
\Sigma_{\theta}
is estimated using an exponential moving average (EMA) over latent feature statistics during training, capturing the scale of variation in each feature dimension. For the reference distribution, we maintain a frozen copy of the pretrained world model and estimate
Σ
old
\Sigma_{\text{old}}
from its latent features; this covariance is fixed throughout training. This construction provides a consistent reference distribution while allowing the current model to adapt its feature scale gradually.
The KL regularization is then defined as:
ℒ
KL
=
𝔼
t
[
D
KL
(
𝒩
(
z
t
,
Σ
θ
)
∥
𝒩
(
z
t
old
,
Σ
old
)
)
]
.
\mathcal{L}_{\text{KL}}=\mathbb{E}_{t}\left[D_{\text{KL}}\Big(\mathcal{N}(z_{t},\Sigma_{\theta})\;\|\;\mathcal{N}(z_{t}^{\text{old}},\Sigma_{\text{old}})\Big)\right].
(9)
The final training objective for the world model is:
ℒ
WM
=
ℒ
video
+
λ
KL
​
ℒ
KL
.
\mathcal{L}_{\text{WM}}=\mathcal{L}_{\text{video}}+\lambda_{\text{KL}}\mathcal{L}_{\text{KL}}.
(10)
This regularization constrains the updated world model to preserve the latent feature geometry expected by the actor, preventing abrupt distribution shifts while still permitting gradual adaptation. In practice, we observe that this approach stabilizes joint training and leads to consistent, though moderate, improvements, reflecting a trade-off between stability and adaptability.
3.4
Action Model RL with Reconstruction-Based Reward
The action model (actor) serves as a translator that converts the world model’s imagined futures into executable actions. Therefore, a natural objective for optimizing the actor is to ensure that the executed trajectory in the real environment faithfully realizes the world model’s predictions.
Formally, let
x
^
t
+
1
:
t
+
H
\hat{x}_{t+1:t+H}
denote the future observations predicted by the world model, and let
x
t
+
1
:
t
+
H
x_{t+1:t+H}
denote the observations obtained by executing the actor in the environment. We define a reconstruction-based reward that measures the consistency between imagined and actual trajectories:
r
t
=
sim
(
x
^
t
+
1
:
t
+
H
,
x
t
+
1
:
t
+
H
)
,
r_{t}=\mathrm{sim}(\hat{x}_{t+1:t+H},x_{t+1:t+H}),
(11)
where
sim
⁡
(
⋅
,
⋅
)
\mathrm{sim}(\cdot,\cdot)
is a similarity function.
This reward directly aligns the actor with the world model: instead of optimizing for task-specific objectives, the actor is encouraged to realize the latent plan encoded in the world model’s predictions. As a result, the actor learns to execute actions that are consistent with the predictive structure of the world model.
We consider several choices for the similarity function, including pixel-level mean squared error, optical flow consistency, DINOv2
[
14
]
feature similarity, and V-JEPA2
[
1
]
feature similarity. These choices correspond to different notions of alignment, ranging from low-level appearance matching to high-level semantic similarity. In particular, optical flow emphasizes motion consistency, while feature-based metrics capture semantic alignment.
Empirically, we find that different similarity functions lead to distinct reward characteristics. Motion-based signals such as optical flow provide stronger discrimination between successful and failed trajectories, while pixel-based reconstruction is more aligned with the training objective of the world model. As we show in Section
4.3
, this trade-off between discriminability and alignment plays a critical role in downstream performance.
Given the reward, we optimize the actor using a policy gradient objective:
∇
ϕ
J
=
𝔼
⁡
[
∇
ϕ
​
log
​
π
ϕ
​
(
a
t
∣
s
t
)
​
A
t
]
,
\nabla_{\phi}J=\mathbb{E}\left[\nabla_{\phi}\log\pi_{\phi}(a_{t}\mid s_{t})A_{t}\right],
(12)
where the advantage
A
t
A_{t}
is computed from the reconstruction-based rewards.
Overall, this formulation enables the actor to ground the world model’s imagined trajectories into real-world execution, effectively bridging the gap between prediction and action. By aligning execution with imagination, the actor inherits the planning capability of the world model and translates it into robust behavior.
