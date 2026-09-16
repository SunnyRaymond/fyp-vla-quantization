# Dreaming Of Others: Latent Teammate Modeling In World Models For Multi-Agent Reinforcement Learning

paper_id: arxiv:2605.31361v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

World models have emerged as a powerful paradigm for reinforcement learning (RL), enabling agents to learn compact latent dynamics and train decision policies from imagined trajectories
(
Hafner et al., 2025
)
. Despite their success in single-agent domains, deploying world models in cooperative multi-agent RL (MARL) remains challenging because the behavior of teammates introduces non-stationary, partially observable structure that is typically compressed as undifferentiated noise.
We advocate extending world models from simulating environments to simulating
others
within them. Concretely, we propose a teammate-conditioned world model that (i) factorizes the latent state into environment and teammate components and (ii) augments the model with a Theory-of-Mind (ToM) head that infers latent embeddings of partner behavior from partial histories. The resulting teammate latents condition the actor and critic during imagination, allowing agents to anticipate partner variability and adapt to unseen collaborators without access to their internal observations or policies.
This focus on collaboration with humans removes many of the simplifications common in multi-agent reinforcement learning. Agents cannot rely on centralized training with privileged information, explicit communication channels, or access to a partner’s policy, goals, or observations. The human partner instead appears as a partially observed, adaptive process whose behavior must be inferred and integrated into the agent’s predictive model.
This paper is a proposal intended to catalyze discussion in the world-models community; we do not report empirical results. We precisely describe the architecture, objectives, and evaluation protocol, argue why modeling teammates as structured latent processes can reduce apparent non-stationarity, and outline how to assess zero-shot and few-shot coordination. The broader goal is human AI collaboration, where partners are partially observable and heterogeneous
(
Carroll et al., 2019
;
Liang et al., 2024
)
.

## Method

3.1
Design Rationale
In cooperative settings, observations reflect both physical dynamics and the evolving behavior of teammates. Treating other agents as exogenous noise induces apparent non-stationarity as partners change policies. We posit that modeling teammates as structured latent processes, separate from environmental latents, can reduce non-stationarity, produce socially consistent imagined rollouts, and enable rapid adaptation to new partners.
3.2
Architecture
At each time step
t
t
, the encoder consumes the controlled agent’s observation
x
t
x_{t}
and action
a
t
0
a_{t}^{0}
to produce a deterministic hidden state
h
t
h_{t}
. We posit a factorized stochastic latent
z
t
=
[
z
t
env
,
z
t
team
]
,
z_{t}=\big[z_{t}^{\text{env}},\;z_{t}^{\text{team}}\big],
where
z
t
env
z_{t}^{\text{env}}
captures environment dynamics and
z
t
team
z_{t}^{\text{team}}
encodes inferred teammate behavior.
Two decoders operate on these latents. An observation decoder maps
z
t
env
z_{t}^{\text{env}}
to
x
^
t
\hat{x}_{t}
. A teammate-policy decoder maps
z
t
team
z_{t}^{\text{team}}
to a predictive distribution
π
^
t
j
​
(
⋅
)
\hat{\pi}_{t}^{j}(\cdot)
over the teammate’s next action. Actions from all agents
(
a
t
0
,
a
t
j
)
(a_{t}^{0},a_{t}^{j})
drive the transition to
h
t
+
1
h_{t+1}
. During actor-critic learning, the hidden state and teammate latents condition policy and value heads to generate
(
a
t
0
,
v
t
,
r
t
)
(a_{t}^{0},v_{t},r_{t})
and support imagination with sampled
z
t
team
z_{t}^{\text{team}}
.
We retain standard Dreamer training for the world model and control components
(
Hafner et al., 2025
)
and add a dedicated ToM objective for teammate modeling, without requiring shared imagination, centralized policies, or explicit communication
(
Lobos-Tsunekawa et al., 2022
;
Toledo and Prorok, 2024
;
Shi et al., 2025
)
.
Figure 1:
World model and teammate modeling.
An RSSM with factorized latent
z
t
=
[
z
t
e
​
n
​
v
,
z
t
t
​
e
​
a
​
m
]
z_{t}=[z_{t}^{env},z_{t}^{team}]
. The decoder reconstructs
x
^
t
\hat{x}_{t}
from
z
t
e
​
n
​
v
z_{t}^{env}
and predicts teammate policy
π
^
t
j
​
(
⋅
)
\hat{\pi}_{t}^{j}(\cdot)
from
z
t
t
​
e
​
a
​
m
z_{t}^{team}
. Actions
(
a
t
0
,
a
t
j
)
(a_{t}^{0},a_{t}^{j})
update the transition to
h
t
+
1
h_{t+1}
. The ToM loss supervises
π
^
t
j
\hat{\pi}_{t}^{j}
.
3.3
Teammate-Modeling Objective
Let
π
t
j
​
(
⋅
)
\pi_{t}^{j}(\cdot)
denote the empirical or behavior distribution of the teammate’s next action at time
t
t
obtained from data, and
π
^
t
j
​
(
⋅
)
\hat{\pi}_{t}^{j}(\cdot)
the model’s prediction from
z
t
team
z_{t}^{\text{team}}
. We minimize a calibrated cross-entropy with temporal regularization:
ℒ
ToM
=
𝔼
t
[
−
∑
a
π
t
j
(
a
)
log
π
^
t
j
(
a
)
]
+
α
KL
(
q
(
z
t
team
∣
h
t
)
∥
p
(
z
t
team
∣
h
t
−
1
,
a
t
−
1
)
)
,
\mathcal{L}_{\text{ToM}}=\mathbb{E}_{t}\Big[-\sum_{a}\pi_{t}^{j}(a)\,\log\hat{\pi}_{t}^{j}(a)\Big]+\alpha\,\mathrm{KL}\!\left(q\!\left(z_{t}^{\text{team}}\mid h_{t}\right)\,\middle\|\,p\!\left(z_{t}^{\text{team}}\mid h_{t-1},a_{t-1}\right)\right),
(1)
where the KL term stabilizes temporal consistency of the inferred teammate latent. In practice,
π
t
j
\pi_{t}^{j}
can be a one hot target from the observed
a
t
j
a_{t}^{j}
or a smoothed label. The total objective augments standard Dreamer training with
λ
ToM
​
ℒ
ToM
\lambda_{\text{ToM}}\mathcal{L}_{\text{ToM}}
.
Figure 2:
Actor-critic imagination.
The hidden state and teammate latents condition the policy and value heads to produce
(
a
t
0
,
v
t
,
r
t
)
(a_{t}^{0},v_{t},r_{t})
. Imagination samples
z
t
team
z_{t}^{\text{team}}
to simulate partner variability for zero-shot and few-shot coordination.
3.4
Deployment and Adaptation
At test time, the model infers
z
t
team
z_{t}^{\text{team}}
online from observed teammate actions and conditions the actor and critic on this embedding. Imagined rollouts sample plausible teammate trajectories, enabling zero-shot coordination with unseen partners and few-shot improvement as more behavior is observed, without access to the partner’s policy or observations.
