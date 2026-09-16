# In-Context Planning with Latent Temporal Abstractions

paper_id: semanticscholar:de1ce2f87e18481920b707937eea4cae6c190cfe
tier: T2
source_used: html_arxiv
warning: none

## Intro

†
†
footnotetext:
Source code:
https://github.com/BaitingLuo/I-TAP
Planning-based reinforcement learning (RL) has delivered strong results in discrete decision-making domains (e.g., board games and video games)
(
Silver et al., 2017
;
Schrittwieser et al., 2020
)
and has shown increasing promise in continuous control
(
Hubert et al., 2021
)
. However, planning-based offline RL for continuous control faces two practical challenges. First, many real environments are effectively partially observable due to latent parameters (e.g., unobserved disturbances or payload changes), which breaks the stationary, full observability assumptions commonly adopted by learned dynamics models used for planning. When these latent factors are not properly handled during planning, they manifest as apparent stochasticity from the planner’s perspective
(
Antonoglou et al., 2022
)
, exacerbating planning complexity. Second, planning over primitive continuous actions induces large branching factors and long effective horizons, making search expensive especially under uncertainty.
To address adaptation under partial observability, recent work has reframed RL as sequence modeling, leveraging Transformers
(
Vaswani et al., 2017
)
trained on trajectories to produce policies conditioned on a finite context window
(
Chen et al., 2021
;
Brown et al., 2020
;
Furuta et al., 2022
;
Liu and Abbeel, 2023
;
Laskin et al., 2023
;
Huang et al., 2024
)
. Conditioning on recent interaction history enables in-context adaptation without test-time gradient updates, and can implicitly capture latent task variables when they are identifiable from history. Yet these sequence-model policies are typically deployed as direct action predictors, which introduces two limitations: (i) Without an explicit decision-time optimizer, they often inherit the constraints and suboptimalities of the offline dataset
(
Son et al., 2025
)
. (ii) In stochastic environments, converting a predictive model into an optimized decision rule is nontrivial when the model is used only as a conditional policy
(
Paster et al., 2022
)
. These issues motivate combining in-context models with explicit planning.
Nevertheless, planning directly in raw continuous action space can be inefficient and inflexible
(
Jiang et al., 2023
)
. Recent planning-based methods reduce complexity by learning temporal abstractions (e.g., options
(
Sutton et al., 1999
)
or macro-actions
(
Mcgovern and Sutton, 1998
)
) and planning over high-level decisions
(
Jiang et al., 2023
;
Luo et al., 2025
)
, which shortens effective horizons and reduces branching. However, these planners are developed under the assumption of a stationary, fully observed Markov decision process (MDP) and learn dynamics models that condition only on the current state. As a result, state aliasing would create apparent stochasticity and increase the search burden.
Motivated by these challenges, we introduce an in-context planning framework that adapts to different latent parameters in a learned discrete latent temporal abstraction space for continuous control under stochastic dynamics. Our premise is that integrating temporal abstraction with in-context adaptation for planning-based RL addresses these issues jointly. Adapting from recent history and planning conditioned on it in a latent temporal abstraction space allows an agent to: (i) decouple adaptation and planning from the native temporal granularity of MDP, thereby shortening the required context, easing the learning of a reliable sequence model prior by modeling a simpler distribution over discrete latent temporal abstractions instead of high-dimensional continuous actions, and reducing the branching factor during planning; (ii) use context to infer global latent parameters that govern the environment’s dynamics (e.g., unobserved perturbation forces), enabling effective adaptation and forecasting across scenario shifts; and (iii) employ an online planner such as Monte Carlo Tree Search (MCTS) for optimization, providing a mechanism to deviate from suboptimal behavior policies in the offline data and handle uncertainty.
Figure 1:
Overview of I-TAP.
Left:
A residual-quantized VAE (RQ-VAE) discretizes continuous observation–action trajectories into a coarse-to-fine token stack.
Right:
Normalized return and per-decision latency as functions of planning horizon and context size on Stochastic MuJoCo, highlighting the importance of a properly sized context window for effective in-context planning under environmental stochasticity and partial observability.
To this end, we propose
In-Context Latent Temporal Abstraction Planner
(I-TAP), which learns a discrete latent temporal-abstraction space and a Transformer-based in-context sequence prior over discrete latent codes from offline data, and performs decision-time planning with these models to enable adaptive decision-making. To improve flexibility and scalability in handling high-dimensional continuous observation-action spaces, we adopt an observation-conditioned residual-quantized VAE (RQ-VAE) to learn this discrete latent temporal-abstraction space, as illustrated in Fig.
1
. RQ-VAE
(
Lee et al., 2022
)
encodes each macro-step into a depth-
D
D
coarse-to-fine stack of code indices drawn from a codebook, providing a compositional discretization with substantially higher effective capacity than a single code. This retains high-fidelity decoding while enabling compact discrete representations, yielding a scalable interface for in-context sequence modeling over abstractions and planning in latent space. MCTS then operates over these latent tokens using context
-
guided priors to balance exploration and exploitation under uncertainty; finally, we decode the selected latent stack to a primitive action sequence and execute the first action.
Our experiments demonstrate that I-TAP can be trained as a single offline model across behavior policies of varying quality and multiple latent parameters, and evaluated from deterministic to highly stochastic environments. Across these settings, I-TAP matches or outperforms strong offline RL and planning-based baselines, while exhibiting in-context adaptation under partial observability and scaling to high-dimensional continuous control through latent temporal abstractions. In summary, we make the following contributions:
(i) We propose I-TAP, an offline RL method that unifies in-context adaptation and online planning for stochastic, partially observable continuous control by planning in a learned latent temporal-abstraction space;
(ii) To improve scalability and flexibility, we learn a residual-quantized temporal abstraction and a context-conditioned latent dynamics model from offline trajectories, via an observation-conditioned RQ-VAE and an autoregressive temporal Transformer;
(iii) We instantiate an in-context planner that performs Monte Carlo Tree Search directly over latent tokens, reducing the effective branching factor and decision horizon, and enabling improvements beyond suboptimal behavior policies.

## Method

POMDP.
We consider a partially observable Markov decision process (POMDP)
M
=
(
𝒮
,
𝒜
,
𝒪
,
P
,
R
,
γ
)
M=(\mathcal{S},\mathcal{A},\mathcal{O},P,R,\gamma)
, where
P
⁡
(
s
t
+
1
∣
s
t
,
a
t
)
P(s_{t+1}\mid s_{t},a_{t})
is the transition kernel,
R
⁡
(
r
t
∣
s
t
,
a
t
)
R(r_{t}\mid s_{t},a_{t})
is the reward distribution,
and
γ
∈
[
0
,
1
)
\gamma\in[0,1)
is the discount factor.
At each time step
t
t
, the agent receives an observation
o
t
∈
𝒪
o_{t}\in\mathcal{O}
, selects an action
a
t
∈
𝒜
a_{t}\in\mathcal{A}
,
the environment transitions according to
P
⁡
(
s
t
+
1
∣
s
t
,
a
t
)
P(s_{t+1}\mid s_{t},a_{t})
, and the agent receives a reward
r
t
∼
R
(
⋅
∣
s
t
,
a
t
)
r_{t}\sim R(\cdot\mid s_{t},a_{t})
. For a fixed context length
c
c
, we define the agent’s context prior to observing
o
t
o_{t}
as
c
t
=
(
o
t
−
c
,
a
t
−
c
,
r
t
−
c
,
…
,
o
t
−
1
,
a
t
−
1
,
r
t
−
1
)
∈
𝒞
c_{t}=(o_{t-c},a_{t-c},r_{t-c},\ldots,o_{t-1},a_{t-1},r_{t-1})\in\mathcal{C}
. The goal is to maximize the expected discounted return
J
=
𝔼
π
,
M
​
[
∑
t
=
0
T
−
1
γ
t
​
r
t
]
J=\mathbb{E}_{\pi,M}\!\left[\sum_{t=0}^{T-1}\gamma^{t}r_{t}\right]
over horizon
T
T
.
Meta-RL.
We define a context-conditioned policy
π
:
𝒞
×
𝒪
→
Δ
⁡
(
𝒜
)
\pi:\mathcal{C}\times\mathcal{O}\rightarrow\Delta(\mathcal{A})
,
where
𝒞
\mathcal{C}
is a fixed-length context space and
Δ
⁡
(
𝒜
)
\Delta(\mathcal{A})
denotes distributions over actions.
At time
t
t
, the algorithm forms a bounded context
c
t
∈
𝒞
c_{t}\in\mathcal{C}
from the interaction history
h
t
h_{t}
(e.g., a sliding window of recent transitions) and samples actions as
a
t
∼
f
⁡
(
c
t
,
o
t
)
a_{t}\sim f(c_{t},o_{t})
. Meta-RL aims to learn an algorithm
π
\pi
that maximizes
J
J
in expectation over a distribution of tasks (POMDPs)
p
⁡
(
M
)
p(M)
:
max
π
⁡
𝔼
M
∼
p
⁡
(
M
)
​
[
𝔼
π
,
M
​
[
∑
t
=
0
H
−
1
γ
t
​
r
t
]
]
.
\max_{\pi}\ \mathbb{E}_{M\sim p(M)}\left[\mathbb{E}_{\pi,M}\!\left[\sum_{t=0}^{H-1}\gamma^{t}r_{t}\right]\right].
In our setting, each
M
M
is induced by an unobserved latent
parameter. In offline meta-RL, we assume access to a dataset of trajectories collected by some behavior algorithms on meta-training tasks, from which the context
c
t
c_{t}
is constructed.
