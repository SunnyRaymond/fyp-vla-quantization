# Learning from Reward-Free Offline Data: A Case for Planning with Latent Dynamics Models

paper_id: semanticscholar:b9eb8ba8467d4e8d50c41d4e66f96fa6ec173193
tier: T2
source_used: html_arxiv
warning: none

## Intro

How can we build a system that performs well on unseen combinations of tasks and environments? One promising approach is to avoid relying on online interactions or expert demonstrations, and instead leverage large collections of existing suboptimal trajectories without reward annotations
[
32
,
53
,
16
]
. Broadly, two dominant fields offer promising solutions for learning from such data: reinforcement learning and optimal control.
While online reinforcement learning has enabled agents to master complex tasks—from Atari games
[
44
]
, Go
[
60
]
, to controlling real robots
[
49
]
—it demands massive quantities of environment interactions.
For instance,
OpenAI et al. [49]
used the equivalent of 100 years of real-time hand manipulation experience to train a robot to reliably handle a Rubik’s cube.
To address this inefficiency,
offline RL methods
[
33
,
41
,
18
]
have been developed to learn behaviors from state–action trajectories with corresponding reward annotations. However, these methods typically train agents for a single task, limiting their reuse in other downstream tasks. To overcome this, recent work has explored learning behaviors from offline reward-free trajectories
[
52
,
67
,
32
,
53
]
.
This reward-free paradigm is particularly appealing as it allows agents to learn from suboptimal data and use the learned policy to solve a variety of downstream tasks.
For example, a system trained on low-quality robotic interactions with cloth can later generalize to tasks like folding laundry
[
9
]
.
Optimal control tackles challenge differently: instead of learning a policy function via trial and error,
it plans actions using a known dynamics model
[
7
,
64
,
63
]
to plan out actions. Since real-world dynamics are often hard to specify exactly, many approaches instead learn the model from data
[
71
,
20
,
77
]
. This model-based approach has shown generalization in manipulation tasks involving unseen objects
[
17
]
. Importantly, dynamics models can be trained directly from reward-free offline trajectories, making this a compelling route
[
16
,
56
]
.
Despite significant advances in RL and optimal control, the role of pre-training data quality on reward-free offline learning remains largely unexplored. Prior work has primarily focused on RL methods trained on data from expert or exploratory policies
[
22
,
76
]
, without isolating the specific aspects of data quality that influence performance. In this work, we address this gap by systematically evaluating the strengths and limitations of various approaches for learning from reward-free trajectories. We assess how different learning paradigms perform under offline datasets that vary in both quality and quantity. To ground our study, we focus on navigation tasks — an essential aspect of many real-world robotic systems — where spatial reasoning, generalization, and trajectory stitching play a critical role. While this choice excludes domains such as manipulation, it offers a controlled yet challenging testbed for our comparative analysis.
Our contributions can be summarized as follows:
1.
We propose two new navigation environments with granular control over the data generation process, and generate a total of
23 datasets
of varying quality;
2.
We evaluate methods for learning from offline, reward-free trajectories, drawing from both reinforcement learning and optimal control paradigms. Our analysis systematically assesses their ability to learn from random policy trajectories, stitch together short sequences, train effectively on limited data, and generalize to unseen environment layouts and tasks beyond goal-reaching;
3.
We demonstrate that learning a latent dynamics model and using it for planning is robust to suboptimal data quality and achieves the highest level of generalization to environment variations;
4.
We present a list of guidelines to help practitioners choose between methods depending on available data and generalization requirements.
To facilitate further research into methods for learning from offline trajectories without rewards, we release code, data, environment visualizations, and more at
latent-planning.github.io
.
Figure 1
:
Overview of our analysis.
We test six methods for learning from offline reward-free
trajectories on 23 different datasets across several navigation environments. We evaluate for six generalization properties required to scale to large offline datasets of suboptimal trajectories. We find that planning with a latent dynamics model (PLDM) demonstrates the highest level of generalization. For a full comparison, see
Table
1
.
Right:
diagram of PLDM. Circles represent variables, rectangles – loss components, half-ovals – trained models.

## Method

In this section, we formally introduce the setting of learning from state-action sequences without reward annotations and overview available approaches. We also introduce a
method we call Planning with a Latent Dynamics Model (PLDM).
3.1
Problem Setting
We consider a Markov decision process (MDP)
ℳ
=
(
𝒮
,
𝒜
,
μ
,
p
,
r
)
{\mathcal{M}}=({\mathcal{S}},{\mathcal{A}},\mu,p,r)
, where
𝒮
{\mathcal{S}}
is the state space,
𝒜
{\mathcal{A}}
is the action space,
μ
∈
𝒫
⁡
(
𝒮
)
\mu\in{\mathcal{P}}({\mathcal{S}})
denotes the initial state distribution,
p
∈
𝒮
×
𝒜
→
𝒮
p\in{\mathcal{S}}\times{\mathcal{A}}\rightarrow{\mathcal{S}}
denotes the transition dynamics (we only consider the deterministic case), and
r
∈
𝒮
→
ℝ
r\in{\mathcal{S}}\rightarrow{\mathbb{R}}
denotes the reward function. We work in the offline setting, where we have access to a dataset of state-action sequences
𝒟
{\mathcal{D}}
which consists of transitions
(
s
0
,
a
0
,
s
1
,
…
,
a
T
−
1
,
s
T
)
(s_{0},a_{0},s_{1},\ldots,a_{T-1},s_{T})
. We emphasize again that the offline dataset in our setting does not contain any reward information. The goal is, given
𝒟
{\mathcal{D}}
, to find a policy
π
∈
𝒮
×
𝒵
→
𝒜
\pi\in{\mathcal{S}}\times{\mathcal{Z}}\rightarrow{\mathcal{A}}
, to maximize cumulative reward
r
z
r_{z}
, where
𝒵
{\mathcal{Z}}
is the space of possible task definitions. Our goal is to make the best use of the offline dataset
𝒟
{\mathcal{D}}
to enable the agent to solve a variety of tasks in a given environment with potentially different layouts. During evaluation, unless otherwise specified,
the agent is tasked to reach a goal state
s
g
s_{g}
, so the reward is defined as
r
g
(
s
)
=
𝕀
[
s
=
s
g
]
r_{g}(s)=\mathbb{I}[s=s_{g}]
, and
𝒵
{\mathcal{Z}}
is equivalent to
𝒮
{\mathcal{S}}
.
3.2
Reward-free Offline Reinforcement Learning
Table 1
:
Road-map of our generalization stress-testing experiments.
We test 4 offline goal-conditioned methods - HIQL, GCIQL, CRL, GCBC; a zero-shot RL method HILP, and a learned latent dynamics planning method PLDM.
★★★
denotes good performance in the specified experiment,
★★✩
denotes average performance, and
★✩✩
denotes poor performance. We see that HILP and PLDM are the
best-performing methods, with PLDM standing out as the only method that reaches competitive performance in all settings.
Property
(Experiment section)
HILP
HIQL
GCIQL
CRL
GCBC
PLDM
Transfer to new environment layouts
(
4.8
)
★✩✩
★✩✩
★✩✩
★✩✩
★✩✩
★★★
Transfer to a new task
(
4.6
)
★★✩
★✩✩
★✩✩
★✩✩
★✩✩
★★★
Data efficiency
(
4.3
)
★✩✩
★★✩
★★★
★★✩
★★✩
★★★
Best-case performance
(
4.2
)
★★★
★★★
★★★
★★★
★★✩
★★★
Can learn from random policy trajectories
(
4.5
)
★★★
★✩✩
★★★
★✩✩
★✩✩
★★✩
Can stitch suboptimal trajectories
(
4.4
)
★★★
★✩✩
★★★
★✩✩
★✩✩
★★✩
In this work, we study methods that solve tasks purely from offline trajectories without reward annotations. Reward-free offline RL methods fall into two categories: goal-conditioned RL and zero-shot methods that treat the task as a latent variable. We evaluate state-of-the-art methods from both categories on goal-reaching, and test zero-shot methods on their ability to transfer to new tasks. The methods we investigate are:
•
GCIQL
[
52
]
– goal-conditioned version of Implicit Q-Learning
[
33
]
, a strong and widely-used method for offline RL;
•
HIQL
[
52
]
– a hierarchical GCRL method which trains two policies: one to generate subgoals, and another one to reach the subgoals. Notably, both policies use the same value function;
•
HILP
[
53
]
– a method that learns state representations from the offline data such that the distance in the
learned representation space is proportional to the number of steps between two states. A direction-conditioned policy is then learned to be
able to move along any specified direction in the latent space;
•
CRL
[
19
]
– uses contrastive learning to learn compatibility between states and possible reachable goals. The learned representation, which has been shown to be directly linked to goal-conditioned Q-function, is then used to train a goal-conditioned policy;
•
GCBC
[
43
,
23
]
– Goal-Conditioned Behavior Cloning - the simplest baseline for goal-reaching.
3.3
Planning with a Latent Dynamics Model
The methods in
Section
3.2
are model-free, none explicitly model the environment dynamics. Since we do not assume known dynamics as in classical control, we can instead learn a dynamics model from offline data, similar to
[
46
,
55
]
, which propose a model-based method for goal-reaching using an image reconstruction objective.
We propose a model-based method named Planning with a Latent Dynamics Model (PLDM), which learns latent dynamics using a reconstruction-free SSL objective and the JEPA architecture
[
39
]
. At test time, we plan in the learned latent space to reach goals.
We opt for an SSL approach that predicts the latents as opposed to reconstructing the input observations
[
26
,
21
,
81
,
3
]
motivated by findings that reconstruction yields suboptimal features
[
2
,
42
]
, while reconstruction-free representation learning works well for control
[
59
,
27
]
.
Appendix
G
provides empirical support: features trained with reconstruction-based methods such as DreamerV3
[
26
]
underperform in test-time planning.
Given agent trajectory sequence
(
s
0
,
a
0
,
s
1
,
…
,
a
T
−
1
,
s
T
)
(s_{0},a_{0},s_{1},...,a_{T-1},s_{T})
, we specify the PLDM world model as:
Encoder
:
\displaystyle\mathrm{Encoder:}
z
0
^
=
z
0
=
h
θ
​
(
s
0
)
\displaystyle\quad\hat{z_{0}}=z_{0}=h_{\theta}(s_{0})
(1)
Predictors
:
\displaystyle\mathrm{Predictors:}
z
^
t
k
=
f
θ
k
​
(
z
^
t
−
1
k
,
a
t
−
1
)
,
∀
k
∈
{
1
,
…
,
K
}
\displaystyle\quad\hat{z}_{t}^{k}=f_{\theta}^{k}(\hat{z}_{t-1}^{k},a_{t-1}),\forall k\in\{1,\ldots,K\}
(2)
where
z
^
t
k
\hat{z}_{t}^{k}
is the latent state predicted by predictor
k
k
and
z
t
z_{t}
is the encoder output at step
t
t
. When
K
>
1
K>1
, we train an ensemble of predictors for uncertainty regularization at test-time. The training objective involves minimizing the distance between predicted and encoded latents summed over all timesteps. Given target and predicted latents
Z
,
Z
^
k
∈
ℝ
H
×
N
×
D
Z,\hat{Z}^{k}\in\mathbb{R}^{H\times N\times D}
, where
H
≤
T
H\leq T
is the model prediction horizon,
N
N
is the batch dimension, and
D
D
the feature dimension, the similarity objective between predictions and encodings is:
ℒ
sim
=
∑
k
=
1
K
∑
t
=
0
H
1
N
​
∑
b
=
0
N
‖
Z
^
t
,
b
k
−
Z
t
,
b
‖
2
2
\displaystyle\mathcal{L}_{\mathrm{sim}}=\sum_{k=1}^{K}\sum_{t=0}^{H}\frac{1}{N}\sum_{b=0}^{N}\|\hat{Z}^{k}_{t,b}-Z_{t,b}\|^{2}_{2}
(3)
To prevent representation collapse, we use a VICReg-inspired
[
4
]
objective, and inverse dynamics modeling
[
40
]
. We show a diagram of PLDM in
Figure
1
. See
Section
D.1.1
for details.
Goal-conditioned planning with PLDM.
In this work, we mainly focus on the task of reaching specified goal states. While methods outlined in
Section
3.2
rely on trained policies to reach the goal, PLDM relies on planning.
At test time, given the current observation
s
0
s_{0}
, goal observation
s
g
s_{g}
, pretrained encoder
h
θ
h_{\theta}
predictor
f
θ
f_{\theta}
, and planning horizon
H
H
, our planning objective is:
∀
k
∈
{
1
,
…
,
K
}
:
z
^
0
k
=
z
0
k
=
h
θ
​
(
s
0
)
,
z
^
t
k
=
f
θ
k
​
(
z
^
t
−
1
k
,
a
t
−
1
)
\displaystyle\forall k\in\{1,\ldots,K\}:\ \hat{z}_{0}^{k}=z_{0}^{k}=h_{\theta}(s_{0}),\ \hat{z}_{t}^{k}=f_{\theta}^{k}(\hat{z}_{t-1}^{k},a_{t-1})
(4)
C
goal
​
(
𝐚
,
s
0
,
s
g
)
=
1
K
​
∑
k
=
1
K
∑
t
=
0
H
‖
h
θ
​
(
s
g
)
−
f
θ
k
​
(
z
^
t
k
,
a
t
)
‖
\displaystyle C_{\text{goal}}(\mathbf{a},s_{0},s_{g})=\frac{1}{K}\sum_{k=1}^{K}\sum_{t=0}^{H}\|h_{\theta}(s_{g})-f_{\theta}^{k}(\hat{z}_{t}^{k},a_{t})\|
(5)
C
uncertainty
​
(
𝐚
,
s
0
,
s
g
)
=
∑
t
=
0
H
γ
t
​
∑
j
=
1
d
Var
⁡
(
{
f
θ
k
​
(
s
t
k
,
a
t
)
j
}
k
=
1
K
)
\displaystyle C_{\text{uncertainty}}(\mathbf{a},s_{0},s_{g})=\sum_{t=0}^{H}\gamma^{t}\sum_{j=1}^{d}\mathrm{Var}(\{f_{\theta_{k}}(s_{t}^{k},a_{t})_{j}\}_{k=1}^{K})
(6)
𝐚
∗
=
arg
⁡
min
𝐚
​
{
C
goal
​
(
𝐚
,
s
0
,
s
g
)
+
β
​
C
uncertainty
​
(
𝐚
,
s
0
,
s
g
)
}
\displaystyle\mathbf{a}^{*}=\arg\min_{\mathbf{a}}\{C_{\text{goal}}(\mathbf{a},s_{0},s_{g})+\beta C_{\text{uncertainty}}(\mathbf{a},s_{0},s_{g})\}
(7)
C
goal
C_{\text{goal}}
is the goal-reaching objective and
C
uncertainty
C_{\text{uncertainty}}
penalizes the model from choosing state-action transitions that deviate from the training distribution, with
γ
∈
[
0
,
1
]
\gamma\in[0,1]
as the temporal discount. This regularization resembles how GCIQL, HIQL, and HILP use expectile regression to learn policies that remain
in-distribution
with respect to the dataset
[
34
]
. See Appendix
E
for ablations on
C
uncertainty
C_{\text{uncertainty}}
.
Following the Model Predictive Control framework
[
45
]
, PLDM re-plans every
i
i
interactions with the environment. By default, we use
i
=
1
i=1
for all experiments, making PLDM
∼
4
\sim 4
x slower than the model-free baselines. The replanning interval
i
i
can be increased to accelerate MPC with only a minor loss in performance (see
Appendix
F
).
We use MPPI
[
72
]
in all our experiments with planning. We note that PLDM is not using rewards, neither explicitly nor implicitly, and should be considered as an optimal control method. We also note that to apply PLDM to a new task, we do not need to retrain the encoder
h
θ
h_{\theta}
and dynamics
f
θ
f_{\theta}
, we only need to change the cost in
Equation
5
. We test this flexibility in
Section
4.6
, where we invert the sign of the cost to make the agent avoid a given state.
Figure 3
:
Left
: The Two-Rooms environment. The agent starts at a random location and is tasked with reaching the goal at another randomly sampled location in the other room using 200 steps or less. Observations are
64
×
64
64\times 64
pixels images.
Right:
Examples of trajectories in the offline data.
Red
: each step’s direction is sampled from Von Mises distribution.
Blue
: each step’s direction is sampled uniformly.
Method
Good-quality data
No door-passing trajectories
CRL
1
89.3   ±
1
0.7
1
14.7   ±
1
4.1
GCBC
1
86.0   ±
1
2.0
11
8.4   ±
1
1.2
GCIQL
1
98.0   ±
1
0.9
1
99.6   ±
1
0.4
HILP
100.0   ±
1
0.0
100.0   ±
1
0.0
HIQL
1
96.4   ±
1
1.3
1
26.3   ±
1
5.6
PLDM
1
97.8   ±
1
0.7
1
34.4   ±
1
2.7
Table 2
:
Performance of tested methods on good-quality data and on data with no trajectories passing through the door. Values are average success rates
(
±
standard error
)
(\pm\text{standard error})
across 3 seeds.
