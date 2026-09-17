# World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry

paper_id: arxiv:2604.01985v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

World models—action-conditioned forward dynamics models that predict future states given specific actions or action chunks—have come to play an increasingly important role in robot learning
[
37
,
120
,
4
,
109
,
137
,
52
]
.
Recent works have shown that, when trained on action-labeled robot interactions alongside action-free internet videos
[
97
,
51
,
126
]
, world models have the potential to not only generate controllable future dynamics but also enable scalable policy evaluation
[
93
,
108
,
140
,
67
]
, policy optimization
[
40
,
123
,
124
,
34
]
, and test-time planning
[
39
,
41
,
53
,
138
,
92
]
.
Despite remarkable progress, building a general-purpose world model that is robust enough for various downstream applications remains difficult. A central challenge is
action following
: predicting future states that faithfully reflect the effects of the given actions
[
101
]
.
Unlike policy learning, which primarily focuses on modeling optimal actions, a world model must be reliable across a much broader action distribution, including suboptimal, exploratory, and even random actions encountered during policy learning or evaluation
[
63
,
135
,
53
]
.
However, collecting robot interactions covering diverse actions is often slow, expensive, and sometimes even unsafe.
Given a limited budget of robot data, deciding which specific interactions to collect remains a pressing challenge.
Previous work has sought to address this through two main approaches. One line of work relies on on-policy exploration,
i.e.
, gathering data by rolling out the policies of interest
[
53
,
35
,
75
]
. While effective for the considered policies, the learned model often degrades sharply beyond predefined policy sets, compromising its generality.
Another line of work focuses on info-max exploration, actively seeking interactions that maximize information gain
[
90
,
100
,
57
]
.
A common proxy for information gain is the prediction error of the world model, estimated before collecting the corresponding transition–a process we refer to as
world model verification
. This verification process, however, often suffers from a practical challenge: existing methods tend to be reliable in well-explored regions where additional data are largely redundant, but unreliable in under-explored regions where verification is critically needed. After all, the interactions that are most informative for exploration are precisely those where the least prior information exists for verification.
This tension raises a central question:
How can we reliably verify the predictions of a world model in under-explored regimes?
To this end, we propose World Action Verifier (WAV), a framework that enables world models to verify their own predictions and self-improve through an asymmetric forward-inverse cycle. The core idea is to decompose action-conditioned forward predictions into two complementary factors:
state plausibility
,
i.e.
, whether a predicted state is visually realistic, and
action reachability
,
i.e.
, whether the predicted transition is physically achievable under the given actions.
This decomposition not only allows each factor to be verified separately, but also admits two crucial asymmetries: (i) the broader availability of action-free data: state plausibility can be verified using internet videos without action labels, which are far more abundant than the action-labeled robot interactions used to train the world model, and (ii) lower dimensionality of action-relevant features: action reachability can be verified based on a compact subset of state features relevant to the actions, which are much lower-dimensional than the full state the world model must predict.
Motivated by these asymmetries, we augment a world model with two additional components: a diverse subgoal generator obtained from video corpora, and a sparse inverse model trained to infer actions from a learned subset of state features.
Together, these components induce a goal-oriented self-improvement cycle: the subgoal generator proposes plausible future states, the inverse model infers actions that could reach them, and the forward world model rolls out those actions to test whether the predicted states are consistent with the proposed subgoals (
Figure
1
).
Theoretically, we show that verification via a sparse inverse process is easier than dense forward generation, particularly in high-dimensional stochastic environments. Empirically, we evaluate WAV on nine tasks spanning MiniGrid
[
19
]
, RoboMimic
[
141
]
, and ManiSkill
[
85
]
.
Compared to existing methods, WAV improves the sample efficiency of world models by
2
×
2\times
and boosts downstream policy performance by more than 22%. Our results suggest that the asymmetries between forward and inverse dynamics offer a promising ingredient for building self-improving world models.
Figure 1:
Overview of World Action Verifier, a framework that enables action-conditioned world models to verify their predictions and self-improve from an asymmetric forward-inverse cycle: (i) a
diverse
subgoal generator proposes plausible future states, (ii) a
sparse
inverse model infers actions from a relevant subset of state features, and (iii) a world model rolls forward and verifies consistency between its predicted state and the proposed state.

## Method

World models excel when grounded in action-labeled interaction data, yet collecting such data at scale is often prohibitively expensive.
In this section, we present World Action Verifier (WAV), a self-improving framework that enables a world model to verify its own predictions and prioritize informative exploration.
We first formalize the verification problem in a semi-supervised setting (
Sec.
2.1
), then decompose it into two more tractable subproblems (
Sec.
2.2
), and finally couple them into a goal-oriented exploration procedure for self-improvement (
Sec.
2.3
).
2.1
Preliminary: Semi-Supervised Verification of World Models
We consider a world model
f
θ
f_{\theta}
as an action-conditioned forward dynamics model,
s
^
t
+
1
=
f
θ
​
(
s
t
,
a
t
)
,
\hat{s}^{t+1}=f_{\theta}(s^{t},a^{t}),
where
s
t
s^{t}
and
a
t
a^{t}
are the state and action, or action chunk, at time
t
t
, and
s
^
t
+
1
\hat{s}^{t+1}
is the predicted successor state.
Following recent training recipes
[
51
,
31
]
, we study a semi-supervised setting with two data sources: a small action-labeled robot interaction dataset
𝒟
act
=
{
(
s
t
,
a
t
,
s
t
+
1
)
}
\mathcal{D}_{\mathrm{act}}=\{(s^{t},a^{t},s^{t+1})\}
and a large action-free video dataset
𝒟
vid
=
{
(
s
t
,
s
t
+
1
,
…
)
}
.
\mathcal{D}_{\mathrm{vid}}=\{(s^{t},s^{t+1},\ldots)\}.
Typically,
𝒟
vid
\mathcal{D}_{\mathrm{vid}}
spans a much broader range of state transitions than
𝒟
act
\mathcal{D}_{\mathrm{act}}
.
Our goal is to improve
f
θ
f_{\theta}
not only on the narrow action distribution represented in
𝒟
act
\mathcal{D}_{\mathrm{act}}
, but also on the broader transition support reflected in
𝒟
vid
\mathcal{D}_{\mathrm{vid}}
.
However, the lack of action labels in online videos poses a key challenge for
action following
: rather than faithfully grounding predictions in the conditioning action, existing world models often hallucinate future states that may look visually plausible but are physically misaligned with the given action
[
101
,
82
]
.
A natural remedy is to collect additional action-labeled robot interactions
[
53
,
35
,
75
]
.
Yet, since large-scale robot interaction data are costly to collect, a critical question arises:
which specific interactions should be prioritized to improve the world model most effectively?
Intuitively, transitions that the model can already predict accurately yield little new knowledge.
Instead, the data budget should be steered toward transitions that are likely to induce large prediction errors.
More formally, for a transition
(
s
t
,
a
t
,
s
t
+
1
)
(s^{t},a^{t},s^{t+1})
, we define the true
prediction error as
ε
⁡
(
s
t
,
a
t
,
s
t
+
1
)
:=
ℓ
⁡
(
f
θ
​
(
s
t
,
a
t
)
,
s
t
+
1
)
,
\varepsilon(s^{t},a^{t};s^{t+1}):=\ell\!\left(f_{\theta}(s^{t},a^{t}),s^{t+1}\right),
(1)
where
ℓ
⁡
(
⋅
,
⋅
)
\ell(\cdot,\cdot)
is a discrepancy measure in the state space.
Since the true successor state
s
t
+
1
s^{t+1}
cannot be observed prior to execution, we aim to construct a
verifier
ε
^
​
(
s
t
,
a
t
,
s
^
t
+
1
)
\hat{\varepsilon}(s^{t},a^{t},\hat{s}^{t+1})
that estimates this error.
From an exploration standpoint, the verifier need not be perfectly calibrated, but it should preserve the relative ranking of prediction errors across candidate actions
[
46
,
98
]
.
That is, given two candidate actions
a
i
t
a_{i}^{t}
and
a
j
t
a_{j}^{t}
, with predicted successor states
s
^
i
t
+
1
=
f
θ
​
(
s
t
,
a
i
t
)
\hat{s}_{i}^{t+1}=f_{\theta}(s^{t},a_{i}^{t})
and
s
^
j
t
+
1
=
f
θ
​
(
s
t
,
a
j
t
)
\hat{s}_{j}^{t+1}=f_{\theta}(s^{t},a_{j}^{t})
, an effective verifier for exploration should satisfy
ε
⁡
(
s
t
,
a
i
t
,
s
i
t
+
1
)
<
ε
⁡
(
s
t
,
a
j
t
,
s
j
t
+
1
)
⟹
ε
^
​
(
s
t
,
a
i
t
,
s
^
i
t
+
1
)
<
ε
^
​
(
s
t
,
a
j
t
,
s
^
j
t
+
1
)
.
\varepsilon(s^{t},a_{i}^{t};s_{i}^{t+1})<\varepsilon(s^{t},a_{j}^{t};s_{j}^{t+1})\;\Longrightarrow\;\hat{\varepsilon}(s^{t},a_{i}^{t},\hat{s}_{i}^{t+1})<\hat{\varepsilon}(s^{t},a_{j}^{t},\hat{s}_{j}^{t+1}).
(2)
2.2
Two Complementary Factors of Verification
A common approach to verifying forward predictions in
Equation
2
is to leverage the internal knowledge of the world model itself,
e.g.
, extracting epistemic uncertainty from a single model
[
90
]
or measuring disagreement across multiple models
[
100
,
57
]
.
However, such verification methods often inherit the blind spots of the learned world model: they provide relatively reliable error estimates in well-explored regimes where the current world model is already accurate, but become much less reliable in under-explored regimes where accurate verification is most critical.
To overcome this issue, we take a different perspective: rather than directly verifying the overall correctness of a forward prediction, we decompose it into sub-conditions that are easier to verify.
More specifically, motivated by the Bayes decomposition,
p
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
=
p
⁡
(
a
t
∣
s
t
,
s
t
+
1
)
​
p
​
(
s
t
+
1
∣
s
t
)
p
⁡
(
a
t
∣
s
t
)
∝
p
⁡
(
s
t
+
1
∣
s
t
)
⏟
state
​
p
⁡
(
a
t
∣
s
t
,
s
t
+
1
)
⏟
action
,
p(s^{t+1}\mid s^{t},a^{t})=\frac{p(a^{t}\mid s^{t},s^{t+1})\,p(s^{t+1}\mid s^{t})}{p(a^{t}\mid s^{t})}\;\propto\;\underbrace{p(s^{t+1}\mid s^{t})}_{\text{state}}\,\underbrace{p(a^{t}\mid s^{t},s^{t+1})}_{\text{action}},
(3)
we view a correct action-conditioned forward prediction as satisfying two complementary criteria:
•
State Plausibility
: whether the predicted next state is plausible under the environment dynamics.
•
Action Reachability
: whether the transition from
s
t
s^{t}
to
s
t
+
1
s^{t+1}
is consistent with the given action.
From this view, a correct prediction should both lie on the manifold of plausible futures and be reachable under the specified action.
Crucially, each condition admits a verification strategy that is more tractable than predicting high-dimensional forward dynamics, which we will describe next.
State verification via distribution asymmetry.
One common failure mode of high-dimensional forward prediction is poor visual plausibility.
For example, a world model post-trained on limited robot interaction data may partially forget the general dynamics learned from video pretraining, resulting in blurry or inconsistent rollouts
[
133
]
.
To detect such errors, we build a state verifier from the first factor in
Equation
3
, namely a state transition prior
g
ϕ
​
(
s
t
+
1
∣
s
t
)
g_{\phi}(s^{t+1}\mid s^{t})
.
Since this prior does not condition on actions, it can be trained not only on action-labeled robot interactions
𝒟
act
\mathcal{D}_{\mathrm{act}}
, but also on the much larger action-free video dataset
𝒟
vid
\mathcal{D}_{\mathrm{vid}}
.
Moreover, as the prediction error of a world model often compounds over longer horizons, we instantiate
s
t
+
1
s^{t+1}
as a future subgoal reached after an action chunk rather than as the immediate next frame.
After training,
g
ϕ
g_{\phi}
allows us to sample a diverse set of
K
K
plausible future subgoals for state verification,
{
s
~
k
t
+
1
}
k
=
1
K
∼
g
ϕ
(
⋅
∣
s
t
)
.
\{\tilde{s}^{t+1}_{k}\}_{k=1}^{K}\sim g_{\phi}(\cdot\mid s^{t}).
(4)
Action verification via dimensionality asymmetry.
State plausibility alone does not imply correct forward prediction: for downstream policy use, predicted transitions must also be reachable under the specified action.
To assess this condition, we build another verifier from the second factor in
Equation
3
, namely an inverse dynamics model
h
ψ
​
(
a
t
∣
s
t
,
s
t
+
1
)
h_{\psi}(a^{t}\mid s^{t},s^{t+1})
that infers which action could connect a current state to a future state.
While the inverse model
h
ψ
h_{\psi}
is action-dependent and cannot leverage more training data than the forward world model, it benefits from a lower effective dimensionality in both its outputs and its relevant inputs.
For instance, in visually complex scenes, an action typically affects only one object at a time rather than the entire scene.
Inspired by the observation that models attending to a compact set of causally relevant features often generalize better
[
76
,
117
]
, we explicitly impose a learnable sparsity mask
M
M
in the inverse dynamics model:
a
^
t
=
h
ψ
​
(
M
⊙
s
t
,
M
⊙
s
t
+
1
)
.
\hat{a}^{t}=h_{\psi}\!\left(M\odot s^{t},\;M\odot s^{t+1}\right).
(5)
As illustrated in
Figure
2
, the subgoal generator
g
ϕ
g_{\phi}
and inverse model
h
ψ
h_{\psi}
provide two complementary components for verification: the former checks whether a candidate future is plausible, while the latter checks whether it is reachable through an inferred action.
Figure 2
:
Decomposing model verification into state plausibility and action reachability.
⬇
#
f:
world
model,
h:
inverse
model
#
s:
current
state,
g:
subgoal
generator
#
D:
current
data,
K:
number
of
candidates
for
each
exploration
iteration
:
s_g
=
v
.
sample
(
s
,
K
)
#
subgoals
a
=
h
.
inverse
(
s
,
s_g
)
#
actions
s_p
=
f
.
predict
(
s
,
a
)
#
outcomes
scores
=
dist
(
s_g
,
s_p
)
#
disagreement
idx
=
argmax
(
scores
)
#
max
surprise
s_n
=
env
.
step
(
a
[
idx
])
D
.
append
((
s
,
a
[
idx
],
s_n
))
f
.
update
(
D
),
h
.
update
(
D
)
Algorithm 1
WAV-Guided Exploration.
2.3
Goal-Oriented Exploration
Given the two verification criteria above, we next couple them into a verification-driven exploration algorithm.
A straightforward design is action-oriented exploration: sample candidate actions, roll out the forward world model
f
θ
f_{\theta}
, and then use the inverse model
h
ψ
h_{\psi}
to check whether the generated state transitions recover the original actions
[
127
]
.
However, this ordering can be brittle in practice.
Among the three components, the forward world model
f
θ
f_{\theta}
is often the least reliable when action-labeled data are scarce.
As such, errors introduced by the initial forward rollout can produce off-manifold states, on which the subsequent inverse model also becomes unreliable.
We therefore pursue a goal-oriented alternative that places the forward world model as the final step in the verification cycle.
At each time step
t
t
, we first sample plausible subgoals from the transition prior, infer actions that could reach those subgoals, and only then verify whether the action-conditioned world model can realize them:
s
t
→
g
ϕ
s
~
1
:
K
t
+
1
→
h
ψ
a
^
1
:
K
t
→
f
θ
s
^
1
:
K
t
+
1
→
ℓ
ε
^
1
:
K
→
max
a
⋆
.
s^{t}\xrightarrow{\mathmakebox[2.0em][c]{\,g_{\phi}\,}}\tilde{s}_{1:K}^{t+1}\xrightarrow{\mathmakebox[2.0em][c]{\,h_{\psi}\,}}\hat{a}_{1:K}^{t}\xrightarrow{\mathmakebox[2.0em][c]{\,f_{\theta}\,}}\hat{s}_{1:K}^{t+1}\xrightarrow{\mathmakebox[2.0em][c]{\,\ell\,}}\hat{\varepsilon}_{1:K}\xrightarrow{\mathmakebox[2.0em][c]{\,\max\,}}a^{\star}.
(6)
For each candidate subgoal
s
~
k
t
+
1
\tilde{s}_{k}^{t+1}
, we measure how far the forward rollout
s
^
k
t
+
1
\hat{s}_{k}^{t+1}
deviates from it.
In practice, this discrepancy
ℓ
\ell
can be computed in the discrete state space, a continuous representation space, or a diffusion noise space, depending on the parameterization of the world model.
As summarized in
Algorithm
1
, we execute
a
⋆
a^{\star}
associated with the largest discrepancy at each time step, add the resulting transition to
𝒟
act
\mathcal{D}_{\mathrm{act}}
, and iteratively update both the forward world model and the inverse dynamics model.
By verifying multiple candidate rollouts in parallel before acting, our method reduces unnecessary real-world interaction and thereby effectively trades scalable computation for improved data efficiency.
