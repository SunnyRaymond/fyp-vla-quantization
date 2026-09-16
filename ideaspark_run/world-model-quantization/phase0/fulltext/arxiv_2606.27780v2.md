# Understanding Rollout Error in Graph World Models

paper_id: arxiv:2606.27780v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

A world model
Ha and Schmidhuber (2018)
represents environment observations as states and predicts future states conditioned on actions. Modern world models trained on large-scale data
Hafner et al. (2018)
;
Hafner et al. (2020)
;
Hafner et al. (2024)
;
Chua et al. (2018)
;
Schrittwieser et al. (2020)
;
Wu et al. (2024)
;
Bruce et al. (2024)
have shown strong planning, prediction, and generation capabilities. Most of them, however, are designed for vector- or image-like states. They do not directly capture environments whose state is a graph. Such environments are increasingly common in agent planning: multi-agent calling trees coordinate distributed execution
Anokhin et al. (2024)
, tool-use benchmarks require planning over external APIs
Li et al. (2023)
;
Qin et al. (2024)
;
Liu et al. (2024b)
;
Trivedi et al. (2024)
, skill graphs encode reusable capabilities
Huang and Huang (2025)
;
Huang et al. (2025)
;
Wang et al. (2025a)
;
Xia et al. (2026)
, and communication networks route actions among interacting entities. Planning in these settings requires reasoning about node states and about how effects travel through relations. The same autoregressive rollout that enables long-horizon planning also creates a risk: prediction errors can compound and degrade downstream decisions
Asadi et al. (2018)
;
Janner et al. (2019)
;
Kakade and Langford (2002)
.
Graph World Models (GWMs) address this setting by autoregressively predicting graph states
G
t
=
(
V
,
E
t
,
X
t
,
A
t
)
G_{t}=(V,E_{t},X_{t},A_{t})
Feng et al. (2025)
. Given a learned transition model
ℱ
θ
​
(
G
^
k
,
a
k
)
\mathcal{F}_{\theta}(\hat{G}_{k},a_{k})
, a GWM rolls out
G
^
1
,
…
,
G
^
H
\hat{G}_{1},\ldots,\hat{G}_{H}
and scores candidate action sequences. Existing work has mostly studied graph foundation models for static graph tasks
Liu et al. (2024a)
;
Chen et al. (2024)
, graph generation or dynamic graph representation learning
You et al. (2018)
;
Vignac et al. (2023)
;
Rossi et al. (2020)
;
You et al. (2022)
, and domain-specific graph planning
Zhang et al. (2021)
;
Ammanabrolu and Riedl (2021)
. Feng et al.
Feng et al. (2025)
introduced a unified GWM with action nodes, but their core setting is fixed-edge (FE) rollout. It remains unclear how GWMs behave under
long-horizon rollout error
and in
dynamic-edge (DE) regimes
, where the structure rolled out by the model becomes part of its future input.
Graph structure makes compounding error harder to analyze than in vector-state world models. In vector settings, rollout error is often summarized by a scalar Lipschitz constant. In graph settings, the same local error can behave very differently depending on topology. An error on a chain may stay localized; an error on a hub, dense graph, or high-spectral-radius topology can spread widely, consistent with spectral analyses of graph propagation, graph-network dynamics, GNN stability, and rewiring studies of harmful information bottlenecks
Battaglia et al. (2016)
;
Battaglia et al. (2018)
;
Sanchez-Gonzalez et al. (2018)
;
Oono and Suzuki (2020)
;
Chamberlain et al. (2021)
;
Tori et al. (2024)
;
Yu et al. (2025)
. The problem is harder when edges are predicted. A node-state error can corrupt future edge prediction, and an edge error can change later message passing. GWM rollout error is therefore not merely scalar prediction error; it is a topology-dependent, and sometimes coupled, node-edge dynamical process.
Existing rollout-error theory is not designed for graph-structured planning. Standard model-based RL analyses bound compounding error with a scalar Lipschitz constant over a fixed-dimensional state vector
Asadi et al. (2018)
;
Janner et al. (2019)
. This misses three aspects of graph rollouts. First, topology is explicit: for a message-passing GWM, node-error growth separates into a topology term
ρ
⁡
(
A
)
\rho(A)
and a model term
∏
ℓ
‖
W
ℓ
‖
2
\prod_{\ell}\|W_{\ell}\|_{2}
, where
A
A
is the adjacency matrix,
ρ
⁡
(
A
)
\rho(A)
is its spectral radius, and
W
ℓ
W_{\ell}
are layer weights. Second, dynamic graphs require coupled error dynamics: node-feature error
e
k
X
=
‖
X
^
k
−
X
k
‖
F
e^{X}_{k}=\|\hat{X}_{k}-X_{k}\|_{F}
and edge error
e
k
A
=
‖
A
^
k
−
A
k
‖
F
e^{A}_{k}=\|\hat{A}_{k}-A_{k}\|_{F}
evolve together rather than through a scalar recursion. Third, rollout error must be tied to planning regret over horizon
H
H
. These gaps motivate our graph-valued analysis, which replaces the scalar constant with
GEAF
^
=
ρ
⁡
(
A
)
​
∏
ℓ
‖
W
ℓ
‖
2
\widehat{\mathrm{GEAF}}=\rho(A)\prod_{\ell}\|W_{\ell}\|_{2}
and uses a joint node-edge operator
B
B
for the recursion of
(
e
k
X
,
e
k
A
)
(e^{X}_{k},e^{A}_{k})
in dynamic-edge GWMs.
We analyze accumulative rollout error in GWMs both theoretically and empirically. The graph error amplification factor
GEAF
^
=
ρ
⁡
(
A
)
​
∏
ℓ
‖
W
ℓ
‖
2
\widehat{\mathrm{GEAF}}=\rho(A)\prod_{\ell}\|W_{\ell}\|_{2}
separates topology-dependent amplification from the model-dependent spectral norm product. For dynamic-edge GWMs, we introduce a joint node-edge error operator
B
B
that captures node-to-node, edge-to-node, node-to-edge, and edge-to-edge propagation. This operator cleanly separates two regimes: fixed-edge (FE) models have no edge-prediction head, so
M
X
=
0
M_{X}=0
and the dynamics reduce to node-only prediction; dynamic-edge (DE) models have an edge-prediction head, so
M
X
>
0
M_{X}>0
and node-edge cross-coupling becomes active.
We evaluate the framework on a seven-topology synthetic benchmark with established baselines, including GCN
Kipf and Welling (2017)
, MPNN
Gilmer et al. (2017)
, GPS
Rampášek et al. (2022)
, and ActionNode GWM
Feng et al. (2025)
. The benchmark includes FE and DE simulators, plus heterogeneous agent-graph testbeds inspired by tool/API and agent benchmarks
Li et al. (2023)
;
Qin et al. (2024)
;
Liu et al. (2024b)
;
Trivedi et al. (2024)
. The results support a regime-dependent view of GWM rollout error. Long rollouts turn small prediction errors into larger planning errors. When graph structure evolves, dynamic-edge training sharply outperforms fixed-edge training, and the learned dynamics exhibit the node-edge coupling predicted by the joint operator. The central empirical point is that GWM stability is not an architecture property alone; it depends on topology, horizon, and rollout regime.
We further propose
Error-Aware GWM
to improve rollout stability on high-spectral-radius graphs. It combines spectral regularization, rollout consistency, and critical-node weighting to control topology-induced amplification while preserving prediction accuracy. Empirically, Error-Aware GWM removes the divergence observed in vanilla GCN rollouts and keeps long-horizon prediction error low. Simpler stabilizers can reduce divergence, but often lose predictive fidelity. This distinction matters: a useful graph world model must be stable enough to roll out and accurate enough for those rollouts to support planning.
Our contributions are summarized as follows:
•
Unified GWM framework with FE/DE instantiations (Section
3
).
We formulate graph-structured agent planning as state-action-transition modeling and support both fixed-edge and dynamic-edge rollouts.
•
Topology-aware rollout error theory (Section
4
).
We develop graph-valued error bounds that account for topology-dependent amplification and node-edge coupling during autoregressive rollout.
•
Error-Aware GWM for stable planning (Section
6
).
We introduce a training objective that combines spectral control, rollout consistency, and critical-node weighting to improve long-horizon stability.
•
Evaluation across synthetic and agent-graph settings (Section
7
).
We test the framework across diverse graph topologies, rollout regimes, and agent-graph environments to study when GWMs are stable or unstable.
•
Scope analysis on real-world benchmarks (Section
7.7.1
).
We evaluate GWMs beyond controlled synthetic dynamics and clarify their practical boundary relative to standard graph methods.

## Method

3.1
Preliminaries: Agent Graph World
A
Graph World Model
(GWM) represents an agent environment as an evolving graph and predicts future graph states for planning. At time
t
t
, the graph state is
G
t
=
(
V
,
E
t
,
X
t
,
A
t
)
.
G_{t}=(V,E_{t},X_{t},A_{t}).
(3.1)
Here,
V
V
is the fixed set of agent, tool, skill, or task nodes;
E
t
E_{t}
is the edge set;
X
t
∈
ℝ
N
×
d
X_{t}\in\mathbb{R}^{N\times d}
is the node-feature matrix; and
A
t
∈
{
0
,
1
}
N
×
N
A_{t}\in\{0,1\}^{N\times N}
is the adjacency matrix. In an agent system, node features can encode agent status, tool outputs, skill availability, task progress, or failure signals. Edges represent calling, dependency, routing, or communication relations.
An action
a
t
a_{t}
is represented as an action node injected into the graph. The learned transition model
ℱ
θ
:
(
G
t
,
a
t
)
↦
G
^
t
+
1
\mathcal{F}_{\theta}:(G_{t},a_{t})\mapsto\hat{G}_{t+1}
(3.2)
predicts the next graph state. During planning, an agent evaluates candidate action sequences
a
1
,
…
,
a
H
a_{1},\ldots,a_{H}
by rolling out
G
^
k
+
1
=
ℱ
θ
(
G
^
k
,
a
k
)
,
k
=
0
,
…
,
H
−
1
,
\hat{G}_{k+1}=\mathcal{F}_{\theta}(\hat{G}_{k},a_{k}),\qquad k=0,\ldots,H-1,
(3.3)
and scoring the predicted outcomes. Thus, GWM planning reduces long-horizon decision-making to simulation over predicted graph trajectories.
3.2
Action Node Mechanism
Following Feng et al.
Feng et al. (2025)
, actions are encoded as action nodes connected to the current graph. This gives a unified representation for three kinds of planning actions:
node-level actions
, which modify a specific agent, skill, tool, or task node;
edge-level actions
, which add, remove, or update dependency or communication edges; and
graph-level actions
, which change global routing, coordination, or allocation patterns.
We use two action-node connection types.
Explicit action nodes
connect to target nodes through task-defined structure, such as a planner calling a tool or assigning a subtask to an executor.
Retrieval-based action nodes
connect through embedding similarity or retrieval, such as a query routed to related skill nodes or auxiliary tools. The same transition model can therefore represent explicit decisions and soft retrieval-based dependencies in one agent graph.
3.3
Two GWM Instantiations
Fixed-Edge GWM (FE-GWM).
In the fixed-edge setting, topology is fixed during rollout and only node states are predicted. A representative GCN-style transition is
X
^
k
+
1
=
tanh
⁡
(
A
~
​
X
^
k
​
W
1
+
U
​
a
k
)
​
W
2
+
ξ
k
,
\hat{X}_{k+1}=\tanh(\tilde{A}\hat{X}_{k}W_{1}+Ua_{k})W_{2}+\xi_{k},
(3.4)
where
A
~
=
D
−
1
/
2
A
D
−
1
/
2
.
\tilde{A}=D^{-1/2}AD^{-1/2}.
(3.5)
Here,
U
​
a
k
Ua_{k}
injects the action signal and
ξ
k
\xi_{k}
denotes rollout noise. This setting covers ActionNode GWM
Feng et al. (2025)
and standard GNN-based world models with fixed agent-graph structure. Since edges are not predicted, FE-GWM propagates node-state error through a fixed graph, with dynamics governed by
ρ
⁡
(
A
)
\rho(A)
and the model weight norms.
Dynamic-Edge GWM (DE-GWM).
In the dynamic-edge setting, both node features and graph structure are predicted autoregressively. In addition to the node transition, DE-GWM uses an edge-prediction head such as
A
^
i
​
j
=
σ
⁡
(
h
i
⊤
​
Q
​
h
j
)
,
\hat{A}_{ij}=\sigma(h_{i}^{\top}Qh_{j}),
(3.6)
where
h
i
,
h
j
h_{i},h_{j}
are node embeddings and
Q
Q
parameterizes pairwise edge scores. The model is trained with
ℒ
=
ℒ
node
+
λ
e
​
BCE
​
(
A
^
,
A
)
+
1
2
​
‖
W
‖
2
​
‖
Q
‖
2
.
\mathcal{L}=\mathcal{L}_{\rm node}+\lambda_{e}\mathrm{BCE}(\hat{A},A)+\tfrac{1}{2}\|W\|_{2}\|Q\|_{2}.
(3.7)
This setting models changing agent interactions, such as task routing, tool calls, validation links, or recovery paths. DE-GWM activates node-edge cross-coupling: node errors can affect future edge prediction, and edge errors can change future message passing. In the joint-operator view, FE-GWM is the special case with no edge-prediction head and
M
X
=
0
M_{X}=0
; DE-GWM has
M
X
>
0
M_{X}>0
and enables node-edge feedback.
