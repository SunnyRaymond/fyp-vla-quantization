# Hierarchical Planning with Latent World Models

paper_id: semanticscholar:477e20594635eb0ced89bbf2dd9ec873278cf48d
tier: T2
source_used: html_arxiv
warning: none

## Intro

Planning with learned world models has emerged as a powerful paradigm for embodied decision making: by simulating the consequences of candidate actions, an agent can reason about long-term outcomes without acting in the environment
[
47
,
10
,
17
]
. Recent advances in latent-space modeling and self-supervised learning have made it possible to train such models directly from large collections of unlabeled, task-agnostic offline trajectories, enabling zero-shot planning from high-dimensional observations such as pixels
[
1
,
46
,
6
,
62
,
13
]
. Yet single-level (flat) planning with these models fails in two distinct regimes. On non-greedy tasks like pick-and-place — where the optimal trajectory must temporarily move away from the goal — the goal-matching cost the planner minimizes is non-monotone along that trajectory, trapping local optimizers in greedy shortcuts even at modest horizons; empirically, even simple real-world pick-and-place is out of reach for state-of-the-art world-model planners such as VJEPA2-AC
[
1
]
(
table
1
). On long-horizon tasks, prediction errors compound over autoregressive rollouts
[
50
,
27
]
and the action search space grows exponentially with horizon
[
25
]
.
Introducing temporal hierarchy into decision-making — reasoning coarsely over long horizons while retaining fine-grained control at short horizons — has long been studied as a way to mitigate both effects, but existing approaches do not transfer cleanly to zero-shot planning on visual world models. Hierarchical reinforcement learning based on options, skills, or hierarchical policies
[
49
,
3
,
19
,
39
]
requires task-specific rewards or task distributions, and even modern goal-conditioned and zero-shot RL methods
[
39
,
41
]
generalize poorly beyond training
[
46
]
. Hierarchical world models have been explored in the RL setting
[
15
,
44
,
21
]
, but remain coupled to policy learning over a specific task distribution. Hierarchical MPC in optimal control
[
12
,
33
,
29
]
is task-agnostic but has been confined to low-dimensional states, hand-engineered features, or known dynamics. It remains unclear how to realize hierarchical zero-shot planning directly on visual world models.
We propose
Hierarchical Planning with Latent World Models
(HWM), a framework for zero-shot hierarchical MPC over learned latent world models (Table
7
contrasts HWM with prior hierarchical model-based methods). HWM trains world models at multiple temporal resolutions within a shared latent space, supervised solely by next-latent prediction, and couples them at planning time: predictions from the coarser model serve as subgoals that the finer-scale MPC matches in latent space; no hierarchical policy, skill, or goal-conditioned controller is needed. To support efficient long-horizon planning, we further introduce a learned action encoder that compresses sequences of primitive actions between waypoint states into latent macro-actions, reducing the dimensionality of the coarser-scale search. To our knowledge, HWM is the first world-model planner to demonstrate zero-shot, non-greedy real-robot manipulation from pixels with a single goal image, solving Franka pick-&-place at 70% success while single-level VJEPA2-AC planner
[
1
]
achieves 0%.
z
1
z_{1}
E
E
s
1
s_{1}
F
(
2
)
F^{(2)}
l
t
1
l_{t_{1}}
z
^
t
2
\hat{z}_{t_{2}}
F
(
2
)
F^{(2)}
l
t
2
l_{t_{2}}
F
(
2
)
F^{(2)}
l
t
H
−
1
l_{t_{H-1}}
z
^
H
\hat{z}_{H}
⋅
⋅
⋅
⋅
⋅
\cdot\ \ \cdot\ \ \cdot\ \ \cdot\ \ \cdot
F
(
1
)
F^{(1)}
a
1
a_{1}
F
(
1
)
F^{(1)}
a
h
a_{h}
z
^
h
\hat{z}_{h}
…
\dots
z
goal
z_{\text{goal}}
E
E
s
goal
s_{\text{goal}}
ℒ
low
\mathcal{L}_{\text{low}}
ℒ
high
\mathcal{L}_{\text{high}}
High-Level Planning
to Goal
arg
⁡
min
{
l
}
⁡
ℒ
high
\arg\min_{\{l\}}\mathcal{L}_{\text{high}}
Low-Level Planning
to
1
st
1^{\text{st}}
Subgoal
arg
⁡
min
{
a
}
⁡
ℒ
low
\arg\min_{\{a\}}\mathcal{L}_{\text{low}}
Model key
F
(
1
)
F^{(1)}
Low-level world model
F
(
2
)
F^{(2)}
High-level world model
Figure 2
:
Hierarchical planning in latent space.
A high-level planner optimizes macro-actions using a long-horizon latent world model to reach the final goal embedding.
The resulting first predicted latent state serves as a subgoal for low-level planning, where a short-horizon world model optimizes primitive actions to reach this subgoal.
We summarize our contributions as follows:
•
We introduce HWM, a hierarchical MPC formulation that couples learned world models at different temporal scales via a shared latent space, enabling direct subgoal transfer across levels without hierarchical policies, skill learning, or task-specific rewards.
•
We show that HWM unlocks a new capability for zero-shot world-model planning: solving non-greedy real-robot tasks from visual inputs, where success requires temporarily moving away from the goal. HWM achieves 70% success on Franka pick-&-place from a single goal image, compared to 0% for VJEPA2-AC
[
1
]
, and compares favorably to several zero-shot VLA baselines under the evaluated setup.
•
We demonstrate consistent gains across three latent world-model backbones — VJEPA2-AC (manipulation), DINO-WM (push manipulation), and PLDM (maze navigation) — and show that hierarchical planning improves both performance (up to
+
44
%
+44\%
and
+
39
%
+39\%
absolute) and efficiency (up to a
3
×
3\times
reduction in planning compute) on long-horizon tasks.

## Method

4.1
Manual Subgoals: Isolating Planning from Representation
Method
P&P
(Manual Subgoals)
P&P
(End-to-End)
Cup
Box
Cup
Box
VJEPA2-AC
80%
80%
0%
0%
VJEPA2-AC (hierarchy)
80%
80%
70%
60%
Table 4
:
Full Franka results, including the manual-subgoal control.
When pick-&-place is manually decomposed into grasp-then-place subtasks via intermediate subgoal images, the flat VJEPA2-AC planner reaches 80%/80%; without this supervision, the flat planner fails completely. HWM recovers 70%/60% of this performance automatically from a single goal image, without manual decomposition.
The single-level VJEPA2-AC planner fails entirely on end-to-end pick-&-place (Table
1
), but this aggregate failure conflates two possible causes: the VJEPA2-AC world model may lack the representational fidelity needed for manipulation tasks, or single-level MPC may be unable to escape the greedy-shortcut traps that non-monotone goal-matching induces (Section
1
). This section disentangles the two by supplying the single-level planner with manually-defined intermediate subgoals.
Setup.
For each of the 10 pick-&-place evaluation episodes per object, we manually annotate a single intermediate subgoal image corresponding to the grasp point. The single-level VJEPA2-AC planner is run in two sequential phases: first reaching the subgoal by grasping the object, then reaching the final goal by moving the object to the goal location.
Results.
Table
4
reports the Franka results for the manual-subgoal setup. With manual subgoals, the single-level VJEPA2-AC planner reaches 80%/80% on pick-&-place cup/box, far above its 0%/0% without subgoals. The VJEPA2-AC world model is therefore competent on these tasks given the right intermediate targets; the failure of single-level planning is hierarchical in nature, not representational. HWM recovers most of this oracle performance automatically (70%/60% from a single goal image), confirming that its contribution is to produce reachable subgoals without manual annotation.
4.2
Do High-Level World Models Improve Long Horizon Prediction?
Figure 6
:
Prediction error (
L
1
L_{1}
) as a function of prediction horizon.
For short horizons (
≤
\leq
1 s), the low-level world model is more accurate.
For longer horizons (
≥
\geq
1.5 s), a single-step prediction from the high-level world model outperforms autoregressive rollouts of the low-level model, reflecting reduced error accumulation.
A key challenge in long-horizon planning with learned world models is error accumulation during autoregressive rollouts, where small one-step errors compound over time. We hypothesize that models trained at longer temporal scales yield more accurate long-horizon predictions by reducing the number of autoregressive steps required. To test this, we condition both low-level and high-level world models—trained on the DROID dataset—on held-out initial observations and predict future states up to 2 seconds ahead, measuring
ℓ
1
\ell_{1}
error to the ground-truth future latent state. Predicting 2 seconds ahead requires up to 16 autoregressive steps for the low-level model, whereas the high-level model produces a single-step prediction. As shown in
Fig.
6
, the low-level model is more accurate for short horizons (
≤
\leq
1 s), while the high-level model achieves lower error for longer horizons (
≥
\geq
1.5 s), supporting a hierarchical strategy in which high-level planning provides long-term guidance and low-level planning handles short-term precision.
4.3
Learning Latent Macro-Actions vs. Handcrafted or Direct Representations
A key question is how to parameterize macro-actions in the hierarchical world model. We compare learned latent macro-actions—obtained by encoding sequences of low-level actions—against simpler alternatives: concatenating primitive actions, and hand-crafted summaries such as net end-effector displacement (delta pose) in robotic settings. Learned macro-actions consistently outperform both.
Compared to concatenating primitive actions, macro-actions achieve substantially higher planning performance (
table
6
) by compressing action sequences into a lower-dimensional space, reducing the high-level planning search complexity.
For the Franka experiments, we instantiate the handcrafted high-level action baseline as the net end-effector displacement between the start and end of each low-level action chunk. As shown in
table
6
, models trained with this delta-pose representation produce plans that align worse with expert actions than those trained with learned macro-actions. This gap suggests that simple displacement summaries can discard important temporal structure: in particular, they can collapse extended, non-greedy trajectories into a single endpoint displacement.
High Level Action
D
∈
[
9
,
12
]
D\!\in\![9,12]
D
∈
[
13
,
16
]
D\!\in\![13,16]
Concat Primitive Actions
52
37
Macro-Action
95
83
Table 5
:
Effect of macro-action parameterization on planning success rate (%) in Diverse Maze. Concatenating primitive actions yields substantially lower success than learned macro-actions.
High Level Action
Cos
↑
\uparrow
ℓ
1
\ell_{1}
↓
\downarrow
Delta Pose
0.80
±
0.02
0.80\pm 0.02
0.088
±
0.005
0.088\pm 0.005
Macro-Action
0.88
±
0.03
\mathbf{0.88\pm 0.03}
0.080
±
0.002
\mathbf{0.080\pm 0.002}
Table 6
:
Action alignment to expert trajectories on Franka for high-level world models using delta-pose vs. learned macro-actions (mean
±
\pm
SE). We report cosine similarity (
↑
\uparrow
) and
ℓ
1
\ell_{1}
(
↓
\downarrow
) between inferred and expert behavior.
Overall, learned macro-actions provide a more effective and general representation for high-level planning, capturing the structure of action sequences relevant for long-horizon prediction.
4.4
Emergence of semantic subgoals
Figure 7
:
Effect of macro-action dimension on hierarchical planning.
Left:
Performance as a function of macro-action dimension
d
d
.
When the latent space has very low capacity, the high-level planner fails to produce valid plans.
As
d
d
increases, success improves; however, beyond a moderate dimensionality, proposed subgoals become harder for the low-level planner to execute, as indicated by a drop in the cosine similarity between the inferred primitive actions from the hierarchical planner and expert behavior.
Right:
Qualitative rollouts illustrating this trade-off.
Moderate latent dimensionality biases the planner toward reachable, greedy subgoals, yielding the best overall performance.
We study how the macro-action dimensionality affects the two components required for successful hierarchical planning: whether the high-level model produces a valid plan to the final goal, and whether the first predicted subgoal is reachable by the low-level planner.
We evaluate these categories in two steps. First, we assess high-level plan validity qualitatively from decoded rollouts: a plan is considered valid if the decoded final high-level prediction realizes the desired goal. Second, among valid plans, we test whether the first high-level subgoal is reachable by the low-level planner by comparing the primitive actions inferred by the hierarchical planner with expert actions. In pick-&-place, reachability has a simple interpretation: a reachable subgoal should induce the same greedy behavior as the expert, namely moving toward and grasping the object before transporting it. By contrast, an unreachable or non-greedy subgoal induces incorrect low-level actions, such as moving directly toward the target location without first picking up the object. Low similarity to the expert action therefore indicates that the high-level plan may look valid globally, but proposes a subgoal that the low-level planner cannot reliably execute.
As shown in
Fig.
7
, when the macro-action space has sufficient capacity (
≥
4
\geq 4
dimensions), the high-level planner typically produces valid plans. However, these subgoals are not always reachable by the low-level planner, as they may require non-greedy action sequences. Restricting the macro-action dimensionality biases the planner toward proposing subgoals that are achievable with greedy behavior. This suggests an optimal regime in which the latent space is expressive enough to encode useful trajectories, but not so expressive that it enables subgoals requiring complex non-greedy execution. Empirically, a macro-action dimension of 4 strikes this balance for the Franka tasks.
Notably, reconstruction fidelity from latent predictions is not tightly correlated with hierarchical planning success. Lower-dimensional macro-actions yield noisier predictions, reflected in higher
ℓ
1
\ell_{1}
error and blurrier reconstructions (
Fig.
7
), as expected under stronger compression. Nevertheless, these predictions often preserve coarse semantic structure, such as contact events or motion direction, which is sufficient for hierarchical planning despite reduced visual precision.
