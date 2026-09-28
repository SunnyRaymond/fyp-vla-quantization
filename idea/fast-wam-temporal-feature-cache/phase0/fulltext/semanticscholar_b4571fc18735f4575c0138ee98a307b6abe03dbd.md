# Evolving Cache Schedules for Fast Diffusion Policy Inference

paper_id: semanticscholar:b4571fc18735f4575c0138ee98a307b6abe03dbd
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Diffusion policies have garnered substantial attention in robotic control for their
ability to model multimodal action distributions through conditional denoising
processes [3, 32, 6, 8]. With scalable transformer denoisers, this formulation has
⋆Corresponding author.
arXiv:2607.20293v1  [cs.CV]  22 Jul 2026

2
S. Wang et al.
1
25
50 74
99
Timestep
1
2
3
4
5
6
7
8
Block
Self Attention
0.02
0.04
0.06
0.08
0.10
1
25
50 74
99
Timestep
1
2
3
4
5
6
7
8
Block
Cross Attention
0.02
0.04
0.06
0.08
0.10
1
25
50 74
99
Timestep
1
2
3
4
5
6
7
8
Block
FFN
0.02
0.04
0.06
0.08
0.10
Fig. 1. Cross-step feature dissimilarity of transformer modules on Push-T. Higher
dissimilarity indicates larger feature variation between denoising steps and therefore
weaker temporal redundancy. The heatmaps show that different blocks exhibit distinct
temporal distributions of high-dissimilarity regions as well as different overall dissim-
ilarity levels, revealing heterogeneous redundancy patterns and degrees across blocks.
FFN denotes the feed-forward network.
become an expressive action-generation module for high-dimensional visuomotor
policies and increasingly complex manipulation settings [18, 1, 28]. However, the
iterative denoising loop also imposes a heavy inference burden that directly limits
the achievable action frequency, making it difficult for diffusion policies to satisfy
the low-latency requirements of real-time, smooth robotic control [10]. Recently,
cache-based acceleration methods provide a natural training-free approach to
reducing the inference cost of diffusion policies. These methods exploit temporal
redundancy along the denoising trajectory by caching intermediate activations
and reusing them at skipped positions, so that only selected block–timestep
positions need to be recomputed [16, 14, 11, 34, 35].
Existing training-free methods typically allocate computation across blocks
uniformly, which ignores heterogeneous redundancy across blocks and leads to a
suboptimal trade-off between performance and efficiency. EfficientVLA, for ex-
ample, refreshes all blocks synchronously with a fixed temporal interval, while
BAC chooses block-specific update timesteps but still assigns a fixed refresh
budget to each block [30, 9]. To examine whether such uniform allocation is
justified, we measure cross-step feature dissimilarity across transformer blocks.
Higher dissimilarity indicates larger feature variation between denoising steps
and therefore weaker computation redundancy. As shown in Fig. 1, blocks dif-
fer not only in the temporal distribution of high-dissimilarity regions, but also
in their overall dissimilarity levels, revealing distinct redundancy patterns and
degrees across blocks.
This observation motivates a global cache scheduling strategy that allocates
one fixed refresh budget over the entire block–timestep lattice, allowing com-
putation to move from redundant blocks to sensitive ones. For concreteness,
we view training-free acceleration as a fixed-budget cache-scheduling problem.
Given cacheable blocks B, denoising steps T , and a budget K, a schedule se-
lects K block–step positions to refresh and reuses cached activations for the
rest. A naive solution is exhaustive search, but enumerating K refreshes over

Evolving Cache Schedules for Fast Diffusion Policy Inference
3
the |B||T | block–step grid yields an exponential search space. Alternatively, one
might extend similarity-based approximation [9] to global scheduling, using fea-
ture similarity to estimate cache-induced errors without exhaustive evaluation.
However, under coupled block schedules, inter-block error propagation makes
update-induced errors difficult to formulate as a tractable optimization objec-
tive [9], limiting its applicability in global cache scheduling.
To bridge this gap, we propose EVO, a training-free framework that searches
global cache schedules with an evolutionary algorithm. EVO represents each
candidate as a complete refresh schedule and optimizes it through selection,
crossover, mutation, and elitism, using closed-loop rollout performance as the
fitness. While rollout-based fitness provides a direct estimate of deployment per-
formance, its high computational cost limits the practicality of large-scale search.
EVO therefore proposes two lightweight mechanisms to make search practical.
Specifically, redundancy-aware initialization uses activation dissimilarity only as
a soft prior, biasing the initial population toward less redundant block–step posi-
tions while keeping random individuals and full-space mutation for exploration.
Meanwhile, target-conditioned early stopping turns search into a goal-driven pro-
cedure: once a quick evaluation finds a candidate that reaches a baseline-relative
target, EVO confirms it with an independent formal evaluation before termina-
tion. The final schedule is then deployed as an offline-optimized cache plugin,
without altering the policy parameters, diffusion sampler, or action interface.
In summary, our contributions are:
– We reveal heterogeneous redundancy in Diffusion Policy that different trans-
former blocks exhibit distinct temporal redundancy patterns and degrees,
motivating a global cache scheduling strategy.
– We propose EVO, a training-free evolutionary framework that searches for
complete global cache schedules using closed-loop rollout performance as the
optimization objective.
– We develop redundancy-aware initialization and target-conditioned early
stopping to reduce the cost of rollout-driven evolutionary search while pre-
serving global exploration.
– We validate EVO on multiple robotic manipulation benchmarks, showing
that it preserves near-full policy performance while achieving up to 8.05×
action-generation speedup.
2

## Method

[25, 12, 13, 33], or reduce computation through pruning and distillation [28, 21,
10, 26]. However, they typically alter the sampling process or require additional
training. In contrast, our method accelerates transformer-based diffusion poli-
cies by reducing inference computation through caching, without retraining the
model or modifying the original sampler.
Cache-based Acceleration. Feature caching stores and reuses intermediate fea-
tures across denoising steps to avoid redundant computation. Existing methods
have shown strong acceleration for image generation in both U-Net diffusion
models and diffusion transformers [16, 29, 22, 35, 24, 15]. These methods typically
rely on fixed refresh rules, token-wise or feature-wise update strategies, or learned
routing networks, and they are mainly evaluated with image quality metrics such
as Fr´echet Inception Distance (FID). In action generation, EfficientVLA applies
feature caching to the diffusion-based action head of Vision-Language-Action
(VLA) models [30], but its cache refresh still follows a fixed-interval rule and does
not capture redundancy differences across blocks and denoising steps. Block-wise
Adaptive Caching (BAC) further selects refresh times for each block based on
feature similarity [9], but its refresh budget remains constrained within each
block. Existing cache-based methods therefore still lack global budget allocation
over the block–timestep lattice. Our method addresses this limitation by formu-
lating cache scheduling as a global allocation problem and directly optimizing
rollout success rate.
Evolutionary Search. Evolutionary algorithms and other black-box optimisers
are well suited to discrete combinatorial problems where gradients are unavail-
able [31, 19, 7, 27]. They have been applied to policy search and neural archi-
tecture optimisation for robotic tasks [23, 4, 5], where candidate architectures
or policies are often evaluated by empirical task performance. This setting is
closely related to our cache scheduling problem: each schedule is a discrete sub-
set of block–timestep refresh positions under a fixed budget, and its quality is
determined by closed-loop rollout performance rather than a differentiable ob-
jective. We therefore use evolutionary search as an offline optimizer for complete
cache schedules. Since black-box search with task-level evaluation often incurs
high evaluation cost, EVO further incorporates redundancy-aware initialization
and target-conditioned early stopping to reduce unnecessary evaluations.
3
Method
EVO accelerates a pretrained transformer-based diffusion policy by optimizing
where cache refreshes are performed during iterative denoising. Rather than as-
signing a uniform or per-block refresh budget, EVO searches for a globally con-
strained subset of block–timestep positions under a fixed computation budget.

Evolving Cache Schedules for Fast Diffusion Policy Inference
5
(a) EVO Framework
Population Set
…
Schedule Population
Dedup & 
Refill
Evolutionary Schedule Search
Selection
Crossover
Mutation
Produce new candidate schedules
Iterative
Evolution
Top 
as Elite Schedules
…
Keep high-fitness Schedules
Rollout Fitness Evaluation
Candidate Schedule 
Simulated Rollouts
…
0
1
Fitness=Success Rate
Best Verified Schedule 
Output
Initial Schedule
Population
Next Generation
Evaluate candidates
9                                      
× Block
Temporal
features 
Dissimilarity 
matrices
…
Initial Schedule
Population
Guided 
Individuals
Random Individuals
(b) Population Initialization
(c) Target-conditioned Early Stopping
Stop & Output
Best Verified  
Pass
Continue Search
Residual Transformer Block
Updated Features
Skipped Features
Candidate Cache Schedule
1-th
N-th
Formal Evaluation
• More rollouts
• Independent seed
• High-confidence test
1
2
1
2
1
2
1
2
Best Quick Fitness
Success Rate
Target Threshold
Trigger
Evaluated Generations
Fig. 2. Framework of EVO, a training-free acceleration framework for global cache
scheduling via evolutionary search. (a) EVO searches for a globally budgeted cache
schedule, using rollout fitness as the selection criterion. (b) Redundancy-aware and
random individuals are combined to initialize the schedule population. (c) Target-
conditioned early stopping performs formal verification for promising schedules and
terminates once the desired performance target is reached.
As illustrated in Fig. 2, the method formulates cache scheduling as a subset-
selection problem over the block–timestep lattice and optimizes complete sched-
ules with evolutionary search, using closed-loop rollout performance as the fit-
ness. Redundancy-aware initialization and target-conditioned early stopping are
introduced to reduce the cost of this offline search. The resulting schedule is then
utilized by the pretrained policy without retraining or modifying the diffusion
sampler.
3.1
Global Cache Scheduling
Let πθ denote a pretrained diffusion policy. For each action query, the policy
starts from initial noise and performs T reverse denoising steps, indexed by
T = {0, . . . , T −1}. At each denoising step, the transformer denoiser propagates
the action representation through a sequence of cacheable computation units.
For an L-layer transformer decoder, the self-attention, cross-attention, and feed-
forward residual branches in each layer are used as cacheable units, yielding
B = 3L units indexed by B = {1, . . . , B}. The cacheable computations of one

6
S. Wang et al.
1
2
3
4
5
6
7
8
Layer
SA
CA
FFN
0.00
0.02
0.04
0.06
0.08
0.10
Success Rate Drop
(a) Single-block cache stress test
0.94
0.96
0.98
1.00
Mean Adjacent-step Cosine Similarity
0.0
0.1
0.2
0.3
Success Rate Drop
Spearman ρ = -0.053
Module Type
SA
CA
FFN
(b)
Feature similarity vs. task sensitivity
Fig. 3. Motivation for rollout-driven global cache scheduling. (a) Under the same sparse
update pattern, different transformer blocks cause different success drops, indicating
non-uniform task sensitivity across layers and module types. (b) The weak correlation
between adjacent-step feature similarity and success drop suggests that feature simi-
larity is useful as a prior but insufficient as the final scheduling objective.
action query can therefore be represented as a two-dimensional block–timestep
lattice B × T .
Each lattice position (b, t) corresponds to the residual-branch computation
of unit b at denoising step t. EVO maintains one residual cache cb for each unit
and resets all caches before the next action query. Let rb,t be the residual output
of unit b at denoising step t. Given a cache schedule S, the residual ˜rb,t used by
the accelerated policy is
˜rb,t =
(
rb,t,
(b, t) ∈S,
cb,
(b, t) /∈S,
cb ←rb,t if (b, t) ∈S.
(1)
Thus, selected positions evaluate the original residual branch and refresh the
cache, whereas unselected positions reuse the most recently cached residual of
the same unit. Cache scheduling only changes which residual computations are
executed, while the policy parameters, observation encoder, diffusion sampler,
and action interface remain unchanged.
Per-block scheduling constrains different blocks to receive comparable refresh
allocations, although their effects on closed-loop control may differ substantially.
This assumption is evaluated by applying the same sparse update pattern to each
block individually while keeping all other blocks fully computed. As shown in
Fig. 3(a), the resulting success drop varies across blocks, indicating clear block-
dependent task sensitivity. This motivates a global allocation strategy in which
refreshes are assigned over the full lattice rather than fixed within each block.
Motivated by this non-uniformity, EVO defines a cache schedule as a fixed-
budget subset of the block–timestep lattice:
S ⊆B × T ,
|S| = K,
(2)
where K denotes the total refresh budget, i.e., the number of lattice positions at
which the original residual computation is executed. Unlike per-block schedul-
ing, this formulation does not pre-assign a fixed number of refreshes to each

Evolving Cache Schedules for Fast Diffusion Policy Inference
7
block, allowing the budget to be allocated globally according to downstream
performance.
3.2
Evolutionary Search over Schedules
The fixed-budget search space contains
 BT
K

feasible schedules, making exhaus-
tive enumeration impractical. EVO therefore employs evolutionary search to
optimize complete cache schedules in this large, discrete, and non-differentiable
space. In the evolutionary algorithm, we represent each schedule S by an indi-
vidual x, which contains K unique refresh positions:
x = {(bi, ti)}K
i=1,
(bi, ti) ∈B × T ,
|x| = K.
(3)
Feature similarity is commonly used as a proxy objective for cache scheduling,
based on the intuition that similar activations across adjacent denoising steps
can be reused with limited error. In closed-loop control, however, activation sim-
ilarity is not necessarily aligned with task-level sensitivity. Fig. 3(b) shows that
blocks with comparable similarity scores can induce different success drops, and
the overall correlation between feature similarity and task sensitivity is weak.
Additional analysis across Kitchen, Square MH, Tool PH, Transport MH, and
Push-T shows a similar trend. Detailed results are provided in Sec. A.4 of the
supplementary material. EVO therefore uses feature similarity only for initial-
ization and adopts rollout success rate as the fitness for schedule optimization.
Given a candidate schedule S, EVO applies the corresponding cached exe-
cution to the pretrained policy, denoted by πθ(S), and estimates its empirical
success rate over evaluation episodes:
F(S) =
1
Neval
Neval
X
n=1
1{Success(τn; S)},
τn ∼πθ(S).
(4)
This fitness evaluates the effect of a cache schedule on the closed-loop behav-
ior of the accelerated policy. EVO optimizes the schedule by maximizing the
downstream rollout success rate under the fixed refresh budget:
S∗= argmax
|S|=K
F(S),
S ⊆B × T .
(5)
The evolutionary procedure iteratively applies tournament selection, set-level
crossover, mutation, repair, and elitism. Tournament selection samples parent
schedules according to their rollout fitness. Set-level crossover exchanges refresh
positions between two parents, while mutation resamples a subset of positions
from the full block–timestep lattice. The repair operator removes duplicate po-
sitions and fills missing positions so that each child remains a feasible schedule
with exactly K unique refreshes. Elitism preserves the best-performing schedules
across generations, and evaluated schedules are stored using canonical signatures
to avoid repeated rollouts. These operators maintain the fixed-budget constraint
while allowing refresh allocations to move freely across blocks and denoising
steps.

8
S. Wang et al.
3.3
Practical Search Mechanisms
Although rollout-based fitness directly reflects closed-loop control performance,
it is expensive because each candidate must be evaluated through environ-
ment rollouts. EVO reduces this offline cost with two search-time mechanisms.
Redundancy-aware initialization improves the starting population, and target-
conditioned early stopping limits unnecessary rollout evaluations. Both mech-
anisms affect only the search process; schedule selection remains governed by
rollout fitness.
Redundancy-aware Initialization. Feature similarity is not used as the final op-
timization objective, but it provides a useful prior for constructing the initial
population. While uniformly random initialization provides broad exploration,
it may allocate early refresh positions to highly redundant parts of the lattice.
EVO therefore uses activation redundancy to bias part of the initial population
toward less redundant block–timestep positions.
Residual activations are collected from a small set of rollout steps generated
by the uncached policy. Let ar,b,t denote the activation for rollout index r, unit
b, and denoising step t. Each block–timestep position is assigned the following
dissimilarity score:
db,t = Er


1
T −1
X
t′̸=t

1 −
⟨ar,b,t, ar,b,t′⟩
∥ar,b,t∥2∥ar,b,t′∥2

.
(6)
A larger db,t indicates that the activation at position (b, t) is less similar to
other denoising steps within the same module, and hence has lower feature re-
dundancy. After normalizing db,t into wb,t, guided individuals sample refresh
positions according to
pb,t =
(ϵ + wb,t)γ
P
(b′,t′)∈B×T (ϵ + wb′,t′)γ .
(7)
This distribution assigns larger initial sampling probabilities to less redundant
positions. The prior is used only for initialization: the initial population also
contains random schedules, and subsequent mutation and repair can sample any
position in B × T . Rollout fitness remains the only selection criterion.
Target-conditioned Early Stopping. To avoid unnecessary evaluations after a
satisfactory schedule has been found, EVO terminates search once a candidate
reaches the prescribed performance target. Let qbase denote the formal rollout
score of the uncached policy. During evolution, candidate schedules are first
screened with a low-cost quick evaluation. When a schedule satisfies
qquick(S) ≥qbase,
(8)
EVO performs an independent formal evaluation using a larger rollout budget
and independent random seeds. The schedule is accepted, and search terminates
only if
qformal(S) ≥qbase −δacc,
(9)

Evolving Cache Schedules for Fast Diffusion Policy Inference
9
where δacc is the allowed absolute performance drop. The quick evaluation there-
fore serves as a screening stage for promising candidates, while the formal eval-
uation provides the final acceptance criterion and mitigates the effect of noisy
quick rollouts. Representative search trajectories in Sec. A.6 of the supplemen-
tary material illustrate that this mechanism reduces unnecessary evaluations
while avoiding premature termination.
3.4
Offline-Optimized Deployment
After search, EVO fixes the best verified schedule and removes the optimizer
from the inference loop. At test time, the deployment wrapper loads S∗, resets
the residual caches at the beginning of each action query, and follows Eq. 1
to determine whether each block–timestep position is refreshed or replaced by
cache reuse. The policy weights, observation encoder, diffusion sampler, and
action interface remain unchanged. The search is performed offline only once for
a given task. The resulting schedule is reused across subsequent episodes drawn
from the same environment distribution, without online adaptation or repeated
search. Further analysis is provided in Sec. 4.3.
4
Experiments
This section evaluates whether EVO can reduce diffusion-policy inference cost
while preserving closed-loop control performance, without retraining or modify-
ing the pretrained policy. We first introduce the experimental setup, including
benchmark tasks, policy backbones, evaluation metrics, and implementation de-
tails. We then compare EVO with representative training-free acceleration base-
lines to evaluate the trade-off between task success rate and inference speed.
Finally, we conduct ablation studies to analyze how global block-step scheduling
and similarity-guided initialization contribute to the overall performance.
Table 1. Benchmark on Proficient Human (PH) demonstration data. Success rates are
evaluated by mean ± standard deviation over three evaluation seeds, and speedups are
measured with full DP-T inference. EVO achieves a favorable trade-off between task
success rate and inference speed, maintaining an average success rate of 0.78 while
reaching up to 8.05× speedup.
Method
Success Rate ↑
AVG
FLOPs
Speed ×
Lift
Can
Square
Transport
Tool
Push-T
Full Precision
1.00 ± 0.00
0.99 ± 0.01
0.85 ± 0.04
0.82 ± 0.00
0.45 ± 0.05
0.66 ± 0.02
0.80
15.77G
–
EfficientVLA (M = 8)
1.00 ± 0.00
0.79 ± 0.06
0.85 ± 0.05
0.69 ± 0.07
0.25 ± 0.10
0.64 ± 0.02
0.70
1.64G
9.62
EfficientVLA (M = 10)
0.89 ± 0.05
0.10 ± 0.05
0.40 ± 0.04
0.23 ± 0.05
0.00 ± 0.00
0.59 ± 0.03
0.37
1.95G
8.09
BAC (M = 8)
1.00 ± 0.00
0.63 ± 0.03
0.86 ± 0.05
0.77 ± 0.08
0.37 ± 0.09
0.66 ± 0.03
0.72
2.33G
6.77
BAC (M = 10)
1.00 ± 0.00
0.63 ± 0.02
0.91 ± 0.05
0.81 ± 0.04
0.48 ± 0.09
0.66 ± 0.01
0.75
2.72G
5.80
EVO (M = 8)
1.00 ± 0.00
0.98 ± 0.04
0.88 ± 0.02
0.74 ± 0.02
0.39 ± 0.08
0.66 ± 0.03
0.78
1.96G
8.05
EVO (M = 10)
1.00 ± 0.00
0.97 ± 0.01
0.87 ± 0.02
0.76 ± 0.04
0.41 ± 0.05
0.69 ± 0.03
0.78
2.33G
6.77

10
S. Wang et al.
Table 2. Benchmark on Mixed Human (MH) demonstration data. Success rates are
evaluated by mean ± standard deviation over three evaluation seeds, and speedups are
measured with full DP-T inference. EVO preserves the full-model average success rate
of 0.79 while achieving up to 7.92× speedup.
Method
Success Rate ↑
AVG
FLOPs
Speed ×
Lift
Can
Square
Transport
Full Precision
1.00 ± 0.00
0.94 ± 0.02
0.69 ± 0.14
0.51 ± 0.01
0.79
15.77G
–
EfficientVLA (M = 8)
1.00 ± 0.00
0.72 ± 0.09
0.53 ± 0.06
0.00 ± 0.00
0.56
1.64G
9.62
EfficientVLA (M = 10)
0.75 ± 0.05
0.09 ± 0.04
0.02 ± 0.02
0.00 ± 0.00
0.22
1.95G
8.09
BAC (M = 8)
1.00 ± 0.00
0.73 ± 0.08
0.61 ± 0.06
0.45 ± 0.06
0.70
2.32G
6.80
BAC (M = 10)
1.00 ± 0.00
0.94 ± 0.02
0.75 ± 0.06
0.47 ± 0.10
0.79
2.71G
5.82
EVO (M = 8)
1.00 ± 0.00
0.96 ± 0.02
0.71 ± 0.06
0.47 ± 0.03
0.79
1.99G
7.92
EVO (M = 10)
1.00 ± 0.00
0.93 ± 0.03
0.77 ± 0.05
0.47 ± 0.03
0.79
2.34G
6.74
4.1
Experimental Setup
Models, Benchmarks, and Metrics. Following the standard setting of Dif-
fusion Policy, we adopt the Transformer-based Diffusion Policy (DP-T) as the
base policy and evaluate EVO by publicly released pretrained checkpoints. We
conduct experiments on multiple robotic manipulation benchmarks, including
RoboMimic tasks, Push-T, Block Push, and Kitchen. For the multi-stage re-
sults in Table 3, BP and Kit denote Block Push and Kitchen, respectively, and
AVG denotes the average over all reported success-rate metrics. The demon-
stration data comprises mixed proficient/non-proficient human (MH) teleopera-
tion demonstrations, proficient human (PH) teleoperation demonstrations, and
expert trajectories generated by scripted Markovian policies for several low-
dimensional tasks. For most manipulation tasks, we treat success rate as the
primary performance metric, while Push-T is evaluated by target-area cover-
age. To measure inference efficiency, we report the FLOPs required for action
generation alongside the speedup relative to the full DP-T model.
Baselines. We leverage the full DP-T policy as the full-precision baseline and
subsequently compare EVO with two representative training-free caching meth-
ods. EfficientVLA refreshes features at fixed denoising intervals as a static schedul-
ing paradigm, while BAC achieves local adaptive scheduling through a block-wise
update policy and an error-propagation repair mechanism. Distinct from prior
works, EVO globally searches for cache schedules over the full block-timestep
lattice with more adaptive and fast performance.
Implementation Details. All methods are evaluated by the final checkpoint
of the publicly released pretrained DP-T policy, with the denoising process set
to 100 steps. EVO adopts a unified genetic-search configuration with population
size, number of elites, and maximum number of generations set to 30, 10, and 30,
respectively. Unless otherwise specified, we evaluate two cache-budget settings,
M = 8 and M = 10, where M denotes the average number of refresh steps per
cacheable module. Since DP-T contains 24 cacheable modules, these settings

Evolving Cache Schedules for Fast Diffusion Policy Inference
11
correspond to total budgets of K = 192 and K = 240 block-timestep update
positions, respectively. During the search, each candidate schedule S is first
evaluated by 20 rollout episodes. Candidates satisfying the target trigger are
then confirmed by 50 rollout episodes with independent random seeds. Search
stops early when the confirmed score satisfies the target performance criterion.
Following the search phase, the selected cache schedule is frozen and deployed
for the final inference evaluation. Final results are ev
