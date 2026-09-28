# BAG: Budget-Aware Gating for Diffusion Caching

paper_id: arxiv:2608.09231v2
tier: T3
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Diffusion Transformers (DiTs) [31] have established them-
selves as the standard architecture for state-of-the-art text-to-
image and text-to-video generation [4, 18, 19, 40, 41, 45].
However, generating a single sample requires dozens of se-
quential forward passes through the network [13, 22], lead-
ing to high latency and substantial serving costs. To mitigate
this computational bottleneck, several acceleration paradigms
have been explored, including fast numerical solvers [26, 42,
54, 57], model distillation into few-step or even single-step
generators [37, 39], and post-training quantization [5, 21].
Among these approaches, diffusion caching [23, 25, 29, 38,
44, 51, 55] offers a particularly lightweight and complemen-
tary strategy. Because internal feature maps evolve gradu-
ally between adjacent denoising steps, certain steps can by-
pass full network evaluation by reusing cached computations.
Caching requires no architecture modifications, retraining, or
alterations to the underlying sampler, making it orthogonal to
other acceleration techniques. Given a T-step sampling pro-
cess and a budget B representing the total number of allowed
function evaluations (NFEs), a caching policy must decide
at each step whether to perform a full computation or reuse
cached features. These T sequential binary decisions form a
cache schedule, which directly governs the trade-off between
inference speed and generation quality.
Existing diffusion caching methods fall into two main
paradigms.
The first paradigm uses online heuristic rules
based on proxy signals like feature drift [7, 16, 23]. While
instance-adaptive and lightweight, these rules lack budget
awareness: they ignore the remaining budget when making
per-step decisions, preventing global computation pacing. In
addition, the realized NFE can only be controlled indirectly
by tuning the threshold.
Conversely, the second paradigm
relies on static offline schedules, using predefined intervals
[27, 29, 38] or search-based timetables [20, 28]. Although
static plans guarantee exact budget adherence via global plan-
ning, their open-loop nature imposes identical schedules on
all instances and requires re-searching for every new budget
constraint. Ultimately, neither paradigm combines real-time
trajectory feedback with global budget pacing.
To quantify the performance lost by current paradigms,
Fig. 3 compares deployed methods against a reference sched-
ule obtained via prompt-specific offline search [1, 17] un-
der identical NFE constraints. Across prompts and budgets,
the prompt-specific schedule preserves full-compute genera-
tion quality substantially better than existing online and static
baselines.
Furthermore, both the offline-optimized budget
allocation and the resulting quality gains vary substantially
1
arXiv:2608.09231v2  [cs.CV]  14 Aug 2026

2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
2.5x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.1x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
3.8x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
4.5x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
5.0x
Figure 1. One gate, any budget. Samples from Qwen-Image-2512 [45] accelerated by a single BAG gate (under 1K parameters), with the
speedup increasing from 2.5× on the left to 5.0× on the right; the label at the top-right corner of each image gives its speedup. The same gate
serves every budget, prompt, and resolution shown, and spends exactly the requested compute budget.
across different prompts. These results demonstrate the inher-
ent suboptimality of existing paradigms: hand-crafted heuris-
tics lack global budget awareness, static schedules lack in-
stance adaptivity. However, prompt-specific search is com-
putationally prohibitive for real-time inference.
To bridge this gap, an effective cache policy must simul-
taneously monitor two key signals at every step: the bud-
get state, which determines the overall pacing of remaining
compute, and the local trajectory, which evaluates whether
the current step warrants full computation. Rather than hand-
crafting complex decision heuristics, we propose learning this
policy directly. Our method, BAG (Budget-Aware Gating), in-
troduces a lightweight gating network of under 1K parameters
that conditions its decisions on two distinct sources of infor-
mation: global budget constraints and local trajectory dynam-
ics. Specifically, the global budget context is captured by three
scalar indicators: the target compute ratio, the fraction of bud-
get remaining, and the relative budget tightness over the re-
maining horizon. Simultaneously, local trajectory dynamics
are reflected by three fine-grained signals: the elapsed steps
since the previous full computation, the relative feature drift
accumulated since that computation, and the step-to-step fea-
ture variation. From these six scalar inputs, the gate outputs a
binary compute-or-reuse decision at a cost negligible relative
to a network forward pass. This dual-context design addresses
the shortcomings of prior paradigms: trajectory feedback en-
ables fine-grained, per-step adaptation, explicit budget con-
ditioning coordinates computation over the remaining hori-
zon, and a learned decision rule replaces hand-crafted, setting-
specific heuristics.
To train the policy network, we formulate the learning pro-
cess as offline-to-online schedule distillation. During train-
ing, we perform offline searches across diverse prompts and
target budgets to construct high-quality reference schedules.
2

1. Search
2. The idea
3. Showtime !
A blue colored dog...
7 steps please ! 
Super Mario, 8K HD...
13 steps max !
A green cup and
a blue phone...
9 steps !
Distill them into
a tiny gate !
An astronaut
riding a horse...
10 steps !
Search done ...
but what about
UNSEEN prompts online?
...
7/50
...
13/50
...
9/50
= compute
= cache
0 / 1
compute 
label
Budget
trajectory
p = 0.9
label
1
compute !
2
cache !
3
cache !
4
compute !
5
cache !
6
cache !
7
cache !
8
compute !
9
compute !
10
cache !
...
Figure 2. BAG overview: learn the scheduler, not the schedule. Among the compared scheduling paradigms, BAG deploys the only per-step
rule that reads both the remaining budget and the realized trajectory. (1) Search (training only, once per backbone): for each (prompt, budget)
cell, a matched-NFE search produces a reference schedule: a full-horizon allocation of exactly B evaluations, optimized to preserve the full-
compute output. (2) Distill: each reference rollout is decomposed into per-step decision examples, from which a tiny gate learns to map budget
state and trajectory feedback to compute-or-reuse decisions. (3) Deploy: on an unseen prompt the frozen gate decides closed-loop at every step,
spending exactly the requested budget.
Because each reference schedule is optimized with full access
to the sampling trajectory and global budget, it serves as a
strong target. We then replay these rollouts, extract the cor-
responding budget and trajectory states at each step, and train
the gate via supervised learning to replicate the reference de-
cisions. Consequently, global horizon planning is performed
once offline, while only the compact learned policy runs dur-
ing inference. By training across a range of target budgets, the
gate learns how offline-optimized compute allocation shifts as
a function of available budget B, effectively learning a gener-
alized scheduling policy rather than a fixed schedule. At infer-
ence, the frozen gate makes closed-loop decisions on unseen
prompts. The target budget B is set directly at runtime and
met exactly by construction. A single checkpoint therefore
supports different budgets and step counts without retraining
or re-search, while remaining robust to changes in seed, reso-
lution, and guidance scale.
We treat high-quality cache schedules across prompts and
budgets as outputs of one decision rule that reads both the
remaining budget and the realized trajectory, and propose
BAG, an offline-to-online schedule distillation framework
that learns a lightweight budget- and trajectory-conditioned
gate from searched reference schedules. Once trained, the
same gate supports all evaluated budgets and sampling step
counts, spends exactly the requested number of NFEs, and
requires no further search or training. Comprehensive exper-
iments on FLUX.1-dev [19], Wan2.1 [41], and Qwen-Image-
2512 [45] demonstrate consistent improvements over state-
of-the-art static and online caching methods across all re-
construction metrics under matched computation, including a
+1.2 dB PSNR gain over the strongest baseline at the ∼5×
FLUX acceleration tier.

## Method

3.1. Preliminaries
Problem setup. A sampler runs T denoising steps (T=50 for
training and the main comparison). A cache schedule is a bi-
nary mask m = (m0, . . . , mT −1), mt ∈{0, 1}: step t runs
the network and refreshes the cache if mt=1, and otherwise
reuses the cached computation through the standard resid-
ual path, which we hold fixed for every scheduling method
studied. The budget fixes the number of function evaluations
(NFEs) to ∥m∥1 = B, with m0=1 forced. Writing xT (m, p)
for the output on prompt p under schedule m, caching seeks to
approximate the same-prompt, same-seed full-compute out-
put xT (1, p) with only B of the T evaluations. We evaluate
reconstruction with PSNR, SSIM [43], and LPIPS [52]; some
works instead report reference-free scores such as ImageRe-
ward [47]. We compare all methods on the three reconstruc-
tion metrics.
Two scheduling paradigms.
Static methods fix the full
schedule before sampling, m = µ(B, T), based only on the
budget and step count. They meet the budget exactly through
full-horizon allocation, but use the same open-loop plan for
every prompt and require a new plan for each budget. Online
threshold rules decide per step, mt = I[ qt > δ ], where
qt is a hand-crafted trajectory signal: decisions respond to
the realized rollout, but the realized computation is only an
indirect consequence of δ, and neither the remaining budget
nor the horizon enters the decision.
Motivation.
Fig. 3 measures what these restrictions cost:
at matched NFE, per-prompt searched schedules reconstruct
the full-compute output markedly better than what either
paradigm deploys (MagCache [30], static in effect, on FLUX;
SeaCache [7] on Wan), and the experiments quantify the mar-
gins. The search itself is far too expensive to run per prompt,
but its product, a full-horizon allocation of the budget, can be
obtained once, offline, and distilled into a policy cheap enough
to consult at every step. This is what BAG does: it keeps the
per-step decision form of the online paradigm and gives it the
budget awareness of the offline one, as a learned rule over an
explicit state,
zt = gθ
 sbud
t
, straj
t

,
(1)
whose two halves supply the two missing ingredients: global
resource context and local rollout context.
3.2. BAG: Budget-Aware Gating
BAG has three stages (Fig. 2): offline reference search, super-
vised distillation, and budget-exact inference.
Offline references. To supervise this rule, for each training
prompt p and budget B, a matched-NFE local search finds
the schedule that best preserves the full-compute output under
exactly B evaluations, the searched reference schedule
ˆm(p, B) ≈arg
min
∥m∥1=B D
 xT (m, p), xT (1, p)

,
(2)
4

Algorithm 1 BAG training (one backbone, run once)
1: Given: prompts P, budgets B, steps T
2: Initialize: example set D ←∅
3: for p ∈P do
4:
run the full-compute rollout
▷target xT (1, p)
5:
for B ∈B do
6:
ˆm ←matched-NFE search
▷Eq. (2), offline
7:
save sbud
t
, straj
t
along its last rollout
▷Eq. (3)
8:
D ←D ∪{(sbud
t
, straj
t
, ˆmt)}t≥1
9:
end for
10: end for
11: z-score inputs; re-weight the positive class
12: fit θ on D
▷Eq. (4), ≈1 min
13: Return: gate gθ
where D is the LPIPS distance. The search runs once per
backbone, at training time only; it holds B fixed and optimizes
only where computation is placed, so its objective is aligned
with full-compute fidelity. Because one gate later serves ev-
ery budget, each training prompt is searched at each training
budget, so the examples record how the allocation shifts as B
changes. Optimizer details and cost are given in App. A.1.
Budget and trajectory state. The gate’s state instantiates the
two contexts of Eq. (1) with six scalars. At step t, with ct
evaluations already spent, it reads
sbud
t
=
h
B
T ,
B−ct
B
,
B−ct
T −t
i
,
straj
t
=
h
t −tlast,
∥xt−xtlast∥
∥xtlast∥
,
∥xt−xt−1∥
∥xt−1∥
i
,
(3)
where tlast is the last computed step and xt the post-patch-
embedding token tensor (conditional branch for CFG mod-
els), the same class of observable that prior heuristics thresh-
old. The budget state expresses the overall compute ratio, the
remaining resource fraction, and the budget pressure over the
remaining horizon: it is what lets the gate pace spending glob-
ally, and, because its entries are ratios of step counts, the same
definition applies at other budgets and step counts. The tra-
jectory state carries cache staleness, cache drift, and the local
step change: together they measure how far the rollout has
moved since the last refresh and how fast it is moving now.
Each scalar is a counter or a single reduction over a tensor the
forward pass already produces, with no extra network evalua-
tion, and the gate learns their joint effect.
Supervised distillation. The per-step supervision comes out
of the search itself (Algorithm 1): its last round rolls out the
returned ˆm, and along this rollout we save the states at every
step, yielding examples D =

(sbud
t
, straj
t
, ˆmt)

over all
training prompts and budgets (t ≥1). The gate is trained by
per-step binary classification,
L(θ) = P
D BCE
 σ(zt), ˆmt

,
(4)
Algorithm 2 BAG budget-exact inference (one prompt)
1: Given: gate gθ, budget B, steps T, cutoff τ=0.5
2: Initialize: m0 ←1; computed count c ←1
3: for t = 1 to T −1 do
4:
read sbud
t
, straj
t
▷cheap online state, Eq. (3)
5:
mt ←I

σ(gθ(sbud
t
, straj
t
)) > τ

▷gate decision
6:
if B−c ≥T−t: mt ←1
▷spend remainder
7:
if B−c = 0: mt ←0
▷budget exhausted
8:
if mt=1: run network; refresh cache; c ←c+1
9:
else: reuse the cached residual
10: end for
11: Return: final sample
▷realized NFE = B exactly
where σ is the logistic sigmoid. The inputs are z-scored, and
the BCE loss is class-balanced. The budget features supply
the sequence context, so a sequence-level allocation problem
is reduced to per-step classification, and the global count
comes out exact at inference. The gate is a small MLP of
under 1K parameters, and one gate is trained per backbone in
about a minute on a single RTX 4090. Because deployment
states are induced by the gate’s own past decisions rather
than the reference rollouts, distillation incurs an off-policy
state-distribution shift.
The reported results are measured
under deployment and already include its cost; correcting
the shift with on-policy relabeling would multiply the offline
cost, so we leave it to future work. Architectures, optimizers,
and further discussion are in App. A.2.
Budget-exact inference. At deployment the gate runs inside
the sampling loop: whenever σ(zt) > τ (a fixed cutoff,
τ=0.5) the step computes and one evaluation is spent, and the
updated budget state enters the next step’s decision. Unlike
the thresholds of prior heuristics, τ does not set how much
is computed: the total is pinned by B itself.
A step with
the budget already spent can only reuse, and a rollout with
exactly as many steps left as budget can only compute them
all; at such steps there is nothing to decide, and at every other
step the gate’s decision stands (Algorithm 2).
Outside the
forced cases, the gate places evaluations from the budget and
trajectory state.
In the main comparisons, it spends all B
before the remainder rule activates and leaves a brief reuse
tail, as do most reference schedules.
The boundary rules
ensure an NFE of B without prescribing those placements, so
the requested budget directly sets the speedup.
4. Experiments
4.1. Setup
Backbones and data. We evaluate on three backbones. On
FLUX.1-dev [19] (50 steps, 10242), we train the gate on 120
GenEval prompts [11] (96 train, 24 validation), searching each
prompt at every budget in {7, 10, 13, 16, 19, 22, 25}, and test
on all 200 DrawBench prompts [36], a disjoint prompt source.
On Wan2.1-T2V-1.3B [41] (50-step UniPC [54], 832×480, 65
5

Method
NFE
Lat. (s)
Speed
PSNR↑
SSIM↑
LPIPS↓
FLUX.1-dev
50
31.13
1.00×
–
–
–
∼5× acceleration (B=9)
10 steps
10
6.23
4.99×
14.73
0.632
0.4823
TeaCache
10.6
6.85
4.54×
16.09
0.672
0.4266
MagCache
10.0
6.25
4.98×
20.03
0.745
0.3014
TaylorSeer
9.0
7.02
4.43×
15.44
0.643
0.4399
BudCache†
9.0
5.63
5.53×
19.49
0.727
0.3275
SeaCache
9.0
6.00
5.19×
19.66
0.755
0.3031
BAG (ours)
9.0
5.99
5.20×
21.27
0.766
0.2887
∼3.8× acceleration (B=13)
15 steps
15
9.33
3.34×
15.77
0.673
0.4118
TeaCache
14.7
9.38
3.32×
17.49
0.724
0.3395
MagCache
13.0
8.11
3.84×
21.70
0.812
0.2111
TaylorSeer
14.0
10.01
3.11×
18.29
0.744
0.2841
BudCache†
13.0
8.11
3.84×
21.55
0.805
0.2136
SeaCache
13.0
8.46
3.68×
21.85
0.821
0.2030
BAG (ours)
13.0
8.48
3.67×
24.46
0.849
0.1675
∼2.4× acceleration (B=20)
25 steps
25
15.49
2.01×
18.10
0.753
0.2929
TeaCache
21.0
13.28
2.34×
18.84
0.765
0.2744
MagCache
20.0
12.47
2.50×
25.74
0.892
0.1105
TaylorSeer
26.0
17.14
1.82×
23.50
0.870
0.1325
BudCache†
20.0
12.46
2.50×
27.52
0.903
0.0898
SeaCache
20.9
13.37
2.33×
27.81
0.914
0.0835
BAG (ours)
20.0
12.81
2.43×
29.18
0.918
0.0773
Table 1. Quantitative results on FLUX.1-dev. One BAG check-
point improves all three metrics at every acceleration tier. Methods
are compared at matched NFE (baselines spend at least as many eval-
uations), with wall-clock latency also reported. Bold: best per tier;
underline: second; top row: full-compute reference. †BudCache un-
der our search (see App. B).
frames), we train on 50 VBench [14] prompts, each searched
at {15, 18, 21, 24}, and test on 100 disjoint VBench prompts
sampled uniformly over all 19 categories. On Qwen-Image-
2512 [45] (50 steps, 10242), the FLUX protocol is reused un-
changed: the gate is trained on schedules searched for the
same 120 GenEval prompts and tested on DrawBench-200.
The gate is also evaluated at budgets it was not trained on:
B=9 and B=20 on FLUX, B=19 on Wan, and both Qwen-
Image budgets. Every BAG number is thus reported on held-
out prompts; the prompt-isolation protocol is in App. A.3.
Protocol. All methods are evaluated against the same-seed,
same-machine 50-step full-compute output of the same back-
bone, with per-prompt seeds. We report realized NFE (func-
tion evaluations per prompt), PSNR/SSIM/LPIPS, and la-
tency. Baseline thresholds are swept on the test set to match
the target NFE, a protocol that favors the baselines.
BAG
spends exactly B by construction, and unless stated other-
wise every BAG number comes from one gate per backbone
at τ=0.5, all budgets served by the same checkpoint.
Baselines.
We compare against TeaCache [23], Mag-
Cache [30], SeaCache [7] (the strongest heuristic overall
in our runs), TaylorSeer [24] (official implementation) as
Method
NFE
Lat. (s)
Speed
PSNR↑
SSIM↑
LPIPS↓
Wan2.1-1.3B
50
184.40
1.00×
–
–
–
∼3.4× acceleration (B=15)
TeaCache
17.0
62.95
2.93×
21.01
0.769
0.1827
MagCache
15.0
55.58
3.32×
20.31
0.746
0.2047
BudCache†
15.0
55.58
3.32×
24.06
0.835
0.1241
SeaCache
15.0
56.70
3.25×
22.99
0.805
0.1470
BAG (ours)
15.0
55.76
3.31×
24.96
0.848
0.1142
∼2.7× acceleration (B=19)
TeaCache
20.0
74.22
2.48×
23.01
0.826
0.1292
MagCache
19.0
70.57
2.61×
25.71
0.878
0.0891
BudCache†
19.0
70.47
2.62×
24.93
0.861
0.1010
SeaCache
19.0
72.29
2.55×
26.28
0.880
0.0871
BAG (ours)
19.0
70.68
2.61×
27.89
0.905
0.0685
∼2.1× acceleration (B=24)
TeaCache
25.0
93.09
1.98×
23.79
0.846
0.1110
MagCache
26.0
96.77
1.91×
26.96
0.906
0.0639
BudCache†
24.0
89.38
2.06×
26.07
0.886
0.0802
SeaCache
24.6
92.30
2.00×
30.72
0.943
0.0395
BAG (ours)
24.0
89.55
2.06×
31.38
0.947
0.0381
Table 2.
Quantitative results on Wan2.1-T2V-1.3B. One BAG
checkpoint improves all three metrics at every acceleration tier.
Methods are compared at matched NFE (baselines spend at least as
many evaluations), with per-video wall-clock latency also reported.
Bold: best per tier; underline: second; top row: full-compute refer-
ence. †BudCache under our search (see App. B).
the representative of the mechanism axis, and, on FLUX
and Qwen-Image, naive step reduction.
Tiers match the
∼5/3.8/2.4× (FLUX) and ∼3.4/2.7/2.1× (Wan) accelera-
tion factors targeted by prior caching work.
For fairness,
BudCache [20] uses our search on FLUX and Wan: we run
the schedule search behind our gate’s labels on BudCache’s
calibration prompt and broadcast the resulting schedule to
all test prompts, isolating its one-prompt-calibration choice
from search strength; on Qwen-Image it runs its official
search protocol, since there the official search spends more
rollouts per schedule than ours. Per-tier operating points and
BudCache’s official protocol are given in App. B.
4.2. Main Results
Quantitative comparison.
Tabs. 1 and 2 present the
matched-computation comparison against both paradigms.
At every tested budget on both backbones, BAG improves all
three reconstruction metrics over the strongest baseline using
the same or fewer NFEs. The PSNR margin over SeaCache,
the strongest online heuristic, reaches +2.6 dB on FLUX
and +2.0 dB on Wan, and LPIPS is reduced by 21–22%
relative to SeaCache at the two tighter Wan tiers. Against
the per-tier runner-up (MagCache at the tightest FLUX tier,
BudCache† at Wan B=15) the PSNR margin ranges from
+0.7 to +2.6 dB. Where SeaCache overshoots the tier, BAG
wins while spending less.
TaylorSeer sits on a different
design axis, forecasting cached features rather than changing
the schedule, and under this protocol it trails at every tier even
6

Figure 4. Qualitative results on FLUX.1-dev at the ∼5× (B=9) and ∼2.4× (B=20) tiers, with the 50-step original leftmost. At matched
compute the accelerated baselines drift in composition and object identity, while BAG stays closer to the full-compute output. More qualitative
results in App. D.
Figure 5. Qualitative results on Wan2.1 at the ∼3.4× tier (B=15,
six frames). At matched NFE BAG follows the full-compute motion
(the horse’s gait, the locomotive’s smoke) while SeaCache blurs fast
motion and drifts. More qualitative results in App. D.
when granted the next-higher point on its discrete NFE grid
(App. B). Tab. 4 reports the reference-free VBench dimen-
sions on Wan at the tightest tier, where every method stays
within 0.003 of the reference on the first four dimensions and
BAG has the highest average among the accelerated methods.
Tab. 3 extends the comparison to Qwen-Image-2512: the
gate again leads every reconstruction metric at both tiers
Method
NFE
Speed
PSNR↑
SSIM↑
LPIPS↓
Qwen-Image-2512
50
1.00×
–
–
–
∼5× acceleration (B=9)
9 steps
9.0
5.56×
14.05
0.549
0.4603
SeaCache
9.0
5.56×
16.70
0.579
0.4314
BudCache
9.0
5.56×
18.76
0.622
0.3543
BAG (ours)
9.0
5.56×
19.13
0.629
0.3120
∼3.2× acceleration (B=15)
15 steps
15.0
3.33×
16.03
0.657
0.3427
SeaCache
15.0
3.33×
20.60
0.769
0.2231
BudCache
15.0
3.33×
21.62
0.801
0.1804
BAG (ours)
15.0
3.33×
23.66
0.801
0.1759
Table 3. Quantitative results on Qwen-Image-2512. One BAG gate
leads all three metrics at both acceleration tiers, with methods com-
pared at matched NFE. Bold: best per tier; underline: second; top
row: full-compute reference.
under matched computation, with PSNR margins of +2.4 and
+3.1 dB over SeaCache, and naive step reduction trails every
caching method. VBench results for all three tiers, FLUX
preference metrics, and further analyses are in App. C.
Qualitative comparison. Fig. 4 compares FLUX outputs at
matched NFE: threshold schedules lose sign text, object iden-
tity, and scene layout, while BAG stays closer to the full-
compute output. Fig. 5 shows the corresponding Wan com-
parison at the tightest tier, and Fig. 1 shows Qwen-Image-
7

Method
Subject↑
