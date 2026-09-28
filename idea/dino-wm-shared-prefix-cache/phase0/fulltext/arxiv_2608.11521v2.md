# Keep the Future, Drop the Rollout: RIFT for World Action Models

paper_id: arxiv:2608.11521v2
tier: T3
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

World action models (WAMs) couple future video prediction
with robot control. A single network predicts a short video
and conditions its next actions on that prediction (Yuan et al.
2026; Li et al. 2026; Bi et al. 2025). In matched settings,
policies that retain this future read achieve higher success
than current-only variants. However, iterative video gener-
ation makes rollout-based systems incur 3.3× to 9.6× the
latency of current-only deployment (fig. 1).
Existing efficient variants remove the future read at de-
ployment. Fast-WAM drops the future branch after future-
prediction co-training (Yuan et al. 2026), whereas PFD dis-
tills a future-conditioned correction into the current-only
path (Fang, Chen, and Cai 2026). Both reduce latency but
retain a gap to policies that explicitly read a generated fu-
ture. Yet the action expert consumes a future representa-
tion, whereas iterative rollout is the process that constructs
it. Existing comparisons change both the availability of this
250
500
1000
2000
Latency / action chunk (ms, log scale)
96.5
97.0
97.5
98.0
98.5
99.0
LIBERO success (%)
future read + rollout
no future read
RIFT: future read, no rollout
Fast-WAM
PFD
Fast-WAM-Joint
Cosmos Policy
Fast-WAM-IDM
LingBot-VA
RIFT (ours)
Figure 1: LIBERO success versus deployment latency.
Points show mean success and bars show ±std over three

## Method

Emb.
Fut.
Roll-
SR
Lat.
Rel.
PT.
read
out
(%)
(ms)
Fast-WAM†
×
×
×
96.8 ±0.27
235.7 1.0×
PFD†
×
×
×
97.3 ±0.12
257.0 1.1×
Fast-WAM-Joint∗
×
✓
✓
98.4 ±0.26
780.2 3.3×
Fast-WAM-IDM∗
×
✓
✓
98.6 ±0.34 1081.2 4.6×
LingBot-VA†
✓
✓
✓
98.5 ±0.08 2270.3 9.6×
Rift (ours)
×
✓
×
98.8 ±0.17
247.9 1.1×
0
20
40
60
80
Success rate (%)
Fast-WAM
Fast-WAM-Joint
Fast-WAM-IDM
RIFT
49.7
68.1
71.4
81.1
Figure 4: Overall LIBERO-Plus OOD robustness. Suc-
cess across all 10,030 variants, with one rollout per variant
and no further training; bars are point estimates.
5.2
Quantitative results
LIBERO.
In Table 1, Rift achieves 98.8% overall success,
close to the 98.4% to 98.6% achieved by rollout-based Joint,
IDM, and LingBot-VA. Unlike these methods, Rift requires
only 247.9 ms per action chunk, reducing latency by 68.2%
to 89.1% while remaining close to current-only Fast-WAM at
235.7 ms. Compared with rollout-free Fast-WAM and PFD,
Rift improves success by 2.0 and 1.5 percentage points,
respectively, at comparable latency.
LIBERO-Plus OOD.
Across four checkpoints and 10,030
variants, Rift achieves the highest overall success rate of
81.1%, a +9.7 percentage-point gain over Fast-WAM-IDM
(fig. 4). This shows robustness to OOD perturbations. See
table 5 for the full breakdown.
RoboTwin 2.0.
The interface transfers to a second em-
bodiment (table 2; per-task rates in table 6). Rift reaches
92.9/92.6 on clean/randomized scenes, the best observed
among the evaluated methods, against 92.5/92.1 for PFD,
92.4/91.4 for rollout-based LingBot-VA, 91.9/91.6 for Fast-
WAM, and 91.0/91.1 for rollout-based Fast-WAM-Joint. We
observe the same performance recovery on RoboTwin 2.0
while retaining the one-pass deployment path.
5.3
Ablations
Anticipation-token supervision.
The base recipe Rift-L2
regresses future latents with a direct L2 loss and reaches
98.37 ±0.12; the conditional-FM recipe reaches 98.8 ±0.17.

Table 2: RoboTwin 2.0 closed-loop success (%) on clean and
domain-randomized scenes.
Values aggregate one check-
point per method; This report follows single-seed evaluation
protocol. Emb. PT.: embodied pretraining.
Method
Emb.
Fut.
Roll-
Clean Rand. AVG
PT.
read
out
LingBot-VA
✓
✓
✓
92.4
91.4
91.9
Fast-WAM-Joint
×
✓
✓
91.0
91.1
91.0
Fast-WAM
×
×
×
91.9
91.6
91.8
PFD
×
×
×
92.5
92.1
92.3
Rift (ours)
×
✓
×
92.9
92.6
92.8
Both use the same one-pass graph and 247.9 ms cost, iso-
lating supervision without deployment overhead. The 0.4
point difference approaches the evaluation’s resolution, and
the full recipe includes its conditioning curriculum. Table 4
gives per-suite rates; we report FM as the recipe.
The number of anticipation tokens.
Figure 5 sweeps m =
2 to full alignment (m = 196); current-only Fast-WAM is
the no-cache m = 0 reference (96.75%).
Rift-L2 rises
from 97.08% to 98.37%; conditional FM exceeds it from
m = 4 and peaks at 98.78%. Even small interfaces beat the
