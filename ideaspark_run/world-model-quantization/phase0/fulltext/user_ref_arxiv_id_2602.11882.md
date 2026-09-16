# [user-supplied anchor] 2602.11882

paper_id: user_ref:arxiv_id:2602.11882
tier: U
source_used: html_arxiv
warning: none

## Intro

World-model planning has become a strong paradigm for sample-efficient control and spatial reasoning, especially when latent dynamics are built on pretrained visual features
(
Hafner et al., 2023
;
Oquab et al., 2023
;
Zhou et al., 2024
)
.
Deployment constraints, however, force these models into tight memory and latency budgets where aggressive quantization can destabilize behavior.
Recent broad surveys and world-model-specific studies report that low-bit degradation is often non-uniform across modules
(
Liu et al., 2025
;
Fu et al., 2026
)
.
We ask whether, near low-bit operating points, planning quality is driven more by average precision or by how bits are allocated between the encoder and predictor. To answer this, we evaluate DINO-WM on the Wall task using paired-goal tests across a set of mixed-bit variants. Our empirical contributions identify when quantization is safe, when it degrades planning, and which allocation choices preserve performance under tight budgets. Concretely, we report three findings: (1) a three-regime pattern in which 8/6 bits remain similar to FP16, 4 bits form a sensitive transition, and 3 bits collapse; (2) the mixed-vs-uniform INT4 direction holds across two planner budgets and difficulty slices; and (3) asymmetric and layerwise ablations implicate the encoder as a primary sensitivity locus.

## Method

A three-regime pattern appears in this environment/checkpoint.
Figure
1
and Table
1
show a stable regime at 8/6 bits, a transition at 4 bits, and collapse at 3 bits.
At budget bA, FP16, uniform INT8, mixed INT8, and uniform INT6 all reach 0.533 success.
At budget bB, FP16 and uniform INT6 both reach 0.650.
In contrast, all 3-bit variants are 0.0 at both budgets.
4-bit is allocation-sensitive.
At 4 bits, mixed INT4 is higher than uniform INT4: 0.267 vs 0.067 at bA, and 0.500 vs 0.200 at bB.
Paired analysis gives +0.20 at bA (95% CI [0.00, 0.40]) and +0.30 at bB ([0.00, 0.55]), with sign-test
p
=
0.109
p=0.109
in both budgets.
We therefore interpret the 4-bit effect as
directional
rather than statistically definitive (paired detail in Appendix Table
2
and forest view in Figure
8
).
Synchronous strict-run update.
In the strict replication (22 cells, 66 episodes), the 8-bit-versus-4-bit regime split remains: FP16/mixed INT8/uniform INT8 are high, while 4-bit variants are lower overall.
However, mixed-vs-uniform INT4 is budget-conditioned in this smaller run: at bA, mixed INT4 exceeds uniform INT4 (0.333 vs 0.167; paired delta +0.167), while at bB, mixed INT4 is lower (0.000 vs 0.167; paired delta -0.167).
Episode-level matchup counts are balanced in pooled view (1 mixed-only win vs 1 uniform-only win; Appendix Table
3
).
We interpret this as a sensitivity signal near the 4-bit frontier: allocation matters, but direction can flip with budget under low sample size.
Fairness check: allocation versus total precision budget.
Mixed INT4 (138.84 MB) is substantially larger than uniform INT4 (68.12 MB), so mixed-vs-uniform alone does not isolate allocation from total precision.
To partially control this, we compare near-size asymmetric variants at bA: E6/P4 (73.19 MB, success 0.300) and E8/P4 (78.25 MB, 0.233) both improve over uniform INT4 (68.12 MB, 0.067), while remaining far smaller than mixed INT4.
This suggests encoder-side bits are useful even under tight size budgets, while higher total precision (for example uniform INT6 at 77.92 MB, success 0.533) remains another strong driver (size-aware context in Appendix Table
4
and map in Figure
7
).
Asymmetric and layerwise evidence supports encoder sensitivity.
At bA, E6/P4 reaches 0.300 success at 73.19 MB, improving over uniform INT4 (0.067 at 68.12 MB), with paired delta +0.233 (95% CI [0.033, 0.433]).
By contrast, increasing predictor bits while keeping encoder at INT4 (E4/P8, E4/P6) does not match mixed INT4.
Numerically, E4/P8 is 0.133 with paired delta
−
0.133
-0.133
relative to mixed INT4, suggesting that spending extra bits on the predictor cannot compensate for encoder degradation at this frontier.
Layerwise INT4 ablation (Appendix Figure
3
) is non-monotonic but peaks when the encoder is fully preserved (0.25 at 100% FP16), supporting the hypothesis that encoder precision is often a key bottleneck near the 4-bit frontier.
Mechanistic diagnostics are consistent with representation degradation.
Across 65 run-level points (variant
×
\times
budget
×
\times
seed), success correlates negatively with divergence metrics from planning logs: Spearman
ρ
=
−
0.928
\rho=-0.928
for mean state distance and
ρ
=
−
0.708
\rho=-0.708
for visual-embedding divergence (Appendix Figure
5
).
This does not prove causality, but it supports a plausible mechanism: low-bit encoder degradation weakens latent geometry used for goal-directed planning.
Variant
Success bA
Success bB
Size (MB)
FP16
0.533
0.650
204.99
Uniform INT6
0.533
0.650
77.92
Mixed INT6
0.533
0.700
143.58
Uniform INT4
0.067
0.200
68.12
Mixed INT4
0.267
0.500
138.84
Uniform INT3
0.000
0.000
63.23
Mixed INT3
0.000
0.000
136.47
Table 1:
Core mixed-bit outcomes from the primary paired run. Means are shown; uncertainty and paired statistics are reported in Appendix Tables
2
and
4
.
Figure 1:
Success–size Pareto frontier for the paired mixed-bit study (budgets bA and bB). Each point is one variant, with vertical bars showing run-level 95% confidence intervals over seeds. Stars denote non-dominated Pareto points (higher success, lower model size). The frontier shows a stable 8/6-bit region, an allocation-sensitive 4-bit transition, and a collapsed 3-bit region.
