# Predict Before You Deploy: Offline Prediction of Quantization-Induced Task Degradation for World Action Models

paper_id: arxiv:2609.19441v1
tier: U
source_used: html_arxiv
warning: none

## Intro

World action models (WAMs) combine video generation and action modeling for robot control
[
1
,
2
,
3
,
4
,
5
]
, but their large backbones impose substantial memory and computational demands on deployment.
Post-training quantization (PTQ)
[
6
,
7
,
8
,
9
]
can reduce these demands without retraining the policy.
However, its deployment value depends on retaining adequate task performance.
Selecting a quantization configuration involves weight and activation precision, grouping, layer coverage, and quantizer design.
These choices produce candidates with different resource requirements and behavioral effects, even at the same nominal bit width.
Evaluating every candidate in closed loop reveals its task performance but requires repeated simulator or robot interaction.
We therefore ask:
can we predict whether a new quantization configuration will degrade task performance before evaluating it in closed loop?
Offline action comparison provides an accessible signal: replay identical observations through the reference and quantized policies and measure their action deviations.
However, task success depends on the consequences of action changes throughout a rollout, and deviation magnitudes need not have the same significance across policies.
Our experiments show sharply different outcomes for grouped and per-channel three-bit Cosmos configurations, as well as a cross-policy ordering reversal: at three bits, Fast-WAM suffers a larger task loss than Cosmos despite a smaller mean action deviation.
Within the initial Cosmos and UVA development sets, acceptable configurations nevertheless have smaller deviations than degraded configurations.
These observations motivate interpreting offline deviation through behavioral evidence specific to the policy and evaluation setting.
We propose PreDE (
Pre
dict Before You
DE
ploy), a policy-calibrated framework for predicting quantization-induced task degradation.
A small development set pairs offline action deviations with closed-loop outcomes to establish acceptance and rejection thresholds.
For each new candidate, PreDE compares its actions with cached reference actions on a fixed observation log.
It accepts candidates at or below the acceptance threshold, rejects those at or above the rejection threshold, and defers intermediate candidates to closed-loop evaluation.
After calibration, issuing predictions requires only offline policy queries.
Deferred candidates require further interaction if a deployment decision is needed.
We provide a post hoc formalization of this rule through a within-setting single-crossing hypothesis:
acceptable and degraded outcome labels can be separated by an unknown deviation threshold, while unresolved outcomes impose no constraint.
Under this hypothesis, the two thresholds delimit the interval of thresholds consistent with the development labels.
PreDE decides where these thresholds agree and defers where they disagree.
This interpretation explains the max/min construction and deferral interval, with conclusions conditional on the ordering hypothesis.
We evaluate PreDE through cross-model quantization measurements, held-out prediction, and physical robot experiments.
The broad evaluation spans five WAMs and four benchmark settings.
Across 20 Cosmos Policy and eight UVA held-out configurations, PreDE issues 21 decisions (75% coverage), all matching the observed acceptable or degraded outcome labels.
These predictions are determined before observing candidate closed-loop outcomes.
Deferred candidates include both acceptable outcomes and a 33-percentage-point loss, demonstrating that deferral does not imply mild degradation.
In a separate 450-trial Franka Research 3 experiment, all tested configurations assigned to high-deviation groups before trials show significant degradation.
This physical grouping study provides complementary behavioral evidence without transferring the simulation thresholds.
On the robot, W4A4 quantization of Cosmos Policy provides a
1.37
×
1.37\times
action-query speedup and approximately 44% lower peak memory.
Our contributions are threefold:
•
We propose PreDE for predicting quantization-induced task degradation from offline action deviations, and formalize its acceptance, rejection, and deferral regions under a within-setting label-ordering hypothesis.
•
We evaluate predictions on 28 held-out configurations from two WAM policies, with decisions determined before observing closed-loop results, and report prediction results, decision coverage, label ordering, and measurement sensitivity.
•
We characterize quantization outcomes across five WAMs and four benchmark settings, and present a separate real-robot study of behavioral degradation and the latency and memory benefits of quantization.

## Method

Fig. 1:
Overview of PreDE.
(a) Closed-loop outcomes and offline action deviations of development configurations establish policy-specific
acceptance and rejection thresholds.
(b) New candidates are compared with cached reference
actions on a fixed observation log and accepted, rejected,
or deferred.
(c) A separate real-robot experiment evaluates deviation groups fixed before trials.
IV-A
Problem formulation and offline measurement
Given a reference policy
π
\pi
, a quantization configuration
q
q
specifies precision, grouping, layer coverage, and quantizer, producing a quantized policy
π
q
\pi_{q}
.
Under a fixed task distribution and control condition, let
p
0
p_{0}
and
p
q
p_{q}
denote their closed-loop success probabilities, with task loss
L
⁡
(
q
)
=
p
0
−
p
q
L(q)=p_{0}-p_{q}
.
The condition fixes observation preprocessing, action parameterization, chunk length, execution rate, and denoising steps.
Our goal is to predict degradation for a new configuration before evaluating it in closed loop.
PreDE uses a fixed observation log
D
D
and a development set
C
C
of quantized configurations with measured closed-loop outcomes (Fig.
1
).
For each observation in
D
=
{
o
i
}
i
=
1
M
D=\{o_{i}\}_{i=1}^{M}
, the reference and candidate receive identical inputs and matched action-generation seeds.
Let
a
i
​
h
​
j
ref
a^{\mathrm{ref}}_{ihj}
and
a
i
​
h
​
j
q
a^{q}_{ihj}
denote their actions at chunk step
h
h
and component
j
j
.
We measure mean absolute component deviation:
δ
¯
​
(
q
,
D
)
=
1
M
​
H
​
d
​
∑
i
=
1
M
∑
h
=
1
H
∑
j
=
1
d
|
a
i
​
h
​
j
q
−
a
i
​
h
​
j
ref
|
,
\bar{\delta}(q;D)=\frac{1}{MHd}\sum_{i=1}^{M}\sum_{h=1}^{H}\sum_{j=1}^{d}\left|a^{q}_{ihj}-a^{\mathrm{ref}}_{ihj}\right|,
(2)
where
H
H
is the chunk length and
d
d
is the number of included action components.
The action representation and included components remain fixed within each calibration setting.
Reference actions are cached, so a new candidate requires
M
M
complete policy queries without environment interaction.
IV-B
Policy-specific calibration
For each
q
∈
C
q\in C
, we measure
δ
¯
​
(
q
,
D
)
\bar{\delta}(q;D)
and the observed task loss
L
^
​
(
q
)
=
p
^
0
−
p
^
q
\widehat{L}(q)=\hat{p}_{0}-\hat{p}_{q}
, using reference and candidate evaluations under matched conditions.
Before calibration, we specify an acceptable observed-loss tolerance
ϵ
A
\epsilon_{A}
, a degradation threshold
ϵ
R
>
ϵ
A
\epsilon_{R}>\epsilon_{A}
, and an evidence criterion
E
⁡
(
q
)
E(q)
.
These define
C
A
\displaystyle C_{A}
=
{
q
∈
C
:
L
^
​
(
q
)
≤
ϵ
A
}
,
\displaystyle=\{q\in C:\widehat{L}(q)\leq\epsilon_{A}\},
(3)
C
R
\displaystyle C_{R}
=
{
q
∈
C
:
L
^
​
(
q
)
≥
ϵ
R
∧
E
⁡
(
q
)
=
1
}
.
\displaystyle=\{q\in C:\widehat{L}(q)\geq\epsilon_{R}\land E(q)=1\}.
(4)
Configurations in
C
A
C_{A}
are
acceptable
, those in
C
R
C_{R}
are
degraded
, and all others are
unresolved
.
The same labeling rule applies to held-out outcomes.
These labels describe finite-sample observations: an acceptable label does not establish a population loss bound, and failure to detect degradation does not establish an acceptable outcome.
Our simulation prediction experiments use
ϵ
A
=
0.01
\epsilon_{A}=0.01
and
ϵ
R
=
0.02
\epsilon_{R}=0.02
, corresponding
to one and two percentage points.
These specify the acceptable observed-loss tolerance
and the minimum loss required for a degraded label,
respectively, and were fixed before held-out evaluation.
A loss exceeding
ϵ
R
\epsilon_{R}
is not sufficient for
a degraded label: it must also satisfy the evidence
criterion
L
^
​
(
q
)
>
2
​
p
^
0
​
(
1
−
p
^
0
)
n
0
+
p
^
q
​
(
1
−
p
^
q
)
n
q
,
\widehat{L}(q)>2\sqrt{\frac{\hat{p}_{0}(1-\hat{p}_{0})}{n_{0}}+\frac{\hat{p}_{q}(1-\hat{p}_{q})}{n_{q}}},
(5)
where
n
0
n_{0}
and
n
q
n_{q}
are the reference and candidate
episode counts.
Thus, the evidence requirement depends on the observed
success rates and sample sizes, rather than a fixed
minimum detectable loss.
We compute the max/min thresholds in Eq. (
1
) from these development labels.
The acceptance threshold is the largest deviation among acceptable configurations, while the rejection threshold is the smallest among degraded configurations.
Unresolved outcomes establish neither threshold.
When both subsets are nonempty and
δ
acc
<
δ
rej
\delta_{\mathrm{acc}}<\delta_{\mathrm{rej}}
, they define the consistent-threshold interval
Θ
⁡
(
C
)
=
[
δ
acc
,
δ
rej
)
\Theta(C)=[\delta_{\mathrm{acc}},\delta_{\mathrm{rej}})
of Sec.
III
.
PreDE calibrates the behavioral interpretation of deviation.
The underlying PTQ method determines the quantizer’s parameters.
Using actual quantized configurations connects the measurement to observed task outcomes without assuming that synthetic action noise reproduces quantization effects.
Hypothesis H formalizes the ordering needed for the conditional interpretation in Proposition
1
.
Its applicability to new configurations is evaluated through held-out predictions.
IV-C
Prediction and evaluation
We freeze the log, measurement protocol, and thresholds before closed-loop evaluation of new configurations.
For
q
∉
C
q\notin C
, PreDE computes its offline deviation and issues
g
⁡
(
q
)
=
{
accept
,
δ
¯
​
(
q
,
D
)
≤
δ
acc
,
reject
,
δ
¯
​
(
q
,
D
)
≥
δ
rej
,
defer
,
δ
acc
<
δ
¯
​
(
q
,
D
)
<
δ
rej
.
g(q)=\begin{cases}\mathrm{accept},&\bar{\delta}(q;D)\leq\delta_{\mathrm{acc}},\\
\mathrm{reject},&\bar{\delta}(q;D)\geq\delta_{\mathrm{rej}},\\
\mathrm{defer},&\delta_{\mathrm{acc}}<\bar{\delta}(q;D)<\delta_{\mathrm{rej}}.\end{cases}
(6)
If either development subset is empty or
δ
acc
≥
δ
rej
\delta_{\mathrm{acc}}\geq\delta_{\mathrm{rej}}
, PreDE defers all candidates.
Acceptance predicts an acceptable outcome, while rejection predicts degradation.
Under H, Proposition
1
excludes the opposite definite label for each issued decision, but it does not exclude an unresolved outcome.
Deferral makes no outcome prediction and calls for closed-loop evaluation if a deployment decision is needed.
This follows the selective prediction principle of withholding decisions when evidence is insufficient rather than deciding on every candidate
[
23
]
.
Predictions are determined before observing candidate closed-loop outcomes.
Decision coverage is the fraction of held-out candidates receiving accept or reject, including those whose eventual outcomes remain unresolved.
We score predictions against the labels defined by Eqs. (
3
)–(
4
).
Accepting a degraded configuration is a false acceptance, while
rejecting an acceptable configuration is a false rejection.
Accepting an acceptable configuration or rejecting a
degraded configuration counts as a correct decision.
An issued decision with an unresolved outcome counts
as neither correct nor incorrect.
Deferred outcomes are reported but not scored.
Evaluated configurations may subsequently enter
C
C
to update the anchors, as described in
Proposition
2
.
Once used for recalibration, they become development
evidence, and the updated rule requires new held-out
evaluation.
This separates initial calibration, offline prediction,
and further interaction for deferred candidates.
TABLE I:
Closed-loop success rates (%) for 33 quantized
model–benchmark pairs.
W8A8 uses ViDiT-Q-based quantization. W4A4 uses
SVDQuant-style quantization. W4 and W3 use HQQ
weight-only quantization.
In an additional single-seed Cosmos/LIBERO development
comparison, RTN W3 achieves 97.40% success with group
size 128 and 20.05% with per-channel quantization.
Model
Benchmark
bf16
W8A8
W4A4
W4
W3
Cosmos-2B
LIBERO (four suites)
98.38
98.12
97.98
98.20
97.87
Cosmos-2B
RoboCasa-24
67.03
66.42
65.58
65.58
61.61
UVA-0.5B
LIBERO-10
89.60
91.20
89.20
88.40
78.60
UVA-0.5B
PushT
97.47
99.61
95.79
96.02
96.79
UVA-0.5B
PushT-M
82.71
75.07
78.66
81.77
77.71
mimic-2B
LIBERO-spatial
90.40
91.40
88.30
92.60
90.20
mimic-2B
LIBERO-object
94.20
92.40
89.60
93.80
95.20
RynnVLA-7B
LIBERO (four suites)
95.30
—
—
95.05
94.75
Fast-WAM-6B
LIBERO (four suites)
95.55
—
95.75
95.55
90.15
