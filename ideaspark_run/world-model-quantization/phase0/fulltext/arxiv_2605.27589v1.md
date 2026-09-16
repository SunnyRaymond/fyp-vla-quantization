# What-If World: A Causal Benchmark for General World Models in Embodied Scenarios

paper_id: arxiv:2605.27589v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Video generation models are increasingly framed as general-purpose world simulators for embodied applications such as autonomous driving and robotic manipulation
[
24
,
45
]
. In these settings, visual realism is typically not enough, and neither is isolated physical plausibility
[
10
,
28
]
. A model can render a car braking hard with flawless kinematics yet produce nearly identical footage when the prompt is changed to a gentler brake. It has captured the visual template of “a car braking” but remains insensitive to the input that should shape the outcome. For action-conditioned simulators, model-based planners, and policy training pipelines, this is the failure that matters most, since rollouts that fail to track the controlled variable cannot be used to compare alternative actions or conditions.
This failure mode is inherently comparative: a single video rarely reveals whether the model would have produced the same footage under a different input. Yet existing benchmarks evaluate each video in isolation. Video-quality benchmarks such as VBench
[
29
]
and EvalCrafter
[
40
]
, together with physics-focused extensions like VideoPhy
[
8
]
and PhyGenBench
[
42
]
, score each generation against a single prompt, so two near-identical videos for "brake gently" and "brake hard" would both pass. Causal benchmarks for visual understanding, including CLEVRER
[
74
]
and NExT-QA
[
71
]
, instead test whether a model perceives causal structure in a video shown to it—not whether the videos it generates track changes in its input. Neither line of work probes the property at stake: that a controlled change in input should yield a correspondingly controlled change in output.
Probing this property in video generation models requires
formalization across three pieces, namely benchmark dimensions,
construction pipeline, and evaluation protocol. First, no prior
work defines what counts as a valid physical intervention
variable in embodied video generation, so the test space must
be established before any selection. Second, controlled
comparison is fragile in a generative setting, where the model
is text-driven yet must produce a physical effect, and
ceteris
paribus
must be approximated through input design rather than
enforced through generation. Third, the evaluation faces two
intertwined difficulties. Causal sensitivity is not a single
property, since a pair can fail in several ways. The intervention
may not have executed, the videos may execute but not diverge,
or the divergence may reflect scene drift rather than the
intervention. Without separating these modes, the evaluation
cannot tell what went wrong. Moreover, these judgments require
semantic reasoning that feature-distance metrics cannot capture,
while the benchmark’s scale rules out human annotation.
To fill these gaps, we build
What-If World
, a benchmark for contrastive intervention in autonomous driving and robotic arm manipulation. For test dimensions, two domain experts apply the three criteria above to real videos and merge survivors by causal mechanism (e.g., braking and acceleration intensity collapse into a single Force/Degree primitive), yielding six primitives across two domains: environment (surface friction, material/medium, obstacle configuration) and agent action (spatial alignment, force/degree, temporal sequencing). For test constructions, each instance is built in three steps.
(1) We filter clips from nuScenes
[
14
]
or DROID
[
32
]
for the
physical
conditions
the target intervention needs. (2) We extract
the frame just before the action begins and pair it with two scene-describing prompt
components (camera perspective and initial scene state),
together anchoring the shared initial state. (3) We author
a prompt pair that differs in exactly one semantic variable, so that any divergence between the two videos
traces back to the intervention.
For evaluations, we introduce the
APEO
framework, scoring each pair on Adherence (intervention executed), Physics (dynamics valid), Environment (scene held still), and Outcome (result diverged as predicted); the first three are scored in both single-video and paired modes, while Outcome is paired-only. The rubric runs at benchmark scale via a VLM judge validated against human annotation.
Applying this protocol to nine state-of-the-art video generation world models (four open-source, five closed-source) with Gemini 3.1-Pro as the VLM judge, we report three findings. First, causal sensitivity in video generation. No model exceeds 52% on our paired metric, and the four open-source models cluster near 28%. Second, most models show a wide gap between per-video and paired evaluation scores, generating pairs of individually convincing videos that fail to differ in the way the input predicts. We call this gap the contrastive bottleneck. Third, where models do score well, the pattern is more consistent with the visual prominence of the intervention’s consequences than with generalizable physical sensitivity—Force/Degree (producing dramatic motion differences) reaches 40.4% while Surface Friction (producing subtle trajectory differences) drops to 14.2%.
In summary, our contributions are as follows.
1.
The What-If World benchmark
, the first contrastive-intervention benchmark for video world models in embodied domains, contributing a six-primitive taxonomy of physical intervention variables that unifies autonomous driving and robotic manipulation, instantiated as 319 paired triplets anchored on real frames from nuScenes and DROID.
2.
The APEO evaluation framework
, a paired protocol scoring Adherence, Physics, Environment, and Outcome in both single-video and paired modes, with a VLM-based implementation validated against human annotation.
3.
A diagnostic study of nine state-of-the-art video generation models
, which surfaces the failure modes characterized above and provides the first quantitative evidence that they are invisible to existing per-video evaluation.

## Method

What-If World aims to benchmark a specific capability: whether a video generation model can simulate the differential physical consequences of a controlled intervention, rather than merely render visually plausible footage. Formally, a model
ℳ
\mathcal{M}
takes an initial frame
x
0
x_{0}
and a text prompt
p
p
to generate a video
𝒱
=
ℳ
⁡
(
x
0
,
p
)
\mathcal{V}=\mathcal{M}(x_{0},p)
. We define the capability under test as
causal intervention
as follows: given a shared
x
0
x_{0}
and a pair of prompts
(
p
+
,
p
−
)
(p^{+},p^{-})
that differ only in the description of one physical variable
v
v
, the model
ℳ
\mathcal{M}
should produce two videos
(
𝒱
+
,
𝒱
−
)
(\mathcal{V}^{+},\mathcal{V}^{-})
that diverge in the direction physics predicts under the intervention. Building this benchmark requires answering three questions:
•
What to test
(Section
3.1
): which physical variables to put under intervention. We construct a six-primitive taxonomy that spans agent actions and environment conditions, retaining only variables that are physically fundamental, embodiment-shared, and operationally isolable.
•
How to test
(Section
3.2
): how to construct each test pair so the divergence can be attributable to the intervened variable. We anchor every test on a real frame
x
0
x_{0}
at the causal branching point and write a contrastive prompt pair that differs only in the description of the control.
•
How to evaluate
(Section
3.3
): how to score the pair jointly. Our APEO framework checks four dimensions (Adherence, Physics, Environment, and Outcome) in both single-video and paired modes, so per-video plausibility is separated from causal correctness.
Figure 1
:
What-If World benchmark overview. Three stages address the challenges of evaluating the causal sensitivity of video world models: (1) what to test, (2) how to test, and (3) how to evaluate.
3.1
The What-If World Causal Taxonomy
To address the
What to test
challenge, we need to identify the variables that govern physical outcomes. A taxonomy of such variables is not just a descriptive convenience but a prerequisite of our evaluation protocol for two reasons: (1) The contrastive pair
(
p
+
,
p
−
)
(p_{+},p_{-})
is only well-defined once we name the variable being changed; without an explicit set of variables to draw from, “changing one physical variable” has no operational meaning. (2) Aggregate scores become actionable diagnostics only when each failure can be traced to a specific variable. A taxonomy lets us report which class of physical phenomena a model fails on, rather than a single opaque “physics” score.
As discussed in the background, existing benchmarks do not supply such a space. We therefore construct a taxonomy of physical primitives, grounded in real-world data
[
11
]
. Two domain experts (one in AD, one in manipulation) randomly selected and reviewed 1,000 real videos from nuScenes
[
14
]
and DROID
[
32
]
to identify variables that change physical outcomes. Candidate variables were retained only if they were: (i)
physically fundamental
, driven by a real physical mechanism rather than a stylistic or perceptual change
[
49
,
31
]
; (ii)
embodiment-shared
, applicable to both driving and robotics so that performance reflects general physical sensitivity rather than domain-specific shortcuts
[
46
,
19
]
; and (iii)
operationally isolable
, meaning we can change this one variable while holding everything else fixed, which is the standard requirement for controlled causal evaluation
[
2
,
51
]
. The experts then merged candidates.
For example, braking intensity and acceleration intensity look like opposite actions and a surface-level taxonomy would keep them apart, but they share the same underlying mechanism: a continuous scalar on the action side (how hard) whose value monotonically determines the outcome until it crosses a categorical boundary (e.g., stopping in time versus not). We therefore merge them into a single Force/Degree primitive. Applying this mechanism-based logic to all candidates and organizing the resulting primitives along the natural cause-and-effect structure of an embodied event yields a two-domain taxonomy: properties of the surrounding environment (
𝒟
env
\mathcal{D}_{\text{env}}
) and parameters of the agent’s action (
𝒟
int
\mathcal{D}_{\text{int}}
), with three primitives in each.
𝒟
env
\mathcal{D}_{\text{env}}
— Environment Domain.
External conditions that change what a given action produces. (1)
Surface Friction
determines whether contact yields traction or slip: hard braking decelerates cleanly on dry asphalt but skids on ice. (2)
Material & Medium
sets the resistance and deformability of the bodies and the medium they move through, which shapes how force translates into motion, so that the same gripper closure holds a rigid block but crushes a sponge. (3)
Obstacle Configuration
fixes what space the agent can actually move into, where a single misplaced cone flips the outcome from clean passage to collision.
𝒟
int
\mathcal{D}_{\text{int}}
— Interaction Domain.
Parameters of the action itself, holding the environment fixed. (1)
Spatial Alignment
is where the action is applied: small positional errors separate a clean merge from a sideswipe, or a grasp from a miss. (2)
Force/Degree
is how hard or how far (force, speed, or angular extent), and scales the outcome continuously until it crosses a categorical boundary, such as stopping in time versus not, or holding versus dropping. (3)
Temporal Sequencing
is when each sub-action fires relative to the others; closing the gripper before rather than after reaching the target turns the same motion into a grasp or a miss.
Table
3
instantiates each primitive in both domains. Each benchmark instance varies exactly one primitive under unanimous expert admission, so that a failure can be attributed to a specific reasoning deficit rather than to poor physics in general.
3.2
Benchmark Construction
Figure 2
:
Benchmark construction pipeline. Stage 1 filters clips from nuScenes and DROID and extracts the causal branching point
x
0
x_{0}
at action onset. Stage 2 writes a contrastive prompt pair
(
p
+
,
p
−
)
(p^{+},p^{-})
that shares scene context and differs only in one physical variable, with the outcome withheld. Each triplet
(
x
0
,
p
+
,
p
−
)
(x_{0},p^{+},p^{-})
is fed to a video model to produce
(
V
+
,
V
−
)
(V^{+},V^{-})
.
To solve the
How to test
challenge, we design each test instance as a controlled probe of the model’s world-simulator capability from a
shared initial state, a change to one physical variable should drive the simulated rollout to diverge as physical law predicts. Two
requirements
[
49
,
51
]
support
the interpretability of this probe. The first is state fixation:
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
should originate from the same
initial state, so that observed divergence can be attributed to
v
v
rather than to differing initial conditions. The second is
single-variable change: only one key variable should differ between
the two prompts, so that divergence can be cleanly traced to a
specific mechanism. Existing video-generation
benchmarks
[
29
,
8
,
42
,
36
]
satisfy neither and only test each video models against a single prompt in isolation.
We meet both requirements with a two-stage construction
(Figure
2
): (1) selecting a real clip and extracting
a frame
x
0
x_{0}
that, together with a fixed scene-state
description, anchors the initial state, and (2) formatting a paired
prompt
(
p
+
,
p
−
)
(p_{+},p_{-})
with only semantic different in target
variables.
Anchoring the state:
x
0
x_{0}
at the causal branching point.
We anchor the initial state through a paired specification: a single image frame
x
0
x_{0}
together with the
camera perspective
and
initial scene state
components of the prompt. The frame fixes what is visually observable (geometry, materials, agent positions, lighting), while the two prompt components supply information that a single frame cannot encode: how the camera will behave over time (e.g.,
“stationary dashboard camera”
) and dynamic context such as the ego vehicle’s speed or its motion relative to other agents. This split mirrors how current video-generation world models are conditioned (image as visual anchor, text as the source of non-visual initial information), and the three together form the shared initial conditioning that we hold fixed across
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
.
Frames are drawn from nuScenes
[
14
]
(autonomous driving) and DROID
[
32
]
(robotic arm manipulation). Within these datasets, a clip is eligible only if it carries the
physical affordances
that make a target primitive testable, such as the ego car driving side-by-side with another vehicle in an overtaking scenario for
Surface Friction
, or an unoccluded gripper near a target object for manipulation primitives. Domain-specific visibility filters further remove parked ego vehicles, heavy occlusion, and pre/after-contact gripper states, so that any divergence between
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
remains visible against an uncluttered scene.
From each qualified clip we take the frame immediately preceding action onset as the visual component of
x
0
x_{0}
, and we refer to this moment as the
causal branching point
. At this moment,
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
can start from an identical state (state fixation), and at the same time, the intervention
v
v
has not yet materialized, so
v
v
retains room to change the future (we call this
physical affordance
). In the overtaking scenario, for instance,
x
0
x_{0}
is the exact moment the ego car and an adjacent vehicle travel side-by-side at identical speeds, where the future trajectory depends entirely on the subsequent action. See Appendix
A.4
for more details.
Isolating the variable: the contrastive prompt pair.
Since text is the primary control input that current video-generation world models accept, we implement the contrastive design in the prompt pair. With
x
0
x_{0}
and the first two prompt components already fixed by the shared initial conditioning above, the only span left for editing is the agent action description. The intervention prompt
p
−
p_{-}
is obtained by editing this span alone, so that
p
+
p_{+}
and
p
−
p_{-}
differ only in how they describe the target variable. For example, “
[stationary dashboard camera]…a gray van drives parallel to the ego car at the same speed; after a short moment, the ego car accelerates aggressively.
” Both prompts withhold the physical consequence: neither states whether the ego car ends up ahead or behind, so the model cannot retrieve the outcome directly from text priors and must instead simulate it from
x
0
x_{0}
and
v
v
, which is the capability the benchmark probes.
The intervention prompt
p
−
p_{-}
is obtained by editing only the
description of the target variable in the action span:
p
−
=
p
+
[
v
←
v
−
]
p_{-}=p_{+}[\,v\leftarrow v_{-}\,]
; for the example above, the ego
action description flips from acceleration to hard deceleration.
Combined with the shared initial conditioning, this restricts the
prompt-level edit to a single, locatable change in the textual
specification, so that any divergence between
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
should be most cleanly traceable to the model’s
handling of
v
v
. The full prompt building process is in Appendix
A.5
.
3.3
Evaluation Metrics
To address
how to evaluate
, our metric must diagnose whether the generated pair
(
𝒱
+
,
𝒱
−
)
(\mathcal{V}_{+},\mathcal{V}_{-})
genuinely reflects the expected causal effect of the intervention. Each video must individually be a valid generation: it should execute its action, obey physics, and maintain a stable scene. Otherwise, comparing
𝒱
+
\mathcal{V}_{+}
to
𝒱
−
\mathcal{V}_{-}
amounts to comparing two broken outputs. Prior video benchmarks
[
29
,
8
,
42
,
36
]
measure exactly this per-video property and stop there.
But individual plausibility is not sufficient: a model can produce two individually plausible videos that ignore
v
v
entirely, generating identical or randomly different rollouts. Detecting this requires comparing
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
jointly. Prior work in vision–language benchmarks
[
58
,
1
]
has shown that paired comparison is essential for exposing exactly this failure mode. However, existing evaluations of video world models have not adopted this design, even though downstream uses such as action-conditioned simulation
[
72
]
and model-based planning
[
27
]
depend on this property.
Table 1
:
The APEO evaluation framework.
Each criterion is scored as a binary success (
1
1
) or fail (
0
0
); a fail corresponds to the negation of the listed pass criterion. Dimensions A, P, and E are evaluated under both single-video and paired modes, whereas O is paired-only.
Dimension
Single-Video Evaluation Criterion of Success (
s
s
)
Paired-Videos Evaluation Criterion of Success (
p
p
)
A
Adherence
𝒱
\mathcal{V}
executes the action prescribed by
p
p
on the intended
target entity.
The actions rendered in
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
are
visually distinguishable, and their difference is aligned with the
direction of the prompt-level controls.
P
Physics
Motion within
𝒱
\mathcal{V}
respects basic physical constraints
throughout (no teleportation, morphing, or ghost forces).
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
share an aligned trajectory prior
intervention and bifurcate thereafter in a manner consistent
with physical law under the prescribed change.
E
Environment
Background, camera viewpoint, and non-target object permanence remain
stable throughout
𝒱
\mathcal{V}
.
The backgrounds of
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
are near-identical;
observable cross-video difference is attributable to the intervened
action rather than to scene-level artefacts.
O
Outcome
Not applicable.
(Outcome is defined only relative to a counterfactual partner.)
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
terminate in measurably distinct
final states, with the divergence aligned to the prediction of
v
v
in
both direction and affected entities.
The APEO framework.
Adopting paired comparison as our evaluation design leaves one question open: what should we compare
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
on? We design a four-dimension framework whose dimensions jointly cover the conditions a pair must meet to reflect the causal effect of
v
v
: Adherence (the intervention itself was executed), Physics (the dynamics were physically valid), Environment (the context in video held still), and Outcome (the result differed in the predicted direction). The four dimensions are intended to be complementary, each focusing on a different aspect of how a pair can fail. Each one also corresponds to concrete downstream consumers of a video world model:
Adherence of Controls (A)
matters for VLA policy
training
[
65
,
34
,
82
]
and
action-conditioned world models
[
72
,
13
]
:
if
𝒱
−
\mathcal{V}_{-}
silently performs a different action, the policy
learns a wrong action–outcome map.
Physical Realism (P)
matters for inverse dynamics
learning
[
6
]
and latent dynamics
modeling
[
27
]
: physically impossible
intermediate frames corrupt the dynamics signal even when endpoints
look correct
[
31
]
.
Environmental
Consistency (E)
matters for synthetic data
augmentation
[
61
,
28
]
: hallucinated
background changes induce shortcut
learning
[
22
]
.
Outcome Divergence (O)
matters for model-based planning and counterfactual
simulation
[
27
,
72
]
: their value
depends on rolling out distinct futures under distinct actions.
Each APEO dimension except
O
O
can be evaluated in two modes
(Table
1
). The single-video checks (
A
s
,
P
s
,
E
s
A_{s},P_{s},E_{s}
)
ask whether each video alone executes the prompted action, obeys
physics, and holds its scene stable; these are necessary for
plausibility but silent on whether the pair diverges correctly. The
paired-video checks (
A
p
,
P
p
,
E
p
,
O
p
A_{p},P_{p},E_{p},O_{p}
) enforce
ceteris
paribus
:
A
p
A_{p}
checks that
𝒱
+
\mathcal{V}_{+}
and
𝒱
−
\mathcal{V}_{-}
execute visually distinct actions;
P
p
P_{p}
checks that their
trajectories overlap before the intervention and diverge plausibly
afterward;
E
p
E_{p}
checks that their backgrounds match, so any visible
difference is attributable to
v
v
; and
O
p
O_{p}
checks that the resulting
states differ in the direction
v
v
predicts.
The two modes catch different failures. A model can pass every
single-video check yet fail
P
p
P_{p}
and
O
p
O_{p}
if both videos are
individually plausible but follow the same trajectory, which is the
contrastive bottleneck (Section
4
) where the model ignores
v
v
entirely. Conversely, a pair can pass
O
p
O_{p}
for the wrong reason,
diverging through hallucinated scene artifacts rather than through
v
v
, a failure only
E
p
E_{p}
catches. Reporting all these
dimensions therefore helps localize where a model’s causal pipeline
fails.
VLM-based evaluation with human verification.
Manual annotation at this scale is infeasible, so we use a VLM judge (Gemini 3.1 Pro),
as is now standard in video generation benchmarks
[
29
,
8
,
36
]
, recent autonomous-driving evaluation pipelines
[
65
]
,
and consistent with the broader LLM-as-a-Judge literature
[
79
,
39
]
. For each APEO check,
the judge receives
(
𝒱
+
,
𝒱
−
)
(\mathcal{V}_{+},\mathcal{V}_{-})
along with
(
p
+
,
p
−
)
(p_{+},p_{-})
and answers a primitive-conditioned binary question
(e.g., “Does
𝒱
−
\mathcal{V}_{-}
exhibit a shorter stopping distance
than
𝒱
+
\mathcal{V}_{+}
?”). We use binary rather than Likert scoring
because graded ratings exhibit position, verbosity, and
central-tendency biases that compress
variance
[
62
]
; binary decisions avoid these while
still aggregating to fine-grained scores through dimension- and
instance-level averaging. We validate against 421 human-annotated samples: the judge
agrees with human labels on 82.30% of decisions averaged
across dimensions, comparable to the inter-human agreement of 84.03%.
