# Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization

paper_id: arxiv:2608.26103v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

Zero-shot cross-task generalization is essential for general-purpose robotic
manipulation: a robot should infer how to act in a task it has not practiced,
using only information available at deployment. Inspired by in-context learning
(ICL)
[
1
,
2
]
, where a model solves a new problem
from input context without parameter updates, we view robotic task generalization
as specifying an unseen task through deployment-time context. This perspective
suggests a route toward open-ended task generalization: the policy could infer
the intended task from context and translate it into executable robot behavior.
Recent vision-language-action (VLA)
models
[
3
,
4
,
5
,
6
,
7
,
8
,
9
,
10
]
and video-action models
[
11
,
12
,
13
,
14
,
15
,
16
]
have substantially expanded the capabilities of robot policies, yet both
predominantly rely on language as the task interface. Language, however, often
underspecifies manipulation tasks: spatial constraints, intermediate states, and
temporal structure can be cumbersome to articulate, and even detailed
instructions provide no direct visual evidence of how the scene should evolve.
Human demonstration videos provide a natural in-context specification of the
intended task
[
17
,
18
]
. By directly presenting the
desired visual state changes and their temporal evolution, they provide concrete
visual evidence of how the scene should evolve. Although a human video does not
provide executable robot actions, it gives the policy a visual reference from
which to infer the desired task evolution and realize it through the robot’s own
embodiment and dynamics.
Exploiting human videos at scale, however, presents two key challenges. First,
large-scale, task-rich human-robot paired data remain scarce and expensive to
collect manually
[
19
,
20
,
21
]
.
Learning from human video instructions requires semantically corresponding robot
trajectories that retain executable actions, yet such pairings are rarely
available at scale. Second, a policy trained on seen tasks can learn shortcuts
that allow it to ignore the in-context video. In particular, the next robot
video-action chunk can often be predicted from the robot history and text
instruction alone. Consequently, the model may perform well on familiar
training tasks without learning to use the human video, and then underuse the
video at test time, precisely when it is needed to specify an unseen task.
To address these challenges, we introduce
Zero-WAM
, a causal
video-action model that follows in-context human video instructions for zero-shot
robotic task generalization. Zero-WAM supports two forms of task
specification within a single policy: language instructions and human
demonstration videos. Given either form of instruction, the model
autoregressively predicts future robot videos and executable actions. Human
videos enable Zero-WAM to condition these predictions on demonstrated visual
state changes, providing information about the intended task evolution beyond
that available from language and robot history alone.
To scale training with human video instructions, we propose an
in-context human video generation pipeline
that uses task-sampled robot trajectories to construct
semantically matched human manipulation videos. Each human video is paired with
its corresponding robot trajectory, which retains the executable robot actions,
to form a human-robot in-context learning (ICL) pair. The resulting
HumanGen
dataset contains 74.2K human-robot ICL pairs spanning 8.6K tasks. We further
perform task-balanced curation of robotic pretraining data by repartitioning
public robot trajectories into more than 6,000 manipulation tasks and sampling
trajectories at the task level. This procedure yields approximately 400K robot
trajectories per training epoch, which we refer to as
Task-diverse VA
. Task-balanced
sampling both supplies task-rich source trajectories for the in-context human video generation pipeline and
prevents video-action pretraining from being dominated by repeated teleoperation
trajectories from a small number of tasks. Together,
HumanGen
and
Task-diverse VA
provide scalable human video task specifications and diverse robot video-action
dynamics for causal video-action pretraining.
Figure
2
summarizes the resulting data composition and the in-context human video generation pipeline.
To ensure that Zero-WAM uses the in-context human video rather than relying
on shortcuts from robot history and text, we introduce an in-context future
chunk prediction (IFP) objective. Standard next-chunk prediction can often be
solved using only local robot history, particularly for tasks observed during
training. IFP instead supervises multiple strided chunks of future robot video
from the current robot-video representation, encouraging the model to encode the
longer-term task evolution conveyed by the human video.
We evaluate Zero-WAM on zero-shot cross-task generalization in RoboTwin
2.0
[
22
]
simulation and on real-world robotic manipulation.
In RoboTwin 2.0, Zero-WAM achieves a
46.95
%
46.95\%
average success rate across
seven unseen tasks, outperforming LingBot-VA
[
11
]
by an
absolute margin of
29.50
29.50
percentage points. Real-world experiments further
demonstrate human-video-guided generalization to unseen task configurations
involving multi-object scenes, long-horizon manipulation, and precision-demand
insertion, with Zero-WAM outperforming LingBot-VA across all three task
families.
Our contributions are summarized as follows:
•
We formulate zero-shot robotic task generalization as in-context
world action modeling, where a single causal policy supports both language
and human videos as task instructions.
•
We propose a scalable in-context human video generation pipeline that automatically converts
task-sampled robot trajectories into semantically matched human video
instructions, yielding the
HumanGen
dataset of 74.2K human-robot ICL
pairs over 8.6K tasks; the same task-level sampling also yields
Task-diverse VA
data, a task-balanced corpus for autoregressive robotic video-action
pre-training.
•
We propose Zero-WAM with an in-context future chunk prediction
(IFP) objective that discourages shortcut learning from seen-task
trajectories and strengthens the use of in-context human video prompts.
•
We demonstrate effective zero-shot cross-task generalization in RoboTwin,
where Zero-WAM substantially improves over video-action baselines, and
further evaluate real-world generalization to unseen task configurations
without collecting corresponding robot data or updating any model parameters.

## Method

Zero-shot robotic task generalization requires an instruction interface that
transfers to unseen tasks, together with training data that covers diverse task dynamics rather
than merely increasing the number of robot trajectories. As an engineering
foundation, we first curate
Task-diverse VA
data by re-sampling public robotic
pre-training data at the task level (
Section
2.1
). Building on the same
task-level sampling, we introduce the
in-context human video generation pipeline
to convert
task-sampled robot videos into human video instructions (
Section
2.2
).
The generated human-robot ICL pairs form
HumanGen
, which includes Pre-train
ICL (External), Pre-train ICL (In-house), Simulation ICL, and Real-world ICL
(
Section
2.3
).
Figure
2
summarizes this data
composition and the pipeline used to generate
HumanGen
data.
Figure 2
:
Data construction and in-context human video generation.
Top:
Task-diverse VA
data provide task-balanced robotic video-action pre-training data,
while
HumanGen
contains Pre-train ICL (External), Pre-train ICL (In-house),
Simulation ICL, and Real-world ICL pairs. Bottom: the in-context human video generation pipeline converts
task-sampled robot videos into human video instructions.
2.1
Task-Diverse Video-Action Data
Task-diverse VA
data are curated from public robotic video-action (VA) pre-training
datasets, including AgiBot
[
23
]
,
InternData-A1
[
24
]
,
Open-X-Embodiment
[
4
]
, RoboCOIN
[
25
]
, and
RoboMIND
[
26
]
. These source datasets are the same public VA
pre-training datasets used by LingBot-VA
[
11
]
. Although these datasets contain large numbers of robot
trajectories and broad task coverage, many trajectories come from repeated
tele-operation of the same task. Using the raw distribution directly would make
pre-training overly influenced by redundant trajectories, preventing the model
from fully benefiting from the task diversity already present in the source
datasets. We address this by re-partitioning each dataset
into tasks, where each task is defined by the combination of the manipulation
action and the object. The task labels are obtained from the original
dataset metadata when available, or parsed from the robot trajectories when the
metadata are insufficient. We then sample a bounded number of trajectories from each
task, with the sampling bound adjusted according to the intra-task diversity of
each source dataset. Across the five source datasets, we sample more than 6,000
tasks and approximately 400K corresponding robot trajectories in each training
epoch. This
balanced video-action corpus is effective for adapting a general-domain video generation
model into an autoregressive robotic video-action model.
2.2
In-Context Human Video Generation Pipeline
Human video demonstrations convey task semantics directly through visual
state changes, are easier to obtain than robot demonstrations, and do not
depend on the robot embodiment used at test time. Scaling task-rich
human-robot in-context learning pairs is therefore an effective route to
cross-task generalization.
Manually collecting such paired human and robot action data is difficult, and the cost becomes especially prohibitive when high task diversity is required. We therefore start from the robotic pre-training corpus and again sample trajectories at the task level, following
Section
2.1
. This gives us a broad set of robot videos with executable action annotations. We then introduce an
in-context human video generation pipeline
that converts these robot trajectories into human
manipulation videos with the same task semantics. To increase the diversity of visual
alignment in the human-robot ICL pairs, the generated human videos include
variations in background, viewpoint, environment style, object instance, and object placement while
preserving the same task semantics.
The bottom part of
Figure
2
shows the in-context human video generation pipeline for generating human video instructions.
For each sampled robot video, we first use a VLM
(Gemini 3.1 Pro
[
27
]
or
Qwen3.6-Plus
[
28
]
) for task analysis. The VLM
extracts task-level information, including the task name, initial object states,
object state changes, and final object states. It also produces an image-editing
prompt that transforms the first robot video frame into the initial observation
of a human manipulation scene. We inject the visual-alignment variations
described above through this image-editing prompt while preserving the task
semantics. Given the first robot frame and the image-editing prompt, an
image editing model (Nano Banana 2
[
29
]
or
Qwen-Image-2.0
[
30
]
) generates the
initial human observation image for the corresponding task. We then use the VLM
to process this edited human observation together with the extracted object-state
information, producing a video-generation prompt that describes how human hands
should manipulate the objects to complete the task. This prompt is sent to a
video generation model (Wan 2.7
[
31
]
or
Kling AI 3.0
[
32
]
) to
synthesize the human manipulation video. Finally, the VLM evaluates each
generated video for task semantic preservation and physical plausibility, and
qualified human videos are paired with their original robot trajectories as ICL
samples.
Table 1
:
Comparison with existing task-level paired human-robot datasets.
“Multi-source” indicates whether the dataset is constructed from multiple data
sources. “Embod.” is the number of robot embodiments. “Human view” specifies
the viewpoint of human videos. “Human acquisition” indicates whether the human
videos require manual collection or are automatically generated. “Align. div.”
indicates whether human-robot pairs include diverse degrees of visual alignment,
such as changes in background, viewpoint, environment style, object instance, or
object placement. “Samples” is the number of human video instructions, and
“Tasks” is the number of task categories.
Dataset
Multi-source
Embod.
Human view
Human acquisition
Align. div.
Samples
Tasks
MIME
[
19
]
✗
1
third
manual
✗
8.3K
20
EgoMimic
[
20
]
✗
1
ego
manual
✗
2.1K
3
BC-Z
[
17
]
✗
1
ego
manual
✓
18.7K
100
EgoHumanoid
[
33
]
✗
1
ego
manual
✗
1.2K
4
RH20T
[
21
]
✗
4
ego & third
manual
✗
110K
147
EgoScale
[
34
]
✗
1
ego
manual
✗
10.3K
344
HumanGen
(ours)
✓
>
>
45
ego & third
auto-generated
✓
74.2K
8.6K
2.3
In-Context HumanGen Dataset
HumanGen
is a collection of human-robot ICL pairs generated with the
in-context human video generation pipeline. Each pair contains a generated human video instruction and
its corresponding robot trajectory with executable actions.
HumanGen
is
organized by data source into Pre-train ICL (External),
Pre-train ICL (In-house), Simulation ICL, and Real-world ICL. Following the same
task-diverse sampling principle used to curate
Task-diverse VA
data, source robot
videos are sampled by task rather than by raw trajectory frequency, so that the ICL
data are not dominated by repeated executions of a small set of tasks.
Pre-train ICL (External).
This subset is built from the same five public robotic video-action datasets
used for
Task-diverse VA
data: AgiBot
[
23
]
,
InternData-A1
[
24
]
,
Open-X-Embodiment
[
4
]
, RoboCOIN
[
25
]
, and
RoboMIND
[
26
]
. These datasets cover diverse robot
embodiments, scenes, and objects, including more than 45 robot embodiments. We sample robot trajectories by
task from each dataset and convert them into semantically matched human videos
using the in-context human video generation pipeline. Specifically, we sample 3,354 tasks and 6,660
trajectories from AgiBot, 261 tasks and 12,515 trajectories from InternData-A1, 438
tasks and 9,290 trajectories from Open-X-Embodiment, 512 tasks and 6,513 trajectories
from RoboCOIN, and 497 tasks and 6,210 trajectories from RoboMIND. In total,
Pre-train ICL (External) contains 5,062 tasks and 41,188 human-robot ICL pairs.
For this subset, we intentionally diversify the visual alignment between the
human and robot videos: we enforce changes in the scene environment, tabletop
background, object instances, and object placement, and balance first-person and
third-person viewpoints. Because task semantics are preserved across these
visual variations, the model is pushed to learn human-robot task
correspondences that are invariant to changes in scenes, objects, and viewpoints,
improving its robustness to visually diverse ICL prompts.
Pre-train ICL (In-house).
To further increase task coverage, we sample robot trajectories by task from our
in-house robotic datasets. This subset covers multiple robot embodiments, such as
bimanual Franka and Galaxea R1 Pro, and contains 3,522 tasks and 30,247
human-robot ICL pairs. When using the in-context human video generation pipeline to produce the corresponding human videos for this subset, we reduce the ratio of third-person
viewpoints and decrease the frequency of scene, object, and object-placement
changes. This produces more visually aligned ICL samples while maintaining task
diversity.
Simulation ICL.
To support cross-task evaluation in simulation, we generate ICL data for 50
RoboTwin tasks
[
22
]
. This subset contains 2,500 ICL samples, with 50 samples for each
task. Among them, 43 tasks are used for post-training (2,150
training samples), while the remaining 7 tasks are reserved as unseen tasks for
zero-shot cross-task evaluation.
Real-world ICL.
Real-world ICL contains human-robot pairs collected on the bimanual Franka
evaluation embodiment. It includes 252 human-robot ICL pairs across the three
real-world task families used in our evaluation: 120 pairs from 30
object-to-container placement training task combinations, 96 pairs from 16
three-object sequential manipulation training task combinations, and 36 pairs
for two-table-leg insertion. This subset is used for post-training so that the model can fit the
evaluation robot kinematics for cross-task evaluation.
Comparison with existing datasets.
Table
1
compares
HumanGen
with existing task-level
paired human-robot datasets. The statistics are taken from their official
papers. Unlike prior datasets that rely on manual human data collection,
HumanGen
uses the in-context human video generation pipeline to automatically generate semantically
matched human videos from robot trajectories, enabling scalable construction of
human-robot ICL pairs.
HumanGen
also combines multiple data sources (public,
in-house, simulation, and real-world), covers substantially more robot
embodiments, contains human data from both ego and third-person views, and
contains paired data with diverse degrees of visual alignment. Most importantly,
HumanGen
contains substantially broader task coverage, which is a key factor
for zero-shot cross-task generalization.
