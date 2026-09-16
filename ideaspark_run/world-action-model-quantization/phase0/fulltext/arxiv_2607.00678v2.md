# ABot-M0.5: Unified Mobility-and-Manipulation World Action Model

paper_id: arxiv:2607.00678v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

Figure 1
:
Overview of ABot-M0.5.
ABot-M0.5 is a granularity-aligned, action-disentangled, and train-test-consistent world-action model for mobile manipulation.
Top:
The 3-stage training pipeline (Pretrain, SFT1, SFT2 ) progressively improves the model. The final Dream-Forcing stage (SFT2) uses model-predicted videos to train inverse dynamics, significantly boosting robustness.
Left & Center:
To solve structural mismatches, we design a Video
→
\rightarrow
Latent Action
→
\rightarrow
Action pipeline for temporal alignment, and use a dual-level architecture to decouple different action spaces (e.g., base movement and arm manipulation), making the framework adaptable to diverse robot embodiments.
Right:
ABot-M0.5 achieves state-of-the-art performance on both simulation benchmarks (radar chart) and real-world mobile manipulation tasks (bar charts).
Bottom:
Real-world execution sequences (e.g., Arrange Flower, Find Toaster) show successful long-horizon mobile manipulation, where blue and green colors distinguish the navigation and manipulation phases.
Mobile manipulation is a defining capability for general-purpose robots: a capable embodied agent must not only manipulate objects, but also navigate through cluttered environments, maintain long-horizon task context, and execute precise interactions under changing viewpoints and scene dynamics
[
60
,
74
,
32
,
24
]
. Despite rapid recent progress in embodied AI, current embodied learning paradigms still fall short of this goal. Reactive Vision-Language-Action (VLA) policies lack explicit world modeling and long-horizon memory
[
5
,
31
,
4
,
54
,
53
]
, while emerging world-model-based methods, though promising, are still not structured in a way that matches the demands of mobile manipulation
[
72
,
3
,
20
,
21
]
.
The central limitation is not merely insufficient model scale, but a structural mismatch between current world-action learning methods and the requirements of mobile manipulation. Effective mobile manipulation requires alignment at three levels. First,
temporal granularity
must be aligned, since coarse video prediction must ultimately support fine-grained, step-level control; otherwise, subtle but crucial dynamics such as contact onset, grasp closure, and alignment correction are easily blurred or lost
[
21
]
. Second,
action space
must be aligned, since navigation and manipulation follow fundamentally different dynamics—for example, low-frequency global base motion versus high-frequency local arm control—yet are often optimized within a single entangled action space, leading to interference and suboptimal specialization
[
6
,
5
]
. Third,
train-test consistency
must be aligned, since actions at deployment time are conditioned on the model’s own predicted future observations rather than ground-truth futures, creating train-test mismatch and compounding errors over long autoregressive rollouts
[
72
,
33
,
20
,
21
]
.
ABot-M0.5 is designed around this alignment principle. Rather than addressing these issues as isolated patches, it provides a unified framework for alignment-aware world-action learning in mobile manipulation. To align temporal granularity, ABot-M0.5 introduces frame-level latent actions between video latents and robot actions, forming a fine-grained intermediate action space that captures local visual state transitions and is less tied to any specific embodiment. This factorizes the direct video-to-action mapping into a video-to-latent-action-to-action pipeline, allowing the model to recover motion intent from coarse visual dynamics and translate it into executable low-level control. To align action structure, ABot-M0.5 adopts a dual-level Mixture-of-Transformers architecture that disentangles both modality-specific representations and heterogeneous action subspaces, enabling base motion and arm manipulation to be modeled separately within a unified system. To align inference conditions, ABot-M0.5 further employs a Dream Forcing training strategy that exposes inverse dynamics learning to self-dreamed videos, directly aligning the conditioning context of training with that of inference, thus improving robustness to prediction errors.
Extensive experiments on challenging mobile manipulation and manipulation benchmarks show that ABot-M0.5 achieves strong performance in both long-horizon and fine-grained manipulation tasks. These results indicate that progress in mobile manipulation depends not only on scaling model capacity or data volume, but also on aligning world modeling, action abstraction, and deployment-time behavior within a single coherent framework
[
44
]
. More broadly, they point to a practical path for extending WAMs from stationary manipulation to mobile manipulation.
Our main contributions are as follows:
•
We identify three core structural bottlenecks that limit existing WAMs in mobile manipulation: temporal granularity mismatch between coarse video prediction and fine-grained control, action structure mismatch between heterogeneous mobility and manipulation behaviors, and context mismatch between training and autoregressive inference.
•
We propose ABot-M0.5, a new WAM architecture for mobile manipulation that addresses these bottlenecks through intermediate latent actions, a dual-level Mixture-of-Transformers design, and Dream Forcing, enabling fine-grained motion abstraction, structured action decoupling, and train-test consistent inverse dynamics learning.
•
We demonstrate strong results on challenging mobile manipulation and manipulation benchmarks, with clear gains in long-horizon task success and fine-grained manipulation accuracy, and validate the contribution of each component through extensive ablations.

## Method

This section formalizes mobile manipulation as an alignment-aware world-action learning problem. Rather than treating mobile manipulation as a simple long-horizon extension of stationary manipulation, we view it as a setting in which world modeling, action modeling, and deployment-time rollout must be jointly aligned. This perspective provides a unified explanation for why current embodied learning methods struggle in mobile manipulation, and motivates the design of ABot-M0.5 in the next section.
2.1
Problem Setting
We consider language-conditioned mobile manipulation tasks in which a robot must execute long-horizon behaviors by jointly performing navigation and object interaction in visually complex environments
[
60
,
74
,
32
]
. At each time step
t
t
, the agent receives a language instruction
l
l
, a multi-view visual observation
o
t
o_{t}
, and optionally a history of past observations and actions. The goal is to generate low-level executable actions
a
t
a_{t}
that complete the task over an extended horizon while remaining consistent with future world evolution.
Formally, let
o
t
=
{
I
t
(
1
)
,
…
,
I
t
(
N
c
)
}
o_{t}=\{I_{t}^{(1)},\dots,I_{t}^{(N_{c})}\}
denote the multi-view observation at time
t
t
, where
N
c
N_{c}
is the number of cameras and
I
t
(
i
)
I_{t}^{(i)}
is the image captured by the
i
i
-th camera. Given an observation history
o
≤
t
o_{\leq t}
, an action history
a
<
t
a_{<t}
, and a language instruction
l
l
, the policy aims to predict future behavior over a horizon
H
H
. In reactive policies, this is often formulated as a direct conditional mapping over a chunk size
H
H
[
6
,
31
,
4
,
10
]
:
a
t
:
t
+
H
−
1
∼
π
(
⋅
∣
o
≤
t
,
a
<
t
,
l
)
.
a_{t:t+H-1}\sim\pi(\cdot\mid o_{\leq t},a_{<t},l).
(1)
Such a formulation is effective when the required action mainly depends on the current observation and the temporal horizon is short. However, it becomes increasingly brittle in mobile manipulation, where future decisions depend not only on the current state, but also on how the environment is expected to evolve under the robot’s own future actions.
World Action Models (WAMs) address this limitation by jointly modeling future observations and future actions
[
72
,
3
,
76
,
71
,
33
]
. Let
z
t
+
1
:
t
+
H
z_{t+1:t+H}
denote the compressed video latent of the future observations over the time horizon
t
+
1
:
t
+
H
t+1:t+H
. Instead of directly predicting actions from the current observation, a WAM models a structured future trajectory:
(
z
t
+
1
:
t
+
H
,
a
t
:
t
+
H
−
1
)
∼
p
(
⋅
∣
o
≤
t
,
a
<
t
,
l
)
.
(z_{t+1:t+H},a_{t:t+H-1})\sim p(\cdot\mid o_{\leq t},a_{<t},l).
(2)
This formulation introduces explicit future world modeling and provides a natural interface for long-horizon rollout, since future prediction and action prediction are embedded in the same autoregressive process.
However, directly applying existing WAM formulations to mobile manipulation remains insufficient. Mobile manipulation differs from stationary manipulation in at least three ways. First, future world evolution spans larger viewpoint changes and more diverse scene transitions due to robot movement. Second, the action space becomes heterogeneous, since both mobility and manipulation must be generated within a single policy. Third, rollout robustness becomes more important, because long-horizon mobile tasks amplify small prediction errors over time.
These differences suggest that mobile manipulation should not be viewed merely as a larger version of stationary manipulation. Instead, it should be treated as a world-action learning problem with stricter structural requirements on representation, control, and rollout.
To make this explicit, the core challenge in mobile manipulation lies in bridging two fundamentally different spaces: the coarse, long-term future video latents
z
t
+
1
:
t
+
H
z_{t+1:t+H}
that capture global world evolution, and the fine-grained, heterogeneous executable robot actions
a
t
:
t
+
H
−
1
a_{t:t+H-1}
. Directly mapping the video latent to executable robot action is notoriously difficult due to the severe granularity and semantic gaps between them.
For notational simplicity in the following text, we will abstract away the explicit horizon
H
H
and denote
z
t
+
1
:
t
+
H
z_{t+1{:}t+H}
and
a
t
:
t
+
H
−
1
a_{t{:}t+H-1}
simply as
z
t
+
1
z_{t+1}
and
a
t
a_{t}
, respectively.
Ideally, a successful mobile manipulation policy should learn a coherent hierarchical process: it must first anticipate how the visual world will evolve (
z
t
+
1
z_{t+1}
), then distill this macroscopic evolution into frame-level intermediate motion intents that capture local visual state transitions, and finally ground these intents into embodiment-specific low-level controls (
a
t
a_{t}
). Formally, this desired hierarchy can be conceptualized as:
Video Latent
​
z
t
+
1
→
Frame-level Motion Intents
⏟
Bridging Space
→
Robot Action
​
a
t
,
\text{Video Latent }z_{t+1}\rightarrow\underbrace{\text{Frame-level Motion Intents}}_{\text{Bridging Space}}\rightarrow\text{Robot Action }a_{t},
(3)
where the intermediate bridging space serves as the crucial link connecting future world dynamics with fine-grained physical execution. However, how to effectively define, learn, and align this intermediate space remains an open question. In the next section, we will address this by introducing
latent actions
to instantiate this bridging space, along with tailored architectures and training strategies to achieve full alignment.
2.2
Core Bottlenecks in Mobile Manipulation
Under the formulation above, the key limitation of existing methods is not merely insufficient model scale, but a structural mismatch between how current WAMs are trained and the requirements of mobile manipulation
[
72
,
3
,
76
]
. We identify three core bottlenecks.
Temporal Granularity Mismatch
Existing WAMs typically model future observations in temporally compressed chunks. This design is computationally efficient and suitable for long-horizon video prediction, but it creates a mismatch between the temporal granularity of world modeling and that of control generation. In practice, future video latents may summarize multiple frames within a chunk, whereas robot actions must often be generated at every frame or control step.
This mismatch is especially problematic in mobile manipulation, where fine-grained interactions determine success. Behaviors such as grasp closure, contact onset, object release, fine alignment, and local collision avoidance often unfold over very short temporal windows. When world modeling is performed only at a coarse chunk level, these local transitions may be smoothed out or omitted, making it difficult for the policy to recover the precise motion intent required for execution.
Action Structure Mismatch
Mobile manipulation introduces a heterogeneous action space that differs substantially from the action space of stationary manipulation. The robot must control both global mobility and local manipulation, and these two forms of behavior obey very different dynamics. Base movement tends to be low-frequency, smooth, and globally oriented. Arm manipulation, by contrast, is higher-frequency, local, and sensitive to contact-rich dynamics. Treating them as a single entangled action space forces the model to optimize over conflicting patterns within one shared representation.
This mismatch has two consequences. First, it increases optimization difficulty. Gradients from mobility-dominated trajectories and manipulation-dominated trajectories may interfere, preventing the model from specializing to either mode effectively. Second, it weakens compositionality. In many mobile manipulation tasks, base control and arm control must coordinate while remaining structurally distinct.
Rollout Condition Mismatch
A third bottleneck arises from the discrepancy between how inverse dynamics is trained and how actions are actually generated at inference time. During training, inverse dynamics is usually conditioned on ground-truth future observations or their latent representations. During inference, however, such ground-truth futures are unavailable. The model must instead act based on its own predicted visual rollouts, which inevitably contain noise, uncertainty, and sometimes severe errors such as blurring, object drift, or hallucinated content.
This train-test mismatch creates a form of exposure bias in world-action learning, related to the distribution-shift studied in sequence prediction and imitation learning
[
2
,
56
]
. The inverse dynamics model is optimized under ideal future conditions, but deployed under imperfect self-generated futures. In long-horizon mobile manipulation, the discrepancy compounds over time and can eventually derail execution.
Summary
These three bottlenecks reveal a shared pattern: current methods are insufficiently aligned with the structure of mobile manipulation. Coarse visual prediction is misaligned with fine control, entangled action learning is misaligned with heterogeneous robot behavior, and ground-truth-conditioned training is misaligned with autoregressive deployment. ABot-M0.5 is designed around this observation. The next section presents the full model, including latent actions for temporal alignment, a dual-level Mixture-of-Transformers for structured action modeling, and Dream-Forcing for rollout-aligned inverse dynamics learning.
