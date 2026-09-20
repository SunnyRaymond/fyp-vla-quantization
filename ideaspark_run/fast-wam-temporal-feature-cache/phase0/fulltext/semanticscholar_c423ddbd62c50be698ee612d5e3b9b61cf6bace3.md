# MotuBrain: An Advanced World Action Model for Robot Control

paper_id: semanticscholar:c423ddbd62c50be698ee612d5e3b9b61cf6bace3
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Recent progress in embodied foundation models has been driven by Vision-Language-Action (VLA) policies [6,
8, 10, 20, 22, 26, 33, 38], which map visual observations and language instructions in pretrained Vision-
Language Models (VLMs) [1, 4] directly to robot actions. Inheriting rich semantic priors, VLAs generalize well
across diverse objects and language instructions, achieving strong performance on a wide range of robotic
tasks. However, since they are primarily pretrained on static image–text data, they often neglect the perception
and prediction of fine-grained world dynamics that are essential for precise robotic control, leading to a
superficial imitation of behaviors rather than temporal understanding of the physics of the world.
With the rise of video generation models [3, 23, 30, 34, 39], a growing body of research has begun exploring
how to adapt these models for world modeling [9, 15–18, 28]. A world model aims to predict how the
environment evolves in response to actions—a capability that aligns directly with what a video generation

## Method

Conceptually, video generation models are naturally suited to this task, as video generation models are
pretrained on vast and diverse web video data, which equips them with rich spatiotemporal priors regarding
object permanence, physical dynamics, and human-object interactions. This enables them to generalize more
effectively and reason about novel scenarios where in-domain data are scarce. As a result, leveraging video
generation models as a foundation for world modeling has emerged as a promising paradigm for scalable
robot policy learning.
Following this insight, early attempts largely followed a two-stage paradigm: Video Generation Model (VGM)
plus Inverse Dynamics Model (IDM) [13, 14, 19, 29, 40]. In this framework, a video diffusion model
pretrained on large-scale web data first predicts future visual trajectories from current observations and
language instructions. Subsequently, an inverse dynamics model infers actions from the generated future
frames. While this paradigm successfully leverages rich spatiotemporal priors from video data to achieve
broad generalization, it suffers from a critical drawback in that errors in video prediction accumulate over
time, which leads to compromised action accuracy and downstream policy performance.
To mitigate this issue, a subsequent line of research has explored World Action Models (WAMs) that unify visual
dynamics and action prediction within a single, jointly optimized objective [5, 21, 24, 25, 27, 36, 37, 41].
Unlike the VGM+IDM pipeline, which decouples forecasting from action inference, WAMs simultaneously
predict future visual states and actions in an aligned manner. This integration offers two key advantages: (1)
it avoids the cascading errors inherent in sequential video prediction, and (2) it overcomes the fragmented
functionality of conventional embodied systems, where semantic understanding, dynamics modeling, and
action generation are typically learned from disparate supervision sources.
In our view, the core source of intelligence in such a unified model is its ability to absorb large-scale
heterogeneous multimodal data under one unified training recipe. In principle, this includes pure video
data without action annotations, robot data with aligned video-language-action trajectories across different
embodiments, and even task-agnostic interaction data with partially missing modalities [32]. By contrast, VLA
learning primarily relies on robot task trajectories with aligned observation-language-action supervision, and
adaptation to the target robot is primarily coupled to embodiment-specific action data [6, 12, 22].
1
arXiv:2604.27792v5  [cs.RO]  15 Jul 2026

Motus [5] was an early step in this direction. It established a unified world-action formulation in which
video and action are modeled in a shared generative framework, so that policy modeling, world modeling,
video generation, inverse dynamics, and joint video-action prediction become different inference modes
of the same model. It also showed that, with UniDiffuser-style continuous multimodal modeling and a
Mixture-of-Transformers design, a world action model can absorb heterogeneous multimodal data rather than
being restricted to embodiment-specific task trajectories.
Building on this foundation, we present Motubrain. Like Motus, Motubrain adopts UniDiffuser [2] to jointly
model and schedule the two continuous modalities, namely video and action, and uses a three-stream Mixture-
of-Transformers architecture to integrate video generation, action modeling, and language conditioning
within one system. This unified formulation again supports inference over five distributions with the same
model: vision-language-action policy modeling, world modeling, video generation, inverse dynamics, and joint
video-action prediction. More importantly, it preserves the key advantage of unified world-action modeling:
the model can learn from a much broader family of multimodal data, including video-only data without action
labels, interaction data without explicit task language, and heterogeneous robot trajectories collected from
different embodiments.
Motubrain further extends this paradigm in several practically important directions. It introduces a unified
multiview representation that supports an arbitrary number of camera views under different camera layouts,
rather than depending on a fixed visual input format. It uses an independent text stream to more tightly couple
high-level semantics with low-level control, making instruction following an explicit part of action generation
and improving semantic understanding. It adopts a unified action representation across embodiments, enabling
the model to capture transferable control regularities rather than overfitting to the action format of a single
robot. Beyond architecture and pre-training, Motubrain also involves a post-training and deployment recipe
tailored to long-horizon real-world control: autoregressive diffusion rollout enables temporally extended
execution, V2A-style asymmetric dependency enables action-only inference without explicitly generating future
video, and real-time chunked closed-loop execution reduces boundary discontinuities during asynchronous
control. Finally, we develop a systems-oriented inference stack including denoising-step reduction, compilation
with CUDA-graph-friendly execution, FP8 quantization, and DiT caching, which together deliver more than
50× end-to-end speedup over the naive baseline and make large world action models practical for real-time
robotic deployment.
These design choices lead to strong empirical performance across both action-centric and world-modeling
evaluations. On RoboTwin 2.0, Motubrain achieves average success rates of 95.8% and 96.1% under the clean
and randomized settings, respectively, with the randomized score exceeding 95%. On WorldArena, Motubrain
attains the strongest reported EWMScore in our comparison, indicating that the same model not only executes
actions effectively but also predicts future world dynamics accurately. Beyond standardized benchmarks,
Motubrain can be adapted to new humanoid embodiments with only 50–100 same-embodiment trajectories,
while retaining the ability to solve long-horizon and dexterous manipulation tasks without relying on an
additional VLM planner, a dual-system decomposition, external memory, or retry-specific data. Together, these
results suggest that unified world action models can simultaneously scale in generality, controllability, and
real-world deployability.
2
Method
In this section, we present the overall methodology of Motubrain from four aspects: model architecture,
pre-training, post-training, and inference. We first introduce the core architectural design and the main
acceleration techniques integrated into Motubrain in Section 2.1. We then describe the pretrained foundation
model used in our system in Section 2.2. After that, we present the post-training pipeline, including sparse
adaptation and step distillation, in Section 2.3. Finally, we describe the inference-time acceleration strategies
deployed in Motubrain in Section 2.4.
2




Video  Model
Action   Model
t
t
Language  Model
Multi-Modal Self-Attention
Text Encoder
Video Encoder
Action Encoder
Instruction: Insert the 
flowers into the vase, 
spray water with the 
watering can.
QKV
QKV
FFN
QKV
FFN
FFN
video
action
 Language Block
 Video Block
 Action Block
 Language Block
 Video Block
 Action Block
L×
N×
M×
Figure 1 Overview of Motubrain’s architecture. Motubrain builds on a unified video-action backbone and adopts a
three-stream Mixture-of-Transformers architecture with text, video, and action streams. It further uses an H-bridge
attention design to balance cross-modal interaction and efficiency, while supporting flexible multiview inputs through
view-dependent 3D RoPE offsets.
2.1
Model Architecture
Motubrain first adopts UniDiffuser [2] to jointly model and schedule the two continuous modalities, i.e., video
and action, so that all interaction patterns between them are captured in a unified generative framework. As a
