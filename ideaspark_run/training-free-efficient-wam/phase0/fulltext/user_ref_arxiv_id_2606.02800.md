# [user-supplied anchor] 2606.02800

paper_id: user_ref:arxiv_id:2606.02800
tier: U
source_used: html_arxiv
warning: none

## Intro

Physical AI agents perceive, reason, and take actions to interact with the real world. However, training such agents directly in the real world is slow, expensive, and could be dangerous. To overcome these bottlenecks, we must construct a training facility to enable safe and scalable learning in simulated worlds, where Physical AI agents acquire two fundamentally coupled capabilities:
understanding
and
generation
. Understanding allows an agent to infer latent representations, semantics, and dynamics from partial observations, and generation empowers the agent to predict and simulate plausible futures, anticipating how the world evolves and how the agent should take actions in response. Prior work has largely treated these two pillars in isolation, leading to separate discriminative models for perception and reasoning, such as Vision-Language Models (VLMs); generative models for world simulation, such as Video Generation Models and Forward Dynamics Models; and action-prediction models, such as Vision-Language-Action Models (VLAs) and World-Action Models (WAMs).
We argue that this paradigm separation is fundamentally limiting: understanding requires reasoning about the future evolution of the world and the consequences of actions, while generation relies on a compact, structured representation of the world and agent behaviors. Unifying them into a single scalable framework is therefore essential for Physical AI. Consider a general home robot instructed to clean a dining table after dinner. Under the current paradigm, the robot must stitch together a disjointed suite of models: a VLM to locate dishware and generate an executable plan, a VLA or WAM to generate action sequences, and a Forward Dynamics Model or “World Model” to simulate and evaluate future states. This fragmented architecture is suboptimal and computationally wasteful. Can we instead design a single, unified model that natively addresses all essential capabilities for Physical AI agents?
We introduce Cosmos 3, a family of omnimodal world models that jointly model language, image, video, audio, and action for both understanding and generation.
Serving as a general-purpose backbone for Physical AI, Cosmos 3 unifies a wide array of distinct model classes into a single framework (
Fig.
1
).
Depending on the input-output configuration, Cosmos 3 seamlessly transitions between multiple operational modes: it can operate as a vision-language model for multimodal understanding and reasoning; a text-to-image generator, a video generator for text-to-video synthesis, image animation (image-to-video), future prediction (video-to-video), or synchronous audio-video generation; a world-action model for joint action prediction and environmental simulation.
By unifying perception, simulation, and execution without architectural modifications, Cosmos 3 eliminates the need for fragmented, task-specific pipelines, enabling scalable learning through shared representations and joint multi-task supervision.
Figure 1
:
Cosmos 3 serves as a general-purpose backbone for Physical AI.
By jointly modeling language, image, video, audio, and action for both understanding and generation, Cosmos 3 unifies a wide range of model classes within a single network architecture, including vision-language models, image generation models, audio-visual generation models, policy or world-action models, forward dynamics models, and inverse dynamics models.
Scaling training data and environments for Physical AI agents remains a persistent bottleneck. Cosmos 3 offers a strong starting point to address this challenge in three ways: (i) synthetic data generation, (ii) task-specific specialization, and (iii) training environment (
Fig.
2
).
In the near term, Cosmos 3 synthesizes high-fidelity, diverse visual data to enhance training for Physical AI agents. We demonstrate how we can post-train Cosmos 3 into a better synthetic data generator in
Sec.
4.2.3
and
Sec.
4.2.4
.
Since agents perceive and interact with environments through diverse embodiments and tasks, Cosmos 3 supports task- and embodiment-specific specialization on top of a shared model. As a powerful mid-training model for Physical AI, Cosmos 3 establishes a better starting point by modeling general world dynamics and action priors while remaining highly amenable to downstream adaptation. In practice, the model can be post-trained on target data for distinct applications without architectural modifications, enabling data-driven specialization that retains a common world representation thanks to its omnimodal design.
Sec.
4.2.5
describes how we post-train Cosmos 3 into a highly capable world-action model on DROID.
In the long term, Cosmos 3 is positioned to generate high-quality, complex training environments for Physical AI agents. To accelerate open research and deployment in Physical AI, we release our code, model checkpoints, curated synthetic datasets, and an evaluation benchmark under the OpenMDW-1.1 License at
github.com/nvidia/cosmos
and
huggingface.co/collections/nvidia/cosmos3
.
Figure 2
:
Cosmos 3 offers a strong starting point for training Physical AI agents.
Cosmos 3 can be post-trained on target data for distinct applications without architectural modifications. In this paper, we demonstrate how we post-train Cosmos 3 for better synthetic data generation (
Sec.
4.2.3
and
Sec.
4.2.4
) and better robot policy (
Sec.
4.2.5
). In the future, we expect Cosmos 3 to play an essential role in generating high-quality, complex environments for training Physical AI agents.
Table 1
:
Cosmos 3 results overview.
Cosmos 3 consistently outperforms specialized open-source baselines across all capabilities. Detailed results can be found in
Sec.
6
. In the table,
∗
denotes post-trained Cosmos 3 variants;
†
\dagger
denotes closed models; gray-colored cells in each row indicate the model does not possess the corresponding capabilities.
Capability
Model
Reasoning
Generation
General
Robotics
Smart infra.
Driving
Text2Image
Text2Video
Image2Video
Audio
FD: Robot
Policy: Robot
Cosmos3-Super
73.7
57.8
62.6
79.3
91.36
∗
80.0
82.8
7.31
26.0
∗
-
Cosmos3-Nano
69.6
55.1
61.0
76.0
84.61
79.4
82.7
7.34
25.5
∗
39.7
∗
Gemini 3.1 Pro
†
77.5
58.2
58.6
47.2
Qwen3-VL-32B
72.8
52.6
56.1
40.7
Qwen3-VL-8B
68.9
48.5
52.7
46.4
Gemma-4-31B
69.8
51.0
51.3
36.6
Gemma-4-E4B
53.1
39.3
29.4
26.0
Gemini 3 Pro Image
†
90.85
Qwen-Image-2512
84.25
Veo-3.1
†
79.1
82.6
7.45
Wan2.2-A14B
78.0
81.3
Ctrl-World
23.0
π
0.5
\pi_{0.5}
28.1
We evaluate Cosmos 3 and its post-trained variants on a wide range of benchmarks, covering essential understanding and generation capabilities for Physical AI.
Tab.
1
provides a summary of our benchmark results, detailed in
Sec.
6
. As shown in the table, Cosmos 3 establishes a new state-of-the-art across most capabilities, being highly competitive or outperforming specialized models.
The technical details of Cosmos 3 are organized as follows.
Sec.
2
introduces the model architecture, including encoders for all modalities, the arrangement of multimodal tokens to enable different generation modes, the Mixture-of-Transformers (MoT) backbone, multimodal position embedding, and model variants.
Sec.
3
outlines the training data for the reasoner and generator training.
Sec.
4
details the training recipes for reasoner and generator.
Sec.
5
describes infrastructure, including data, training, serving, and evaluation.
Sec.
6
presents our experimental results.
Sec.
7
discusses related work and
Sec.
8
concludes the paper.

## Method

Cosmos 3 is capable of processing multimodal inputs and generating multimodal outputs. Beyond language, vision (image and video), and audio, Cosmos 3 treats action as a core modality, introducing a dedicated class of action tokens. These action tokens bridge the physical world with language-based reasoning and video-based world modeling, linking directly to physically grounded control signals for real-world interaction. Cosmos 3 integrates modality-specific encoders to project different modalities into a unified representation space, which is then processed by a Mixture-of-Transformers (MoT) backbone. During inference, language tokens are generated via next-token prediction, while other modalities are generated through iterative denoising.
2.1
Encoders
Given an input sequence of language, vision, audio, and action, the first step is to embed them into a unified representation space using modality-specific encoders. To enable the shared transformer parameters and positional embeddings to distinguish between different modalities, we add a learnable, modality-specific embedding vector to each non-language modality before feeding it into the MoT backbone.
2.1.1
Image and Video
We adopt two separate encoders for visual input. For visual understanding, we use a ViT encoder pre-trained with vision-language alignment. For visual generation, we use the video VAE encoder from Wan2.2-TI2V-5B
(
Wan et al., 2025a
)
. The ViT encoder has a
16
×
16
16\times 16
patch size, followed by a two-layer MLP that merges
2
×
2
2\times 2
tokens and projects them into the latent space of the transformer. Following Qwen3-VL
(
Bai et al., 2025b
)
, we also aggregate visual features from ViT via DeepStack
(
Meng et al., 2024
)
and insert text–based video timestamps interleaved with video frames
(
Chen et al., 2024b
)
. The VAE compresses the input video temporally by
4
×
4\times
and spatially by
32
×
32
32\times 32
, implemented as
16
×
16
16\times 16
spatial compression followed by a
2
×
2
2\times 2
patch merge. We use a linear layer to project each VAE token into the transformer’s hidden dimension before feeding the latents into the MoT backbone. The ViT encoder for understanding is jointly trained with the backbone, while the VAE encoder for generation is kept frozen during training.
2.1.2
Audio
For audio generation, we adopt the audio VAE architecture from
Lee et al. (2025b)
. The raw stereo audio sampled at 48 kHz is encoded with a hop size of 1920 samples, resulting in 25 tokens per second of audio. The audio VAE is frozen during training. As with the other non-text modalities, audio tokens are projected into the transformer’s hidden dimension using a linear layer before entering the MoT backbone.
2.1.3
Action
We support action modeling across diverse embodiments, including autonomous vehicles, camera motion, robots, and egocentric human motion (head and hands). Since each domain exposes its own native control space—such as joint trajectories, steering commands, body poses, or camera transformations—we map them into a unified action interface that enables consistent multimodal reasoning, generation, and policy learning across domains.
Action representations.
We use actions to denote causal variables that induce changes in the world state.
Given consecutive video tokens, an action token
a
t
a_{t}
represents the transition from the previous state
v
t
−
1
v_{t{-}1}
to the current state
v
t
v_{t}
.
Each embodiment source is transformed into a compact representation that captures a shared underlying geometric structure across different action domains, as illustrated in
Fig.
3
.
At a high level, actions can include up to three components: ego poses for the agent’s main observation frame, effector poses for the agent’s effectors, and grasp states for the manipulation state.
To avoid embodiment-specific controller details such as Proportional–Integral–Derivative (PID) parameters or low-level actuation interfaces, ego and effector poses are represented as pseudo-actions derived from state differences.
For consecutive
SE
⁡
(
3
)
\mathrm{SE}(3)
poses
𝐓
t
−
1
\mathbf{T}_{t-1}
and
𝐓
t
\mathbf{T}_{t}
, we represent motion as the relative transform
Δ
​
𝐓
t
=
𝐓
t
−
1
−
1
​
𝐓
t
\Delta\mathbf{T}_{t}=\mathbf{T}_{t-1}^{-1}\mathbf{T}_{t}
.
We use the 6D representation following
Zhou et al. (2019)
and the OpenCV convention for rotations where the z-axis is along the fingers/grippers and x-axis is to the right.
Grasp states, however, are treated differently: rather than representing temporal differences, they directly encode the current manipulation state at time t.
For cameras and autonomous vehicles, actions are represented by ego poses only, without any effector poses or grasp states. For egocentric data, we use head-camera pose deltas as ego poses, wrist-pose deltas as effector poses, and fingertip positions in each wrist frame as grasp states
(
Yang et al., 2025d
)
. For robotic data, we use head-camera pose deltas as ego poses, end-effector flange-pose deltas as effector poses
(
Lyu et al., 2026
)
, and continuous gripper open/close values as grasp states.
Figure 3
:
Unified action representation.
We map heterogeneous embodiment controls into compact action vectors built from shared geometric components. Ego and effector motions are encoded as relative-pose pseudo-actions using 3D translation and 6D rotation (an over-parameterized rotation representation by Zhou
Zhou et al. (2019)
, as the degree of freedom of rotation is 3), while grasp states directly encode the current manipulation state, such as fingertip positions for hands or gripper open/close values for robots. Domain-aware input and output projections handle heterogeneous action-vector lengths while preserving the shared semantic space.
Action tokenization.
Our action representation maps diverse embodiments into a shared latent action space while preserving embodiment-specific structure and semantics.
We therefore use domain-aware input and output projection layers with separate weight matrices for each embodiment domain
(
Zheng et al., 2026
)
, while sharing the MoT backbone.
For an input
𝐱
∈
ℝ
d
in
(
k
)
\mathbf{x}\in\mathbb{R}^{d_{\text{in}}^{(k)}}
, such as an egocentric action vector concatenating the head-pose delta, left and right wrist-pose deltas, and fingertip coordinates, and domain identifier
k
∈
{
1
,
…
,
K
}
k\in\{1,\ldots,K\}
, the input projection is:
𝐳
=
𝐖
in
(
k
)
​
𝐱
+
𝐛
in
(
k
)
\mathbf{z}=\mathbf{W}_{\mathrm{in}}^{(k)}\mathbf{x}+\mathbf{b}_{\mathrm{in}}^{(k)}
(1)
where
𝐳
∈
ℝ
d
model
\mathbf{z}\in\mathbb{R}^{d_{\text{model}}}
is the latent action token,
𝐱
\mathbf{x}
denotes the normalized action vector, and
𝐖
in
(
k
)
∈
ℝ
d
model
×
d
in
(
k
)
\mathbf{W}_{\mathrm{in}}^{(k)}\in\mathbb{R}^{d_{\text{model}}\times d_{\text{in}}^{(k)}}
and
𝐛
in
(
k
)
∈
ℝ
d
model
\mathbf{b}_{\mathrm{in}}^{(k)}\in\mathbb{R}^{d_{\text{model}}}
are the domain-specific input projection matrix and bias.
To decode the tokens back to the original action space, we use a domain-specific output projection:
𝐱
=
𝐖
out
(
k
)
​
𝐳
+
𝐛
out
(
k
)
\mathbf{x}=\mathbf{W}_{\mathrm{out}}^{(k)}\mathbf{z}+\mathbf{b}_{\mathrm{out}}^{(k)}
(2)
where
𝐖
out
(
k
)
∈
ℝ
d
in
(
k
)
×
d
model
\mathbf{W}_{\mathrm{out}}^{(k)}\in\mathbb{R}^{d_{\text{in}}^{(k)}\times d_{\text{model}}}
and
𝐛
out
(
k
)
∈
ℝ
d
in
(
k
)
\mathbf{b}_{\mathrm{out}}^{(k)}\in\mathbb{R}^{d_{\text{in}}^{(k)}}
are the domain-specific output projection matrix and bias.
All projection parameters are initialized from scratch and optimized jointly with the MoT backbone.
We convert the predicted 6D rotation back to a
3
×
3
3\times 3
SO
⁡
(
3
)
\mathrm{SO}(3)
rotation matrix using singular value decomposition (SVD).
2.2
Token Arrangement and Generation Mode
Cosmos 3 is a unified model that supports various modalities and tasks.
Different tasks can be formulated as interleaved multimodal sequences, each consisting of a series of segments from different modalities. Given a task, all segments are first encoded into embeddings using the modality-specific encoders described above. Once embedded, tokens from different modalities are packed using a unified format that applies across all tasks, which we describe next.
2.2.1
Token Arrangement
The input token sequence consists of two subsequences: an autoregressive (AR) subsequence followed by a diffusion (DM) subsequence.
The
AR subsequence
is responsible for reasoning and understanding. It contains language tokens as well as video and image tokens embedded by the ViT encoder.
All AR tokens are routed to a dedicated set of parameters in the transformer decoder layers.
The
diffusion subsequence
follows the AR subsequence and contains video and image tokens from the VAE encoder, as well as audio and action tokens. During generation, the model iteratively denoises the noisy diffusion tokens to produce the corresponding clean tokens. Diffusion tokens are routed to a separate parameter set from that used by AR tokens, while still interacting with AR tokens through joint attention in each of the transformer decoder layers.
For any given task, we apply the same format to arrange these tokens: (1) autoregressive tokens are placed before diffusion tokens; (2) within the diffusion subsequence, for each modality, clean conditioning tokens are placed before noisy diffusion tokens; and (3) within both the conditioning and diffusion subsequence, tokens are ordered by vision, audio, and action modality.
By using this unified format, Cosmos 3 can support various generation tasks, which we detail below.
2.2.2
Generation Mode
Cosmos 3 supports different modalities: language, vision, audio, and action.
We denote clean vision, audio, and action tokens as
v
v
,
s
s
, and
a
a
, respectively, and their noisy counterparts with tildes:
v
~
\tilde{v}
,
s
~
\tilde{s}
, and
a
~
\tilde{a}
.
Given these modalities, the supported generation modes are listed as follows:
•
Language.
For language generation, the input contains only the autoregressive subsequence, and the generation-specific diffusion parameters are not activated. Image and video inputs, if present, are embedded by the ViT encoder and placed in the autoregressive subsequence. In this setting, Cosmos 3 operates like a standard VLM.
•
Text-to-Image.
In this mode, the autoregressive subsequence contains the language tokens, while the diffusion subsequence contains the noisy target image tokens embedded by the VAE encoder. The entire sequence of tokens becomes:
𝐒
T2I
=
[
𝐒
AR
,
v
~
1
]
,
\mathbf{S}_{\mathrm{T2I}}=[\mathbf{S}_{\mathrm{AR}},\;\tilde{v}_{1}],
(3)
where
𝐒
AR
≜
[
l
1
,
…
,
l
n
,
⟨
EOS
⟩
,
⟨
BOG
⟩
]
\mathbf{S}_{\mathrm{AR}}\triangleq[l_{1},\ldots,l_{n},\langle\text{EOS}\rangle,\langle\text{BOG}\rangle]
is the AR prefix shared by all modes below (
l
1
,
…
,
l
n
l_{1},\ldots,l_{n}
are the language tokens;
⟨
EOS
⟩
\langle\text{EOS}\rangle
and
⟨
BOG
⟩
\langle\text{BOG}\rangle
are the end-of-sentence and begin-of-generation special tokens), and
v
~
1
\tilde{v}_{1}
is the noisy image token.
•
Text-to-Video (+Audio).
This mode is similar to Text-to-Image, but the diffusion subsequence contains the noisy target video tokens instead. When audio is (optionally) generated jointly, noisy audio tokens are appended after the noisy vision tokens. In summary, the packed sequence becomes:
𝐒
T2V
+
Audio
=
[
𝐒
AR
,
v
~
1
:
N
,
s
~
]
,
\mathbf{S}_{\mathrm{T2V+Audio}}=[\mathbf{S}_{\mathrm{AR}},\;\tilde{v}_{1:N},\;\tilde{s}],
(4)
where
N
N
is the number of latent video frames.
•
Image-to-Video/Video-to-Video (+Audio).
This mode introduces an initial conditioning image or a number of initial video frames, and the model generates the complete continuation conditioned on them and the text prompt. In the diffusion subsequence, the clean conditioning image or video tokens are followed by the noisy target video tokens:
𝐒
V2V
=
[
𝐒
AR
,
v
1
:
P
,
v
~
P
+
1
:
N
]
,
\mathbf{S}_{\mathrm{V2V}}=[\mathbf{S}_{\mathrm{AR}},\;v_{1:P},\;\tilde{v}_{P+1:N}],
(5)
where
P
P
is the number of conditioning latent frames. When
P
=
1
P=1
, the task becomes Image-to-Video, while
P
>
1
P>1
corresponds to Video-to-Video. When audio is also generated, the audio tokens are appended similarly to the Text-to-Video case.
•
Video transfer.
In this task, the input consists of a control video (
\eg
, edge, or depth) together with a text description, and the model generates the corresponding RGB video. The token layout is similar to that of Video-to-Video, with the control-video tokens used as conditioning tokens and the RGB video tokens used as noisy target tokens:
𝐒
Transfer
=
[
𝐒
AR
,
v
1
:
N
ctrl
,
v
~
1
:
N
]
,
\mathbf{S}_{\mathrm{Transfer}}=[\mathbf{S}_{\mathrm{AR}},\;v^{\mathrm{ctrl}}_{1:N},\;\tilde{v}_{1:N}],
(6)
where
v
ctrl
1
:
N
v^{\mathrm{ctrl}}_{1:N}
are the clean VAE-encoded tokens of the control video.
•
Action.
Cosmos 3 supports three generation modes for action—forward dynamics, inverse dynamics, and joint video-action prediction (policy). For a trajectory with consecutive video tokens, each action token
a
t
a_{t}
represents the transition from
v
t
−
1
v_{t{-}1}
to
v
t
v_{t}
.
Forward dynamics predicts future visual states conditioned on observed context and clean action tokens, while inverse dynamics infers the action tokens that explain an observed visual transition.
In policy mode, the model jointly predicts action and video tokens, enabling it to generate both the intervention and its expected visual consequence under the same sequence model.
The conditional directions are summarized in
Fig.
4
.
Figure 4
:
Action sequence configurations.
For a video-action data sample, Cosmos 3 constructs different training modes by varying which tokens are clean and which are noisy. The diagram shows a local temporal window in which action tokens lie between adjacent video tokens:
a
t
a_{t}
connects
v
t
−
1
v_{t{-}1}
to
v
t
v_{t}
, and
a
t
+
1
a_{t{+}1}
connects
v
t
v_{t}
to
v
t
+
1
v_{t{+}1}
. Forward dynamics mode denoises vision tokens conditioned on clean action tokens; inverse dynamics mode denoises action tokens conditioned on clean vision tokens; and video-action (policy) mode denoises both vision and action tokens. Language and special tokens are omitted for compactness.
2.3
Mixture-of-Transformers (MoT) Architecture
Cosmos 3 adopts a
Mixture-of-Transformers (MoT)
architecture that processes a unified sequence of tokens from different modalities. At the layer level, each transformer decoder layer contains two sets of parameters: one for reasoning tasks, which processes tokens from the AR subsequence (reasoner), and one for generation tasks, which processes tokens from the diffusion subsequence (generator). Although Cosmos 3 shares similarities with unified generation models such as
Deng et al. (2025)
in its decoder-layer structure, it differs in its training strategy, positional embeddings, and overall capabilities.
Figure 5
:
Mixture-of-Transformers (MoT) architecture of Cosmos 3.
Left:
a single transformer operates on one token sequence comprising the autoregressive (
AR
) and diffusion (
DM
) subsequences: AR carries discrete text tokens and, optionally, ViT-encoded vision tokens, ending with
<EOS>
and a begin-of-generation token
<BOG>
, while DM carries continuous tokens from their respective encoders, noise-perturbed during training. Here we visualize all input tokens as noisy for simplicity; for generation modes such as image-to-video or video transfer, clean conditioning tokens precede the noisy targets within DM; see
Sec.
2.2.2
.
Within each transformer block, AR tokens and DM tokens are processed by independent LayerNorms and MLPs (all co-initialized from a pre-trained VLM) and meet only at a shared self-attention operator. Let
𝐐
\mathbf{Q}
,
𝐊
\mathbf{K}
, and
𝐕
\mathbf{V}
be query, key, and value vectors in attention, where the subscript indicates which tower it is in.
𝐐
AR
\mathbf{Q}_{\mathrm{AR}}
attends causally over
𝐊
AR
,
𝐕
AR
\mathbf{K}_{\mathrm{AR}},\mathbf{V}_{\mathrm{AR}}
only, while
𝐐
DM
\mathbf{Q}_{\mathrm{DM}}
attends bidirectionally over the concatenated
[
𝐊
AR
;
𝐊
DM
]
[\mathbf{K}_{\mathrm{AR}};\mathbf{K}_{\mathrm{DM}}]
and
[
𝐕
AR
;
𝐕
DM
]
[\mathbf{V}_{\mathrm{AR}};\mathbf{V}_{\mathrm{DM}}]
.
In this way, diffusion is conditioned on the AR context, while AR remains autoregressively self-contained. Outputs are next-token predictions for
Reasoner and denoised tokens for Generator (trained in practice with a flow-matching objective predicting velocity; we show the clean target here for clarity).
Right:
the attention mask, causal for AR and full for diffusion.
2.3.1
Dual-Tower Layer Structure
A standard transformer decoder layer consists of a self-attention operation, a feed-forward network, and some normalization layers. Instead of processing all token types with the same parameters, the MoT design uses two pathways, as shown in
Fig.
5
. Each pathway is a standard transformer layer with its own parameters, including layer normalization modules, attention projection matrices, and feed-forward networks. The two pathways are both initialized from the weights of a pre-trained Vision-Language Model (VLM), allowing Cosmos 3 to inherit strong language and visual reasoning capabilities while learning to generate high-fidelity videos. During both training and inference, the AR subsequence at the front is routed to the reasoner tower, while the diffusion subsequence at the back is routed to the generator tower.
2.3.2
Dual-Stream Joint Attention
Although the two towers use independent parameters, tokens from the diffusion subsequence interact with the AR subsequence through a dual-stream joint attention operation. Here we denote the query, key, and value vectors of the AR and diffusion subsequences as
𝐐
AR
\mathbf{Q}_{\text{AR}}
,
𝐊
AR
\mathbf{K}_{\text{AR}}
,
𝐕
AR
\mathbf{V}_{\text{AR}}
,
𝐐
DM
\mathbf{Q}_{\text{DM}}
,
𝐊
DM
\mathbf{K}_{\text{DM}}
, and
𝐕
DM
\mathbf{V}_{\text{DM}}
, respectively.
Autoregressive subsequence attention.
Tokens in the AR subsequence attend only to tokens within the AR subsequence using
causal self-attention
; that is, each token can attend only to preceding tokens in the same sequence. This is fully consistent with the autoregressive property inherited from the VLM backbone, allowing the model to preserve the text-generation capability of the pre-trained VLM:
𝐎
AR
=
Attn
causal
⁡
(
𝐐
AR
,
𝐊
AR
,
𝐕
AR
)
.
\mathbf{O}_{\text{AR}}=\operatorname{Attn}_{\text{causal}}\!\bigl(\mathbf{Q}_{\text{AR}},\;\mathbf{K}_{\text{AR}},\;\mathbf{V}_{\text{AR}}\bigr).
(7)
Diffusion subsequence attention.
Tokens in the DM subsequence use
full bidirectional attention
, with the union of AR and DM tokens serving as the keys and values. This allows each diffusion token to freely attend to the text prompts from the autoregressive subsequence, as well as to all other conditional and diffusion tokens in the sequence, thereby maintaining temporal and spatial consistency:
𝐎
DM
=
Attn
full
⁡
(
𝐐
DM
,
[
𝐊
AR
;
𝐊
DM
]
,
[
𝐕
AR
;
𝐕
DM
]
)
,
\mathbf{O}_{\text{DM}}=\operatorname{Attn}_{\text{full}}\!\bigl(\mathbf{Q}_{\text{DM}},\;[\mathbf{K}_{\text{AR}};\,\mathbf{K}_{\text{DM}}],\;[\mathbf{V}_{\text{AR}};\,\mathbf{V}_{\text{DM}}]\bigr),
(8)
where
[
⋅
;
⋅
]
[\cdot\,;\cdot]
denotes concatenation along the sequence dimension. We note that AR tokens are never updated based on DM tokens, preserving the causal integrity of the conditioning pathway.
2.4
Multimodal Position Embedding
Position embeddings inject temporal and spatial structure into the attention mechanism, encouraging tokens to attend more strongly to semantically and geometrically relevant tokens, often nearby in space or time. Since Cosmos 3 jointly models language, vision, audio, and action tokens within a unified attention framework, designing a position-embedding scheme that generalizes consistently across modalities is inherently challenging. Inspired by 3D Multimodal RoPE (MRoPE)
(
Bai et al., 2025a
)
, we design a 3D MRoPE with absolute temporal indexing to align video, audio, and action tokens along the same physical temporal axis. The original 3D MRoPE divides the hidden dimension of each attention head into temporal, height, and width components, where the temporal component records only the discrete token index. This design is sufficient for image and video understanding tasks, but it is inadequate for our setting, where video, audio, and action tokens may be generated simultaneously at different frame or sampling rates. In this case, tokens from different modalities must be aligned to an absolute physical temporal axis. We first introduce the base formulation, which follows the original 3D MRoPE design, and then describe our extensions and modifications, especially our absolute temporal modulation, which aligns the absolute temporal axis.
2.4.1
Position Index Allocation
Autoregressive tokens.
For backward compatibility with language generation and image/video understanding models, position indices for all language tokens and ViT-encoded media tokens in the AR subsequence follow the original 3D MRoPE design. For language tokens,
t
=
h
=
w
t=h=w
is set to the same monotonically increasing value, reducing 3D MRoPE to standard 1D RoPE behavior. For tokens from the ViT encoder,
t
t
is shared by all tokens from the same frame, while the
h
h
and
w
w
indices vary independently according to the spatial location of each token. The allocation of the position index in the autoregressive subsequence is identical to the 3D MRoPE design in Qwen3-VL
(
Bai et al., 2025a
)
.
Figure 6
:
Illustrative coordinate assignment under 3D MRoPE.
Left:
A packed token sequence containing language, video (two frames,
2
×
2
2\times 2
spatial grid each), audio, and action tokens. Each token receives a
(
t
,
h
,
w
)
(t,h,w)
triplet. Language tokens use
t
=
h
=
w
t=h=w
; video tokens vary on all three axes; action and audio tokens use temporal coordinates only (
h
=
w
=
0
h=w=0
). A modality offset
k
k
separates the text and vision temporal ranges.
Right:
FPS modulation maps frame indices to scaled temporal positions so that equal real-world durations occupy equal position ranges at 16, 24, and 30 FPS, where 24 FPS is our base frame-per-second.
Diffusion tokens.
As illustrated in
Fig.
6
, video tokens vary across all three axes:
t
t
advances with the temporal latent frame index, while
h
h
and
w
w
tile over the spatial grid
(
0
​
…
​
H
−
1
,
0
​
…
​
W
−
1
)
(0\ldots H{-}1,\;0\ldots W{-}1)
independently per frame. Image tokens are treated as single-frame videos and vary only in
(
h
,
w
)
(h,w)
. Both spatial and temporal indices are reset to zero at the start of each vision segment, so the model treats
t
t
,
h
h
, and
w
w
as absolute within-video coordinates rather than positions in the global sequence. For example, in the video transfer task where the user provides a text prompt together with controlled video frames such as depth maps, both the clean control-video tokens and the noisy generated-video tokens start from the temporal offset of the last token in the autoregressive subsequence. All
audio tokens
and
action tokens
only carry temporal coordinates. The spatial indices are set to zero (
h
=
w
=
0
h=w=0
). For audio tokens, the temporal index advances with each audio hop; for action tokens, the temporal index advances with each sampling step.
Autoregressive and diffusion token margin.
In practice, we find that directly letting the diffusion tokens start from the temporal offset of the last autoregressive token leads to over-saturation and checkerboard artifacts in the initial video frames. This effect is especially pronounced in larger variants of Cosmos 3, such as the Super model. We hypothesize that this occurs because the last language token and the vision tokens from the first frame occupy adjacent temporal positions, resulting in nearly identical temporal embeddings. To address this issue, inspired by
Cao et al. (2025)
, we insert a fixed temporal gap between the autoregressive and diffusion subsequences, uniformly shifting the temporal indices of all the subsequent vision, audio, and action tokens. This creates a buffer in positional space that provides a clearer text-to-vision transition signal without requiring architectural changes or additional learnable embeddings. In all of our models, we set the gap to be
15000
15000
.
2.4.2
Absolute Temporal Modulation
A single unit step along the temporal dimension may correspond to different physical time intervals across modalities or data sources. For example, when encoding videos at 60 FPS and 24 FPS, respectively, a temporal-index increment for 24-FPS video tokens corresponds to a physical time interval that is 2.5 times longer than that of 60-FPS video tokens. Similar discrepancies also arise for action and audio tokens, where different data sources may use different sampling rates. FPS modulation is designed to align tokens with different temporal resolutions onto a shared physical temporal axis by modulating the effective size of each temporal increment.
We first define the temporal steps per second (TPS) to characterize the physical temporal resolution. For video tokens, TPS is given by the video frame rate divided by the temporal compression factor, which is 4 in our case due to the video VAE encoder. For audio tokens, TPS is computed as
TPS
audio
=
48000
1920
≈
25
\mathrm{TPS}_{\mathrm{audio}}=\frac{48000}{1920}\approx 25
(48 kHz, 1920 hop size). For action tokens, TPS is exactly the sampling frequency of the action data.
We then associate a unit length along the temporal dimension with a base TPS, denoted as
TPS
base
\mathrm{TPS}_{\mathrm{base}}
. For tokens in a given diffusion subsequence, we compute their corresponding TPS. When the temporal index needs to be increased by one unit step, the temporal increment
δ
​
t
\delta t
with the modulation is computed as
δ
​
t
=
TPS
base
TPS
.
\delta t=\frac{\mathrm{TPS}_{\mathrm{base}}}{\mathrm{TPS}}.
(9)
Since video constitutes the majority of our training data, and 24 FPS is the most common frame rate in our setting, we set
TPS
base
=
24
4
=
6
\mathrm{TPS}_{\mathrm{base}}=\frac{24}{4}=6
where
4
4
is our video tokenizer’s temporal compression ratio.
2.5
Model Variants
Cosmos 3 is trained at three model scales:
Edge
,
Nano
, and
Super
, spanning a wide range of computational budgets from on-device deployment to large datacenter inference.
Edge
is a 4B-parameter model built upon a dense 2B-parameter transformer,
Nano
is a 16B-parameter model built upon a dense 8B-parameter transformer, and
Super
is a 64B-parameter model built upon a dense 32B-parameter transformer. All variants are initialized from pre-trained vision-language models (VLMs) and adopt the Mixture-of-Transformers (MoT) architecture described above.
Tab.
2
summarizes the key architectural hyperparameters for each variant. Cosmos3-Nano and Cosmos3-Super models are released in this paper. Cosmos3-Edge model will be included in a later release.
Cosmos3-Edge
uses the design of a 2B dense transformer of
28
28
layers,
2048
2048
hidden size,
16
16
attention heads,
8
8
key-value heads, a head dimension of
128
128
, and
9216
9216
FFN dimension. We train the LLM from scratch using the Megatron codebase. The design of the LLM largely follows the Qwen3-1.7B architecture, with two notable differences: it removes QK normalization and uses ReLU-squared as the FFN activation, which is paired with the Edge FFN dimension reported in
Tab.
2
.
Cosmos3-Nano
adapts the Qwen3-VL 8B
(
Bai et al., 2025b
)
architecture, with
36
36
layers in the LLM, a hidden size of
4096
4096
,
32
32
attention heads, 8 key-value heads, a head dimension of
128
128
, and a FFN dimension of
12,288
12{,}288
.
Cosmos3-Super
adapts the Qwen3-VL 32B
(
Bai et al., 2025b
)
architecture, with
64
64
layers in the LLM, a hidden size of
5120
5120
,
64
64
attention heads,
8
8
key-value heads, a head dimension of
128
128
, and a FFN dimension of
25,600
25{,}600
.
Table 2
:
Cosmos 3 MoT model variants.
All models share the dual-tower MoT architecture. “LLM Layers” refers to the number of transformer decoder layers; each layer carries independent parameter sets for the reasoner and generator towers.
Edge
uses a dense 2B parameter transformer trained from scratch, while
Nano
and
Super
are initialized from pre-trained Qwen3-VL weights.
Variant
LLM Layers
Hidden Dim
Attn Heads
KV Heads
Head Dim
FFN Dim
Cosmos3-Edge
28
2,048
16
8
128
9,216
Cosmos3-Nano
36
4,096
32
8
128
12,288
Cosmos3-Super
64
5,120
64
8
128
25,600
