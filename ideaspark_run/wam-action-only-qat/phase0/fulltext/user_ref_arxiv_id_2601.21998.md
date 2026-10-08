# [user-supplied anchor] 2601.21998

paper_id: user_ref:arxiv_id:2601.21998
tier: U
source_used: html_arxiv
warning: none

## Intro

Figure 1
:
LingBot-VA
: An Autoregressive World Model for Robotic Manipulation.
(1)
Pretraining:
LingBot-VA
is pretrained on diverse in-the-wild videos and robot action data, enabling strong generalization across scenes and objects.
(2)
Comprehensive Evaluation:
We conduct extensive experiments on real-world tasks (long-horizon, deformable objects, and precision manipulation) and simulation benchmarks, significantly outperforming state-of-the-art methods including
π
0.5
\pi_{0.5}
.
(3)
Versatile Capabilities:
Beyond policy learning, our model supports visual dynamics prediction and inverse dynamics inference from robot videos.
(4)
Emergent Properties:
Our causal world modeling approach exhibits long-range temporal memory and strong few-shot adaptation ability.
Vision-Language-Action (VLA) models have emerged as a promising paradigm for general-purpose robotic manipulation
[
12
,
11
,
34
,
7
]
, demonstrating impressive capabilities in grounding linguistic instructions into visual perceptions across diverse objects and unstructured environments.
However, beneath their apparent success lies a significant challenge:
representation entanglement
.
Most existing VLAs adopt a feedforward paradigm that maps current observations to action sequences
[
17
,
91
]
, requiring a single neural network to simultaneously learn visual scene understanding, physical dynamics, and motor control from a unified supervision signal.
This entanglement can create a bottleneck—the model must compress heterogeneous knowledge, ranging from high-dimensional visual semantics to low-dimensional motor commands, into a shared representation space.
This often leads to limited sample efficiency and suboptimal generalization.
Without explicit modeling of environmental evolution
[
25
,
82
,
26
]
, reactive policies may rely on pattern matching rather than a principled understanding of physical dynamics.
Recent attempts to bring world modeling into robotic policies span interactive neural simulators (e.g., UniSim
[
86
]
), chunk-based video-action diffusion models (e.g., UVA
[
40
]
and UWM
[
97
]
), and offline video generators for subgoal synthesis (e.g. Gen2Act
[
4
]
, Act2Goal
[
95
]
).
While conceptually appealing, these approaches face three primary limitations for effective closed-loop control.
First,
the reactivity gap
: chunk/open-loop generation often rolls out long segments without incorporating real-time feedback, making it hard to adapt to disturbances.
Second,
limited long-term memory
: chunk-wise generation can introduce inconsistencies over long horizons when history is not persistently cached.
Third,
causality
: bidirectional attention within a segment allows future tokens to influence past predictions, which diverges from the causal nature of physical reality where the present depends only on the past.
These observations motivate an autoregressive formulation for robust closed-loop reasoning.
We propose
LingBot-VA
, an
autoregressive diffusion
world model that addresses these limitations through a unified video-action framework.
Unlike autoregressive language models that predict discrete tokens, our model operates in a continuous latent space via flow matching
[
46
,
50
]
, autoregressively generating chunks of video and action representations through iterative denoising.
While our approach conceptually separates visual dynamics prediction and action decoding
[
22
,
27
]
, the key architectural insight is to
interleave
video and action tokens into a single autoregressive sequence.
Both modalities are jointly processed through a Mixture-of-Transformers (MoT) architecture
[
43
]
with shared attention.
Within this unified autoregressive generation process, latent imagination and action inference occur jointly: at each autoregressive step, the model generates predicted future visual states through iterative denoising while simultaneously decoding the corresponding actions, allowing both streams to mutually condition on one another.
This integration, built upon a large-scale pretrained video diffusion backbone
[
79
]
, offers several advantages:
(i) Reactive AR loop
: because video and action tokens form a unified sequence, each autoregressive step allows the system to recalibrate based on the latest real-world observation, enabling timely adjustments to both the predicted future and motor commands;
(ii) Persistent context through KV-cache
: the cached key-value pairs preserve the interleaved video-action trajectory, providing a rich context that helps mitigate temporal drift;
(iii) Causal consistency
: causal attention masking over the unified sequence ensures that both predicted visual states and action commands are governed by preceding states, respecting the temporal arrow of physical dynamics.
By incorporating real-world observations at each step, this formulation helps mitigate the
distribution drift
that often affects open-loop methods in long-horizon tasks.
A primary challenge in deploying large-scale autoregressive video-action models is inference latency; generating high-fidelity video tokens through iterative denoising is computationally intensive.
We address this through two complementary strategies.
First, we introduce
Noisy History Augmentation
, a training scheme that enables
partial denoising
at inference time.
The key insight is that action decoding does not always require pixel-perfect reconstruction; instead, it can rely on robust semantic structures.
By training the action decoder to predict from partially noisy latent representations, we significantly reduce the computational overhead while maintaining precise action prediction.
Second, we design an
asynchronous coordination
pipeline that overlaps computation with execution: while the robot executes current actions, the world model predicts future visual states and plans subsequent sequences.
This parallelized architecture, combined with variable chunk-size training, facilitates high-frequency closed-loop control without compromising prediction quality.
We evaluate
LingBot-VA
across diverse manipulation tasks in both simulation and real-world environments.
Our method demonstrates competitive performance compared to state-of-the-art VLA policies, particularly in long-horizon tasks requiring temporal consistency.
Our contributions are summarized as follows:
•
Autoregressive Video-Action World Modeling:
We introduce an autoregressive diffusion framework that
architecturally unifies
visual dynamics prediction and action inference within a single interleaved sequence while maintaining their
conceptual distinction
. This formulation supports persistent memory through KV cache and causal consistency via attention masking.
•
Mixture-of-Transformers Architecture with Asynchronous Execution:
We design a dual-stream MoT architecture with asymmetric capacity and introduce a partial denoising strategy combined with asynchronous coordination to enable efficient robotic control.
•
Superior Long-Horizon and Precision Performance:
Extensive real-world and simulation experiments demonstrate consistent state-of-the-art performance, with particularly strong improvements on long-horizon and high-precision manipulation tasks. Our method also achieves significantly improved sample efficiency and strong generalization to novel scenes and object configurations.

## Method

3.1
Problem Statement & Approach Overview
We study robotic manipulation as a sequential decision-making problem under partial observability.
At each timestep
t
t
, the agent receives a visual observation
o
t
∈
𝒪
o_{t}\in\mathcal{O}
and executes
an action
a
t
∈
𝒜
a_{t}\in\mathcal{A}
, which induces a transition in the underlying physical world and
produces the next observation
o
t
+
1
o_{t+1}
.
Vision-Language-Action (VLA) Policies.
Most existing VLA policies learn a direct, reactive mapping from observation history to actions:
a
t
∼
π
θ
(
⋅
∣
o
t
)
,
a_{t}\sim\pi_{\theta}(\cdot\mid o_{t}),
(5)
through imitation learning on robot demonstration data.
While this end-to-end approach has shown impressive results, it suffers from a fundamental
coupling problem: the model must simultaneously learn visual scene understanding, physical dynamics,
and motor control from a single supervision signal of paired observations and actions.
This entanglement leads to poor sample efficiency and limited generalization, as the model struggles
to disentangle visual reasoning from action prediction without explicit dynamics modeling.
Our Approach.
Unlike VLA policies that directly learn action distributions, we adopt a world modeling perspective:
instead of learning
π
⁡
(
a
t
∣
o
t
)
\pi(a_{t}\mid o_{t})
, we predict how the visual world will evolve,
then infer actions based on these predictions.·
Our approach operates in two stages:
(Stage 1) Visual dynamics prediction:
\displaystyle\text{(Stage 1) Visual dynamics prediction:}\quad
o
t
+
1
∼
p
θ
(
⋅
∣
o
≤
t
)
,
\displaystyle o_{t+1}\sim p_{\theta}(\cdot\mid o_{\leq t}),
(6)
(Stage 2) Inverse dynamics:
\displaystyle\text{(Stage 2) Inverse dynamics:}\quad
a
t
∼
g
ψ
(
⋅
∣
o
t
,
o
t
+
1
)
.
\displaystyle a_{t}\sim g_{\psi}(\cdot\mid o_{t},o_{t+1}).
Stage 1 learns to predict future visual observations given observation history.
Stage 2 uses an inverse dynamics model to decode actions from desired visual transitions.
This decomposition enables Stage 1 to leverage large-scale video data for learning physical priors,
while Stage 2 only requires robot demonstrations to ground visual predictions in executable actions.
Method Overview.
Figure
2
illustrates the details of our framework.
Our method consists of three key components, detailed in the following subsections:
(§
3.2
) Autoregressive Video-Action World Modeling
describes how we
model visual dynamics in latent space and decode actions from predicted state transitions—this is the
core formulation
of our approach;
(§
3.3
) LingBot-VA: Unified Architecture & Training
presents our unified model
for video-action pretraining, including the architecture design and training objective—this is the
instantiation
of our formulation;
(§
3.4
) Real-time Deployment & Asynchronous Inference
introduces our deployment
strategy that enables real-time control through parallelized prediction and execution—this is the
practical realization
for robotic control.
3.2
Autoregressive Video-Action World Modeling
Previous video world models either focus on open-ended video prediction
[
54
]
or learn action-conditioned interactive environments
[
13
,
56
]
primarily for game or simulation domains, which may not directly transfer to precise robotic manipulation.
To leverage rich visual dynamics priors from video data for robot manipulation, we propose a unified video-action world modeling framework that jointly models visual observations and robot actions within a single autoregressive process.
Unlike prior approaches that either decouple video prediction from action inference
[
16
,
27
]
or rely on bidirectional diffusion within segments
[
97
]
, our method unifies video and action within a single
causal autoregressive
framework, enabling persistent memory through KV cache and seamless integration of real-time observations.
World Dynamics with Autoregressive Modeling.
Recent world models for robotics often adopt bidirectional video generation approaches
[
20
,
42
,
4
,
24
]
or learn interactive simulators
[
86
]
, which face fundamental limitations for closed-loop control.
Open-loop methods that generate entire long sequences in one shot incur prohibitive computational cost
and cannot incorporate real-time feedback for error correction.
Chunk-based diffusion methods that generate video segments sequentially
[
22
,
97
]
suffer from two critical issues:
(1) they lack persistent memory across chunks, as each chunk is generated independently without access to the full history, leading to temporal inconsistencies and drift over long horizons;
(2) the bidirectional attention within each chunk violates causality, preventing seamless integration with real-time observations during execution.
The physical world, however, is inherently causal and autoregressive: the present state depends only on the past, and we cannot observe the future before it occurs.
This fundamental property motivates our autoregressive world modeling approach, which offers three critical advantages over chunk-based diffusion for robotic control:
(1) Persistent Memory
: by explicitly conditioning on the complete observation history through causal attention and KV cache, the model maintains long-term context and temporal coherence across the entire trajectory, avoiding the “amnesia” problem of chunk-based methods;
(2) Causal Consistency
: the unidirectional dependency structure naturally aligns with closed-loop execution, where new observations can be seamlessly incorporated as they arrive;
(3) Efficiency
: chunk-wise prediction with parallel generation within each chunk balances computational efficiency with autoregressive flexibility, enabling high-frequency control with real-time error correction.
We formalize this as an autoregressive process: at each step, the world model predicts the next chunk of
K
K
video frames using conditional flow matching:
o
t
+
1
:
t
+
K
∼
p
θ
(
⋅
∣
o
≤
t
)
,
o_{t+1:t+K}\sim p_{\theta}(\cdot\mid o_{\leq t}),
(7)
where tokens within each chunk are generated in parallel via bidirectional attention, while maintaining causal structure across chunks.
This chunk-wise formulation balances generation efficiency with autoregressive flexibility for closed-loop correction.
Video-Action State Encoding.
Operating directly on pixel-level video observations is computationally prohibitive due to the high dimensionality and redundancy of raw visual data.
We leverage a causal video VAE
[
79
]
to compress visual observations into compact latent tokens
z
t
=
E
⁡
(
o
t
∣
o
<
t
)
∈
ℝ
N
×
C
z_{t}=E(o_{t}\mid o_{<t})\in\mathbb{R}^{N\times C}
, where
N
N
is the number of spatial tokens after passing into video VAE, and
C
C
is the channel number.
By conditioning on previous latent states, the encoder maintains temporal coherence while processing observations sequentially, naturally aligning with our autoregressive world modeling framework.
To align robot actions with visual tokens, we project action vectors to token embeddings
a
t
∈
ℝ
D
a_{t}\in\mathbb{R}^{D}
via a lightweight MLP
ϕ
⁡
(
⋅
)
\phi(\cdot)
where
D
D
is the dimension of the video token after patchfication, enabling unified interleaving of visual and action tokens as in prior approaches
[
22
,
5
]
.
Latent Video State Transition.
While standard video generation models predict future frames based solely on visual history, robotic manipulation requires accounting for the embodiment’s physical state and interaction with the environment.
During deployment, the robot’s state evolves through continuous interaction: each action modifies the embodiment’s configuration (e.g., gripper position, joint angles), which in turn influences how the scene evolves.
In many manipulation settings, actions encode absolute pose information (e.g., end-effector poses in world coordinates), so the action history
a
<
t
a_{<t}
effectively captures the trajectory of the embodiment’s configuration.
Conditioning on action history thus provides knowledge of how the robot has moved and interacted with objects, consistent with prior action-conditioned video/world models
[
86
,
22
,
97
]
.
We extend our autoregressive formulation to condition on both observation and action histories:
z
t
+
1
:
t
+
K
∼
p
θ
(
⋅
∣
z
≤
t
,
a
<
t
)
,
z_{t+1:t+K}\sim p_{\theta}(\cdot\mid z_{\leq t},a_{<t}),
(8)
where
z
t
z_{t}
is the latent visual state and
a
t
a_{t}
is the action token.
This enables the world model to ground predictions in the embodiment’s state, ensuring that predicted observations reflect the robot’s physical interaction with the scene.
Inverse Dynamics for Action Decoding.
Once the world model predicts future visual states, we leverage these predictions to plan actions.
Rather than directly predicting actions from current observations, we employ an inverse dynamics model that infers actions by conditioning on desired future observations, enabling the policy to reason about
what action leads to a desired visual outcome
.
However, simply conditioning on the current and next states
(
z
t
,
z
t
+
1
)
(z_{t},z_{t+1})
is insufficient for accurate action prediction.
The action history
a
<
t
a_{<t}
encodes the embodiment’s state trajectory for determining feasible actions, while the observation history
z
<
t
z_{<t}
provides temporal context for multi-step interactions (e.g., whether an object was previously grasped).
We therefore formulate inverse dynamics as:
a
t
:
t
+
K
−
1
∼
g
ψ
(
⋅
∣
z
^
t
+
1
:
t
+
K
,
z
≤
t
,
a
<
t
)
,
a_{t:t+K-1}\sim g_{\psi}(\cdot\mid\hat{z}_{t+1:t+K},z_{\leq t},a_{<t}),
(9)
where the inverse dynamics model
g
ψ
g_{\psi}
takes as input the predicted chunk of visual states
z
^
t
+
1
:
t
+
K
\hat{z}_{t+1:t+K}
inferred by Eq.
8
, observation history
z
≤
t
z_{\leq t}
, and action history
a
<
t
a_{<t}
.
This mirrors recent IDM-based policies
[
20
,
1
,
55
,
22
,
73
]
that leverage future targets to infer feasible actions while maintaining consistency with embodiment dynamics.
3.3
LingBot-VA: Unified Architecture & Training
Architecture.
To jointly model video and action generation, we leverage a dual-stream diffusion transformer architecture that performs conditional flow matching for autoregressive prediction.
Our model consists of two parallel transformer backbones: a video stream initialized from Wan2.2-5B (a large-scale pretrained video generation model with dimension
d
v
d_{v}
[
79
]
), and an action stream with same depth but significantly smaller width
d
a
≪
d
v
d_{a}\ll d_{v}
.
This asymmetric design is motivated by the observation that action distributions are inherently simpler than visual data requiring fewer parameters to model effectively while maintaining expressive capacity for visual dynamics.
Video Sparsification.
Video frames exhibit significant temporal redundancy, especially in robotic manipulation where scenes evolve gradually.
We sparsify the video sequence by temporally downsampling frames by a factor of
τ
=
4
\tau=4
, reducing visual tokens while improving efficiency
[
5
]
.
Since actions evolve at higher frequency than visual changes, we interleave the downsampled video tokens with action tokens in temporal order: for each video frame
o
t
o_{t}
, we associate
τ
\tau
consecutive actions
{
a
t
,
1
,
a
t
,
2
,
…
,
a
t
,
τ
}
\{a_{t,1},a_{t,2},\ldots,a_{t,\tau}\}
, forming a unified sequence
[
z
t
,
a
t
,
1
,
a
t
,
2
,
…
,
a
t
,
τ
,
z
t
+
1
,
…
]
[z_{t},a_{t,1},a_{t,2},\ldots,a_{t,\tau},z_{t+1},\ldots]
for joint modeling.
This design means that predicting
K
K
video frames corresponds to generating
τ
​
K
\tau K
actions, enabling high-frequency control while maintaining efficient video generation.
Mixture-of-Transformer Block.
To enable interaction while preserving modality-specific feature spaces, we employ a Mixture-of-Transformers (MOT) architecture
[
5
,
43
,
19
]
, where video and action tokens are processed by separate transformer blocks at each layer, then fused via cross-modal attention
[
5
]
.
At each layer, the video and action streams independently compute their query, key, and value matrices using separate QKV projection matrices, maintaining distinct feature spaces for each modality.
To align dimensions for cross-modal fusion, action tokens are first projected to the video dimension via a linear layer, participate in joint self-attention, then projected back to their original dimension via a residual connection that preserves the action-specific representations.
This MOT design allows video and action to mutually influence each other through attention while maintaining separate parameterizations, preventing interference between modality-specific feature representations.
For action decoding, the final action stream outputs are mapped to low-dimensional action vectors via a linear projection head.
Action Network Initialization.
Proper initialization of the action stream is critical for training stability and convergence.
We find that training the action network from scratch leads to unstable optimization and slow convergence, as the action tokens’ output distribution initially diverges significantly from the video distribution, disrupting the joint attention mechanism.
To address this, we initialize the action network weights by interpolating the pretrained video weights according to the action dimension, then apply a scaling factor
α
=
d
v
/
d
a
\alpha=\sqrt{d_{v}/d_{a}}
to preserve output variance, where
d
v
d_{v}
and
d
a
d_{a}
are the video and action dimensions.
This initialization strategy ensures that action tokens start with output distributions comparable to video tokens, stabilizing early-stage training and accelerating convergence.
Variable Chunk Size Training.
To enable flexible deployment, we randomly sample the chunk size
K
K
from a predefined range during training.
By training with variable chunk sizes (e.g.,
K
∈
[
1
,
8
]
K\in[1,8]
), the model learns to generate coherent predictions across different temporal horizons.
At inference time, this allows freely selecting the chunk size to balance computational efficiency and planning horizon—larger chunks reduce the number of autoregressive steps but require longer per-step computation, while smaller chunks enable more frequent closed-loop correction.
In our experiments, we use
K
=
4
K=4
for deployment as a practical trade-off.
Teacher Forcing for Unified Video-Action Training.
In §
3.2
, we formulated both visual dynamics prediction (Eq. 7) and inverse dynamics (Eq. 8) as autoregressive modeling problems, where each prediction conditions on the history of observations and actions.
This unified autoregressive formulation enables a natural training strategy: we can treat the interleaved video-action sequence as a single unified sequence and train the model using standard next-token prediction, analogous to language modeling in NLP
[
76
]
.
Figure 3
:
Teacher Forcing Attention Mask
: Causal attention mask for unified video-action pretraining. Each token can only attend to preceding tokens in the temporal sequence.
Specifically, given an episode with interleaved tokens, we train the model to predict each token conditioned on all preceding tokens in the sequence.
This is implemented via teacher forcing: during training, we use ground-truth tokens from the dataset as context for predicting subsequent tokens, rather than model-generated predictions.
The causal dependency structure is enforced through attention masking (Figure
3
)—each token can only attend to tokens that appear earlier in the temporal sequence.
Importantly, teacher forcing is particularly well-suited for robotic manipulation: unlike pure generative modeling where it leads to train-test distribution mismatch, robot policies naturally retrieve real-world observations during deployment, directly matching the training regime.
This formulation offers two key benefits: (1) unifying video and action prediction under a single training objective enables end-to-end learning of world dynamics and action inference; (2) by processing episodes in parallel with causal attention masking, we efficiently optimize both components across all timesteps in a single forward pass.
Noisy History Augmentation.
The primary bottleneck during inference remains video token generation—the number of video tokens are much larger than action tokens, and each requires multiple denoising steps through the flow matching process.
To address this, we introduce a noise augmentation strategy during training that enables
partial denoising
at test time.
The key insight is that action prediction does not require fully denoised video representations; the inverse dynamics model can learn to extract action-relevant information from partially noisy video states.
Specifically, during training, we randomly augment the video history
z
≤
t
z_{\leq t}
with noise following the same interpolation scheme as flow matching:
z
~
≤
t
=
{
(
1
−
s
aug
)
​
ϵ
+
s
aug
​
z
≤
t
,
p
=
0.5
,
s
aug
∈
[
0.5
,
1
]
,
ϵ
∼
𝒩
⁡
(
0
,
I
)
z
≤
t
,
1
−
p
=
0.5
\tilde{z}_{\leq t}=\begin{cases}(1-s_{\text{aug}})\epsilon+s_{\text{aug}}z_{\leq t},&p=0.5,\quad s_{\text{aug}}\in[0.5,1],\;\epsilon\sim\mathcal{N}(0,I)\\
z_{\leq t},&1-p=0.5\end{cases}
(10)
This augmentation trains the action decoder to predict actions from partially noisy video representations.
At inference time, this enables a significant speedup: instead of fully denoising video tokens from
s
=
0
s=0
to
s
=
1
s=1
, we only need to denoise to
s
=
0.5
s=0.5
, halving the number of denoising steps for video generation while maintaining action prediction quality.
Training Objective.
We jointly optimize both video and action using flow matching with the noisy history augmentation described above.
For video tokens
z
t
z_{t}
, the dynamics loss supervises velocity field prediction conditioned on (potentially noisy) history:
ℒ
dyn
=
𝔼
t
,
s
,
z
t
+
1
,
ϵ
​
[
‖
v
θ
​
(
z
t
+
1
(
s
)
,
s
,
z
~
≤
t
,
a
<
t
|
c
)
−
z
˙
t
+
1
(
s
)
‖
2
]
,
\mathcal{L}_{\mathrm{dyn}}=\mathbb{E}_{t,s,z_{t+1},\epsilon}\left[\|v_{\theta}(z_{t+1}^{(s)},s,\tilde{z}_{\leq t},a_{<t}|c)-\dot{z}_{t+1}^{(s)}\|^{2}\right],
(11)
where
s
∈
[
0
,
1
]
s\in[0,1]
is flow time,
z
t
+
1
(
s
)
=
(
1
−
s
)
​
ϵ
+
s
​
z
t
+
1
z_{t+1}^{(s)}=(1-s)\epsilon+sz_{t+1}
with
ϵ
∼
𝒩
⁡
(
0
,
I
)
\epsilon\sim\mathcal{N}(0,I)
,
z
˙
t
+
1
(
s
)
=
z
t
+
1
−
ϵ
\dot{z}_{t+1}^{(s)}=z_{t+1}-\epsilon
,
z
~
≤
t
\tilde{z}_{\leq t}
is the augmented history (Eq. 10), and
c
c
is the language instruction.
For action tokens
a
t
a_{t}
, the inverse dynamics loss conditions on current and next observations:
ℒ
inv
=
𝔼
t
,
s
,
a
t
,
ϵ
​
[
‖
v
ψ
​
(
a
t
(
s
)
,
s
,
z
~
≤
t
+
1
,
a
<
t
|
c
)
−
a
˙
t
(
s
)
‖
2
]
,
\mathcal{L}_{\mathrm{inv}}=\mathbb{E}_{t,s,a_{t},\epsilon}\left[\|v_{\psi}(a_{t}^{(s)},s,\tilde{z}_{\leq t+1},a_{<t}|c)-\dot{a}_{t}^{(s)}\|^{2}\right],
(12)
where
a
t
(
s
)
=
(
1
−
s
)
​
ϵ
+
s
​
a
t
a_{t}^{(s)}=(1-s)\epsilon+sa_{t}
with
ϵ
∼
𝒩
⁡
(
0
,
I
)
\epsilon\sim\mathcal{N}(0,I)
,
z
~
t
,
z
~
t
+
1
\tilde{z}_{t},\tilde{z}_{t+1}
are the (potentially noisy) current and next video tokens, and
c
c
is the language instruction.
The complete objective is
ℒ
=
ℒ
dyn
+
λ
​
ℒ
inv
\mathcal{L}=\mathcal{L}_{\mathrm{dyn}}+\lambda\mathcal{L}_{\mathrm{inv}}
.
3.4
Real-time Deployment & Asynchronous Inference
KV Cache for Efficient Autoregressive Inference.
Our autoregressive formulation naturally enables KV cache acceleration during inference.
Since each prediction step conditions on the history of observations and actions, we cache the key-value pairs from previous tokens to avoid redundant computation.
At each autoregressive step, only the new tokens (current observation and predicted actions) require full attention computation, while cached history tokens are reused.
Algorithm
1
describes the complete inference procedure with KV cache.
Algorithm 1
KV Cache Inference
1:
Initial observation
o
0
o_{0}
, chunk size
K
K
, KV cache
𝒞
\mathcal{C}
2:
z
0
←
E
⁡
(
o
0
)
z_{0}\leftarrow E(o_{0})
,
𝒞
←
{
z
0
}
\mathcal{C}\leftarrow\{z_{0}\}
3:
t
←
0
t\leftarrow 0
4:
loop
5:
Sample
ϵ
∼
𝒩
⁡
(
0
,
I
)
\epsilon\sim\mathcal{N}(0,I)
⊳
\triangleright
Generate video chunk (integrate to
s
=
0.5
s=0.5
)
6:
z
~
t
+
1
:
t
+
K
←
ϵ
+
∫
0
0.5
v
θ
(
z
t
+
1
:
t
+
K
(
s
)
,
s
∣
𝒞
)
d
s
\tilde{z}_{t+1:t+K}\leftarrow\epsilon+\int_{0}^{0.5}v_{\theta}(z^{(s)}_{t+1:t+K},s\mid\mathcal{C})\,ds
7:
Sample
ϵ
∼
𝒩
⁡
(
0
,
I
)
\epsilon\sim\mathcal{N}(0,I)
⊳
\triangleright
Generate action chunk (integrate to
s
=
1
s=1
)
8:
a
t
:
t
+
K
−
1
←
ϵ
+
∫
0
1
v
ψ
(
a
t
:
t
+
K
−
1
(
s
)
,
s
∣
z
~
t
:
t
+
K
,
𝒞
)
d
s
a_{t:t+K-1}\leftarrow\epsilon+\int_{0}^{1}v_{\psi}(a^{(s)}_{t:t+K-1},s\mid\tilde{z}_{t:t+K},\mathcal{C})\,ds
9:
for
i
=
t
i=t
to
t
+
K
−
1
t+K-1
do
10:
Execute
a
i
a_{i}
, receive
o
i
+
1
o_{i+1}
⊳
\triangleright
Execute and collect observations
11:
z
i
+
1
←
E
⁡
(
o
i
+
1
)
z_{i+1}\leftarrow E(o_{i+1})
12:
end
for
13:
𝒞
←
𝒞
∪
{
z
t
+
1
:
t
+
K
,
a
t
:
t
+
K
−
1
}
\mathcal{C}\leftarrow\mathcal{C}\cup\{z_{t+1:t+K},a_{t:t+K-1}\}
⊳
\triangleright
Update KV cache
14:
t
←
t
+
K
t\leftarrow t+K
15:
end loop
Asynchronous Prediction and Execution.
Despite the efficiency gains from KV cache and partial denoising, autoregressive prediction still incurs non-negligible latency that can violate real-time control requirements.
To address this, we introduce an asynchronous inference strategy that pipelines action prediction with execution, effectively hiding prediction latency.
We illustrate the difference between synchronous and asynchronous inference in
Fig.
4
.
The key insight is to overlap computation with execution (
Fig.
4
B): While the robot executes the current action chunk
a
t
a_{t}
, the model simultaneously predicts the subsequent action chunk
a
t
+
1
a_{t+1}
conditioned on the most recent real observation
z
t
−
1
z_{t-1}
(received after the execution of
a
t
−
1
a_{t-1}
).
For simplicity, we use
z
t
z_{t}
to denote latent observations (ignoring the video VAE compression) instead of
o
t
o_{t}
in this section.
We discard all history data before timestamp
t
−
1
t-1
and use the hat notation
^
\hat{\penalty\ }
to mark predicted visual content. Consequently, the model’s active context is limited to the executed action chunk
a
t
−
1
a_{t-1}
, the recent ground-truth observation
z
t
−
1
z_{t-1}
, the currently executing action
a
t
a_{t}
, and its corresponding visual forecast
z
^
t
\hat{z}_{t}
. A naive auto-regressive implementation (
Fig.
4
B-1) is to store these tokens into the KV cache and predict
z
^
t
+
1
\hat{z}_{t+1}
. However, we observed that such a design frequently leads to open-loop degradation and trajectory drift. Because the video generative model inherently favors temporal smoothness, it tends to "continue" the hallucinated video
z
^
t
\hat{z}_{t}
while ignoring the critical physical feedback provided by the real observation
z
t
−
1
z_{t-1}
, eventually causing the model to lose its capacity to react to the environment.
To mitigate this, we introduce a Forward Dynamics Model (FDM) grounded step into our inference pipeline (
Fig.
4
B-2). Instead of relying on stale forecasts, we replace it by executing a forward dynamics pass: the model uses the recent feedback
z
t
−
1
z_{t-1}
and "imagines" the resulting visual state
z
t
z_{t}
after applying action
a
t
a_{t}
. By caching this feedback-grounded prediction instead of a stale forecast, we force the model to re-align with environmental feedback before predicting
z
t
+
1
z_{t+1}
. This design enhances our asynchronous algorithm into a robust closed-loop system, enabling the robot to effectively perceive and react to real-world changes.
Algorithm
2
formalizes this asynchronous pipeline. During post training, we additionally incorporate a forward dynamics prediction loss:
ℒ
fdm
=
𝔼
t
,
s
,
z
^
t
+
1
,
ϵ
​
[
‖
v
ψ
​
(
z
~
t
+
1
,
s
,
z
t
,
a
t
,
z
~
<
t
,
a
^
<
t
|
c
)
−
z
˙
t
+
1
(
s
)
‖
2
]
,
\mathcal{L}_{\mathrm{fdm}}=\mathbb{E}_{t,s,\hat{z}_{t+1},\epsilon}\left[\|v_{\psi}(\tilde{z}_{t+1},s,z_{t},a_{t},\tilde{z}_{<t},\hat{a}_{<t}|c)-\dot{z}_{t+1}^{(s)}\|^{2}\right],
(13)
Figure 4
:
Asynchronous pipeline design overview
: The traditional synchronous pipeline (A) suffers from delays caused by blocked computations, while the asynchronous pipeline (B) addresses this issue by enabling parallel computation and execution. However, a naive asynchronous implementation (B-1) relies on outdated visual predictions. In contrast, we improve and refine asynchronous prediction through forward dynamic prediction (B-2), which updates stale predictions with recent real-world observations.
Algorithm 2
Asynchronous Inference and Execution
1:
Initial observation
o
0
o_{0}
, chunk size
K
K
, KV cache
𝒞
\mathcal{C}
2:
z
0
←
E
⁡
(
o
0
)
z_{0}\leftarrow E(o_{0})
;
𝒞
←
{
z
0
}
\mathcal{C}\leftarrow\{z_{0}\}
3:
z
~
1
:
K
,
a
0
:
K
−
1
←
Predict
(
𝒞
)
\tilde{z}_{1:K},a_{0:K-1}\leftarrow\textsc{Predict}(\mathcal{C})
⊳
\triangleright
Cold Start
4:
ObsQueue
←
∅
\texttt{ObsQueue}\leftarrow\emptyset
⊳
\triangleright
Thread-safe queue for incoming real observations
5:
t
←
0
t\leftarrow 0
6:
loop
7:
parallel
:
8:
Branch A: Robot Execution
9:
async
Executor
(
a
t
:
t
+
K
−
1
,
ObsQueue
)
\textsc{Executor}(a_{t:t+K-1},\texttt{ObsQueue})
⊳
\triangleright
Execute pre-computed actions
10:
Branch B: Inference with FDM Grounding
11:
if
t
>
0
t>0
then
12:
o
t
−
K
+
1
:
t
←
ObsQueue.dequeue
(
)
o_{t-K+1:t}\leftarrow\texttt{ObsQueue.dequeue}()
⊳
\triangleright
Get real observation
13:
z
t
−
K
+
1
:
t
←
E
(
o
t
−
K
+
1
:
t
)
z_{t-K+1:t}\leftarrow E(o_{t-K+1:t})
14:
𝒞
←
𝒞
∪
{
z
t
−
K
+
1
,
t
,
a
t
−
K
:
t
−
1
}
\mathcal{C}\leftarrow\mathcal{C}\cup\{z_{t-K+1,t},a_{t-K:t-1}\}
⊳
\triangleright
Cache feedback
15:
end if
16:
𝒞
tmp
←
𝒞
∪
{
a
t
:
t
+
K
−
1
}
\mathcal{C}_{\text{tmp}}\leftarrow\mathcal{C}\cup\{a_{t:t+K-1}\}
⊳
\triangleright
Cache action being executed
17:
z
t
+
1
:
t
+
K
←
FDM
(
𝒞
tmp
)
z_{t+1:t+K}\leftarrow\textsc{FDM}(\mathcal{C}_{\text{tmp}})
⊳
\triangleright
Imagine visual outcome
18:
𝒞
tmp
←
𝒞
tmp
∪
{
z
t
+
1
:
t
+
K
}
\mathcal{C}_{\text{tmp}}\leftarrow\mathcal{C}_{\text{tmp}}\cup\{z_{t+1:t+K}\}
⊳
\triangleright
Update cache
19:
z
~
t
+
K
+
1
:
t
+
2
​
K
,
a
t
+
K
:
t
+
2
​
K
−
1
←
Predict
(
𝒞
tmp
)
\tilde{z}_{t+K+1:t+2K},a_{t+K:t+2K-1}\leftarrow\textsc{Predict}(\mathcal{C}_{\text{tmp}})
20:
t
←
t
+
K
t\leftarrow t+K
21:
end loop
