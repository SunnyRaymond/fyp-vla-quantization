# Geometric Action Model for Robot Policy Learning

paper_id: arxiv:2606.17046v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

A long-standing goal in robotics is to build generalist manipulation policies that can follow natural-language instructions and manipulate arbitrary objects across diverse scenes
[
20
,
18
,
19
,
6
]
. To achieve this, a general manipulation model must not only recognize objects and parse instructions, but also reason about how the physical world will evolve under its own actions. This requires a unified understanding of language, visual appearance, scene geometry, robot state, and physical dynamics.
Recent progress has therefore increasingly relied on large-scale foundation models as pretrained substrates for robot policies. Vision-language-action models (VLAs) build on vision-language models whose representations are aligned with natural language, and learn to map visual and linguistic tokens to robot actions
[
55
,
20
,
18
,
6
,
16
,
30
]
. Video world-action models (WAMs) instead leverage pretrained video generation models, using their world prediction priors to jointly model future frames and actions
[
19
,
29
]
. While these approaches have shown impressive language-conditioned manipulation ability, they are fundamentally in 2D: 3D cues such as depth, scale, and occlusion are left implicit in monocular cues that the action decoder must disentangle on its own, leading to limited generalization across environment changes, especially in robot initial state and camera viewpoint
[
12
,
45
]
.
To overcome this limitation, recent work incorporates 3D geometric information into robot policies. One line learns policies directly on explicit 3D observations such as raw point clouds
[
51
,
14
]
, demonstrating the value of geometry for generalization but typically requiring task-specific encoders trained from scratch. With the emergence of Geometric Foundation Models (GFMs)
[
22
,
41
,
42
]
, some works transfer pretrained geometric priors into VLA policies, either distilling selected GFM features into the VLA backbone through representation alignment
[
49
,
21
,
37
]
or attaching a lightweight action head on top of a GFM’s final features
[
31
,
13
]
. These improve spatial awareness, but use the GFM only as a static feature extractor: its multi-layer geometric structure is never repurposed as the policy’s own temporal and action-generating substrate.
In this work, we propose
Geometric Action Model (GAM)
, which directly repurposes a GFM as a manipulation policy by using it as a shared medium for perception, future-state prediction, and action decoding. We show that by jointly predicting future action and geometry, geometric world dynamics can be inherently incorporated into robot policies.
Specifically, we split the pretrained GFM at an intermediate layer: the shallow layers serve as an observation encoder, while the remaining layers serve as a decoder block. Given the current visual observation, the observation encoder extracts spatially meaningful scene representations. To model how the world evolves over time, we insert a causal transformer at the intermediate layer that predicts future feature representations. This predictor is conditioned on task language, proprioception, and action history by introducing them as additional tokens at each timestep. The predicted future-state tokens are then processed by the remaining GFM decoder together with an action token, allowing the backbone to produce both future geometry and robot actions. An intuitive comparison between existing paradigms and our proposed framework is illustrated in Figure
2
.
Across diverse simulation
[
23
,
12
,
27
]
and real-world benchmarks, GAM matches or exceeds the success rate of current foundation-model-scale baselines such as VLAs and WAMs while using substantially fewer trainable parameters and substantially faster inference
(55
×
\times
faster)
, and shows improved generalization to unseen scenarios. GAM especially achieves outstanding performance in camera perturbation settings
(
↑
\uparrow
9.7%p)
, which requires geometric understanding priors.
Our contributions are as follows:
•
We introduce
Geometric Action Model (GAM)
, a manipulation policy that combines temporal world modeling, latent feature-space prediction, and a geometric foundation-model substrate in a single shared-backbone architecture.
•
We show that action and geometry can be predicted in a shared token space: a single autoregressive sequence and a single backbone forward pass produce both action tokens and future-scene tokens, decoded by a lightweight action regression head and a depth head.
•
We demonstrate that across diverse simulation and real-world manipulation benchmarks, GAM is simultaneously more accurate, more robust, faster, and lighter than current foundation-model-scale alternatives.

## Method

Figure 3:
Main architecture of GAM.
Problem Formulation.
We consider language-conditioned robot manipulation. At each timestep
t
t
, the robot receives a multi-view RGB observation
o
t
=
{
I
v
,
t
}
v
=
1
V
o_{t}=\{I_{v,t}\}_{v=1}^{V}
from
V
V
fixed cameras, a proprioceptive state
s
t
∈
ℝ
d
s
s_{t}\in\mathbb{R}^{d_{s}}
describing the robot’s joint configuration and end-effector pose, and a natural-language task instruction
ℓ
\ell
that is held constant throughout an episode. The policy
π
θ
\pi_{\theta}
must produce an action chunk
a
^
t
∈
ℝ
C
×
d
a
\hat{a}_{t}\in\mathbb{R}^{C\times d_{a}}
of length
C
C
, encoding the next
C
C
delta-pose or joint commands to be executed open-loop before the next observation is acquired. We learn a policy:
π
θ
:
(
{
o
t
−
H
+
1
,
…
,
o
t
}
,
{
s
t
−
H
+
1
,
…
,
s
t
}
,
{
a
t
−
H
,
…
,
a
t
−
1
}
,
ℓ
)
↦
a
^
t
\pi_{\theta}\colon\;\big(\{o_{t-H+1},\ldots,o_{t}\},\,\{s_{t-H+1},\ldots,s_{t}\},\,\{a_{t-H},\ldots,a_{t-1}\},\,\ell\big)\;\mapsto\;\hat{a}_{t}
(3)
from a dataset of
N
N
expert demonstrations
𝒟
=
{
(
τ
i
,
ℓ
i
)
}
i
=
1
N
\mathcal{D}=\{(\tau_{i},\ell_{i})\}_{i=1}^{N}
, where each trajectory
τ
i
=
(
o
t
,
s
t
,
a
t
)
t
=
1
T
i
\tau_{i}=(o_{t},s_{t},a_{t})_{t=1}^{T_{i}}
pairs a sequence of observations, states, and executed action chunks with a fixed instruction
ℓ
i
\ell_{i}
. The policy conditions on a context window of
H
H
recent timesteps.
Overview.
In the following sections, we explain how we transform a pretrained GFM into a
language-conditioned world-action model. Our key idea is to split the GFM into two parts and insert a causal temporal predictor between them. This design lets GAM formulate future prediction directly inside the GFM latent space, enabling all predictive computation to be performed in the GFM’s geometric representation space.
Concretely, our framework operates in three sequential stages inside the GFM. First, the
observation encoder
(§
4.1
) repurposes the shallow layers of the GFM to extract latent geometric features from multi-view observations. Next, the
causal future predictor
(§
4.2
) operates at the split layer, where it combines these geometric features with language, proprioception, and action history to
predict future latent tokens. Finally, during
feature propagation and decoding
(§
4.3
), the predicted future tokens are routed through the remaining deep GFM blocks to simultaneously decode future geometry and the final action chunk
a
^
t
\hat{a}_{t}
. Figure
3
(a) shows the overall architecture of our model.
4.1
Observation Encoder
We first reuse the shallow layers of the pretrained GFM as the observation encoder. Let
L
s
L_{s}
denote the split layer where the causal future predictor is inserted. The original GFM transformer stack is then decomposed into an encoder and a decoder:
E
≤
L
s
=
f
(
L
s
)
∘
⋯
∘
f
(
1
)
,
D
>
L
s
=
f
(
M
)
∘
⋯
∘
f
(
L
s
+
1
)
.
E_{\leq L_{s}}=f^{(L_{s})}\circ\cdots\circ f^{(1)},\qquad D_{>L_{s}}=f^{(M)}\circ\cdots\circ f^{(L_{s}+1)}.
(4)
Here, the choice of
L
s
L_{s}
is important because
L
s
L_{s}
must be deep enough to extract sufficiently rich visual features from the raw observations, yet shallower than the earliest layer used in the DPT head
L
s
<
m
1
L_{s}<m_{1}
, so that predicted future states can be decoded into future geometries by the DPT heads.
After defining this split layer
L
s
L_{s}
, for each timestep
t
′
t^{\prime}
in the context window, we tokenize the multi-view RGB observation
o
t
′
=
{
I
v
,
t
′
}
v
=
1
V
o_{t^{\prime}}=\{I_{v,t^{\prime}}\}_{v=1}^{V}
using the original GFM patch embedding. This produces the initial multi-view token sequence:
𝐙
t
′
(
0
)
=
[
𝐳
1
,
t
′
(
0
)
,
…
,
𝐳
V
,
t
′
(
0
)
]
∈
ℝ
V
⁡
(
1
+
P
)
×
d
,
\mathbf{Z}_{t^{\prime}}^{(0)}=\big[\mathbf{z}_{1,t^{\prime}}^{(0)},\ldots,\mathbf{z}_{V,t^{\prime}}^{(0)}\big]\in\mathbb{R}^{V(1+P)\times d},
(5)
where each view contributes one camera token and
P
P
patch tokens. The observation encoder maps these tokens to the split-layer representation
𝐙
t
′
(
L
s
)
\mathbf{Z}_{t^{\prime}}^{(L_{s})}
. By applying this encoding independently to each timestep in the context window, the output of this stage is a sequence of per-timestep geometric latent states
{
𝐙
t
−
H
+
1
(
L
s
)
,
…
,
𝐙
t
(
L
s
)
}
\{\mathbf{Z}_{t-H+1}^{(L_{s})},\ldots,\mathbf{Z}_{t}^{(L_{s})}\}
.
4.2
Causal Future Predictor
After the observation encoder, GAM performs temporal prediction directly at the split layer
L
s
L_{s}
, forecasting the next latent geometric state from current and past observations while conditioning on the task instruction, proprioception, and action history. To this end, we insert a causal future predictor
g
ϕ
g_{\phi}
between the shallow encoder
E
≤
L
s
E_{\leq L_{s}}
and the deep decoder
D
>
L
s
D_{>L_{s}}
.
For each timestep
t
′
t^{\prime}
in the context window, the encoder provides latent tokens
𝐙
t
′
(
L
s
)
\mathbf{Z}_{t^{\prime}}^{(L_{s})}
, and we embed the proprioceptive state
s
t
′
s_{t^{\prime}}
and previous action
a
t
′
−
1
a_{t^{\prime}-1}
as tokens:
𝐩
t
′
=
ψ
s
​
(
s
t
′
)
,
𝐪
t
′
=
ψ
a
​
(
a
t
′
−
1
)
,
\mathbf{p}_{t^{\prime}}=\psi_{s}(s_{t^{\prime}}),\qquad\mathbf{q}_{t^{\prime}}=\psi_{a}(a_{t^{\prime}-1}),
(6)
with
ψ
s
,
ψ
a
\psi_{s},\psi_{a}
lightweight projection layers, and the instruction
ℓ
\ell
into language tokens
𝐋
ℓ
\mathbf{L}_{\ell}
with a pretrained text encoder. We then form a per-timestep token block by concatenating the encoded GFM tokens with the proprioception and action-history tokens
𝐔
t
′
=
[
𝐩
t
′
;
𝐪
t
′
;
𝐙
t
′
(
L
s
)
]
\mathbf{U}_{t^{\prime}}=[\mathbf{p}_{t^{\prime}};\mathbf{q}_{t^{\prime}};\mathbf{Z}_{t^{\prime}}^{(L_{s})}]
. The full input to the causal future predictor is
𝐗
=
[
𝐋
ℓ
;
𝐔
t
′
−
H
+
1
;
…
;
𝐔
t
′
]
\mathbf{X}=[\mathbf{L}_{\ell};\mathbf{U}_{t^{\prime}-H+1};\ldots;\mathbf{U}_{t^{\prime}}]
.
The combined sequence
𝐗
\mathbf{X}
is then processed through block-causal self-attention
[
43
]
, ensuring the model incorporates past and present contexts without future leakage, as illustrated in Figure
3
(b). At the final layer of the predictor
g
ϕ
g_{\phi}
, we read off the predictions from their respective sequence slots. Specifically, the hidden states corresponding to the geometry slots forecast the latent geometric tokens of the future frame, denoted as
𝐙
~
t
′
+
1
(
L
s
)
\tilde{\mathbf{Z}}_{t^{\prime}+1}^{(L_{s})}
. Concurrently, the hidden state of the designated previous-action slot is projected to produce a predicted next action token
𝐚
~
t
′
∈
ℝ
d
\tilde{\mathbf{a}}_{t^{\prime}}\in\mathbb{R}^{d}
, in direct analogy to next-token prediction in a causal language model. By jointly forecasting action and geometric latents in this layer, we ensure that action tightly interacts with spatial representations.
This design of introducing a causal transformer predictor
g
ϕ
g_{\phi}
allows the pretrained GFM to acquire language-conditioned temporal world modeling with
minimal
architectural modification. Only the inserted
g
ϕ
g_{\phi}
needs to learn how to fuse language, proprioception, and action history with GFM latent features. The resulting predictions,
𝐙
~
t
′
+
1
(
L
s
)
\widetilde{\mathbf{Z}}_{t^{\prime}+1}^{(L_{s})}
and
𝐚
~
t
′
\tilde{\mathbf{a}}_{t^{\prime}}
, are then passed to the remaining GFM blocks for joint geometry and action decoding.
4.3
Feature Propagation and Action Decoding
Following the causal future predictor, the single action token
𝐚
~
t
′
\tilde{\mathbf{a}}_{t^{\prime}}
is replicated
V
V
times to form a set of per-view action tokens
{
𝐚
~
v
,
t
′
}
v
=
1
V
\{\tilde{\mathbf{a}}_{v,t^{\prime}}\}_{v=1}^{V}
, where
𝐚
~
v
,
t
′
=
𝐚
~
t
′
\tilde{\mathbf{a}}_{v,t^{\prime}}=\tilde{\mathbf{a}}_{t^{\prime}}
. Concatenated with the geometry tokens, they are fed through the remaining GFM blocks
D
>
L
s
D_{>L_{s}}
. We perform this
feature propagation
by appending each view’s corresponding action token
𝐚
~
v
,
t
′
\tilde{\mathbf{a}}_{v,t^{\prime}}
directly to its geometry token sequence for each timestep:
𝐙
~
t
′
+
1
(
M
)
=
(
f
(
M
)
∘
⋯
∘
f
(
L
s
+
1
)
)
(
[
[
𝐙
~
1
,
t
′
+
1
(
L
s
)
;
𝐚
~
1
,
t
′
]
,
…
,
[
𝐙
~
V
,
t
′
+
1
(
L
s
)
;
𝐚
~
V
,
t
′
]
]
)
.
\tilde{\mathbf{Z}}_{t^{\prime}+1}^{(M)}=\big(f^{(M)}\circ\cdots\circ f^{(L_{s}+1)}\big)\!\Big(\Big[\big[\tilde{\mathbf{Z}}_{1,t^{\prime}+1}^{(L_{s})};\,\tilde{\mathbf{a}}_{1,t^{\prime}}\big],\ldots,\big[\tilde{\mathbf{Z}}_{V,t^{\prime}+1}^{(L_{s})};\,\tilde{\mathbf{a}}_{V,t^{\prime}}\big]\Big]\Big).
(7)
To prevent future leakage, we extend the predictor’s causal mask strategy to the GFM’s remaining global attention layers (
f
global
(
m
)
f_{\text{global}}^{(m)}
).
Finally, the propagated features are decoded by two heads. The lightweight action head
h
act
h_{\text{act}}
aggregates action tokens over the context window to regress the executable action chunk
a
^
t
′
\hat{a}_{t^{\prime}}
, while the original GFM depth head
h
depth
h_{\text{depth}}
decodes geometry tokens into action-aligned future depth maps. The GFM’s deep blocks, originally pretrained to decode shallow features into 3D geometry, are thus repurposed here as the decoder of the world model’s predicted future.
4.4
Training and Inference
The policy is trained end-to-end by minimizing a multi-task objective over action execution, world modeling, and geometric decoding:
ℒ
total
=
λ
act
​
ℒ
act
+
λ
feat
​
ℒ
feat
+
λ
depth
​
ℒ
depth
,
\mathcal{L}_{\text{total}}=\lambda_{\text{act}}\mathcal{L}_{\text{act}}+\lambda_{\text{feat}}\mathcal{L}_{\text{feat}}+\lambda_{\text{depth}}\mathcal{L}_{\text{depth}},
(8)
where the
λ
\lambda
factors balance each term and
ℋ
=
{
t
−
H
+
1
,
…
,
t
}
\mathcal{H}=\{t-H+1,\ldots,t\}
is the context window. The
action
loss
ℒ
act
\mathcal{L}_{\text{act}}
is an
ℓ
1
\ell_{1}
regression between the decoded action chunk
a
^
t
′
\hat{a}_{t^{\prime}}
and the expert action
a
t
′
a_{t^{\prime}}
over all
t
′
∈
ℋ
t^{\prime}\in\mathcal{H}
. The
future-feature
loss
ℒ
feat
\mathcal{L}_{\text{feat}}
anchors the predictor
g
ϕ
g_{\phi}
to temporal geometric transitions by aligning predicted future tokens
𝐙
~
t
′
+
1
(
L
s
)
\tilde{\mathbf{Z}}_{t^{\prime}+1}^{(L_{s})}
with the actual next frame
𝐙
t
′
+
1
(
L
s
)
\mathbf{Z}_{t^{\prime}+1}^{(L_{s})}
extracted from frozen GFM:
ℒ
feat
=
∑
t
′
∈
ℋ
‖
𝐙
~
t
′
+
1
(
L
s
)
−
𝐙
t
′
+
1
(
L
s
)
‖
1
.
\mathcal{L}_{\text{feat}}=\sum_{t^{\prime}\in\mathcal{H}}\left\|\tilde{\mathbf{Z}}_{t^{\prime}+1}^{(L_{s})}-\mathbf{Z}_{t^{\prime}+1}^{(L_{s})}\right\|_{1}.
(9)
The
future-depth
loss
ℒ
depth
\mathcal{L}_{\text{depth}}
grounds the predicted future in valid 3D structure by supervising the decoded depth
D
~
t
′
+
1
=
h
depth
​
(
𝐙
~
t
′
+
1
(
m
∗
)
)
\tilde{D}_{t^{\prime}+1}=h_{\text{depth}}(\tilde{\mathbf{Z}}_{t^{\prime}+1}^{(m^{*})})
using depth head
h
depth
h_{\text{depth}}
against ground-truth future depth
D
t
′
+
1
D_{t^{\prime}+1}
, adopting the scale-invariant and gradient-matching penalties of the GFM
[
22
,
42
]
.
At inference, we maintain the historical context online with key-value caching, so each step processes only the new observation
o
t
o_{t}
and previous action
a
t
−
1
a_{t-1}
in a single feed-forward pass.
