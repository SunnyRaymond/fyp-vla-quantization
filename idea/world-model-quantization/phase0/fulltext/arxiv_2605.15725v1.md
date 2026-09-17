# DiLA: Disentangled Latent Action World Models

paper_id: arxiv:2605.15725v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

World models
(
Friston, 2010
;
Sutton, 1991
;
Ha and Schmidhuber, 2018
;
Hafner et al., 2020
)
have emerged as a cornerstone for autonomous agents
(
Reed et al., 2022
)
, enabling planning
(
Sobal et al., 2025
)
, simulation
(
He et al., 2025
)
, and policy learning
(
Hafner et al., 2020
;
Hansen et al., 2024
;
Hafner et al., 2024
)
in complex environments. By learning to predict future states, these models implicitly capture the underlying latent dynamics of the physical world
(
Garrido et al., 2025
;
Bar et al., 2024
;
Zhou et al., 2024
;
Assran et al., 2025
)
. However, traditional approaches rely heavily on action-labeled datasets, which are scarce and expensive to scale compared to the vast availability of unlabeled video data
(
Venkataramanan et al., 2023
)
. To bridge this gap, Latent Action Models (LAMs)
(
Bruce et al., 2024
;
Schmidt and Jiang, 2024
)
have been introduced to infer latent actions directly from unlabeled videos, serving as surrogates for explicit control signals
(
Ye et al., 2024
;
Chen et al., 2024b
;
Bu et al., 2025
)
. A LAM consists of two core components: an Inverse Dynamics Model (IDM), which extracts latent actions from consecutive frames, and a Forward Dynamics Model (FDM), which predicts future states by conditioning past observations on these inferred actions.
Figure 1
:
Co-evolving of latent actions and disentanglement.
To resolve the "LAM Trade-off",
DiLA
jointly learns abstract latent actions and content-structure disentanglement. By imposing a restricted predictive bottleneck, the latent action model drives the disentanglement of spatial structures from content semantics. Conversely, this disentanglement provides structural layout inputs that facilitate the learning of highly abstract latent actions.
Despite their promise, current LAMs face a fundamental dilemma, which we term the "
LAM Trade-off
"
(
Gao et al., 2025
;
Liu et al., 2025
)
: the tension between action abstraction and generation quality. To ensure actions are transferable and abstract, models typically impose strong predictive bottlenecks, such as Vector Quantization
(
Van Den Oord et al., 2017
;
Bruce et al., 2024
;
Schmidt and Jiang, 2024
;
Ye et al., 2024
)
or variational bottleneck
(
Kingma and Welling, 2013
;
Gao et al., 2025
)
. While these priors encourage abstraction, they often disrupt the intrinsic manifold of the latent action space, leading to over-simplification and degraded video generation fidelity
(
Garrido et al., 2026
)
. Conversely, relaxing these bottlenecks improves generation but yields entangled representations where latent actions capture irrelevant visual details rather than pure dynamics
(
Nikulin et al., 2025
)
.
Most existing approaches prioritize abstraction at the expense of generation quality
(
Schmidt and Jiang, 2024
;
Ye et al., 2024
;
Chen et al., 2024b
)
. They often adopt a disjoint, two-stage training paradigm, discarding the FDM in favor of a separate, pre-trained diffusion model for video generation
(
Bruce et al., 2024
;
Gao et al., 2025
)
. To learn more abstract latent actions, some recent works extract optical flow or depth maps as inputs or supervision targets
(
Kim et al., 2025
;
Nikulin et al., 2025
;
Bi et al., 2025
)
. Conceptually, these methods function as an explicit separation of motion-related spatial layouts from content details.
This observation motivates our investigation into two pivotal questions:
1.
Can disentanglement learning and latent action learning be co-optimized?
2.
How can we simultaneously achieve abstract latent actions and high-fidelity prediction?
In this paper, we argue that the key to resolving the "LAM Trade-off" lies in
disentanglement
. We propose
DiLA
, a novel
D
isentangled
L
atent
A
ction world model that learns to decouple video sequences into
structure
(dynamics-relevant spatial layout) and
content
(dynamics-irrelevant visual appearance and texture). The separation is functional: features needed to model latent dynamics under the prediction bottleneck are assigned to the structure pathway. As illustrated in Fig.
1
, instead of learning latent actions from entire visual features,
DiLA
forces the latent action to predict only the structural layout dynamics. This constraint facilitates the learning of content-invariant latent actions and further enforces disentanglement: to minimize structure prediction error, the model is incentivized to distill motion dynamics into the latent action while offloading static visual details to a separate content pathway. The structure pathway captures motion-related spatial information (e.g., positions and shapes), whereas the content pathway maintains visual details for high-fidelity generation. We show that while difficult individually, the co-evolution of disentanglement and latent action learning creates a synergistic loop that significantly enhances representation learning.
To validate the capabilities of
DiLA
, we conducted experiments on a large-scale suite of datasets covering human activity, robot manipulation, and outdoor navigation. Our benchmarks demonstrate superior performance against leading methods, particularly in maintaining high generation quality while transferring actions across different embodiments. We also show that
DiLA
learns a structured, interpretable latent action space that effectively supports downstream visual planning. Collectively, these results confirm that disentangling structure from content is key to achieving both abstract action learning and high-fidelity generation.
Our contributions are summarized as follows:
•
We present
DiLA
, the first disentangled latent action world model that reconciles the inherent trade-off between latent action abstraction and generation fidelity via content-structure disentanglement.
•
Co-evolution of latent action and disentanglement. The predictive bottleneck acts as a driving force to isolate spatial layouts from appearance. In turn, this separation facilitates the learning of abstract latent action.
•
We demonstrate
DiLA
’s capabilities through extensive experiments, covering cross-embodiment action transfer, visual planning, and the interpretable analysis of latent manifolds on out-of-distribution datasets.

## Method

This section details
DiLA
, a disentangled latent action world model that achieves disentanglement by processing video sequences through two specialized pathways. The first, a
structure pathway
, isolates motion-related spatial layouts to learn latent actions that are invariant to visual contents, enforced via a strict information bottleneck. The second, a
content pathway
, extracts and memorizes temporal-invariant visual features over time. Future embeddings are generated by a Fusion Decoder that recombines these structural and content representations. We employ a DINOv2 encoder
(
Oquab et al., 2024
)
for feature extraction and an RAE decoder
(
Zheng et al., 2025
)
for visualization. Specifically,
DiLA
formulates prediction entirely in the latent space (similar to JEPA
(
LeCun, 2022
;
Assran et al., 2025
)
), eliminating the need for pixel-level reconstruction during training. The model architecture is illustrated in Fig.
2
.
To be specific,
DiLA
initiates processing by extracting visual embeddings
𝒆
0
:
t
\boldsymbol{e}_{0:t}
from video clips
𝒐
0
:
t
\boldsymbol{o}_{0:t}
, which are then refined by a spatial-temporal Transformer
(
Ye et al., 2024
)
. Spatial attention layers model global spatial dependencies, while temporal attention layers utilize causal masking to restrict information flow to historical contexts. To further enforce temporal causality, we integrate rotary position embeddings
(
Su et al., 2024
)
into the temporal attention.
Figure 3
:
Action transfer.
(A)
Cross-embodiment and intra-domain transfer
. Left: Human-to-robot latent action transfer. Middle: Semantic transfer across diverse objects and viewpoints. Right: Intra-domain transfer (human-to-human and robot-to-robot). (B)
Navigation transfer
. Action transfer between virtual simulations and real-world navigation environments.
The structure pathway.
First, a structure encoder compresses tokens into structure embeddings
𝒔
0
:
t
\boldsymbol{s}_{0:t}
. The IDM then takes these embeddings to compute abstract latent actions
𝒛
0
:
t
−
1
\boldsymbol{z}_{0:t-1}
. In self-supervised settings, the temporal difference of structure embeddings (
Δ
𝒔
0
:
t
−
1
\Delta\boldsymbol{s}_{0:t-1}
) effectively represents dominant motion changes. Accordingly, the IDM processes these differences using 3D convolutional blocks: spatial kernels capture translation-invariant global features, while temporal kernels aggregate bidirectional context to form time-dependent latent actions (
d
z
=
256
d_{z}=256
). The FDM is designed as a lightweight spatial-temporal Transformer. It generates the next state by predicting displacement vectors based on the current state
𝒔
0
:
t
−
1
\boldsymbol{s}_{0:t-1}
, conditioned on the latent actions
𝒛
0
:
t
−
1
\boldsymbol{z}_{0:t-1}
using AdaLN-zero
(
Peebles and Xie, 2023
)
. The final prediction is formulated as a residual update:
𝒛
0
:
t
−
1
\displaystyle\boldsymbol{z}_{0:t-1}
=
IDM
(
Δ
𝒔
0
:
t
−
1
)
\displaystyle=\text{IDM}(\Delta\boldsymbol{s}_{0:t-1})
(1)
𝒔
^
1
:
t
\displaystyle\hat{\boldsymbol{s}}_{1:t}
=
𝒔
0
:
t
−
1
+
FDM
(
𝒔
0
:
t
−
1
,
𝒛
0
:
t
−
1
)
.
\displaystyle=\boldsymbol{s}_{0:t-1}+\text{FDM}(\boldsymbol{s}_{0:t-1},\boldsymbol{z}_{0:t-1}).
Crucially, the temporal difference
Δ
𝒔
0
:
t
−
1
\Delta\boldsymbol{s}_{0:t-1}
serves a dual purpose: it acts as an information bottleneck to enforce abstraction and implicitly functions as the driving force for disentanglement. Since this pathway is tasked with predicting the structure embedding
𝒔
\boldsymbol{s}
(rather than the full visual embedding
𝒆
\boldsymbol{e}
) using these abstract latent actions, a dual optimization pressure emerges. To minimize prediction error, the model is incentivized to: (1) distill only abstract dynamics into the latent action, and simultaneously (2) ensure the target
𝒔
\boldsymbol{s}
retains only dynamics-correlated spatial layouts. This mutual adaptation renders the prediction task tractable. Consequently, the model naturally purges content details from
𝒔
\boldsymbol{s}
, as such high-entropy information cannot be effectively compressed into the low-dimensional latent action, which would otherwise impede accurate prediction.
The content pathway.
A content encoder is used to compress tokens into embeddings
𝒄
0
:
t
\boldsymbol{c}_{0:t}
. To aggregate historical content information, we utilize the Mamba architecture
(
Gu and Dao, 2024
)
. Unlike traditional RNNs, Mamba offers superior training parallelization and stable memory retention over long sequences. Functionally, the Mamba module mimics the principle of Slow Feature Analysis
(
Wiskott and Sejnowski, 2002
)
: it aggregates static features as they are revealed, differing from the structure pathway’s focus on dynamics. In a POMDP, content is static in the world state but dynamic in the belief state due to partial observability. We offload these belief updates to the content pathway, thereby allowing the structure pathway to specialize strictly in modeling physical dynamics. Furthermore, this long-term memory allows the model to preserve information about temporarily occluded backgrounds, as well as to infer unobserved regions in new scenes.
The output of the memory module at time
t
t
is formulated as:
𝒄
t
mem
=
Mamba
​
(
𝒄
t
,
𝒉
t
−
1
)
.
\boldsymbol{c}_{t}^{\text{mem}}=\text{Mamba}(\boldsymbol{c}_{t},\boldsymbol{h}_{t-1}).
Fusing content and structure.
We employ a spatial-attention Transformer equipped with a dual cross-attention mechanism as the Fusion Decoder: with
𝒔
^
\hat{\boldsymbol{s}}
acting as the queries, the first cross-attention module utilizes the content memory
𝒄
mem
\boldsymbol{c^{\text{mem}}}
as keys and values, followed by a second that attends to the initial visual embedding
𝒆
0
\boldsymbol{e}_{0}
as keys and values. Conditioning on
𝒆
0
\boldsymbol{e}_{0}
is crucial, as it supplies high-frequency details lost during compression. The decoding process at time
t
t
is:
Dec
θ
​
(
𝒔
^
t
+
1
,
𝒆
0
,
𝒄
t
mem
)
=
𝒆
^
t
+
1
.
\text{Dec}_{\theta}(\hat{\boldsymbol{s}}_{t+1},\boldsymbol{e}_{0},\boldsymbol{c}_{t}^{\text{mem}})=\hat{\boldsymbol{e}}_{t+1}.
(2)
Latent rollouts.
Unlike prior approaches
(
Gao et al., 2025
;
Routray et al., 2025
)
that perform rollouts in high-dimensional observation space,
DiLA
generates rollouts directly within the latent structure space via autoregressive iteration. At time step
t
t
, the model predicts the subsequent structure state by applying the FDM to the previously predicted state
𝒔
^
t
\hat{\boldsymbol{s}}_{t}
and the current latent action
𝒛
t
\boldsymbol{z}_{t}
:
𝒔
^
t
+
1
=
𝒔
^
t
+
FDM
​
(
𝒔
^
t
,
𝒛
t
)
.
\hat{\boldsymbol{s}}_{t+1}=\hat{\boldsymbol{s}}_{t}+\text{FDM}(\hat{\boldsymbol{s}}_{t},\boldsymbol{z}_{t}).
(3)
Upon obtaining
𝒔
^
t
+
1
\hat{\boldsymbol{s}}_{t+1}
, we reconstruct the visual embedding
𝒆
^
t
+
1
\hat{\boldsymbol{e}}_{t+1}
via the Fusion Decoder (Eq.
2
). Subsequently, the content memory is updated to
𝒄
^
t
+
1
m
​
e
​
m
\hat{\boldsymbol{c}}_{t+1}^{mem}
with this newly generated
𝒆
^
t
+
1
\hat{\boldsymbol{e}}_{t+1}
to condition the next step of the rollout.
Figure 4
:
Content and structure disentanglement.
(A)
Rebinding
: Structure from a source sequence is fused with content from a reference sequence. The output retains the source’s spatial dynamics and the reference’s appearance. (B)
Motion Isolation
: Fixing the structure embedding
𝒔
\boldsymbol{s}
results in a static sequence, confirming that content memory
𝒄
mem
\boldsymbol{c}^{\text{mem}}
encodes no motion information.
Training objectives.
We train
DiLA
using a self-supervised
teacher-forcing
paradigm that relies solely on video sequences, thereby eliminating the need for ground-truth action labels. Throughout training, both the DINOv2 encoder and the RAE decoder remain frozen. The total training objective is a weighted combination of visual latent prediction, structure prediction, latent action consistency, and regularization losses:
ℒ
total
\displaystyle\mathcal{L}_{\text{total}}
=
λ
𝒆
​
ℒ
𝒆
+
λ
𝒔
​
ℒ
𝒔
+
λ
𝒛
​
ℒ
𝒛
+
λ
reg
​
ℒ
reg
\displaystyle=\lambda_{\boldsymbol{e}}\mathcal{L}_{\boldsymbol{e}}+\lambda_{\boldsymbol{s}}\mathcal{L}_{\boldsymbol{s}}+\lambda_{\boldsymbol{z}}\mathcal{L}_{\boldsymbol{z}}+\lambda_{\text{reg}}\mathcal{L}_{\text{reg}}
(4)
ℒ
𝒆
\displaystyle\mathcal{L}_{\boldsymbol{e}}
=
‖
𝒆
t
−
𝒆
^
t
‖
2
\displaystyle=\|\boldsymbol{e}_{t}-\hat{\boldsymbol{e}}_{t}\|_{2}
ℒ
𝒔
\displaystyle\mathcal{L}_{\boldsymbol{s}}
=
‖
Δ
​
𝒔
t
−
FDM
​
(
𝒔
t
,
𝒛
t
)
‖
2
\displaystyle=\|\Delta\boldsymbol{s}_{t}-\text{FDM}(\boldsymbol{s}_{t},\boldsymbol{z}_{t})\|_{2}
ℒ
𝒛
\displaystyle\mathcal{L}_{\boldsymbol{z}}
=
‖
IDM
​
(
𝒔
t
+
1
−
𝒔
t
)
−
IDM
​
(
𝒔
^
t
+
1
−
𝒔
t
)
‖
2
\displaystyle=\|\text{IDM}(\boldsymbol{s}_{t+1}-\boldsymbol{s}_{t})-\text{IDM}(\hat{\boldsymbol{s}}_{t+1}-\boldsymbol{s}_{t})\|_{2}
ℒ
reg
\displaystyle\mathcal{L}_{\text{reg}}
=
‖
𝒛
t
‖
2
+
∑
t
m
t
⋅
(
cos
⁡
(
𝒛
t
fwd
,
𝒛
t
bwd
)
+
1
)
2
∑
t
m
t
+
ϵ
\displaystyle=\|\boldsymbol{z}_{t}\|_{2}+\frac{\sum_{t}m_{t}\cdot(\cos(\boldsymbol{z}_{t}^{\text{fwd}},\boldsymbol{z}_{t}^{\text{bwd}})+1)^{2}}{\sum_{t}m_{t}+\epsilon}
+
0.5
⋅
exp
(
−
10
⋅
σ
𝒛
)
,
\displaystyle+0.5\cdot\exp(-10\cdot\sigma_{\boldsymbol{z}}),
where
𝒛
t
fwd
=
IDM
​
(
𝒔
t
+
1
−
𝒔
t
)
\boldsymbol{z}_{t}^{\text{fwd}}=\text{IDM}(\boldsymbol{s}_{t+1}-\boldsymbol{s}_{t})
and
𝒛
t
bwd
=
IDM
​
(
𝒔
t
−
𝒔
t
+
1
)
\boldsymbol{z}_{t}^{\text{bwd}}=\text{IDM}(\boldsymbol{s}_{t}-\boldsymbol{s}_{t+1})
represent the forward and backward latent actions, respectively. The mask
m
t
=
𝕀
[
∥
𝒔
t
+
1
−
𝒔
t
∥
>
τ
]
m_{t}=\mathbb{I}[\|\boldsymbol{s}_{t+1}-\boldsymbol{s}_{t}\|>\tau]
filters out static frames. Adopting the principle of group symmetry
(
Koyama et al., 2023
;
Hayashi et al., 2025
)
, we employ a cosine similarity objective in
ℒ
reg
\mathcal{L}_{\text{reg}}
to enforce that inverse temporal transitions yield opposite latent action vectors. This geometric constraint compels the latent space to align with meaningful motion dynamics while suppressing stochastic, irrelevant distractors. To further constrain this manifold, we introduce additional regularization terms targeting the norm and variance of the latent actions. We constrain the vector norm to maintain a compact manifold and prevent topological distortion, while regularizing the variance to maximize information entropy.
