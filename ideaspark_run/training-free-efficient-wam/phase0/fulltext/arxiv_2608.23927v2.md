# GlanceWAM: Sparse Test-Time Imagination for World-Action Models

paper_id: arxiv:2608.23927v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

Video generative models capture rich physical priors over object dynamics, contact physics, and 3D scene evolution
(
NVIDIA, 2025a
;
Wan Team, 2025
;
SkyReels Team, 2025
)
, offering a promising foundation for autonomous robots to anticipate the consequences of their actions. In robot manipulation, world-action models (WAMs) leverage these predictive backbones through two distinct pathways:
(i)
representation shaping
, where future-prediction losses enrich shared visual features, and
(ii)
visual foresight
, where the model explicitly synthesizes future visual states to guide downstream policy execution
(
Kim et al., 2026
)
. However, existing WAMs couple future prediction and action decoding synchronously at the control rate (Figure
1
a)
(
Kim et al., 2026
;
Li et al., 2026c
)
. This tight coupling incurs two fundamental costs: prohibitive inference latency and myopic foresight. Over the short duration of an action chunk the scene barely changes, so predicting only that far ahead reduces world modeling to near-trivial reconstruction of the current observation.
Figure 1:
Synchronous imagination versus sparse lookahead foresight.
(a)
Existing WAMs either skip test-time foresight or imagine video synchronously with every action chunk, paying
K
K
heavy video denoising steps per chunk.
(b)
GlanceWAM glances ahead asynchronously to a single latent lookahead frame
H
f
≈
3
H_{f}\approx 3
s away and reuses it across
≈
4
{\approx}4
action chunks, each decoded in
48
48
ms.
(c)
Success rate on RoboCasa kitchen (24 tasks, demos only).
To bypass this latency bottleneck, recent approaches advocate abandoning test-time visual imagination entirely, relegating video modeling purely to offline pretraining or feature regularization
(
Yuan et al., 2026b
;
Cai et al., 2026
)
. While this strategy eliminates generative overhead during control, it discards explicit visual foresight—depriving downstream policies of the future visual destinations and spatial guidance needed for long-horizon manipulation. Conversely, architectures that retain visual foresight continue to couple dense future synthesis directly to high-frequency action chunks
(
Kim et al., 2026
;
Li et al., 2026c
)
, remaining bounded by prohibitive diffusion latencies and near-static prediction horizons. Current world-action models are thus caught in a
speed–success dilemma
: either synthesize video synchronously at the control rate and forfeit real-time reactivity, or discard test-time imagination and forfeit the performance benefits of visual foresight.
In this paper, we show that this dilemma is not an intrinsic property of visual foresight, but an artifact of coupling imagination synchronously to high-frequency control chunks. Downstream policies do not require dense, frame-by-frame future video of the immediate milliseconds; they primarily need a distant spatial destination (“where to go”). We introduce GlanceWAM, a framework that decouples visual imagination from real-time control on a single shared video DiT backbone. GlanceWAM makes visual foresight fast through asynchronous execution off the critical path: the model glances ahead on a slow clock to generate a single lookahead frame seconds into the future (
H
f
≈
3
H_{f}\approx 3
s) in the background, completely isolating generative sampling latency from the high-frequency control loop. Crucially, the entire imagination-and-control loop operates purely in latent space without decoding to raw pixels, allowing the action head to decode short action chunks at the control rate (
48
48
ms) in a single forward pass. This architecture is enabled by two key mechanisms:
(i)
a non-interfering attention mask (prefix-LM) that isolates video representations by preventing lookahead tokens from contaminating observation encodings, and
(ii)
staleness-robust horizon training that supervises the policy across varying time offsets (
u
∼
𝒰
(
0
,
H
f
]
u\sim\mathcal{U}(0,H_{f}]
) to tolerate lookahead aging.
Trained purely on demonstrations, GlanceWAM establishes a new state-of-the-art among world-action models across both manipulation success rate and inference speed, resolving the speed–success dilemma in practice. On the 24-task RoboCasa kitchen benchmark
(
Nasiriany et al., 2024
)
, GlanceWAM achieves
72.2
%
72.2\%
success, outperforming both synchronous Cosmos Policy (
67.1
%
67.1\%
) and imagination-free co-training (
64.4
%
64.4\%
), while reaching
99.0
%
99.0\%
on LIBERO
(
Liu et al., 2023
)
. Concurrently, its action decoding executes in
48
48
ms per chunk on a single NVIDIA A100 GPU (
24
×
24\times
lower control-path latency than synchronous WAMs), operating comfortably within real-time control budgets. In single-arm and bimanual real-robot manipulation, GlanceWAM achieves higher average success than
π
0.5
\pi_{0.5}
(
Physical Intelligence, 2025
)
without any robot-data pretraining (§
4.3
). Systematic diagnostics further confirm that lookahead conditioning is causally load-bearing and remains robust under asynchronous execution delays, demonstrating that world-action models do not need to choose between speed and imagination—sparse, asynchronous visual foresight delivers both.

## Method

In this section, we present GlanceWAM, a world-action model that decouples test-time visual foresight from high-frequency action execution on one shared video DiT backbone, read by a lightweight flow-matching action head
(
Lipman et al., 2023
;
NVIDIA, 2025b
)
.
We formalize the dual-timescale problem (§
3.1
), then detail the architecture (§
3.2
), staleness-robust co-training (§
3.3
), and asynchronous latent-space inference (§
3.4
).
Figure 2:
GlanceWAM architecture.
(a)
In co-training, the video DiT predicts the future target
𝐱
t
+
H
f
\mathbf{x}_{t+H_{f}}
, while the action head learns inverse dynamics toward a lookahead frame
𝐱
t
+
u
\mathbf{x}_{t+u}
at a random offset
u
∼
𝒰
(
0
,
H
f
]
u\sim\mathcal{U}(0,H_{f}]
, conditioned on
Δ
=
u
\Delta=u
, so it sees every lookahead age it will meet at deployment. Two separate VAE passes and the prefix-LM mask
𝐌
\mathbf{M}
keep the lookahead out of video prediction.
(b)
At inference, the lookahead latent
𝐳
^
la
\hat{\mathbf{z}}_{\text{la}}
is generated asynchronously once per
H
f
H_{f}
and never decoded; each
0.8
0.8
s action chunk reads the held
𝐳
^
la
\hat{\mathbf{z}}_{\text{la}}
at the decaying offset
Δ
\Delta
in
48
48
ms, within the offset range seen in training.
3.1
Problem Formulation and Dual-Timescale Setup
Setting and video diffusion foundation.
We consider language-conditioned visuomotor manipulation from demonstration trajectories.
At each decision step
t
t
, the policy receives observation history
𝐨
≤
t
\mathbf{o}_{\leq t}
and instruction
𝐜
\mathbf{c}
, predicting an action chunk
𝐚
t
=
[
a
t
,
…
,
a
t
+
H
a
−
1
]
∈
ℝ
H
a
×
D
a
\mathbf{a}_{t}=[a_{t},\dots,a_{t+H_{a}-1}]\in\mathbb{R}^{H_{a}\times D_{a}}
spanning control horizon
H
a
H_{a}
(
16
16
steps at
20
​
Hz
=
0.8
​
s
20\,\text{Hz}=0.8\,\text{s}
)
(
Zhao et al., 2023
;
Chi et al., 2023
;
Kim et al., 2026
)
.
Our framework builds upon a latent video diffusion transformer (SkyReels-V2-DF, 1.3B)
(
SkyReels Team, 2025
)
that compresses video frames into latent representations
𝐳
i
\mathbf{z}_{i}
via a frozen causal video VAE encoder
ℰ
\mathcal{E}
(
Wan Team, 2025
)
(reproducibility details in Appendix
E
).
We write
𝐨
\mathbf{o}
for observed frames, available at deployment, and
𝐱
\mathbf{x}
for future frames of the demonstration video, available only in training; both are encoded by
ℰ
\mathcal{E}
into latents
𝐳
\mathbf{z}
.
Under diffusion forcing
(
Chen et al., 2024
)
, each latent frame carries an independent noise level
τ
i
∈
[
0
,
1
]
\tau_{i}\in[0,1]
: clean observation frames have
τ
i
=
0
\tau_{i}=0
, while generative targets draw
τ
i
\tau_{i}
from a shifted logit-normal
(
Esser et al., 2024
)
.
Dual-timescale formulation.
A visual forward world model anticipates future scene evolution
𝐱
t
+
H
f
\mathbf{x}_{t+H_{f}}
over a foresight horizon
H
f
H_{f}
conditioned on context.
Existing world-action models couple foresight synchronously to the control rate (
H
f
≡
H
a
H_{f}\equiv H_{a}
)
(
Kim et al., 2026
;
Li et al., 2026c
)
, incurring heavy sampling delays and myopic foresight (§
1
).
To resolve this, GlanceWAM decouples the foresight horizon (
H
f
≈
3.0
H_{f}\approx 3.0
s on a slow background clock) from the control rate (
H
a
=
0.8
H_{a}=0.8
s on a fast latent clock).
This decoupling introduces a central challenge: a lookahead latent generated once per
H
f
H_{f}
is held and reused across
≈
H
f
/
H
a
{\approx}H_{f}/H_{a}
consecutive action chunks.
Consequently, the policy must act against visual foresight whose temporal offset
Δ
\Delta
decays from
H
f
H_{f}
toward
0
0
between refreshes — a staleness that the training interface must anticipate.
3.2
Unified Latent World-Action Architecture
Three-role training window.
Training a dual-timescale world-action model requires supervising two concurrent capabilities: generating visual foresight and conditioning actions on that foresight.
To supervise both in a single forward pass, each training sequence provides three distinct visual inputs (Figure
2
a):
(i)
Observation history
𝐨
≤
t
\mathbf{o}_{\leq t}
(
τ
=
0
\tau=0
) provides clean visual context.
(ii)
A noised
future target
𝐱
t
+
H
f
\mathbf{x}_{t+H_{f}}
(
τ
∈
(
0
,
1
)
\tau\in(0,1)
) at the full foresight horizon
H
f
H_{f}
supervises the video backbone’s generative forward dynamics.
(iii)
A clean
lookahead condition
𝐱
t
+
u
\mathbf{x}_{t+u}
(
τ
=
0
\tau=0
) at a randomized intermediate offset
u
∼
𝒰
(
0
,
H
f
]
u\sim\mathcal{U}(0,H_{f}]
provides teacher-forced visual guidance for action execution.
Separating the future target from the lookahead condition is essential: while the world model must learn to predict long-horizon transitions at
H
f
H_{f}
, the policy at deployment executes against a held lookahead whose remaining offset
Δ
\Delta
decays over time. Sampling
u
∈
(
0
,
H
f
]
u\in(0,H_{f}]
exposes the policy to this varying offset during training.
The three roles occupy dedicated 3D rotary position embedding (RoPE) temporal slots (
0
0
,
1
1
, and
2
2
).
Two-pass causal visual encoding.
Encoding these three frames into latent tokens requires preventing temporal information leakage during compression.
Because standard 3D causal video VAEs
(
Wan Team, 2025
;
SkyReels Team, 2025
)
aggregate features temporally across frames, encoding all three frames in a single pass would allow future information from
𝐱
t
+
H
f
\mathbf{x}_{t+H_{f}}
to contaminate the lookahead latent
𝐳
t
+
u
\mathbf{z}_{t+u}
.
To guarantee strict causal isolation, we encode the inputs in two independent passes of
ℰ
\mathcal{E}
:
Pass 1
(generative stream) computes
ℰ
⁡
(
[
𝐨
≤
t
,
𝐱
t
+
H
f
]
)
=
[
𝐳
≤
t
,
𝐳
t
+
H
f
]
\mathcal{E}([\mathbf{o}_{\leq t},\mathbf{x}_{t+H_{f}}])=[\mathbf{z}_{\leq t},\mathbf{z}_{t+H_{f}}]
to provide standard video co-training supervision;
Pass 2
(policy stream) computes
ℰ
⁡
(
[
𝐨
≤
t
,
𝐱
t
+
u
]
)
=
[
𝐳
≤
t
,
𝐳
t
+
u
]
\mathcal{E}([\mathbf{o}_{\leq t},\mathbf{x}_{t+u}])=[\mathbf{z}_{\leq t},\mathbf{z}_{t+u}]
, ensuring the lookahead latent depends solely on past context and its own frame.
Figure 3:
Non-interfering attention mask.
3-class prefix-LM mask
𝐌
\mathbf{M}
.
Non-interfering 3-class attention mask.
A second leakage path arises inside the transformer: under full self-attention, video generation queries could attend directly to the clean lookahead frame, turning future prediction into a trivial copying shortcut.
To eliminate representation contamination, we design a structured 3-class prefix-LM block mask
𝐌
∈
{
0
,
1
}
S
×
S
\mathbf{M}\in\{0,1\}^{S\times S}
(Figure
3
) implemented via FlexAttention
(
He et al., 2024
)
:
(i)
Observation queries attend exclusively to observations (
𝐌
⁡
(
𝐳
obs
,
𝐳
obs
)
=
1
\mathbf{M}(\mathbf{z}_{\text{obs}},\mathbf{z}_{\text{obs}})=1
).
(ii)
Future prediction queries attend to observations and future targets (
𝐌
⁡
(
𝐳
fut
,
{
𝐳
obs
,
𝐳
fut
}
)
=
1
\mathbf{M}(\mathbf{z}_{\text{fut}},\{\mathbf{z}_{\text{obs}},\mathbf{z}_{\text{fut}}\})=1
), but are strictly blocked from lookahead tokens (
𝐌
⁡
(
𝐳
fut
,
𝐳
la
)
=
0
\mathbf{M}(\mathbf{z}_{\text{fut}},\mathbf{z}_{\text{la}})=0
).
(iii)
Lookahead queries attend to observations and themselves (
𝐌
⁡
(
𝐳
la
,
{
𝐳
obs
,
𝐳
la
}
)
=
1
\mathbf{M}(\mathbf{z}_{\text{la}},\{\mathbf{z}_{\text{obs}},\mathbf{z}_{\text{la}}\})=1
).
Because no non-lookahead tokens attend to the lookahead frame (
𝐌
⁡
(
⋅
,
𝐳
la
)
=
0
\mathbf{M}(\cdot,\mathbf{z}_{\text{la}})=0
), the backbone representations for observations and future targets remain mathematically identical to standard video co-training, ensuring all policy gains stem strictly from the lookahead conditioning channel.
3.3
Staleness-Robust Co-Training
Joint flow-matching objective.
We train the shared video backbone
θ
dit
\theta_{\text{dit}}
and the action head
ϕ
act
\phi_{\text{act}}
end-to-end via joint conditional flow matching
(
Lipman et al., 2023
)
.
The video objective supervises forward dynamics velocity prediction
𝐯
θ
\mathbf{v}_{\theta}
on the noised future target
𝐳
t
+
H
f
(
τ
)
\mathbf{z}_{t+H_{f}}^{(\tau)}
:
ℒ
video
(
θ
dit
)
=
𝔼
τ
,
ϵ
,
𝐳
‖
𝐯
θ
(
𝐳
t
+
H
f
(
τ
)
,
τ
∣
𝐳
≤
t
,
𝐜
)
−
(
ϵ
−
𝐳
t
+
H
f
)
‖
2
.
\mathcal{L}_{\text{video}}(\theta_{\text{dit}})=\mathbb{E}_{\tau,\bm{\epsilon},\mathbf{z}}\left\|\mathbf{v}_{\theta}\left(\mathbf{z}_{t+H_{f}}^{(\tau)},\tau\mid\mathbf{z}_{\leq t},\mathbf{c}\right)-(\bm{\epsilon}-\mathbf{z}_{t+H_{f}})\right\|^{2}.
(1)
Simultaneously, the action head learns an inverse dynamics model by flow matching on the continuous action chunk
𝐚
t
∈
ℝ
H
a
×
D
a
\mathbf{a}_{t}\in\mathbb{R}^{H_{a}\times D_{a}}
, predicting its velocity
𝐮
ϕ
\mathbf{u}_{\phi}
conditioned on the DiT backbone’s multi-layer visual representations
𝐡
\mathbf{h}
, task instruction
𝐜
\mathbf{c}
, and lookahead offset
Δ
\Delta
:
ℒ
action
(
θ
dit
,
ϕ
act
)
=
𝔼
σ
,
ϵ
a
,
𝐚
‖
𝐮
ϕ
(
𝐚
t
(
σ
)
,
σ
∣
[
𝐡
obs
;
𝐡
la
]
,
𝐜
,
Δ
)
−
(
ϵ
a
−
𝐚
t
)
‖
2
,
\mathcal{L}_{\text{action}}(\theta_{\text{dit}},\phi_{\text{act}})=\mathbb{E}_{\sigma,\bm{\epsilon}_{a},\mathbf{a}}\left\|\mathbf{u}_{\phi}\left(\mathbf{a}_{t}^{(\sigma)},\sigma\mid[\mathbf{h}_{\text{obs}};\mathbf{h}_{\text{la}}],\mathbf{c},\Delta\right)-(\bm{\epsilon}_{a}-\mathbf{a}_{t})\right\|^{2},
(2)
where
σ
\sigma
is the Beta-distributed action flow timestep
(
Black et al., 2024a
)
, and the overall loss is
ℒ
=
ℒ
video
+
ℒ
action
\mathcal{L}=\mathcal{L}_{\text{video}}+\mathcal{L}_{\text{action}}
.
Staleness-robust horizon randomization.
During asynchronous deployment, a lookahead frame generated at time
t
0
t_{0}
is held across multiple control cycles, meaning subsequent action chunks at
t
0
+
k
⋅
H
a
t_{0}+k\cdot H_{a}
execute with an aging visual guide whose remaining offset decays toward zero.
To make the policy inherently robust to this staleness without frequent re-generation, we pair the randomized offset sampling
u
∼
𝒰
(
0
,
H
f
]
u\sim\mathcal{U}(0,H_{f}]
with explicit temporal conditioning:
the action head receives the exact offset
Δ
=
u
\Delta=u
via a sinusoidal time embedding
(
Vaswani et al., 2017
)
.
Exposing the policy to all intermediate offsets during training teaches it to seamlessly follow visual foresight regardless of where the current execution step falls within the refresh cycle.
To retain robust control when foresight is absent or degraded, we apply lookahead token dropout with probability
p
=
0.1
p=0.1
.
Multi-layer visual extraction.
Rather than extracting features solely from the final DiT block, the action head cross-attends to concatenated representations
[
𝐡
obs
;
𝐡
la
]
[\mathbf{h}_{\text{obs}};\mathbf{h}_{\text{la}}]
pooled across four uniformly spaced transformer layers
{
5
,
12
,
19
,
26
}
\{5,12,19,26\}
(Figure
2
a).
This multi-layer conditioning combines low-level spatial details from shallow layers with high-level semantic destinations from deep layers, providing a
+
0.7
%
+0.7\%
improvement on RoboCasa kitchen (§
4.4
).
3.4
Asynchronous Latent-Space Inference
Figure 4:
Imagined lookaheads.
Observation
𝐨
t
\mathbf{o}_{t}
, the GlanceWAM lookahead
𝐳
^
la
\hat{\mathbf{z}}_{\text{la}}
(decoded for display only), and the real frame
𝐱
t
+
3
​
s
\mathbf{x}_{t+3\,\text{s}}
, in RoboCasa kitchen (top) and real IMETA Y1 rollouts (bottom). The lookahead gets the target layout right while coarsening texture. Wrist view inset; more in Figure
10
.
Pure latent-space control path.
At test time, the model conditions directly on its own generated visual foresight (Figure
2
b).
Once per horizon
H
f
H_{f}
(
∼
3.0
{\sim}3.0
s), the video DiT runs an ODE flow sampler for
1
1
–
10
10
steps to generate the lookahead latent
𝐳
^
la
\hat{\mathbf{z}}_{\text{la}}
from current observation tokens
𝐳
≤
t
\mathbf{z}_{\leq t}
.
Crucially,
𝐳
^
la
\hat{\mathbf{z}}_{\text{la}}
is never decoded to raw RGB pixels (Figure
4
decodes some for display): it is retained entirely within the normalized latent space of the causal VAE, directly serving as the slot-
2
2
conditioning tokens for subsequent action forward passes.
Eliminating VAE decoding from the control loop removes substantial computational overhead and preserves fine-grained spatial representations.
Asynchronous amortization.
While the lookahead latent is held, the action head decodes subsequent
0.8
0.8
s action chunks in real time (
48
48
ms per chunk on a single NVIDIA A100 GPU).
Each chunk conditions on the decaying lookahead offset
Δ
=
H
f
−
(
t
mod
H
f
)
∈
(
0
,
H
f
]
\Delta=H_{f}-(t\bmod H_{f})\in(0,H_{f}]
, matching the training distribution of
u
u
(§
3.3
).
Because a single lookahead frame serves approximately
4
4
consecutive action chunks (
H
f
/
H
a
≈
4
H_{f}/H_{a}\approx 4
), video sampling overhead is amortized across control cycles (
∼
17
%
{\sim}17\%
at
10
10
steps, and
∼
2
%
{\sim}2\%
at the
1
1
-step regime validated in §
4.5
).
By pipelining lookahead generation on a background thread behind active action execution, the lookahead proposer leaves the critical control path entirely, enabling low-latency, closed-loop manipulation.
