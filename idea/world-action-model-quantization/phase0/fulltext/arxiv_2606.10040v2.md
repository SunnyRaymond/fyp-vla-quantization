# Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination

paper_id: arxiv:2606.10040v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

Robot control requires understanding how the physical scene will evolve during interaction. World-Action Models (WAMs)
[
27
,
11
]
address this by coupling future video prediction
[
7
,
12
,
29
]
with action generation
[
5
,
17
,
14
,
35
]
. By predicting how observations change over time, WAMs embed rich physical dynamics and world priors into the control policy, making them a promising robot learning paradigm. Yet the strongest systems still rely on very large video generators
[
29
,
14
]
, based on the belief that sharper, more photorealistic futures will yield better actions. That belief comes with a cost: heavy compute, high latency, and steep hardware demands that block real-time deployment.
A different picture is taking shape. High-quality control does not require photorealistic video. What the policy truly needs is a future representation that preserves task-relevant geometry, motion tendencies, and contact cues. For example, VPP
[
12
]
shows that action generation remains effective even when the denoising process is reduced to a single step, while Fast-WAM
[
37
]
demonstrates that WAMs can remain competitive even when explicit future generation is skipped during inference. Building on this insight, we aim not for perfect images but for action-centric futures, and we reframe efficiency as a modeling problem by proposing
Efficient-WAM
.
Our core idea is to make the video branch smaller and smarter within a Mixture-of-Transformers (MoT) framework
[
1
,
16
]
through structured pruning guided by world-knowledge transfer from the foundation model WAN-2.2-5B
[
26
]
. This distillation step defines what the model must keep to remain action-faithful: channels and pathways that encode geometry, dynamics, and contact. Once the backbone has been carved down around these essentials, two complementary accelerations follow naturally. First, token density can fall without harming control. Because pruning concentrates capacity on task-relevant structure, the model can predict lower-resolution future latents that still carry the cues needed by the action expert. Computation and memory scale down with token count, while the distilled priors preserve the information that matters. Second, denoising can be asymmetric. The pruned video branch no longer needs a long sampling schedule to hallucinate photorealistic detail, whereas the action branch benefits from a richer trajectory refinement. Allocating fewer steps to video and more to action reduces latency where it counts while preserving decision quality.
Unlike early-exit or dynamic layer-skipping methods
[
38
,
32
]
or single-step denoising and diffusion-policy distillation methods
[
24
,
23
,
28
]
that trade stability for short-term gains, our design integrates model size, token budget, and sampling depth into a coherent, action-centric system that achieves massive efficiency gains with minimal compromise to control performance. Pruning focuses representation on control-critical content. Lower token density then becomes a safe consequence rather than a risky shortcut. Asymmetric denoising exploits the increased reliability of the pruned video predictor to further reduce sampling. The three levers reinforce one another, producing a compact future-imagination module that remains aligned with the controller’s needs.
We evaluate Efficient-WAM in simulation and real-world manipulation. Despite intentionally coarse future predictions, it achieves 86.7% average success in simulation and 66.25% in real-world tasks, comparable to or better than heavyweight WAM baselines. By jointly optimizing model size, token budget, and denoising, Efficient-WAM reduces per-chunk latency to 98 ms on a local consumer GPU. Our core contributions are:
•
We identify the critical deployment bottleneck of WAMs and introduce an ”action-centric future imagination” design principle, demonstrating that WAMs can be effectively decoupled from the pursuit of photorealistic video generation.
•
We propose Efficient-WAM, a novel architecture that holistically reduces inference costs by optimizing model size, token count, and denoising steps. This unified approach enables low-latency, real-world deployment while preserving strong world priors and control performance.
•
We decompose WAM inference cost into model size, visual tokens, and denoising steps, showing how pruning enables lower-resolution future latents and shorter video-side sampling.

## Method

3.1
Design Formulation
A World-Action Model (WAM) explicitly models the joint distribution of future scene evolution and control actions. Given a current observation
o
o
, a language instruction
l
l
, and a robot state
s
s
, the joint prediction objective is formulated as
p
(
𝐳
v
,
a
1
:
H
∣
o
,
l
,
s
)
p(\mathbf{z}^{v},a_{1:H}\mid o,l,s)
, where
𝐳
v
\mathbf{z}^{v}
represents the explicit future visual latents and
a
1
:
H
a_{1:H}
is the action chunk.
To make this joint prediction tractable, our architecture factorizes the distribution into a future-imagination process and a future-conditioned action generation process:
p
(
𝐳
v
,
a
1
:
H
∣
o
,
l
,
s
)
=
p
ϕ
​
(
𝐳
v
∣
o
,
l
)
⏟
video branch
⋅
p
θ
(
a
1
:
H
∣
o
,
l
,
s
,
𝐳
v
)
⏟
action branch
.
p(\mathbf{z}^{v},a_{1:H}\mid o,l,s)=\underbrace{p_{\phi}(\mathbf{z}^{v}\mid o,l)}_{\text{video branch}}\cdot\underbrace{p_{\theta}(a_{1:H}\mid o,l,s,\mathbf{z}^{v})}_{\text{action branch}}.
(1)
Here,
p
ϕ
p_{\phi}
predicts the future dynamic context, while
p
θ
p_{\theta}
extracts executable control signals from this imagination. While Efficient-WAM preserves this joint formulation, it fundamentally questions whether
𝐳
v
\mathbf{z}^{v}
must be photorealistic.
To connect this formulation to efficiency, we focus on the computation required to produce the future representation
𝐳
v
\mathbf{z}^{v}
. Denote the active video model size by
ℳ
v
\mathcal{M}_{v}
, the future prediction resolution by
r
v
r_{v}
, the resulting number of future video tokens by
N
tok
v
​
(
r
v
)
N_{\text{tok}}^{v}(r_{v})
, and the video denoising budget by
K
v
K_{v}
. For a fixed implementation family, the video-side cost can be described as a function:
𝒞
video
=
ℱ
video
​
(
ℳ
v
,
N
tok
v
​
(
r
v
)
,
K
v
)
,
\mathcal{C}_{\text{video}}=\mathcal{F}_{\text{video}}\!\left(\mathcal{M}_{v},\,N_{\text{tok}}^{v}(r_{v}),\,K_{v}\right),
(2)
This abstraction highlights the controllable factors of video-side computation. Efficient-WAM follows an
action-centric design principle
: policies need structural and dynamic cues, not photorealistic details. We therefore systematically compress the three factors in Eq. (
2
) via a compact video expert distilled from WAN-2.2-5B, low-resolution future latents, and asymmetric video-action denoising. The following sections detail these components.
Figure 2:
Efficient-WAM architecture.
The model utilizes a multiscale video-latent layout where high-resolution current observations and low-resolution future latents are concatenated. A compact video expert and an action expert interact via layer-wise MoT to predict optimal action chunks.
3.2
Compact Architecture with World-Knowledge Transfer
Large-scale video generation backbones encode rich world priors, yet their massive parameter counts make them ill-suited for the stringent latency requirements of real-time robotic control. To address this challenge, we adopt a compact Mixture-of-Transformers (MoT) architecture. By using a lightweight video expert and a dedicated action expert, our design retains necessary world priors while significantly reducing the computational cost.
To build the video expert, we prune WAN-2.2-5B by reducing transformer depth and layer width. Instead of random initialization, we copy weights from selected teacher layers via layer slicing. This structured transfer is highly intentional. It ensures the student inherits fundamental physical priors, such as task-relevant geometry, motion tendencies, and contact cues. At the same time, it sheds the parameter capacity dedicated to high-fidelity pixel rendering. To stabilize this knowledge transfer, we supplement the standard video flow-matching objective with a teacher-guided distillation loss that aligns intermediate hidden states and temporal changes. This effectively distills the teacher’s physical world understanding into an action-centric backbone.
As illustrated in Figure
2
, this compact video expert is coupled layer-wise with the action expert. The task instruction is injected via cross-attention, while robot states and noisy actions are embedded as action tokens. At each MoT layer, action tokens attend to the video tokens to extract future context before being mapped back to the action stream. During the main action training stage, the compact video expert is frozen. This preserves the stable world priors while optimizing the lightweight action expert for precise control.
3.3
Coarse Future Prediction with Multiscale Video-Latent Layout
Standard WAMs typically predict future videos at the uniform resolution of the input observation, wasting computational capacity on action-irrelevant visual details. Efficient-WAM mitigates this via a
multiscale video-latent layout
. Specifically, the current observation is encoded via a VAE into high-resolution condition tokens (e.g.,
384
×
320
384\times 320
). Conversely, target future frames are spatially downsampled to a reduced
future video size
(e.g.,
192
×
160
192\times 160
) before VAE encoding, yielding token-sparse, low-resolution future latents. Both sets of latents are patchified and concatenated to form a unified visual context. The action expert then performs joint video-action attention over this multiscale token sequence. This ensures the action branch retains high-fidelity spatial details of the current state while utilizing the low-resolution future latents merely as a coarse dynamic guide. This design stems from our core hypothesis: effective control requires preserving task-relevant geometry, motion tendencies, and contact cues, rather than generating visually sharp future frames. As demonstrated in our ablations, this intentional degradation in future token density preserves action accuracy while substantially reducing attention cost and accelerating inference.
3.4
Asymmetric Video-Action Denoising
In generative WAMs, video and action branches conventionally share the same iterative denoising schedule. However, visual structure and precise control coordinates converge at different rates. Action generation requires precise multi-step sampling to yield safe, executable trajectories. Future video only needs to provide coarse dynamic context. Because global structural cues, like object geometry and contact boundaries, emerge in the very first few denoising steps, executing a long sampling schedule to hallucinate photorealistic textures is computationally wasteful.
We exploit this divergence by introducing training-free asymmetric video-action denoising during inference. We allocate a larger denoising budget to the action branch (e.g., 5 to 10 steps) and refresh the video branch with far fewer steps (e.g., only the initial 2 steps). Between video refresh steps, the model reuses cached video features to condition the ongoing action refinement. This scheduling drastically reduces computational overhead by ceasing video generation once the actionable dynamics are clear, yielding significant acceleration with negligible impact on task success.
3.5
Training Objectives
Efficient-WAM trains both branches via conditional flow matching
[
18
,
19
]
. Let
𝐱
1
\mathbf{x}_{1}
denote the target data (clean future video latents
𝐱
1
v
\mathbf{x}_{1}^{v}
or action chunks
𝐱
1
a
\mathbf{x}_{1}^{a}
), and
𝐱
0
∼
𝒩
⁡
(
0
,
I
)
\mathbf{x}_{0}\sim\mathcal{N}(0,I)
. We define the interpolation path
𝐱
t
=
(
1
−
t
)
​
𝐱
0
+
t
​
𝐱
1
\mathbf{x}_{t}=(1-t)\mathbf{x}_{0}+t\mathbf{x}_{1}
and target velocity
𝐮
t
=
𝐱
1
−
𝐱
0
\mathbf{u}_{t}=\mathbf{x}_{1}-\mathbf{x}_{0}
. The unified objective is:
ℒ
FM
=
𝔼
t
,
𝐱
0
,
𝐱
1
​
[
‖
f
⁡
(
𝐱
t
,
t
,
c
)
−
𝐮
t
‖
2
2
]
\mathcal{L}_{\text{FM}}=\mathbb{E}_{t,\mathbf{x}_{0},\mathbf{x}_{1}}\left[\left\|f(\mathbf{x}_{t},t;c)-\mathbf{u}_{t}\right\|_{2}^{2}\right]
(3)
where
f
f
is the respective prediction network and
c
c
provides conditioning. Training proceeds in three phases. First, we adapt the compact video expert using
ℒ
stage-1
=
ℒ
video-FM
+
λ
distill
​
ℒ
distill
\mathcal{L}_{\text{stage-1}}=\mathcal{L}_{\text{video-FM}}+\lambda_{\text{distill}}\mathcal{L}_{\text{distill}}
, where
ℒ
distill
\mathcal{L}_{\text{distill}}
explicitly transfers hidden representations and temporal motion cues from the full WAN model. Second, we attach the action expert and train it with the video branch frozen, utilizing a joint loss
ℒ
stage-2
=
ℒ
action-FM
+
λ
v
​
ℒ
video-FM
\mathcal{L}_{\text{stage-2}}=\mathcal{L}_{\text{action-FM}}+\lambda_{v}\mathcal{L}_{\text{video-FM}}
. This ensures the future-imagination branch remains aligned while the action expert learns executable control. Finally, a third phase co-trains both experts end-to-end using the same joint objective for unified refinement.
