# DC-WAM: Dynamic-Centric Visual Supervision and Reasoning for World-Action Models

paper_id: arxiv:2607.25918v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Figure 1:
Effect of dynamic-centric supervision and routing.
Our method achieves a relatively low PSNR, but improves policy success on LIBERO-Plus evaluation, suggesting that appearance-level reconstruction fidelity is not aligned with control-relevant future prediction.
Vision-Language-Action (VLA) models map visual observations and language instructions directly to robot actions, leveraging the semantic priors of large vision-language backbones
(
Kim et al. 2024
;
Black et al. 2024
)
. World-Action Models (WAMs) further couple action generation with future visual prediction, using anticipated scene evolution as an additional learning signal for robot control
(
Zhu et al. 2025
;
Li et al. 2025
;
Ye et al. 2026b
;
Li et al. 2026b
;
Kim et al. 2026
)
. By jointly modeling what the robot should do and how the scene may change, WAMs provide dense temporal supervision beyond action imitation alone.
However, uniform RGB future prediction entangles manipulation dynamics with appearance factors such as texture, illumination, background, and sensor noise. Although these factors dominate reconstruction errors, they often change across environments without altering the underlying manipulation dynamics, making appearance-centric supervision potentially brittle under visual distribution shifts. Consequently, an RGB video branch may allocate substantial capacity to appearance reconstruction rather than the gripper motion, object displacement, and contact events that are more directly useful for action learning.
Recent efficient WAMs show that strong action policies do not necessarily require complete future videos to be iteratively generated during execution
(
Yuan et al. 2026
;
Li et al. 2026a
;
Zhang et al. 2026c
)
. Meanwhile, other methods reduce the ambiguity of RGB prediction by introducing structured future representations, such as semantic masks, point trajectories, optical flow, or geometric states
(
Yu et al. 2026
;
Guan et al. 2026
;
Ranasinghe et al. 2026
;
Liu et al. 2026
;
Zhang et al. 2026a
)
. Although these representations provide more structured supervision, they typically modify or expand the predicted future-state modality. We instead ask whether the existing RGB-based WAM can be redirected toward dynamic-centric representations that remain stable across appearance shifts, without additional modality-specific prediction.
We propose
DC-WAM
, a dynamic-centric World-Action Model that redistributes the supervision and reasoning toward interaction-induced dynamics. At the supervision level, DC-WAM replaces uniform latent reconstruction with two complementary training signals. First, temporal-difference supervision emphasizes changes between adjacent visual flow fields, reducing the influence of temporally persistent appearance components. Second, tracker-guided flow matching reweights the visual objective toward localized regions with strong gripper, object, and contact motion. The tracker-derived targets are constructed offline from training videos and are not required during policy execution. At the reasoning level, we introduce
DynaRoute
, a lightweight module that predicts token-wise dynamic relevance and converts it into an attention bias. This routes visual attention toward future tokens associated with interaction-induced motion.
We retain an action-conditioned visual branch, allowing video supervision to regularize action representations through an explicit action-to-visual pathway
(
Ye et al. 2026a
)
. The RGB branch is used during training but can be removed at deployment following Fast-WAM-style inference, preserving efficient action generation.
Experiments show that DC-WAM substantially improves robustness under distribution shift. As summarized in Fig.
1
, the proposed method progressively improves LIBERO-Plus success, demonstrating that DC-WAM improves OOD generalization while maintaining strong clean-condition performance.
In summary, our contributions are threefold:
•
We propose DC-WAM, which redirects the existing RGB video branch from appearance-dominated reconstruction toward interaction-induced visual dynamics, without introducing additional modality-specific prediction branches or deployment-time inputs.
•
We introduce complementary dynamic-centric supervision and reasoning mechanisms: temporal-difference supervision suppresses persistent appearance components, tracker-guided flow matching emphasizes localized manipulation dynamics, and DynaRoute routes
visual attention toward dynamically relevant tokens.
•
Trained only on clean demonstrations, DC-WAM maintains strong in-distribution performance while substantially improving OOD success and reducing ID–OOD degradation on LIBERO-Plus and real-world manipulation under unseen visual perturbations.

## Method

Overview.
We consider language-conditioned robotic manipulation from visual observations. Given the current RGB image
𝐨
t
∈
ℝ
3
×
H
×
W
\mathbf{o}_{t}\in\mathbb{R}^{3\times H\times W}
, language instruction
ℓ
\ell
, and proprioceptive state
𝐬
t
\mathbf{s}_{t}
, a World-Action Model (WAM) jointly predicts an action chunk and future visual evolution:
p
θ
(
𝐚
t
:
t
+
T
a
−
1
,
𝐨
^
t
+
1
:
t
+
T
v
∣
𝐨
t
,
ℓ
,
𝐬
t
)
,
p_{\theta}\!\left(\mathbf{a}_{t:t+T_{a}-1},\mathbf{\hat{o}}_{t+1:t+T_{v}}\mid\mathbf{o}_{t},\ell,\mathbf{s}_{t}\right),
(1)
where
T
a
T_{a}
and
T
v
T_{v}
denote the action and visual prediction horizons, respectively.
Architecture.
DC-WAM is built on the Wan2.2 backbone
(
Wan et al. 2025
)
and follows a visual-action MoT architecture with two branches: an RGB visual branch for future latent video prediction and an action branch for action-chunk generation. We keep the original RGB-based visual branch and do not introduce additional modality-specific experts as extra token vectors in MoT computation.
The visual branch is action-conditioned through the MoT attention mask. Future visual tokens can attend to action tokens, while action tokens are prevented from attending to future visual tokens and can only attend to the current observed visual tokens, avoiding future leakage during action prediction.
Building on this backbone, DC-WAM modifies the training and forward computation of the RGB visual branch in two ways: dynamic-centric visual supervision and DynaRoute attention bias, as illustrated in Fig.
2
.
We describe these two components next.
Tracker-Derived Dynamic Map Construction
We construct tracker-derived dynamic maps offline at the episode level, providing concentration on manipulation dynamics. They are not used during policy execution.
Given an episode with
T
ep
T_{\mathrm{ep}}
frames, we sample candidate points either uniformly over the image or within foreground regions produced by SAM
(
Kirillov et al. 2023
)
. An off-the-shelf point tracker
(
Karaev et al. 2024
)
estimates their trajectories throughout the episode. Let the position of the
n
n
-th tracked point at global time
t
t
be
𝐩
t
,
n
=
(
x
t
,
n
,
y
t
,
n
)
∈
[
0
,
W
)
×
[
0
,
H
)
,
\mathbf{p}_{t,n}=(x_{t,n},y_{t,n})\in[0,W)\times[0,H),
(2)
where
t
=
0
,
…
,
T
ep
−
1
t=0,\ldots,T_{\mathrm{ep}}-1
. We compute the frame-wise motion magnitude using the backward temporal difference:
d
t
,
n
=
‖
𝐩
t
,
n
−
𝐩
t
−
1
,
n
‖
2
,
t
=
1
,
…
,
T
ep
−
1
,
d_{t,n}=\left\|\mathbf{p}_{t,n}-\mathbf{p}_{t-1,n}\right\|_{2},\quad t=1,\ldots,T_{\mathrm{ep}}-1,
(3)
and set
d
0
,
n
=
0
d_{0,n}=0
.
Dynamic points at time
t
t
are selected by
𝒫
dyn
t
=
{
n
∣
d
t
,
n
>
δ
mot
}
,
\mathcal{P}^{t}_{\mathrm{dyn}}=\left\{n\mid d_{t,n}>\delta_{\mathrm{mot}}\right\},
(4)
where
δ
mot
\delta_{\mathrm{mot}}
filters out static points and small tracking fluctuations.
We rasterize the selected dynamic points onto the VAE visual-token grid. Let
(
x
k
,
y
k
)
(x_{k},y_{k})
denote the center of the
k
k
-th visual token in the original image coordinate system, where
k
=
1
,
…
,
H
z
​
W
z
k=1,\ldots,H_{z}W_{z}
and
H
z
×
W
z
H_{z}\times W_{z}
is the spatial resolution of the VAE visual latent. The spatial response between point
n
n
and token
k
k
is
κ
t
,
n
,
k
=
exp
⁡
[
−
λ
⁡
(
(
x
k
−
x
t
,
n
)
2
σ
x
2
+
(
y
k
−
y
t
,
n
)
2
σ
y
2
)
]
,
\kappa_{t,n,k}=\exp\left[-\lambda\left(\frac{(x_{k}-x_{t,n})^{2}}{\sigma_{x}^{2}}+\frac{(y_{k}-y_{t,n})^{2}}{\sigma_{y}^{2}}\right)\right],
(5)
with
σ
x
=
W
W
z
​
σ
p
,
σ
y
=
H
H
z
​
σ
p
.
\sigma_{x}=\frac{W}{W_{z}}\sigma_{p},\quad\sigma_{y}=\frac{H}{H_{z}}\sigma_{p}.
Unless otherwise specified, we set
λ
=
0.25
\lambda=0.25
and
σ
p
=
1.25
\sigma_{p}=1.25
.
The unnormalized token-level dynamic response is obtained by aggregating motion-weighted kernel responses:
b
t
,
k
=
∑
n
∈
𝒫
dyn
t
d
t
,
n
​
κ
t
,
n
,
k
.
b_{t,k}=\sum_{n\in\mathcal{P}^{t}_{\mathrm{dyn}}}d_{t,n}\kappa_{t,n,k}.
(6)
Finally, we normalize the response over the entire episode and within each camera view:
m
t
,
k
∗
=
b
t
,
k
max
t
′
,
k
′
⁡
b
t
′
,
k
′
+
ϵ
∈
[
0
,
1
]
.
m^{*}_{t,k}=\frac{b_{t,k}}{\max_{t^{\prime},k^{\prime}}b_{t^{\prime},k^{\prime}}+\epsilon}\in[0,1].
(7)
This episode-level normalization preserves both spatial and temporal saliency: it highlights tokens near strong tracked motion and assigns larger relevance values to time steps where interaction-induced changes are more pronounced.
An important consequence of this construction is its reduced
sensitivity to temporally persistent appearance shifts.
Because the dynamic map is derived from inter-frame point
displacements rather than RGB reconstruction errors, static or
slowly varying changes in illumination and background texture
do not directly contribute to the supervision target, provided
that the underlying point trajectories remain stable.
Consequently, the visual branch is encouraged to prioritize
gripper motion, object displacement, and contact-related changes
instead of fitting nuisance appearance variations. This provides
a natural source of robustness to appearance-level OOD
perturbations, such as lighting and background changes, that alter
visual appearance without changing the underlying manipulation
dynamics.
These maps are used to reweight the original visual flow-matching loss toward sparse interaction regions and to supervise the dynamic relevance predicted by DynaRoute, whose outputs are defined on the DiT visual-token grid. We further downsample the map to the DiT token resolution
H
D
×
W
D
H_{D}\times W_{D}
:
m
~
t
,
k
∗
=
𝒟
tok
(
m
t
,
⋅
∗
)
k
,
k
=
1
,
…
,
H
D
W
D
,
\widetilde{m}^{*}_{t,k}=\mathcal{D}_{\mathrm{tok}}\left(m^{*}_{t,\cdot}\right)_{k},\quad k=1,\ldots,H_{D}W_{D},
(8)
where
𝒟
tok
\mathcal{D}_{\mathrm{tok}}
denotes patch-wise downsampling from the VAE latent grid to the DiT token grid.
Dynamics-Aware Attention Bias
DynaRoute predicts token-wise relevance for future visual tokens and
converts it into an additive key-side bias in the visual branch.
It encourages the video expert to prioritize interaction-induced
dynamics, such as object displacement and robot-environment contact,
rather than attending uniformly to all visual tokens.
Let
𝐕
τ
∈
ℝ
B
×
S
v
×
d
v
\mathbf{V}_{\tau}\in\mathbb{R}^{B\times S_{v}\times d_{v}}
denote the DiT visual-token sequence, where
S
v
=
T
v
​
K
D
S_{v}=T_{v}K_{D}
,
K
D
=
H
D
​
W
D
K_{D}=H_{D}W_{D}
, and
d
v
d_{v}
is the visual hidden
dimension. We decompose it as
𝐕
τ
=
[
𝐕
obs
,
𝐕
τ
fut
]
,
\mathbf{V}_{\tau}=[\mathbf{V}^{\mathrm{obs}},\mathbf{V}_{\tau}^{\mathrm{fut}}],
(9)
where
𝐕
obs
\mathbf{V}^{\mathrm{obs}}
contains the clean current-observation tokens, while
𝐕
τ
fut
\mathbf{V}^{\mathrm{fut}}_{\tau}
represents the
future visual tokens at timestep
τ
\tau
. The observation tokens remain clean at all timesteps. Let
𝐀
τ
\mathbf{A}_{\tau}
denote the action tokens at the same timestep, and let
𝐒
\mathbf{S}
denote the fused language and proprioceptive conditioning tokens.
DynaRoute is evaluated once at each diffusion timestep:
𝐳
τ
=
G
ψ
​
(
sg
⁡
(
𝐕
τ
)
,
sg
⁡
(
𝐀
τ
)
,
𝐒
,
τ
)
∈
ℝ
B
×
S
v
,
\mathbf{z}_{\tau}=G_{\psi}\!\left(\operatorname{sg}(\mathbf{V}_{\tau}),\operatorname{sg}(\mathbf{A}_{\tau}),\mathbf{S},\tau\right)\in\mathbb{R}^{B\times S_{v}},
(10)
where
sg
⁡
(
⋅
)
\operatorname{sg}(\cdot)
denotes stop-gradient. The predicted dynamic relevance is
𝐠
τ
=
σ
⁡
(
𝐳
τ
)
∈
(
0
,
1
)
B
×
S
v
,
\mathbf{g}_{\tau}=\sigma(\mathbf{z}_{\tau})\in(0,1)^{B\times S_{v}},
(11)
and is supervised by the downsampled tracker-derived map
𝐦
~
∗
\widetilde{\mathbf{m}}^{*}
defined in Eq.
8
.
For each future visual token
i
i
, we convert its relevance into a log-space bias:
b
i
=
α
​
log
⁡
(
max
⁡
(
g
i
,
ϵ
)
)
,
b_{i}=\alpha\log\!\left(\max(g_{i},\epsilon)\right),
(12)
where
α
\alpha
controls the routing strength and
ϵ
\epsilon
ensures numerical stability. Low-relevance tokens therefore receive stronger negative biases. We further center the bias over visual tokens to stabilize the bias scale:
b
¯
i
=
b
i
−
1
S
v
​
∑
q
=
1
S
v
b
q
.
\bar{b}_{i}=b_{i}-\frac{1}{S_{v}}\sum_{q=1}^{S_{v}}b_{q}.
(13)
Let
𝐐
ℓ
V
\mathbf{Q}_{\ell}^{V}
,
𝐊
ℓ
V
\mathbf{K}_{\ell}^{V}
, and
𝐕
ℓ
V
\mathbf{V}_{\ell}^{V}
denote the visual queries, keys, and values at the
ℓ
\ell
-th MoT layer. The same bias is shared across all layers at the current diffusion timestep:
Attn
ℓ
V
←
V
=
softmax
⁡
(
𝐐
ℓ
V
​
(
𝐊
ℓ
V
)
⊤
d
+
𝐁
¯
)
​
𝐕
ℓ
V
,
\operatorname{Attn}_{\ell}^{V\leftarrow V}=\operatorname{softmax}\!\left(\frac{\mathbf{Q}_{\ell}^{V}(\mathbf{K}_{\ell}^{V})^{\top}}{\sqrt{d}}+\bar{\mathbf{B}}\right)\mathbf{V}_{\ell}^{V},
(14)
where
𝐁
¯
\bar{\mathbf{B}}
is obtained by broadcasting
𝐛
¯
\bar{\mathbf{b}}
over attention heads and visual-query positions.
Additional implementation details of DynaRoute, together with further analyses of its routing behavior and effectiveness, are provided in Appendix.
Action-only inference with routed video cache.
As illustrated in Fig.
4
, DC-WAM follows
Fast-WAM-style action-only inference at deployment. Rather than
iteratively denoising future video, it executes the video branch only
once to construct a routed visual key-value cache
𝒞
V
\mathcal{C}_{V}
for
the action branch.
At the initial diffusion step
τ
init
=
1
\tau_{\mathrm{init}}=1
, we form the
pseudo-video and noisy action inputs as
𝐕
~
=
[
𝐕
obs
,
ϵ
V
]
,
ϵ
V
∼
𝒩
⁡
(
𝟎
,
𝐈
)
,
\widetilde{\mathbf{V}}=[\mathbf{V}^{\mathrm{obs}},\boldsymbol{\epsilon}^{V}],\quad\boldsymbol{\epsilon}^{V}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
(15)
and
𝐀
~
=
ϵ
A
,
ϵ
A
∼
𝒩
⁡
(
𝟎
,
𝐈
)
,
\widetilde{\mathbf{A}}=\boldsymbol{\epsilon}^{A},\quad\boldsymbol{\epsilon}^{A}\sim\mathcal{N}(\mathbf{0},\mathbf{I}),
(16)
where
𝐕
obs
\mathbf{V}^{\mathrm{obs}}
is the clean observed-frame latent and
ϵ
V
\boldsymbol{\epsilon}^{V}
occupies the future visual slots. DynaRoute is evaluated once to produce
𝐛
¯
cache
=
Bias
⁡
(
G
ψ
​
(
sg
⁡
(
𝐕
~
)
,
sg
⁡
(
𝐀
~
)
,
𝐒
,
τ
init
)
)
,
\bar{\mathbf{b}}_{\mathrm{cache}}=\mathrm{Bias}\!\left(G_{\psi}\!\left(\operatorname{sg}(\widetilde{\mathbf{V}}),\operatorname{sg}(\widetilde{\mathbf{A}}),\mathbf{S},\tau_{\mathrm{init}}\right)\right),
(17)
which is injected during video-cache prefill. After constructing
𝒞
V
\mathcal{C}_{V}
, the video branch is no longer executed and the action
branch reuses the cached visual keys and values for all subsequent
denoising steps.
This design preserves train-inference consistency for DynaRoute, as it closely matches the high-noise regime used during training. At large diffusion timesteps, DynaRoute infers dynamic relevance primarily from the clean observation, language instruction, and proprioceptive state; at lower-noise timesteps, it can additionally exploit partially preserved information in the action and future-visual tokens.
Figure 4:
DynaRoute during training and inference. During training, a layer-shared bias is predicted at each diffusion timestep and injected into video-query attention.
At deployment, DynaRoute is evaluated once at
τ
init
=
1
\tau_{\mathrm{init}}=1
to construct the routed visual cache
𝒞
V
\mathcal{C}_{V}
. The video branch is then disabled and the cache is
reused throughout action denoising.
Training Objective
The complete training objective is
ℒ
=
ℒ
FM
A
+
ℒ
TD
V
+
ℒ
TrackFM
V
+
ℒ
Route
,
\mathcal{L}=\mathcal{L}_{\mathrm{FM}}^{A}+\mathcal{L}_{\mathrm{TD}}^{V}+\mathcal{L}_{\mathrm{TrackFM}}^{V}+\mathcal{L}_{\mathrm{Route}},
(18)
where the four terms supervise action generation, dense temporal dynamics, sparse interaction regions, and DynaRoute relevance prediction, respectively.
Flow matching.
For a clean sample
𝐱
\mathbf{x}
and Gaussian noise
ϵ
∼
𝒩
⁡
(
𝟎
,
𝐈
)
\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
, we use the linear interpolation
𝐱
τ
=
(
1
−
τ
)
​
𝐱
+
τ
​
ϵ
,
τ
∼
𝒰
⁡
(
0
,
1
)
,
\mathbf{x}_{\tau}=(1-\tau)\mathbf{x}+\tau\boldsymbol{\epsilon},\quad\tau\sim\mathcal{U}(0,1),
(19)
with the flow target
𝐮
=
ϵ
−
𝐱
.
\mathbf{u}=\boldsymbol{\epsilon}-\mathbf{x}.
(20)
For actions, the model predicts
𝐮
^
A
\widehat{\mathbf{u}}^{A}
and is trained with
ℒ
FM
A
=
𝔼
⁡
[
‖
𝐮
^
A
−
𝐮
A
‖
2
2
]
.
\mathcal{L}_{\mathrm{FM}}^{A}=\mathbb{E}\left[\left\|\widehat{\mathbf{u}}^{A}-\mathbf{u}^{A}\right\|_{2}^{2}\right].
(21)
To emphasize state transitions, we impose a temporal-difference loss on the visual flow velocity at the VAE latent resolution
(
Gao et al. 2026
)
. Let
𝐮
^
t
V
\widehat{\mathbf{u}}^{V}_{t}
and
𝐮
V
t
=
ϵ
t
V
−
𝐱
t
V
\mathbf{u}^{V_{t}}=\boldsymbol{\epsilon}^{V}_{t}-\mathbf{x}^{V}_{t}
denote the predicted visual velocity and the target flow velocity at time step t. The temporal difference objective
ℒ
TD
V
\mathcal{L}_{\mathrm{TD}}^{V}
enforces temporal consistency by matching adjacent-frame transitions between the predicted and target visual velocity fields:
ℒ
TD
V
=
𝔼
⁡
[
∑
t
=
1
T
v
−
1
‖
(
𝐮
𝐭
^
V
−
𝐮
^
t
−
1
V
)
−
(
𝐮
t
V
−
𝐮
t
−
1
V
)
‖
2
2
]
.
\mathcal{L}_{\mathrm{TD}}^{V}=\mathbb{E}\Bigg[\sum_{t=1}^{T_{v}-1}\Big\|(\widehat{\mathbf{u_{t}}}^{V}-\widehat{\mathbf{u}}^{V}_{t-1})-(\mathbf{u}^{V}_{t}-\mathbf{u}^{V}_{t-1})\Big\|_{2}^{2}\Bigg].
(22)
This temporal-difference loss suppresses temporally invariant appearance components, but it does not explicitly localize task-critical interaction regions.
We therefore use the tracker-derived dynamic map
𝐦
∗
∈
[
0
,
1
]
T
v
×
H
z
​
W
z
\mathbf{m}^{*}\in[0,1]^{T_{v}\times H_{z}W_{z}}
to reweight the original visual FM error on the VAE latent grid.
Since
𝐦
∗
\mathbf{m}^{*}
is constructed from thresholded tracked motion, most static background cells have zero or near-zero weights, making the map spatially sparse.
For each latent frame
t
t
and cell
p
p
, let
e
t
,
p
V
=
‖
𝐮
^
t
,
p
V
−
𝐮
t
,
p
V
‖
2
2
.
e^{V}_{t,p}=\left\|\widehat{\mathbf{u}}^{V}_{t,p}-\mathbf{u}^{V}_{t,p}\right\|_{2}^{2}.
(23)
The tracker-guided objective is
ℒ
TrackFM
V
=
𝔼
⁡
[
∑
t
,
p
m
t
,
p
∗
​
e
t
,
p
V
∑
t
,
p
m
t
,
p
∗
+
ϵ
]
.
\mathcal{L}_{\mathrm{TrackFM}}^{V}=\mathbb{E}\left[\frac{\sum_{t,p}m^{*}_{t,p}e^{V}_{t,p}}{\sum_{t,p}m^{*}_{t,p}+\epsilon}\right].
(24)
Thus, TrackFM redistributes the visual FM loss toward sparse regions with strong tracked motion, such as the end effector, manipulated objects, and contact areas.
Together, the temporal-difference and TrackFM objectives provide complementary dense and sparse dynamic supervision.
DynaRoute Relevance Prediction.
The downsampled token-level target
𝐦
~
∗
∈
[
0
,
1
]
T
v
×
K
D
\widetilde{\mathbf{m}}^{*}\in[0,1]^{T_{v}\times K_{D}}
from Eq. (
8
) is used as the supervision target for DynaRoute. Given the predicted relevance
𝐠
∈
[
0
,
1
]
T
v
×
K
D
\mathbf{g}\in[0,1]^{T_{v}\times K_{D}}
, we combine binary cross-entropy with soft Dice losses:
ℒ
Route
=
ℒ
BCE
​
(
𝐠
,
𝐦
~
∗
)
+
λ
Dice
​
ℒ
Dice
​
(
𝐠
,
𝐦
~
∗
)
,
\mathcal{L}_{\mathrm{Route}}=\mathcal{L}_{\mathrm{BCE}}\left(\mathbf{g},\widetilde{\mathbf{m}}^{*}\right)+\lambda_{\mathrm{Dice}}\mathcal{L}_{\mathrm{Dice}}\left(\mathbf{g},\widetilde{\mathbf{m}}^{*}\right),
(25)
where
ℒ
Dice
=
1
−
2
​
∑
t
,
k
g
t
,
k
​
m
~
t
,
k
∗
+
ϵ
∑
t
,
k
g
t
,
k
+
∑
t
,
k
m
~
t
,
k
∗
+
ϵ
.
\mathcal{L}_{\mathrm{Dice}}=1-\frac{2\sum_{t,k}g_{t,k}\widetilde{m}^{*}_{t,k}+\epsilon}{\sum_{t,k}g_{t,k}+\sum_{t,k}\widetilde{m}^{*}_{t,k}+\epsilon}.
(26)
The BCE term provides token-wise relevance supervision, while the Dice term mitigates the imbalance caused by sparse dynamic regions.
