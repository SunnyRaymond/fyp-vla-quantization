# JEPA-WAM: Learning Vision-Language-Action Policies with Joint-Embedding World Modeling

paper_id: arxiv:2608.09381v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Vision-language-action (VLA) policies have achieved strong performance across diverse manipulation tasks
(
Kim et al. 2024
;
Black et al. 2024
;
Liu et al. 2025
)
, but their action prediction objectives model state transitions only implicitly, which can limit robustness under distribution shift.
World action models (WAMs) address this by explicitly modeling future states alongside action generation
(
Cen et al. 2025
;
Li et al. 2026b
;
Ye et al. 2026b
)
.
However, video generation based WAMs incur substantial deployment cost due to iterative future prediction.
This motivates latent WAMs, which retain predictive world modeling without generating future observations.
Figure 1:
Overview and performance of JEPA-WAM.
Top: JEPA-WAM couples latent transition prediction in the V-JEPA space with action generation through a shared predictor.
Bottom: JEPA-WAM maintains strong in-distribution performance while improving generalization under out-of-distribution shifts across LIBERO, RoboTwin 2.0, and real-world manipulation; the same transition supervision also benefits pretrained
π
0.5
\pi_{0.5}
.
Without explicit future-frame generation, latent WAMs must address two complementary design questions:
what predictive target should be learned
, and
how should predictive supervision be integrated with action generation?
Regarding the target, existing methods often reuse intermediate features from pretrained video generators
(
Yuan et al. 2026
)
or compress future observations into a small number of latent tokens or subgoals
(
Luo et al. 2026a
;
Chen et al. 2026
)
.
Although efficient, generator features are optimized for iterative future-frame generation rather than explicitly representing state changes, while compact future representations may lose fine-grained spatial structure.
Regarding policy integration, existing approaches either use predicted future representations as additional context for the action module
(
Ma et al. 2026
)
or introduce a separate prediction objective or latent dynamics module alongside the policy
(
Sun et al. 2026
)
.
The former may expose the action module to redundant future-state information, whereas the latter may only weakly influence the representations from which actions are generated.
These limitations motivate a latent WAM that learns a spatially structured representation of the observed transition and uses predictive supervision to directly shape the policy backbone responsible for action conditioning.
To address these limitations, we introduce JEPA-WAM, a latent WAM that retains transition modeling without explicit future generation.
To answer the first question about what predictive representation to learn, we build JEPA-WAM in the pretrained V-JEPA 2.1 representation space and construct a target that represents the transition rather than the absolute future state.
V-JEPA’s video pretraining produces temporally consistent representations
(
Mur-Labadia et al. 2026
)
, allowing the jointly encoded current–future observations to capture their temporal relation.
Rather than reconstructing a unique future observation, the joint target captures stable and changing regions and evolving local object and spatial relations, while preserving dense patch-level structure instead of being globally pooled or compressed, allowing the joint target to preserve fine-grained spatial information.
This formulation can also be applied to pretrained VLA policies; we instantiate it in
π
0.5
\pi_{0.5}
with an auxiliary transition prediction branch while preserving its original pathways.
To answer the second question about how transition prediction should be used for action generation, JEPA-WAM uses a shared predictor for transition modeling and action generation.
In a single forward pass, it predicts the dense transition target from the current observation while producing dedicated representations to condition the action generation.
This allows transition supervision to directly shape the same backbone used for action generation, while the action expert does not need to rely on the full predicted transition representation.
At deployment, latent transition prediction is removed and only action generation is retained.
As summarized in Figure
1
, JEPA-WAM maintains competitive in-distribution (ID) performance while showing strong generalization under visual and spatial out-of-distribution (OOD) shifts. On LIBERO-Plus, JEPA-WAM achieves 79.2, the best result among methods without robot-policy pretraining. When applied to the transition target in the pretrained
π
0.5
\pi_{0.5}
, it improves the average from 84.5 to 86.3, achieving the best overall result. JEPA-WAM further generalizes well to randomized bimanual manipulation on RoboTwin 2.0 and to real-world manipulation under visual and spatial shifts. Controlled ablations support the joint current–future target, patch-level spatial supervision, and direct transition prediction through the shared backbone.
Our contributions are threefold:
•
We introduce JEPA-WAM, a latent WAM built in a pretrained V-JEPA representation space, where a shared predictor couples latent transition modeling with continuous action generation.
•
We construct a spatially structured joint current–future transition target that preserves dense patch-level information, and instantiate the same target formulation in pretrained VLA policies.
•
We demonstrate strong OOD generalization on LIBERO-Plus, RoboTwin 2.0, and real-world manipulation, with the pretrained
π
0.5
\pi_{0.5}
instantiation achieving the best overall result on LIBERO-Plus.

## Method

JEPA-WAM is a latent world action model built in a pretrained V-JEPA representation space.
It uses V-JEPA to represent the current visual state and define the transition target, while a shared predictor couples transition modeling with continuous action generation.
As shown in Figure
3
, it predicts a spatially structured joint current–future target while producing representations for action conditioning.
3.1
V-JEPA Representation Space and Joint Current–Future Target
We use pretrained V-JEPA 2.1 as the latent representation space of JEPA-WAM, representing both the current visual state and the joint current–future target for temporal supervision.
A frozen V-JEPA 2.1 encoder
E
J
E_{J}
provides dense patch-level representations rather than a globally pooled representation.
We refer to these representations as spatially structured because the tokens retain their patch organization within each view and are arranged in a fixed camera order across views.
We denote the resulting current representation as
Z
t
Z_{t}
.
During training, we construct a joint current–future target
Y
t
,
t
+
δ
Y_{t,t+\delta}
by encoding the observations at time
t
t
and
t
+
δ
t+\delta
together in the same representation space.
Current visual representation.
Let
V
V
index the available camera views,
O
t
v
O_{t}^{v}
denote the current observation from view
v
v
. The frozen V-JEPA encoder processes each view independently, and we concatenate the resulting visual tokens in a fixed camera order:
Z
t
=
Concat
v
∈
V
⁡
E
J
​
(
O
t
v
)
∈
ℝ
N
vis
×
d
J
,
Z_{t}=\operatorname{Concat}_{v\in V}E_{J}(O_{t}^{v})\in\mathbb{R}^{N_{\mathrm{vis}}\times d_{J}},
(1)
where
N
vis
N_{\mathrm{vis}}
is the total number of visual tokens and
d
J
d_{J}
is the V-JEPA feature dimension.
Thus, V-JEPA defines the latent visual state space of JEPA-WAM, with both the current representation and the transition target below expressed in this space.
The fixed camera and patch-token ordering is retained in the prediction target defined below.
Joint current–future target.
During training, each current observation is paired with an observation collected
δ
\delta
steps later.
For each camera view, we stack the current and future observations along the temporal dimension and jointly encode them with the frozen V-JEPA encoder:
Y
t
,
t
+
δ
\displaystyle Y_{t,t+\delta}
=
Concat
v
∈
V
⁡
sg
⁡
[
E
J
​
(
Stack
time
⁡
(
O
t
v
,
O
t
+
δ
v
)
)
]
,
\displaystyle=\operatorname{Concat}_{v\in V}\operatorname{sg}\!\left[E_{J}\!\left(\operatorname{Stack}_{\mathrm{time}}(O_{t}^{v},O_{t+\delta}^{v})\right)\right],
(2)
∈
ℝ
N
vis
×
d
J
.
\displaystyle\in\mathbb{R}^{N_{\mathrm{vis}}\times d_{J}}.
where
sg
\operatorname{sg}
denotes stop-gradient, and
δ
\delta
is a benchmark-specific temporal offset.
V-JEPA 2.1 uses modality-specific tokenizers, with its video tokenizer grouping every two frames into one temporal tubelet.
Therefore, the two-frame joint input produces the same spatial token grid as a single image, so
Y
t
,
t
+
δ
Y_{t,t+\delta}
and
Z
t
Z_{t}
share the same camera and spatial-token ordering.
Unlike a future-only target
E
J
​
(
O
t
+
δ
v
)
E_{J}(O_{t+\delta}^{v})
, which represents the future observation in isolation, the joint target makes both temporal endpoints available to the pretrained V-JEPA encoder.
Rather than requiring reconstruction of a complete or unique future observation, it emphasizes their visual relation: which regions remain stable or change, and how local object and spatial relations differ across time.
Together with its dense patch-level organization, this provides task-shared visual temporal supervision without compressing the transition into a small set of global latent tokens.
3.2
Shared Predictor for Transition Prediction and Action Generation
Having defined the joint current–future target, we next couple its prediction with action generation through a shared predictor.
We instantiate the shared predictor
F
θ
F_{\theta}
with Qwen2.5-0.5B, which processes the visual representation together with the task instruction and produces dedicated representations for action conditioning.
The same predictor is also supervised to predict the joint current–future target, allowing temporal supervision to directly optimize the backbone used for action generation.
Visual interface alignment.
To bridge the frozen V-JEPA encoder and the Qwen predictor, we introduce a lightweight visual projector
P
vis
P_{\mathrm{vis}}
that maps the current V-JEPA representation
Z
t
Z_{t}
into the predictor input space.
Following the single-stage finetuning setup of Prismatic
(
Karamcheti et al. 2024
)
, we keep the V-JEPA encoder frozen while jointly finetuning
P
vis
P_{\mathrm{vis}}
and the full Qwen2.5-0.5B backbone. During robot-policy training, the V-JEPA encoder, visual projector, and base Qwen weights are frozen, while the Qwen LoRA adapters are optimized together with the transition prediction head and action expert.
Shared predictor.
To couple latent transition prediction with action generation, we introduce a shared predictor
F
θ
F_{\theta}
that supports both prediction of the joint current–future target and extraction of dedicated representations for action conditioning.
The projected visual tokens, task instruction
ℓ
\ell
, and dedicated action placeholder tokens
P
act
P_{\mathrm{act}}
are processed by
F
θ
F_{\theta}
:
(
Q
t
wm
,
C
t
)
=
F
θ
​
(
P
vis
​
(
Z
t
)
,
ℓ
,
P
act
)
,
(Q_{t}^{\mathrm{wm}},C_{t})=F_{\theta}(P_{\mathrm{vis}}(Z_{t}),\ell,P_{\mathrm{act}}),
(3)
where
Q
t
wm
Q_{t}^{\mathrm{wm}}
denotes the hidden states at the visual-token positions and is used to predict the joint current–future target, while
C
t
C_{t}
denotes the dedicated representations used for action generation.
Q
t
wm
Q_{t}^{\mathrm{wm}}
preserves the fixed camera and spatial-token ordering of
Z
t
Z_{t}
, whereas
C
t
C_{t}
aggregates the preceding visual and task context.
Q
t
wm
Q_{t}^{\mathrm{wm}}
learns visual temporal structure rather than a complete instruction-conditioned future.
By sharing the predictor, supervision from latent transition prediction updates the same backbone from which action relevant representations are extracted, allowing the learned temporal visual patterns to benefit action generation.
Latent transition prediction.
For the transition prediction branch of the shared predictor, the hidden states
Q
t
wm
Q_{t}^{\mathrm{wm}}
produced by
F
θ
F_{\theta}
are mapped back to the V-JEPA representation space through a lightweight prediction head
G
ϕ
G_{\phi}
:
Y
^
t
,
t
+
δ
=
G
ϕ
​
(
Q
t
wm
)
∈
ℝ
N
vis
×
d
J
.
\hat{Y}_{t,t+\delta}=G_{\phi}(Q_{t}^{\mathrm{wm}})\in\mathbb{R}^{N_{\mathrm{vis}}\times d_{J}}.
(4)
Because
Q
t
wm
Q_{t}^{\mathrm{wm}}
preserves the fixed camera and spatial ordering of
Z
t
Z_{t}
,
Y
^
t
,
t
+
δ
\hat{Y}_{t,t+\delta}
maintains patch-level correspondence with the joint current–future target
Y
t
,
t
+
δ
Y_{t,t+\delta}
.
This spatial correspondence allows the model to capture spatially localized changes between the current and future observations at the patch level.
We therefore optimize the mean patch-level cosine distance:
ℒ
wm
=
1
B
​
N
vis
​
∑
b
=
1
B
∑
n
=
1
N
vis
(
1
−
cos
⁡
(
Y
^
t
,
t
+
δ
,
n
(
b
)
,
Y
t
,
t
+
δ
,
n
(
b
)
)
)
,
\mathcal{L}_{\mathrm{wm}}=\frac{1}{BN_{\mathrm{vis}}}\sum_{b=1}^{B}\sum_{n=1}^{N_{\mathrm{vis}}}\left(1-\cos\left(\hat{Y}_{t,t+\delta,n}^{(b)},Y_{t,t+\delta,n}^{(b)}\right)\right),
(5)
where
B
B
denotes the batch size.
Since the V-JEPA encoder and visual projector are frozen and the target representation is stop-gradient,
ℒ
wm
\mathcal{L}_{\mathrm{wm}}
updates
F
θ
F_{\theta}
together with the prediction head
G
ϕ
G_{\phi}
, providing patch-level temporal supervision to the shared predictor.
Action prediction and generation.
The dedicated action placeholders produce
C
t
C_{t}
, which is provided to the DiT action expert
A
ψ
A_{\psi}
.
Following the StarVLA action-head design
(
StarVLA Community 2026
)
, we use conditional flow matching with future tokens and proprioceptive state
s
t
s_{t}
.
Let
a
≡
a
t
:
t
+
H
−
1
a\equiv a_{t:t+H-1}
denote a demonstrated action chunk.
Given
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
and a flow time
τ
\tau
sampled using a Beta-based schedule, we define
a
τ
=
(
1
−
τ
)
​
ϵ
+
τ
​
a
a_{\tau}=(1-\tau)\epsilon+\tau a
and optimize
ℒ
act
=
𝔼
ϵ
,
τ
​
[
‖
A
ψ
​
(
a
τ
,
τ
,
s
t
,
C
t
)
−
(
a
−
ϵ
)
‖
2
2
]
.
\mathcal{L}_{\mathrm{act}}=\mathbb{E}_{\epsilon,\,\tau}\left[\left\|A_{\psi}(a_{\tau},\tau,s_{t},C_{t})-(a-\epsilon)\right\|_{2}^{2}\right].
(6)
Unless otherwise specified, we use the velocity-prediction objective above. For RoboTwin 2.0, we instead use x-prediction and directly predict the clean action trajectory from the noisy trajectory; see Appendix
B.2
.
3.3
Joint Training and Deployment
During policy training, we jointly optimize latent transition prediction and action generation:
ℒ
=
ℒ
act
+
λ
wm
​
ℒ
wm
,
\mathcal{L}=\mathcal{L}_{\mathrm{act}}+\lambda_{\mathrm{wm}}\mathcal{L}_{\mathrm{wm}},
(7)
where
λ
wm
\lambda_{\mathrm{wm}}
balances the two objectives.
Both losses update the shared predictor
F
θ
F_{\theta}
. Detailed optimization settings are provided in Appendix A.
At deployment, the target branch and prediction head are removed.
3.4
Transfer to Pretrained VLA Policies
Figure 4:
Transfer of the proposed transition supervision to a pretrained VLA policy.
The same joint current–future target can supervise pretrained VLAs without modifying their original perception or action pathways (Figure.
4
).
Given a pretrained VLA policy, we introduce a set of future tokens and use their output hidden states to predict the joint current–future target.
Let
R
t
∈
ℝ
H
f
​
W
f
×
d
f
R_{t}\in\mathbb{R}^{H_{f}W_{f}\times d_{f}}
denote these hidden states, where
H
f
×
W
f
H_{f}\times W_{f}
denotes their coarse spatial grid and
d
f
d_{f}
is the hidden dimension.
We arrange them as a coarse two-dimensional feature map,
R
~
t
=
Reshape
⁡
(
R
t
)
∈
ℝ
H
f
×
W
f
×
d
f
.
\tilde{R}_{t}=\operatorname{Reshape}(R_{t})\in\mathbb{R}^{H_{f}\times W_{f}\times d_{f}}.
(8)
We first reshape the future-token representations into a coarse two-dimensional feature map, then apply a lightweight projection and spatial upsampling to match the feature dimension and spatial resolution of the joint current–future target
Y
t
,
t
+
δ
Y_{t,t+\delta}
:
Y
^
t
,
t
+
δ
=
Upsample
⁡
(
P
sp
​
(
R
~
t
)
)
∈
ℝ
N
vis
×
d
J
.
\hat{Y}_{t,t+\delta}=\operatorname{Upsample}\left(P_{\mathrm{sp}}(\tilde{R}_{t})\right)\in\mathbb{R}^{N_{\mathrm{vis}}\times d_{J}}.
(9)
where
P
sp
P_{\mathrm{sp}}
denotes the lightweight projection used to match the target feature dimension. This alignment enables the pretrained policy to receive the same patch-level supervision from the joint current–future target while preserving its original action pathways. Additional implementation details are provided in Appendix A.4.
