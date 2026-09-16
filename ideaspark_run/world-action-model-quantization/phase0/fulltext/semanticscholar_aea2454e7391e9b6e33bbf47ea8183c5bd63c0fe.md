# OA-WAM: Object-Addressable World Action Model for Robust Robot Manipulation

paper_id: semanticscholar:aea2454e7391e9b6e33bbf47ea8183c5bd63c0fe
tier: T2
source_used: html_arxiv
warning: none

## Intro

Robot manipulation naturally unfolds at the object level. An instruction
like “put the red mug on the green tray” is not a request to replay a
training trajectory; it requires the robot to identify the
language-named objects in the current scene, reason about their spatial
and functional relations to surrounding distractors, and generate
correct, robust closed-loop actions. Vision-Language-Action (VLA)
policies have made rapid progress on standard manipulation benchmarks
by scaling vision-language backbones, robot data, and action
decoders
[
102
,
34
,
60
]
;
World Action Models (WAMs) build on this by jointly predicting actions
and future world states, supplying extra temporal and causal
supervision
[
9
,
90
,
89
]
.
However, robustness benchmarks show that high standard-benchmark scores
do not imply reliable scene understanding: under modest perturbations
of object layout, camera viewpoint, robot initial state, background,
lighting, or sensor noise, policies often collapse from near-saturated
success rates
[
19
,
61
,
23
]
, and
remain insensitive to paraphrased or even meaningless instruction
tokens
[
19
,
99
]
—both signs that target
selection is bound to training layouts and visual context rather than
to the language-named objects. Most existing WAMs still represent the
predicted world as full-frame observations, image/video token streams,
or shared global vision-action
latents
[
101
,
10
,
95
]
; such representations
capture how scenes evolve but provide the action decoder no stable
interface for deciding
which
object to act on. When camera,
layout, or robot pose changes, the target object often remains visible
yet becomes entangled with background and neighboring content inside
holistic tokens, causing action selection to drift.
Object-centric work attempts to fill this gap: slot-based dynamics
models improve sample efficiency and relational
reasoning
[
20
,
29
]
, and object-state grounding turns
object representations into operational interfaces between language and
action
[
13
,
26
,
88
]
. Yet many such
methods rely on Slot-Attention-style discovery from
scratch
[
50
,
81
,
82
]
,
which is sensitive to fixed slot counts, occlusion, and cluttered
scenes; and even with stable slots, each slot still mixes identity,
appearance, pose, and local context, so the action decoder may still
drift to the wrong instance under perturbation.
Based on this diagnosis, we propose OA-WAM, an Object-Addressable
WAM. Each frame is decomposed into
N
+
1
N{+}1
slots whose vectors are
split into a frozen identity address
addr
k
\mathrm{addr}_{k}
—computed once
per episode from the language label and the initial DINOv3 feature,
and never updated thereafter—and a time-varying content
cnt
k
t
\mathrm{cnt}_{k}^{t}
refreshed every step from SAM 3 + DINOv3 masks.
Slots are fused with text, image-VQ, proprioception, and past-action
streams into a block-causal sequence processed by a 7B Chameleon-style
trunk, from which a world head regresses next-frame per-slot
(
cnt
^
,
pose
^
)
(\hat{\mathrm{cnt}},\hat{\mathrm{pose}})
and a flow-matching action
head decodes a 16-step continuous action chunk in a single forward
pass. The OA constraint is enforced as two parameter-free tensor-level
operations applied at
each
of the 32 transformer layers: the
cross-slot key projection reads only the
addr
\mathrm{addr}
subvector,
and a forward hook resets the residual
addr
\mathrm{addr}
slice back to
the cached identity. Because
addr
\mathrm{addr}
is frozen per episode and
gradients into it are blocked by the reset, slot-routing keys at
every transformer layer are architecturally constrained to depend
only on the frozen identity address.
Conditional on correct
upstream slot extraction
, this removes a specific failure mode of
holistic backbones—scene-context shifts can no longer rewrite which
slot the action head reads from—while content, pose, and language
remain free to flow through the residual stream and value
projections; the constraint is on key routing rather than an
end-to-end semantic guarantee, and inherits any errors made by the
SAM 3 + Qwen3-VL slot pipeline.
Across three benchmarks, OA-WAM validates this design where the
hypothesis predicts. On in-distribution LIBERO and SimplerEnv WidowX
(Visual Matching) it reaches
97.8
97.8
and
79.3
79.3
avg, leading all
published VLA and WAM baselines (Table
1
). On
the LIBERO-Plus robustness benchmark (Table
2
) it
sets a new SOTA on the geometric axes most aligned with the
hypothesis—camera
80.5
80.5
(
+
4.7
+4.7
% over Cosmos-Policy),
robot-initial-state
89.6
89.6
(within
0.1
0.1
% of X-VLA), and a
Camera/Robot/Layout Geo Avg of
84.3
84.3
(
+
4.8
+4.8
% above
π
0.5
\pi_{0.5}
)—and is competitive on the seven-axis overall aggregate
(
83.9
83.9
vs
π
0.5
\pi_{0.5}
’s
85.7
85.7
), with the residual gap concentrated
on Sensor Noise where photometric corruption disrupts slot extraction
at the perception stage rather than the action policy itself. A causal
slot-intervention test gives OA-WAM a swap-binding cosine of
0.87
0.87
versus
≤
0.09
\leq 0.09
for all eight holistic baselines—direct
behavioural evidence that target selection is grounded in the explicit
address subspace—and an OA-isolation ablation shows that turning off
the addr-only key projection alone drops LP camera by
13.3
13.3
% while
leaving in-distribution LIBERO essentially unchanged (
−
1.5
-1.5
%), the
asymmetric signature of an OOD-specific inductive bias rather than a
generic capacity gain.
Figure 2:
OA-WAM architecture.
Multi-modal inputs are encoded into each token streams:
object-slot tokens via SAM3+DINOv3, projected by a learnable slot adapter;
Only slot tokens introduce learnable parameters; the others reuse frozen
embed_tokens
.
Tokens are assembled into a block-causal sequence terminated by a learnable action query
[ACT-Q]
and processed by the slot-aware backbone.
The world head reads slot hiddens to predict next-frame per-slot
(
cnt
^
,
pose
^
)
(\hat{\mathrm{cnt}},\hat{\mathrm{pose}})
as auxiliary supervision;
the action head decodes a
16
16
-step action chunk.
Our contributions are as follows:
•
We model the poor robustness of existing WAMs under
manipulation perturbations as a lack of object addressability: the
world state should not only predict the future, but also provide
stable per-object states so that language-conditioned action
generation can directly query task-relevant objects.
•
We propose OA-WAM, which uses per-slot
addr
/
content
\mathrm{addr}/\mathrm{content}
state tokens, a block-causal
world-action sequence, temporally aligned world/action heads, and
a two-part OA constraint (addr-only key projection plus per-layer
address-stream reset) that together architecturally decouple
object-identity addressing from time-varying content modeling.
•
We match SOTA-level performance on standard LIBERO and
SimplerEnv and lead on the camera and robot-initial-state axes of
LIBERO-Plus most directly aligned with our hypothesis; ablations
and slot-intervention tests confirm the object-addressable
interface is the key source of this robustness pattern.

## Method

3.1
Problem setup
At step
t
t
the agent observes RGB frames
𝐈
t
\mathbf{I}_{t}
(third-person and wrist views), proprioception
𝐪
t
∈
ℝ
7
\mathbf{q}_{t}\!\in\!\mathbb{R}^{7}
, a language instruction
ℓ
\ell
, and the past
T
−
1
T{-}1
executed actions
𝐚
<
t
\mathbf{a}_{<t}
. The policy
π
θ
\pi_{\theta}
produces, in a single forward pass, an action chunk
𝐀
t
=
(
𝐚
t
,
…
,
𝐚
t
+
H
−
1
)
∈
ℝ
H
×
7
\mathbf{A}_{t}\!=\!(\mathbf{a}_{t},\dots,\mathbf{a}_{t+H-1})\!\in\!\mathbb{R}^{H\times 7}
(
H
=
16
H{=}16
) and a per-object next-frame state prediction
𝒮
^
t
+
1
=
{
(
𝐜
^
k
t
+
1
,
𝐩
^
k
t
+
1
)
}
k
=
1
N
\widehat{\mathcal{S}}_{t+1}\!=\!\{(\hat{\mathbf{c}}_{k}^{t+1},\hat{\mathbf{p}}_{k}^{t+1})\}_{k=1}^{N}
:
(
𝐀
t
,
𝒮
^
t
+
1
)
∼
π
θ
​
(
𝐈
≤
t
,
𝐪
≤
t
,
ℓ
,
𝐚
<
t
)
.
\bigl(\mathbf{A}_{t},\,\widehat{\mathcal{S}}_{t+1}\bigr)\;\sim\;\pi_{\theta}\!\bigl(\mathbf{I}_{\leq t},\,\mathbf{q}_{\leq t},\,\ell,\,\mathbf{a}_{<t}\bigr).
(1)
OA-WAM parameterizes
π
θ
\pi_{\theta}
over an object-level input representation: each frame is decomposed into
N
+
1
N{+}1
slots, and cross-slot attention keys are constrained to depend only on object identity.
3.2
Object-slot tokenization and unified sequence
Frozen foundation perception produces six parallel token streams that share a single Chameleon-7B sequence
[
11
,
46
]
: BPE text; Qwen3-VL
[
1
]
noun phrases (used only as SAM 3 prompts and excluded from the trunk); Chameleon VQ-GAN image codes; SAM 3
[
7
]
+DINOv3
[
70
]
+pose object slots;
256
256
-bin discretized proprioception; and
256
256
-bin discretized past actions. Five streams reuse the pretrained Chameleon embedding table; only the slot stream introduces new parameters. For each slot
k
k
at frame
t
t
we form a
320
320
-dimensional slot vector
𝐬
k
t
=
[
𝐚𝐝𝐝𝐫
k
⏟
32
∥
𝐜𝐧𝐭
k
t
⏟
256
∥
𝝅
t
⏟
16
∥
𝝆
k
⏟
16
]
∈
ℝ
320
,
\mathbf{s}_{k}^{t}\;=\;\bigl[\,\underbrace{\mathbf{addr}_{k}}_{32}\;\big\|\;\underbrace{\mathbf{cnt}_{k}^{t}}_{256}\;\big\|\;\underbrace{\boldsymbol{\pi}^{t}}_{16}\;\big\|\;\underbrace{\boldsymbol{\rho}_{k}}_{16}\,\bigr]\;\in\;\mathbb{R}^{320},
(2)
where
𝐚𝐝𝐝𝐫
k
=
f
addr
(
[
ℓ
k
∥
𝐟
k
(
0
)
]
)
\mathbf{addr}_{k}\!=\!f_{\mathrm{addr}}([\boldsymbol{\ell}_{k}\|\mathbf{f}_{k}^{(0)}])
is computed once at
t
=
0
t{=}0
from the language label and the initial DINOv3 feature and is fixed throughout the episode (object identity),
𝐜𝐧𝐭
k
t
=
f
cnt
​
(
𝐫𝐚𝐰
k
t
)
\mathbf{cnt}_{k}^{t}\!=\!f_{\mathrm{cnt}}(\mathbf{raw}_{k}^{t})
is recomputed each frame (time-varying state),
𝝅
t
\boldsymbol{\pi}^{t}
is a sinusoidal embedding of the frame index, and
𝝆
k
\boldsymbol{\rho}_{k}
is a learned lookup over three role labels (
robot
,
object
,
padding
). A slot adapter
f
ϕ
:
ℝ
320
→
ℝ
4096
f_{\phi}\!:\!\mathbb{R}^{320}\!\to\!\mathbb{R}^{4096}
projects each slot into the trunk hidden dimension, and slot embeddings replace
⟨
slot
⟩
\langle\texttt{slot}\rangle
placeholder positions in the input sequence via the LLaVA-style
masked_scatter
pattern
[
47
,
37
]
. The full sequence is block-causal across frame groups and terminates in a learnable query token
[act_q]
, whose final hidden state is read by the action head. Slot capacity is fixed at
N
max
=
16
N_{\max}{=}16
with masked padding for unused positions; multi-instance noun phrases are disambiguated at
t
=
0
t{=}0
by the Qwen3-VL relation graph, and identity persists across frames via SAM 3 concept tracking (App.
B
).
3.3
Object-addressable attention
Figure 3:
OA attention mask.
Block-causal across frames; within-frame slots are bidirectional (red diagonal).
W
K
W_{K}
reads only
addr
k
\mathrm{addr}_{k}
(first 32 dims).
The
7
7
B trunk is a Chameleon-style multimodal autoregressive transformer (
32
32
layers, hidden dimension
4096
4096
,
32
32
attention heads). At slot-typed positions, the standard self-attention is replaced by a slot-aware variant in which the key-projection input is restricted to the address subvector (Fig.
3
; within-frame slots attend bidirectionally for permutation equivariance, block-causally across frames):
𝐊
k
(
ℓ
)
=
W
K
(
ℓ
)
⋅
mask
≤
32
​
(
𝐱
k
(
ℓ
)
)
,
𝐐
k
(
ℓ
)
=
W
Q
(
ℓ
)
𝐱
k
(
ℓ
)
,
𝐕
k
(
ℓ
)
=
W
V
(
ℓ
)
𝐱
k
(
ℓ
)
,
\begin{gathered}\mathbf{K}_{k}^{(\ell)}=W_{K}^{(\ell)}\!\cdot\!\mathrm{mask}_{\leq 32}\!\bigl(\mathbf{x}_{k}^{(\ell)}\bigr),\\
\mathbf{Q}_{k}^{(\ell)}=W_{Q}^{(\ell)}\mathbf{x}_{k}^{(\ell)},\quad\mathbf{V}_{k}^{(\ell)}=W_{V}^{(\ell)}\mathbf{x}_{k}^{(\ell)},\end{gathered}
(3)
mask
≤
32
\mathrm{mask}_{\leq 32}
zeros all coordinates beyond the first
32
32
. This is equivalent to projecting only
addr
k
\mathrm{addr}_{k}
through a
32
32
-dimensional slice of
W
K
W_{K}
, but is implemented as a pre-projection mask so that the pretrained
W
K
W_{K}
is reused without introducing OA-specific parameters; non-slot positions use the unmodified base attention (per-module accounting in App.
D
). After every transformer block, a forward hook overwrites
𝐱
k
(
ℓ
+
1
)
[
1
:
32
]
←
𝐚𝐝𝐝𝐫
k
\mathbf{x}_{k}^{(\ell+1)}[1{:}32]\!\leftarrow\!\mathbf{addr}_{k}
at slot positions while leaving the remaining
4064
4064
coordinates untouched, preventing address drift through residual updates. Combined with Eq. (
3
), this ensures by construction that the slot-routing key signal at every layer depends only on the frozen identity address:
conditional on correct slot extraction
, which slot the action head attends to is selected by
addr
\mathrm{addr}
and not by
𝐜𝐧𝐭
k
t
\mathbf{cnt}_{k}^{t}
, scene context, or non-slot tokens, while time-varying content still flows through values and the residual stream. This is an architectural property of key routing rather than a semantic guarantee of grounding correctness end-to-end—if SAM 3 misses an object or two slots are initialized with ambiguous addresses, the routing constraint cannot recover from upstream errors. Holistic VLA backbones
[
34
,
60
,
56
]
entangle identity, appearance, and surrounding context within a single patch-token stream; OA-WAM separates them at the tensor level.
3.4
Prediction heads
Three heads read from the final hidden states
𝐇
∈
ℝ
B
×
L
×
4096
\mathbf{H}\!\in\!\mathbb{R}^{B\times L\times 4096}
. The
world head
h
ψ
h_{\psi}
takes per-slot hiddens through two parallel MLPs—a content branch (
4096
→
1024
→
256
4096{\to}1024{\to}256
) and a pose branch (
4096
→
256
→
9
4096{\to}256{\to}9
)—and is supervised by mean squared error,
ℒ
world
=
1
N
​
∑
k
=
1
N
m
k
obj
​
(
‖
𝐜
^
k
t
+
1
−
𝐜
k
t
+
1
‖
2
2
+
λ
p
​
‖
𝐩
^
k
t
+
1
−
𝐩
k
t
+
1
‖
2
2
)
;
\mathcal{L}_{\mathrm{world}}=\frac{1}{N}\sum_{k=1}^{N}m_{k}^{\mathrm{obj}}\!\Bigl(\bigl\|\hat{\mathbf{c}}_{k}^{t+1}-\mathbf{c}_{k}^{t+1}\bigr\|_{2}^{2}+\lambda_{p}\bigl\|\hat{\mathbf{p}}_{k}^{t+1}-\mathbf{p}_{k}^{t+1}\bigr\|_{2}^{2}\Bigr);
(4)
the robot slot is excluded from this loss since its future is determined by the action under generation. The
action head
h
ξ
h_{\xi}
reads the
[act_q]
hidden and is a flow-matching MLP
[
44
,
3
]
that predicts a velocity field
𝐯
ξ
\mathbf{v}_{\xi}
on the full
16
16
-step action chunk; training minimizes the conditional flow-matching objective
ℒ
act
=
𝔼
τ
,
ϵ
​
‖
𝐯
ξ
​
(
𝐀
t
τ
,
τ
,
𝐇
act_q
)
−
(
𝐀
t
−
ϵ
)
‖
2
2
,
𝐀
t
τ
=
τ
​
𝐀
t
+
(
1
−
τ
)
​
ϵ
,
\mathcal{L}_{\mathrm{act}}=\mathbb{E}_{\tau,\boldsymbol{\epsilon}}\!\left\|\mathbf{v}_{\xi}\!\bigl(\mathbf{A}_{t}^{\tau},\tau,\mathbf{H}_{\textsc{act\_q}}\bigr)-(\mathbf{A}_{t}-\boldsymbol{\epsilon})\right\|_{2}^{2},\;\;\mathbf{A}_{t}^{\tau}=\tau\,\mathbf{A}_{t}+(1-\tau)\,\boldsymbol{\epsilon},
(5)
with
τ
∼
𝒰
⁡
(
0
,
1
)
\tau\sim\mathcal{U}(0,1)
and
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
. At inference, the chunk is decoded by
4
4
-step forward Euler integration in a single forward pass, avoiding the intra-chunk error accumulation of autoregressive action decoders
[
9
,
10
]
. An
auxiliary image-VQ head
reuses the trunk’s
lm
​
_
​
head
\mathrm{lm\_head}
to predict next-frame VQ tokens with a weighted cross-entropy
ℒ
vq
\mathcal{L}_{\mathrm{vq}}
and introduces no new parameters.
3.5
Training objective
The total training loss combines the three head terms with two auxiliary regularizers adapted from prior object-centric VLA work
[
26
,
2
]
:
ℒ
⁡
(
θ
)
=
ℒ
act
+
λ
w
​
ℒ
world
+
λ
v
​
ℒ
vq
+
λ
c
​
ℒ
compose
+
λ
r
​
ℒ
role
.
\mathcal{L}(\theta)\;=\;\mathcal{L}_{\mathrm{act}}\;+\;\lambda_{w}\,\mathcal{L}_{\mathrm{world}}\;+\;\lambda_{v}\,\mathcal{L}_{\mathrm{vq}}\;+\;\lambda_{c}\,\mathcal{L}_{\mathrm{compose}}\;+\;\lambda_{r}\,\mathcal{L}_{\mathrm{role}}.
(6)
ℒ
compose
\mathcal{L}_{\mathrm{compose}}
enforces invariance under random distractor permutation and insertion;
ℒ
role
\mathcal{L}_{\mathrm{role}}
aligns the action-head attention with language-extracted target/reference labels when available. The four weights
{
λ
w
,
λ
v
,
λ
c
,
λ
r
}
\{\lambda_{w},\lambda_{v},\lambda_{c},\lambda_{r}\}
are fixed (non-learnable) hyperparameters with values
{
0.5
,
0.04
,
0.1
,
0.05
}
\{0.5,\,0.04,\,0.1,\,0.05\}
;
λ
c
\lambda_{c}
is linearly warmed from
0
0
to its final value over the first
30
%
30\%
of training and
λ
r
\lambda_{r}
is annealed to
0
0
after the first half of training. The slot adapter and prediction heads are aligned and finetuned on standard LIBERO demonstrations atop a frozen
7
7
B slot-aware trunk, yielding
∼
127
\sim\!127
M trainable parameters (
80
80
M LoRA on
{
q
,
k
,
v
,
o
,
gate
,
up
,
down
}
​
_proj
\{q,k,v,o,\mathrm{gate},\mathrm{up},\mathrm{down}\}\text{\_proj}
and
47
47
M for the prediction heads); LIBERO-Plus is held out as out-of-distribution evaluation. Pseudocode appears in Appendix
A
; remaining implementation details—tokenization, slot-adapter and head architectures, trunk integration, sequence template, loss components and augmentations, training pipeline and hyperparameters, inference latency, and the equivariance proof—are in Appendices
B
–
G
.
