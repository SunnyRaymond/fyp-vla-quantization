# DELE-w0.5: Inferring Action from Future Latent State for Robotic Manipulation

paper_id: arxiv:2608.22067v4
tier: T3
source_used: html_arxiv
warning: none

## Intro

Embodied intelligence has achieved rapid development and achieved strong performance for robotic manipulation
[
17
,
19
]
.
Vision-Language-Action (VLA) and World Action Model (WAM) are two popular model architectures.
VLA is a three-step framework, see Figure
1
(a): First, the input comprises multimodal data, including human task instructions and the robot’s current observations;
Then, a vision-language model (VLM) backbone performs the task planning according to the input data;
Finally, an action expert decodes the task planning to the commands that are executable by the robot.
For some classic work of VLA, refer to
[
4
,
55
,
15
,
12
]
.
Since the information density of language is lower than that of vision, VLA models require a large amount of data to adapt to the downstream tasks.
As a result, generalization of VLA is poor, when light, color, or the task changes, the success rate decays rapidly, extensive empirical results have shown this view, see
[
46
,
14
]
.
WAM
[
10
,
46
,
14
]
fundamentally discards the framework of VLA, WAM’s core lying in the construction of a unified end-to-end learning objective, which integrates the prediction of world state evolution and action generation, see Figure
1
(b). The critical technology of WAM relies on self-supervised future state prediction on large-scale video data,
then WAM develops the action generation as a conditional denoising process according to the future state prediction.
1.1
Limitation of WAM
A dominant line of recent World-Action Models (WAMs) builds robot control on top of video generation and jointly predicts future visual trajectories and actions
[
46
,
47
]
,
which brings some useful spatiotemporal priors into the policy. However, it also creates a structural mismatch between the objective of visual generation and the objective of robot control. Future videos are high-dimensional and appearance-sensitive. Most intermediate frames describe how the scene looks during a transition, rather than the physical consequence that determines the subsequent action. Therefore, the model is required to solve a substantially harder problem than control itself. It must first reconstruct dense visual evolution and then extract a compact action signal from it. Additionally, multi-frame video latents are much larger than the corresponding robot action sequence
[
50
]
.
They dominate the sequence length, memory consumption, and computational budget of joint world-action learning. Consequently, WAM training inherits the heavy cost of large video generators. Recent methods introduce compact backbones, latent downsampling, or lightweight adaptation to make video-action co-training more practical
[
22
]
. These techniques reduce the cost of the existing pipeline, but do not remove its underlying redundancy.
The same limitation becomes more critical during inference. Before an action can be executed, the model must perform iterative denoising over high-dimensional future video-action latents
[
1
]
. The control loop is therefore bottlenecked by synthesizing a visual trajectory that is not itself executed by the robot
[
18
]
.
Therefore, the fundamental limitation of video-generation-based WAMs is not merely their computational cost. The deeper problem is that they optimize visual trajectory reconstruction as a surrogate for action-relevant world prediction. For robot control, a world model should predict how the world will become after an interaction, rather than reproduce how the world looks at every intermediate moment. This motivates a more direct formulation that models action-relevant future states and infers robot actions without relying on video generation in the control loop.
Action
Action Expert
VLM
Language
Vision
(a) VLA
Video
Action
Transformer
Causal DiT
Vision
Language
(b) WAM
Future State
Action
Projection
Projection
Causal Attention
Vision
Language
(c) DELE-w0.5
Figure 1
:
Different Models for Manipulation.
1.2
Our Work
We reformulate the current WAM of visual world modeling for robotic manipulation.
It is different from recent video generation techinique based WAMs that model the future world through synthesizing dense visual trajectories, the proposed
𝙳𝙴𝙻𝙴
\mathtt{DELE}
-
𝚠𝟶
​
.5
\mathtt{w0.5}
argues that robot control does not require reconstructing the entire visual evolution of the real world, but only requires to predict the future latent state with downstream actions. With this observation, we formulate WAM as prediction problem of a future latent state and action, where the
𝙳𝙴𝙻𝙴
\mathtt{DELE}
-
𝚠𝟶
​
.5
\mathtt{w0.5}
directly infers the future visual state and corresponding robot behaviors, without any generating intermediate video frames.
Rethinking World Model for Robotic Manipulation
.
We argue that existing video-generation-based WAMs formulate unnecessary intermediate objectives for robotic manipulation.
Although predicting future videos provides rich spatiotemporal priors, it implicitly treats visual evolution as the primary target of world modeling.
However, for robot robotic manipulation, a world model should not reproduce how the world looks at every intermediate moment.
Instead, it should predict how the world will become after an action is executed.
Most intermediate frames in generated videos only describe the visual transition between states, which provides limited information about the final physical consequence that determines subsequent actions.
Therefore, video-generation-based WAMs force the model to learn the distribution of visual evolution rather than directly modeling action-relevant future states.
This design introduces substantial computational overhead without necessarily improving control capability.
We argue that future-state prediction with causal relevance to robot actions is a more fundamental objective for embodied world models than reconstructing complete visual trajectories.
Inferring Action from Future-State
.
With this observation, we propose
𝙳𝙴𝙻𝙴
\mathtt{DELE}
-
𝚠𝟶
​
.5
\mathtt{w0.5}
that infers robot actions from predicted future states rather than generating future visual trajectories.
Different from existing video-generation-based approaches that model the continuous evolution of the visual world,
𝙳𝙴𝙻𝙴
\mathtt{DELE}
-
𝚠𝟶
​
.5
\mathtt{w0.5}
treats future prediction as a state transition problem and focuses on the physical states that are causally relevant to subsequent actions.
The key insight is that a world model for robotic manipulation does not need to reproduce how the world visually evolves at every intermediate moment, but should capture how the world will become after an action is executed.
By directly modeling action-relevant future states,
𝙳𝙴𝙻𝙴
\mathtt{DELE}
-
𝚠𝟶
​
.5
\mathtt{w0.5}
avoids the high-dimensional visual redundancy introduced by video representations, and eliminates the costly iterative generation process required by video-based WAMs.
This formulation provides a more compact and efficient paradigm for world-action modeling, enabling significantly cheaper training and low-latency inference for real-world robotic deployment. Finally, we comparse the three formulation from probabilistic inference as follows,
•
VLA infers
p
θ
​
(
\bA
t
|
\bO
t
,
\bq
t
,
\bL
)
p_{\theta}(\bA_{t}|\bO_{t},\bq_{t},\bL)
;
•
WAM infers
p
θ
(
\bA
t
,
\bO
t
:
t
+
Δ
​
t
|
\bO
t
,
\bq
t
,
\bL
)
p_{\theta}(\bA_{t},\bO_{t:t+\Delta t}|\bO_{t},\bq_{t},\bL)
;
•
𝙳𝙴𝙻𝙴
\mathtt{DELE}
-
𝚠𝟶
​
.5
\mathtt{w0.5}
infers
p
θ
(
\bA
t
,
\bO
t
+
Δ
​
t
|
\bO
t
,
\bq
t
,
\bL
)
p_{\theta}(\bA_{t},\bO_{t+\Delta t}|\bO_{t},\bq_{t},\bL)
,
where
\bO
t
:
t
+
Δ
​
t
\bO_{t:t+\Delta t}
is the video from current observation
\bO
t
\bO_{t}
to the predicted future state
\bO
t
+
Δ
​
t
\bO_{t+\Delta t}
.

## Method

The framwork of
𝙳𝙴𝙻𝙴
\mathtt{DELE}
-
𝚠𝟶
​
.5
\mathtt{w0.5}
adopts a dual-stream attention mechanism architecture, which contains two blocks: condition stream and noise stream.
4.1
Condition Stream Block
4.1.1
Input and Tokenization
For each time
t
t
, we encode the language instruction input
\bL
\bL
and current state input
\bO
t
\bO_{t}
according to the methods (
1
) and (
2
) correspondingly,
and obtain the language token
\bl
^
\hat{\bl}
, latent space of vision
\bz
^
t
\hat{\bz}_{t}
.
Since the output dimensions of the vision encoder and the language encoder are not necessarily the same,
then we apply the projection operator to ensure that feature information
\bl
\bl
and
\bz
t
\bz_{t}
have the same hidden dimension,
\bl
=
𝙿𝚛𝚘𝚓𝚎𝚌𝚝𝚒𝚘𝚗
⁡
(
\bl
^
)
,
\bz
t
=
𝙿𝚛𝚘𝚓𝚎𝚌𝚝𝚒𝚘𝚗
⁡
(
\bz
^
t
)
,
\displaystyle\bl=\mathtt{Projection}(\hat{\bl}),~~\bz_{t}=\mathtt{Projection}(\hat{\bz}_{t}),
(6)
where in this paper, we set hidden dimension with the value of
1024
1024
, the
𝙿𝚛𝚘𝚓𝚎𝚌𝚝𝚒𝚘𝚗
⁡
(
⋅
)
\mathtt{Projection}(\cdot)
operator we used is the linear neural networks.
4.1.2
Single-Stream Attention Block
To integrate the information of language instruction with current observations, we employ a single-stream attention block
[
20
]
that concatenates the multimodal tokens (i.e.,
\bl
\bl
and
\bz
t
\bz_{t}
), and feeds them into a shared Transformer encoder, where the attention block enables unified, source-agnostic cross-modal interactions. Concretely, firstly, we concatenate the language and vision tokens as follows,
\bZ
t
c
=
𝚌𝚊𝚝
⁡
(
\bl
,
\bz
t
)
=
:
[
\bl
,
\bz
t
]
.
\displaystyle\bZ^{\mathrm{c}}_{t}=\mathtt{cat}(\bl,\bz_{t})=:[\bl,\bz_{t}].
(7)
Then we apply the standard self-attention block
[
36
]
to refine the concatenated information as follows,
\bZ
^
t
c
=
𝙰𝚝𝚝𝚎𝚗𝚝𝚒𝚘𝚗
⁡
(
\bZ
t
c
)
,
\displaystyle\hat{\bZ}^{\mathrm{c}}_{t}=\mathtt{Attention}(\bZ^{\mathrm{c}}_{t}),
(8)
which exploites the complementary information gain from language and vision,
and shows the generalization robustness in embodied artificial intelligence.
4.1.3
AdaLN and Output of Condition Stream
Furthermore,
𝙳𝙴𝙻𝙴
\mathtt{DELE}
-
𝚠𝟶
​
.5
\mathtt{w0.5}
applies the AdaLN facilitates the integration of conditioning information into the feature representation, which resides the dynamic learning of intermediate representations in response to the incoming conditioning signal, and which ia a critical prerequisite for robots to attain fine-grained control.
After the single-stream attention block, with the fused information
\bZ
^
t
c
\hat{\bZ}^{\mathrm{c}}_{t}
, we obtain the query, keys, values as the output of condition stream as follows,
\bQ
c
=
\displaystyle\bQ_{\mathrm{c}}=
QKNorm
⁡
{
\bW
q
​
(
RMSNorm
⁡
(
\bZ
^
t
c
)
⊙
𝜸
c
)
}
,
\displaystyle\mathrm{QKNorm}\left\{\bW_{q}(\mathrm{RMSNorm}(\hat{\bZ}^{\mathrm{c}}_{t})\odot\bm{\gamma}_{\mathrm{c}})\right\},
(9)
\bK
c
=
\displaystyle\bK_{\mathrm{c}}=
QKNorm
⁡
{
\bW
k
​
(
RMSNorm
⁡
(
\bZ
^
t
c
)
⊙
𝜸
c
)
}
,
\displaystyle\mathrm{QKNorm}\left\{\bW_{k}(\mathrm{RMSNorm}(\hat{\bZ}^{\mathrm{c}}_{t})\odot\bm{\gamma}_{\mathrm{c}})\right\},
(10)
\bV
c
=
\displaystyle\bV_{\mathrm{c}}=
\bW
v
​
(
RMSNorm
⁡
(
\bZ
^
t
c
)
⊙
𝜸
c
)
,
\displaystyle\bW_{v}(\mathrm{RMSNorm}(\hat{\bZ}^{\mathrm{c}}_{t})\odot\bm{\gamma}_{\mathrm{c}}),
(11)
where
RMSNorm
⁡
(
⋅
)
\mathrm{RMSNorm}(\cdot)
is root mean square normalization
[
48
]
;
𝜸
c
\bm{\gamma}_{\mathrm{c}}
is the scaling factor for the condition stream with following linear structure,
𝜸
c
=
1
+
\bW
Adaln
​
𝐭
c
+
𝐛
Adaln
,
\displaystyle\bm{\gamma}_{\mathrm{c}}=1+\bW_{\mathrm{Adaln}}\mathbf{t}_{\mathrm{c}}+\mathbf{b}_{\mathrm{Adaln}},
(12)
\bW
Adaln
\bW_{\mathrm{Adaln}}
and
𝐛
Adaln
\mathbf{b}_{\mathrm{Adaln}}
are the learnable weights,
𝐭
c
\mathbf{t}_{\mathrm{c}}
is the timestep
τ
=
1
\tau=1
for condition stream, i.e.,
𝐭
c
​
=
(
5
)
​
𝚃𝚒𝚖𝚎𝚜𝚝𝚎𝚙
​
𝙴𝚖𝚋𝚎𝚍𝚍𝚒𝚗𝚐
​
(
1
)
;
\mathbf{t}_{\mathrm{c}}\overset{\eqref{timestep-embedding}}{=}\mathtt{Timestep~Embedding}(1);
QKNorm
⁡
(
⋅
)
\mathrm{QKNorm}(\cdot)
is a normalization operator maps any vector
𝐱
\mathbf{x}
as follows,
𝐱
′
=
QKNorm
⁡
(
𝐱
)
=
𝐱
/
‖
𝐱
‖
2
;
\displaystyle\mathbf{x}^{\prime}=\mathrm{QKNorm}(\mathbf{x})={\mathbf{x}}/{\|\mathbf{x}\|_{2}};
(13)
and
QKNorm
⁡
(
⋅
)
\mathrm{QKNorm}(\cdot)
applies
ℓ
2
\ell_{2}
normalization along the head dimension of each query and key matrix prior to multiplying them.
4.2
Noise Stream Block
For each time
t
t
, we need to add noise to the clear data
\bA
t
\bA_{t}
and next state
\bO
t
+
Δ
​
t
\bO_{t+\Delta t}
.
Let
τ
∼
𝒰
[
0
,
1
]
\tau\sim\mathcal{U}_{[0,1]}
,
\bepsilon
∼
\calN
​
(
𝟎
,
\bI
)
\bepsilon\sim\calN(\bm{0},\bI)
, where
𝒰
[
0
,
1
]
\mathcal{U}_{[0,1]}
is the uniform distribution on
[
0
,
1
]
[0,1]
,
we obtain noise action as follows,
\bA
t
τ
=
τ
​
\bA
t
+
(
1
−
τ
)
​
\bepsilon
.
\displaystyle\bA^{\tau}_{t}=\tau\bA_{t}+(1-\tau)\bepsilon.
(14)
The noise addition of image is on the latent space:
\bz
^
t
+
Δ
​
t
=
\displaystyle\hat{\bz}_{t+\Delta t}=
𝚅𝚒𝚜𝚒𝚘𝚗
​
𝙴𝚗𝚌𝚘𝚍𝚎𝚛
​
(
\bO
t
+
Δ
​
t
)
,
\displaystyle\mathtt{Vision~Encoder}(\bO_{t+\Delta t}),
(15)
then we obtain the noise image as follows,
\bz
t
+
Δ
​
t
τ
=
τ
​
\bz
^
t
+
Δ
​
t
+
(
1
−
τ
)
​
\bepsilon
.
\displaystyle\bz^{\tau}_{t+\Delta t}=\tau\hat{\bz}_{t+\Delta t}+(1-\tau)\bepsilon.
(16)
Furthermore, following the same steps from (
7
), (
8
), (
9
), (
10
) and (
11
), we obtain the query, keys, values as the output of noise stream, denoted them as
\bQ
n
\bQ_{\mathrm{n}}
,
\bK
n
\bK_{\mathrm{n}}
and
\bV
n
\bV_{\mathrm{n}}
.
4.3
Flow Matching Objective
We concatenate the information from condition stream and noise stream as follows,
\bQ
=
𝚌𝚊𝚝
⁡
(
\bQ
c
,
\bQ
n
)
,
\bK
=
𝚌𝚊𝚝
⁡
(
\bK
c
,
\bK
n
)
,
\bV
=
𝚌𝚊𝚝
⁡
(
\bV
c
,
\bV
n
)
.
\displaystyle\bQ=\mathtt{cat}(\bQ_{\mathrm{c}},\bQ_{\mathrm{n}}),~~~\bK=\mathtt{cat}(\bK_{\mathrm{c}},\bK_{\mathrm{n}}),~~~\bV=\mathtt{cat}(\bV_{\mathrm{c}},\bV_{\mathrm{n}}).
(17)
With the joint attention block and projection layer, we obtain the predicted action chunk
\bA
^
t
\hat{\bA}_{t}
, and predicted next latent observations
\bz
^
t
+
Δ
​
t
\hat{\bz}_{t+\Delta t}
as follows,
\bA
^
t
τ
,
\bz
^
t
+
Δ
​
t
τ
=
𝙿𝚛𝚎𝚓𝚎𝚌𝚝𝚒𝚘𝚗
⁡
(
𝙼𝚊𝚜𝚔𝚎𝚍
​
𝙰𝚝𝚝𝚎𝚗𝚝𝚒𝚘𝚗
​
(
\bQ
,
\bK
,
\bV
)
)
.
\displaystyle\hat{\bA}^{\tau}_{t},~\hat{\bz}^{\tau}_{t+\Delta t}=\mathtt{Prejection}(\mathtt{Masked~Attention}(\bQ,\bK,\bV)).
(18)
Finally, we regress the velocity field according the following way,
ℒ
a
=
\displaystyle\mathcal{L}_{\mathrm{a}}=
𝔼
τ
∼
𝒰
[
0
,
1
]
,
\bepsilon
∼
\calN
​
(
𝟎
,
\bI
)
​
[
‖
𝐮
a
​
(
\bA
t
τ
,
\bz
t
+
Δ
​
t
τ
,
\bq
t
,
𝐥
)
−
(
\bA
t
−
\bepsilon
)
‖
2
2
]
,
\displaystyle\mathbb{E}_{\tau\sim\mathcal{U}_{[0,1]},\bepsilon\sim\calN(\bm{0},\bI)}\left[\left\|\mathbf{u}_{\mathrm{a}}(\bA^{\tau}_{t},\bz^{\tau}_{t+\Delta t},\bq_{t},\mathbf{l})-(\bA_{t}-\bepsilon)\right\|_{2}^{2}\right],
(19)
ℒ
o
=
\displaystyle\mathcal{L}_{\mathrm{o}}=
𝔼
τ
∼
𝒰
[
0
,
1
]
,
\bepsilon
∼
\calN
​
(
𝟎
,
\bI
)
​
[
‖
𝐨
a
​
(
\bA
t
τ
,
\bz
t
+
Δ
​
t
τ
,
\bq
t
,
𝐥
)
−
(
\bz
^
t
+
Δ
​
t
−
\bepsilon
)
‖
2
2
]
.
\displaystyle\mathbb{E}_{\tau\sim\mathcal{U}_{[0,1]},\bepsilon\sim\calN(\bm{0},\bI)}\left[\left\|\mathbf{o}_{\mathrm{a}}(\bA^{\tau}_{t},\bz^{\tau}_{t+\Delta t},\bq_{t},\mathbf{l})-(\hat{\bz}_{t+\Delta t}-\bepsilon)\right\|_{2}^{2}\right].
(20)
The overall training objective is
ℒ
=
ℒ
a
+
λ
​
ℒ
o
,
\displaystyle\mathcal{L}=\mathcal{L}_{\mathrm{a}}+\lambda\mathcal{L}_{\mathrm{o}},
(21)
where
λ
\lambda
balances the action learning the next state prediction.
Training
𝐋
\mathbf{L}
𝐎
t
\mathbf{O}_{t}
𝐀
t
\mathbf{A}_{t}
𝐎
t
+
Δ
​
t
\mathbf{O}_{t+\Delta t}
𝐋
\mathbf{L}
𝐎
t
\mathbf{O}_{t}
𝐀
t
\mathbf{A}_{t}
𝐎
t
+
Δ
​
t
\mathbf{O}_{t+\Delta t}
Inference
𝐋
\mathbf{L}
𝐎
t
\mathbf{O}_{t}
𝐀
t
\mathbf{A}_{t}
𝐋
\mathbf{L}
𝐎
t
\mathbf{O}_{t}
𝐀
t
\mathbf{A}_{t}
Figure 3
:
Attention masks for training and inference.
Rows are query-token groups and columns are key/value-token groups.
𝐋
\mathbf{L}
,
𝐎
t
\mathbf{O}_{t}
,
𝐀
t
\mathbf{A}_{t}
, and
𝐎
t
+
Δ
​
t
\mathbf{O}_{t+\Delta t}
denote language, current observation, action, and predicted future observation, respectively.
During training (left),
𝐋
\mathbf{L}
and
𝐎
t
\mathbf{O}_{t}
attend bidirectionally;
𝐀
t
\mathbf{A}_{t}
attends to
{
𝐋
,
𝐎
t
,
𝐀
t
}
\{\mathbf{L},\mathbf{O}_{t},\mathbf{A}_{t}\}
; and
𝐎
t
+
Δ
​
t
\mathbf{O}_{t+\Delta t}
attends to all groups.
During inference (right),
𝐎
t
+
Δ
​
t
\mathbf{O}_{t+\Delta t}
is omitted.
Colored and white cells indicate permitted and masked attention, respectively.
Multi-view tokens are merged into each observation group, and padded language positions are masked along both dimensions.
4.4
Training and Inference
During training, the input tokens are grouped as language, current multi-view observation, action, and future observation.
The language and current-observation tokens form the conditioning group and attend bidirectionally to one another.
They cannot attend to either the action tokens or the predicted future-observation tokens, which prevents information from the prediction targets from leaking into the conditioning representation.
The action tokens attend to the entire conditioning group and to the action group itself, but they cannot access the predicted future observations.
Therefore, robot actions are learned only from the language instruction, the current multi-view observations, and the dependencies within the action sequence.
In contrast, the predicted future-observation tokens attend to all token groups, allowing the model to learn the physical state transition conditioned on the instruction, current observations, and robot actions.
During inference, the two predicted future-observation groups are removed, while the visibility rules for the conditioning and action groups remain unchanged.
The model therefore predicts the action sequence without generating any future visual tokens, which shortens the inference sequence, removes unnecessary visual generation, and enables efficient low-latency robot control.
