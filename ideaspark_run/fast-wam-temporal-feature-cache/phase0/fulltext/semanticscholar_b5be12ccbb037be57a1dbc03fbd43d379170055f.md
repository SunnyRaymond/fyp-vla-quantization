# Fast-WAM: Do World Action Models Need Test-time Future Imagination?

paper_id: semanticscholar:b5be12ccbb037be57a1dbc03fbd43d379170055f
tier: U
source_used: html_arxiv
warning: none

## Intro

Building general-purpose embodied agents requires policies that can not only map visual observations to actions, but also reason about how the physical world evolves under interaction. This has motivated growing interest in World Action Models (WAMs), which combine future visual prediction and action modeling in a unified framework. Compared with standard Vision-Language-Action (VLA) models, WAMs are appealing because modeling future observations may help capture physical dynamics and task-relevant temporal structure.
Most existing WAMs follow an imagine-then-execute paradigm: they first generate future observations, then predict actions conditioned on the imagined future. While intuitive, this design incurs substantial test-time latency due to iterative video denoising
[
1
,
2
,
3
,
4
,
5
]
. More fundamentally, it remains unclear whether explicit future imagination is actually necessary for strong action performance. The effectiveness of WAMs may stem from two distinct sources:
(1)
the video prediction objective during training, which may help the model acquire stronger physical priors and action-conditioned representations, and
(2)
explicit future generation during inference, which may provide additional foresight for action prediction. Existing WAM systems typically entangle these two factors, making it difficult to determine which one is actually responsible for the observed gains.
In this paper, we revisit this design choice and ask a simple question:
do WAMs need to imagine future observations at test time, or do they benefit primarily from learning to model them during training?
Our key idea is to decouple the video prediction objective used in WAM training from explicit future generation at inference time. If the main value of world modeling lies in shaping better latent representations during training, then a WAM should be able to retain this benefit without paying the test-time cost of future video synthesis.
Figure 1
:
Three representative WAM paradigms. (A) Joint-modeling WAMs denoise future video and action tokens together. (B) Causal WAMs first generate future observations and then condition action prediction on the generated future representation. (C) Fast-WAM retains video co-training during training but removes explicit future generation at inference time, directly predicting actions from latent world representations in a single forward pass.
Based on this perspective, we propose
Fast-WAM
, a WAM architecture that preserves video co-training during training but skips future prediction at test time. Instead of using a pretrained video generation model to iteratively synthesize future frames during inference, Fast-WAM repurposes a pretrained video Diffusion Transformer (DiT) as a single-pass world encoder for action generation. Concretely, we build Fast-WAM with a Mixture-of-Transformer (MoT) architecture with shared attention, consisting of a video DiT and an action expert DiT, as illustrated in Figure
1
(C). During training, the video prediction objective shapes the video DiT to encode physically meaningful motion and interaction structure. During inference, the video DiT processes the observation context in a single forward pass and provides latent world representations for action denoising, avoiding explicit future video denoising and enabling efficient real-time control.
To study our central question in a controlled way, we instantiate Fast-WAM into variants that mirror representative imagine-then-execute WAM designs. For simplicity, we focus on single action chunk generation and omit the outer auto-regressive loop. As shown in Figure
1
, existing WAMs can be broadly grouped into two representative paradigms:
(A)
future videos and actions are jointly denoised with shared attention
[
4
,
6
,
5
]
; and
(B)
actions are predicted after, and conditioned on, generated future videos
[
3
,
7
,
8
]
. We also implement a no-video-co-training variant, which serves as a direct control for the role of the training objective itself. Together, these controlled comparisons allow us to isolate the contribution of test-time future imagination from that of video co-training during training.
Experiments on simulation benchmarks (LIBERO and RoboTwin) show that Fast-WAM achieves strong results without any embodied pretraining, demonstrating strong data efficiency. On real-world robotic tasks, Fast-WAM remains highly effective while running at only 190 ms latency, making it more than 4
×
\times
faster than existing imagine-then-execute WAM approaches. More importantly, controlled comparisons show that Fast-WAM stays close to imagine-then-execute variants, while removing the video co-training objective causes a much larger performance drop. These results suggest that the main value of video prediction in WAMs may lie in improving world representations during training rather than in explicitly generating future observations at test time.
Our contributions are three-fold:
•
We identify and study a basic question in WAMs: whether their gains come primarily from video modeling during training or from explicit future imagination during inference.
•
We propose Fast-WAM, a WAM architecture that retains video co-training during training while eliminating future prediction at test time, enabling real-time inference.
•
Through controlled comparisons on simulation and real-world benchmarks, including variants with and without video co-training, we show that much of the benefit of WAMs comes from the video co-training objective itself, while explicit future generation at inference time appears to be less critical than previously assumed.

## Method

3.1
Problem Formulation
We consider embodied policy learning from visual observations and language instructions. Let
o
o
denote the current observation,
l
l
denote the task instruction, and
a
1
:
H
a_{1:H}
denote an action chunk of horizon
H
H
. A standard visuomotor policy models the conditional distribution
p
(
a
1
:
H
∣
o
,
l
)
,
p(a_{1:H}\mid o,l),
(1)
which directly maps the current perceptual context to a sequence of actions. World Action Models (WAMs) augment this formulation by introducing future visual observations as an intermediate variable. Let
v
1
:
T
v_{1:T}
denote future visual observations over a prediction horizon
T
T
. Many existing WAMs follow an
imagine-then-execute
factorization:
p
(
a
1
:
H
∣
o
,
l
)
=
∫
p
(
v
1
:
T
∣
o
,
l
)
p
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
v
1
:
T
)
d
v
1
:
T
,
p(a_{1:H}\mid o,l)=\int p(v_{1:T}\mid o,l)\,p(a_{1:H}\mid o,l,v_{1:T})\,dv_{1:T},
(2)
where the model first predicts future observations and then conditions action generation on the imagined future. In practice, this is typically implemented either by jointly denoising future video and actions within a shared model, or by first generating future video and then feeding it to an inverse dynamics or action prediction module. Some prior WAMs further wrap this formulation in an outer auto-regressive rollout, which we omit here for simplicity and controlled comparison.
Our starting point is the observation that the effectiveness of WAMs may arise from two distinct factors:
(i)
the video prediction objective used during training, which can encourage the model to learn physically meaningful latent representations, and
(ii)
explicit future generation during inference, which may provide additional foresight for action prediction. Existing WAM formulations usually couple these two factors, since the same model both learns from future video prediction and explicitly synthesizes future observations at test time. We design Fast-WAM to decouple these two factors. During training, it retains world modeling as a co-training signal; during inference, however, it does not explicitly generate future observations. Instead, Fast-WAM predicts actions directly from the current observation and instruction,
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
)
,
p_{\theta}(a_{1:H}\mid o,l),
(3)
while using latent world representations shaped by video co-training. In this sense, Fast-WAM has a direct-policy interface at test time, similar to standard VLA policies, while its representation learning remains grounded in WAM-style video modeling during training. Formally, let
z
⁡
(
o
,
l
)
z(o,l)
denote the latent world representation produced by the video backbone conditioned on the current context. Fast-WAM uses this representation to parameterize the action distribution,
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
)
=
p
θ
(
a
1
:
H
∣
z
(
o
,
l
)
)
.
p_{\theta}(a_{1:H}\mid o,l)=p_{\theta}(a_{1:H}\mid z(o,l)).
(4)
The key difference from imagine-then-execute WAMs is that
z
⁡
(
o
,
l
)
z(o,l)
is obtained by a single forward encoding pass, rather than by explicitly sampling or denoising future observations
v
1
:
T
v_{1:T}
at inference time.
3.2
Model Architecture
Overview.
Fast-WAM is designed to preserve the training benefits of world modeling while removing the inference cost of explicit future imagination. During training, it jointly learns action prediction and video modeling, encouraging the visual backbone to capture physically meaningful motion and interaction structure. During inference, Fast-WAM does not explicitly generate future observations. Instead, it keeps only the clean latent tokens of the first observation frame, processes them with the video model in a single forward pass, and uses the resulting latent world representation for direct action generation. This gives Fast-WAM a direct-policy interface at test time while retaining WAM-style video supervision during training.
Architecture.
Fast-WAM is built on top of the video Diffusion Transformer (DiT) from Wan2.2-5B
[
36
]
, which serves as the world modeling backbone. We also reuse its pretrained text encoder and video VAE: task language is encoded by the built-in T5 encoder and provided to all tokens through cross-attention, while visual observations are mapped into latent video tokens by the pretrained VAE. On top of this backbone, we introduce an action expert DiT for action chunk generation. The full model is organized as a Mixture-of-Transformer (MoT) architecture with shared attention between the video and action branches, as illustrated in Figure
2(a)
.
We organize the input tokens into three groups: clean latent tokens of the first observation frame, which serve as the shared visual anchor; noisy latent tokens of future video frames, which are used only during training for video modeling; and action tokens, which are processed by the action expert for action generation. All token groups attend to the language embeddings through cross-attention. A structured attention mask controls the information flow between these groups. During training, future noisy video tokens attend bidirectionally within the video branch and can access the clean first-frame tokens; action tokens attend bidirectionally within the action branch and can also access the clean first-frame tokens. Crucially, action tokens cannot attend to future video tokens, and the clean first-frame tokens do not attend to any other tokens. This ensures that both video modeling and action prediction are grounded in the same visual context while preventing future information from leaking into the action branch. We provide the full training and inference masks in Figure
2(b)
.
At inference time, Fast-WAM removes the future video branch entirely: only the clean first-frame latent tokens are retained and passed through the video backbone once to produce latent world features for the action expert. Since no future noisy video tokens are instantiated and no explicit future video denoising is performed, Fast-WAM incurs substantially lower inference cost than standard imagine-then-execute WAMs.
(a)
Fast-WAM model architecture.
(b)
Training and inference masks.
Figure 2
:
Fast-WAM architecture and the structured attention mask used to disentangle video co-training from action generation.
Training objective.
Fast-WAM is trained with a joint flow matching objective over action tokens and future video latents. Given a target variable
y
y
(either an action chunk
a
1
:
H
a_{1:H}
or future video latents
z
1
:
T
z_{1:T}
), we sample Gaussian noise
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
and a time step
t
∈
(
0
,
1
)
t\in(0,1)
, and construct the interpolated sample
y
t
=
(
1
−
t
)
​
y
+
t
​
ϵ
.
y_{t}=(1-t)\,y+t\,\epsilon.
(5)
The model is trained to predict the corresponding velocity field with a standard flow matching objective
ℒ
FM
​
(
y
)
=
𝔼
y
,
ϵ
,
t
​
[
‖
f
θ
​
(
y
t
,
t
,
o
,
l
)
−
(
ϵ
−
y
)
‖
2
2
]
.
\mathcal{L}_{\mathrm{FM}}(y)=\mathbb{E}_{y,\,\epsilon,\,t}\left[\left\|f_{\theta}(y_{t},\,t,\,o,\,l)-(\epsilon-y)\right\|_{2}^{2}\right].
(6)
We instantiate this objective for both action generation and video co-training. For action prediction, we set
y
=
a
1
:
H
y=a_{1:H}
and optimize
ℒ
act
=
ℒ
FM
(
a
1
:
H
)
.
\mathcal{L}_{\mathrm{act}}=\mathcal{L}_{\mathrm{FM}}(a_{1:H}).
(7)
For video co-training, we set
y
=
z
1
:
T
y=z_{1:T}
, where
z
1
:
T
z_{1:T}
denotes the latent tokens of future video frames produced by the pretrained VAE, and optimize
ℒ
vid
=
ℒ
FM
(
z
1
:
T
)
.
\mathcal{L}_{\mathrm{vid}}=\mathcal{L}_{\mathrm{FM}}(z_{1:T}).
(8)
The overall training objective is
ℒ
=
ℒ
act
+
λ
​
ℒ
vid
,
\mathcal{L}=\mathcal{L}_{\mathrm{act}}+\lambda\mathcal{L}_{\mathrm{vid}},
(9)
where
λ
\lambda
balances action learning and video co-training.
3.3
Controlled Variants for Disentangled WAM Design
To answer our central question, namely whether the benefit of WAMs comes primarily from video co-training during training or from explicit future imagination during inference, we design a set of controlled variants under a shared implementation framework. We instantiate representative imagine-then-execute design patterns from recent WAMs while keeping the backbone, tokenization, and training recipe as aligned as possible. This controlled setup allows us to isolate the contribution of test-time future generation from that of the video co-training objective itself.
As illustrated in Figure
1
(A) and (B), we consider two representative imagine-then-execute variants that capture the dominant design patterns in recent WAMs. The first variant, named Fast-WAM-Joint, follows the joint-generation paradigm, where future video tokens and action tokens are denoised together within a shared model, so that action generation remains coupled with future video modeling throughout the denoising process
[
4
,
6
,
5
]
. The second variant, named Fast-WAM-IDM, follows the video-then-action paradigm, where future video tokens are generated first from the current observation and language context, and action prediction is then conditioned on the resulting future representation
[
3
,
7
,
8
]
. In both cases, we preserve the defining inference structure of the corresponding paradigm, together with key training choices used in recent WAMs, while implementing them within our shared framework for controlled comparison with Fast-WAM.
We further construct a Fast-WAM variant without video co-training. This variant keeps the architecture and inference procedure unchanged, and removes only the video modeling objective during training. It therefore serves as a direct control for the role of video co-training itself. Together, these controlled variants isolate the two factors that are typically entangled in prior WAMs: video co-training during training and explicit future imagination during inference.
