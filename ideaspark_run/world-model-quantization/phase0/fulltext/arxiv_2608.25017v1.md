# Rollout-Decoded Reconstruction for Long-Horizon Prediction in Latent World Models

paper_id: arxiv:2608.25017v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

A latent world model predicts by free-running: an encoder compresses the last observation into a latent state, a transition iterates that latent forward on its own outputs, and a decoder turns each rollout latent into a predicted observation
[
9
,
8
,
10
,
11
]
. The decoder, however, is trained only on latents anchored to observations: encoder outputs, and predictions a single teacher-forced step from them. Nothing in the standard objective ties decoding quality to the drifted latents a long rollout visits, so a decoder that is sharp near observations can fail many steps into the rollout it exists to decode. PlaNet
[
9
]
named the direct fix, decoding the rollout during training, and set it aside; no latent world model since has made it the object of study (Section
2
).
We propose Rollout-Decoded Reconstruction (RDR), a loss term that closes this gap directly. During training, the model is rolled free-running from an encoded initial state, exactly as evaluation will roll it; every rollout latent is decoded; and the reconstruction error against ground truth is penalized. The term adds no parameters, changes no architecture, and reduces to the standard objective when its weight
λ
\lambda
is zero. Every preregistered comparison is therefore an A/B at fixed data, seeds, budget, and parameter count, differing in one flag; the added cost is training-time compute (Appendix
C
).
On the Kuramoto–Sivashinsky (KS) equation
[
14
,
20
]
, a chaotic PDE with exact ground truth and a standard long-horizon metric, RDR raises valid prediction time (VPT, the time to first crossing of normalized error 0.5) from
3.87
±
0.23
3.87\pm 0.23
to
6.97
±
0.42
6.97\pm 0.42
time units: a 1.80
×
\times
improvement at an identical 193,568 parameters, confirmed on fresh seeds and at both evaluation horizons, with the direction holding in 10 of 10 preregistered configurations at ratios of 1.71–2.50
×
\times
(Section
4.2
). The effect is consistent with the training-distribution account: a variant that spends +25.7% more parameters on the same objective performs worse in 7 of 10 configurations, which rules out added capacity as the explanation, and both arms’ rollout latents drift far off the posterior distribution while RDR decodes at matched distance with lower error (Section
5
). A latent-width sweep suggests the advantage grows as the latent widens while the standard objective loses horizon; the trend is descriptive (Section
5.3
). Preliminary control experiments show RDR more robust to a planner–training rollout mismatch, and a fixed-epoch margin that an optimizer-step-matched control mostly removes (Section
6
). At matched budget an observation-space predictor reaches the same horizon on this fully observed system; the comparison bounds the latent bottleneck itself, and the RDR contrast is within-latent throughout (Section
4.3
).
RDR is a training objective. Symmetry reduction, discrete latents, KL balancing, and latent overshooting all leave a decoder in place with its training distribution unchanged, so RDR applies on top of each; whether the improvement transfers to those settings is untested. Every quantitative claim traces to archived evaluation artifacts produced under a preregistered protocol, and each figure is rendered by a script that re-asserts every quoted number against those artifacts.

## Method

3.1  Model and base objective
The model is a deliberately plain encoder–transition–decoder triplet. An MLP encoder
E
ϕ
E_{\phi}
maps a field snapshot
u
t
∈
ℝ
64
u_{t}\in\mathbb{R}^{64}
to a latent
z
t
∈
ℝ
d
z_{t}\in\mathbb{R}^{d}
(two hidden layers of 256, GELU). A state-space predictor
f
θ
f_{\theta}
, a four-layer S5 stack
[
21
]
with state size 64, advances the latent one step. A decoder
D
ψ
D_{\psi}
, an MLP with one hidden layer of 512, maps latents back to field space. A target encoder
E
¯
\bar{E}
, the exponential moving average of
E
ϕ
E_{\phi}
with decay 0.999, provides latent regression targets; we write
z
¯
t
=
E
¯
​
(
u
t
)
\bar{z}_{t}=\bar{E}(u_{t})
and use
posterior
, by analogy with RSSM practice, for latents computed from observed data.
Four standard terms train the triplet. With
sg
\mathrm{sg}
the stop-gradient and
z
^
t
+
k
\hat{z}_{t+k}
the free-running rollout (
z
^
t
=
E
ϕ
​
(
u
t
)
\hat{z}_{t}=E_{\phi}(u_{t})
,
z
^
t
+
k
+
1
=
f
θ
​
(
z
^
t
+
k
)
\hat{z}_{t+k+1}=f_{\theta}(\hat{z}_{t+k})
, gradients flowing through the whole chain):
L
TF
\displaystyle L_{\mathrm{TF}}
=
1
T
​
∑
t
‖
f
θ
​
(
z
¯
t
)
−
sg
⁡
(
z
¯
t
+
1
)
‖
2
\displaystyle=\tfrac{1}{T}\textstyle\sum_{t}\big\|f_{\theta}(\bar{z}_{t})-\mathrm{sg}(\bar{z}_{t+1})\big\|^{2}
teacher-forced one-step latent prediction,
L
R
\displaystyle L_{\mathrm{R}}
=
1
K
​
∑
k
=
1
K
‖
z
^
t
+
k
−
sg
⁡
(
z
¯
t
+
k
)
‖
2
\displaystyle=\tfrac{1}{K}\textstyle\sum_{k=1}^{K}\big\|\hat{z}_{t+k}-\mathrm{sg}(\bar{z}_{t+k})\big\|^{2}
multi-step latent rollout consistency,
L
OBS
\displaystyle L_{\mathrm{OBS}}
=
1
T
​
∑
t
‖
D
ψ
​
(
f
θ
​
(
z
¯
t
)
)
−
u
t
+
1
‖
2
\displaystyle=\tfrac{1}{T}\textstyle\sum_{t}\big\|D_{\psi}(f_{\theta}(\bar{z}_{t}))-u_{t+1}\big\|^{2}
decode the teacher-forced prediction,
L
recon
\displaystyle L_{\mathrm{recon}}
=
1
T
​
∑
t
‖
D
ψ
​
(
E
ϕ
​
(
u
t
)
)
−
u
t
‖
2
\displaystyle=\tfrac{1}{T}\textstyle\sum_{t}\big\|D_{\psi}(E_{\phi}(u_{t}))-u_{t}\big\|^{2}
reconstruction anchor on the online encoder.
No term exposes the decoder to a latent more than one teacher-forced step from an observation.
3.2  The RDR objective
RDR decodes the same free-running rollout that
L
R
L_{\mathrm{R}}
constrains and that evaluation scores, and penalizes its error against the ground-truth fields:
L
RDR
=
1
K
​
∑
k
=
1
K
‖
D
ψ
​
(
z
^
t
+
k
)
−
u
t
+
k
‖
2
,
L
=
L
TF
+
α
e
​
L
R
+
L
OBS
+
L
recon
+
λ
​
L
RDR
.
L_{\mathrm{RDR}}=\frac{1}{K}\sum_{k=1}^{K}\big\|D_{\psi}(\hat{z}_{t+k})-u_{t+k}\big\|^{2},\qquad L=L_{\mathrm{TF}}+\alpha_{e}\,L_{\mathrm{R}}+L_{\mathrm{OBS}}+L_{\mathrm{recon}}+\lambda\,L_{\mathrm{RDR}}.
(1)
The gradient reaches the decoder directly and, through the rollout chain, the predictor and the encoder, so all three components are shaped by the trajectory the model will actually produce. The curriculum is standard for rollout losses:
α
e
\alpha_{e}
ramps linearly from 0 to 1 between epochs 2 and 5, the rollout branch (and with it the RDR term) opens after the two warm-up epochs, and the rollout is pure free-running throughout. Setting
λ
=
0
\lambda{=}0
recovers the baseline exactly; we call that arm
posterior-only
. The operating weight is
λ
=
0.3
\lambda{=}0.3
, and the result is flat across
λ
∈
{
0.1
,
0.3
,
0.6
,
1.0
}
\lambda\in\{0.1,0.3,0.6,1.0\}
(Section
5.2
).
The term adds no parameters. Both arms already compute the free-running rollout for
L
R
L_{\mathrm{R}}
, so the marginal cost is the
K
K
additional decoder evaluations per window and their backward pass, incurred at training time only; inference is unchanged. Appendix
C
reports step counts, decoder-evaluation counts, wall-clock, and hardware for every arm.
