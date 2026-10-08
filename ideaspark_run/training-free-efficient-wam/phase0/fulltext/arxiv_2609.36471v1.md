# Staircase Policy: Streaming Inference for World-Action Models with Large Action Chunks

paper_id: arxiv:2609.36471v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Figure 1:
Success rate versus measured inference throughput on LIBERO. The measurement protocol, per-method results, and methods omitted from the figure are provided in Appendix
C
.
Vision-language-action (VLA) models have achieved increasingly strong performance in robotic manipulation. These models typically combine a large pretrained vision-language backbone with an action decoder. Recent architectures often use a flow-matching-based Diffusion Transformer (DiT) as the action decoder to generate continuous actions directly
(
Kim et al., 2024
;
Black et al., 2024
)
. This design is becoming increasingly common in generalist manipulation policies and supports long action chunks in a single forward pass.
A recent extension is the World-Action Model (WAM)
(
Chen et al., 2026a
;
Team et al., 2026
;
Chen et al., 2026b
)
, which introduces an additional future-prediction module to predict a future observation or its latent representation.
Unlike WAMs that explicitly predict future images, JEPA-style variants predict a latent representation rather than pixels
(
Miao et al., 2026
;
Lin et al., 2026b
)
. We study one instance of the latter family, where a lightweight predictor produces the future latent representation. Specifically, the current observation and the predicted future representation are then jointly used to condition action generation. Despite this more efficient design, future prediction still introduces additional inference overhead on top of an already expensive policy
(
Chen et al., 2026a
;
Yuan et al., 2026b
)
.
To sustain high-frequency control, the policy must generate new action chunks sufficiently frequently: a low-level controller may execute actions at several hundred hertz, while the policy inference frequency is typically below 10 Hz.
A single WAM inference requires a forward pass through a multi-billion-parameter backbone and a future-prediction module, followed by iterative action denoising.
This inference cost becomes particularly limiting for onboard deployment on edge devices
(
Yang et al., 2026
;
Sun et al., 2026e
)
. Improving the action throughput of WAMs without degrading task performance is therefore important for practical deployment.
Despite slow policy inference, action chunking provides a practical way to sustain high-frequency control
(
Zhao et al., 2023
)
. Given an observation
o
t
o_{t}
, the policy generates
H
H
future actions in a single forward pass, allowing one expensive inference to be amortized over multiple control steps. Effective throughput, however, depends not on the generated horizon
H
H
, but on the executed horizon
H
exec
≤
H
H_{\mathrm{exec}}\leq H
: the number of generated actions actually executed before the next policy update. In practice,
H
exec
H_{\mathrm{exec}}
is often substantially smaller than
H
H
. A deployed system typically executes only the first few actions of a chunk, often one to twenty, acquires a new observation, queries the policy again, and discards the remaining actions
(
Lu et al., 2026
)
.
This limited execution horizon is necessary because the reliability of later actions decreases as execution proceeds farther from the observation on which the chunk was conditioned. Tracking and contact errors accumulate while the robot acts without a new observation, uncertainty increases with the prediction horizon, and changes in the scene during execution are not reflected in the original conditioning observation. Increasing the generated chunk length alone therefore does not necessarily improve effective throughput. With the model weights and denoising budget fixed, increasing the executed horizon from
10
10
to
50
50
actions reduces success by
12.8
12.8
points for LaWAM
(
Chen et al., 2026a
)
and
26.8
26.8
points for
π
0.5
\pi_{0.5}
(
Intelligence et al., 2025
)
(Sec.
5.1
). The resulting challenge is to support a long execution horizon while allowing unexecuted actions to be continuously updated using the latest observation.
To address this challenge, we propose
Staircase Policy
, a streaming inference and training framework for World-Action Models with large action chunks: it adds a lightweight predictor of the future observation latent to a flow-matching VLA, turning it into the JEPA-style WAM we call an
S-WAM
.
Staircase Policy
uses future prediction to maintain reliable actions over longer execution horizons.
It maintains a buffer of sub-chunks at staggered denoising stages.
First, it
progressively generates and immediately executes
near-term sub-chunks, allowing action execution to begin before the full chunk has been generated. Second, it
continuously refines unexecuted actions
: once a near-term sub-chunk has finished executing, a new observation is available, and
Staircase Policy
re-encodes that observation, re-predicts the future latent representation, and advances all remaining actions by one denoising step under the updated condition.
This design supports long execution horizons while maintaining task performance:
S-WAM
achieves
97.7
%
97.7\%
on LIBERO and
87.9
%
87.9\%
on LIBERO-Plus, with
292.7
292.7
executed actions per second (
642.9
642.9
with additional optimizations) and a
73.3
73.3
ms time-to-first-action (TTFA).
Our contributions are summarized as follows:
•
We propose
Staircase Policy
, a streaming inference and training framework
for World-Action Models that pipelines the generation and execution of large action chunks.
Staircase Policy
maintains sub-chunks at staggered denoising stages, allowing near-term actions to be executed while later actions continue to be generated.
•
We introduce
observation-conditioned refinement
for long action chunks. As new observations arrive during execution,
Staircase Policy
updates the predicted future latent and continuously refines unexecuted actions, allowing the chunk to adapt without repeatedly invoking the full policy.
•
We show that
future-prediction error can serve as a practical signal for adaptive chunking
. By comparing the predicted future latent with the subsequently observed one,
Staircase Policy
can decide when to terminate the current chunk and replan.
•
We demonstrate
strong efficiency, robustness, and transferability
across multiple simulation benchmarks, policy backbones, and real-robot tasks.
S-WAM
substantially improves action throughput and TTFA while maintaining strong task performance, including under perturbations and dynamic-scene evaluations.

## Method

Staircase Policy
is a streaming inference and training framework that turns a flow-matching VLA into a JEPA-style WAM. The design is motivated by two properties of long-horizon action generation.
Denoising requirements vary across the action horizon.
Near-term actions are generally easier to predict, while actions farther into the future are less reliable
(
Lu et al., 2026
)
. This suggests allocating denoising computation non-uniformly across the chunk. Because flow-matching trajectories are approximately straight
(
Liu et al., 2022
;
Lipman et al., 2022
)
, a single velocity evaluation can already provide a useful estimate toward the data endpoint.
Staircase Policy
therefore emits the first sub-chunk after one denoising step and gives each subsequent sub-chunk one additional step (Sec.
3.2
).
Visual conditioning should be refreshed during execution.
Object poses, contacts, and robot configuration change as actions are executed, while the task instruction and high-level semantic context remain largely unchanged within a chunk.
Staircase Policy
therefore refreshes only the lightweight vision-side future predictor at each sub-chunk boundary, while reusing the cached vision-language context. This provides updated visual conditioning at millisecond-scale cost without repeatedly invoking the substantially more expensive backbone (Sec.
3.3
).
Figure 2:
Overview of
Staircase Policy
. Here
t
i
t_{i}
denotes a time step and
t
i
∗
t_{i}^{*}
the predicted future.
3.1
Setup
The host is a flow-matching VLA mapping an observation
o
o
and instruction
ℓ
\ell
to a chunk of
H
H
actions
(
Zhao et al., 2023
)
: a
backbone
of billions of parameters emits a semantic context
e
e
and a compact latent plan
z
z
(
Bruce et al., 2024
)
, and a flow-matching
expert
v
θ
v_{\theta}
denoises a buffer
x
∈
ℝ
H
×
d
a
x\in\mathbb{R}^{H\times d_{a}}
toward the data along the linear path from noise
(
Lipman et al., 2022
)
. We require only that the expert be callable as a
single
Euler step against a cached backbone context.
Staircase Policy
adds a lightweight
future predictor
consisting of a frozen vision encoder
h
=
E
⁡
(
o
)
h=E(o)
and a decoder
g
g
that predicts the feature map of a near-future observation, the
future latent
f
^
=
g
⁡
(
h
,
z
)
\hat{f}=g(h,z)
. The expert is conditioned on
c
=
(
h
,
f
^
,
e
)
c=(h,\hat{f},e)
, and we refer to the resulting streaming World-Action Model as an
S-WAM
. The future predictor we introduce follows the latent world model of LaWAM
(
Chen et al., 2026a
)
, retaining its predictor architecture while training it under the streaming schedule described below.
Under conventional inference, the entire action buffer is denoised for
n
n
steps, after which the policy executes a prefix of
H
exec
≤
H
H_{\mathrm{exec}}\leq H
actions without taking a new observation and discards the remainder.
Increasing
H
exec
H_{\mathrm{exec}}
improves executed-action throughput, but also requires executing later actions, which are less reliable as execution moves farther from the conditioning observation.
Staircase Policy
changes how the action chunk is denoised and executed: instead of fully denoising the entire chunk before execution, it progressively generates near-term actions while continuously updating the remaining actions with new observations, without modifying the host backbone or flow expert.
Algorithm 1
Staircase Policy
inference for one chunk of
H
=
K
​
G
H=KG
actions
1:
once per chunk:
(
e
,
z
)
←
backbone
⁡
(
o
0
,
ℓ
)
(e,z)\leftarrow\mathrm{backbone}(o_{0},\ell)
2:
x
∼
𝒩
⁡
(
0
,
I
)
x\sim\mathcal{N}(0,I)
;
τ
←
0
\tau\leftarrow 0
3:
for
j
=
0
,
…
,
K
−
1
j=0,\dots,K-1
do
4:
h
j
←
E
⁡
(
o
j
)
h_{j}\leftarrow E(o_{j})
;
f
^
j
←
g
⁡
(
h
j
,
z
)
\hat{f}_{j}\leftarrow g(h_{j},z)
⊳
\triangleright
refresh, Eq. (
2
)
5:
v
←
v
θ
(
x
,
τ
∣
h
j
,
f
^
j
,
e
)
v\leftarrow v_{\theta}(x,\tau\mid h_{j},\hat{f}_{j},e)
⊳
\triangleright
one expert pass
6:
emit
A
^
j
=
[
x
+
(
1
−
τ
)
v
]
j
​
G
:
(
j
+
1
)
​
G
\hat{A}_{j}=[\,x+(1-\tau)\,v\,]_{jG:(j+1)G}
⊳
\triangleright
readout, Eq. (
1
)
7:
x
←
x
+
1
K
​
v
x\leftarrow x+\frac{1}{K}v
;
τ
←
τ
+
1
K
\tau\leftarrow\tau+\frac{1}{K}
8:
execute
A
^
j
\hat{A}_{j}
(
G
G
steps); observe
o
j
+
1
o_{j+1}
9:
end
for
3.2
The Denoising Staircase
We partition the chunk into
K
K
contiguous sub-chunks of size
G
G
, so that the chunk length is
H
=
K
​
G
H=KG
. These two hyper-parameters are the only scheduling parameters varied in our study (Sec.
5.2
); unless otherwise specified, we use
H
=
50
H{=}50
,
K
=
5
K{=}5
,
G
=
10
G{=}10
. Algorithm
1
states the resulting inference loop.
The buffer starts as pure noise at a single shared flow time
τ
=
0
\tau{=}0
, and each query advances
all
positions by one Euler step of size
1
/
K
1/K
,
x
←
x
+
1
K
​
v
θ
​
(
x
,
τ
∣
c
j
)
x\leftarrow x+\frac{1}{K}v_{\theta}(x,\tau\mid c_{j})
with
τ
←
τ
+
1
K
\tau\leftarrow\tau+\frac{1}{K}
, giving the grid on the right of Fig.
2
. Immediately after the
j
j
-th update (
j
=
0
,
…
,
K
−
1
j=0,\dots,K{-}1
) the
j
j
-th sub-chunk is
read out
by extrapolating the same velocity estimate to
τ
=
1
\tau{=}1
,
A
^
j
=
[
x
+
(
1
−
τ
)
v
θ
(
x
,
τ
∣
c
j
)
]
j
​
G
:
(
j
+
1
)
​
G
,
\hat{A}_{j}=\big[\,x+(1-\tau)\,v_{\theta}(x,\tau\mid c_{j})\big]_{\,jG:(j+1)G},
(1)
and handed to the controller while the rest of the buffer keeps denoising. We denote the full extrapolated buffer before slicing as
A
^
j
full
\hat{A}_{j}^{\,\mathrm{full}}
.
Sub-chunk
j
j
is therefore emitted after
j
+
1
j{+}1
forward passes of the DiT action expert. Although different sub-chunks have different readout depths, all buffer positions remain at the same flow time
τ
\tau
, so the expert always receives inputs from the shared-
τ
\tau
distribution used during training. Per
H
H
executed actions, the schedule requires one backbone pass and
K
K
denoising steps.
3.3
Refreshing the Future Predictor
The denoising staircase creates
K
K
sub-chunk boundaries within each chunk. At every boundary, a new observation is available when the expert performs its next denoising update. After completing sub-chunk
j
−
1
j{-}1
, the robot observes
o
j
o_{j}
, and
Staircase Policy
recomputes the future prediction as
h
j
=
E
⁡
(
o
j
)
,
f
^
j
=
g
⁡
(
h
j
,
z
)
,
c
j
=
(
h
j
,
f
^
j
,
e
)
,
h_{j}=E(o_{j}),\quad\hat{f}_{j}=g(h_{j},z),\quad c_{j}=(h_{j},\,\hat{f}_{j},\,e),
(2)
while keeping
z
z
and
e
e
fixed at their chunk-start values. Thus, the expensive VLA backbone runs one forward pass per full chunk, whereas the lightweight future predictor runs once per sub-chunk boundary, as illustrated in Fig.
2
.
Each refreshed condition is applied to all unexecuted positions in the buffer. Consequently, changes observed at boundary
j
j
can influence every subsequent action in the current chunk without requiring a full policy replan. This observation-conditioned refresh is therefore what allows the staircase schedule to maintain action quality over a long execution horizon.
Because the refresh runs as a chain, the predicted future is itself a usable signal:
f
^
j
−
1
\hat{f}_{j-1}
targets
h
j
h_{j}
, which is encoded one sub-chunk later, so every boundary yields a prediction–realization pair that a policy denoising the chunk only once never has. Their discrepancy
δ
j
=
MSE
⁡
(
f
^
j
−
1
,
h
j
)
\delta_{j}=\operatorname{MSE}(\hat{f}_{j-1},\,h_{j})
grows when the scene departs from what the model anticipated at planning time, for instance because the target was displaced. If
δ
j
\delta_{j}
exceeds a threshold, the chunk ends after sub-chunk
j
j
and the backbone plans a new one (right of Fig.
2
), making the executed horizon data-dependent rather than fixed at
K
K
sub-chunks.
We explore such signal in detail in Sec.
5.4
.
3.4
Training Under the Deployment Schedule
Standard training does not expose the policy to several states encountered by Algorithm
1
, including partially denoised buffers, one-step readouts, changes in conditioning within a chunk, and self-predicted future latents. We therefore train the model using the same streaming schedule used at inference time. We further find that training this way converges considerably faster than conventional chunk training (Sec.
4.3
). Each demonstration window provides the
K
+
1
K{+}1
observations at the sub-chunk boundaries. At every boundary, the condition uses the
self-predicted
g
⁡
(
h
j
,
z
)
g(h_{j},z)
rather than the ground-truth
h
j
+
1
h_{j+1}
, matching the information available during deployment.
The forward rollout remains sequential, while the action buffer is detached between boundaries to avoid backpropagation through the full
K
K
-step chain. The
K
K
expert graphs are optimized in a shared backward pass.
Supervision targets the quantity the controller consumes, the readout (
1
) rather than the velocity. At boundary
j
j
each buffer position carries weight
1
1
if it lies in the emitted sub-chunk and
λ
ne
=
0.25
\lambda_{\mathrm{ne}}{=}0.25
otherwise, and the losses pool boundaries:
ℒ
act
=
∑
j
,
p
w
j
​
(
p
)
​
‖
A
^
j
full
​
(
p
)
−
A
⁡
(
p
)
‖
2
2
d
a
​
∑
j
,
p
w
j
​
(
p
)
,
ℒ
f
=
1
K
​
∑
j
=
0
K
−
1
MSE
⁡
(
g
⁡
(
h
j
,
z
)
,
h
j
+
1
)
.
\mathcal{L}_{\mathrm{act}}=\frac{\sum_{j,p}w_{j}(p)\,\|\hat{A}_{j}^{\,\mathrm{full}}(p)-A(p)\|_{2}^{2}}{d_{a}\sum_{j,p}w_{j}(p)},\qquad\mathcal{L}_{\mathrm{f}}=\frac{1}{K}\sum_{j=0}^{K-1}\operatorname{MSE}\!\left(g(h_{j},z),\,h_{j+1}\right).
(3)
Executed positions receive full weight, while the lower-weight dense term supervises all other valid positions in the buffer. The future latent remains attached to the computation graph: gradients from
ℒ
act
\mathcal{L}_{\mathrm{act}}
propagate through
f
^
j
\hat{f}_{j}
into
g
g
and
z
z
. Thus, the future predictor is optimized both for latent prediction and for its utility in action generation. The total objective is
ℒ
=
ℒ
act
+
β
⁡
(
ℒ
f
+
ℒ
z
)
\mathcal{L}=\mathcal{L}_{\mathrm{act}}+\beta(\mathcal{L}_{\mathrm{f}}+\mathcal{L}_{z})
with
β
=
0.1
\beta=0.1
, where
ℒ
z
\mathcal{L}_{z}
distills
z
z
toward the frozen latent-action encoder of the pretraining stage.
