# Efficient World Action Model Inference with Adaptive Intermediate States

paper_id: arxiv:2609.34608v1
tier: H
source_used: html_arxiv
warning: none

## Intro

World Action Models (WAMs) jointly model robot actions and how the environment may evolve under those actions, extending visuomotor policies with future-aware prediction for closed-loop control and planning
(
Kim et al., 2026
;
Yuan et al., 2026
;
Bi et al., 2026
)
.
However, WAM inference is computationally expensive because many WAMs generate action horizons, often together with future representations, through iterative diffusion or flow solvers
(
Chi et al., 2025
;
Black et al., 2025a
;
Hou et al., 2024
)
.
During closed-loop control, WAM inference involves multiple nested computation loops: an outer replanning loop repeatedly solves for new action horizons as observations arrive, an inner denoising loop iteratively refines future trajectories or representations, and the Transformer repeatedly processes intermediate states across network layers. As a result, WAMs repeatedly recompute highly related inference states along the observation-to-action path, introducing substantial GPU computation and latency overhead that ultimately limits closed-loop control frequency.
Figure 1:
The main idea of
WAMachine
.
WAMachine
adapts state across replans, denoising steps, and Transformer layers to reduce computation and observation-to-action latency.
K
K
denotes native denoising steps;
k
1
k_{1}
and
k
2
k_{2}
denote steps before and after real observation arrival, respectively;
k
1
+
k
2
<
K
k_{1}+k_{2}<K
describes the illustrated accepted warm-start path.
A natural direction for accelerating WAM inference is to preserve useful computation from earlier inference and adapt it as the control loop and inference context evolve.
Existing approaches explore this opportunity at different points in the inference pipeline.
RTI-DP and STEP leverage information from preceding control steps to construct better initializations for subsequent diffusion inference
(
Duan et al., 2025
;
Li et al., 2026
)
;
RTC and FutureRTC overlap action generation with physical execution to reduce computation exposed on the control path
(
Black et al., 2025b
;
Jiang et al., 2026
)
;
and TeaCache, DiCache, and C
3
ache reuse intermediate features or residuals across denoising steps or inference chunks to reduce repeated Transformer computation
(
Liu et al., 2025
;
Bu et al., 2026
;
Zhao et al., 2026
)
.
These methods demonstrate the value of reusing previously computed information in WAM inference.
However, existing approaches exploit reuse only at isolated points in the inference pipeline, capturing fragmented forms of redundancy rather than the continuous evolution of inference states.
In contrast, WAM inference exhibits
state continuity
: intermediate states remain informative across replans, denoising steps, and network layers, while evolving with the task context and computation.
The central challenge is therefore to preserve and adapt inference states across replans, denoising steps, and network layers, enabling efficient closed-loop WAM inference without redundant recomputation.
To address this challenge, we view closed-loop WAM inference as a continuously evolving stateful process, where inference states generated throughout the control loop remain informative as the task context changes, exposing a fundamental opportunity for cross-stage state transport.
We propose
WAMachine
1
1
1
WAMachine
embodies a state-machine view of WAMs, where inference evolves through the preservation and adaptation of computational states.
, a training-free stateful inference framework that preserves and adapts evolving inference states across replans, denoising steps, and Transformer layers, transforming isolated reuse opportunities into a unified acceleration framework for closed-loop WAM inference.
Specifically,
WAMachine
exploits state continuity within the three nested computation loops of WAM inference: closed-loop replanning, iterative denoising, and diffusion Transformer (DiT) forward execution
(
Peebles and Xie, 2023
)
.
Across closed-loop replanning,
Trajectory Remapping
reduces redundant trajectory generation by carrying informative replan state from the preceding replan into the next replan.
Since consecutive replans typically share the same task objective and exhibit gradual state evolution, the preceding solution provides a strong initialization signal despite updated observations.
Trajectory Remapping remaps these states to the new planning context, allowing WAMs to bypass unnecessary exploration from noise and accelerate closed-loop replanning without additional training.
Across iterative denoising,
Observation Rebinding
reduces latency exposed to the control loop by continuing partially completed inference across physical execution.
Future-aware WAMs provide predictions of upcoming observations, enabling anticipatory inference before the next observation is available.
By rebinding retained denoising states to the realized observation, Observation Rebinding preserves computation that passes consistency checks and refreshes inference otherwise.
Across DiT forward execution,
Residual Rescaling
reduces repeated Transformer computation by reusing temporally coherent intermediate representations across denoising steps.
Our analysis reveals that hidden states in the middle layers evolve smoothly across adjacent denoising iterations, enabling residual-level reuse with lightweight consistency checking.
Residual Rescaling selectively rescales retained residual states and refreshes them through full computation of the middle layers when the probe checks fail, reducing DiT inference cost.
We evaluate
WAMachine
using three representative WAM architectures on LIBERO
(
Liu et al., 2023
)
and RoboTwin 2.0
(
Chen et al., 2026
)
.
Our results show that
WAMachine
achieves 1.47–3.05
×
\times
speedups in observation-to-action latency and 2.23–3.27
×
\times
speedups in GPU inference time per replan relative to the native runtimes, while retaining 96.69–99.54% of native task success in large-sample evaluations.
Notably, ablations on Fast-WAM-IDM show that the complete system achieves the lowest observation-to-action latency among the tested variants and retains substantial acceleration even without CUDA Graphs, demonstrating the effectiveness of stateful inference in reducing both computation and control latency.
Our main contributions are as follows:
•
We reveal state continuity as a new acceleration opportunity in WAM inference and introduce stateful inference for preserving and adapting evolving inference states.
•
We propose
WAMachine
, a training-free framework that transports inference states across replanning, denoising, and Transformer execution through trajectory remapping, observation rebinding, and residual rescaling.
•
WAMachine
achieves 1.47–3.05
×
\times
speedups in observation-to-action latency and 2.23–3.27
×
\times
speedups in GPU inference time per replan for three representative WAMs on LIBERO and RoboTwin 2.0, while retaining 96.69–99.54% of native task success.

## Method

WAMachine
exploits state continuity through a training-free stateful inference framework that preserves and adapts inference state as the control loop evolves; Figure
3
illustrates the overall framework.
For each new replan, Trajectory Remapping preserves replan state from the preceding replan and adapts it through remapping to construct an initialization for the next replan.
During physical action execution, Observation Rebinding preserves denoising state produced by a bounded anticipatory inference prefix advanced under the predicted future.
When the real observation arrives, inference continues from the retained denoising state through observation rebinding if consistency checks pass; otherwise, it restarts from the remapped initialization.
For each remaining denoising step, Residual Rescaling preserves exact layer state and adapts it to the current input through residual rescaling guided by a shallow probe.
Figure 2:
Evidence for state continuity.
(a)
Consecutive replans show high cosine similarity and small relative
L
2
L_{2}
distance between trajectory latents at matching early stages (top); remapped initialization enables three-step inference with low action RMSE and 96–100% success on LIBERO (bottom).
(b)
Most anticipatory prefixes are ready within the action execution window (below
y
=
x
y=x
); color shows action RMSE after observation rebinding relative to full inference under the real condition.
(c)
Relative
L
2
L_{2}
changes between adjacent layer hidden states are small in the middle Action-DiT layers (top); residual rescaling and state refresh control relative output
L
2
L_{2}
error compared with repeatedly reusing the first-step residual
R
1
R_{1}
(bottom).
Analysis protocols appear in Appendix
E
.
3.1
Trajectory Remapping across Closed-Loop Replans
Although each replan receives a new observation, the high-level instruction remains unchanged within an embodied task, while the robot and environment typically change gradually.
We therefore compare trajectory latents between consecutive replans on Cosmos Policy.
As shown in Figure
2
(a), these latents remain highly similar at matching early denoising stages, and remapped initialization enables three-step inference with low action RMSE while largely preserving task success across LIBERO suites.
These observations motivate Trajectory Remapping to preserve replan state from the preceding replan and adapt it to the new planning context, enabling refinement with fewer denoising steps.
Specifically, let
t
t
index closed-loop replans and
r
r
index native solver stages.
For either the video or action branch, let
x
¯
t
\bar{x}_{t}
denote the final denoised output of replan
t
t
, and let
x
t
r
x_{t}^{r}
denote an intermediate trajectory latent retained at stage
r
r
with noise level
σ
r
>
0
\sigma_{r}>0
.
To express the retained latent relative to the final output, we divide their difference by
σ
r
\sigma_{r}
and normalize it to obtain the denoising direction
d
t
d_{t}
:
d
t
=
𝒩
⁡
(
x
t
r
−
x
¯
t
σ
r
)
,
d_{t}=\mathcal{N}\!\left(\frac{x_{t}^{r}-\bar{x}_{t}}{\sigma_{r}}\right),
(1)
where
𝒩
\mathcal{N}
is a model-specific normalization operator.
The pair
(
x
¯
t
,
d
t
)
(\bar{x}_{t},d_{t})
forms the retained replan state:
x
¯
t
\bar{x}_{t}
provides the denoised endpoint, while
d
t
d_{t}
provides a normalized direction that can be rescaled to the noise level used to initialize the next replan.
Let
𝒜
t
\mathcal{A}_{t}
denote the trajectory remapping operator for the video or action branch, defined according to the model’s execution protocol.
When horizon alignment is required,
𝒜
t
\mathcal{A}_{t}
shifts reusable slots in the branch’s temporal coordinates; otherwise, it preserves relative slot indices without shifting.
Let
ℳ
t
\mathcal{M}_{t}
denote the set of target slots that receive state from the preceding replan under
𝒜
t
\mathcal{A}_{t}
.
At entry stage
b
0
b_{0}
with noise level
σ
b
0
\sigma_{b_{0}}
, we initialize replan
t
+
1
t+1
using remapped state for slots in
ℳ
t
\mathcal{M}_{t}
and fresh noise elsewhere:
x
t
+
1
b
0
​
[
j
]
=
{
𝒜
t
​
(
x
¯
t
)
​
[
j
]
+
σ
b
0
​
𝒜
t
​
(
d
t
)
​
[
j
]
,
j
∈
ℳ
t
,
ξ
t
+
1
b
0
​
[
j
]
,
j
∉
ℳ
t
,
x_{t+1}^{b_{0}}[j]=\begin{cases}\mathcal{A}_{t}(\bar{x}_{t})[j]+\sigma_{b_{0}}\mathcal{A}_{t}(d_{t})[j],&j\in\mathcal{M}_{t},\\[2.0pt]
\xi_{t+1}^{b_{0}}[j],&j\notin\mathcal{M}_{t},\end{cases}
(2)
where
j
j
indexes trajectory slots and
ξ
t
+
1
b
0
\xi_{t+1}^{b_{0}}
is fresh noise drawn from the model’s initialization distribution at noise level
σ
b
0
\sigma_{b_{0}}
.
The entry stage
b
0
b_{0}
is fixed for each model profile: a later stage reduces denoising computation but leaves fewer steps to adapt to the new planning context.
Inference may begin under the predicted future through Observation Rebinding; after the real observation arrives, subsequent denoising uses it as conditioning.
For the first replan,
WAMachine
initializes inference with fresh noise.
Figure 3:
The framework of
WAMachine
.
Left:
Trajectory Remapping remaps replan state from the preceding replan to initialize the next replan.
Top right:
Observation Rebinding advances an anticipatory prefix during action execution, then rebinds retained denoising state to the real observation for continuation if consistency checks pass; otherwise, inference restarts from the remapped initialization under the real condition.
Bottom:
Residual Rescaling uses a shallow probe to rescale retained residuals and skip the middle layers; if consistency checks fail, it performs full computation of these layers and refreshes the retained layer state.
3.2
Observation Rebinding across Denoising Steps
Future-aware WAMs predict upcoming observations, enabling anticipatory inference during physical action execution.
Figure
2
(b) shows that most inference prefixes on Cosmos Policy are ready within this execution window.
Observation Rebinding exploits this window by advancing a bounded inference prefix under the predicted future and retaining the resulting denoising state for rebinding to the real observation.
Let
F
^
t
\widehat{F}_{t}
denote the future predicted at replan
t
t
.
Together with the current robot state
s
t
s_{t}
and task instruction
g
g
, it serves as input to a model-specific encoder
ψ
\psi
that constructs the predicted condition for replan
t
+
1
t+1
:
c
^
t
+
1
=
ψ
⁡
(
F
^
t
,
s
t
,
g
)
.
\widehat{c}_{t+1}=\psi\!\left(\widehat{F}_{t},s_{t},g\right).
(3)
Starting from the initialization produced by Trajectory Remapping, Observation Rebinding advances inference only to a predefined rebinding stage and retains the current latent and native solver history as denoising state.
Bounding the anticipatory prefix limits computation under the predicted future, and the prefix does not issue robot actions.
When the real observation arrives, the encoder
ψ
\psi
constructs the real condition
c
t
+
1
c_{t+1}
.
The consistency check compares predictions
u
^
m
\widehat{u}_{m}
with observations
u
m
u_{m}
for each component
m
∈
𝒱
m\in\mathcal{V}
, such as images and robot states, before encoding.
RMSE
e
m
e_{m}
measures their discrepancy after normalization by
δ
m
\delta_{m}
.
The consistency score
S
S
aggregates these errors through a weighted root mean square, where
w
m
w_{m}
denotes the weight of component
m
m
:
e
m
=
mean
⁡
[
(
u
^
m
−
u
m
δ
m
)
2
]
,
S
=
∑
m
∈
𝒱
w
m
​
e
m
2
∑
m
∈
𝒱
w
m
.
e_{m}=\sqrt{\operatorname{mean}\!\left[\left(\frac{\widehat{u}_{m}-u_{m}}{\delta_{m}}\right)^{2}\right]},\qquad S=\sqrt{\frac{\sum_{m\in\mathcal{V}}w_{m}e_{m}^{2}}{\sum_{m\in\mathcal{V}}w_{m}}}.
(4)
Here,
mean
\operatorname{mean}
averages over the elements of each component.
Let
τ
\tau
denote the consistency threshold and
κ
\kappa
count consecutive accepted rebindings, capped at
κ
max
\kappa_{\max}
.
Rebinding proceeds only when
S
<
τ
S<\tau
and
κ
<
κ
max
\kappa<\kappa_{\max}
.
Each accepted rebinding increments
κ
\kappa
, while a refresh under the real condition resets it to zero.
To prevent error accumulation,
WAMachine
forces a refresh at the next replan when
κ
\kappa
reaches
κ
max
\kappa_{\max}
and skips the corresponding anticipatory prefix.
Appendix
B.3
lists model-specific components, normalization scales, weights, and thresholds.
If the consistency checks pass, Observation Rebinding rebinds the retained denoising state by replacing
c
^
t
+
1
\widehat{c}_{t+1}
with
c
t
+
1
c_{t+1}
and continues inference from that state.
Otherwise, inference restarts from the remapped initialization under the real condition.
3.3
Residual Rescaling across DiT Layers
Each denoising step requires Transformer computation, both within the anticipatory prefix and during inference under the real condition.
Figure
2
(c, top) shows small relative
L
2
L_{2}
changes between adjacent layer hidden states in the middle Action-DiT layers of Fast-WAM-IDM, while Figure
2
(c, bottom) shows that adaptive rescaling and refresh help control relative output
L
2
L_{2}
error.
These observations motivate Residual Rescaling to adapt retained layer state to the current input through residual rescaling guided by a shallow probe.
When consistency checks fail, Residual Rescaling performs full computation of the middle layers and retains the exact layer state for later reuse.
Consider a DiT with
L
L
layers, indexed from
0
0
to
L
−
1
L-1
.
Layer boundaries
0
<
q
<
s
<
e
<
L
0<q<s<e<L
partition the DiT into a head
[
0
,
q
)
[0,q)
, a shallow probe
[
q
,
s
)
[q,s)
, middle layers
[
s
,
e
)
[s,e)
, and a tail
[
e
,
L
)
[e,L)
.
Let
h
ℓ
​
(
i
)
h_{\ell}(i)
denote the hidden state after the first
ℓ
\ell
layers at denoising step
i
i
, with
h
0
​
(
i
)
h_{0}(i)
denoting the DiT input.
After full computation at denoising step
a
a
, Residual Rescaling retains two residuals as the exact layer state:
p
a
=
h
s
​
(
a
)
−
h
q
​
(
a
)
,
R
a
=
h
e
​
(
a
)
−
h
s
​
(
a
)
,
p_{a}=h_{s}(a)-h_{q}(a),\qquad R_{a}=h_{e}(a)-h_{s}(a),
(5)
where
p
a
p_{a}
is the shallow probe residual and
R
a
R_{a}
is the cumulative residual of the middle layers.
At a subsequent denoising step
i
i
, Residual Rescaling executes the head and shallow probe on the current input to obtain the probe residual
p
i
=
h
s
​
(
i
)
−
h
q
​
(
i
)
.
p_{i}=h_{s}(i)-h_{q}(i).
(6)
Residual Rescaling estimates how to rescale the retained residual
R
a
R_{a}
by comparing the current probe residual
p
i
p_{i}
with the retained probe residual
p
a
p_{a}
.
It fits
α
i
​
p
a
\alpha_{i}p_{a}
to
p
i
p_{i}
by least squares, then clips the estimated rescaling factor
α
i
\alpha_{i}
to the allowed range:
α
i
=
clamp
⁡
(
mean
⁡
(
p
i
⊙
p
a
)
max
⁡
(
mean
⁡
(
p
a
⊙
p
a
)
,
ϵ
)
,
α
min
,
α
max
)
,
\alpha_{i}=\operatorname{clamp}\!\left(\frac{\operatorname{mean}(p_{i}\odot p_{a})}{\max(\operatorname{mean}(p_{a}\odot p_{a}),\epsilon)},\alpha_{\min},\alpha_{\max}\right),
(7)
where
⊙
\odot
denotes elementwise multiplication and
mean
\operatorname{mean}
averages over probe elements.
The denominator has a positive lower bound
ϵ
\epsilon
, and
clamp
\operatorname{clamp}
bounds the rescaling factor
α
i
\alpha_{i}
within
[
α
min
,
α
max
]
[\alpha_{\min},\alpha_{\max}]
.
Reuse requires directional agreement between
p
i
p_{i}
and
p
a
p_{a}
and a small relative fitting error of
α
i
​
p
a
\alpha_{i}p_{a}
:
cos
⁡
(
p
i
,
p
a
)
≥
γ
cos
,
‖
p
i
−
α
i
​
p
a
‖
2
‖
p
i
‖
2
≤
γ
fit
.
\cos(p_{i},p_{a})\geq\gamma_{\mathrm{cos}},\qquad\frac{\|p_{i}-\alpha_{i}p_{a}\|_{2}}{\|p_{i}\|_{2}}\leq\gamma_{\mathrm{fit}}.
(8)
Here
γ
cos
\gamma_{\mathrm{cos}}
sets the minimum cosine similarity,
γ
fit
\gamma_{\mathrm{fit}}
bounds the relative fitting error, and
∥
⋅
∥
2
\|\cdot\|_{2}
denotes the
L
2
L_{2}
norm.
For joint execution of the video and action branches, Residual Rescaling estimates separate rescaling factors and skips the middle layers only when both branches pass the consistency checks.
For independent execution, it decides reuse separately for each branch.
Appendix
B.2
lists the layer boundaries, rescaling bounds, and consistency thresholds for each model.
Residual Rescaling skips the middle layers if the checks pass, approximating their output as
h
~
e
​
(
i
)
=
h
s
​
(
i
)
+
α
i
​
R
a
.
\widetilde{h}_{e}(i)=h_{s}(i)+\alpha_{i}R_{a}.
(9)
Accepted reuse preserves the retained layer state for subsequent denoising steps.
If the consistency checks fail, Residual Rescaling resumes full computation of the middle layers from the already computed
h
s
​
(
i
)
h_{s}(i)
.
It then refreshes the retained layer state with the exact probe residual
p
i
p_{i}
and the newly computed cumulative residual of the middle layers.
Both paths reuse the completed head and shallow probe computation, then execute the tail layers and compute the model output.
After observation rebinding, Residual Rescaling applies the same consistency checks under the real condition before reusing the retained layer state.
3.4
State Preservation and Adaptation
WAMachine
coordinates the three mechanisms to preserve and adapt replan state, denoising state, and layer state at complementary scopes.
Trajectory Remapping supplies the remapped initialization from which Observation Rebinding advances anticipatory inference during action execution, retaining denoising state for rebinding to the real observation.
Within both anticipatory inference and inference under the real condition, Residual Rescaling uses a shallow probe to rescale retained layer state and skip the middle layers when consistency checks pass.
If the retained denoising state fails the consistency checks, inference restarts from the remapped initialization, so the replan state remains useful.
After observation rebinding succeeds, Residual Rescaling still checks the retained layer state under the real condition and refreshes it through full computation of the middle layers if its checks fail.
This design preserves useful state across the three scopes while allowing each mechanism to adapt or refresh its state as the inference context changes.
Model-specific integration details and CUDA Graphs preparation appear in Appendix
B
.
