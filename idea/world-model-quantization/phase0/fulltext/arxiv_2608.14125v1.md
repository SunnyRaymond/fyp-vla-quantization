# Traj-LeWM: Path-Aware World-Model Planning via Latent Trajectory Cost

paper_id: arxiv:2608.14125v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Figure 1:
Overview of Traj-LeWM. Stage 1 encodes the observation trajectory
and goal into latent representations used by LTC. Stage 2 trains LTC
with synthetic and mined closed-loop trajectory preferences while
retaining LeWM’s next-step prediction objective. Stage 3 ranks CEM
candidates using the joint endpoint-plus-LTC score.
Joint-embedding predictive architectures (JEPAs) support model-based
control by predicting future observations in compact latent spaces rather
than reconstructing decision-irrelevant pixel details
(
LeCun 2022
;
Assran et al. 2023
)
. LeWM
(
Maes et al. 2026
)
extends this paradigm into a lightweight
visual world model trained end-to-end from pixels and achieves strong
performance in goal-conditioned planning. However, this performance does
not remain consistent on more challenging tasks: LeWM’s success rate
drops from
96
%
96\%
on Push-T to
74
%
74\%
on OGBench-Cube, whereas DINO-WM
reports
86
%
86\%
success on Cube
(
Zhou et al. 2024
)
. Across tasks, we empirically find that candidates with the same start and
goal and similar endpoint costs can produce different execution outcomes,
suggesting that endpoint information alone cannot reliably capture
candidate quality. This motivates using complete-trajectory information
in both representation learning and candidate ranking.
We attribute this behavior to two missing uses of trajectory information.
First, LeWM’s next-step prediction loss supervises local transitions
between adjacent states, while SIGReg only regularizes the distribution
of latent representations; neither requires the shared encoder to
preserve information about a complete trajectory relative to its task
goal. Consequently, although autoregressive rollout produces a sequence
of predicted states, the representation space has not been directly
trained to assess the goal-conditioned quality of a complete candidate
trajectory. Second, LeWM ranks candidates only by the distance between
their predicted latent endpoints and the goal representation. A predicted
endpoint close to the goal does not guarantee that the same action
sequence will perform well when executed in the environment, and
candidates with similar predicted endpoints cannot be distinguished
using their different intermediate evolutions.
To address these limitations, we propose Traj-LeWM. It retains LeWM’s
original prediction objective and endpoint score while introducing a
goal-conditioned latent trajectory cost (LTC) over complete latent
trajectories. The overall framework is illustrated in
Fig.
1
. LTC maps the goal-relative evolution of a complete
latent trajectory to a learned scalar cost. During training, LTC is
learned from pairwise preferences that favor goal-matched expert
trajectories over synthetic negative trajectories and closed-loop
execution failures; synthetic preferences also shape the shared encoder.
During planning, LTC is combined with endpoint distance so that candidate
ranking uses both endpoint and intermediate-path information. Across four
simulated tasks, Traj-LeWM consistently outperforms LeWM, while evaluation
on a physical Franka FR3 further demonstrates its real-robot feasibility.
Controlled experiments separately show that trajectory supervision
improves the shared representation and that LTC uses intermediate-path
information to improve candidate ranking.
In summary, our contributions are threefold:
•
We introduce a learned goal-conditioned latent trajectory cost
that evaluates complete predicted trajectories. By aggregating
goal-relative latent evolution over time, LTC complements endpoint
distance with a path-sensitive planning signal.
•
We develop a trajectory-preference learning mechanism based on
synthetic negative trajectories and closed-loop execution failures.
Synthetic preferences introduce trajectory-level, goal-conditioned
supervision into the shared representation, while mined failures
further train LTC without changing LeWM’s original predictor
objective.
•
We conduct a comprehensive evaluation on four simulated tasks
and a physical Franka FR3. Compared with LeWM, Traj-LeWM improves
success rates by
3
3
,
14
14
,
7
7
, and
7
7
percentage points on Push-T,
OGBench-Cube, Reacher, and Two-Room, respectively, and increases
real-robot success from
50
%
50\%
to
70
%
70\%
over the same
20
20
tasks.
Controlled endpoint-only, endpoint-matched, and fixed-endpoint
analyses, together with ablations, further verify the effects of
trajectory-level representation shaping and intermediate-path-aware
candidate ranking.

## Method

Overview.
Traj-LeWM extends LeWM with a goal-conditioned latent trajectory cost (LTC),
S
ψ
​
(
τ
∣
g
)
S_{\psi}(\tau\mid g)
, that evaluates complete latent trajectories. As illustrated in
Fig.
1
, trajectory-level preferences train LTC and shape the shared representation, while during planning LTC supplements endpoint distance with a path-sensitive signal for ranking candidate action sequences.
Motivation: Two Gaps in LeWM
LeWM training and planning.
The agent receives pixel observations
o
t
∈
𝒪
o_{t}\in\mathcal{O}
and executes actions
a
t
∈
𝒜
a_{t}\in\mathcal{A}
. Following LeWM
(
Maes et al. 2026
)
, an encoder
ℰ
θ
:
𝒪
→
ℝ
d
\mathcal{E}_{\theta}:\mathcal{O}\rightarrow\mathbb{R}^{d}
maps each observation to a latent state
z
t
=
ℰ
θ
​
(
o
t
)
z_{t}=\mathcal{E}_{\theta}(o_{t})
, and an action-conditioned predictor
ℱ
ϕ
\mathcal{F}_{\phi}
predicts the next latent state. We write the goal observation as
g
g
and its representation as
z
g
=
ℰ
θ
​
(
g
)
z_{g}=\mathcal{E}_{\theta}(g)
. LeWM trains the encoder and predictor with a next-step latent prediction loss
ℒ
pred
\mathcal{L}_{\mathrm{pred}}
and uses SIGReg, denoted by
ℛ
sigreg
\mathcal{R}_{\mathrm{sigreg}}
, to prevent representation collapse:
ℒ
LeWM
≜
ℒ
pred
+
λ
sig
​
ℛ
sigreg
.
\mathcal{L}_{\mathrm{LeWM}}\triangleq\mathcal{L}_{\mathrm{pred}}+\lambda_{\mathrm{sig}}\mathcal{R}_{\mathrm{sigreg}}.
(1)
At inference time, LeWM freezes the world model and autoregressively rolls out each candidate action sequence
a
t
:
t
+
H
−
1
a_{t:t+H-1}
. With a real-prefix length
L
L
, the context-augmented predicted latent trajectory is
τ
^
t
(
a
)
=
(
z
t
−
L
+
1
:
t
,
z
^
t
+
1
:
t
+
H
)
.
\hat{\tau}_{t}(a)=\big(z_{t-L+1:t},\hat{z}_{t+1:t+H}\big).
(2)
LeWM scores a candidate solely by the distance between its predicted endpoint and the goal representation:
C
0
​
(
τ
^
t
​
(
a
)
,
g
)
≜
‖
z
^
t
+
H
−
z
g
‖
2
2
.
C_{0}(\hat{\tau}_{t}(a),g)\triangleq\big\|\hat{z}_{t+H}-z_{g}\big\|_{2}^{2}.
(3)
It then uses CEM to minimize
Eq.
3
and executes the selected action sequence in a receding-horizon manner. We omit the current-time subscript below and write the predicted trajectory as
τ
^
\hat{\tau}
.
Gap 1: Missing trajectory-level, goal-conditioned supervision.
The prediction loss constrains local transitions, whereas SIGReg shapes the distribution of latent representations. Neither term in
Eq.
1
jointly uses an ordered complete trajectory and its task goal. The original objective therefore provides no explicit trajectory-level, goal-conditioned supervision to the shared representation.
Gap 2: Endpoint scoring is insensitive to intermediate paths.
Eq.
3
reads only the predicted endpoint. For any two intermediate latent-state sequences
u
1
:
H
−
1
u_{1:H-1}
and
v
1
:
H
−
1
v_{1:H-1}
that share an initial state
z
0
z_{0}
and endpoint
z
H
z_{H}
,
C
0
(
(
z
0
,
u
1
:
H
−
1
,
z
H
)
,
g
)
=
C
0
(
(
z
0
,
v
1
:
H
−
1
,
z
H
)
,
g
)
.
C_{0}\big((z_{0},u_{1:H-1},z_{H}),g\big)=C_{0}\big((z_{0},v_{1:H-1},z_{H}),g\big).
(4)
The endpoint term therefore cannot directly use information about how a candidate reaches its endpoint. Intermediate dynamics may nevertheless provide additional evidence about candidate quality under the specified goal when endpoint distance alone is insufficient. Together, these gaps leave trajectory-level, goal-conditioned information without an explicit role in either representation learning or candidate scoring.
Goal-Conditioned Trajectory Cost
Designing a goal-conditioned path functional.
To use information along the complete predicted path rather than only
its endpoint, we formulate trajectory quality as a goal-conditioned path
functional over an ordered sequence of latent states and instantiate it
as a latent trajectory cost (LTC). LTC maps a predicted trajectory to a learned scalar cost by
aggregating latent states, their temporal changes, and goal-relative
information over time. Intermediate-path information therefore
contributes directly to trajectory evaluation rather than being
discarded by endpoint-only scoring. The ranking semantics of this cost
are learned from trajectory preferences: goal-matched expert
trajectories should receive lower costs than synthetic negative
trajectories and execution failures.
We implement this functional using four components:
(i) latent state and local evolution
, represented by the current
state
z
t
z_{t}
and the first difference
Δ
​
z
t
=
z
t
+
1
−
z
t
\Delta z_{t}=z_{t+1}-z_{t}
between adjacent latent states;
(ii) goal conditioning
, introduced at every step through the
goal-relative quantity
d
t
g
=
z
g
−
z
t
d_{t}^{g}=z_{g}-z_{t}
;
(iii) temporal position
, represented by the normalized phase
η
t
=
t
/
max
⁡
(
T
−
1
,
1
)
\eta_{t}=t/\max(T-1,1)
; and
(iv) learned trajectory aggregation
, in which
ℓ
ψ
\ell_{\psi}
maps
the information at each step to a nonnegative contribution and these
contributions are averaged over the trajectory. The goal-relative term
allows the same latent state and change to receive different evaluations
under different goals, while the phase term distinguishes where an
evolution occurs along the trajectory. Averaging over time produces a
length-normalized cost for the complete trajectory.
Definition 1
(Goal-Conditioned Latent Trajectory Cost)
.
Given a latent trajectory
τ
=
(
z
0
,
…
,
z
T
)
\tau=(z_{0},\ldots,z_{T})
with
T
≥
1
T\geq 1
and goal representation
z
g
z_{g}
, define
S
ψ
​
(
τ
∣
g
)
=
1
T
​
∑
t
=
0
T
−
1
ℓ
ψ
​
(
z
t
,
Δ
​
z
t
,
d
t
g
,
η
t
)
.
S_{\psi}(\tau\mid g)=\frac{1}{T}\sum_{t=0}^{T-1}\ell_{\psi}\!\left(z_{t},\Delta z_{t},d_{t}^{g},\eta_{t}\right).
(5)
Here
ℓ
ψ
:
ℝ
3
​
d
+
1
→
ℝ
≥
0
\ell_{\psi}:\mathbb{R}^{3d+1}\!\to\!\mathbb{R}_{\geq 0}
is a learnable nonnegative
per-step contribution function, whose aggregation over time defines the
cost of the complete trajectory.
Under the learned trajectory preferences, a lower
S
ψ
S_{\psi}
indicates a
trajectory preferred as goal-matched expert behavior, whereas a higher
value indicates a less preferred trajectory, such as a synthetic
negative or an execution failure. Unlike
C
0
C_{0}
, which reads only the
endpoint, LTC is structurally able to use intermediate latent states,
their changes, and their relations to the goal.
Trajectory Preference Supervision
Write a goal-conditioned example as
x
=
(
τ
,
g
)
x=(\tau,g)
and abbreviate
S
ψ
​
(
x
)
=
S
ψ
​
(
τ
∣
g
)
S_{\psi}(x)=S_{\psi}(\tau\mid g)
. For a preference pair
(
x
+
,
x
−
)
(x^{+},x^{-})
, define the cost margin as
m
ψ
​
(
x
+
,
x
−
)
=
S
ψ
​
(
x
−
)
−
S
ψ
​
(
x
+
)
m_{\psi}(x^{+},x^{-})=S_{\psi}(x^{-})-S_{\psi}(x^{+})
and use the pairwise logistic loss
ℓ
β
​
(
x
+
,
x
−
)
=
log
⁡
(
1
+
exp
⁡
[
−
m
ψ
​
(
x
+
,
x
−
)
β
]
)
,
\ell_{\beta}(x^{+},x^{-})=\log\!\left(1+\exp\!\left[-\frac{m_{\psi}(x^{+},x^{-})}{\beta}\right]\right),
(6)
where
β
>
0
\beta>0
is a temperature. Minimizing
Eq.
6
encourages
S
ψ
​
(
x
+
)
<
S
ψ
​
(
x
−
)
S_{\psi}(x^{+})<S_{\psi}(x^{-})
, so that a goal-matched expert trajectory is preferred to its corresponding negative.
Synthetic preferences.
For an expert trajectory
τ
+
=
(
z
0
+
,
…
,
z
T
+
)
\tau^{+}=(z_{0}^{+},\ldots,z_{T}^{+})
, let the goal representation be
z
g
=
z
T
+
z_{g}=z_{T}^{+}
and define the positive example
x
+
=
(
τ
+
,
g
)
x^{+}=(\tau^{+},g)
. The first negative replaces the goal with a batchwise cyclically shifted goal
g
′
g^{\prime}
:
x
gm
−
=
(
τ
+
,
g
′
)
.
x^{-}_{\mathrm{gm}}=(\tau^{+},g^{\prime}).
(7)
The second negative adds noise to the intermediate states. Draw
ϵ
t
∼
𝒩
⁡
(
0
,
σ
2
​
s
batch
2
​
I
)
\epsilon_{t}\sim\mathcal{N}(0,\sigma^{2}s_{\mathrm{batch}}^{2}I)
and fix
ϵ
0
=
ϵ
T
=
0
\epsilon_{0}=\epsilon_{T}=0
; the perturbed states are
z
t
−
=
z
t
+
+
ϵ
t
,
0
≤
t
≤
T
.
z_{t}^{-}=z_{t}^{+}+\epsilon_{t},\qquad 0\leq t\leq T.
(8)
Let
τ
jit
−
=
(
z
0
−
,
…
,
z
T
−
)
\tau^{-}_{\mathrm{jit}}=(z_{0}^{-},\ldots,z_{T}^{-})
and
x
jit
−
=
(
τ
jit
−
,
g
)
x^{-}_{\mathrm{jit}}=(\tau^{-}_{\mathrm{jit}},g)
. Goal mismatch trains LTC to assign an expert trajectory a lower cost with its corresponding goal than with an unrelated goal. Endpoint-preserving perturbations train LTC to distinguish different intermediate paths even when the initial and final states are fixed.
We stop gradients through the synthetic negative branch while retaining the current encoder’s computation graph for the expert branch. Synthetic preferences therefore update LTC directly and update the encoder through the expert branch; their effect on negative encodings is mediated indirectly by the cost landscape learned by LTC. The corresponding loss is
ℒ
path
=
𝔼
⁡
[
w
gm
​
ℓ
β
​
(
x
+
,
x
gm
−
)
+
w
jit
​
ℓ
β
​
(
x
+
,
x
jit
−
)
w
gm
+
w
jit
]
.
\mathcal{L}_{\mathrm{path}}=\mathbb{E}\!\left[\frac{w_{\mathrm{gm}}\ell_{\beta}(x^{+},x^{-}_{\mathrm{gm}})+w_{\mathrm{jit}}\ell_{\beta}(x^{+},x^{-}_{\mathrm{jit}})}{w_{\mathrm{gm}}+w_{\mathrm{jit}}}\right].
(9)
Closed-loop failure preferences.
Synthetic negatives expose predefined structural differences but cannot
cover errors produced by the model during actual planning. After each
training epoch, we therefore use the current model to perform
endpoint-only CEM planning and execute the resulting plan in a resettable
training environment. We deliberately omit LTC during failure mining so
that it does not filter out failures selected by endpoint-only scoring;
these episodes provide negative examples of planning errors that the
endpoint term alone cannot identify. The endpoint-only planner is used
only for collecting training failures, whereas final planning uses the
joint endpoint-plus-LTC score defined below. The environment’s success
criterion supplies a binary episode-level label. For each failed episode
i
i
, we encode the executed observations frame by frame as
τ
i
−
=
(
ℰ
θ
​
(
o
i
,
0
exec
)
,
…
,
ℰ
θ
​
(
o
i
,
T
i
−
exec
)
)
\tau^{-}_{i}=(\mathcal{E}_{\theta}(o^{\mathrm{exec}}_{i,0}),\ldots,\mathcal{E}_{\theta}(o^{\mathrm{exec}}_{i,T_{i}^{-}}))
and pair it with an expert trajectory
τ
i
+
\tau^{+}_{i}
having the same initial condition and goal:
(
x
i
+
,
x
i
−
)
=
(
(
τ
i
+
,
g
i
)
,
(
τ
i
−
,
g
i
)
)
.
(x_{i}^{+},x_{i}^{-})=\big((\tau^{+}_{i},g_{i}),(\tau^{-}_{i},g_{i})\big).
(10)
Positive and negative trajectories may differ in length because
Eq.
5
averages each trajectory over its own number of
transitions. We insert these preference pairs sequentially into a
bounded FIFO buffer
ℬ
\mathcal{B}
. The corresponding buffer loss is
ℒ
mined
=
𝔼
(
x
+
,
x
−
)
∼
ℬ
​
ℓ
β
​
(
x
+
,
x
−
)
.
\mathcal{L}_{\mathrm{mined}}=\mathbb{E}_{(x^{+},x^{-})\sim\mathcal{B}}\ell_{\beta}(x^{+},x^{-}).
(11)
The trajectory and goal representations inserted into the buffer are
detached, so
ℒ
mined
\mathcal{L}_{\mathrm{mined}}
updates only LTC in the current backward pass. It
first changes LTC parameters and then indirectly influences the encoder
through
ℒ
path
\mathcal{L}_{\mathrm{path}}
in subsequent batches.
We intentionally exclude the mined observations from predictor training,
because using them as transition targets would introduce online dynamics
adaptation and confound it with the effect of LTC. Closed-loop mining is
instead an integral part of the proposed trajectory-preference module:
it supplies failure preferences only for calibrating LTC, while LeWM’s
predictor data and objective remain unchanged.
Joint Training and Gradient Flow
Writing the LeWM objective defined in
Eq.
1
as
ℒ
LeWM
\mathcal{L}_{\mathrm{LeWM}}
, the complete training objective is
ℒ
Traj
​
-
​
LeWM
=
ℒ
LeWM
+
λ
path
​
ℒ
path
+
λ
mined
​
ℒ
mined
.
\mathcal{L}_{\mathrm{Traj\text{-}LeWM}}=\mathcal{L}_{\mathrm{LeWM}}+\lambda_{\mathrm{path}}\mathcal{L}_{\mathrm{path}}+\lambda_{\mathrm{mined}}\mathcal{L}_{\mathrm{mined}}.
(12)
Table
1
summarizes the direct gradient flow from each loss in one backward pass. This design preserves the predictor’s training signal while allowing trajectory preferences to shape the shared encoder through the synthetic expert branch. Closed-loop failure preferences primarily calibrate LTC against the model’s own failure modes, while the synthetic expert branch supplies trajectory-level supervision to the encoder.
Loss
Encoder
ℰ
θ
\mathcal{E}_{\theta}
Predictor
ℱ
ϕ
\mathcal{F}_{\phi}
LTC
S
ψ
S_{\psi}
ℒ
pred
\mathcal{L}_{\mathrm{pred}}
✓
✓
ℛ
sigreg
\mathcal{R}_{\mathrm{sigreg}}
✓
ℒ
path
\mathcal{L}_{\mathrm{path}}
✓
✓
ℒ
mined
\mathcal{L}_{\mathrm{mined}}
✓
Table 1:
Direct gradient flow from each loss term.
Each epoch has two stages. First, we jointly optimize
Eq.
12
on offline expert batches: we construct goal-mismatched and endpoint-preserving perturbations to compute
ℒ
path
\mathcal{L}_{\mathrm{path}}
and, when the buffer is nonempty, sample closed-loop preferences to compute
ℒ
mined
\mathcal{L}_{\mathrm{mined}}
. We then disable gradients, perform closed-loop planning with the current model, and add newly discovered failed trajectories to
ℬ
\mathcal{B}
. Training alternates between these two stages.
Trajectory-Sensitive Candidate Scoring
During planning, LTC reads, as defined in
Eq.
2
, the context-augmented predicted latent trajectory
τ
^
\hat{\tau}
. To eliminate the scale difference between LTC output and endpoint distance, we first define the calibrated LTC score
S
~
ψ
​
(
τ
^
∣
g
)
=
IQR
⁡
(
C
0
)
max
⁡
(
IQR
⁡
(
S
ψ
)
,
ϵ
)
​
S
ψ
​
(
τ
^
∣
g
)
.
\widetilde{S}_{\psi}(\hat{\tau}\mid g)=\frac{\operatorname{IQR}(C_{0})}{\max\!\left(\operatorname{IQR}(S_{\psi}),\epsilon\right)}S_{\psi}(\hat{\tau}\mid g).
(13)
Here
ϵ
>
0
\epsilon>0
is a small constant for numerical stability.
IQR
⁡
(
x
)
=
Q
0.75
​
(
x
)
−
Q
0.25
​
(
x
)
\operatorname{IQR}(x)=Q_{0.75}(x)-Q_{0.25}(x)
denotes the interquartile range; the interquartile ranges of
C
0
C_{0}
and
S
ψ
S_{\psi}
are both estimated over the same set of candidates produced by endpoint-only CEM. The combined score is defined as
C
λ
​
(
τ
^
,
g
)
=
C
0
​
(
τ
^
,
g
)
+
λ
​
S
~
ψ
​
(
τ
^
∣
g
)
,
C_{\lambda}(\hat{\tau},g)=C_{0}(\hat{\tau},g)+\lambda\widetilde{S}_{\psi}(\hat{\tau}\mid g),
(14)
where
C
0
C_{0}
evaluates only the predicted endpoint,
S
~
ψ
\widetilde{S}_{\psi}
reads the complete predicted latent trajectory, and
λ
≥
0
\lambda\geq 0
controls the relative strength of the LTC signal. When
λ
=
0
\lambda=0
, candidates are ranked solely by the endpoint term, although the encoder may still have been shaped by trajectory preference training.
Thus, the combined score retains endpoint-based goal matching while incorporating information from complete predicted trajectories into candidate selection.
