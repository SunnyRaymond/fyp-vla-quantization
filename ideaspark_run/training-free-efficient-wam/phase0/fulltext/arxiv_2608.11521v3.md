# Keep the Future, Drop the Rollout: RIFT for World Action Models

paper_id: arxiv:2608.11521v3
tier: T3
source_used: html_arxiv
warning: none

## Intro

Figure 1:
Success rate vs. deployment latency.
Points show mean success rates, and
error bars indicate
±
1
\pm 1
standard deviation across three evaluation runs per method.
Latency is measured in milliseconds per action chunk on one A800.
Grey squares
denote current-only methods;
blue circles
denote methods that read a future representation produced by rollout.
Rift
achieves the highest success rate with latency close to the minimum.
World action models couple future video prediction with robot control. A single network predicts a short video and conditions its next actions on that prediction
(
Yuan et al. 2026
;
Li et al. 2026
;
Bi et al. 2025
)
. In matched settings, rollout-based policies achieve higher success than current-only variants. However, iterative video generation makes rollout-based systems incur
3.3
×
3.3\times
to
9.6
×
9.6\times
the latency of current-only deployment (
fig.
1
).
Fast-WAM drops the future branch after future-prediction co-training
(
Yuan et al. 2026
)
, whereas PFD distills a future-conditioned correction into the current-only path
(
Fang, Chen, and Cai 2026
)
. Both reduce latency but retain a gap to policies that explicitly read a generated future. Yet the action expert consumes a future representation, whereas iterative rollout is the process that constructs it. Existing comparisons change both the availability of this representation and the process used to construct it, so their separate contributions remain unresolved. We therefore ask whether a WAM can preserve explicit test-time future conditioning without iterative video rollout.
To separate these factors, we examine the future cache, which contains the keys and values of future tokens at each layer. Joint co-denoising updates this cache at every denoising step, whereas the generate-then-act inverse dynamics model (IDM) uses a fixed final-clean cache computed from the fully denoised future video. Their similar success suggests that the representation, rather than its iterative trajectory, may provide the shared benefit. Success alone cannot reveal what the action reads.
We therefore intervene on this cache. The attention mask prevents video tokens from attending to action tokens, so the cache is an action-independent intervention site. We record it and replay action denoising with non-target inputs fixed, masking attention to future tokens, editing future values under the recorded keys, or replacing both keys and values with one fixed final-clean cache. Each closed-loop intervention uses the same initial state and policy seed as the unmodified policy. We measure success rate (SR) and end-effector average displacement error (EE-ADE), the average drift from the unmodified end-effector trajectory.
Across
2,000
2{,}000
paired trials on all 40 LIBERO tasks, masking yields
18.7
18.7
cm EE-ADE and reduces success from
98.4
%
98.4\%
to
9.7
%
9.7\%
. Spatial permutation and temporal swapping yield similar EE-ADE (
14.3
14.3
and
15.6
15.6
cm) but sharply different success (
65.2
%
65.2\%
and
0.7
%
0.7\%
), showing that the action reads future content at its assigned positions. In contrast, reusing one fixed final-clean K/V cache yields only
1.9
1.9
cm EE-ADE and
97.9
%
97.9\%
success. Actions depend on future content and its organization, yet one fixed cache nearly preserves execution.
This cache still requires iterative video generation. We therefore ask whether one forward pass can produce an effective, complete future cache. We propose
Rift
(
Rollout-free Imagination via Future Tokens
), which places learned anticipation tokens at future temporal positions and fills their per-layer K/V cache with one video-backbone pass. We shape the anticipation states with conditional flow matching, using a distributional objective rather than direct L2 regression to the single observed future. Deployment requires one cache prefill followed by the ordinary action flow, without video diffusion or video decoding at test time.
On LIBERO,
Rift
achieves
98.8
%
98.8\%
overall success with a
3.1
3.1
–
9.2
×
9.2\times
inference speedup over the evaluated rollout-based baselines. Without further training, it achieves
81.1
%
81.1\%
overall success on LIBERO-Plus, a
+
9.7
+9.7
percentage-point improvement over the strongest evaluated baseline. On RoboTwin 2.0, it achieves the highest observed success rates among the evaluated methods in both clean and randomized scenes. On three real-world tasks,
Rift
achieves
45.3
%
45.3\%
average success, a
+
6.0
+6.0
percentage-point improvement over Fast-WAM-Joint, with
60.7
%
60.7\%
lower action-chunk latency.
We make three contributions.
1.
We introduce a paired closed-loop intervention protocol
to measure how future caches affect executed trajectories and task success.
2.
We show that actions are sensitive to future content and its spatial and temporal organization.
Yet one fixed final-clean K/V cache nearly preserves execution in the evaluated co-denoising settings.
3.
We develop a rollout-free future interface.
Rift
constructs a complete future K/V cache in one backbone pass and achieves higher overall success than the evaluated baselines across three simulation benchmarks and three real-world tasks. Its latency remains close to current-only Fast-WAM on LIBERO and the real robot.

## Method

Figure 2:
A fixed final-clean K/V cache nearly preserves Oracle execution.
Masking attention to future tokens or perturbing future values alters trajectories and reduces success across four WAMs.
For Joint and Cosmos-2, one final-clean K/V cache obtained from rollout is reused throughout action denoising, yielding trajectories and success rates close to Oracle.
Oracle denotes each model’s unmodified policy.
Bars show end-effector trajectory deviation from Oracle, with success rates reported on the right.
N/A marks structurally redundant or unsupported interventions.
3.1
The channel we edit
With paired closed-loop interventions, we test whether actions depend on the future cache,
whether future values must stay at assigned positions, and whether the complete K/V cache must evolve
during denoising. Each model–intervention estimate uses
2,000
2{,}000
paired trials across all
40
40
LIBERO tasks.
Figure
2
reports executed
EE-ADE and success rate.
Fast-WAM-Joint, Fast-WAM-IDM, Cosmos Policy, and LingBot-VA construct futures
differently but all provide per-layer keys and values associated with future tokens. We call these the
future cache
;
Oracle
denotes the unmodified checkpoint used as the reference for interventions.
Because video tokens don’t attend to action tokens, the cache is action-independent
given observation (
o
o
), language (
l
l
), and video-generation randomness. This property supports record and replay. The record
pass generates video normally and stores per-layer future K/V at every action-denoising step. Value
corruptions retain the recorded key trajectory and edit only the matched future values. The
final-clean control instead takes the final clean future from Oracle’s iterative generation,
prefills its complete K/V once, and reuses that fixed cache at every action-denoising step. All
non-target inputs remain fixed. Exact replay of the unedited trajectory reproduces
Oracle
. Shuffle and noise alter only future values, with recorded keys fixed.
We interpret these out-of-distribution perturbations alongside the frozen-present and final-clean K/V controls.
3.2
Scoring each edit
Action chunks are not directly comparable across architectures, so we execute paired
policies. For episode
i
i
, intervention
I
I
and
Oracle
share the initial state and policy seed. Let
𝐱
i
,
t
I
\mathbf{x}^{I}_{i,t}
and
𝐱
i
,
t
O
\mathbf{x}^{O}_{i,t}
denote their recorded end-effector positions after environment step
t
t
. EE-ADE averages their distance over the common executed prefix
T
i
T_{i}
:
EE
​
-
​
ADE
i
⁡
(
I
)
=
1
T
i
​
∑
t
=
1
T
i
‖
𝐱
i
,
t
I
−
𝐱
i
,
t
O
‖
2
\operatorname{EE\text{-}ADE}_{i}(I)=\frac{1}{T_{i}}\sum_{t=1}^{T_{i}}\left\lVert\mathbf{x}^{I}_{i,t}-\mathbf{x}^{O}_{i,t}\right\rVert_{2}
(1)
We exclude the reset pose and use recorded simulator positions rather than integrating predicted actions. When runs end at different times, only their common prefix contributes. We first average EE-ADE over episodes within each task, then average the task means with equal weight.
Appendix
B
defines the bootstrap procedure that resamples tasks. EE-ADE measures drift from
Oracle
; success measures task completion.
High EE-ADE with a similar success rate indicates altered end-effector motion at comparable task performance.
When the success rate also decreases, the motion deviation is accompanied by reduced task success.
3.3
Intervention set
The interventions map directly to the three questions. Masking blocks attention to future tokens;
norm-matched noise replaces future values; and frozen-present supplies plausible, non-predictive values.
Spatial shuffle permutes values within frames, temporal swap exchanges value frames, and final-clean
replay replaces the evolving complete cache with the same final-clean K/V at every action step. We apply each supported edit to Fast-WAM-Joint, Fast-WAM-IDM, Cosmos Policy, and LingBot-VA, comparing each with its Oracle reference.
Claims are therefore within-model; cross-model magnitudes remain descriptive because architectures
and checkpoints differ.
3.4
Finding 1: WAM action experts use future values at their assigned positions
Across all four WAMs, masking yields
11.8
11.8
–
20.4
20.4
cm EE-ADE and reduces success
from
98.4
98.4
–
98.6
%
98.6\%
to
0.0
0.0
–
32.0
%
32.0\%
. Under recorded keys, noise yields
10.5
10.5
–
17.9
17.9
cm and at most
40.9
%
40.9\%
success, while frozen-present values yield
18.1
18.1
–
21.3
21.3
cm and at most
6.5
%
6.5\%
. These within-model effects show that action experts use
meaningful future values.
Position also matters: spatial shuffle yields
5.0
5.0
–
19.8
19.8
cm and
0.0
0.0
–
84.5
%
84.5\%
success across all four, while temporal swap yields
15.6
15.6
–
16.3
16.3
cm and
0.0
0.0
–
69.0
%
69.0\%
on Joint, IDM, and LingBot-VA. Every supported edit changes execution and lowers
own-model success, so future values are not an unordered pool. Severity differs by interface:
temporal swap is worse for Joint and IDM, spatial shuffle for LingBot-VA, and Cosmos-2 exposes no
temporal swap. Thus, position sensitivity is shared, not a universal severity ordering.
3.5
Finding 2: One fixed final-clean K/V cache nearly preserves execution
Finding 1 establishes content and position sensitivity, not whether the complete
cache must evolve. Where supported, final-clean replay replaces the entire future-cache trajectory
with one final-clean cache, holding both K and V fixed at every action-denoising step.
For Joint and Cosmos-2, final-clean K/V replay gives
1.9
/
1.7
1.9/1.7
cm EE-ADE and
97.9
/
98.2
%
97.9/98.2\%
success; the corresponding Oracle policies reach
98.4
/
98.4
%
98.4/98.4\%
. These are the smallest nonzero EE-ADEs
among compatible edits. IDM and LingBot-VA already expose one fixed final-clean K/V cache, so this
replay is identical by construction and structurally N/A.
The fixed cache nearly preserves Oracle execution in Joint and Cosmos-2,
but still requires iterative video generation.
The full-cache intervention also does not isolate the separate
contributions of keys and values.
Section
4
tests whether a single forward pass can construct
the complete future K/V cache used by the action expert.
