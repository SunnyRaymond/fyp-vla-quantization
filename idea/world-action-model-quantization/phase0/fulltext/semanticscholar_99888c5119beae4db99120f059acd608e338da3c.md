# MV-WAM: Manifold-Aware World Action Model with Value Augmentation

paper_id: semanticscholar:99888c5119beae4db99120f059acd608e338da3c
tier: T2
source_used: html_arxiv
warning: none

## Intro

Generalization is fundamental to deploying robotic manipulation policies in the real world
Gao et al. (2025)
. Beyond the training distribution, policies must achieve visual generalization to unseen textures, lighting, backgrounds, and object configurations. Recent vision-language-action (VLA) models
Brohan et al. (2022)
;
Driess et al. (2023)
;
Kim et al. (2024)
, built upon large-scale vision-language pretraining
Radford et al. (2021b)
;
Alayrac et al. (2022)
;
Achiam et al. (2023)
;
Radford et al. (2021a)
, have substantially advanced manipulation policies in semantic and task-level generalization. However, this progress has not extended to visual and environmental generalization
Gao et al. (2025)
;
Shi et al. (2023)
,
as policies trained on limited demonstrations cannot capture the visual diversity of real-world deployment. World Action Models (WAMs) extend the VLA paradigm by further leveraging large-scale action-free video data which captures far richer visual diversity than demonstration data alone
Hafner et al. (2019)
;
Ha and Schmidhuber (2018)
;
Ye et al. (2026)
, broadening visual generalization to regimes that curated demonstrations cannot reach. Yet realizing this promise has proven non-trivial.
Existing WAMs differ chiefly in how they integrate video prior into policies
Bi et al. (2025)
;
Li et al. (2026)
;
Ye et al. (2026)
;
Hu et al. (2025)
;
Yuan et al. (2026)
.
The most loose coupling is achieved by
decoupled
approaches, where a separate video model and policy interact only through generated visual predictions or latent representations
Du et al. (2023)
;
Yang et al. (2024)
;
Hu et al. (2024b)
;
Tian et al. (2024)
.
Fully unified
approaches eliminate this distinction altogether, tokenizing visual and action signals into a single stream processed by shared transformer parameters
Cheang et al. (2024)
;
Zhu et al. (2024)
. Approaches based on
Mixture-of-Transformers (MoTs)
Liang et al. (2024)
go further still, introducing modality-specific experts while sharing global attention, and currently achieve the strongest in-domain performance among WAM designs.
Across these designs, all report consistent improvements over non-WAM baselines on in-domain and out-of-distribution (OOD) scenes alike. Despite these advances, as shown in Appendix
D.1
, in-domain performance steadily climbs while OOD gains lag behind, leaving a wide and in some cases growing generalization gap. Regardless of the architecture, the gains from video prior seem to flow disproportionately into visual prediction rather than action robustness, which is precisely what WAMs were designed to improve. Motivated by these observations, we aim to explore the following question:
Why does deeper video-action coupling not translate to greater action generalization?
We trace this gap to a manifold mismatch deeply embedded in current WAM
architectures. Visual observations are high-dimensional and densely structured,
governed by spatial and photometric regularities, whereas robot actions are
low-dimensional, temporally structured, and precision-critical over control
dimensions
Brohan et al. (2022)
;
Cheang et al. (2024)
. These two modalities thus inhabit
intrinsically heterogeneous representational manifolds
Yang et al. (2026)
. Joint optimization over
incompatible manifolds inevitably compromises action learning
Yu et al. (2020)
,
leaving action representations under-trained and brittle. This is further aggravated
by the sheer volume of visual tokens, which naturally dominates the shared objective.
This misalignment is masked in-domain by strong cross-modal correlations, but under
distribution shift, visual perturbations propagate through shared parameters and
degrade action prediction
Gao et al. (2025)
. This structural mismatch is
architecture-agnostic, explaining why neither decoupled, fully unified, nor MoT-based
designs have succeeded in closing the generalization gap.
As illustrated in Fig.
1
, we propose
MV-WAM
(Manifold-Aware World Action Model with Value Augmentation), a novel framework
that jointly generates future visual imaginations, robot actions, and value estimation within a unified architecture. At its core, MV-WAM adopts a
MoTs backbone
Liang et al. (2024)
in which each transformer layer hosts two modality-specific experts, one
processing visual tokens and the other processing action-value tokens.
Modality-specific experts share a global attention space while maintaining separate parameters.
The visual expert is initialized from a large-scale pretrained video
generative model
Chi et al. (2025b)
, allowing MV-WAM to inherit a rich visual
prior over physical dynamics, object interactions, and scene geometry that
demonstration data alone cannot provide. Built on this framework, three principled components collectively enable action-free video to serve as a direct source of policy generalization. First, a
modality-adaptive optimization
scheme extends the plain MoTs into a manifold-aware multi-objective MoTs formulation, with each modality supervised according to the intrinsic structure of its target manifold. This eliminates cross-modal optimization interference and alleviates gradient imbalance, preventing each modality’s learning signal from corrupting the other.
Second, a
cross-modality causal mask
hierarchically routes information across modalities, grounding each action token in its corresponding predicted visual frame and each value token in both visual and action contexts, ensuring that action generation is always conditioned on the predicted visual future while preserving modality-specific structure. Third, a
progress-valued regulation
mechanism uses value
tokens, trained with Monte Carlo task returns, to jointly estimate task
progress and detect visual-action misalignment during execution. During online execution, when
predicted values drop below a learned threshold, MV-WAM triggers a
value-guided rollback, enabling the policy to autonomously identify and
recover from execution deviations.
We evaluate MV-WAM extensively across simulation and real-world settings. On the RoboTwin 2.0
Chen et al. (2025)
benchmark,
covering 50 manipulation tasks across both clean scenes
and randomized scenes with perturbed lighting, backgrounds,
and object configurations. MV-WAM achieves state-of-the-art performance under
both settings, reaching a 55.7% success rate on unseen scenarios
and outperforming the strongest baseline by 29.3%
while matching its in-domain performance. On real-world dual-arm manipulation, MV-WAM further achieves 77.5% mean success rate across four tasks of varying difficulty, where baseline methods fail to generalize to physical deployment. Notably, MV-WAM achieves these gains with significantly fewer parameters than competing WAMs, suggesting that manifold-aware design rather than model scale is the key driver of generalization. Ablation studies further confirm
that each architectural component contributes substantively to the OOD
performance.
Our contributions are summarized as follows.
(1) MV-WAM framework.
We propose MV-WAM, a unified world action
model built on MoTs that jointly generates visual predictions, robot actions,
and value estimates, with a manifold-aware multi-objective formulation and
a cross-modality causal mask that ground action generation in predicted
visual futures while respecting the intrinsic geometry of each modality.
(2) Progress-valued regulation.
We introduce a progress-valued
regulation mechanism that estimates task completion and detects visual-action
misalignment, enabling autonomous deviation
detection and value-guided recovery without human intervention.
(3) State-of-the-art generalization.
Extensive experiments on
simulated and real-world manipulation tasks demonstrate state-of-the-art
performance across in-domain, zero-shot, and few-shot settings.

## Method

Figure 2
:
Overview of
MV-WAM
. (a) Detailed architecture comprises a Video Expert and an Action-Value Expert. (b) Manifold-aware Training. (c) Two-stage training, video-only pretraining followed by joint video-action training. (d) Value-guided rollback for online execution correction.
3.1
Novel End-to-End Framework
To close the generalization gap arising from the structural mismatch between visual and action modalities, we propose a Manifold-Aware Unified Framework named
MV-WAM
, which unifies video world modeling, robotic action prediction, and value estimation within a single Transformer-based diffusion backbone. As illustrated in Fig.
2
, our model is built upon stacked Diffusion Transformer (DiT) blocks comprising a Video Expert and an Action-Value Expert, coupled through a Mixture-of-Transformers (MoTs)
Liang et al. (2024)
mechanism that enables cross-modal interaction. Rather than adopting the plain MoTs, we extend it into a manifold-aware multi-objective MoTs.
Dual Branch Expert architecture
Both the Video Expert and Action-Value Expert adopt DiT backbones with identical depth and standard internal configurations. Task instructions
ℒ
\mathcal{L}
are uniformly encoded via the umT5
Chung et al. (2023)
text encoder and injected into both experts through cross-attention. Specifically, the Video Expert is instantiated based on WoW-1.3B
Chi et al. (2025b)
. It first employs a spatiotemporal VAE to compress raw visual observation frames into compact latent representations, which are concatenated with mask tokens along the channel dimension, patched into visual tokens, and fed into the DiT backbone for video diffusion modeling. In contrast, the Action-Value Expert takes multi-view visual states
𝒔
t
\boldsymbol{s}_{t}
encoded by SigLIP2
Tschannen et al. (2025)
as visual condition, with language embeddings injected into odd transformer layers and SigLIP2 visual embeddings into even layers. Within this expert, action tokens and state tokens share a unified encoder-decoder architecture for action generation, while value tokens are processed by an independent encoder-decoder for separate value estimation.
For precise temporal alignment during joint modeling, the Rotary Positional Embedding (RoPE) scaling factor of the Action-Value Expert is set to
1
/
4
1/4
of that used in the Video Expert. This synchronization establishes a unified temporal grid between action tokens and video latents, enabling temporally coherent cross-modal interaction. We further adopt a structured causal attention mask to regulate information flow while preserving modality-specific priors. Video tokens are restricted to self-attention only within the visual stream to maintain high-fidelity video generation, whereas action and state tokens can attend to both video context and other action tokens. value tokens attend to visual, action, and state tokens jointly, enabling holistic assessment of task progress and visual-action alignment. This asymmetric masking ensures that action generation is always conditioned on visual predictions while video generation remains uncontaminated by action signals, and that value estimation has access to the full cross-modal context necessary for reliable task assessment.
Manifold-Aware Dual Prediction Objectives
Instead of adopting a single unified diffusion objective for both modalities, we tailor two distinct manifold-aware prediction losses according to their intrinsic geometric properties. Visual and action data reside on Riemannian manifolds with significantly different curvatures, and optimizing both with one shared objective leads to severe geometric mismatch and suboptimal generalization. Our framework adopts separate flow-matching transformation functions for each branch. The Video Expert, defined on the high-curvature visual manifold, directly predicts the velocity field following standard flow matching:
ℒ
v
=
𝔼
τ
,
𝒛
0
,
𝒛
1
​
‖
π
θ
v
​
(
𝒛
τ
,
τ
,
𝒄
)
−
(
𝒛
1
−
𝒛
0
)
‖
2
2
,
\mathcal{L}_{v}=\mathbb{E}_{\tau,\boldsymbol{z}_{0},\boldsymbol{z}_{1}}\left\|\pi_{\theta}^{v}(\boldsymbol{z}_{\tau},\tau,\boldsymbol{c})-(\boldsymbol{z}_{1}-\boldsymbol{z}_{0})\right\|_{2}^{2},
(2)
where
𝒛
0
\boldsymbol{z}_{0}
denotes clean video latents,
𝒛
1
\boldsymbol{z}_{1}
is the noise prior, and
𝒛
τ
=
(
1
−
τ
)
​
𝒛
0
+
t
​
𝒛
1
\boldsymbol{z}_{\tau}=(1-\tau)\boldsymbol{z}_{0}+t\boldsymbol{z}_{1}
.
π
θ
v
\pi_{\theta}^{v}
and
π
θ
a
\pi_{\theta}^{a}
denote the Video Expert and Action-Value Expert respectively.
For the Action-Value Expert, both action and value tokens reside on low-curvature manifolds with similar geometric structure. Using action as a representative example, the optimization objective is equivalent to directly regressing the clean action latent
𝒂
0
\boldsymbol{a}_{0}
:
ℒ
a
=
𝔼
τ
,
𝒂
0
,
𝒂
1
​
‖
π
θ
a
​
(
𝒂
τ
,
τ
,
𝒄
)
−
𝒂
0
‖
2
2
.
\mathcal{L}_{a}=\mathbb{E}_{\tau,\boldsymbol{a}_{0},\boldsymbol{a}_{1}}\left\|\pi_{\theta}^{a}(\boldsymbol{a}_{\tau},\tau,\boldsymbol{c})-\boldsymbol{a}_{0}\right\|_{2}^{2}.
(3)
The entire framework is jointly optimized by
ℒ
=
λ
v
​
ℒ
v
+
λ
a
​
ℒ
a
\mathcal{L}=\lambda_{v}\mathcal{L}_{v}+\lambda_{a}\mathcal{L}_{a}
, where all balancing weights
λ
v
\lambda_{v}
and
λ
a
\lambda_{a}
are empirically set to 1. During inference, the Video Expert denoises latents from the predicted velocity field, while the Action-Value Expert directly outputs the optimized action representation. Benefiting from the manifold-aware dual objectives, MV-WAM simultaneously achieves high-fidelity visual world modeling and robust robotic action generation.
3.2
Progress-Valued Regulation
While manifold-aware objectives ensure geometric alignment during training, they provide no explicit signal to assess task progress or detect visual-action misalignment at inference time. In the absence of such feedback, execution deviations remain undetected and unrecoverable, directly undermining the policy’s generalization under distribution shift. To this end, we introduce a
Progress-Valued Regulation
mechanism that equips the Action-Value Expert with a dedicated value prediction capability, enabling the model to bias denoising trajectories toward high-progress outcomes.
Monte Carlo Value Estimation.
Inspired by the paradigm of Cosmos Policy
Kim et al. (2026)
, we estimate the task value function
c
c
with Monte Carlo (MC) returns computed from executed trajectories. For each state-action pair in a trajectory, the value target is defined as the empirical discounted return accumulated from the current step
t
t
to the end of the trajectory:
c
^
​
(
s
t
,
a
t
)
=
∑
i
=
0
H
−
t
γ
i
​
r
​
(
s
t
+
i
,
a
t
+
i
)
,
\hat{c}(s_{t},a_{t})=\sum_{i=0}^{H-t}\gamma^{i}r(s_{t+i},a_{t+i}),
(4)
where
H
H
is the trajectory horizon and
γ
\gamma
is the discount factor(we set
γ
=
0.99
\gamma=0.99
). This MC target converts sparse task outcomes into a dense progress signal along the demonstration trajectory, serving as the supervision target for the value token.
Value-Guided Rollback for Online Execution.
During closed-loop execution, the predicted value token serves as a dynamic progress monitor to detect and correct execution deviations in real-time. Specifically, we maintain a rolling buffer of recent state-Action-Value triplets and track the peak value
c
peak
=
max
i
<
t
⁡
c
^
i
c_{\text{peak}}=\max_{i<t}\hat{c}_{i}
achieved so far. At each control step, if the newly predicted value drops significantly below this peak (i.e.,
c
^
t
<
c
peak
−
δ
\hat{c}_{t}<c_{\text{peak}}-\delta
, where
δ
\delta
is a task-aware threshold), the system flags the current action chunk as potentially erroneous. Upon triggering, the policy initiates a
rollback-and-resample
protocol: it reverts the execution state to the cached checkpoint corresponding to
c
peak
c_{\text{peak}}
, effectively discarding the faulty action sequence. From this restored high-value state, the diffusion process is re-initialized to conditionally resample a new action chunk. Leveraging the inherent stochasticity of diffusion sampling, this correction can be performed efficiently without regenerating the entire trajectory or requiring additional environment queries. This value-triggered backtracking mechanism acts as a self-correcting loop that prevents error accumulation during long-horizon tasks, ensuring that the model consistently recovers from execution deviations and navigates toward high-progress regions of the action manifold.
