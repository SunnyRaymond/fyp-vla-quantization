# Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure

paper_id: arxiv:2607.05966v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

World models have become a load-bearing component of recent embodied AI, serving as latent simulators for planning
[
3
,
4
,
5
]
and as generative environments for self-supervised learning
[
6
,
7
]
. A widely-noted failure mode is the deterioration of imagined rollouts over long horizons, conventionally attributed to
compounding error
[
8
]
.
This framing is accurate but underspecified: it does not distinguish what kind of error compounds or which feature dimensions deteriorate. Across four recent
Observations in driving VLM/VLA and trajectory-prediction benchmarks indicate that the deterioration carries a specific structural signature that the compounding-error framing obscures.
We adopt the classical mechanics distinction between kinematic and dynamic motion. We define
kinematic
as motion described purely through position, velocity, and acceleration time series, without invoking the forces or physical constraints that produced it. As
dynamic
, we define motion that requires those constraints (e.g., mass, friction, contact) to be reproduced correctly.
Central Claim
Current world models imagine
kinematically
rather than
dynamically
:
they extrapolate position-velocity-acceleration trajectories that are internally consistent with linear kinematic update rules, but inconsistent with the physical constraints that produce real motion.
Kinematic fallback is a third, structurally distinct account of long-horizon world-model failure, alongside the two positions that dominate the literature:
predictable-representation engineering
(the Dreamer line
[
5
]
) and
error-compounding bounds
(MBPO
[
8
]
). Predictable-representation engineering attributes long-horizon reliability to stable, predictable latent representations and pursues it through normalization and balancing techniques; a single algorithm with fixed hyperparameters across 150+ tasks empirically validates the position, yet it is silent on
what
those representations should contain. Error-compounding bounds derive an explicit quadratic-in-horizon bound on the gap between model-based and true returns under policy distribution shift, and respond pragmatically by limiting model trust to short branched rollouts from real states. A world model whose latent contains rich kinematic features but no dynamic features satisfies both: stable enough for Dreamer-line predictability, accurate enough inside the training regime for Janner-line short-rollout bounds, yet still biased toward kinematic continuation once conditioning pushes its rollouts across a physical-regime boundary. The three accounts are therefore not mutually exclusive but different layers of the same failure surface; because they predict distinct empirical signatures, the protocol of Section
III
can target the kinematic-fallback layer specifically.
We make three contributions. We (i)
recast long-horizon world-model failure in kinematic-vs-dynamic terms
, distinguishing a structural-content layer from the variance-engineering and error-compounding layers studied in prior work; (ii)
introduce imagined kinematic-consistency error (iKCE) together with a conditioning-perturbation protocol
that operationalize this account as a falsifiable evaluation diagnostic; and (iii)
instantiate the diagnostic on an open-weight checkpoint
(DreamerV3 on DMC walker-walk), where it exhibits both predicted signatures of kinematic imagination, with controls ruling out the principal confounds. The two signatures are a kinematic-null residual
∼
\sim
180
×
\times
above matched physics at
T
=
16
T{=}16
, and statistical invariance of the imagined rollouts to a friction sweep that crosses the empirical gait-collapse boundary. The diagnostic signature is this regime-invariance, not the absolute iKCE magnitude: a trivially kinematic predictor would produce zero iKCE.

## Method

Our diagnosis is motivated by four existing observations, each inconclusive alone and explained only in isolation by its original authors, but jointly forming a coherent structural signature.
(i) Representational diagnostic on driving VLMs and VLAs.
Schäfer et al. [11]
present EgoDyn-Bench, a video-QA diagnostic that decouples physical reasoning from visual perception. (i), the weighted physics consistency rate (WPCR) saturates with a single static frame: rising from
∼
\sim
20 with no visual input to
∼
\sim
97 with one frame and remaining essentially flat as additional frames are added or temporally shuffled. (ii), reintroducing video to a text-only baseline recovers only
∼
\sim
2.6pp on balanced accuracy under the best encoding, while text-only input already achieves 59.6% BAcc. The authors characterize this as a “functional decoupling between vision and language”: ego-motion understanding is derived almost exclusively from the language modality, with visual observations
Implication for world models is structural.
Contributing static context rather than temporal evidence. Imagined rollouts that depend on such encoders for motion-conditional features extrapolate from representations that under-encode the temporal dynamics they would need to imagine correctly.
(ii) Sensor-degraded behavioral diagnostic.
Priyadershi and Frtunikj [10]
stress-test Alpamayo R1, a 10B-parameter driving VLA, across 1,996 scenarios under eight sensor perturbations. Under heavy Gaussian noise (
σ
=
70
\sigma=70
), the authors characterize the failure mode as one where the trajectory decoder “fails via collapsing kinematic priors while the language branch continues producing coherent but safety-irrelevant explanations.”
Independent observation.
This is an independent observation of the same kinematic-fallback failure mode, in a third-party VLA under naturalistic sensor degradation rather than controlled diagnostic stimuli.
(iii) Open-loop trajectory-prediction baselines.
Zhai et al. [13]
train a 3-layer MLP that consumes only the ego vehicle’s kinematic state and matches perception-based end-to-end planners on nuScenes open-loop L2 (0.29 m vs. 0.37 m for VAD-Base), with no camera, LiDAR, or HD-map input. The authors read this as a benchmark artifact, attributing it to the trajectory distribution of nuScenes and to a coarse collision-evaluation grid, and call for rethinking the open-loop evaluation scheme. We accept the empirical finding but read its significance differently.
Third independent signature.
We read the same result as a third independent signature of kinematic imagination: when an ego-state-only predictor saturates the dominant open-loop metric, perception-based planners on this benchmark are not doing substantially more than kinematic extrapolation.
(iv) Physics-consistency scoring on fine-tuned VLAs.
Gao et al. [2]
introduce a Kinematic Consistency Error (KCE) that scores a trajectory by checking each predicted next position against a closed-form kinematic extrapolation of the current state, and reuse it as a training loss for a fine-tuned 4B VLA. On their style-conditioned benchmark, KCE shows no monotonic relationship to model scale or modality: the strongest generalist (Gemini-3-Pro, 0.06–0.11 m) and the smaller fine-tuned models (0.08–0.12 m) overlap, with no ordering by parameter count or sensor richness.
A data/training deficit, not a capacity one.
A deficit that is invariant to model scale and modality is unlikely to reflect a capacity limitation. It points instead to the training signal and data distribution rather than to model size.
Conclusion.
The four observations are concentrated in the driving and VLA setting but employ heterogeneous methodological registers, from representational probing to behavioral perturbation to physics-consistency scoring, and converge on a single structural deficit: the learned representation is dominated by kinematic features, and the dynamic features required for physical-regime-conditional behavior are systematically under-represented.
