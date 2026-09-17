# Low-Rank Dynamics-Effective Latent Carriers for Counterfactual Rollout in Learned World Models

paper_id: arxiv:2608.15156v3
tier: T3
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

World models have progressed from compact learned simulators (Ha and Schmidhuber 2018) and latent 
planning (Hafner et al. 2019) to imagination-based control (Hafner et al. 2023) , generative interactive 
environments (Bruce et al. 2024) , and action-conditioned video models for planning (Assran et al. 2025). 
This advance makes latent state itself a scientific object: not only whether a model predicts, but what its 
internal state retains and whether that information can guide future behavior. 
The key question is whether the latent state can support the intended counterfactual behavior, not whether 
its coordinates have an obvious physical interpretation. DeepMDP formalized latent compression that 
preserves transition and reward structure relevant to control (Gelada et al. 2019). Recent work argues that 
latent states should be judged by sufficiency for a specified function, such as prediction, planning, 
memory, grounding, or counterfactual reasoning (K.W. Kim 2026), and that world models should 
preserve distinctions needed to answer intervention queries (Thorpe et al. 2026). Counterfactual behavior 
therefore provides a stricter test than decodability alone. 

* Equal contribution 
Source codes and data can be found at  https://github.com/lysea8282/dynamic-effective-latent-carriers 

Physical dynamics sharpen the problem. Physion and related benchmarks show that learned models can 
make useful physical predictions (Bear et al. 2021), but behavioral success does not reveal how task-
relevant information is organized internally. A decodable physical quantity need not be used by the 
transition dynamics, while functionally relevant state may be distributed across hidden coordinates. 
A useful analogy is coarse-graining: complex systems can admit coarser predictive descriptions, as in 
rigorous micro-to-macro limits such as hard-sphere dynamics to fluid equations (Deng, Hani, and Ma 
2025). The analogy is inspiring but has limits. Neural latent coordinates are learned and non-identifiable, 
so a useful intervention interface need not align with named physical variables or a fixed set of human-
interpretable axes. 
Recent interpretability work makes this distinction concrete. Distributed Alignment Search allows causal 
variables to align with distributed neural representations (Geiger et al. 2024) , while causal-probing work 
emphasizes completeness and selectivity (Canby et al. 2025). In world and video models, physical 
information has been probed in distributed representations (Joseph et al. 2026), probe-derived hidden-
state shifts can alter predictions (Zhang 2026), and Concept Activation Vectors can steer physical 
plausibility judgments (Alam 2026). These results establish that physical information can be readable and 
steerable, but they do not by themselves show that an edited hidden state can autonomously generate the 
intended future. 
Our central question then asks: can a limited hidden-state intervention place a recurrent world model on 
an intended counterfactual trajectory and then let the model's own dynamics continue it? We require a 
one-shot edit to produce a sustained autonomous rollout that passes registered specificity and control tests, 
rather than merely changing an immediate readout or one-step prediction. 
We study a controlled 192-dimensional recurrent world model of two-object motion in a deterministic two-
dimensional collision environment. The primary model family is trained on factual trajectories together with 
locally edited counterfactual trajectories. We call bounded edits to one velocity component “Single” and same-
object edits to two velocity components “Joint.” After verifying that the model can natively roll out the edited 
state, we fit checkpoint-specific candidate carriers from training-only counterfactual-minus-factual hidden 
differences and an affine map from the factual state and requested edit to carrier coefficients. At test time, the 
predicted patch is applied once, followed by twelve autonomous transitions without future observations, 
teacher forcing, repeated hidden-state correction, or physical-state clamping. 
For Single edits, rank 4 is the smallest rank on the preregistered grid that satisfies all development criteria. 
The frozen procedure replicates on independently trained checkpoints, remains effective at nearby anchors, 
and passes the preregistered specificity and integrity controls. We therefore interpret the rank-4 carrier as a 
compact, checkpoint-specific intervention interface for sustained counterfactual rollout—not as the model's 
intrinsic state dimension. 
The same Single-derived carrier and Single-only affine map also support bounded same-object Joint requests, 
even though no Joint examples are used to fit the map. Across the matched training regimes, broader 
counterfactual support was associated mainly with better Joint rollout accuracy and more additive Joint hidden 
responses. This additive structure is partly, but not entirely, captured by the rank-4 subspace, and the effect of 
the patch quickly spreads through the rest of the hidden state. A position-edit stress test fails the required 
specificity controls. Together, these results support a compact intervention-entry interface rather than a closed 
four-dimensional dynamical state.

## Method

This section defines the intervention problem and the notation used throughout the paper. We first 
consider bounded Single edits, which define the confirmatory rank-selection and replication assays, and 
then a same-object Joint extension. Single denotes an edit to one velocity component of one object; Joint 
denotes simultaneous edits to both velocity components of the same object. Formal definitions are 
collected in Appendix A, with environment, model, and training details in Appendix B. 
3.1 World-model setting 
We study two circular objects moving in a bounded, deterministic two-dimensional arena. Each object 
has a two-dimensional position and velocity, and the model receives noisy numeric object-state 
observations. All trajectories are generated with this simulator; no external physics dataset or pretrained 
trajectory dataset is used. This controlled setting gives an exact simulator-grounded counterfactual
