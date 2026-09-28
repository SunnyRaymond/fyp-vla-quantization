# OnlineCache: Learning Dynamic Caching Policies with Error Correction for Efficient Diffusion Inference

paper_id: semanticscholar:24aa6f3f5aefcd2463ee56a14077f2bcf11c3fcb
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

In recent years, diffusion models (Sohl-Dickstein et al., 2015; Song and Ermon, 2019; Ho et al.,
2020; Dhariwal and Nichol, 2021) have emerged as a dominant paradigm in generative AI, delivering
unparalleled synthesis quality across a wide range of modalities, including images (Rombach et al.,
2022), videos (Blattmann et al., 2023), audio (Kong et al., 2021), and 3D content (Poole et al.,
2023). This success is largely attributed to the adoption of highly scalable transformer-based
backbones (Peebles and Xie, 2023). However, such performance gains come at the cost of substantial
computational overhead. As these models become increasingly complex and large-scale, they exhibit
slow inference and high deployment costs, posing serious challenges for real-time applications.
To mitigate these computational bottlenecks, various acceleration techniques have been explored in
recent years. Most existing approaches rely on modifying the underlying model, such as network
pruning (Fang et al., 2023), low-bit quantization (Shang et al., 2023), and knowledge distillation (Luh-
man and Luhman, 2021), all of which reduce inference cost by altering model structure or parameter
∗Equal contribution.
†Corresponding author.
Preprint.
arXiv:2607.29398v1  [cs.LG]  31 Jul 2026

(a) Sample-level heterogeneity. OnlineCache dynamically
allocates computation across prompts of varying difficulty.
(b) Timestep-level heterogeneity. OnlineCache adaptively
identifies timesteps that are suitable for caching.
Figure 1: Comparison of static strategy (ERTACache) and ours (OnlineCache) in different senarios of diffusion acceleration.
(a) Under different prompt complexities, ERTACache adopts a uniform caching strategy, whereas OnlineCache dynamically
adapts (e.g., 70.00% vs. 53.33% cache ratio), achieving improved generation quality under a comparable average speedup. (b)
Under the same 66.67% caching ratio, ERTACache’s "one-policy-fits-all" strategy is suboptimal. In contrast, OnlineCache
identifies critical timesteps and achieves a 37.48% reduction in L1 error, along with substantially improved perceptual quality.
representation. In contrast, an emerging line of work focuses on cache-based acceleration, which
exploits the temporal redundancy across diffusion steps by reusing intermediate representations (Ma
et al., 2024a,b; Chen et al., 2024). Owing to its model-agnostic nature and minimal architectural
intervention, cache-based methods offer a complementary and increasingly attractive alternative.
However, most existing cache-based methods rely on static or heuristic-driven rules to determine
intermediate feature reuse, which fundamentally constrains their flexibility and performance. Static
strategies such as ERTACache (Peng et al., 2025) adopt a fixed caching schedule—applying the
same set of cached steps to all samples regardless of input complexity. More adaptive methods like
TeaCache (Liu et al., 2025a), which utilize heuristic threshold, still remain suboptimal due to their
greedy, local decisions that lack global quality awareness. Through an empirical analysis of diffusion
trajectories (as evidenced in Figure 1), we argue that such rigid, rule-based policies are insufficient to
handle the intrinsic heterogeneity of the denoising process, which manifests at two levels:
▷Sample-level heterogeneity: Figure 1a demonstrates generation difficulty varies significantly
across prompts. While static method allocates computation uniformly, our dynamic policy adap-
tively allocates resources across samples, assigning more computation to difficult cases and less to
easy ones, thereby achieving better performance under a comparable average speedup.
▷Timestep-level heterogeneity: Figure 1b shows that error sensitivity fluctuates along the denoising
trajectory. Static method inevitably caches sensitive steps while wasting computation on redundant
ones, resulting in suboptimal quality. In contrast, OnlineCache identifies critical timesteps and
significantly reduces overall errors (37.48% L1 reduction; LPIPS improves from 0.3920 to 0.2179).
These observations reveal a fundamental mismatch between rigid scheduling and complex denoising.
We argue that effective acceleration should move beyond fixed rules and adopt a dynamic, instance-
aware policy. To this end, we propose OnlineCache, a framework that formulates the caching
schedule as a sequential decision-making problem. Instead of relying on heuristics, we employ a
lightweight policy network to evaluate latent states at each denoising timestep and decide whether
to reuse cached representations or perform full computation. The policy is optimized via policy
gradient (Sutton et al., 1999) to maximize a trajectory-level reward, endowing the model with a global
perspective for efficient allocation of computational budgets. Specifically, we introduce two variants
of OnlineCache: (i) a vanilla variant that trains a standalone policy network to learn adaptive caching
decisions, and (ii) a bilevel optimization (BLO) variant that jointly trains the policy while optimizing
an error corrector to compensate for caching-induced trajectory deviations.
In the BLO variant, policy optimization is formulated as the outer objective to effectively balance the
trade-off between acceleration and generation quality, while the error corrector is optimized as the
inner objective to minimize local error. This bilevel coupling ensures that the policy learns to skip
computation strictly conditional on the corrector’s capacity to repair the induced error, leading to a
2

Numerical Integration:
Next Timestep  
(until 
)
Current Noisy
Sample 
Timestep 
State 
 Preparation
Statistical Feature Extraction
, 
, 
Concatenate
Classifier
Reuse 
  Corrector 
(Cache)
Corrected Residual:
Diffusion Model
(Compute)    Update Cache

Compute Residual:
Output:
Figure 2: Overall inference pipeline of OnlineCache, where the policy network dynamically determines caching decisions
and the corrector compensates for cache-induced approximation errors.
superior Pareto frontier between efficiency and quality. Moreover, both the policy and the corrector
are designed as lightweight MLPs, introducing negligible inference-time latency in practice.
Our core contributions can be summarized as follows:
▷We propose OnlineCache, a dynamic, instance-aware cache-based acceleration framework, with
two variants: a policy-only vanilla model and a bilevel optimization model that jointly trains policy
with an error corrector, enabling a globally effective speed-quality trade-off.
▷We develop a practical training pipeline for OnlineCache, incorporating multiple optimization
strategies and engineering techniques to ensure robust convergence, stable joint training, and
efficient plug-and-play inference, without any architectural modifications to the backbone. Code
and trained policy networks for several mainstream diffusion models will be released.
▷We conduct extensive experiments and ablation studies to validate the effectiveness of our approach.
OnlineCache accelerates 1.88× on DIT-XL/2 while maintaining competitive FID relative to full-
step inference. On FLUX.1-DEV, it attains nearly 3× speedup with consistently high generation
quality, surpassing prior cache-based methods. It also extends seamlessly to video generation.
2
Related Works
Cache-based Acceleration.
Cache-based acceleration leverages temporal redundancy across diffu-
sion steps to reduce inference cost without modifying the backbone model. Early methods mostly
adopt fixed caching schedules or architecture-specific reuse, such as periodic feature reuse in Deep-
Cache (Ma et al., 2024a) and FORA (Selvaraju et al., 2024). To improve robustness, later methods
introduce calibration or correction mechanisms, including Block Caching (Wimbauer et al., 2024),
∆-DiT (Chen et al., 2024), and ERTACache (Peng et al., 2025). More recent work moves toward
dynamic and learning-based policies, exemplified by TeaCache (Liu et al., 2025a), DiCache (Bu et al.,
2025), and router-based or compensation-based approaches (Ma et al., 2024b; Liu et al., 2025b). Our

## Method

Beyond caching, diffusion acceleration has also been explored through pruning, quantization, distilla-
tion, and improved samplers. We refer readers to Appendix B for a more comprehensive discussion.
3
Methodology
In this section, we detail OnlineCache, a dynamic caching framework for diffusion acceleration.
We formulate the caching decision as a sequential decision-making problem and solve it via Policy
Gradient (Sutton et al., 1999). Specifically, we introduce two versions: (i) a vanilla variant trains a
separate policy network and (ii) a bilevel optimization variant that jointly trains the policy while
tuning an error corrector for caching-induced deviations.
3.1
Problem Formulation
We consider a diffusion-based generative process that progressively denoises a latent variable from
xN to x0 over a sequence of timesteps {tN, . . . , t1}. During inference, a diffusion model Mdiff takes
3

the current noisy sample xt at timestep t and predicts the next-step intermediate state. Here, we
take the v-prediction parameterization as an example, while the x- and ϵ-prediction variants follow
analogously. Specifically, the model output can be formulated in a residual form as:
vt = Mdiff(xt, t) = xt + rt,
(1)
where rt = vt −xt denotes the residual at timestep t. The sample at the next timestep xt−1 is
then obtained by applying a numerical integration step: xt−1 = xt −vt · ∆t, where ∆t denotes the
integration step size. Previous works, such as ERTACache (Peng et al., 2025), have consistently shown
that in diffusion process, the residual feature rt often evolves smoothly across adjacent timesteps.
This temporal redundancy suggests that rt can be well approximated by a previously computed and
cached residual from a recent timestep, denoted as rcached.
We introduce a binary variable at ∈{0, 1} per timestep: at = 1 signifies caching (skip the heavy
computation and reuse rcached), and at = 0 signifies computing. The update rule is defined as:
vt = xt + ˆrt,
where ˆrt =
rcached,
if at = 1,
Mdiff(xt, t) −xt,
if at = 0.
(2)
When at = 0, we update the cache with the exact residual: ˆrt →rcached. This mechanism allows us
to skip the expensive computation of Mdiff entirely when the cached residual is deemed sufficient.
3.2
Policy Network Architecture and Design
The decision to reuse residuals is non-trivial: aggressive caching may cause error accumulation and
quality degradation, while conservative reuse diminishes efficiency gains. To balance these factors,
we formulate caching as a Markov Decision Process (MDP) and train a lightweight policy πθ to
maximize a reward signal balancing quality and speedup. Crucially, the effectiveness of the policy
depends on a well-designed state representation input, guided by the following two key observations:
▷Statistical Sufficiency. Latent features during diffusion often exhibit Gaussian-like behavior,
making the channel-wise mean µ(·) and standard deviation σ(·) informative global summaries.
This is theoretically grounded in Latent Diffusion Models, where the VAE imposes a KL constraint
that regularizes the latent space toward N(0, I); combined with the Gaussian initialization at xT ,
intermediate states naturally retain Gaussian-like properties. Empirically, as shown in Figure 3,
analysis of hidden states from the FLUX.1-DEV (Labs et al., 2025) model across early, middle,
and final timesteps confirms this behavior. Accordingly, we extract µ(·) and σ(·) from both the
current latent xt and the cached residual rcached. In addition, we include max(|rcached|) to capture
salient localized updates that may be obscured by average statistics.
▷Temporal Awareness. Diffusion dynamics are non-stationary, evolving from coarse structural
formation to fine-grained refinement. The time embedding embtime(t) supplies phase information,
enabling the policy to adapt its caching behavior across different denoising stages.
Figure 3: Statistical feature design for pol-
icy input. We extract (µ(·), σ(·)) to char-
acterize the latent distribution, motivated
by the approximately Gaussian behavior of
latent representations in diffusion models.
Based on the above analysis, let D be the channel dimension
of the latent space. The state st provided to the policy network
πθ(at|st) is a concatenated vector in R6×D:
st = Concat


µ(xt), σ(xt),
µ(rcached), σ(rcached), max(|rcached|),
embtime(t)

.
(3)
The policy πθ is parameterized by a lightweight Multilayer Per-
ceptron(MLP). The architecture consists of: (i) An input pro-
jection layer compressing the feature dimension from 6 × D
to 2 × Nhidden, followed by LayerNorm and ReLU. (ii) A bot-
tleneck layer reducing dimensions to Nhidden/2, followed by
ReLU. (iii) A final projection to a scalar probability pt via a
Sigmoid activation. The action is then sampled stochastically
with at ∼Bernoulli(pt). To assess the validity of these de-
sign choices, we perform a controlled ablation over the state
components, as analyzed in Section 4.4.1.
4

3.3
Reward Design and Policy Optimization
We design a composite reward function R(τ) evaluated over the full generation trajectory τ. The
reward encourages high visual fidelity while enforcing a specific acceleration rate.
▷Quality Reward (Rq): We measure the perceptual similarity between the image generated by our
dynamic policy, Ipred, and the ground-truth image generated with full compute, Igt. We utilize the
combination of LPIPS (Zhang et al., 2018) and SSIM (Wang et al., 2004):
Rq = λq (1 −LLPIPS(Igt, Ipred) + LSSIM(Igt, Ipred)) .
(4)
▷Acceleration Constraint (Ra): To enforce a target compute budget, we penalize deviations from a
target cache ratio ρtarget (e.g., 0.5, aiming for the policy to converge to caching roughly 50%):
Ra = −λa

1
T
T
X
t=1
at −ρtarget
 .
(5)
▷Stability Penalty (Rs): To prevent error explosion from consecutive caching, we apply a penalty if
the number of consecutive skips kt exceeds a tolerance threshold ktol:
Rs = −λs
 T
X
t=1
1(kt>ktol) · (kt −ktol)
!
.
(6)
The total reward for trajectory τ is the sum: R(τ) = Rq + Ra + Rs. Since Ra and Rs are always
negative, they serve as penalty terms. We optimize the policy parameters θ using the REINFORCE
algorithm (Monte Carlo Policy Gradient). To bridge the gap between raw rewards and stable gradient
updates, we transform the total reward into a normalized advantage score and incorporate entropy
regularization. For a detailed derivation of the reverse gradient flow, please refer to Appendix A.
However, the raw reward signals R(τ) can vary significantly in magnitude, leading to high variance in
gradient estimation. In practice, we follow a standard approach and compute a normalized advantage
score for each trajectory relative to the current batch B. Let R(τi) be the cumulative reward for the
i-th trajectory in a batch of size B. The advantage A(τi) for trajectory i is then standardized:
A(τi) =
R(τi) −1
B
PB
j=1 R(τj)
r
1
B
PB
j=1

R(τj) −1
B
PB
k=1 R(τk)
2
+ ϵ
,
(7)
where ϵ is a small constant for numerical stability. This normalization assigns positive rewards to
above-average trajectories and penalizes below-average ones.
To encourage exploration and prevent the policy from collapsing to a deterministic behavior early in
training, we introduce an entropy bonus. Since our policy πθ(st) outputs a scalar probability pt for
the Bernoulli distribution, the entropy at timestep t can be defined as:
H(pt) = −[pt log pt + (1 −pt) log(1 −pt)] .
(8)
Finally, the total loss L(θ) combines the policy gradient and entropy terms. We introduce λe to
control the exploration-exploitation trade-off, for a batch of trajectories, the objective is:
L(θ) = 1
B
B
X
i=1
"
−
 T
X
t=1
log πθ(ai,t | si,t)
!
· A(τi) −λe
1
T
T
X
t=1
H(πθ(si,t))
#
.
(9)
3.4
Bilevel Optimization Framework
To further improve performance, we introduce an error corrector implemented as a lightweight MLP,
which compensates for the drift induced by cache reuse. We formulate the joint training as a bilevel
optimization problem. The framework consists of an outer-loop objective that optimizes the policy
and a inner-loop objective that minimizes the local error. Formally, let θ and ϕ denote the parameters
of the policy πθ and corrector Cϕ, respectively. The optimization problem can be defined as:
min
θ
Louter(θ, ϕ∗(θ)) = −Eτ∼πθ[R(τ)]
s.t.,
ϕ∗(θ) ∈arg min
ϕ Linner(θ, ϕ),
(10)
where R(τ) is the cumulative reward defined in Section 3.3.
5

3.4.1
Outer-loop: Policy Optimization
In the outer loop, we optimize the policy parameters θ while keeping the error corrector ϕ fixed,
which represents the optimal solution ϕ∗under the current policy. The policy network learns to
make binary caching decisions at based on observed latent statistics. The optimization objective is to
minimize the negative expected reward via the policy gradient defined in Equation 9, where A(τi)
denotes the normalized advantage of the trajectory generated through the interaction between the
current policy and the corrector. The update of θ can be expressed as:
dLouter
dθ
=
∂Louter
∂θ
| {z }
Direct Gradient
+ ∂Louter
∂ϕ∗
· dϕ∗(θ)
dθ
|
{z
}
Implicit Gradient
.
(11)
The direct gradient term characterizes how variations in the policy θ directly influence the reward
R(τ). In contrast, the implicit gradient reflects the fact that changing θ induces a shift in the inner-
level optimum ϕ∗, since the corrector is trained conditional on the current caching decisions. This
variation in ϕ∗in turn influences the outer-level reward.
3.4.2
Inner-loop: Corrector Training
The inner loop aims to find the optimal corrector ϕ∗that compensates for the numerical errors
introduced when the policy network chooses to skip the transformer computation (at = 1). When
a step is cached, the accelerated output is calculated as ˆvt(ϕ) = vt + Cϕ(xt, rcached, embtime(t)),
where vt is computed according to Equation 2 under the condition at = 1. Given the pre-computed
ground-truth intermediate states vgt
t (Appendix C.2.1 reports an ablation on the choice of corrector
training target), the inner objective Linner is defined as the Mean Squared Error over the cached steps:
Linner(θ, ϕ) =
X
t : at=1
ˆvt(ϕ) −vgt
t
2
2 .
(12)
3.4.3
Practical BLO Implementation
Computing the exact implicit gradient dϕ∗(θ)
dθ
in Equation 11 is intractable in our setting for two
reasons: (i) The dependency of the inner objective Linner on the policy parameters θ is mediated by
the discrete sampling of actions at ∼πθ. This non-differentiable sampling operation prevents the
flow of gradients from the inner-loop loss back to the policy parameters via standard backpropagation.
(ii) Even with continuous relaxation tricks (e.g., Gumbel-Softmax), calculating the exact implicit
gradient typically involves computing Hessian-vector products, which imposes a prohibitive memory
and computational burden given the high-dimensional context of diffusion models.
Therefore, we adopt a First-Order Approximation of the BLO objective. Specifically, we simplify
the optimization by alternating between the two levels: (i) Policy update (Outer-loop): We update
θ using the policy gradient, assuming the current corrector ϕ is a localized constant approximation
of the optimal ϕ∗(θ). (ii) Corrector training (Inner-loop): We update ϕ via supervised fine-tuning
based on the trajectories generated by the new policy πθ. This alternating scheme can be viewed as a
coordinate descent approach to the BLO problem. By iteratively updating the policy and adapting the
corrector to shifts in the action distribution, the framework converges to a joint equilibrium, enabling
our OnlineCache to deliver substantial speedups while preserving generation quality.
4
Experiments
We design three sets of main experiments to evaluate our proposed OnlineCache:
▷FLUX.1-DEV (Targeting SOTA Caching Methods 4.1): For image generation, we primarily
compare against the SOTA caching methods and further provide a thorough generalization analysis.
▷DIT-XL/2 (Targeting Learning-Based Methods 4.2): OnlineCache introduces an extra training
phase. We therefore compare with representative learning-based cache methods (FastCache, L2C),
demonstrating that our approach achieves superior performance even within trainable frameworks.
▷COGVIDEOX-2B (Targeting Cross-Modality Generalization 4.3): For video generation, our goal
is to validate the modality-agnostic nature of OnlineCache. Comparing with the strong baseline
TeaCache is sufficient to show effective generalization beyond image diffusion models.
6

4.1
FLUX.1-DEV Experiment
Table 1: Quantitative comparison on FLUX. We evaluate FLUX-series models on diverse datasets to demonstrate the strong
performance and generalization capability of OnlineCache. Lat and CR denote Latency and Cache Ratio, respectively. Rows
with the same background color denote comparable settings, and bold indicates the best result within each group.
Method
Efficiency
Visual Quality
CR
Speed ↑
Lat ↓
LPIPS ↓
SSIM ↑
PSNR ↑
Block1: FLUX.1-dev, MSCOCO, 512×512, 800 samples
FLUX(30 steps)
1.00×
5.099
–
–
–
0.00%
ERTACache
2.68×
1.900
0.252
0.739
19.852
66.67%
TeaCache
2.60×
1.960
0.329
0.669
17.006
-
OC-BLO(Ours)
2.96×
1.725
0.245
0.739
21.442
72.22%
TeaCache
1.40×
3.646
0.142
0.831
22.453
-
OC-BLO(Ours)
2.07×
2.463
0.125
0.850
24.768
56.32%
Block2: FLUX.1-dev, MSCOCO, 1024×1024, 500 samples
FLUX(30 steps)
1.00×
15.470
–
–
–
0.00%
FLUX(12 steps)
2.46×
6.291
0.390
0.679
16.619
60.00%
OC-BLO(Ours)
3.25×
4.758
0.374
0.694
20.265
74.81%
ERTACache
2.62×
5.912
0.281
0.755
20.382
66.67%
OC-BLO(Ours)
2.75×
5.631
0.275
0.762
21.343
68.38%
Block3: FLUX.1-schnell, Parti-Prompts, 512×512, 1632 samples
FLUX(4 steps)
1.00×
0.884
–
–
–
0.00%
OC-BLO(Ours)
1.31×
0.674
0.070
0.844
23.261
34.40%
OC-BLO(Ours)
1.66×
0.534
0.131
0.671
22.649
57.72%
FLUX(6 steps)
1.00×
1.212
–
–
–
0.00%
OC-BLO(Ours)
2.31×
0.525
0.197
0.574
20.232
74.59%
Method
Efficiency
Visual Quality
CR
Speed ↑
Lat ↓
LPIPS ↓
SSIM ↑
PSNR ↑
Block4: FLUX.1-dev, Parti-Prompts, 512×512, 1632 samples
FLUX(30 steps)
1.00×
5.090
–
–
–
0.00%
FLUX(12 steps)
2.31×
2.199
0.425
0.587
13.958
60.00%
TaylorSeer(N=6,O=1)
3.18×
1.599
0.405
0.578
15.033
76.68%
TaylorSeer(N=6,O=2)
2.97×
1.714
0.383
0.593
16.009
76.68%
TaylorSeer(N=5,O=1)
2.81×
1.812
0.341
0.636
17.795
73.33%
TaylorSeer(N=5,O=2)
2.67×
1.904
0.338
0.638
18.370
73.33%
OC-BLO(Ours)
3.18×
1.601
0.316
0.668
19.928
73.39%
ERTACache
2.68×
1.896
0.251
0.746
20.319
66.67%
TaylorSeer(N=4,O=1)
2.45×
2.077
0.274
0.689
19.195
70.00%
TaylorSeer(N=4,O=2)
2.38×
2.142
0.267
0.693
19.690
70.00%
OC-BLO(Ours)
2.93×
1.739
0.248
0.740
21.286
70.92%
Block5: FLUX.1-dev, Parti-Prompts, 512×512, 1632 samples
FLUX(30 steps)
1.00×
5.090
–
–
–
0.00%
OC-BLO(Ours)
3.84×
1.327
0.372
0.618
18.810
79.67%
OC-BLO(Ours)
4.04×
1.261
0.388
0.603
18.505
81.05%
OC-BLO(Ours)
4.26×
1.195
0.410
0.586
18.057
82.37%
OC-BLO(Ours)
4.50×
1.130
0.418
0.578
17.673
83.84%
OC-BLO(Ours)
4.78×
1.065
0.443
0.554
16.908
85.28%
OC-BLO(Ours)
5.25×
0.969
0.478
0.524
16.104
87.27%
4.1.1
Quantitative Results on FLUX.1-DEV
We evaluate OnlineCache with Bilevel Optimization (i.e., OC-BLO) on MSCOCO (Lin et al., 2014),
generating 800 images at a 512 × 512 resolution. Quantitative results are presented in Table 1 ,
Block1. Using FLUX.1-DEV as the baseline without caching, we compare our method against widely
adopted cache-based approaches, including TeaCache and ERTACache. Under identical settings,
OnlineCache demonstrates consistent performance improvements in synthesis quality and efficiency.
4.1.2
Generalization Analysis
To assess the robustness and applicability of our proposed OnlineCache, we conduct comprehensive
generalization experiments across four dimensions:
▷Cross-Resolution Generalization (Block2): We evaluate the policy and corrector trained at
512 × 512 directly on 1024 × 1024 generation without retraining. OnlineCache achieves a 3.25×
speedup, outperforming a 12-step reduction baseline (2.46×) across all metrics. Compared to
ERTACache, our method also attains a higher speedup while maintaining superior fidelity. This
scalability is largely attributed to the policy design (Section 3.2), which utilizes global statistics
(channel-wise mean and standard deviation) rather than raw spatial tensors. Since sequence length
L varies with image resolution, adopting such resolution-agnostic statistics avoids coupling the
policy to specific sequence lengths, thereby improving generalization across different resolution.
▷Cross-Model Generalization (Block3): While direct cross-model transfer is generally constrained
by architectural disparities (e.g., varying latent spaces, channel dimensions, and prediction targets),
training a model-specific policy remains highly practical. The policy introduces negligible overhead
(76.6MB for 24GB FLUX.1-DEV) and trains efficiently (see Table 4). Furthermore, the learned
policy exhibits zero-shot transferability when underlying architectures are closely aligned. For
instance, applying the FLUX.1-DEV policy to its distilled variant, FLUX.1-SCHNELL (sharing
the same 3072-dimensional hidden state space), achieves a 2.31× speedup with LPIPS < 0.2.
▷Cross-Dataset Generalization (Block4): We directly evaluate the MSCOCO-trained OnlineCache
on the Parti-Prompts (Yu et al., 2022) dataset. The policy maintains strong performance, indicating
that it captures fundamental denoising trajectory dynamics rather than dataset-specific artifacts,
rendering domain-specific retraining unnecessary. Comparisons with the state-of-the-art TaylorSeer
further validate the sustained efficacy of our approach under data distribution shifts.
▷Acceleration Ratio Generalization (Block5): During training with the acceleration constraint
(Equation 5), cache ratio converges to ∼0.43, yielding a 1.65× speedup. At inference, the overall
acceleration ratio can be flexibly controlled via arithmetic modifications to the policy’s final output
logits. Even under extreme acceleration (scaling from 3.84× to 5.25×), performance degrades
gracefully rather than collapsing. Specifically, LPIPS increases monotonically from 0.372 to 0.478,
indicating robust and reliable instance-aware decisions under tight computational budgets.
7

4.2
DIT-XL/2 Experiment
Table 2: Quantitative comparison on DiT-XL/2. We compare OnlineCache against other learning-based acceleration
approaches under diverse sampling configurations. The "FastCache baselines” are reused from the official FastCache paper (Liu
et al., 2025b) (Tables 10 and Table 12). IS and Prec denote Inception Score and Precision, respectively. Note that sFID, IS,
Prec, and Recall are not reported for FastCache Baselines, as they were not evaluated in the official paper.
Method
FID ↓
sFID ↓
IS ↑
Prec ↑
Recall ↑
Speed ↑
FastCache Baselines
Baseline
4.45
-
-
-
-
1.00×
TeaCache
5.09
-
-
-
-
1.84×
AdaCache
4.64
-
-
-
-
1.26×
L2C
6.88
-
-
-
-
1.69×
FBCache
4.48
-
-
-
-
1.63×
FastCache
4.46
-
-
-
-
1.74×
OnlineCache with BLO (DDIM, 30 steps, CFG 1.7)
Baseline
4.44
8.53
304.46
0.85
0.50
1.00×
OC-BLO
4.41
8.61
284.57
0.84
0.50
1.54×
OC-BLO
4.43
8.74
271.91
0.84
0.50
1.70×
Method
FID ↓
sFID ↓
IS ↑
Prec ↑
Recall ↑
Speed ↑
More OnlineCache Configurations
DPM++, 50 steps, CFG 1.5
Baseline
3.47
9.20
268.44
0.82
0.57
1.00×
OC-BLO
3.28
8.05
257.19
0.82
0.56
1.88×
DPM++, 25 steps, CFG 1.5
Baseline
3.77
10.63
264.17
0.82
0.55
1.00×
OC-BLO
3.69
7.65
229.30
0.78
0.57
1.69×
DPM++, 15 steps, CFG 1.5
Baseline
3.95
11.09
254.78
0.82
0.55
1.00×
OC-BLO
4.07
8.42
189.45
0.71
0.58
1.65×
Table 2 presents the results on ImageNet-256 using DIT-XL/2 (Peebles and Xie, 2023). To com-
prehensively evaluate the performance of our method, we conduct comparisons against a range
of widely-used methods, including training-free approaches (TeaCache (Liu et al., 2025a), Ada-
Cache (Kahatapitiya et al., 2024), FBCache (Cheng, 2025)), and learning-based caching methods
such as L2C (Ma et al., 2024b), FastCache (Liu et al., 2025b). To ensure fairness, we align our
settings with FastCache. Following standard protocols, we generate 50k samples acros
