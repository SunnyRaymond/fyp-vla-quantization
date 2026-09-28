# Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache

paper_id: semanticscholar:d3e2550ef75f159ad3a47ca1e66dd8ec612d2266
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

In recent years, diffusion models [15, 20, 21, 36, 45] have
achieved remarkable success in image and video genera-
tion, leading to widespread applications in domains such
*Equal contribution. †Corresponding authors.
Noise
Data
High Deviation
(a) Fixed schedule
Noise
Data
High Deviation
(b) Locally adaptive schedule
Noise
Data
Low Deviation
Calibration
Optimal Schedule
(c) DPCache
Original trajectory
Computed trajectory
Cached trajectory
Computed timestep
Cached timestep
Trajectory deviation
Figure 1. (a) Fixed schedule is inflexible and unable to identify
critical timesteps, resulting in large deviation from the true tra-
jectory. (b) Locally adaptive schedule makes greedy, short-sighted
decisions that often skip essential timesteps, leading to irreversible
deviation. (c) DPCache identifies a globally optimal sequence of
key timesteps through calibration and achieves low cumulative tra-
jectory deviation.
as robotics, intelligent content creation, and virtual real-
ity. However, diffusion models, particularly diffusion trans-
formers (DiTs) [34], are often characterized by their large-
scale architectures and reliance on multi-step iterative sam-
pling during inference, resulting in substantial computa-
tional overhead and prolonged latency, which severely hin-
der their practical deployment in real-world scenarios. As a

## Method

global path planning problem, selecting an optimal se-
quence of key timesteps to minimize deviation from the
full-step denoising trajectory.
• We propose a Path-Aware Cost Tensor that quantifies the
path-dependent error of skipping intermediate timesteps
along the denoising trajectory, and employ dynamic pro-
gramming to efficiently identify the optimal sampling
schedule for a fixed number of sampling steps.
• DPCache achieves up to 4.87× acceleration across im-
age and video generation models including DiT, FLUX,
and HunyuanVideo, consistently outperforming prior
caching-based methods in both efficiency and generation
quality, establishing a new state of the art in training-free
diffusion sampling acceleration.
2. Related Works
2.1. Sampling Step Reduction
Dynamics redesign accelerates diffusion by redefining
the sampling trajectory or numerical integration scheme.
DDIM [43] introduces a deterministic sampling trajectory,
allowing high-quality generation with fewer steps.
The
DPM-Solver family [29, 30, 60] employs high-order numer-
ical solvers for ordinary differential equations (ODEs), sig-
nificantly reducing the number of sampling steps. Rectified
Flow [28] learns a straight transport path from noise to data
by modeling instantaneous velocity, while MeanFlow [11]
replaces this with average velocity to promote stability and
improve fidelity in generation.
Step distillation reduces sampling steps by distilling a
full-step teacher model into a few-step student. Progres-
sive distillation [39] establishes this paradigm by iteratively
compressing step counts through trajectory-level supervi-
sion. Subsequent methods refine the distillation objective:
Distribution Matching Distillation (DMD) [3, 52, 53] aligns
the student and the teacher at the marginal distribution level,
while adversarial diffusion distillation [3, 24, 40, 51] incor-
porates GAN-based discriminators [12] to improve percep-
tual fidelity. In parallel, consistency models [31, 33, 44, 46]
eliminate the need for a teacher altogether, instead en-
forcing the self-consistency of network predictions across
timesteps. However, these methods require additional train-
ing and are thus computationally expensive.
2

2.2. Per-Step Computation Optimization
Denoising model compression reduces computational
overhead by compressing the model through parameter re-
duction strategies. Pruning methods [2, 7, 55, 58] sparsify
weights or attention structures to reduce parameter count.
Quantization [19, 22, 42, 49] enables low-bit inference by
approximating full-precision parameters with compact nu-
meric representations. Some approaches leverage knowl-
edge distillation [10, 23, 56] to transfer the knowledge of
a larger teacher model to a lightweight student model. In
parallel, token merging methods [1, 8] dynamically fuse se-
mantically similar tokens to shorten the sequence length and
alleviate the computational cost of attention. Nevertheless,
these methods typically introduce irreversible degradation
in representational capacity, often necessitating nontrivial
fine-tuning or retraining to recover generation quality.
Feature caching accelerates diffusion sampling by
reusing previously computed intermediate features across
timesteps, thereby avoiding redundant forward passes.
DeepCache [32] first demonstrates that high-level feature
maps in U-Net-based diffusion models exhibit strong inter-
step similarity and can be safely reused with minimal qual-
ity degradation. This insight has since been extended to Dif-
fusion Transformers [5, 41]. To enhance robustness, adap-
tive strategies such as AdaCache [18] and TeaCache [25]
further refine reuse decisions based on feature distance or
predicted deviation.
Parallel efforts focuses on mitigat-
ing errors introduced by caching. For instance, GOC [35]
corrects feature drift via gradient-based propagation, while
OmniCache [6] employs stage-aware filtering to suppress
cache-induced noise. Recently, TaylorSeer [26] proposes a
cache-then-forecast paradigm that predicts future features
via Taylor expansion. HiCache [9] improves the prediction
accuracy with Hermite polynomials. SpeCa [27] employs a
lightweight verifier network to dynamically decide whether
to accept the predicted features. Despite these advances, ex-
isting caching-based methods typically rely on fixed or lo-
cally adaptive schedules. Our proposed DPCache reframes
sampling acceleration as a global path planning problem,
explicitly optimizing the sequence of caching and computa-
tion decisions to minimize overall error accumulation.
3. Method
3.1. Preliminary
Diffusion models. Diffusion models are a class of genera-
tive models that synthesize realistic samples through a for-
ward diffusion process and a reverse denoising process. The
forward diffusion process is a Markov chain that progres-
sively adds Gaussian noise to a clean data point x0 over T
timesteps according to a predefined noise schedule {βt}T
t=1:
q(xt|xt−1) = N(xt;
p
1 −βtxt−1, βtI)).
(1)
After T timesteps, the latent xT is approximately dis-
tributed as standard Gaussian noise. The reverse process
aims to recover data by iteratively denoising from xT ∼
N(0, I) back to x0. It is modeled as a learned Markov
chain with Gaussian transitions:
pθ(xt−1|xt) = N(xt−1; µθ(xt, t), Σθ(xt, t)),
(2)
where µθ and Σθ are trained to approximate the true reverse
dynamics. As sampling involves T sequential timesteps,
fast and accurate sampling remains a fundamental require-
ment for diffusion models.
Feature caching in diffusion models.
Existing fea-
ture caching methods generally adopt a periodic temporal
caching strategy with interval N. At a selected timestep
t, the model performs a full forward pass and caches the
output of each layer l as hl
t = Fl(xt). The cached fea-
ture stack H = {hl
t}L
l=1 is then reused for the next N −1
timesteps without further computation:
Fl(xt−k) = hl
t, k = 1, 2, ..., N −1.
(3)
TaylorSeer [26] addresses the error accumulation inherent
in na¨ıve reuse by introducing Taylor-series-based predictor.
It maintains a history of finite differences up to order m,
forming an enriched cache:
H = {hl
t, ∆hl
t, ..., ∆mhl
t}L
l=1.
(4)
TaylorSeer uses the cached differences to approximate fea-
tures at timestep t −k via a truncated Taylor expansion:
Fl
pred,m(xt−k) = hl
t +
m
X
i=1
∆ihl
t
i! · N i (−k)i.
(5)
Our method builds on this predictive paradigm but uses a
globally optimized caching strategy, instead of fixed or lo-
cally adaptive schedules.
3.2. Overview
Existing caching-based acceleration methods typically op-
timize the sampling schedule without accounting for the
global structure of the denoising trajectory. To address this
