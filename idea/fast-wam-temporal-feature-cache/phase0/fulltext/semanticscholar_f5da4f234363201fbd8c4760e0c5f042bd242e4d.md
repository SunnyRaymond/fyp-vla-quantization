# ReCache: Learning Budget-Aware Caching Schedules for Diffusion Models via REINFORCE

paper_id: semanticscholar:f5da4f234363201fbd8c4760e0c5f042bd242e4d
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Diffusion models have become one of the dominant paradigms for high-fidelity image and video
generation [22, 50, 4]. Recent transformer-based architectures, such as DiT [46], have further
improved scalability and generation quality, enabling increasingly powerful text-to-image and text-to-
video systems. However, these advances come at a substantial inference cost: generating a single
sample requires repeatedly evaluating a large denoising network over many sampling steps. As model
size, resolution, and video length continue to grow, reducing the cost of diffusion inference has
become an important practical challenge.
A prominent line of work accelerates diffusion sampling through feature caching. The key observation
is that consecutive denoising steps often perform highly redundant computations: intermediate
activations change smoothly along the reverse trajectory and can therefore be reused or approximated
across nearby timesteps [44, 58]. Existing caching methods typically differ along two axes: what to
∗Equal contribution. Correspondence: Mishan Aliev <maliev@hse.ru>, Eva Neudachina <eneudachina@hse.ru>, Ilya
Bykov <philurame@gmail.com>.
Preprint.
arXiv:2606.06060v1  [cs.CV]  4 Jun 2026

Figure 1: ReCache preserves text and visual fidelity under high acceleration on FLUX.1-dev [3].
Under a 5.4× acceleration, heuristic baselines (TaylorSeer [37], DPCache [12]) fail to maintain
image quality and corrupt text spelling. By framing scheduling as an RL problem, ReCache allocates
computations to the most critical steps, closely matching Full inference at the same computational
cost.
cache and when to cache. While some approaches reuse selected model components according to a
fixed schedule [44, 58, 11], others determine caching steps dynamically using error-based heuristics
[35]. Furthermore, recent methods go beyond direct reuse by forecasting future features [37, 17].
Despite these differences, most methods still rely on hand-crafted scheduling rules to decide at which
denoising steps the expensive full computation should be performed.
In this work, we focus on the scheduling problem, arguing that caching schedules should be learned
rather than hand-crafted. Existing approaches, including both uniform spacing and adaptive heuristics,
often aim to keep the accelerated sampling process close to the original full-inference trajectory at
intermediate denoising steps. However, this objective can be suboptimal under a fixed computational
budget: different denoising timesteps contribute unequally to the final generation, and the steps that
best preserve intermediate trajectory consistency may not coincide with those most important for
the final output. This suggests that schedule design should not only minimize local approximation
errors along the denoising path, but should also account for how each recomputation decision affects
the final sample. We therefore argue that caching schedules should be optimized directly for output
consistency, i.e., for preserving the perceptual and semantic quality of the generated sample relative
to full inference.
To achieve this, we frame caching-schedule selection as a Reinforcement Learning (RL) problem.
We propose ReCache, a budget-aware method that learns a scheduling policy for diffusion feature
caching. Once a policy is trained for a specific model and caching setup, ReCache takes a target
budget and predicts a schedule, determining exactly where full computation is necessary and where
cached features are sufficient. The policy is optimized via a reward with two objectives: minimizing
the distance to the full-inference samples and preserving the perceptual and semantic quality of
the generated images. As shown in Figure 1, for a fixed computational budget, our learned policy
produces outputs that more closely match the full-inference generations than those from uniform or
heuristic-adaptive baselines.
We evaluate ReCache on FLUX.1-dev [3], HunyuanVideo [28], and Wan2.1 [67], and combine it
with recent caching mechanisms such as TaylorSeer [37] and HiCache [17]. A single trained policy
(for a given caching mechanism) adapts to multiple FLOPs budgets at inference time, avoiding
the need to design a separate schedule for each acceleration regime. Across all evaluated settings,
ReCache consistently outperforms hand-crafted scheduling baselines at the same computational
cost, demonstrating that learning budget-aware schedules is a simple and effective way to improve
diffusion caching methods.
In summary, our main contributions are:
• We frame caching schedule selection as a budget-aware RL problem that directly optimizes output
consistency under a fixed compute budget, instead of relying on hand-crafted trajectory-consistency
heuristics.
• We propose ReCache, a lightweight scheduling method compatible with diverse diffusion models
and caching mechanisms.
2

• We show that ReCache consistently improves over hand-crafted scheduling baselines across
multiple budgets, caching strategies, and state-of-the-art image and video models, including FLUX,
HunyuanVideo, and Wan2.1.
2
Preliminaries
Diffusion and Flow Matching models [59, 22, 62, 34, 38, 1] generate samples by numerically
integrating a learned ODE that transports a Gaussian latent z ∼N(0, I) to a data sample. A diffusion

## Method

an ODE solver. Each step evaluates a learned velocity v(xt, t) parameterized as a neural network,
and this evaluation dominates the per-step cost. In modern image and video models, this network
is a diffusion transformer [46, 15, 3, 67, 28], whose forward pass through stacked attention and
feedforward blocks is what caching methods aim to amortize. Since our method is agnostic to the
specific timestep schedule, we index inference steps by their position {1, 2, . . . , N} throughout the
paper.
Feature caching.
Recent works observe that the intermediate activations of vθ change slowly
between adjacent inference steps and propose to amortize the per-step cost by reusing them. The
inference steps are split into two groups: cache steps, where the network is evaluated in full and
selected activations are stored, and reuse steps, where the missing activations are reconstructed from
the cache rather than recomputed. A caching schedule s is a subset s ⊆{1, . . . , N} of cache steps,
while its size |s| = k is a caching budget — it controls the number of full model evaluations in the
generation process.
Apart from a problem of choosing a caching schedule, caching mechanisms M set a rule of how
the reuse steps reconstruct activations that a full forward trajectory would have produced. Direct
reuse mechanisms copy cached activations from the most recent cache step, either as block outputs
(FORA [58]) or as residuals added to the current step’s input (∆-DiT [11]). Feature forecasting
mechanisms instead extrapolate the current activations from values cached at several previous cache
steps: TaylorSeer [37] fits a Taylor polynomial to the trajectory of cached features, and HiCache [17]
replaces the Taylor basis with one better suited to diffusion feature trajectories. Our experiments
evaluate ReCache on top of both direct and forecasting methods, but ReCache is agnostic to the
choice of M.
Caching schedules.
Existing methods predominantly propose policies of choosing caching sched-
ules s via hand-crafted rules. Many works place cache steps at uniform intervals along the inference
trajectory [11, 58, 37, 17]; others select them adaptively per generation, e.g. by triggering a cache
step whenever the activation error exceeds a threshold [35, 6]. Both families either fix the budget
implicitly or expose it only through threshold hyperparameters, motivating the budget-aware learned
policy we develop in Section 3.
An extended discussion of these prior approaches and broader related work is provided in Appendix A.
3
Method
We introduce ReCache, a framework that trains a schedule policy: given a target budget k, the
policy selects k inference steps to recompute (Figure 2). We cast the training as a stochastic policy-
optimization problem and parameterize the policy as a budget-conditioned distribution over k-subsets.
3.1
Policy Optimization
To evaluate a schedule’s quality, we combine two complementary signals. The first signal asks that
the cached generation stay close to the full-inference one. This target provides strong supervision,
since the full-inference output is a direct reference for what the model would have produced without
caching. Under aggressive caching, however, exact agreement with the full-inference trajectory
is difficult to achieve, and the policy may need to settle for an approximation. The second signal
is a reward model [72, 70, 27] that scores generation quality directly, letting the policy fine-tune
perceptual details that the distance term alone cannot recover.
3

Figure 2: Overview of the ReCache method. Given a target computational budget k, a lightweight
MLP policy network predicts importance logits to perform a top-k selection of inference steps for
full computation. During the inference with caching, the selected steps are fully computed (green
blocks), while the intermediate steps reuse or forecast features (red blocks). The policy is trained
using the REINFORCE algorithm without backpropagating through the diffusion process. The
reward formulation balances a Fidelity term (minimizing the distance to the xfull output from the full
inference) and a Quality term (maximizing the perceptual quality of the generated xcache).
Concretely, let z ∼N(0, I) and xfull = G(z), and denote by xcache = G(z | s, M) the output
produced under schedule s and caching mechanism M. Given a distance d and a reward model R,
Limage(s) = d (xcache, xfull) −αiq R (xcache) .
(1)
Direct minimization of Limage over schedules is infeasible: the loss is non-differentiable in s, and
the search space contains
 N
k

subsets. We instead lift the problem to a stochastic policy and train a
distribution pθ over k-subsets to minimize the expected loss
min
θ
Es∼pθ Limage(s).
(2)
Without further care, the policy can collapse to a deterministic schedule that is locally good but
globally suboptimal. We counter this with an entropy bonus −λentropyH(θ) that keeps the policy
exploring during training, yielding the regularized loss
Lreg(s) = Limage(s) + λentropy log pθ(s).
(3)
The regularized expectation Epθ Lreg(s) can be optimized over θ with policy-gradient methods. We
use REINFORCE [69] with a leave-one-out (LOO) baseline [29, 49] for variance reduction: given n
i.i.d. samples s(1), . . . , s(n) ∼pθ and their mean Lreg = 1
n
P
i Lreg(s(i)),
∇θ Epθ(s) Lreg(s) ≈
1
n −1
n
X
i=1

Lreg(s(i)) −Lreg

∇θ log pθ(s(i)).
(4)
3.2
Budget-conditioned Policy Over k-subsets
The policy must select a k-subset s ⊆{1, . . . , N}. We parameterize it with the Plackett-Luce
distribution [47]: each step i is assigned an importance logit θi, and a sample is drawn by sampling
without replacement from the corresponding categorical distribution k times. Applying the chain rule
4

yields the closed-form log-probability
log pθ(s) =
k
X
i=1

θsi −log
X
j /∈{s1,...,si−1}
exp(θj)

.
(5)
The relative importance of each step may in principle depend on the budget. At small k, only the
most critical steps survive, and the policy must concentrate its mass on those steps. At larger k, the
policy has more room to spread mass across the trajectory. We do not assume that the optimal step
ranking is budget-invariant: we condition the logits on k through a lightweight MLP with parameters
ϕ, θ = MLPϕ(k), and train a single budget-adaptive policy by sampling k ∼q(k) at every training
step, so that one trained model covers a range of inference budgets without retraining. Empirically
(Section 4.3), the trained policy produces nested schedules across budgets: the (k+1)-schedule
extends the k-schedule by a single step rather than reordering its existing selections. This suggests
that under the caching objective the underlying step-importance ranking is largely budget-invariant.
The MLP parameterization makes no architectural commitment to this; nestedness is discovered, not
imposed, and a single set of k-independent logits would in retrospect suffice.
In practice we never sample sequentially. The Gumbel-Top-k trick [30, 19, 64] produces a Plackett-
Luce sample in one shot by perturbing each logit with independent Gumbel noise and returning the
top-k indices:
gi = θi −log(−log ui),
ui
i.i.d.
∼U(0, 1),
s = (s1, . . . , sk) = argtopk(g1, . . . , gN).
(6)
At inference we drop the noise and use the deterministic schedule s = argtopk(θ1, . . . , θN).
3.3
Training Algorithm
Algorithm 1 summarizes ReCache training. Before training begins, we pre-sample a small dataset
of (x1, xfull) pairs, where each pair consists of an initial noise sample and the corresponding full-
inference output of G. At each training step, we draw a target budget k ∼q(k), obtain logits
θ = MLPϕ(k), sample n schedules from the resulting Plackett-Luce distribution via the Gumbel-
Top-k trick, and update ϕ with the LOO estimator (Eq. 4) of the regularized loss Lreg.
Algorithm 1 ReCache schedule policy training
Input: diffusion model G with N backbone steps, caching mechanism M, number of leave-one-out
samples n, distance function d, reward function R, loss coefficient αiq, regularizer coefficient
λentropy;
repeat
for all elements in batch do
z ∼N(0, I), xfull = G(z)
▷pre-sampled noise and full-inference image
k ∼q(k)
▷sample caching budget
θ = MLPϕ(k)
▷logits parameterized by MLP
for i = 1 to n do
Sample s(i) = (s(i)
1 , . . . , s(i)
k ) ∼pθ
▷via Eq. 6
x(i)
cache = G(z | s(i), M)
▷generate image with cached inference
Limage(s(i)) = d
 x(i)
cache, xfull

−αiq R(x(i)
cache)
Compute log pθ(s(i))
▷via Eq. 5
Lreg(s(i)) = Limage(s(i)) + λentropy log pθ(s(i))
end for
Lreg = 1
n
Pn
i=1 Lreg(s(i))
L(s(i)) =
 Lreg(s(i)) −Lreg

.stopgrad()
∇θL(s) =
1
n−1
Pn
i=1 L(s(i)) ∇θ log pθ(s(i))
end for
Accumulate gradients across batch elements and update ϕ via backprop through θ = MLPϕ(k).
until converged
5

4
Experiments
4.1
Experimental Setup
We evaluate ReCache on three widely used image and video generation backbones: FLUX.1-dev [3]
for text-to-image generation, and HunyuanVideo [28] and Wan2.1 [67] for text-to-video generation.
For FLUX.1-dev, we use N = 50 inference steps and generate images at 1024 × 1024 resolution.
HunyuanVideo uses N = 50 steps and generates videos at 480 × 640 × 65 resolution, while Wan2.1
uses N = 25 steps and 480 × 832 × 81.
Baselines.
ReCache is a scheduling policy and is agnostic to the underlying caching mechanism.
We therefore evaluate it across several computational budgets—7, 9, and 13 full-computation steps—
and combine it with state-of-the-art caching mechanisms. For feature-forecasting methods, including
TaylorSeer [37] and HiCache [17], we compare ReCache with a uniform schedule. DiCache [6]
introduces Dynamic Cache Trajectory Alignment as a caching mechanism and Online Probe Profiling
as a scheduling strategy. DPCache [12] focuses on schedule selection and uses a modified TaylorSeer
mechanism called Taylor-DP. For fair comparison with DiCache and DPCache, we keep the corre-
sponding caching mechanism fixed and replace only the schedule with ReCache. We also include a
naive step-reduction baseline, where the model is run with fewer denoising steps and no caching. All
results are grouped by caching mechanism and budget.
