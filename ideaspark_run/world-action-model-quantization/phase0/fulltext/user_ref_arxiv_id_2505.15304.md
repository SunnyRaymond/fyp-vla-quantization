# [user-supplied anchor] 2505.15304

paper_id: user_ref:arxiv_id:2505.15304
tier: U
source_used: html_arxiv
warning: none

## Intro

In recent years, deep neural network (DNN)-based policy models have significantly advanced robot manipulation and autonomous driving
[
57
,
27
,
11
,
3
,
4
,
33
]
, primarily by surpassing traditional search-based methods through imitation learning (IL) from expert data. This success has led to growing interest in adopting foundation models for large-scale IL, aiming to improve generalization and robustness beyond limited data, facilitating policy transfer across different robot embodiments, tasks, or environments
[
3
,
4
]
. Vision-Language Action (VLA) models
[
27
,
11
,
3
,
4
]
extend pre-trained vision-language models to robotics via next-token prediction, integrating visual and textual information for enhanced manipulation capabilities. Despite their potential as foundation models supporting cross-embodiment transfer, IL-based VLA models often suffer from slow inference speeds, high computational costs, and substantial memory usage
[
52
]
, making deployment on resource-constrained, battery-powered robots challenging.
Quantization techniques reduce DNN inference costs by converting full-precision (FP)
1
1
1
We use BFloat16
[
25
]
and Float32 as baseline FP for OpenVLA and CILRS, unless noted otherwise.
weights and activations to lower precision
[
6
,
12
,
34
,
16
]
, and extensive research focuses on mitigating the resulting numerical errors to preserve accuracy. Post-training quantization (PTQ) adjusts weights and activations to reduce quantization loss, whereas quantization-aware training (QAT) incorporates quantization directly in training to enhance robustness. These strategies are well studied in computer vision
[
54
,
38
,
31
,
60
]
and natural language processing
[
34
,
15
,
53
]
, but their effects on IL-based applications, such as robot manipulation or autonomous driving, remain underexplored.
In this work, we investigate how quantization errors affect IL policy decisions and note two key observations: 1) Quantization errors generally have minor impact across most timesteps, producing small action discrepancies (e.g.,
t
1
t_{1}
in Fig.
1
). 2) Certain critical states, however, experience large deviations in actions due to quantization errors (Fig.
1
-
t
2
t_{2}
: PTQ,
t
3
t_{3}
: QAT), ultimately causing mission failures (Fig.
1
-
t
4
t_{4}
). This differs from typical long-sequence prediction, where errors accumulate at each timestep to deviate from the FP policy’s action distribution; here, the quantized policy must specifically address a small number of mission-critical states.
Figure 1
:
Comparison of action discrepancy (L2-norm) between the quantized and full-precision (FP) policies in
OpenVLA
using the PTQ, QAT, and SQIL, and evaluation of the average success rate on the LIBERO benchmark. For fair comparison, episode timelines are segmented into four sub-tasks and realigned based on successful transitions.
To address this, we present saliency-aware quantized imitation learning (SQIL), a simple yet effective method for IL-based robotic control. SQIL integrates quantization into fine-tuning, increasing robustness against quantization errors. Unlike traditional QAT for supervised learning, SQIL incorporates quantization-robust action distillation (QRD): 1) identifying mission-critical states via saliency scores, and 2) applying an importance-weighted loss to emphasize correct actions at these states. Consequently, the quantized policy more closely follows the FP policy’s decisions for crucial states. As seen in Fig.
1
, SQIL suppresses action discrepancies at critical timesteps (
t
2
t_{2}
,
t
3
t_{3}
), leading to mission success and fully recovering the FP’s success rate (Fig.
1
(d)).
We validate SQIL’s generalization capability across extensive simulation benchmarks with environment variations, real-world tasks, and cross-domain tasks (self-driving, physics simulation), consistently recovering full-precision performance. For robot manipulation, evaluations of
OpenVLA
[
27
]
on the LIBERO benchmark
[
35
]
show that our 4-bit weight-quantized models achieve success rates comparable to the FP policy, delivering a 2.5× speedup and 2.5× energy savings on edge GPUs. In autonomous driving, tests of
CILRS
[
57
]
on the NoCrash benchmark
[
8
]
confirm that our 4-bit weight- and activation-quantized policy maintains the FP policy’s performance, realizing a 3.7× speedup and 3.1× energy savings on low-end GPUs. These findings mark the first successful demonstration of recovering and efficiently deploying quantized IL-based policies.
Contributions.
Building on the observations and results above, this paper makes four primary contributions:
•
First Systematic Study of Quantized IL.
To our knowledge, this is the
first comprehensive analysis of quantized IL
that identifies the significance of
mission-critical states
—moments where fine-grain control is essential. We found that most failures in quantized policies stem from coarse action control during physical interactions (e.g., grasping or releasing objects). This insight explains
why naive quantization fails
.
•
SIS: Policy-Driven Critical State Detection.
SIS uses the policy’s action sensitivity to surface mission-critical states, surpassing vision-language key-frame (KF) detectors
[
28
]
.
•
SQIL: Saliency-Aware Quantized Imitation Learning.
By coupling 4-bit QAT with SIS-weighted loss (QRD), SQIL achieve a 2–4
×
\times
speedup and energy savings while maintaining a success rate within 1% of the FP baseline.
•
Broad, Cross-Domain Validation.
Experiments on robot manipulation, autonomous driving, and MuJoCo control confirm SQIL’s generality in both simulation and real-world.

## Method

Motivated by the previous observations, we propose enabling the quantized policy to learn compensations for quantization errors in mission-critical states. The proposed method, saliency-aware quantized imitation learning (SQIL), consists of two key components: 1) a saliency-based state-importance score (SIS) to identify mission-critical states without supervision, and 2) a quantization-robust action distillation (QRD) method that selectively emphasizes learning from the SIS-identified mission-critical states.
Figure 4
:
Comparison of SIS and keyframe (KF) on trajectory.
LIBERO Benchmark
QAT
SQIL (KF)
SQIL (SIS)
Avg. Success Rate (%)
71.6 ± 0.5
72.6 ± 0.7
73.2 ± 0.6
Table 1
:
Saliency metric comparison: Keyframe (KF) and SIS.
4.1
Saliency-based State-Importance Score
From Fig.
1
, we see that mission-critical states depend on both the task and the environment, making it non-trivial to identify which states need extra attention during imitation learning. Therefore, we quantify the state importance score (SIS) to identify mission-critical states. Motivated by
[
18
]
, we maintain that states showing large action discrepancies under visual perturbation are mission-critical, as these discrepancies reveal the crucial visual regions for successful decision-making. We measure the perturbation-based action saliency at each position of a state as follows:
S
π
​
(
s
t
,
k
)
=
1
2
​
‖
π
⁡
(
s
t
)
−
π
⁡
(
ϕ
⁡
(
s
t
,
k
)
)
‖
2
,
S_{\pi}(s_{t},k)=\frac{1}{2}\left\|\pi(s_{t})-\pi\left(\phi(s_{t},k)\right)\right\|^{2},
(4)
where
ϕ
⁡
(
s
t
,
k
)
\phi(s_{t},k)
applies a perturbation (e.g. a Gaussian blur at position
k
k
. Higher
S
π
​
(
s
t
,
k
)
S_{\pi}(s_{t},k)
indicates that local modifications at
k
k
have a greater impact on the policy’s output
We then define SIS as the average saliency across all positions
k
k
:
S
​
I
​
S
π
s
t
=
𝔼
k
​
[
S
π
FP
​
(
s
t
,
k
)
]
.
SIS^{s_{t}}_{\pi}=\mathbb{E}_{k}\left[S_{\pi^{\text{FP}}}(s_{t},k)\right].
(5)
A higher SIS suggests that the state has more perturbation-sensitive positions, often found in mission-critical scenarios. As shown in Fig.
3
, SIS effectively highlights the states at the “Pick-Up” (timestep
d
d
) and “Drop” (timestep
g
g
) actions, highlighting its ability to detect moments requiring fine-grained control.
Compared to vision-language-based key-frame (KF) detection
[
28
]
, which often activates only at coarse sub-task boundaries (e.g., “drawer open”, “object placed”), SIS responds to subtle yet decisive robot–environment interactions—such as initial grasp or release—by measuring how small perturbations alter the policy’s output (Fig.
4
). Quantitatively, replacing SIS with KF in SQIL reduces success by 1.1% on LIBERO manipulation tasks (Table
1
), confirming that policy-driven saliency is essential to robust performance under quantization.
Algorithm 1
Saliency-Aware Quantized Imitation Learning
Input:
Pre-trained policy
π
FP
\pi^{\text{FP}}
, quantized policy
π
θ
Q
\pi^{\text{Q}}_{\theta}
, Expert dataset
𝒟
E
\mathcal{D}_{\text{E}}
, Number of epochs
N
N
Output:
Quantized policy
π
θ
Q
\pi^{\text{Q}}_{\theta}
Initialize
π
θ
Q
\pi^{\text{Q}}_{\theta}
from
π
FP
\pi^{\text{FP}}
Calculate
S
​
I
​
S
π
FP
SIS_{\pi^{\text{FP}}}
with
π
FP
\pi^{\text{FP}}
,
𝒟
E
\mathcal{D}_{\text{E}}
using Eq. (
5
)
for
epoch = 1 to
N
N
do
for
each trajectory
τ
i
\tau_{i}
in
𝒟
E
\mathcal{D}_{\text{E}}
do
for
each state-action pair
(
s
t
,
a
t
)
(s_{t},a_{t})
in
τ
i
\tau_{i}
do
Calculate
ℒ
QAT
\mathcal{L}^{\text{QAT}}
using Eq. (
3
)
Calculate
ℒ
QRD
\mathcal{L}^{\text{QRD}}
with
S
​
I
​
S
π
F
​
P
SIS_{\pi^{FP}}
using Eq. (
6
)
ℒ
SQIL
=
ℒ
QAT
+
ℒ
QRD
\mathcal{L}^{\text{SQIL }}=\mathcal{L}^{\text{QAT}}+\mathcal{L}^{\text{QRD}}
(Eq. (
7
))
Compute
∂
ℒ
SQIL
∂
θ
≈
∂
ℒ
SQIL
∂
θ
Q
\frac{\partial\mathcal{L}^{\text{SQIL }}}{\partial\theta}\approx\frac{\partial\mathcal{L}^{\text{SQIL }}}{\partial\theta_{\text{Q}}}
Update
π
θ
Q
\pi^{\text{Q}}_{\theta}
with
∂
ℒ
SQIL
∂
θ
\frac{\partial\mathcal{L}^{\text{SQIL }}}{\partial\theta}
end
for
end
for
end
for
return
Quantized policy
π
θ
Q
\pi^{\text{Q}}_{\theta}
4.2
Quantization-Robust Action Distillation
We selectively focus on mission-critical states using the state importance score (SIS). To achieve this, we propose quantization-robust action distillation (QRD), which leverages the FP policy and demonstration data to reduce quantization errors by distilling the FP policy’s action distributions into the quantized model. Its loss function measures the discrepancy between the quantized and FP policies’ action distributions:
ℒ
QRD
​
(
θ
)
=
α
t
⋅
𝔼
τ
i
∼
𝒟
E
​
[
1
|
τ
i
|
​
∑
s
t
∈
τ
i
D
⁡
(
π
Q
​
(
s
t
)
,
π
FP
​
(
s
t
)
)
]
,
\displaystyle\mathcal{L}^{\text{QRD}}(\theta)=\alpha_{t}\cdot\mathbb{E}_{\tau_{i}\sim\mathcal{D}_{\text{E}}}\left[\frac{1}{|\tau_{i}|}\sum_{s_{t}\in\tau_{i}}D({\pi}^{\text{Q}}(s_{t}),{\pi}^{\text{FP}}(s_{t}))\right],
(6)
where
α
t
=
β
\alpha_{t}=\beta
if
S
​
I
​
S
π
FP
s
t
>
T
SIS^{s_{t}}_{\pi^{\text{FP}}}>T
, and
α
t
=
1
\alpha_{t}=1
otherwise;
D
D
is a discrepancy metric (e.g., L2-norm), and
π
⁡
(
s
t
)
{\pi}(s_{t})
is the probability distribution of all possible actions for state
s
t
s_{t}
. The hyperparameter
β
(
>
1
)
\beta(>1)
applies extra weight to high-importance states, and
T
T
is a threshold that selects the top
p
p
=20% of SIS values. Unlike conventional KD methods, QRD employs the weighting
α
t
\alpha_{t}
to amplify the loss for states identified by SIS (i.e.,
S
​
I
​
S
π
FP
s
t
>
T
SIS^{s_{t}}_{\pi^{\text{FP}}}>T
). Training convergence is not sensitive to the hyperparameters
D
⁡
(
)
,
β
,
T
D(),\beta,T
, and we use the same values (details in Supplementary Sec.
3
) for all the experiments across robot control, self-driving, and physics simulation.
Figure 5
:
Comparison of attention visualization in
important states
(pick up the black bowl) for tasks successfully completed on the LIBERO-Spatial benchmark. Additional examples are provided in the
supplementary materials.
4.3
Saliency-Aware Quantized Imitation Learning
To enhance conventional QAT for mission-critical states, we introduce saliency-aware quantized imitation learning (SQIL), which selectively strengthens IL’s weight adjustment for salient states. SQIL’s loss function combines QAT and QRD:
ℒ
SQIL
​
(
θ
)
=
ℒ
QAT
​
(
θ
)
+
ℒ
QRD
​
(
θ
)
,
\mathcal{L}^{\text{SQIL }}(\theta)=\mathcal{L}^{\text{QAT}}(\theta)+\mathcal{L}^{\text{QRD}}(\theta),
(7)
This combined approach effectively reduces quantization errors, enabling the quantized policy to generalize comparably to the FP policy.
ℒ
QAT
\mathcal{L}^{\text{QAT}}
maximizes the quantized policy’s log-likelihood of expert actions, while
ℒ
QRD
\mathcal{L}^{\text{QRD}}
aligns the quantized policy’s action distribution with the FP policy. By applying selective weighting
α
t
\alpha_{t}
, QRD emphasizes states requiring precise control instead of uniformly minimizing the discrepancy across all states.
Algorithm
1
outlines the overall SQIL procedure. We reuse the same expert dataset and training hyperparameters, and SIS can be precomputed once for
𝒟
E
\mathcal{D}_{E}
and
π
FP
\pi^{\text{FP}}
. Consequently, SQIL offers a turnkey solution for efficiently quantizing an FP policy using an existing expert dataset. As shown in Fig.
1
, SQIL suppresses action discrepancies at mission-critical timesteps (
t
2
t_{2}
,
t
3
t_{3}
), leading to mission success and fully recovering the FP’s success rate (Fig.
1
(d)).
4.4
Analysis
In this section, we quantitatively answer two questions about SQIL’s efficacy: 1) Does it restore the disrupted saliency of the quantized policy? 2) Do QAT and QRD within SQIL work synergistically to recover the action density of the quantized policy?
Saliency Visualization
.
To qualitatively assess how quantization affects IL policy behavior and how SQIL addresses this issue, we visualize the saliency map using Eq.
4
. Fig.
5
shows a robot executing “Pick-up the black bowl” under different policies. The FP policy exhibits high saliency on objects of interest (e.g., the robot arm, the target bowl, and the destination plate). In contrast, the PTQ policy focuses on irrelevant regions, indicating that quantization errors misidentify salient areas. Although QAT and QRD alone attempt to restore expert or FP actions, applying loss with uniform importance across all states limits their ability to fix the saliency. In contrast, SQIL’s saliency map closely mirrors the FP policy, indicating that it effectively recovers the quantized policy’s desired focus. We confirm this quantitatively in Table
2
, which shows SQIL achieving lower average divergence from the FP saliency map than PTQ.
Method
Average Saliency Divergence
↓
\downarrow
LIBERO
-Spatial
LIBERO
-Object
LIBERO
-Goal
LIBERO
-Long
PTQ
0.0821
0.0268
0.0612
0.0640
SQIL
0.0559
0.0198
0.0458
0.0521
Table 2
:
Comparison of AttDiv with
OpenVLA
on the LIBERO.
Figure 6
:
Comparison of action distributions for various quantization methods using
CILRS
(W4A4) on NoCrash-dense at state
s
j
s_{j}
. The red circle denotes action
a
j
a_{j}
from the demonstration dataset
𝒟
E
\mathcal{D}_{\text{E}}
, and the diamond shape marks the highest probability point for each distribution.
Method
Quantizer (INT4)
Success Rate %
↑
\uparrow
LIBIERO-Spatial
LIBERO-Object
LIBERO-Goal
LIBERO-Long
FP
-
84.0 ± 0.9
83.9 ± 0.3
76.6 ± 0.6
50.7 ± 1.2
PTQ
AWQ
80.1 ± 0.5
81.3 ± 0.4
74.3 ± 0.6
47.2 ± 0.6
QAT
AWQ
80.9 ± 0.7
82.4 ± 0.4
75.7 ± 0.5
47.3 ± 0.3
SQIL
AWQ
83.9 ± 0.5
83.5 ± 0.
5
76.3 ± 0.4
49.2 ± 1.0
PTQ
QuaRot
81.2 ± 1.0
81.7 ± 0.6
74.8 ± 0.4
47.6 ± 0.9
QAT
QuaRot
81.8 ± 0.9
82.8 ± 0.5
75.2 ± 0.6
48.0 ± 0.8
SQIL
QuaRot
83.8 ± 0.8
83.7 ± 0.5
76.3 ± 0.5
49.4 ± 0.9
Table 3
:
Comparison of success rate across various INT4 quantization with
OpenVLA
and scenarios in the LIBERO benchmark.
Language Instruction Generalization
Illumination Generalization
Task
#Trials
# Successes
↑
\uparrow
Light
Success Rate(%)
↑
\uparrow
FP
SQIL W4
Intensity
FP
SQIL W4
Pick up
the black bowl on the wooden cabinet and
place it
on the plate
50
34
29
100%
50.7
49.2
Grab
the black bowl from the wooden cabinet and
set it
on the plate
50
25
28
80%
50.5
49.1
Remove
the black bowl from the wooden cabinet and
place it
on the plate
50
31
34
60%
49.5
48.8
Lift
the black bowl from the wooden cabinet and
set it
on the plate
50
28
27
–
–
–
Table 4
:
Generalization comparisons of
OpenVLA
: Language instruction generalization in LIBERO-Spatial (left) and illumination generalization in LIBERO-Long (right).
Action Distribution Comparison.
To evaluate the synergy of QAT and QRD, we compare the action distributions of the FP policy and policies quantized by PTQ, QAT, QRD, and SQIL in the self-driving model (CILRS). Fig.
6
depicts each policy’s distribution, with circle and diamond markers indicating the maximum-likelihood actions of the FP and quantized policies, respectively. PTQ introduces large quantization errors, causing serious divergence from FP. Although QAT aligns the maximum likelihood to expert actions, it can generate overly sharp peaks that distort the FP distribution. QRD aligns the quantized distribution with the FP policy’s shape but may overlook expert actions when the FP distribution is broad. By combining both, SQIL preserves the FP policy’s overall behavior while emphasizing high-quality expert actions, resulting in better decisions and improved performance.
