# Gated Relational Alignment via Confidence-based Distillation for Efficient VLMs

paper_id: arxiv:2601.22709v5
tier: U
source_used: html_arxiv
warning: none

## Intro

Vision-language models (VLMs) have become a central paradigm for multimodal intelligence, demonstrating strong capabilities across visual question answering, complex scene understanding, image-text reasoning, vision-language-action systems, and broader multimodal visual applications
(
Bai et al., 2023
;
Liu et al., 2023
;
Liu et al., 2024a
;
Kawaharazuka et al., 2025
;
Liang et al., 2025
)
. However, these capabilities come with substantial computational cost. Modern VLMs typically contain billions of parameters and require large memory bandwidth, making them difficult to deploy on resource-constrained platforms. This motivates low-bit compression methods that can reduce inference cost while preserving the reasoning and perception ability of full-precision models.
Figure 1
:
Throughput-accuracy Pareto analysis on LLaVA-1.5-7B. Bubble size indicates GPU memory footprint.
GRACE
achieves higher throughput than BF16 baselines while maintaining stronger accuracy than existing quantization methods.
Quantization is one of the most effective tools for efficient deployment. Post-training quantization (PTQ) is attractive because it requires little or no retraining
(
Frantar et al., 2022
;
Shao et al., 2023
;
Lin et al., 2024
)
, but aggressive INT4 quantization often causes severe degradation in VLMs due to heterogeneous multimodal distributions and cross-modal sensitivity
(
Wang et al., 2024a
;
Li et al., 2025
)
. Quantization-aware training (QAT) provides a stronger alternative by exposing the model to low-precision constraints during optimization. Recent studies in language and vision models suggest that accurate quantization requires not only preserving task performance but also explicitly managing quantization-induced errors
(
Liu et al., 2024d
;
Liu et al., 2025
;
Tian et al., 2025
)
. Nevertheless, QAT for VLMs remains underexplored, since VLMs combine visual representations, multimodal projection, and language decoding, each with different sensitivity to low-bit perturbations
(
Sun et al., 2024
;
Jin et al., 2025
)
.
Meanwhile, knowledge distillation has emerged as a powerful tool for compressing VLMs
(
Cao et al., 2025
;
Lee et al., 2025
)
. A natural idea is to combine KD with QAT, using a strong teacher to guide the low-bit student. However, this combination is non-trivial. First, teacher predictions are not uniformly reliable: high-entropy or biased predictions may introduce noisy supervision. Second, a quantized student has limited representational capacity and cannot faithfully absorb all teacher information. This issue is especially important for VLMs, where predictions can be affected by prior preferences, class-level confusion, and prompt-induced uncertainty
(
Tian et al., 2026
)
. Therefore, effective QAT-based VLM compression requires a mechanism that identifies which teacher information is worth preserving under a strict bit budget.
We interpret this problem through the Information Bottleneck (IB) principle
(
Tishby et al., 2000
;
Tishby and Zaslavsky, 2015
)
. Quantization imposes a hard capacity constraint by reducing representational precision, while distillation provides dense supervision about task-relevant information. From this perspective, VLM compression becomes a capacity allocation problem: the quantized student must retain high-value teacher knowledge while discarding noisy or redundant information. Standard QAT relies mainly on task losses, which are often too sparse to guide this allocation, especially for multimodal tasks where supervision is distributed across visual tokens, text tokens, and answer logits. We therefore use the teacher as a proxy for task relevance and dynamically regulate distillation according to confidence and information preservation.
Building upon this insight, we propose
GRACE
(
G
ated
R
elational
A
lignment via
C
onfidence-based Distillation for
E
fficient VLMs), a unified framework for QAT-based VLM compression. GRACE consists of three complementary components: (i)
confidence-gated decoupled knowledge distillation
, which suppresses noisy supervision from uncertain teacher predictions; (ii)
relational centered kernel alignment
, which transfers visual-token relational structure rather than point-wise features; and (iii) an
adaptive information-bottleneck controller
, which dynamically balances teacher guidance with the student’s quantized capacity. For low-bit training, GRACE further employs group-wise learned step-size quantization.
Extensive experiments on LLaVA-1.5 and Qwen2-VL demonstrate the effectiveness of GRACE. Our distilled LLaVA-1.5-7B achieves 69.0% average accuracy, improving the 7B baseline by
3.8%
and nearly matching the 13B teacher. More importantly, as shown in Figure
1
, our INT4 Qwen2-VL-2B not only recovers but surpasses the full-precision baseline across all benchmarks, such as
79.1 vs. 73.7
on ScienceQA. GRACE also delivers significant inference speedup and memory reduction when deployed with real INT4 kernels.
Our contributions are summarized as follows:
•
We formulate QAT-based VLM compression from an Information Bottleneck perspective, connecting low-bit quantization, capacity allocation, and teacher-guided knowledge preservation.
•
We propose GRACE, a unified framework that combines confidence-gated distillation, relational CKA, adaptive IB control, and group-wise learned step-size quantization for efficient VLM compression.
•
We show that GRACE enables INT4-quantized VLMs to surpass BF16 baselines while achieving substantial inference speedup and memory reduction.
Conflict of Interest Disclosure.
All authors are affiliated with academic institutions (ETH Zurich and University of Bologna) and declare no financial conflicts of interest; the models evaluated in this work (LLaVA-1.5 and Qwen2-VL) are publicly released open-source models.
Figure 2
:
Correlation between teacher entropy and error rate on ScienceQA (LLaVA-1.5 13B). Higher entropy consistently corresponds to higher prediction error, supporting entropy as a confidence signal for gated distillation.
Figure 3
:
Multi-layer attention visualization of LLaVA-1.5 13B (top) and 7B (bottom). Given the question “What object is being used as the telephone receiver?”, the 13B model progressively localizes the banana, while the 7B model shows scattered attention.

## Method

3.1
Overview
Quantization restricts the capacity of the model to a fixed bit budget, forcing the network to prioritize which information to retain. We present GRACE, a framework that formulates this capacity allocation problem through the lens of the Information Bottleneck principle: the quantized student must retain task-relevant knowledge from the teacher while discarding redundant information that cannot be faithfully represented under bit-width constraints. As illustrated in Figure
4
, a frozen teacher and a trained quantized student process the same input to produce output distributions
P
T
P_{T}
and
P
S
P_{S}
, respectively. The student is trained using a combination of cross-entropy loss, confidence-gated decoupled distillation loss, and relational alignment loss, with the distillation weight dynamically adjusted by an adaptive IB controller. We provide a detailed description of each component in the following.
3.2
Confidence-Gated Decoupled Knowledge Distillation
Standard knowledge distillation
(
Hinton et al., 2015
)
minimizes the KL divergence between teacher and student output distributions. However, this approach treats all teacher predictions equally, ignoring the varying reliability of teacher supervision across different tokens. We address this limitation through two mechanisms: decoupled knowledge distillation and confidence-based gating.
Decoupled Knowledge Distillation.
Following
(
Zhao et al., 2022
)
, we decompose the distillation loss into two components that capture distinct aspects of the teacher’s knowledge. Let
P
T
P_{T}
and
P
S
P_{S}
denote the probability distributions of the teacher and student on the vocabulary, and let
y
y
be the ground-truth label. We define the following two components of the distillation loss:
•
Target Class Knowledge Distillation (TCKD):
This component captures the teacher’s confidence in the correct answer by comparing binary distributions over the target versus non-target classes:
ℒ
TCKD
=
D
KL
(
[
P
T
t
,
1
−
P
T
t
]
∥
[
P
S
t
,
1
−
P
S
t
]
)
\mathcal{L}_{\text{TCKD}}=D_{\text{KL}}\left([P_{T}^{t},1-P_{T}^{t}]\,\|\,[P_{S}^{t},1-P_{S}^{t}]\right)
(1)
where
P
T
t
=
P
T
​
(
y
)
P_{T}^{t}=P_{T}(y)
and
P
S
t
=
P
S
​
(
y
)
P_{S}^{t}=P_{S}(y)
are the probabilities assigned to the target class.
•
Non-target Class Knowledge Distillation (NCKD):
This term transfers the “dark knowledge” embedded in the teacher’s distribution over incorrect classes:
ℒ
NCKD
=
D
KL
(
P
^
T
nt
∥
P
^
S
nt
)
\mathcal{L}_{\text{NCKD}}=D_{\text{KL}}\left(\hat{P}_{T}^{\text{nt}}\,\|\,\hat{P}_{S}^{\text{nt}}\right)
(2)
where
P
^
nt
\hat{P}^{\text{nt}}
denotes the renormalized distribution over non-target classes.
The per-token
Decoupled Knowledge Distillation
(DKD) loss combines these components:
ℒ
DKD
(
i
)
=
α
⋅
ℒ
TCKD
(
i
)
+
β
dkd
⋅
ℒ
NCKD
(
i
)
\mathcal{L}_{\text{DKD}}^{(i)}=\alpha\cdot\mathcal{L}_{\text{TCKD}}^{(i)}+\beta_{\text{dkd}}\cdot\mathcal{L}_{\text{NCKD}}^{(i)}
(3)
where
i
i
indexes tokens and
α
\alpha
,
β
dkd
\beta_{\text{dkd}}
are weighting coefficients. TCKD captures the teacher’s confidence in the ground-truth token, while NCKD transfers the relational structure among all other tokens in the vocabulary. Following
(
Zhao et al., 2022
)
, we set
β
dkd
>
α
\beta_{\text{dkd}}>\alpha
to emphasize the rich “dark knowledge” encoded in non-target distributions, which has been shown to be more informative for knowledge transfer than target-class probabilities alone.
Confidence-Based Gating.
As demonstrated in Section
2.1
, teacher entropy exhibits a strong correlation with prediction errors, indicating that high-entropy outputs constitute unreliable supervision signals. Motivated by this observation, we introduce a confidence-based gating mechanism that adaptively modulates the distillation loss according to teacher certainty.
For each token
i
i
, we compute the entropy of the teacher’s output distribution:
H
i
=
H
(
P
T
(
i
)
)
=
−
∑
v
P
T
(
i
)
(
v
)
log
P
T
(
i
)
(
v
)
H_{i}=H(P_{T}^{(i)})=-\sum_{v}P_{T}^{(i)}(v)\log P_{T}^{(i)}(v)
(4)
We normalize this entropy as
h
~
i
=
H
i
/
log
⁡
|
V
|
∈
[
0
,
1
]
\tilde{h}_{i}=H_{i}/\log|V|\in[0,1]
, where
|
V
|
|V|
denotes the vocabulary size. The confidence weight for token
i
i
is then defined as:
g
i
=
exp
⁡
(
−
h
~
i
)
g_{i}=\exp\left(-\tilde{h}_{i}\right)
(5)
This exponential formulation assigns high weights to confident teacher predictions (low entropy) while suppressing noisy supervision from uncertain predictions (high entropy). The gated DKD loss aggregates per-token losses with confidence weighting:
ℒ
GDKD
=
∑
i
g
i
⋅
ℒ
DKD
(
i
)
∑
i
g
i
\mathcal{L}_{\text{GDKD}}=\frac{\sum_{i}g_{i}\cdot\mathcal{L}_{\text{DKD}}^{(i)}}{\sum_{i}g_{i}}
(6)
where the summation is over all valid tokens in the batch.
Information-Theoretic Justification.
The gated distillation loss admits a principled interpretation under the Information Bottleneck framework. By defining the importance-reweighted distribution:
p
g
​
(
i
)
≜
g
i
∑
j
g
j
p_{g}(i)\triangleq\frac{g_{i}}{\sum_{j}g_{j}}
(7)
we can express
ℒ
GDKD
=
𝔼
p
g
​
[
ℒ
DKD
(
i
)
]
\mathcal{L}_{\text{GDKD}}=\mathbb{E}_{p_{g}}[\mathcal{L}_{\text{DKD}}^{(i)}]
. This reveals that confidence gating implements a
bottleneck on the supervision signal
, allocating distillation capacity towards tokens where the teacher posterior is sharp (low entropy), i.e., where the teacher provides higher information content.
Theorem 3.1
(Effect of Confidence Gating)
.
Let
w
i
=
g
i
/
∑
j
g
j
w_{i}=g_{i}/\sum_{j}g_{j}
denote the normalized weights. The gated loss satisfies the following:
ℒ
GDKD
=
ℒ
¯
DKD
+
N
⋅
Cov
⁡
(
w
i
,
ℒ
DKD
(
i
)
)
\mathcal{L}_{\text{GDKD}}=\bar{\mathcal{L}}_{\text{DKD}}+N\cdot\mathrm{Cov}\left(w_{i},\mathcal{L}_{\text{DKD}}^{(i)}\right)
(8)
where
ℒ
¯
DKD
=
1
N
​
∑
i
ℒ
DKD
(
i
)
\bar{\mathcal{L}}_{\text{DKD}}=\frac{1}{N}\sum_{i}\mathcal{L}_{\text{DKD}}^{(i)}
is the unweighted average. Since
w
i
w_{i}
decreases monotonically with
h
~
i
\tilde{h}_{i}
, positive correlation between entropy and loss implies
Cov
⁡
(
w
i
,
ℒ
DKD
(
i
)
)
<
0
\mathrm{Cov}(w_{i},\mathcal{L}_{\text{DKD}}^{(i)})<0
.
The proof is provided in Appendix
B.2
.
KL Divergence as Information Gap.
Beyond filtering unreliable supervision, the KL-based distillation objective itself admits an information-theoretic interpretation. Let
Y
T
Y_{T}
denote a pseudo-label sampled from the teacher distribution, i.e.,
Y
T
∣
X
=
x
∼
P
T
(
⋅
∣
x
)
Y_{T}\mid X=x\sim P_{T}(\cdot\mid x)
, and let
Z
S
=
f
S
​
(
X
)
Z_{S}=f_{S}(X)
represent the student’s learned representation. Using the student’s output
P
S
P_{S}
as a variational decoder yields:
Proposition 3.2
(Variational Lower Bound and KL Gap)
.
I
(
Z
S
;
Y
T
)
≥
I
(
X
;
Y
T
)
−
𝔼
[
D
KL
(
P
T
∥
P
S
)
]
I(Z_{S};Y_{T})\geq I(X;Y_{T})-\mathbb{E}\left[D_{\text{KL}}\left(P_{T}\|P_{S}\right)\right]
(9)
The proof is provided in Appendix
B.3
.
This result provides a clean interpretation: the KL divergence quantifies the
information gap
between the maximum teacher information
I
⁡
(
X
,
Y
T
)
I(X;Y_{T})
and the information captured by the student
I
⁡
(
Z
S
,
Y
T
)
I(Z_{S};Y_{T})
. Consequently, minimizing
ℒ
GDKD
\mathcal{L}_{\text{GDKD}}
directly maximizes the mutual information between the student’s representation and the teacher’s knowledge.
3.3
Relational Centered Kernel Alignment
As demonstrated in Section
2.2
, larger teacher models develop superior visual attention patterns that logit-level distillation alone cannot transfer. While relational knowledge distillation methods
(
Park et al., 2019
)
have shown that transferring inter-sample structural relationships improves student learning, these approaches operate at the batch level and fail to capture the fine-grained
intra-sample
relational structures among visual tokens that are critical for visual reasoning. We propose Relational Centered Kernel Alignment (RCKA) to explicitly transfer this token-level relational knowledge to the student.
Figure 5
:
Visualization of pairwise visual token similarity from LLaVA-1.5 13B. The heatmap shows cosine similarity between token 0 (yellow box, located in the sky region) and all other tokens in the spatial grid, where Patch X and Patch Y denote the horizontal and vertical grid coordinates respectively.
Relational Structure in Visual Representations.
We first investigate the internal structure of teacher representations. As illustrated in Figure
5
, we visualize pairwise similarities between visual tokens from the 13B teacher. By selecting a token from the sky region (yellow box) and computing its similarity with all other tokens, we observe that the sky token exhibits high similarity with other sky tokens while showing minimal similarity with the aircraft or ground regions. This demonstrates that the teacher’s intermediate hidden states from the language model backbone encode rich relational structures, organizing visual tokens into semantically coherent regions.
RCKA for Relational Knowledge Transfer.
Motivated by these observations, we propose
Relational Centered Kernel Alignment (RCKA)
to transfer the teacher’s relational visual knowledge to the student. While CKA
(
Kornblith et al., 2019
)
has been explored for knowledge distillation
(
Saha et al., 2022
;
Chen et al., 2025
)
, our approach differs in several key aspects: (i) existing methods perform
layer-wise
alignment between corresponding layers, whereas RCKA targets
intra-sample
relational structures among visual tokens; (ii) prior relational KD methods like RKD
(
Park et al., 2019
)
capture
inter-sample
relationships at the batch level, while RCKA preserves fine-grained pairwise similarities that organize visual tokens into semantically coherent regions (e.g., sky tokens clustering together, as shown in Figure
5
); and (iii) RCKA specifically addresses VLMs, where visual tokens processed through the LLM backbone develop rich relational structures critical for visual reasoning, which logit-level distillation cannot transfer.
Specifically, let
V
T
∈
ℝ
n
×
d
T
V_{T}\in\mathbb{R}^{n\times d_{T}}
and
V
S
∈
ℝ
n
×
d
S
V_{S}\in\mathbb{R}^{n\times d_{S}}
denote the visual token representations extracted from the penultimate layers of the teacher and student LLM backbones, respectively, where
n
n
is the number of visual tokens. Text tokens are excluded entirely from the Gram matrix computation to ensure RCKA transfers visual perception capabilities without conflating visual and linguistic representations. We select the penultimate layer because prior work
(
Kornblith et al., 2019
;
Chen et al., 2025
)
has demonstrated that intermediate layers capture richer semantic features than the final layer, which tends to be task-specific.
Gram Matrix Computation.
We compute normalized Gram matrices that capture pairwise similarities among visual tokens:
K
T
=
V
¯
T
​
V
¯
T
⊤
,
K
S
=
V
¯
S
​
V
¯
S
⊤
K_{T}=\bar{V}_{T}\bar{V}_{T}^{\top},\quad K_{S}=\bar{V}_{S}\bar{V}_{S}^{\top}
(10)
where
V
¯
=
V
/
‖
V
‖
2
\bar{V}=V/\|V\|_{2}
denotes row-wise
ℓ
2
\ell_{2}
normalization. Each entry
(
K
)
i
​
j
(K)_{ij}
measures the cosine similarity between visual tokens
i
i
and
j
j
.
Centered Kernel Alignment.
To ensure invariance to the mean of representations, we apply centering to the Gram matrices:
K
~
=
H
​
K
​
H
,
where
H
=
I
n
−
1
n
​
𝟏
n
​
𝟏
n
⊤
\tilde{K}=HKH,\quad\text{where}\quad H=I_{n}-\frac{1}{n}\mathbf{1}_{n}\mathbf{1}_{n}^{\top}
(11)
Here,
I
n
I_{n}
is the identity matrix and
𝟏
n
\mathbf{1}_{n}
is the all-ones vector.
To compare the relational structures encoded in
K
~
T
\tilde{K}_{T}
and
K
~
S
\tilde{K}_{S}
, we adopt Centered Kernel Alignment (CKA)
(
Kornblith et al., 2019
)
, which builds upon the Hilbert-Schmidt Independence Criterion (HSIC). HSIC is a kernel-based statistical measure of dependence between two sets of variables: given centered kernel matrices, it computes their normalized inner product in the space of Hilbert-Schmidt operators as
HSIC
​
(
K
T
,
K
S
)
=
1
(
n
−
1
)
2
​
Tr
​
(
K
~
T
​
K
~
S
)
\text{HSIC}(K_{T},K_{S})=\frac{1}{(n-1)^{2}}\text{Tr}(\tilde{K}_{T}\tilde{K}_{S})
. Intuitively, HSIC quantifies how well the pairwise relationships in one representation predict those in another. If visual tokens that are similar in the teacher’s representation are also similar in the student’s, HSIC will be high.
CKA normalizes HSIC to obtain a similarity measure bounded in
[
0
,
1
]
[0,1]
:
CKA
​
(
K
T
,
K
S
)
=
HSIC
​
(
K
T
,
K
S
)
HSIC
​
(
K
T
,
K
T
)
⋅
HSIC
​
(
K
S
,
K
S
)
\text{CKA}(K_{T},K_{S})=\frac{\text{HSIC}(K_{T},K_{S})}{\sqrt{\text{HSIC}(K_{T},K_{T})\cdot\text{HSIC}(K_{S},K_{S})}}
(12)
This normalization is crucial for our setting: it renders CKA invariant to isotropic scaling of representations and, importantly, enables meaningful comparison even when the teacher and student have different hidden dimensions (
d
T
≠
d
S
d_{T}\neq d_{S}
). Unlike point-wise alignment methods such as MSE that require matching dimensions, CKA operates on
n
×
n
n\times n
Gram matrices and thus bypasses the need for projection layers.
The RCKA loss encourages the student to preserve the teacher’s relational structure:
ℒ
RCKA
=
1
−
CKA
​
(
K
T
,
K
S
)
\mathcal{L}_{\text{RCKA}}=1-\text{CKA}(K_{T},K_{S})
(13)
By aligning relational structures rather than point-wise features, RCKA facilitates knowledge transfer even when
d
T
≠
d
S
d_{T}\neq d_{S}
, enabling the student to acquire the geometric properties underlying the teacher’s superior visual perception and progressive attention refinement.
3.4
Group-wise Learned Step-size Quantization
To enable efficient low-bit inference, we employ QAT with learned step sizes
(
Esser et al., 2019
)
. Unlike PTQ methods that fix quantization scales after calibration, our method treats the scales as learnable parameters and optimizes them jointly with model weights under the distillation objective.
Group-wise Quantization.
VLM weights exhibit heterogeneous distributions across layers and channels due to their multimodal structure
(
Wang et al., 2024a
;
Guo et al., 2025
)
. Per-tensor quantization is often too coarse to capture such variation, while channel-wise quantization still ignores fine-grained differences within each channel. Inspired by microscaling formats
(
Rouhani et al., 2023
)
, we adopt group-wise quantization, which partitions weights into contiguous groups and assigns each group an independent scale.
For a weight matrix
W
W
, we flatten it and partition it into
G
=
d
out
​
d
in
/
g
G=d_{\mathrm{out}}d_{\mathrm{in}}/g
groups, denoted as
W
→
{
W
i
}
i
=
0
G
−
1
W\rightarrow\{W_{i}\}_{i=0}^{G-1}
with
W
i
∈
ℝ
g
W_{i}\in\mathbb{R}^{g}
. We use
g
=
128
g=128
by default. Each group has a learnable log-scale
θ
i
\theta_{i}
with
s
i
=
exp
⁡
(
θ
i
)
s_{i}=\exp(\theta_{i})
, which guarantees positive scales and stabilizes multiplicative updates. Quantization is applied only to the LLM decoder linear layers, while the vision-side and output-related modules remain in BF16 to preserve visual encoding and output distribution quality.
Learned Step-size Quantization.
Following LSQ
(
Esser et al., 2019
)
, we parameterize scales in log space and initialize each group scale by minimizing fake-quantization reconstruction error. For group
W
i
W_{i}
, we sample
K
=
20
K=20
candidate scales from
𝒮
i
=
[
0.3
,
1.2
]
⋅
max
⁡
(
|
W
i
|
)
/
Q
p
\mathcal{S}_{i}=[0.3,1.2]\cdot\max(|W_{i}|)/Q_{p}
. We define
q
i
​
(
s
)
\displaystyle q_{i}(s)
=
clip
⁡
(
⌊
W
i
/
s
⌉
,
−
Q
n
,
Q
p
)
,
\displaystyle=\mathrm{clip}\left(\left\lfloor W_{i}/s\right\rceil,-Q_{n},Q_{p}\right),
(14)
s
i
(
0
)
\displaystyle s_{i}^{(0)}
=
arg
⁡
min
s
∈
𝒮
i
⁡
‖
W
i
−
s
​
q
i
​
(
s
)
‖
2
2
.
\displaystyle=\arg\min_{s\in\mathcal{S}_{i}}\left\|W_{i}-sq_{i}(s)\right\|_{2}^{2}.
We set
θ
i
(
0
)
=
log
⁡
s
i
(
0
)
\theta_{i}^{(0)}=\log s_{i}^{(0)}
. This MSE-based initialization is particularly important for 4-bit QAT, where the quantization grid is coarse and max-based scales can leave large rounding or clipping errors. During training, each group is fake-quantized as
W
^
i
=
s
i
​
q
i
​
(
s
i
)
\widehat{W}_{i}=s_{i}q_{i}(s_{i})
, where
⌊
⋅
⌉
\lfloor\cdot\rceil
denotes round-to-nearest. We use the straight-through estimator for rounding, allowing both
W
W
and
θ
\theta
to receive gradients from the downstream distillation objectives.
Quantization as an Information Capacity Constraint.
The Information Bottleneck principle
(
Tishby et al., 2000
)
seeks to preserve task-relevant information while limiting representation complexity. In our setting, low-bit quantization naturally imposes such a constraint by reducing each weight from 16-bit precision to
b
b
-bit precision. We therefore view quantized distillation as preserving teacher-relevant information under a hard capacity limit:
max
⁡
I
⁡
(
Z
S
,
Y
T
)
s
.
t
.
I
⁡
(
Z
S
,
X
)
≤
C
b
.
\max I(Z_{S};Y_{T})\quad\mathrm{s.t.}\quad I(Z_{S};X)\leq C_{b}.
(15)
Here,
Z
S
Z_{S}
denotes the student representation and
Y
T
Y_{T}
denotes teacher-provided task-relevant information. Since the quantizer physically enforces the capacity limit
C
b
C_{b}
, we do not add a separate penalty on
I
⁡
(
Z
S
,
X
)
I(Z_{S};X)
. Instead, we require the quantized student to retain sufficient teacher knowledge by constraining the distillation loss, written as
min
⁡
ℒ
task
\min\mathcal{L}_{\mathrm{task}}
subject to
ℒ
distill
≤
τ
\mathcal{L}_{\mathrm{distill}}\leq\tau
. The corresponding Lagrangian is
ℒ
task
+
β
⁡
(
ℒ
distill
−
τ
)
,
\mathcal{L}_{\mathrm{task}}+\beta\left(\mathcal{L}_{\mathrm{distill}}-\tau\right),
(16)
where
β
\beta
is adapted online through projected dual ascent on the smoothed distillation loss, as described in Sec.
3.2
. This formulation connects quantization and distillation: quantization limits the student’s information capacity, while distillation guides how this limited capacity should be allocated.
Joint Optimization.
The model weights
W
W
and log-scales
θ
\theta
are jointly optimized with
ℒ
total
=
ℒ
CE
+
β
​
ℒ
GDKD
+
ω
​
ℒ
RCKA
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{CE}}+\beta\mathcal{L}_{\mathrm{GDKD}}+\omega\mathcal{L}_{\mathrm{RCKA}}
.
Their updates are
W
←
W
−
η
W
​
∇
W
ℒ
total
,
θ
←
θ
−
η
θ
​
∇
θ
ℒ
total
.
W\leftarrow W-\eta_{W}\nabla_{W}\mathcal{L}_{\mathrm{total}},\qquad\theta\leftarrow\theta-\eta_{\theta}\nabla_{\theta}\mathcal{L}_{\mathrm{total}}.
(17)
We place
θ
\theta
in a separate optimizer group with no weight decay and use
η
θ
=
10
​
η
W
\eta_{\theta}=10\eta_{W}
. This larger learning rate helps the scales quickly track the changing weight distribution during QAT, which is especially important in early training. Empirically, this follows standard LSQ practice
(
Esser et al., 2019
)
and improves 4-bit stability. The joint update allows weights to migrate toward the quantization grid while the scales co-evolve to preserve task-relevant information.
Table 1:
Comparison with knowledge distillation methods on LLaVA-1.5. All student models use Vicuna-7B as the language backbone. Best results among 7B models are
bolded
, second best are
underlined
.
Method
VQA
v2
GQA
TextVQA
VizWiz
POPE
SQA
MME
MMB
Avg.
LLaVA-1.5-13B (Teacher)
80.0
63.5
61.3
53.6
85.9
71.6
1531.3
67.7
69.1
LLaVA-1.5-7B (Baseline)
78.5
62.0
58.2
50.0
85.9
66.8
1510.7
64.3
66.5
MoVE-KD-v1.0
[CVPR’25]
79.5
63.2
58.3
52.3
86.9
69.3
1524.5
66.3
68.0
↑
\uparrow
1.5%
MoVE-KD-v1.1
[CVPR’25]
79.9
63.9
59.6
52.7
86.3
69.8
1509.1
67.4
68.5
↑
\uparrow
2.0%
HAWAII
[NeurIPS’25]
79.1
62.8
58.7
53.9
87.3
70.5
1540.2
66.9
68.5
↑
\uparrow
2.0%
GRACE (Ours)
79.7
63.3
61.5
52.9
86.2
71.7
1525.8
66.6
69.0
↑
\uparrow
2.5%
Table 2:
Comparison with quantization methods on LLaVA-1.5-7B. Best results among INT4 models are
bolded
, second best are
underlined
.
Bitwidth
Method
VQA
v2
GQA
TextVQA
VizWiz
POPE
SQA
MMB
Avg.
BF16
LLaVA-1.5-13B (Teacher)
80.0
63.5
61.3
53.6
85.9
71.6
67.7
69.1
BF16
LLaVA-1.5-7B (Baseline)
78.5
62.0
58.2
50.0
85.9
66.8
64.3
66.5
8-bit
GRACE (Ours)
79.2
62.8
60.4
52.5
85.9
71.3
66.1
68.3
↑
\uparrow
1.8%
4-bit
RTN
76.4
61.2
57.6
48.5
84.7
65.3
62.6
65.2
↓
\downarrow
1.3%
4-bit
AWQ
[MLSys’24]
77.0
61.3
57.2
49.3
85.1
66.2
63.1
65.6
↓
\downarrow
0.9%
4-bit
QAT
77.3
61.5
57.3
49.8
85.0
66.5
63.9
65.9
↓
\downarrow
0.6%
4-bit
GRACE (Ours)
78.3
61.2
59.2
51.6
85.2
70.1
65.0
67.2
↑
\uparrow
0.7%
3.5
Adaptive IB Controller
Having established that confidence-gated distillation maximizes
I
⁡
(
Z
S
,
Y
T
)
I(Z_{S};Y_{T})
(Section
3.2
) while quantization constrains
I
⁡
(
Z
S
,
X
)
I(Z_{S};X)
(Section
3.4
), we now derive a principled mechanism to balance these competing objectives. Since directly estimating mutual information is intractable for high-dimensional vocabularies, we employ
ℒ
GDKD
\mathcal{L}_{\text{GDKD}}
as a surrogate, justified by Proposition
3.2
and consistent with practical IB applications
(
Alemi et al., 2016
;
Fischer, 2020
)
. This yields the following constrained optimization formulation:
min
θ
⁡
ℒ
CE
​
(
θ
)
s.t.
ℒ
GDKD
​
(
θ
)
≤
τ
\min_{\theta}\mathcal{L}_{\text{CE}}(\theta)\quad\text{s.t.}\quad\mathcal{L}_{\text{GDKD}}(\theta)\leq\tau
(18)
where
τ
\tau
serves as the
information preservation budget
, controlling the minimum amount of teacher knowledge to be retained. The Lagrangian relaxation yields:
ℒ
⁡
(
θ
,
β
)
=
ℒ
CE
​
(
θ
)
+
β
⁡
(
ℒ
GDKD
​
(
θ
)
−
τ
)
,
β
≥
0
\mathcal{L}(\theta,\beta)=\mathcal{L}_{\text{CE}}(\theta)+\beta\left(\mathcal{L}_{\text{GDKD}}(\theta)-\tau\right),\quad\beta\geq 0
(19)
Rather than treating
β
\beta
as a fixed hyperparameter, we update it via projected dual ascent with EMA smoothing:
β
←
Π
[
β
min
,
β
max
]
​
(
β
+
η
⋅
(
ℒ
^
GDKD
−
τ
)
)
\beta\leftarrow\Pi_{[\beta_{\min},\beta_{\max}]}\left(\beta+\eta\cdot(\widehat{\mathcal{L}}_{\text{GDKD}}-\tau)\right)
(20)
This update rule implements intuitive feedback: when
ℒ
^
GDKD
>
τ
\widehat{\mathcal{L}}_{\text{GDKD}}>\tau
, the student struggles to retain teacher knowledge, prompting
β
\beta
to increase; conversely, when the constraint is satisfied,
β
\beta
decreases to prioritize task performance.
The complete GRACE training objective combines all components:
ℒ
total
=
ℒ
CE
+
β
⋅
ℒ
GDKD
+
ω
⋅
ℒ
RCKA
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{CE}}+\beta\cdot\mathcal{L}_{\text{GDKD}}+\omega\cdot\mathcal{L}_{\text{RCKA}}
(21)
where
β
\beta
adapts dynamically via Eq. (
20
) and
ω
\omega
is a fixed weight for relational alignment. A detailed derivation connecting this formulation to IB theory is provided in Appendix
B.4
.
Table 3:
Ablation study on distillation components. GDKD: Confidence-Gated Decoupled Knowledge Distillation; RCKA: Relational Centered Kernel Alignment; Adaptive IB: Adaptive Information Bottleneck Controller.
GDKD
RCKA
Adaptive IB
MMBench
SEED-Bench
ScienceQA
Avg.
Qwen2-VL-7B (Teacher)
80.7
76.9
83.3
80.3
Qwen2-VL-2B (Baseline)
71.6
72.7
73.7
72.7
✗
✗
✗
74.2
73.3
76.8
74.8
(+2.1%)
✓
✗
✗
76.1
74.2
79.4
76.6
(+3.9%)
✗
✓
✗
76.5
74.7
79.2
76.8
(+4.1%)
✗
✗
✓
75.3
73.7
77.6
75.5
(+2.8%)
✓
✓
✗
77.1
75.1
80.2
77.5
(+4.8%)
✓
✗
✓
76.8
74.6
79.9
77.1
(+4.4%)
✗
✓
✓
76.9
75.0
79.8
77.2
(+4.5%)
✓
✓
✓
77.9
75.7
81.0
78.2
(+5.5%)
Table 4:
Ablation study on quantization. We compare QAT alone, QAT with naive KD, and QAT with GRACE.
Precision
KD Method
MMBench
SEED-Bench
ScienceQA
Avg.
BF16
Qwen2-VL-7B (Teacher)
80.7
76.9
83.3
80.3
BF16
Qwen2-VL-2B (Baseline)
71.6
72.7
73.7
72.7
8-bit
QAT only
71.2
72.3
73.0
72.2
(
−
-
0.5%)
8-bit
+ Naive KD
73.8
73.2
76.3
74.4
(+1.7%)
8-bit
+
GRACE
77.4
75.5
80.8
77.9
(+5.2%)
4-bit
QAT only
70.4
71.4
72.2
71.3
(
−
-
1.4%)
4-bit
+ Naive KD
73.0
72.1
75.2
73.4
(+0.7%)
4-bit
+
GRACE
76.9
75.6
79.1
77.2
(+4.5%)
