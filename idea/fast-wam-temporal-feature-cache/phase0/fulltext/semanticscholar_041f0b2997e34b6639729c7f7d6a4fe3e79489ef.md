# BlockVLA: Accelerating Autoregressive VLA via Block Diffusion Finetuning

paper_id: semanticscholar:041f0b2997e34b6639729c7f7d6a4fe3e79489ef
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Autoregressive (AR) models have demonstrated exceptional performance in multimodal understanding via the
next-token prediction paradigm. In the realm of discrete Vision-Language-Action Models (VLA) in robotics,
frameworks (e.g. RT-2 (Zitkovich et al., 2023), OpenVLA (Kim et al., 2025), π0-FAST
(Pertsch et al.,
2025)) tokenize continuous actions into discrete tokens, enabling models to inherit the profound reasoning and
grounding capabilities of web-scale pretrained Vision-Language Models (VLMs) (Karamcheti et al., 2024;
Beyer et al., 2024; Yang et al., 2025a; Chen et al., 2025) through large-scale robotic pretraining (O’Neill
et al., 2024; Ebert et al., 2022; Khazatsky et al., 2024; Bu et al., 2025) and subsequent supervised fine-tuning
(SFT) on specific tasks (Bai et al., 2025). However, the inherent sequential decoding of AR models imposes
critical bottlenecks: high inference latency and compounding execution errors during long-horizon tasks, which
frequently lead to catastrophic failure. To address these limitations, discrete Diffusion Language Models
(dLLM) (Liang et al., 2025; Wen et al., 2025; Ye et al., 2025a; Chen et al., 2026) have emerged as a promising
alternative. As shown in Table 1, by leveraging discrete diffusion processes, dLLMs enable parallel decoding
to accelerate inference while effectively mitigating cumulative errors. While dLLMs, including dVLMs (Austin
et al., 2021; Nie et al., 2025; Bie et al., 2025; Ye et al., 2025b; You et al., 2025; Yang et al., 2025b), promise
to alleviate the sequential decoding bottleneck through parallel token refinement, their practical speedup in
robotic applications remains constrained. The iterative denoising process typically requires a large number of
denoising function evaluations (NFEs), where each NFE denotes one model forward pass during refinement, to
produce high-fidelity action sequences. In addition, the bidirectional iterative decoding of conventional dLLMs
limits the direct reuse of prefix states through standard KV caching (Liu et al., 2025; Wu et al., 2026b),
often offsetting the throughput gains from parallelization. Beyond inference efficiency, adapting dLLMs to
robotics also faces a model availability challenge. Modern VLMs are still overwhelmingly dominated by AR
arXiv:2605.13382v1  [cs.RO]  13 May 2026

Figure 1 Quantitative results on training efficiency and inference throughput. (a) Average success rate on LIBERO: Our
BlockVLA outperforms the baseline in overall average success rate across four task suites and exhibits superior training
efficiency with significantly faster convergence. (b) Inference throughput comparison: Our BlockVLA achieves 3.3×
acceleration than the DDVLA baseline on a single NVIDIA RTX 4090 GPU.
backbones, while mature diffusion-native VLMs remain scarce. Given the prohibitive cost of training dLLMs
from scratch, adapting pretrained AR-based VLMs to the diffusion paradigm (Gong et al., 2025; Zeng et al.,
2025; Cheng et al., 2025; Fu et al., 2025) offers a practical way to inherit large-scale visual-linguistic knowledge.
Nevertheless, this adaptation is challenging because the causal factorization of AR models differs from the
bidirectional iterative refinement used in diffusion models, creating optimization and deployment barriers for
efficient robotic policy learning.
Table 1 Comparison of different modeling paradigms for VLA models.
Feature
Autoregressive
Discrete Diffusion
Block Diffusion (Ours)
Less Cumulative Error
✗
✓
✓
KV-Caching
✓
✗
✓
Parallel Decoding
✗
✓
✓
Efficient Seq Scaling
✓
✗
✓
Inference speed
Slow
Fast
Faster
To address these limitations, we propose BlockVLA, which adapts the block diffusion paradigm (Arriola
et al., 2025a; Fu et al., 2025; Wu et al., 2026a; Chen et al., 2024) to robotic manipulation. Unlike standard
dLLMs (Arriola et al., 2025a; Cheng et al., 2025) that treat the entire sequence as homogeneous training
targets, our framework accounts for the asymmetric nature of VLA training targets, where visual-linguistic
inputs serve as persistent conditions and only actions are generated. We specialized the masking strategy to
accommodate this asymmetry, ensuring that parallel action denoising remains strictly grounded in the multi-
modal prefix to achieve superior training-inference consistency. Our method restructures action generation by
maintaining an AR structure at the block level while performing parallel denoising within each block. This
hybrid design preserves the causal logic and KV-Cache efficiency of AR models, ensuring temporal consistency.
Simultaneously, intra-block parallel diffusion enables high-throughput action generation. Crucially, by retaining
the global causal structure, Block Diffusion facilitates a smoother transition for adapting pretrained AR
backbones than bidirectional approaches, effectively balancing reasoning stability with execution speed.
We implement our framework using OpenVLA (Kim et al., 2025) as the pretrained backbone and conduct
extensive evaluations on the LIBERO (Liu et al., 2023a) and SimplerEnv (Li et al., 2025) robotic benchmarks.
2

As shown in Figure 1(a), our experimental results demonstrate that, with a limited training budget of 50k
steps on only 2 GPUs, our BlockVLA achieves an average success rate of 91.7% on LIBERO, significantly
outperforming the baseline’s 83.2%, with comparable gains on Object and Long suites. Notably, our Block
Diffusion achieves an inference speedup of 3.3× compared to standard discrete diffusion baseline, which is
shown in Figure 1(b). Furthermore, our analysis of success rate curves across training steps reveals that Block
Diffusion exhibits superior training efficiency, with success rates converging faster than the baseline. This
advantage is particularly pronounced in complex, long-horizon tasks, where the synergy of causal block-wise
structure and intra-block parallel generation allows the model to capture temporal dependencies more robustly
and with substantially fewer training iterations.
In summary, our primary contributions are as follows:
• We present BlockVLA, a novel framework that pioneers the adaptation of pretrained Autoregressive VLAs
into a Discrete Diffusion paradigm through Block Diffusion. By reconciling block-level causality with
intra-block parallelism, our method preserves KV-Cache while enabling high-throughput action generation.
• We provide a comprehensive investigation into the Block Diffusion training paradigm tailored for VLA’s
multi-modal asymmetry. By distinguishing visual-linguistic context as persistent conditions from action
generation targets, we design a specialized masking strategy that ensures training-inference consistency.
Our analysis of various attention patterns and masking behaviors offers actionable insights into bridging
the gap between causal pretraining and bidirectional diffusion for action sequences.
• We conduct extensive experiments on the LIBERO and SimplerEnv benchmarks. Our results demonstrate
that BlockVLA achieves approximately a 3.3× inference acceleration compared to standard Discrete
Diffusion baselines. Furthermore, our approach exhibits superior training efficiency, achieving higher
success rates with fewer training iterations, particularly in challenging long-horizon tasks.
2

## Method

large-scale pretrained AR VLAs into efficient Discrete Diffusion policies via Block Diffusion.
2.3
Discrete Vision-Language-Action (VLA) Models
Inspired by LLM advances, VLA models have emerged as a promising paradigm for general-purpose robotic
manipulation (Li et al., 2024; Intelligence et al., 2025; Cui et al., 2025; Bai et al., 2026b,a). A major line of work
focuses on discrete VLA modeling (Kim et al., 2025; Wang et al., 2025; Qu et al., 2025; Cen et al., 2025; Zhao
et al., 2025), where AR backbones generate tokenized robot actions. While systems like OpenVLA demonstrate
strong transfer capabilities, their token-by-token decoding introduces high latency and compounding errors.
Recent studies have explored diffusion-based VLA generation (Song et al., 2026; Ye et al., 2025a; Wen et al.,
2025; Chen et al., 2026) to address these issues, yet many directly adopt dLLM formulations that treat
perception and action as a homogeneous sequence. Although DDVLA (Liang et al., 2025) adapts AR-based
VLAs into discrete diffusion, its reliance on extensive iterative denoising limits efficiency. In this work, we
leverage the asymmetric nature of perception and action by treating visual-linguistic features as stable prefix
anchors. By tailoring a block-wise causal structure with a Diffusion Forcing objective, we ensure that parallel
action denoising remains strictly grounded in the multi-modal context. This design enables continuous prefix
KV-cache reuse, achieving a hardware-friendly balance between reasoning depth and execution reactivity.
3
Method
As illustrated in Figure 2, BlockVLA is a semi-autoregressive discrete diffusion framework for robotic action
generation. It preserves autoregressive dependencies across action blocks while enabling parallel denoising
within each block, thereby combining global temporal consistency with efficient local refinement. We first
review the preliminaries of discrete diffusion and block diffusion language modeling in Section 3.1, and then
formulate their extension to VLA policy learning. We then present the architecture of BlockVLA in Section 3.2,
including the pretrained backbone, action tokenization, block-wise diffusion formulation, masking strategy,
and token-shift design. Finally, Section 3.3 describes the block-wise training and inference pipeline.
3.1
Preliminaries: Discrete and Block Diffusion Language Model
Unlike continuous diffusion models (Ho et al., 2020; Song et al., 2021) that operate in continuous space using
Gaussian noise xt = αtx0 + σtϵ where ϵ ∼N(0, I) , Discrete Diffusion Language Models (Austin et al., 2021;
Nie et al., 2025; Bie et al., 2025; Han et al., 2023) (dLLMs) define the diffusion process directly on the discrete
vocabulary V of size V . Let x0 = [x(1)
0 , . . . , x(L)
0
] denote a sequence of L tokens, where each x(i)
0
∈{0, 1}1×V
is a one-hot row vector. The forward process independently corrupts each token, where the probability of the
intermediate state xt = [x(1)
t , . . . , x(L)
t
] ∈RL×V is given by:
q(xt | x0) =
L
Y
i=1
q(x(i)
t
| x(i)
0 ) =
L
Y
i=1
Categorical

x(i)
t ; p = x(i)
0 ¯Qt

,
with
¯Qt = ¯αtI + (1 −¯αt)1m⊤,
(1)
where ¯αt denotes the retention coefficient determined by the noise schedule, 1 ∈RV ×1 denotes the all-one
vector, and m ∈RV ×1 denotes the one-hot vector corresponding to the absorbing [MASK] token. Under this
formulation, ¯Qt ∈RV ×V ensures that each token either stays in its original state or transitions to the mask
state with probabilities defined by the schedule. The model pθ(x0 | xt) is then trained to predict the original
tokens at the masked positions. The training objective minimizes the cross-entropy loss over masked tokens:
4

Figure 2 Overall architecture of the proposed BlockVLA framework. The adaptation process is illustrated in three
stages. Left: We initialize our model from a pretrained Autoregressive VLA, inheriting its robust vision-language
alignment and causal reasoning capabilities. Middle: During the Adaption SFT phase, we illustrate the shift in training
objectives. While the baseline transitions from a standard causal mask to a fully bidirectional mask (top), our method
employs a Blockwise causal masking strategy (bottom). This encourages intra-block parallel denoising while preserving
global causal dependencies. Right: Adapted Diffusion VLA during inference. In contrast to standard Discrete Diffusion,
which requires full-sequence iterations, our Block Diffusion maintains an autoregressive flow at the block level, enabling
efficient KV Cache reuse and significantly reducing the deployment latency of discrete diffusion-based robotic policies.
L(θ) = −Et,x0,xt
"
1
t
L
X
i=1
1
h
x(i)
t
= [MASK]
i
log pθ(x(i)
0
| xt)
#
.
(2)
where 1[·] is the indicator function that isolates the loss to the masked indices, and the factor 1/t follows
the standard reweighting used in masked dLLMs. During inference, the generation process departs from the
standard token-by-token autoregressive suite. It begins with a sequence entirely composed of [MASK] tokens.
In each iteration, the model performs a parallel forward pass to predict probability distributions for all masked
positions. Following a predefined noise schedule, only a subset of tokens with the highest prediction confidence
is retained and fixed, while remaining uncertain positions are re-masked for subsequent refinement. This
iterative "predict-and-refine" strategy allows the model to utilize bidirectional context and perform robust
error correction, ultimately leading to a coherent and precise action sequence within a few steps.
Block Diffusion (Arriola et al., 2025a) emerges as a semi-autoregressive paradigm combining the advantages of
global coherence and parallel generation efficiency. In this paradigm, the clean sequence x0 is partitioned into
B ordered blocks [x1
0, x2
0, . . . , xB
0 ], where each block xb
0 contains L′ tokens xb
0 = [(xb
0)(1), (xb
0)(2), . . . , (xb
0)(L′)].
The generation follows a hybrid scheme: autoregressive modeling between blocks and discrete diffusion within
each block. The joint likelihood of the sequence factorizes as:
log pθ(x0) =
B
X
b=1
log pθ(xb
0 | x<b
0 ).
(3)
For each block xb
0, the conditional distribution is modeled using a discrete diffusion process, where we
independently assign timesteps (t1, t2, . . . , tB) to the B blocks. We distinguish two conditioning strategies for
the block-level history:
5

• Teacher Forcing: (Arriola et al., 2025a; Cheng et al., 2025) The model conditions on the clean ground-truth
previous blocks, which is formulated as pθ(xb
0 | xb
tb, x<b
0 ). This ensures stable training by providing an
uncorrupted history.
• Diffusion Forcing: (Chen et al., 2024; Wang et al., 2026; Wu et al., 2026a) The model conditions on
potentially “noisy” previous blocks from earlier diffusion steps, formulated as pθ(xb
0 | xb
tb, x<b
t′ ). This aligns
the training distribution with the iterative refinement during inference, enhancing the model’s robustness
to past prediction errors.
For a sequence of length L partitioned into B blocks, each block with length L′ = L/B, assuming L is divisible
by B. The overall learning objective LBD is defined as the average masked denoising loss across all blocks:
LBD(θ) = 1
B
B
X
b=1
Etb,x0,xb
tb

−1
tb
L′
X
i=1
1
h
(xb
tb)(i) = [MASK]
i
log pθ

(xb
0)(i) | xb
tb, h<b

,
(4)
where 1[·] isolates the loss to the masked indices within block b, and h<b denotes the block-level history:
h<b = x<b
0
for Teacher Forcing and h<b = x<b
t′
for Diffusion Forcing. This unified architecture breaks the
autoregressive bottleneck by allowing parallel token generation within blocks while maintaining the structural
consistency of the global sequence.
3.2
Architecture of BlockVLA
3.2.1
Backbone and Action Tokenization
BlockVLA builds upon Discrete Diffusion VLA (DDVLA) (Liang et al., 2025) and the multimodal architecture
of OpenVLA (Kim et al., 2025). As illustrated in Figure 2, we preserve the Prismatic-7B (Karamcheti
et al., 2024) backbone, utilizing SigLIP (Zhai et al., 2023) and DINO v2 (Oquab et al., 2023) to extract
complementary visual representations that are linearly projected into the Llama (Touvron et al., 2023)
embedding space. While OpenVLA generates actions autoregressively and DDVLA uses full-sequence discrete
diffusion, BlockVLA retains the unified transformer structure but introduces a block-wise formulation. By
enforcing block-wise causal constraints over action tokens, our framework enables efficient diffusion-based
action generation.
Following the action discretization scheme used in RT-2 (Zitkovich et al., 2023) and OpenVLA (Kim et al.,
2025), we discretize each continuous action dimension into 256 bins using quantile-based binning. The bin
boundaries are computed from the 1st to 99th percentiles of the action distribution to reduce the influence of
outliers. The binary gripper command is represented as an independent token. Each single-timestep action is
represented by 7 tokens, including 3 translation tokens, 3 rotation tokens, and 1 gripper token. Given an
action horizon H and action dimension D, we flatten the action chunk into a one-dimensional action sequence
a of length HD and concatenate it with the multimodal prefix:
x = [BOS, v, p, l, a(1)
1 , . . . , a(D)
1
, . . . , a(1)
H , . . . , a(D)
H , EOS],
(5)
where v, p, and l denote the visual, proprioceptive, and language tokens, respectively. The token a(d)
h
denotes
the discretized value of the d-th action dimension at timestep h. We use c = [BOS, v, p, l] to denote the
multimodal prefix context, and reserve a for the flattened action token sequence.
3.2.2
Block Diffusion Formulation of BlockVLA
To improve the efficiency of diffusion-based action generation, BlockVLA partitions the flattened action
sequence a into B consecutive blocks, denoted as a = [a1, a2, . . . , aB]. Let ab denote the b-th action block,
and let a<b denote all preceding action blocks. The model maintains an autoregressive dependency across
blocks while performing parallel denoising within each block. This design differs from standard full-sequence
discrete diffusion, where the entire action sequence is repeatedly refined. By restricting iterative denoising to
local blocks, BlockVLA reduces the effective computation associated with repeated denoising steps and allows
previously generated blocks to serve as fixed causal context.
6

(1)
Block-wise masking strategies for Teacher Forcing
and Diffusion Forcing.
(2) Training and inference with or without token shift.
Figure 3 Visualization of the block-wise masking strategy and token-shift design in BlockVLA.
During generation, completed blocks are treated as prefix context for subsequent blocks. Therefore, the
hidden states associated with the multimodal prefix and completed action blocks can be reused through prefix
KV caching. In contrast, conventional bidirectional diffusion decoding updates the entire sequence during
each denoising step, making direct reuse of standard KV-cache states difficult. BlockVLA thus provides a
practical compromise between autoregressive decoding and diffusion-based parallel refinement: global temporal
coherence is preserved through block-level causality, while local action tokens are generated in parallel within
each block, effectively balancing computational efficiency with high-fidelity trajectory prediction.
3.2.3
Masking Strategy
We implement BlockVLA using a structured block-wise attention mask. As shown in Figure 3(1), we consider
two training variants that differ in the context exposed to the current noisy action block: Teacher Forcing in
(c) conditions on clean preceding blocks, whereas Diffusion Forcing in (b) conditions on noisy or partially
denoised preceding blocks.
In the Teacher Forcing variant (Arriola et al., 2025a), the b-th block is trained to predict the clean target block
from its noisy version, conditioned on the prefix context and clean history blocks:
pθ(ab
0 | ab
tb, a<b
0 , c),
(6)
where ab
tb is the noisy current block at noise level tb, ab
0 is its clean target, a<b
0
denotes clean preceding action
blocks, and c = [BOS, v, p, l] is the multimodal prefix. To expose the clean history without leaking the current
or future targets, we concatenate the noisy action sequence and the clean action sequence:
xinput = [BOS, v, p, l, at, EOS, a0],
(7)
where at = [a1
t1, . . . , aB
tB] and t = (t1, . . . , tB) indicate that each block can be corrupted with an independent
noise level, and a0 denotes the clean action sequence. The attention mask allows tokens in ab
tb to attend to c
and a<b
0 , but prevents access to a≥b
0 . Meanwhile, the Teacher Forcing implementation requires two additional
details to ensure consistency between training and inference:
• Loss masking. The appended clean sequence a0 is used only as auxiliary conditioning context. It does not
participate in the diffusion loss, and its CE targets are set to the ignore index −100.
• Position-indexalignment. The appended clean sequence should not be treated as a new temporal continuation
of the noisy sequence. Therefore, corresponding tokens in at and a0 are assigned identical RoPE position
indices (Su et al., 2024). Equivalently, the position ids of the clean suffix are shifted back to match the
action positions in at, so that clean previous blocks provide KV context at the correct action positions.
7

In the Diffusion Forcing variant (Chen et al., 2024), no clean action suffix is appended. Thus, we only need to
corrupt each block in the action sequence.
xinput = [BOS, v, p, l, at, EOS].
(8)
The b-th block is instead trained as
pθ(ab
0 | ab
tb, a<b
t′ , c),
(9)
where a<b
t′
denotes preceding blocks that may have different noise levels and may remain noisy or partially
denoised. Thus, Diffusion Forcing differs from Teacher Forcing in the source of block-level history: it conditions
on imperfect previous blocks rather than clean previous blocks. This better matches the inference-time pattern,
where previously generated blocks may contain residual prediction errors.
Finally, we impose a constraint on the prefix context c. Tokens in c use bidirectional attention to support
multimodal fusion among BOS, visual, proprioceptive, and language tokens. However, prefix tokens are
not allowed to attend to action tokens.
This constraint prevents leakage from future actions, keeping
the conditioning pattern consistent between training and inference: all action blocks condition on the full
multimodal prefix, but the prefix representation is computed independently of the action sequence. Compared
with the bidirectional mask in DDVLA, our mask preserves parallel denoising within each block while
introducing causal dependencies across action blocks.
3.2.4
Token Shift
Standard autoregressive language models are trained with a one token shift as shown in Figure 3(2), where
the hidden state at position n −1 predicts the token at position n. Prior work on adapting pretrained AR
models to diffusion language modeling (Gong et al., 2025) suggests that preserving this shift may help align
diffusion training with pretrained AR representations. DDVLA (Liang et al., 2025) follows this design.
Table 2
Analysis of token shift on LIBERO-Object for
DDVLA baseline (Success Rate %).
Variant
Training Steps
5k
10k
15k
20k
30k
40k
w token shift
51.2
56.6
80.2
90.6
95.0
94.2
w/o token shift
59.6
66.0
81.0
94.2
95.2
95.8
We empirically revisit this choice for VLA policy
learning. As shown in Table 2, which aligns with
recent findings in Efficient-DLM (Fu et al., 2025),
removing the token shift does not degrade perfor-
mance on LIBERO-Object; instead, it yields higher
success rates throughout most training stages. This
