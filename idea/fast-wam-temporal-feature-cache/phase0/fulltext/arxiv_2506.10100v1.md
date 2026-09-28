# EfficientVLA: Training-Free Acceleration and Compression for Vision-Language-Action Models

paper_id: arxiv:2506.10100v1
tier: U
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Building upon advances in multimodal understanding from models integrating vision and language [1,
2, 3, 4, 5], Vision-Language-Action (VLA) models enable transformative embodied intelligence.
These systems, such as OpenVLA [6], CogACT [7], pi0 [8] and RT-2 [9], directly translate multimodal
inputs into executable actions, successfully tackling complex robotic manipulation and reasoning
tasks using large-scale datasets [10, 11]. Many cutting-edge VLAs couple a Vision-Language

## Method

distribution [7, 12, 13, 14]. However, the significant computational and memory overheads of these
Diffusion-based VLA architectures during the inference time pose critical barriers to their practical
deployment, particularly for real-time interaction on resource-constrained robotic platforms.
Diffusion-based VLA architectures typically comprise a vision encoder to extract features, a large
language model (LLM) [15, 16, 17, 18, 19] core for multimodal reasoning, and a diffusion-based
action decoder to predicts the final actions through multiple denoising steps. While this modular
design underpins their powerful capabilities, it inherently results in substantial computational and
Preprint. Under review.

Table 1: Module-wise inference characteristics of a baseline VLA model (CogACT, Left) compared to
our proposed EfficientVLA (Right). EfficientVLA demonstrates significant improvements in overall
inference speed and computational efficiency (FLOPs).
Vision Module
Language Module
Action Module
#Param (M)
802.3
6738.9
89.0
Vision Token
256
256
-
Denoising Steps
-
-
10
Inference Time (ms)
24.9
134.5
51.5
FLOPs (G)
405.50
3726.55
57.96
Vision Module
Language Module
Action Module
#Param (M)
802.3
3971.1 (↓41%)
89.0
Vision Token
256
56 (↓78%)
-
Denoising Steps
-
-
2 (↓80%)
Inference Time (ms)
24.9
58.9 (↓56%)
26.2 (↓49%)
FLOPs (G)
405.50
792.58 (↓78%)
11.72 (↓80%)
(b) Layer-wise output cosine similarity (c) Timestep-wise output cosine similarity
Attention
MLP
Layer = 4
Memory 
Bound
Computation
Bound
Number of Visual Tokens
FLOPs (T)
Time (s)
(a) Token-wise inference bottlenecks
Layer = 0
1                    8                     16                     24                    32
1                    8                     16                    24                        32
1.0        0.9        0.8         0.7         0.6         0.5        0.4          0.3
Figure 1: VLA inference bottleneck and redundancy analysis: (a) Visual token pruning impact on
FLOPs and inference time, revealing computation-bound and memory-bound regimes. (b) High
inter-layer cosine similarity of LLM hidden states, indicating depth-wise redundancy. (c) Temporal
cosine similarity of MLP/attention features in diffusion steps, showing computational redundancy.
memory overhead. Our findings (Table 1) indicate that the language module and the iterative diffusion
head are primary contributors to overall latency and computational load. Furthermore, as illustrated
in Figure 1 (a), while visual token pruning initially reduces inference time in computation-bound
scenarios, its efficacy quickly diminishes as the system becomes memory-bound by the LLM.
Prior VLA acceleration efforts have largely focused on isolated tweaks, delivering minimal overall
gains. These fragmented approaches often fail because they ignore the integrated nature of VLA,
where optimizing one module in isolation merely shifts bottlenecks. Gains are limited by unaddressed
inefficiencies elsewhere, such as the memory demands of LLM or the computational intensity of
action head. For example, methods like TinyVLA [14] and DeeR-VLA [20] focus on specialized
model architectures rather than broadly applicable inference acceleration frameworks for pre-trained
VLAs. Other approaches, such as Mole-VLA [21], tackle LLM layer redundancy but require costly
retraining and overlook other pipeline stages. Similarly, VLA-Cache [22] caches static visual
tokens but provides limited speedup, constrained by the significant memory footprint of LLM and
computational demands of the action head. Consequently, these existing approaches fall short of
providing a truly holistic solution to navigate the complex landscape of VLA inefficiencies.
To develop a more effective acceleration strategy, we systematically analyze the inference characteris-
tics and multifaceted redundancies within each VLA module. In many Diffusion-based VLAs, the
diffusion action head operates as a separate module, guided by features extracted from the VLM. This
separation may underutilize the full reasoning capacity of VLM for action generation, questioning the
necessity of its entire scale. As illustrated in Figure 1 (b), the language module demonstrates shows
considerable depth-wise representational redundancy with high inter-layer hidden state similarity. The
visual processing pathway exacerbates this issue by processing superfluous tokens, characterized by
low task-relevance or high informational overlap due to visual similarity, which strains computational
resources and intensifies the memory-bound condition of LLM. As shown in Figure 1 (c), the iterative
diffusion action head displays significant temporal redundancy. The high similarity of its intermediate
features across adjacent denoising steps implies extensive and near-static recomputations.
Motivated by this, we introduce EfficientVLA, a structured, training-free acceleration framework for
Diffusion-based VLAs that systematically targets these issues. Using a similarity-derived importance
metric to target the primary memory bottleneck of the language module and its observed depth-wise
redundancy (Figure 1 (b)), EfficientVLA employs a similarity-derived importance metric to prune
functionally inconsequential layers, thus reducing the depth of the model and the demands for memory
2

without retraining. To manage the initial computational load from visual inputs before the memory
of LLM limit is reached (Figure 1 (a)), our visual token pruning strategy tackles both task-relevant
and inherent image redundancies by first selecting critical task-aligned tokens, then augmenting this
set to ensure representational diversity while maintaining high task relevance. Lastly, EfficientVLA
addresses temporal redundancy in the compute-intensive action generator (highlighted by high feature
similarity across timesteps, Figure 1 (c)) by caching and reusing intermediate attention and MLP
outputs, thus curtailing redundant computations. This synergistic, structured approach provides a
more holistic alleviation of GPU compute and memory bottlenecks than isolated optimizations.
The main contributions of this work are summarized as follows:
1. We present a systematic analysis identifying critical computational and memory-bound bottlenecks,
alongside multifaceted redundancies within contemporary Diffusion-based Vision-Language-
Action (VLA) architectures, thereby motivating the need for structured acceleration.
2. We propose EfficientVLA, a novel training-free, structured inference acceleration framework that
synergistically prunes redundant layers from the language module based on their informational
impact and strategically selects a compact, task-focused subset of visual tokens by considering
both VLA task relevance and inherent image feature diversity.
3. Our framework further enhances efficiency by exploiting temporal redundancies in the diffusion-
based action head, introducing a caching mechanism for intermediate attention and MLP computa-
tions during the iterative denoising process.
4. We demonstrate the efficacy of EfficientVLA through extensive experiments on the CogACT in
the SIMPLER environment [23], achieving a 1.93× inference speedup and reducing FLOPs to
28.9%, all while incurring a minimal accuracy degradation of only 0.6%. This will facilitate the
application of large-scale VLAs on the resource-constrained robotics platforms in the real world.
2
