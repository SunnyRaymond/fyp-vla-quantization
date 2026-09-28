# [user-supplied anchor] 2502.02175

paper_id: user_ref:arxiv_id:2502.02175
tier: U
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Learning a robust and generalizable policy for robotic manipulation through policy learning has
long been a challenging problem [1], with traditional reinforcement learning approaches [2, 3] often
suffering from poor robustness and limited generalization. Recently, the rapid advancement of
foundational Vision-Language Models (VLMs) [4, 5] has demonstrated remarkable capabilities in
multimodal understanding and generalization. Leveraging large-scale real-world robotic datasets
[6, 7], pioneering works [8–11] have introduced Vision-Language-Action (VLA) models, which
integrate vision and language modalities to directly generate robotic actions in an end-to-end manner.
This emerging paradigm holds great promise for enhancing the adaptability and generalization of
robotic control systems, but leaves a large computational demand.
To mitigate the extensive cost of VLA models, existing works often adopt generic acceleration
techniques, such as model lightweighting [12], quantization [13], and early-exit [14]. While effective
to some extent, these methods often require architectural modifications or retraining, and more
importantly, they lack task-specific design tailored to the intrinsic characteristics of VLA tasks. As a

## Method

ity in robotic perception. Rather than
recomputing all vision tokens at every
timestep, VLA-Cache identifies tokens
that exhibit minimal change between ad-
jacent frames and reuses their cached key-
value (KV) representations to bypass re-
dundant computation. However, we ob-
serve that not all visually static tokens
can be safely reused. Some tokens, such
as those near the gripper or target ob-
ject, may appear visually unchanged but
remain semantically active and crucial
for accurate action generation. Naively
reusing all static tokens results in a sig-
nificant performance drop as shown in
Table 1. To mitigate this, VLA-Cache
incorporates a lightweight filtering mechanism based on decoder attention scores to exclude task-
relevant tokens from reuse, ensuring that semantically critical regions are always recomputed with
up-to-date features. Moreover, we observe that attention patterns vary across decoder layers, with
deeper layers exhibiting more concentrated focus. To further optimize reuse, VLA-Cache employs
a layer-adaptive caching strategy that dynamically adjusts the reuse ratio per layer based on at-
tention entropy, prioritizing precise updates in sensitive regions. These two mechanisms together
enable substantial reduction in decoding overhead, especially in large-scale language decoders (e.g.,
LLaMA [15], Gemma [16]), which typically dominate the compute cost in VLA systems.
The resulting method VLA-Cache offers a training-free and plug-and-play solution for acceler-
ating VLA models without sacrificing action performance. We evaluate VLA-Cache on robotic
manipulation tasks across two simulated environments (LIBERO [17] and SIMPLER [18]) and three
state-of-the-art VLA models (OpenVLA [11], CogAct [19], and OpenVLA-OFT [20]). VLA-Cache
consistently delivers over 1.7× acceleration with only minor drops in task success rate. Furthermore,
we demonstrate its real-world applicability by deploying it on a Kinova Jaco2 robot arm, achieving
practical speedup under real-time control scenarios.
2
