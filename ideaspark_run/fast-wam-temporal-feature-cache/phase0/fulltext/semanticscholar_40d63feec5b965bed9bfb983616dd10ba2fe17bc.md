# Dream-Tac: A Unified Tactile World Action Model for Contact-Rich Robot Manipulation

paper_id: semanticscholar:40d63feec5b965bed9bfb983616dd10ba2fe17bc
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

World models learn predictive representations of environmental
dynamics by forecasting future observations[1, 2, 4, 5, 31], thereby
endowing agents with the ability to anticipate how the environment
will evolve over time. Building on this paradigm, world action
models further transfer such predictive knowledge into the decision-
making process[26, 40, 46], allowing policies to inherit dynamics
priors acquired from large-scale web-data pretraining and to use
these priors to guide action generation.
Although world action models have shown promising perfor-
mance in general manipulation tasks [26, 28, 46], they remain
limited in contact-rich and fine-grained settings. This limitation
stems largely from the fact that vision alone often fails to capture
the physical interaction cues required for precise control, includ-
ing contact states, local geometry, and fine-grained object prop-
erties [6, 11, 17, 24, 50]. As a result, visually conditioned policies
continue to struggle in manipulation settings where success de-
pends on accurately perceiving contact. Tactile sensing naturally
addresses this gap by directly providing contact and local inter-
action signals that are ambiguous or entirely unavailable in RGB
observations [29]. As shown in Fig. 1 (a), RGB images do not pro-
vide a clear indication of contact, whereas tactile variations reveal
subtle physical interaction cues that compensate for the lack of
fine-grained contact perception in vision. This makes tactile sens-
ing particularly important for manipulation tasks where accurate
awareness of physical contact is critical [30].
Incorporating tactile feedback into policy learning provides an
effective pathway toward precise decision-making [11, 27, 45, 50],
particularly within world action models. Tactile signals are inher-
ently temporal and event-driven, while world models are designed
to capture temporal dynamics, making them naturally well-suited
for modeling such interaction patterns [21]. However, these signals
are sparse and transient in practice; long periods of stasis are of-
ten punctuated by brief, critical events like contact onset, slip, or
release. Consequently, treating tactile tokens symmetrically with
other modalities risks diluting the precise interaction cues most
essential for successful manipulation. This raises a key question:
how can tactile signals be incorporated into world action models
in a selective and interaction-aware manner?
Motivated by these observations, we propose Dream-Tac, a uni-
fied tactile world action model that integrates tactile perception
into a generative framework by jointly predicting future visual
observations, robot actions, and future tactile observations. Built
upon a pretrained video generative backbone [2], Dream-Tac ex-
tends world action modeling to contact-rich manipulation through
a novel contact-aware attention bias. This mechanism mitigates the
sparsity of tactile signals by adaptively amplifying the influence
of touch only when contact dynamics become salient. Rather than
treating tactile tokens uniformly, Dream-Tac prioritizes interaction-
relevant signals, enabling tighter coupling between future visual
states, tactile dynamics, and precise action generation. Extensive
real-world experiments validate the effectiveness of out method,
achieving the highest success rate among four strong baselines and
outperforming Cosmos Policy [26] by 31.6%.
While these tactile-aware mechanisms enhance perception, they
significantly increase the computational burden of the world ac-
tion model. This challenge is particularly pronounced in world
action models built upon Diffusion Transformers [35, 35], which
involve quadratic self-attention and iterative multi-step denoising.
The inclusion of tactile sequences further increases the tempo-
ral and modality complexity, making real-time deployment more
challenging. To mitigate this, Dream-Tac incorporates a dual-level
acceleration system. First, we optimize training by re-implementing
the gated-bias attention with a FlashAttention-based formulation
[13, 14, 42], achieving up to 2.94× training speedup. Second, we
accelerate inference via diffusion-step caching, yielding a 1.8×
speedup at test time. These system-level optimizations are critical
for ensuring Dream-Tac remains feasible for the high-frequency
execution required in precise robotic manipulation.
In summary, our contributions are four-fold:
• We propose Dream-Tac, a unified generative world action

## Method

tile signals, and robot actions.
• We propose a contact-aware attention mechanism that adap-
tively emphasizes tactile signals during salient interaction
events.
• We develop a dual-level acceleration system featuring a
FlashAttention-based gated bias for efficient training and a
diffusion-step caching strategy for faster inference.
• Experiments on contact-rich manipulation tasks demon-
strate that Dream-Tac significantly outperforms state-of-
the-art baselines in success rate while maintaining compet-
itive generation quality and execution efficiency.
2
