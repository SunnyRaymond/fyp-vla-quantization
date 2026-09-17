# From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation

paper_id: arxiv:2605.12167v1
tier: T3
source_used: html_arxiv
warning: none

## Intro

Vision–Language–Action (VLA) models
(
Brohan et al., 2022
;
Zitkovich et al., 2023
;
O’Neill et al., 2024
;
Bjorck et al., 2025
;
Black et al., 2025
;
Cheang et al., 2024
)
have become a powerful paradigm for robot manipulation, learning to act directly from perception and language understanding, as shown in Figure
1
(a). By tightly coupling visual understanding with action prediction, VLA systems demonstrate strong generalization across tasks and settings.
Alongside this direct perception-to-action route, another line of research explores imagination-based manipulation
(
Hu et al., 2025b
;
Liao et al., 2025
;
Feng et al., 2025a
;
Pai et al., 2025
;
Liang et al., 2025
;
Zhu et al., 2025
;
Chen et al., 2025a
)
, where video generation models predict future observations to inform action selection, as shown in Figure
1
(b). Advances in video generation enable models to forecast plausible scene evolutions, capturing task semantics and object interactions beyond the current observation. Existing approaches typically follow one of two strategies: either using predicted frames as additional visual inputs to condition the policy
(
Hu et al., 2025b
;
Liao et al., 2025
)
, or directly decoding generated futures into actions
(
Feng et al., 2025a
;
Chen et al., 2025a
)
. The former leaves the burden of translating visual change into control to the action head, while the latter tightly couples execution to the accuracy of video prediction, making control highly sensitive to compounding prediction errors. As a result, effectively exploiting imagined futures for robot manipulation remains challenging.
A fundamental challenge is that visual predictions are not inherently action-oriented. Video generation models are optimized for perceptual realism rather than manipulation-relevant structure. As a result, even accurate future videos may fail to expose the physical causes of state transitions, making the mapping from imagined images to actions indirect and unstable.
This gap motivates inverse-dynamics-style latent action models
(
Ye et al., 2025
;
Chen et al., 2025e
;
Bu et al., 2025b
;
Baker et al., 2022
;
Lee et al., 2024
;
Schmidt and Jiang, 2024
;
Bruce et al., 2024
)
, which model the action responsible for a transition between observations. Following works such as LAPA
(
Ye et al., 2025
)
and UniVLA
(
Bu et al., 2025b
)
, we use IDM terminology in this broader latent-action sense: the model infers latent actions from observation transitions, and these latent actions are later decoded into continuous controls rather than treated as executable actions. By explicitly modeling the causal link from perception to action, inverse dynamics extracts manipulation-relevant information from visual change. This property suggests a simple but powerful idea: imagined future frames can be converted into executable representations by inferring the actions implied by predicted transitions. While inverse dynamics has been explored in robot learning, existing uses are typically task-specific or limited in scale, and do not provide (i) inverse dynamics models pretrained on large-scale robot datasets that can robustly operate on generated futures, nor (ii) a modality-aware set of inverse dynamics models that explicitly captures complementary semantic, geometric, and motion cues.
To make this connection robust, we pretrain inverse dynamics models on large-scale robot datasets
(
O’Neill et al., 2024
;
Bu et al., 2025a
)
, enabling them to generalize across diverse manipulation behaviors and visual variations. Such pretraining allows the models to reliably infer actions not only from real observations but also from imperfect or generated future frames.
Moreover, effective manipulation depends on complementary cues. Semantic changes convey task intent, depth captures geometric structure, and motion patterns reflect interaction dynamics. A single latent action representation is often insufficient. We therefore train three modality-aware inverse dynamics models, specialized for semantic
(
Ravi et al., 2024
)
, depth
(
Yang et al., 2024a
)
, and flow-based motion
(
Karaev et al., 2025
)
information, and combine their outputs as a mixture of latent actions.
Based on this design, we propose
MoLA
(Mixture of Latent Actions), an imagination-based robot manipulation model whose core contribution is a latent-action interface between video generation and policy execution, as shown in Figure
1
(c). MoLA uses predicted future videos as an imagination space, converts them into a mixture of latent actions via pretrained inverse dynamics models, and conditions the policy on these action-centric representations. In doing so, MoLA enables reasoning over imagined futures directly in action space rather than image space.
We evaluate MoLA on standard simulated manipulation benchmarks, including LIBERO
(
Liu et al., 2023a
)
, CALVIN
(
Mees et al., 2022
)
, and LIBERO-Plus
(
Fei et al., 2025
)
, and further validate its effectiveness on real-world robot manipulation tasks. Results show consistent improvements in success rate, temporal consistency, and generalization over methods that directly condition on predicted visual futures.
In summary, our main
contributions
are three-fold:
(i)
We introduce a latent-action interface for imagination-based robot manipulation, transforming predicted future videos into executable representations via inverse dynamics.
(ii)
We propose MoLA, a manipulation model that couples video generation and action through a mixture of pretrained, modality-aware inverse dynamics models.
(iii)
We demonstrate the effectiveness of MoLA through extensive experiments on simulated benchmarks and real-world manipulation tasks.

## Method

3.1
Overview
MoLA (Mixture of Latent Actions) is an imagination-based manipulation framework that uses MoIDM as a latent-action interface to transform predicted future visual trajectories into structured, executable action representations, as shown in Figure
2
. Given the current observation and task instruction, MoLA first employs a video generation model to synthesize future visual rollouts as an imagination space. These imagined futures are then processed by a mixture of modality-aware inverse dynamics models to infer complementary latent actions that explain the underlying visual transitions. The resulting mixture of latent actions is finally decoded into executable robot control commands by a diffusion-based action head.
In the following sections, we first introduce the video generation model used for future imagination (Section
3.2
). We then describe the mixture of inverse dynamics models that extracts modality-specific latent actions (Section
3.3
), followed by the action head that generates executable policies from the latent action mixture (Section
3.4
). The overall training procedure is presented in Section
3.5
.
3.2
Video generation model
Our MoLA adopts Stable Video Diffusion (SVD)
(
Blattmann et al., 2023
)
as the video generation backbone to provide visual imagination for robot manipulation. Given the current RGB observation
o
t
rgb
o_{t}^{\text{rgb}}
and task instruction
l
l
, the model predicts a sequence of future frames
o
^
t
:
t
+
H
rgb
\hat{o}_{t:t+H}^{\text{rgb}}
that captures the anticipated evolution of the scene under task execution.
Following the latent diffusion framework, SVD encodes video clips into a compact latent space and performs conditional denoising to generate future visual trajectories. The denoising process is conditioned on the latent embedding of the current observation and the instruction representation, enabling the model to synthesize task-consistent future rollouts.
During inference, MoLA restricts the generation process to a single denoising step for efficiency. Despite this simplification, the predicted latent futures retain sufficient visual and motion-relevant information to reflect meaningful scene dynamics for downstream action reasoning
(
Hu et al., 2025b
)
. For multi-view robotic setups, future predictions are generated independently for each camera stream.
The video generation model serves as an imagination module that provides temporally coherent and visually informative future rollouts, which are subsequently transformed into action-centric representations by the inverse dynamics models.
3.3
Mixture of inverse dynamics models (MoIDM)
Predicted future frames from the video generation model provide an imagination space, but they are not inherently action-centric. MoLA therefore introduces a mixture of modality-aware inverse dynamics models (MoIDM), which forms the core latent-action interface of the method by converting visual transitions into discrete latent actions between imagined futures and policy execution. We describe
(i)
the pretraining framework of each inverse dynamics model, and
(ii)
how MoIDM is integrated into MoLA for joint fine-tuning with the action head.
Inverse dynamics pretraining.
Each inverse dynamics model adopts a modality-specific spatiotemporal transformer, together with a modality-specific VQ codebook
(
Van Den Oord et al., 2017
)
for discrete latent actions, as shown in the left part of Figure
2
. Given a current RGB frame
o
t
rgb
o_{t}^{\text{rgb}}
and a future RGB frame
o
t
+
k
rgb
o_{t+k}^{\text{rgb}}
, we first extract RGB features using a ViT-based
(
Dosovitskiy, 2020
)
image encoder:
h
t
rgb
=
E
rgb
​
(
o
t
rgb
)
,
h
t
+
k
rgb
=
E
rgb
​
(
o
t
+
k
rgb
)
.
h_{t}^{\text{rgb}}=E_{\text{rgb}}(o_{t}^{\text{rgb}}),\qquad h_{t+k}^{\text{rgb}}=E_{\text{rgb}}(o_{t+k}^{\text{rgb}}).
(1)
To model the temporal interaction between the current and future observations, we introduce a set of learnable latent action queries, which are initialized and interact with the RGB features through a spatiotemporal transformer:
h
~
t
→
t
+
k
=
T
(
m
)
​
(
q
(
m
)
,
h
t
rgb
,
h
t
+
k
rgb
)
,
\tilde{h}_{t\rightarrow t+k}=T^{(m)}\!\left(q^{(m)},h_{t}^{\text{rgb}},h_{t+k}^{\text{rgb}}\right),
(2)
where
q
(
m
)
q^{(m)}
denotes the latent action queries and
T
(
m
)
T^{(m)}
denotes the modality-specific spatiotemporal transformer that captures cross-frame correspondences.
The transformer outputs are then mapped to discrete latent actions through a modality-specific vector-quantized codebook:
z
t
→
t
+
k
(
m
)
=
VQ
(
m
)
​
(
h
~
t
→
t
+
k
)
,
z_{t\rightarrow t+k}^{(m)}=\mathrm{VQ}^{(m)}\!\left(\tilde{h}_{t\rightarrow t+k}\right),
(3)
yielding compact action tokens that capture the causal transition between the two frames.
To induce modality awareness, each IDM is trained with distinct reconstruction targets. The predicted latent action is combined with the current RGB feature and decoded by a shared ViT-based RGB decoder to reconstruct the future RGB frame:
o
^
t
+
k
rgb
=
D
rgb
​
(
h
t
rgb
,
z
t
→
t
+
k
(
m
)
)
,
\hat{o}_{t+k}^{\text{rgb}}=D_{\text{rgb}}\!\left(h_{t}^{\text{rgb}},z_{t\rightarrow t+k}^{(m)}\right),
(4)
which is optimized for
all
inverse dynamics models.
In addition, modality-specific supervision is provided using frozen foundation models. We employ Depth Anything v2
(
Yang et al., 2024a
)
to extract depth features and depth ground-truth maps, SAM2
(
Ravi et al., 2024
)
to extract semantic features and segmentation targets, and CoTracker3
(
Karaev et al., 2025
)
to extract motion flow features and motion flow targets. Let
h
t
(
m
)
=
F
(
m
)
​
(
o
t
rgb
)
,
g
t
+
k
(
m
)
=
F
(
m
)
​
(
o
t
+
k
rgb
)
,
h_{t}^{(m)}=F^{(m)}(o_{t}^{\text{rgb}}),\qquad g_{t+k}^{(m)}=F^{(m)}(o_{t+k}^{\text{rgb}}),
(5)
where
F
(
m
)
F^{(m)}
denotes the corresponding foundation encoder and
g
t
+
k
(
m
)
g_{t+k}^{(m)}
represents the modality-specific future ground-truth.
A ViT-based modality decoder then reconstructs the future modality representation:
g
^
t
+
k
(
m
)
=
D
(
m
)
​
(
h
t
(
m
)
,
z
t
→
t
+
k
(
m
)
)
.
\hat{g}_{t+k}^{(m)}=D^{(m)}\!\left(h_{t}^{(m)},z_{t\rightarrow t+k}^{(m)}\right).
(6)
Each inverse dynamics model is pretrained independently with its own modality reconstruction objective, resulting in three specialized IDMs that share the same RGB-based action inference pipeline while becoming sensitive to semantic, geometric, or motion cues through distinct supervision.
Integration within MoLA (joint fine-tuning).
In the MoLA framework, the video generation model is frozen and used to generate imagined future RGB rollouts
o
^
t
:
t
+
H
rgb
\hat{o}_{t:t+H}^{\text{rgb}}
. For the predicted future frame
o
^
t
+
k
rgb
\hat{o}_{t+k}^{\text{rgb}}
, MoIDM infers modality-specific discrete latent actions through the spatiotemporal transformer and modality-specific VQ codebooks, producing a mixture of latent actions:
𝒵
t
→
t
+
k
=
{
z
t
→
t
+
k
(
sem
)
,
z
t
→
t
+
k
(
depth
)
,
z
t
→
t
+
k
(
flow
)
}
.
\mathcal{Z}_{t\rightarrow t+k}=\left\{z_{t\rightarrow t+k}^{(\text{sem})},\;z_{t\rightarrow t+k}^{(\text{depth})},\;z_{t\rightarrow t+k}^{(\text{flow})}\right\}.
(7)
All inferred latent actions, together with the corresponding generated visual features, are provided as conditioning inputs to the action head. During downstream training, the parameters of MoIDM and the action head are jointly optimized end-to-end, allowing the latent actions to adapt to the characteristics of imagined futures and the control objective, while keeping the video generation model fixed, as shown in the right part of Figure
2
.
3.4
Action head
We employ the Diffusion Transformer
(
Peebles and Xie, 2023
;
Reuss et al., 2024
)
architecture as the action decoding module, which is trained from scratch to generate executable control policies. The action head takes as input the mixture of latent action representations extracted from multiple pretrained inverse dynamics models, together with visual features of the predicted future frames generated by the video model, and predicts action sequences in a conditional generative manner.
Instead of standard diffusion training, the model is optimized using a flow matching objective
(
Lipman et al., 2023
)
, which learns a continuous transformation from noisy action samples to clean target actions under the given conditional features. Through this process, the policy effectively captures complex and multimodal action distributions while enabling efficient and stable training.
Figure 3
:
Experiments are conducted on the CALVIN ABC-D, LIBERO, and LIBERO-Plus benchmarks, as well as on a real-world UR5e robot. We evaluate MoLA across three simulated benchmarks and the real-world setup.
3.5
Model training
We train MoLA in three stages to progressively align video imagination, latent action inference, and policy execution.
Stage I: Video generation model fine-tuning.
We first fine-tune the video generation model on large-scale robot manipulation datasets to adapt it to task-specific visual dynamics and viewpoints. The training datasets and configurations are summarized in Appendix Table
6
.
Stage II: MoIDM pretraining.
Next, we pretrain the mixture of inverse dynamics models using similar robot datasets as in Stage I.
Stage III: End-to-end fine-tuning.
Finally, we integrate the frozen video generation model with MoIDM and the diffusion-based action head. The MoIDM and action head are jointly fine-tuned end-to-end using downstream manipulation demonstrations.
