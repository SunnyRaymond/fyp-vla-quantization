# Adaptive-WAM: Quality-Guided Early-Exit Planning from Intermediate Video-Diffusion Features

paper_id: semanticscholar:10879d5b79fa8703bc8ba7b60aae52d9961696f5
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Planning in open-world traffic requires more than recogniz-
ing the current scene: an autonomous policy must anticipate
how surrounding actors and road context may evolve under its
decisions. Large video models provide temporal and motion
priors for this purpose, and driving-specific world models
adapt such priors to controllable future prediction, simula-
tion, and planning (Wan et al. 2025; Hu et al. 2023a; Gao
et al. 2024a,b). Video generation is therefore an appealing
representation-learning objective for autonomous driving.
∗Corresponding author.
Recent world-action models (WAMs) connect predictive
representations to control either by decoding actions from
video-model features or by jointly generating visual futures
and trajectories (Xia et al. 2025; Liu et al. 2026; Chen, Wang,
and Zhang 2025; Yuan et al. 2026). Although these designs
strengthen the connection between prediction and action,
their deployed computation remains either coupled to itera-
tive future rollout or tied to a predetermined backbone path.
This is unnecessarily rigid: deployment requires only a low-
dimensional ego trajectory, and the quality of a plan may
become sufficient before the full video backbone has been
evaluated.
We therefore ask: how much of a video diffusion model
must be executed to make a reliable driving decision? We
separate two axes that are often conflated: the video dif-
fusion timestep, which controls the latent noise level, and
the DiT depth, which controls how many transformer blocks
are evaluated. On NAVSIM (Dauner et al. 2024), five tested
video timesteps change the score of a fixed layer by at most
0.15 points, whereas fixed exits exhibit substantial depth-
dependent variation in action quality. The best intermediate
exit outperforms the full-depth exit, yet the final adaptive pol-
icy exceeds every fixed single-trajectory exit. This evidence
motivates allocating computation according to the quality of
the current plan rather than committing to one readout depth.
Based on this observation, we propose Adaptive-WAM, a
quality-aware, layer-adaptive world-action model. We retain
a Wan2.2-TI2V-5B (Wan et al. 2025) backbone and attach a
ReCogDrive-style five-step trajectory DiT (Li et al. 2025e)
to six intermediate blocks. Adaptive routing decodes one
trajectory at each attempted exit and retains the best trajec-
tory accumulated across the evaluated exits. A lightweight
DINOv2-Small (Oquab et al. 2024) scorer predicts NAVSIM
planning sub-scores for each decoded trajectory. If the high-
est predicted score among the attempted exits passes a thresh-
old, the controller returns its trajectory; otherwise, backbone
execution continues to the next exit. The default planning
path skips the remaining video denoising loop, the uncon-
ditional classifier-free-guidance branch, and VAE video de-
coding. Because trajectory rewards are highly saturated and
frequently tied, the scorer predicts metric components and
acts primarily as an exit-quality verifier rather than imposing
a strict total ordering over trajectories. We therefore evaluate
both tie-aware selection and consequential large-gap errors.
arXiv:2608.06008v1  [cs.RO]  6 Aug 2026

Future frames 
Action
Transformer block
N×
Image
Ego info
(a)  Video gen-based WAM
Visual &
Action
Supervision
Future frames
Action
···
···
Future frames
Action
···
···
Vision-language world model
Text
Image
Action
Text
Image
Action
(b)  VLA-based WAM
Future frames (optional)
Action
Multi-exit
heads
early exit
Quality
scorer
Trajectory
head
(shallow)
early exit
Quality
scorer
Trajectory
head
(middle)
Quality
scorer
Trajectory
head
(deep)
continue
continue
DiT backbone
N×
Image
Ego info
(c)  Adaptive-WAM
Figure 1: Comparison of predetermined and adaptive WAM interfaces. (a) Video-backbone WAMs follow a predetermined
frame/action generation path; (b) multimodal WAMs generate visual and action streams along a predetermined path; (c)
Adaptive-WAM decodes one trajectory per attempted exit and routes by predicted quality. Future-video prediction supervises
training but is not required by the deployed planner.
Across scorer-backbone ablations with Wan, ResNet, ViT,
and DINO features, the best Wan variant improves the diag-
nostic score by only 0.03 points over DINOv2-Small while
incurring substantially higher inference cost.
Experiments support three conclusions. First, intermedi-
ate diffusion features provide a strong planning substrate:
the best fixed-exit model improves from 86.56 PDMS after
imitation learning to 90.62 after DiffGRPO-style refinement,
while the adaptive single-trajectory planner reaches 90.79
PDMS. Second, adaptive exits improve the performance–
computation frontier. At the selected threshold, the adaptive
policy reaches 90.79 PDMS versus 90.62 for the strongest
fixed single-trajectory exit. Its average end-to-end planning
latency is 170 ms, approximately 10% lower than the 190 ms
fixed block-15 planner and 47% lower than the 320 ms fixed
full-depth planner. Third, the learned representation trans-
fers across datasets, reaching 0.88 m average L2 error and
0.08% collision rate on nuScenes (Caesar et al. 2020) with-
out target-domain fine-tuning.
Our contributions are:
• We systematically diagnose how video-noise level and
DiT depth affect driving, revealing robustness to the five
tested video-noise levels and distinct depth-dependent
quality–computation trade-offs.
• We introduce a multi-exit world-action architecture
whose learned trajectory-quality controller dynamically
selects the required DiT depth without completing video
generation.
• We evaluate planning, transfer, trajectory scoring, and ef-
ficiency on NAVSIM v1/v2 and nuScenes, obtaining state-
of-the-art planning performance among compared world-

## Method

trajectory exit at lower average cost than the strongest
one.
2
