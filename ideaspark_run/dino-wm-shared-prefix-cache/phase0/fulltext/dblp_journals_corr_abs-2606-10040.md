# Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination

paper_id: dblp:journals/corr/abs-2606-10040
tier: H
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Robot control requires understanding how the physical scene will evolve during interaction. World-
Action Models (WAMs) [1, 2] address this by coupling future video prediction [3, 4, 5] with action
generation [6, 7, 8, 9]. By predicting how observations change over time, WAMs embed rich phys-
ical dynamics and world priors into the control policy, making them a promising robot learning
paradigm. Yet the strongest systems still rely on very large video generators [5, 8], based on the
belief that sharper, more photorealistic futures will yield better actions. That belief comes with a
cost: heavy compute, high latency, and steep hardware demands that block real-time deployment.
A different picture is taking shape. High-quality control does not require photorealistic video. What
the policy truly needs is a future representation that preserves task-relevant geometry, motion ten-
dencies, and contact cues. For example, VPP [4] shows that action generation remains effective
even when the denoising process is reduced to a single step, while Fast-WAM [10] demonstrates
that WAMs can remain competitive even when explicit future generation is skipped during infer-
ence. Building on this insight, we aim not for perfect images but for action-centric futures, and we
reframe efficiency as a modeling problem by proposing Efficient-WAM.
Our core idea is to make the video branch smaller and smarter within a Mixture-of-Transformers
(MoT) framework [11, 12] through structured pruning guided by world-knowledge transfer from
the foundation model WAN-2.2-5B [13]. This distillation step defines what the model must keep to
remain action-faithful: channels and pathways that encode geometry, dynamics, and contact. Once
the backbone has been carved down around these essentials, two complementary accelerations fol-
low naturally. First, token density can fall without harming control. Because pruning concentrates
capacity on task-relevant structure, the model can predict lower-resolution future latents that still
carry the cues needed by the action expert. Computation and memory scale down with token count,
while the distilled priors preserve the information that matters. Second, denoising can be asymmet-
ric. The pruned video branch no longer needs a long sampling schedule to hallucinate photorealistic
detail, whereas the action branch benefits from a richer trajectory refinement. Allocating fewer steps
to video and more to action reduces latency where it counts while preserving decision quality.
Unlike early-exit or dynamic layer-skipping methods [14, 15] or single-step denoising and diffusion-
policy distillation methods [16, 17, 18] that trade stability for short-term gains, our design integrates
model size, token budget, and sampling depth into a coherent, action-centric system that achieves
massive efficiency gains with minimal compromise to control performance. Pruning focuses repre-
sentation on control-critical content. Lower token density then becomes a safe consequence rather
than a risky shortcut. Asymmetric denoising exploits the increased reliability of the pruned video
predictor to further reduce sampling. The three levers reinforce one another, producing a compact
future-imagination module that remains aligned with the controller’s needs.
We evaluate Efficient-WAM in simulation and real-world manipulation. Despite intentionally coarse
future predictions, it achieves 86.7% average success in simulation and 66.25% in real-world tasks,
comparable to or better than heavyweight WAM baselines. By jointly optimizing model size, token
budget, and denoising, Efficient-WAM reduces per-chunk latency to 98 ms on a local consumer
GPU. Our core contributions are:
• We identify the critical deployment bottleneck of WAMs and introduce an ”action-centric future
imagination” design principle, demonstrating that WAMs can be effectively decoupled from the
pursuit of photorealistic video generation.
• We propose Efficient-WAM, a novel architecture that holistically reduces inference costs by
optimizing model size, token count, and denoising steps. This unified approach enables low-
latency, real-world deployment while preserving strong world priors and control performance.
• We decompose WAM inference cost into model size, visual tokens, and denoising steps, show-
ing how pruning enables lower-resolution future latents and shorter video-side sampling.
2

2
Related Works
Recent World-Action Models (WAMs) couple future visual prediction with action generation to in-
ject physical priors into robot policies [7, 19, 8, 9, 11]. While these approaches demonstrate the
value of future prediction, they often inherit the computationally heavy design of video generators:
large backbones, dense visual tokens, and iterative denoising. Emerging evidence suggests that
pixel-level fidelity is not always necessary for control. Being-H0.7 [20] avoids raw-pixel predic-
tion, Fast-WAM [10] skips explicit future generation at inference, and recent WAM variants explore
action-centered or asynchronous video-action designs [21, 22]. Efficient-WAM builds on this direc-
tion but retains a lightweight future-imagination branch, asking how compact the video branch can
be while still preserving useful guidance for action generation.
Efficiency has also been studied in VLA and generative robot policies through compact architec-
tures, quantization, early-exit, token compression, dynamic layer activation, and action-sampling
acceleration [23, 14, 24, 15, 25, 26, 16, 17, 18]. These methods mainly optimize the policy back-
bone or action sampler, and are complementary to our focus on the video-imagination bottleneck
in WAMs. To preserve world priors after compression, we further draw on knowledge distillation,
Transformer distillation, and structural pruning [27, 28, 29, 30]. Unlike generic model compression,
our goal is not to reproduce a large video generator, but to transfer spatiotemporal knowledge from
WAN-2.2-5B [13] into a compact, action-oriented video expert.
3
Method
3.1
Design Formulation
A World-Action Model (WAM) explicitly models the joint distribution of future scene evolution and
control actions. Given a current observation o, a language instruction l, and a robot state s, the joint
prediction objective is formulated as p(zv, a1:H | o, l, s), where zv represents the explicit future
visual latents and a1:H is the action chunk.
To make this joint prediction tractable, our architecture factorizes the distribution into a future-
imagination process and a future-conditioned action generation process:
p(zv, a1:H | o, l, s) = pϕ(zv | o, l)
|
{z
}
video branch
· pθ(a1:H | o, l, s, zv)
|
{z
}
action branch
.
(1)
Here, pϕ predicts the future dynamic context, while pθ extracts executable control signals from
this imagination. While Efficient-WAM preserves this joint formulation, it fundamentally questions
whether zv must be photorealistic.
To connect this formulation to efficiency, we focus on the computation required to produce the future
representation zv. Denote the active video model size by Mv, the future prediction resolution by
rv, the resulting number of future video tokens by N v
tok(rv), and the video denoising budget by Kv.
For a fixed implementation family, the video-side cost can be described as a function:
Cvideo = Fvideo(Mv, N v
tok(rv), Kv) ,
(2)
This abstraction highlights the controllable factors of video-side computation. Efficient-WAM fol-
lows an action-centric design principle: policies need structural and dynamic cues, not photoreal-
istic details. We therefore systematically compress the three factors in Eq. (2) via a compact video
expert distilled from WAN-2.2-5B, low-resolution future latents, and asymmetric video-action de-
noising. The following sections detail these components.
3

Block
MHA
FFN
Block
Block
FFN
MHA

×30
×12
“Put the bottles into the 
storage box, and arrange 
them from left to right in 
descending order of size.”
Compact WAN 0.8B
Action expert 
0.2B
VAE encoder
Denoising
timesteps
0
10
Training
Inference
Mixture-of-transformer
First frame
Future frames
Downsampling
WAN 2.2 5B
Slice and distill
KV cache
KV cache
Compact WAN 
0.8B
Block
...
...
Noisy action
Figure 2: Efficient-WAM architecture. The model utilizes a multiscale video-latent layout where
high-resolution current observations and low-resolution future latents are concatenated. A compact
video expert and an action expert interact via layer-wise MoT to predict optimal action chunks.
3.2
Compact Architecture with World-Knowledge Transfer
Large-scale video generation backbones encode rich world priors, yet their massive parameter counts
make them ill-suited for the stringent latency requirements of real-time robotic control. To ad-
dress this challenge, we adopt a compact Mixture-of-Transformers (MoT) architecture. By using
a lightweight video expert and a dedicated action expert, our design retains necessary world priors
while significantly reducing the computational cost.
To build the video expert, we prune WAN-2.2-5B by reducing transformer depth and layer width.
Instead of random initialization, we copy weights from selected teacher layers via layer slicing. This
structured transfer is highly intentional. It ensures the student inherits fundamental physical priors,
such as task-relevant geometry, motion tendencies, and contact cues. At the same time, it sheds the
parameter capacity dedicated to high-fidelity pixel rendering. To stabilize this knowledge transfer,
we supplement the standard video flow-matching objective with a teacher-guided distillation loss
that aligns intermediate hidden states and temporal changes. This effectively distills the teacher’s
physical world understanding into an action-centric backbone.
As illustrated in Figure 2, this compact video expert is coupled layer-wise with the action expert. The
task instruction is injected via cross-attention, while robot states and noisy actions are embedded as
action tokens. At each MoT layer, action tokens 

## Method

Jiajun Li1,*
Tiecheng Guo2,*
Yifan Ye2,*
Rongyu Zhang2
Xiaowei Chi3,‡
Qianpu Sun2
Ying Li2
Yunfan Lou2
Yan Huang4
Zhihe Lu5
Meng Guo2
Shanghang Zhang2,B
1The University of Hong Kong
2Peking University
3Muka Robotics
4Institute of Automation, Chinese Academy of Sciences
5Nanjing University
*Equal contribution
‡Project lead
BCorrespondence: shanghang@pku.edu.cn
Project page: https://efficientwam.github.io/
Abstract: World-Action Models (WAMs) have emerged as a promising paradigm
for embodied control by coupling future visual prediction with action generation.
However, most existing WAMs rely on photorealistic future prediction, which in-
curs high inference latency and makes real-time robot deployment difficult. This
motivates a more efficient WAM design that preserves the control benefits of fu-
ture visual prediction while reducing its inference cost. We introduce Efficient-
WAM, a World-Action Model that reduces the cost of future imagination while
preserving its control benefit. Efficient-WAM improves inference efficiency via
a compact video expert transferred from WAN-2.2-5B, token-sparse video la-
tents, and asymmetric video-action denoising that allocates fewer sampling steps
to video than to actions. Instead of optimizing the future branch for visual fidelity,
Efficient-WAM treats future video prediction as a compact guidance signal for
action generation. Comprehensive experiments on RoboTwin 2.0 and real-world
manipulation tasks show that Efficient-WAM maintains strong action performance
despite visibly coarse future predictions. While maintaining competitive control
capabilities, our 1B-parameter model can reduce per-chunk latency to around 100
ms during physical deployment, achieving a 30x speedup over existing WAMs.
Keywords: World-Action Models, Robot Manipulation, Efficient Robot Learning
Inputs
Observations
Instructions
State
“Pick up LEGO 
blocks and 
place it into 
the bin with 
the matching 
color.”
Ot+1
Ot+2
Ot+k
...
Traditional WAM
Efficient-WAM
Mixture-of-
transformer
heavy
























dense
Model 
size
Tokens 
inputs
Denoising 
timesteps
more
Inference
speed
slow 
lightweight
























sparse
fewer
Correct 
action
fast 
Low-cost future imagination
Low inference latency
π0.5
motus
ours
3215
113
~100
Performance in simulation
π0.5
motus
ours
88.7
82.7
86.7
 ms/chunk
 %
Performance in real world
π0.5
motus
ours
63.8
53.8
66.3
 %
Figure 1: Overview of Efficient-WAM. Efficient-WAM uses low-cost future imagination to capture
task-relevant object and robot dynamics without photorealistic video generation. Compared with
prior WAMs, it achieves lower latency and strong task success in simulation and real-world settings.
arXiv:2606.10040v2  [cs.RO]  10 Jun 2026

1
