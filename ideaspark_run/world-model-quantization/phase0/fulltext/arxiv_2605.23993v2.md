# Nano World Models: A Minimalist Implementation of Future Video Prediction

paper_id: arxiv:2605.23993v2
tier: T3
source_used: html_arxiv
warning: none

## Intro

World Models
(
Ha and Schmidhuber, 2018
;
Dawid and LeCun, 2023
)
have emerged as a cornerstone of spatial intelligence
(
Yang et al., 2025b
;
Wang et al., 2026
)
and real-world decision-making
(
Richens et al., 2025
;
Guo et al., 2025
)
, generating high-fidelity futures by conditioning on the agent’s history and actions. Especially in the past few months predating the release of this manuscript, we have witnessed significant advances in industry-scale World Models
(
Google, 2025
;
Team et al., 2026
)
. Yet, for the broader community, the gap between reading about these models and deploying them remains disappointingly wide.
This manuscript accompanies
Nano World Models
: a minimalist, batteries-included repository for advancing a careful and scientific approach to world-model design. The motivation for this project is simple: while industry-scale world models achieve stunning visual effects, they are built around a handful of simple, well-established techniques: video diffusion
(
Ho et al., 2022b
;
Blattmann et al., 2023
)
, diffusion forcing
(
Chen et al., 2024
)
, consistency distillation
(
Song et al., 2023
)
etc.
We posit that as world model algorithm stabilizes, a shift in research focus lies from inventing new techniques to developing a more nuanced understanding of subtler scientific design decisions, including architectural choices, training objectives, and, of course, data composition and scaling behavior. However, one cannot truly understand the science behind a model without being able to easily experiment with it. Especially with the current fragmented landscape of world modeling research, diverse datasets
(
Brohan et al., 2022
;
Pearce and Zhu, 2022
;
Zhou et al., 2024
)
, training recipes
(
Yang et al., 2023
;
Lipman et al., 2024
;
Li and He, 2025
)
, evaluation protocols
(
Vafa et al., 2024
;
Zhang et al., 2026
)
and downstream tasks
(
Alonso et al., 2024
;
Quevedo et al., 2026
;
Guo et al., 2025
)
scattered across numerous sources makes rigorous scientific studies extremely hard.
1.1
Contributions.
We introduce
Nano World Models
, a minimalist, batteries-included implementation of world models, as a usable starting point admist the aforementioned fragmented landscape. Our repo is built around specializing the diffusion-forcing
(
Chen et al., 2024
)
model. We enable hydra-based configurations
(
Yadan, 2019
)
, modular code for data loading, model design, training, evaluation, and downstream tasks, as well as well-curated docs. In greater detail, we support the following features:
•
Generative Modeling Objectives.
Centered around Diffusion Forcing
(
Chen et al., 2024
)
,
Nano World Models
supports a variety of generative modeling and prediction objectives. It supports Diffusion
(
Rombach et al., 2022
;
Peebles and Xie, 2023
)
and Flow-Matching
(
Lipman et al., 2022
;
Lipman et al., 2024
)
, with
𝐱
\mathbf{x}
,
�
\bm{\epsilon}
and
𝐯
\mathbf{v}
prediction objectives
(
Li and He, 2025
)
.
•
Architectural Sizes and Design.
Nano World Models
is built for support of architectures of varying sizes. Following naming conventions from the image generation community
(
Ma et al., 2024a
;
Peebles and Xie, 2023
)
, we include four variants of varying sizes: NanoWM-S (40M), NanoWM-B (160M), NanoWM-L (600M) and NanoWM-XL (830M). For action injection methods,
Nano World Models
supports element-wise addition, AdaLN
(
Huang and Belongie, 2017
)
, AdaLN fused with timestep injection, FiLM
(
Perez et al., 2018
)
and cross-attention.
•
Environments.
Nano World Models
supports diverse environments, ranging from simple simulation environments to game simulation and robot manipulation. For simple simulation environments,
Nano World Models
currently supports environments drawn from standard robotics benchmarks, namely D4RL
(
Fu et al., 2020
)
and DeepMind Control Suite
(
Tassa et al., 2018
)
. These environments include maze navigation (
Maze
,
Wall
), fine-grained control for tabletop pushing (
PushT
) and deformable object manipulation with an XArm (
Rope
,
Granular
). For game simulation, we support the well-celebrated CS:GO dataset
(
Pearce and Zhu, 2022
)
and for robot manipulation, we support the widely used RT-1
(
Brohan et al., 2022
)
dataset.
•
Logging and Evaluations.
Nano World Models
supports both Tensorboard
(
Abadi et al., 2016
)
and Wandb
(
Biewald, 2020
)
logging systems. Loggings include callback-style validation, reliable per-step checkpointing and system utilization informations. Evaluations are fixed-seed reproducible.
•
Long-Horizon Generation.
Nano World Models
goes beyond the training context. Empowering by the auto-regressive capability of diffusion forcing, as well as sliding window approaches,
Nano World Models
can produce temporally consistent long video generations with 4x the training horizon.
1.2
World-Modeling as
Tool-Use
Nano World Models
goes beyond next state prediction, supporting multiple applications where world modeling serves as a tool for downstream tasks.
•
3D Scene Generation.
Beyond serving as a video predictor, a world model can also act as a generative prior for constructing 3D-consistent scenes.
Nano World Models
supports exporting generated video rollouts into downstream 3D pipelines
(
Lin et al., 2025
;
Chen et al., 2026
)
, where multi-view or temporally adjacent predictions can be lifted into scene representations such as point clouds, Gaussian splats, or neural fields. This provides a lightweight bridge between 2D video generation and 3D scene synthesis, extracting generations from video world models to persistent 3D scenes.
•
Goal-Conditioned Planning.
Nano World Models
further supports goal-conditioned planning, where the world model is used as a simulator for evaluating candidate action sequences before execution. Given an initial observation and a desired goal state, the model can roll out possible futures under different action proposals, enabling planning by trajectory optimization. This turns the learned dynamics model into a tool for decision-making: rather than directly learning a policy for every task, users can query the model to imagine, compare, and select action sequences that are most likely to reach the goal.
1.3
Mission Statement
Thanks to its modular design,
Nano World Models
makes experimentation a matter of changing modular configs rather than rewriting pipelines. Datasets, tasks, model sizes, prediction objectives and overriding any specific configuration, can be swapped from a single command line. In addition, we release everything: code, data and more than a dozen pretrained model checkpoints of all sizes.
World Models need the World. Our hope is to build a Babel tower for world model research: datasets, objectives, architectures, and tasks all speaking the same language. We invite the global community to join us, contribute, and build this future together. Learn more:
Blog:
https://simchowitzlabpublic.github.io/nano-world-model
Models:
https://huggingface.co/collections/knightnemo/nano-world-model
Code:
https://github.com/simchowitzlabpublic/nano-world-model

## Method

Having defined world modeling as conditional sequence generation in
Section
2
, we now describe the modeling and software
abstractions powering
Nano World Models
. The central design principle is to
treat diffusion forcing as a unified interface: prediction objectives,
architectures, action-conditioning mechanisms, latent spaces, environments, and
rollout procedures can be exchanged while preserving the same training and
sampling pipeline.
3.1
Diffusion Forcing as a Unified Interface
Popular generative models produce samples through iterative computation.
Diffusion models iteratively denoise corrupted samples, while flow-matching
models learn a vector field that transports a simple base distribution to the
data distribution. Diffusion forcing extends this view to sequence modeling by
allowing different frames in the same trajectory to occupy different stages of
the generation process.
We introduce a noise index set
K
⊂
R
\mdmathbb{K}\subset\mdmathbb{R}
. For an encoded trajectory
𝐱
1
:
T
+
H
\mathbf{x}_{1:T+H}
, diffusion forcing assigns each frame
𝐱
t
\mathbf{x}_{t}
a noise index
k
t
∈
K
k_{t}\in\mdmathbb{K}
. The model is trained on noised trajectories together with their
noise-index schedule
𝐤
=
(
k
1
,
…
,
k
T
+
H
)
\mathbf{k}=(k_{1},\dots,k_{T+H})
. Context frames may be kept clean
or nearly clean, while future frames may be assigned higher-noise indices. By
changing only this schedule, Nano World Models can express teacher-forced
prediction, masked future prediction, and autoregressive rollout using the same
model interface.
3.2
Generative Objectives
Nano World Models supports multiple generative objectives under the same
diffusion-forcing interface. For diffusion objectives, the model can be trained
with
𝐱
\mathbf{x}
-prediction,
�
\bm{\epsilon}
-prediction, or
𝐯
\mathbf{v}
-prediction targets
(
Li and He, 2025
)
. For
flow-matching objectives, the model predicts the velocity field induced by a
chosen interpolant, such as the linear interpolant between data and noise.
Importantly, these objectives differ only in how the noised input and training
target are constructed. The backbone architecture, conditioning interface,
dataset loader, and sampling code remain shared. This allows objective choices
to be studied as a controlled experimental axis rather than as separate
implementations.
3.3
Nano World Models
Architecture
NanoWM uses a transformer backbone over latent video tokens. For VAE-style encodings, each frame is divided into spatial patches, and projected into a hidden dimension, and processed
by transformer blocks that apply interleaved spatial-temporal attention
(
Ho et al., 2022a
;
Ma et al., 2024b
)
. We follow the
naming convention used in image and video diffusion models: the letter denotes
the model family, while the suffix denotes the latent patch size. For example,
NanoWM-B/2 is the base model with patch size
2
2
, whereas NanoWM-B/4 and
NanoWM-B/8 use coarser latent patches. Nano World Models supports four architecture families: NanoWM-S, NanoWM-B,
NanoWM-L, and NanoWM-XL. These provide a scaling axis from small models for fast
iteration to larger models for higher-capacity prediction.
Action Conditioning.
For action-conditioned world modeling, NanoWM conditions predictions on action
sequences
𝐚
T
:
T
+
H
−
1
\mathbf{a}_{T:T+H-1}
. The repository supports several action-injection
mechanisms. The simplest embeds actions into the transformer hidden dimension
and adds them to the corresponding frame tokens. More expressive variants inject
actions through adaptive layer normalization, fuse action and timestep
conditioning, apply FiLM-style modulation, or use cross-attention from video
tokens to action tokens. These mechanisms expose a spectrum of conditioning strategies, from lightweight
element-wise injection to higher-capacity interactions between actions and visual
dynamics.
3.4
Latent Observation Spaces
Following recent practice in video generation and robotic world modeling,
Nano World Models
predicts encoded observations rather than raw observations directly
(
Huang et al., 2025
;
Zhou et al., 2024
;
Maes et al., 2026b
)
.
The choice of encoding is not merely an implementation detail: recent work
suggests that reconstruction-oriented and semantics-oriented latent spaces can
lead to different tradeoffs in visual fidelity, planning performance, and
representation quality
(
Jha et al., 2026
)
. This makes the latent
representation itself an experimental axis.
Supported Latent Spaces.
Nano World Models
support three types of latent spaces: VAE
(
Rombach et al., 2022
)
, Web-DINO
(
Fan et al., 2025
)
and V-JEPA 2.1
(
Mur-Labadia et al., 2026
)
. VAE latents provide a reconstruction-oriented space that can be decoded back
into RGB frames, making them natural for video generation and perceptual evaluation. DINO features provide a self-supervised representation space that emphasizes semantic and geometric information useful for downstream prediction
and planning. V-JEPA features provide a video-pretrained representation space designed around predictive visual features. Supporting these latent spaces under the same training interface allows
Nano World Models
to compare reconstruction-oriented and representation-oriented
world modeling without changing the rest of the pipeline.
3.5
Datasets and Environment Interface
Nano World Models
uses a shared dataset and environment interface for diverse sources of sequential data. Each dataset exposes observation sequences, optional action sequences, frame windows, and metadata through the same loader
abstraction. This allows simulation environments
(
Tassa et al., 2018
;
Fu et al., 2020
)
, gaming datasets
(
Pearce and Zhu, 2022
)
and robot manipulation datasets
(
Brohan et al., 2022
)
to share the same
modeling code.
Figure 2:
Qualitative rollouts across domains.
Representative
ground-truth (GT) sequences and
Nano World Models
rollouts from Point Maze, Wall, Rope,
Granular, PushT, and RT-1. The same dataset and environment interface
exposes these domains to the training and sampling code, allowing
grid-world navigation, simulated control, and robot-video prediction to be
compared under a shared rollout format.
3.6
Long-Horizon Rollouts
Although models are trained on finite windows, diffusion forcing naturally
supports generation beyond the training horizon.
Nano World Models
enables long-horizon generation via sliding window and auto-regressive generation: generated frames are treated as context for generation of new future frames, and the perceptive field for attention on the temporal axis follows a sliding window procedure. This enables temporally extended
rollouts while preserving the same local denoising interface used during short-horizon sampling.
3.7
Logging, Evaluation and Reproducibility
Figure 3:
Logging and fixed-seed evaluation.
Nano World Models
logs training
curves, validation metrics, and qualitative prediction panels through
Weights & Biases. The same callback-style evaluation pipeline records
PSNR, SSIM, LPIPS, FID, reconstruction videos, predicted rollouts, and
ground-truth clips under a shared run.
Nano World Models
includes fixed-seed validation, checkpointing, logging, and standardized evaluation scripts. For logging, we support both both Tensorboard
(
Abadi et al., 2016
)
and Wandb
(
Biewald, 2020
)
logging systems. To further ensure reproducibility,
Nano World Models
open-sources all final checkpoints for supported environments, and ablated design choices.
3.8
World-Modeling as
Tool-Use
Figure 4:
Exporting rollouts to 3D scene assets.
A generated CSGO
rollout is decoded to RGB frames and passed to a downstream depth and camera
estimation pipeline. The resulting geometry can be visualized as a persistent
point cloud together with the source RGB frame and estimated depth map.
Exporting Video Rollouts to 3D Scene Assets.
To use generated futures as inputs to 3D reconstruction tools,
Nano World Models
provides a
rollout export interface that saves predicted frames together with the metadata
needed by downstream reconstruction pipelines. A generated trajectory is first
decoded into RGB frames at the model’s training resolution. When the source
domain has a different native aspect ratio, frames can be remapped to the native
resolution before reconstruction. The exported video or frame sequence can then
be passed to off-the-shelf multi-view depth and camera estimation systems
(
Lin et al., 2025
;
Chen et al., 2026
)
, whose
outputs are converted into persistent 3D representations such as point clouds.
This design keeps the world model independent of any particular 3D backend:
Nano World Models
supplies temporally coherent visual rollouts, while the reconstruction
module handles geometry estimation and visualization.
MPC Interface for Goal-Conditioned Planning.
For planning,
Nano World Models
exposes the world model as a batched rollout function. At
each replanning step, the planner receives the current observation context, a
goal specification, and a population of candidate action sequences. The world
model predicts a future trajectory for each candidate in parallel, and an
objective module assigns a scalar score to each rollout, such as distance to a
goal state, task progress, or environment-specific reward. The planner updates
the candidate distribution, selects the best sequence, executes its first action,
and then repeats the procedure with the newly observed context. This separates
the learned dynamics model from the planning algorithm and reward definition,
allowing the same checkpoint to be reused across different goal specifications
and trajectory optimizers.
