# BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving

paper_id: semanticscholar:770bf76926e528e00439005beec3e3a533c3c2f1
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Closed-loop planning with reactive agents is a critical challenge in autonomous driving, which re-
quires effective interaction with complex and dynamic traffic environments (Jia et al., 2024). Diffu-
sion models have become a powerful paradigm for this task due to their ability to model complex,
multi-modal distributions and incorporate flexible guidance (Liao et al., 2025; Zheng et al., 2025b;
Yang et al., 2024; Xing et al., 2025). A key challenge, however, is to determine which sources of
guidance information are most salient and how to integrate them effectively into these models to
produce plans that are not only plausible but also safe and reactive in real-world driving conditions.
A promising source for guidance is to leverage typical human expert driving behaviors, often repre-
sented as coarse anchor trajectories, as they provide a strong prior for safe and sensible maneuvers,
constraining the vast solution space. Recently, DiffusionDrive (Liao et al., 2025) implements this
strategy by training a denoiser on a truncated diffusion schedule, starting from a noisy version of the
anchor rather than pure Gaussian noise. While achieving state-of-the-art empirical performance, this

## Method

diffusion process that it is trained on, which diverges from the core principle of diffusion models
and can lead to unpredictable behaviors and compromised performance.
To address this, we introduce BridgeDrive, a principled diffusion framework that integrates anchor-
based guidance for autonomous driving planning using a theoretically sound diffusion bridge for-
mulation. Instead of heuristically truncating the diffusion process, we formally define the planning
task as learning a diffusion process that bridges the gap from a given coarse anchor trajectory to a
refined, context-aware final trajectory plan. This formulation ensures that the forward and denoising
processes are perfectly symmetric, allowing our model to learn a direct and robust transformation
* denotes equal contribution. Corresponding author: Shu Liu.
1
arXiv:2509.23589v4  [cs.AI]  5 Mar 2026

Published as a conference paper at ICLR 2026
from anchors to final trajectories. By adhering to the principles of diffusion, our method fully lever-
ages the expressive power of anchors for guidance while maintaining diffusion models’ ability to
represent diverse human-like driving behaviors. Furthermore, our approach is compatible with effi-
cient ODE-based samplers, enabling real-time performance crucial for on-road deployment. Empiri-
cally, we achieve 74.99% and 89.25% success rate on the Bench2Drive closed-loop benchmark with
PDM-Lite and LEAD datasets, respectively, outperforming previous SOTA by 7.72% and 2.45%.
2
PRELIMINARIES
2.1
AUTONOMOUS DRIVING PLANNING AND EVALUATION
The planning task in autonomous driving can be formulated as predicting future trajectories of
the ego-vehicle based on raw sensor inputs. Conventionally, there are two trajectory representa-
tions (Renz et al., 2025): (1) Temporal speed waypoints x := xtemp ∈RNpoint×2, represent equal
temporal-spaced (e.g., every 0.25 seconds) future coordinates of ego-vehicle, which inherently con-
tain speed control information. (2) Geometric path waypoints x := (xgeo, v) ∈RNpoint×2 × R,
represent equal geometric-spaced (e.g., every 1 meter) future coordinate of ego-vehicle; for geomet-
ric path waypoints-based planning, the model needs to predict the speed v of ego-vehicle. In this
paper, we choose to use geometric path waypoints as our model output, which differs from Diffu-
sionDrive Liao et al. (2025) where temporal speed waypoints are used. This design choice is based
on prior works (Chitta et al., 2023; Zimmerlin et al., 2024) and our ablation study in Section 4.
