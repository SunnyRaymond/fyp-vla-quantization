# DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

paper_id: arxiv:2411.04983v2
tier: U
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Robotics and embodied AI have seen tremendous progress
in recent years. Advances in imitation learning and rein-
forcement learning have enabled agents to learn complex
behaviors across diverse tasks (Agarwal et al., 2022; Zhao
et al., 2023; Lee et al., 2024; Ma et al., 2024; Hafner et al.,
2024; Hansen et al., 2024; Haldar et al., 2024; Jia et al.,
1Courant Institute, New York University 2Meta AI. Correspon-
dence to: Gaoyue Zhou <gz2123@nyu.edu>.
2024). Despite this progress, generalization remains a ma-
jor challenge (Zhou et al., 2023). Existing approaches pre-
dominantly rely on policies that, once trained, operate in
a feed-forward manner during deployment—mapping ob-
servations to actions without any further optimization or
reasoning. Under this framework, successful generalization
inherently requires agents to possess solutions to all possi-
ble tasks and scenarios once training is complete, which is
only possible if the agent has seen similar scenarios during
training (Reed et al., 2022; Brohan et al., 2023b;a; Etukuru
et al., 2024). However, it is neither feasible nor efficient to
learn solutions for all potential tasks and environments in
advance.
Instead of learning the solutions to all possible tasks dur-
ing training, an alternate is to fit a dynamics model on
training data and optimize task-specific behavior at runtime.
These dynamics models, also called world models (Ha &
Schmidhuber, 2018), have a long history in robotics and
control (Sutton, 1991; Todorov & Li, 2005; Williams et al.,
2017). More recently, several works have shown that world
models can be trained on raw sensory data (Hafner et al.,
2019; Micheli et al., 2023; Robine et al., 2023; Hansen
et al., 2024; Hafner et al., 2024). This enables flexible use
of model-based optimization to obtain policies as it circum-
vents the need for explicit state-estimation. Despite this,
significant challenges remains in its use for solving general-
purpose tasks.
To understand the challenges in world modeling, let us con-
sider the two broad paradigms in learning world models:
online and offline. In the online setting, access to the envi-
ronment is often required so data can be continuously col-
lected to improve the world model, which in turn improves
the policy and the subsequent data collection. However,
the online world model is only accurate in the cover of the
policy that was being optimized. Hence, while it can be
used to train powerful task-specific policies, it requires re-
training for every new task even in the same environment.
Instead, in the offline setting, the world model is trained
on an offline dataset of collected trajectories in the environ-
ment, which removes its dependence on the task specificity
given sufficient coverage in the dataset. However, when
required to solve a task, methods in this domain require
1
arXiv:2411.04983v2  [cs.RO]  1 Feb 2025

DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning
strong auxiliary information which can take the form of ex-
pert demonstrations (Pathak et al., 2018; Wang et al., 2023),
structured keypoints (Ko et al., 2023; Wen et al., 2024), ac-
cess to pretrained inverse models (Du et al., 2023; Ko et al.,
2023) or dense reward functions (Ding et al., 2024), all of
which reduce the generality of using offline world models.
The central question to building better offline world models
is if there is alternate auxiliary information that does not
compromise its generality?
In this work, we present DINO-WM, a new and simple

## Method

dataset of trajectories. DINO-WM models the world dy-
namics on compact embeddings of the world, rather than the
raw observations themselves. For the embedding, we use
pretrained patch-features from the DINOv2 model, which
provides both a spatial and object-centric representation
prior. We conjecture that this pretrained representation en-
ables robust and consistent world modeling, which relaxes
the necessity for task-specific data coverage. Given these
visual embeddings and actions, DINO-WM uses the ViT
architecture to predict future embeddings. Once this model
is trained on the offline dataset, planning to solve tasks is
constructed as visual goal reaching, i.e. to reach a future
desired goal given the current observation. Since the pre-
dictions by DINO-WM are high quality (see Figure 4),
we can simply use model predictive control with inference-
time optimization to reach desired goals without any extra
information during testing.
DINO-WM is experimentally evaluated on six environ-
ment suites spanning maze navigation, sliding manipulation,
robotic arm control, and deformable object manipulation
tasks. Our experiments reveal the following findings:
• DINO-WM produce high-quality future world modeling
that can be measured by improved visual reconstruction
from trained decoders. On LPIPS metrics for our hardest
tasks, this improves upon prior state-of-the-art work by
56% (See Section 4.7).
• Given the latent world models trained using DINO-WM,
we show high success for reaching arbitrary goals on
our hardest tasks, improving upon prior work by 45% on
average (See Section 4.3).
• DINO-WM can be trained across environment variations
within a task family (e.g. different maze layouts for nav-
igation or different object shapes for manipulation) and
achieve higher rates of success compared to prior work
(See Section 4.5).
Code and models for DINO-WM are open-sourced to
ensure reproducibility and videos of planning are made
available on our anonymous project website: https:
//dino-wm.github.io.
