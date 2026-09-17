# Feedback World Model Enables Precise Guidance of Diffusion Policy

paper_id: arxiv:2605.15705v1
tier: H
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

World models, which predict how environment states evolve under actions, have emerged as a promising tool
for improving the robustness and generalization of robotic policies (Sun et al., 2026a; Yuan et al., 2026; Ye
et al., 2026; Su et al., 2026; Hou et al., 2026). Existing approaches primarily incorporate world models in two
ways. One line integrates predicted dynamics, preferences, or rewards into policy training, using them as
auxiliary supervision or optimization signals (Sun et al., 2026a; Yuan et al., 2026; Sun et al., 2026b; Liu et al.,
2026). The other leverages world models at inference time to evaluate or plan over candidate actions, directly
influencing decision making during execution (Sun and Song, 2025; Yan et al., 2026). Across both paradigms,
the key idea is to move beyond pure behavior cloning by grounding policy learning and execution in predicted
environment dynamics, encouraging actions that lead to plausible and task-consistent future states (Quevedo
et al., 2025; Sun and Song, 2025).
However, these benefits become difficult to realize when policies are deployed beyond the training distribution.
In robotic manipulation tasks, even small variations in initial pose, object configuration, or visual observation
can shift the system into out-of-distribution (OOD) regimes. Under such shifts, the reliability of the world
model becomes critical: inaccurate predictions can directly mislead both policy learning and test-time
guidance (Quevedo et al., 2025). A common strategy to improve robustness is to scale up data or model
capacity, for example through large pretrained video models (Ye et al., 2026) or additional real-world rollouts
for finetuning (Sun and Song, 2025; Liu et al., 2026). While effective, this approach substantially increases
1
arXiv:2605.15705v1  [cs.RO]  15 May 2026

training cost and is often impractical in data-limited robotic settings (Ye et al., 2026; Yuan et al., 2026).
Notably, deployment itself provides an underexplored signal: after each action, the robot observes the true
next state, which directly exposes the mismatch between predicted and actual transitions. This real-time
feedback offers a natural opportunity to improve prediction reliability without additional data collection or
model scaling.
Despite this, existing methods largely treat world models as static predictors at inference time. While new
observations are incorporated for subsequent predictions, they are rarely used to correct the model’s internal
prediction state. As a result, prediction errors can persist and accumulate over time, especially in long-horizon
or OOD scenarios. This limitation points to a key missing capability: can a world model actively exploit real-time
feedback during inference to maintain reliable prediction under distribution shift, without requiring additional data
or larger models?
To this end, we propose a feedback world model that leverages real-time observations during policy inference
to correct future predictions online. Instead of operating as a static open-loop predictor, our model maintains
a lightweight feedback state that is updated after each environment interaction. The observed mismatch
between predicted and observed transitions is used to iteratively correct subsequent predictions, mitigating
error accumulation without requiring additional training data or parameter updates. This mechanism can be
interpreted as a latent-space observer and, under a linear feedback formulation, admits theoretical convergence
guarantees. Furthermore, we introduce action-aware guidance strategy for diffusion policy inference. Rather
than uniformly comparing predicted observations, we emphasize components that are more directly influenced
by the robot’s actions, such as end-effector motion and object pose. This focuses guidance on controllable,
task-relevant changes while reducing interference from action-irrelevant variations.
We thoroughly evaluate our method on four tasks from the LIBERO-10 task suite in the LIBERO-Plus
benchmark (Fei et al., 2025), three representative Robomimic tasks (Mandlekar et al., 2022), and two real-world
manipulation tasks, focusing on data-limited world model and OOD settings induced by robot initial-state
perturbations. Notably, the proposed feedback mechanism reduces world model prediction error by up to
76.4% under OOD conditions, without additional training data. Built on these improved predictions, the
downstream diffusion policy achieves the best overall performance, increasing the average success rate by 30%.
To summarize, our contributions are threefold: (1) We propose a feedback world model that incorporates
real-time observations as online corrective signals, enabling reliable prediction under distribution shift with
minimal overhead, and providing theoretical convergence guarantees. (2) We introduce an action-aware
guidance strategy that emphasizes controllable, action-relevant components to improve policy guidance. (3)
We validate the proposed approach on both simulated and real-world robotic tasks, demonstrating consistent
improvements in world model prediction accuracy and downstream policy performance.
2

## Method

predictions, compensating for model errors using real-time observations without additional training
data or parameter updates. We show that this process can be interpreted as a latent-space observer
and admits convergence guarantees under mild conditions. We further introduce action-aware guidance
to better translate corrected predictions into control by emphasizing action-controllable components
while suppressing irrelevant variations. Experiments on LIBERO-Plus, Robomimic, and real-world
manipulation tasks demonstrate that our method substantially improves both prediction accuracy
and policy performance under distribution shift. In particular, it reduces world model prediction error
by up to 76.4% and improves out-of-distribution (OOD) success rate by 30%. These results show that
incorporating real-time feedback at inference time provides a simple yet powerful alternative to static
world modeling.
Correspondence: jianfei.yang@ntu.edu.sg, antu0001@e.ntu.edu.sg
Project site: https://lorenzo-0-0.github.io/Feedback_World_Model/
1
