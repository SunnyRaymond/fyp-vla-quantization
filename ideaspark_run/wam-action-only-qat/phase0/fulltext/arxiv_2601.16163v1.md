# Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning

paper_id: arxiv:2601.16163v1
tier: U
source_used: html_arxiv
warning: none

## Intro

†
†
footnotetext:
Correspondence to: Moo Jin Kim
<moojink@cs.stanford.edu>
.
Large pretrained video generation models have shown impressive ability to generate physically plausible and temporally coherent videos
(
NVIDIA et al., 2025
;
Wan et al., 2025
;
Yang et al., 2024
;
Bao et al., 2024
;
Kong et al., 2024
;
Zheng et al., 2024
)
. Unlike pretrained vision-language models—which learn semantic concepts from static image-text pairs and have been popularized as robot policy backbones by recent vision-language-action (VLA) model research (
(
Brohan et al., 2023
;
Kim et al., 2024
;
Intelligence et al., 2025
;
Li et al., 2025b
)
)—pretrained video generation models learn temporal causality,
implicit physics, and motion patterns from millions of videos.
These spatiotemporal priors hold significant value for robotics applications. In this work, we explore how to effectively leverage video models for robotic control and how they can incorporate policy rollout data to refine their world models and enable more effective planning.
Prior works have made significant progress on adapting video models for robotic manipulation, leveraging both robot action data and “action-less” Internet video data to train generalizable policies and perform new tasks with small amounts of demonstrations
(
Liang et al., 2025
;
Zhong et al., 2025
;
Hu et al., 2024
;
Liao et al., 2025
;
Unitree, 2025
;
Feng et al., 2025
;
Yang et al., 2025
;
Wang et al., 2025
)
. However, these works often require multiple training stages (e.g., video fine-tuning followed by action module training) and introduce new architectural components, such as separate action diffusers or inverse dynamics models. Other works avoid these complexities by training unified video-action models
(
Li et al., 2025a
;
Zhu et al., 2025
)
, but they do not leverage pretrained video models due to their custom design, limiting their ability to capitalize on the spatiotemporal priors.
In this work, we address these limitations with
Cosmos Policy
: an effective robot policy that is adapted from a pretrained video model (Cosmos-Predict2-2B
(
NVIDIA et al., 2025
)
) through a single stage of post-training on robot demonstrations. Unlike prior works which carefully design separate action modules and algorithms, Cosmos Policy makes no architectural modifications and instead leverages the pretrained model’s core learning mechanism to capture action distributions. Since video models are effective at modeling complex, high-dimensional, multimodal distributions and can generate temporally coherent videos with hundreds of frames, we hypothesize that their learning algorithms are well-suited for representing actions alongside other modalities. Following this reasoning, we directly fine-tune a video model to simultaneously generate robot actions, future state images, and future state values (expected total cumulative rewards), all of which we encode as latent frames within the model’s latent diffusion sequence. With future state and value predictions, Cosmos Policy can use best-of-N sampling to plan by generating candidate actions, imagining their resulting future states, ranking these states by predicted value, and executing the highest-value action. This search process produces trajectories that are more likely to succeed at the task
Our main contribution is the
Cosmos Policy approach
for fine-tuning pretrained video models to incorporate different modalities that enable
visuomotor control and planning. We evaluate our method in two modes: first as a direct policy (without planning) and then with model-based planning using the future state and value predictions. As a direct policy, Cosmos Policy achieves a new state of the art in both the LIBERO and RoboCasa simulation benchmarks (98.5% and 67.1% average success rates, respectively), outperforming diffusion-based policies trained from scratch, video-based policies (e.g., UVA, Video Policy), and even fine-tuned VLAs (e.g.,
π
0.5
\pi_{0.5}
, OpenVLA-OFT, CogVLA, UniVLA, DP-VLA, GR00T-N1.5).
It also achieves the highest average success rate (93.6%) among state-of-the-art policies in challenging real-world bimanual manipulation tasks. Further, when enhanced with model-based planning,
we observe
a 12.5 percent higher task completion rate on average in two challenging real-world manipulation tasks. In these experiments, we show that Cosmos Policy can incorporate past experiences from policy rollouts to refine its world model and value function and plan more effectively. Lastly, we compare our model-based planning approach to a model-free variant and study their relative advantages.

## Method

In this section, we discuss how to adapt Cosmos-Predict2 into a unified model that predicts actions, future states, and values. We also discuss leveraging policy rollout data to enable effective planning.
4.1
Latent Frame Injection: Incorporating New Modalities
The original Cosmos-Predict2 model takes as input an image and a textual description to generate a short video for a single camera view. It does not support robot proprioception as input, robot actions or state values as output, nor multiple camera views—all of which are desired or required for manipulation policies.
Rather than designing new model components or making architectural modifications as done in prior works, we propose to encode additional modalities as new latent frames that are directly injected into the video model’s latent diffusion sequence. Given a
(
1
+
T
′
)
×
H
′
×
W
′
×
16
(1+T^{\prime})\times H^{\prime}\times W^{\prime}\times 16
sequence of latent frames, which originally correspond to images in a video, we interleave new modalities (robot state, action chunk, and state values) and images from additional camera views by inserting new latent frames between existing image latent frames. For multiple camera viewpoints, the process is simpler: we simply insert the additional camera images at the image sequence level as shown in the top row of Figure
2
) (and the model is subsequently fine-tuned to handle these additional viewpoints).
We now discuss an illustrative example of latent injection for incorporating new modalities. For a robotic platform with two static third-person cameras and a wrist-mounted camera, our latent sequence contains 11 latent frames: (1) a blank placeholder,
*
*
*
This is an implementation detail that is necessary due to the VAE’s
(
1
+
T
4
)
(1+\frac{T}{4})
temporal compression scheme discussed in Section
3
, which is commonly used in models such as Cosmos-Predict2 and Wan2.1. See Appendix
A.1
for details.
(2) robot proprioception (e.g., end-effector pose or joint angles), (3) wrist camera image, (4) first third-person camera image, (5) second third-person camera image, (6) action chunk, (7) future robot proprioception, (8) future wrist camera image, (9) future first third-person camera image, (10) future second third-person camera image, and (11) future state value. Among these, (2), (6), (7), and (11) represent new modalities while (3), (5), (8), and (10) represent additional camera views (assuming that the first third-person camera is the “primary” camera). To encode the new modalities as latent frames, we fill each
H
′
×
W
′
×
C
′
H^{\prime}\times W^{\prime}\times C^{\prime}
latent volume with normalized and duplicated copies of the robot proprioception, action chunk, or value (where normalization simply consists of rescaling to
[
−
1
,
+
1
]
[-1,+1]
). See Figure
2
for an illustration. This ordering of modalities in the sequence represents
(
s
,
a
,
s
′
,
V
⁡
(
s
′
)
)
(s,a,s^{\prime},V(s^{\prime}))
, and it allows for autoregressive decoding of actions, future state, and future state value from left to right (see Section
4.2
for further discussions on this).
Note that
s
s
and
s
′
s^{\prime}
only consist of the observations at time
t
t
and
t
+
K
t+K
, respectively, where
K
K
is the action chunk size. In other words, we do not use input history nor predict future frames across multiple subsequent timesteps. Lastly, latent injection is flexible and can be adapted for any particular robot setup: for example, for a robot with only one third-person camera, one can simply remove the latent frames corresponding to additional camera viewpoints, and this would result in only seven total latent frames.
See Figure
8
for a more detailed version of Figure
2
. Implementation details for latent injection are provided in Appendix
A.1
.
4.2
Joint Training of Policy, World Model, & Value Function
Implementing joint training objectives.
Now that we have a latent diffusion scheme that incorporates additional modalities and camera views that are compatible with robotic policy learning, we can adapt the video model into a policy by training on robot data. For each training step, we sample a batch of
(
s
,
a
,
s
′
,
V
⁡
(
s
′
)
)
(s,a,s^{\prime},V(s^{\prime}))
tuples.
†
†
†
We technically sample the empirical return
G
t
=
γ
H
−
t
​
R
​
(
s
H
,
a
H
)
G_{t}=\gamma^{H-t}R(s_{H},a_{H})
but write “
V
⁡
(
s
′
)
V(s^{\prime})
” for notational simplicity and readability.
50 percent of the batch is sampled from the
demonstrations dataset
and is used to train the policy (
p
⁡
(
a
,
s
′
,
V
⁡
(
s
′
)
|
s
)
p(a,s^{\prime},V(s^{\prime})|s)
), while the other 50 percent is sampled from the
rollouts dataset
and is split into two halves: one half for training the world model (
p
(
s
′
,
V
(
s
′
)
|
s
,
a
)
p(s^{\prime},V(s^{\prime})|s,a)
) and the other half for training the value function (
p
⁡
(
V
⁡
(
s
′
)
|
s
,
a
,
s
′
)
p(V(s^{\prime})|s,a,s^{\prime})
). The conditioning scheme—i.e., which part of the latent diffusion sequence is used as conditioning and which part is used as the target to generate—determines which of these three functions is being trained (see Figure
12
for more details). Initially, the rollouts dataset is simply a superset of the demonstrations dataset that also includes failed demonstrations, if they exist. (Failed demonstrations are those that do not successfully complete the task when replayed in the environment due to human error during data collection, e.g., in the LIBERO and RoboCasa simulations where roughly 10 to 20 percent of demonstrations fail when replayed. In certain environments where teleoperation data is collected more carefully, such as our real-world ALOHA environment, failed demonstrations do not exist; in this case, the demonstrations dataset and rollouts dataset are equal.)
Note that policy and world model training involves auxiliary targets, i.e., the policy is trained to model not just
p
⁡
(
a
|
s
)
p(a|s)
but rather
p
⁡
(
a
,
s
′
,
V
⁡
(
s
′
)
|
s
)
p(a,s^{\prime},V(s^{\prime})|s)
, and the world model learns not just
p
⁡
(
s
′
|
s
,
a
)
p(s^{\prime}|s,a)
but rather
p
(
s
′
,
V
(
s
′
)
|
s
,
a
)
p(s^{\prime},V(s^{\prime})|s,a)
. We find in Section
5.2
that the auxiliary supervision improves policy performance. Also, note that the
V
⁡
(
s
′
)
V(s^{\prime})
predictions are conditioned on the full latent prefix (i.e., all of
(
s
,
a
,
s
′
)
(s,a,s^{\prime})
) during initial Cosmos Policy training. However, when we later fine-tune this base checkpoint on policy rollout data to produce a model with more accurate future state and value predictions, we can choose to condition the value generation on a subset of
(
s
,
a
,
s
′
)
(s,a,s^{\prime})
via input masking. The choice of the input mask determines whether the value function represents the state value
V
⁡
(
s
′
)
V(s^{\prime})
or state-action value
Q
⁡
(
s
,
a
)
Q(s,a)
; we compare these variations in planning experiments (Section
5.3
).
Parallel vs. autoregressive decoding.
Since Cosmos Policy learns to both jointly and conditionally predict the targets
(
a
,
s
′
,
V
⁡
(
s
′
)
)
(a,s^{\prime},V(s^{\prime}))
based on apportioned training samples, it can generate actions, future states, and values either jointly in parallel or autoregressively from left to right. Parallel decoding offers greater speed, while autoregressive decoding may provide higher-quality predictions and allow for separate checkpoints to be used for the policy versus the world model and value function. For direct policy evaluation without planning, only the actions are required for task execution, while the latter two outputs can be discarded. Therefore, we use parallel decoding in this case. For evaluations with planning, we enable autoregressive decoding for higher-quality future state and value predictions.
4.3
Planning with Cosmos Policy’s World Model and Value Function
Cosmos Policy can be deployed as (1) a direct policy without planning or (2) a planning policy using future state and value predictions to search for higher-quality actions. However, training on demonstrations alone is insufficient for effective planning since the data only covers successful outcomes,
‡
‡
‡
Some demonstration datasets, including the LIBERO simulation benchmark training set, includes suboptimal behaviors that may not lead to success when replayed in the training environment, due to human errors during teleoperation. Typically, however, most (if not all) trajectories in a demonstration dataset used for imitation learning are successful.
which means that the world model and value function see a narrow state-action distribution and may struggle to generalize beyond that distribution. We thus find it critical to collect policy rollout data and learn from these experiences.
Learning from rollout experiences.
We collect rollout data by deploying Cosmos Policy in diverse initial conditions and recording the trajectory as well as the episode outcome (success/fail or a fractional score). Given the rollout dataset, we fine-tune our Cosmos Policy checkpoint, with heavier weighting on the world model and value function predictions: 90 percent of each training batch is split evenly between training the world model and value function, while only 10 percent is used to train the policy.
Once we have the fine-tuned checkpoint for refined world modeling and policy learning, we propose
dual deployment
: the original Cosmos Policy checkpoint serves as the policy (we thus call it the “policy model”), while the refined checkpoint serves as the world model and value function (we thus call it the “planning model”). This ensures that the refined world model and value function are trained on on-policy data collected by the original policy.
Model-based planning.
Given the policy model and the planning model, we implement best-of-N sampling as follows: (1) sample multiple action proposals from the policy, (2) use the planning model to predict the future state and value for each proposal, (3) select and deploy the action that leads to the predicted state with the highest predicted value.
For greater accuracy and better modeling of potentially multimodal future state and value distributions, we ensemble the predictions by querying the world model three times per action and the value function five times per future state, resulting in fifteen total value predictions for each action proposal. We aggregate these via “majority mean”: we determine whether the majority predict success or failure (via a fixed threshold) and then average values within the majority group. This approach is more robust to outliers than naive averaging when value predictions are bimodal or exhibit high variance.
To speed up the search process, we use parallelized inference, using N GPUs in best-of-N sampling. We also execute the full action chunk (rather only part of it, as done in receding-horizon control) to avoid further increases in computational cost.
Figure 3:
Cosmos Policy in the ALOHA robot tasks.
Cosmos Policy can successfully execute real-world robotic control tasks that require long-horizon, high-precision manipulation and have high action multimodality.
