# InternW0: A Foundational Physical World Model for Efficient Real-World Interactions

paper_id: arxiv:2609.27656v1
tier: T3
source_used: pdf_arxiv_pymupdf
warning: none

## Intro



## Method

enabling the agent to translate its understanding of the environment and the task into informed de-
cisions (Ha and Schmidhuber, 2018; Hafner et al., 2025).
Recent advances in physical world modeling have increasingly focused on world action models
(WAMs), which couple future visual prediction with robot action generation and provide a promis-
ing way to ground predictive dynamics in executable interaction (Wang et al., 2026a; Yuan et al.,
2026; Bi et al., 2026; Ye et al., 2026). A unified WAM is expected to support heterogeneous em-
bodiments and richer physical modalities such as proprioception, force, and tactile feedback, while
avoiding the latency introduced by synchronously updating expensive video prediction and fast
action generation. However, these remain insufficiently addressed by existing WAMs.
In this report, we develop the InternW world model series to provide this connection through a
shared model of perception, physical dynamics, and action. Following the resource-constrained
perspective from Shanghai AI Laboratory (Chen et al., 2026c), a physical world model is viewed as
a compact approximation of physical state transitions under finite sensing and computational re-
sources. This perspective motivates three requirements of world modeling: omnimodal perception
and interaction, asynchronous processing at multiple frequencies, and local environment modeling
under external influences.
• Omnimodal interfaces: InternW aims to unify diverse modality inputs and predict actions
within a shared physical representation, enabling embodied agents to perceive, reason, and
interact with a wide range of physical environments.
• Asynchronous and multi-frequency signal processing: InternW accommodates different
sensing and control frequencies to balance prediction quality, responsiveness, and compu-
tational cost.
• Local environment and action modeling: InternW estimates local physical state from partial
observations and predicts the effects of both agent actions and external disturbances, updating
its estimates as new evidence arrives.
Shared Representation and Specialized Experts.
These requirements motivate the InternW ar-
chitecture described here at the series level. We draw on the modality-specific parameterization
principle of mixture-of-transformers (MoT) (Liang et al., 2025). Our series-level design adapts this
principle to modalities (or data channels) for language semantics, visual dynamics, spatial geome-
try, contact mechanics, and body actions. Modality-specific processing thus supports joint reason-
ing about task intent, object configuration, contact conditions, and executable motion. The cross-
modality interfaces are extensible, allowing individual InternW models to select the modalities (or
data channels) and computational capacity appropriate for their deployment setting.
Latent Dynamics for Prediction and Action.
At the center of the framework, latent dynamics
connect the estimated physical state to possible future states and interactions. Learning latent
dynamics for planning from visual observations has been explored in prior model-based control
work (Hafner et al., 2019). To express this role conceptually, let Ht contain the timestamped obser-
vations and executed actions available up to time t, and let zt denote a compact representation of
local state and its uncertainty. We write
zt = Fθ(Ht),
pθ(zt+∆| zt, ut:t+∆, ∆),
(1)
2

where ut:t+∆denotes a candidate action sequence and ∆is the prediction horizon. Unobserved
external influences contribute to the uncertainty of the transition distribution. This formulation
specifies the modeling role rather than prescribing a particular probabilistic implementation. Pre-
diction interfaces translate the latent representation into future observations or task-relevant states,
while action interfaces generate commands conditioned on the task and available physical informa-
tion. Their shared representation connects what the agent expects to happen with what it chooses
to do. Under finite computational resources, we seek latent predictions that preserve physical in-
formation relevant to task outcomes and action choices, consistent with control-centric world mod-
eling (Hansen et al., 2024). In the InternW framework, we additionally require predictive informa-
tion to become available in time to influence ongoing execution. We therefore regard predictive
accuracy, decision relevance, and timeliness as three design criteria for our framework.
Duplex Interaction and Closed-Loop Feedback.
InternW couples perception, prediction, and ac-
tion through duplex interaction: it continues to receive new observations while generating predic-
tions and taking actions. Duplex interaction describes this concurrent input–output flow, whereas
asynchronous, multi-frequency processing determines when each sensor stream and computa-
tional component is updated. Together, these mechanisms allow visual, contact, and body-state
feedback to refine subsequent outputs while an operation is in progress. Each executed action
produces new observations that update the local physical representation and inform the next
prediction–action cycle. In scientific workflows, this closed loop connects experimental operations
to their measured outcomes and supplies evidence for subsequent model refinement. Correct oper-
ation requires timestamp alignment and causal consistency: each update must account for actions
that have already been executed while remaining able to revise commands that have not yet been
issued.
Environment
Geometry
Video
Contact
Semantic
Action
InternW
Series
Shared Physical
Representation
Task Planning
Place the egg
in the plate.
Locate
Grasp
Place
Video Prediction
3D Prediction
Contact Prediction
Action Generation
Future Physical States
Predict the consequences of actions in the physical world
Multimodal Observations
Observe and understand the current physical world
Task
“ Make me 
a breakfast.”
Visual
3D
Contact
State
Figure 1: Framework of InternW world model series.
Multimodal Feedback for Contact-Aware
Manipulation.
InternW is designed to
incorporate force, tactile, and proprio-
ceptive feedback into action generation
alongside visual predictive context. These
physical modalities provide complemen-
tary information about contact and the
robot’s physical state, including changes
that may not be directly observable in im-
ages (Lee et al., 2019; Chen et al., 2023;
Song et al., 2026; Yu et al., 2026).
Be-
cause these signals arrive at different tem-
poral resolutions, the model separates
long-horizon predictive context from re-
cent feedback used to update each ac-
tion.
This enables local action correc-
tion at every feedback step without re-
quiring the full predictive representation
to be recomputed. Specifically, InternW0
fuses the temporal history of interaction
forces with visual and proprioceptive ob-
servations, while jointly predicting the
end-effector poses and six-dimensional in-
teraction wrenches.
Each wrench com-
prises three force and three moment com-
ponents (Lynch and Park, 2017). The his-
torical force inputs encode observed con-
tact, whereas the predicted wrenches represent anticipated interaction loads. Together, they pro-
vide information about both past contact and anticipated loads to support contact-aware action
3

selection. Active hybrid position–force interaction is a core manipulation capability that coordi-
nates motion with contact-force regulation (Raibert and Craig, 1981; Li et al., 2026c).
InternW0 as an Initial Instantiation.
Building on the above design principles, InternW0 con-
nects physical prediction and responsive interaction through a mixture-of-transformers (MoT) ar-
chitecture. Specialized video and action experts are jointly optimized through video–action flow
matching, coupling future visual dynamics with executable motion while preserving distinct pro-
cessing pathways. Within the action expert, force feedback is integrated with visual and propri-
oceptive observations, enabling changes in contact and body state to condition subsequent action
updates. This fusion provides a direct pathway from real-time sensory feedback to local motion ad-
justment. To coordinate future prediction with real-time observation, an observation-conditioned
context-routing interface uses current visual features to query predictive video representations and
construct chunk-specific context for the action expert. The resulting duplex interaction allows
longer-horizon predictive context to guide execution while incoming observations continually in-
form subsequent actions, without requiring a new prediction cycle for every interaction update.
Domain-specific state and action encoders, action decoders, and soft prompts adapt this shared ar-
chitecture to heterogeneous datasets and embodiments. These mechanisms concretely realize the
framework’s multimodal interfaces, asynchronous processing, and observation-driven local up-
dates. Within the InkStone scientific discovery platform (InkStone, Shanghai AI Laboratory), our
system connects the scientific reasoning and tool-use capabilities of Intern-S2-Preview (Bai et al.,
2026) to physical experimentation through InternW0. Execution records and measured outcomes
can support subsequent evaluation and model improvement. The following sections detail the
architecture and training pipeline, data recipe, simulation benchmarks, real-robot and wet-lab ex-
periments, as well as the future work of InternW world model series.
2
InternW0 Architecture and Training Pipeline
2.1
Architecture
InternW0 instantiates the preceding framework with a mixture-of-transformers (MoT) backbone
comprising a video expert for future visual prediction and a lightweight action expert for robot con-
trol. Each joint layer contains modality-specific blocks with separate parameters and token streams,
connected through an observation-conditioned video-context interface. This structure allows the
two experts to use different capacities while coupling visual prediction with action generation.
The central design goal is to keep future predictions useful as new sensory signals arrive during
execution. This requires coordinating prediction and control at different timescales: repeatedly
recomputing a complete predictive plan at every control update is expensive, whereas executing
against an unchanged plan context risks conditioning the policy on an increasingly stale view of
the world.
InternW0 addresses this tension through asynchronous duplex inference, in which predictive video
modeling and action generation operate at different temporal scales while remaining coupled
through an observation-conditioned context interface. The heterogeneous pretraining stage learns
this coupling from visual observations, proprioceptive states, and robot actions. For contact-rich
downstream tasks, the action interface is further extended during post-training with force and tac-
tile observations and joint prediction of future interaction signals. Embodiment-specific interfaces
and soft prompts support transfer across heterogeneous control spaces and sensing configurations.
The interaction is intentionally asymmetric. The video expert builds a shared predictive repre-
sentation from visual observations and language without consuming robot-specific action tokens
or domain identities. The action expert, in turn, reads the video context together with the lat-
est observation, proprioception, language conditioning, and embodiment-specific interfaces. This
separation keeps physical prediction broadly shared while allowing control to specialize across
heterogeneous embodiments.
4

IDM-mask
f0
f1
a1
f0
f1
a1
Video DiT
Learn from vast video data
Action DiT
Learn action dynamics across domains
Ego & Internet
Videos
No action labels required
Heterogeneous
Robot Data
State–action supervision
Required
Optional
(Posttrain)
Context
routing
Vision
Encoder
Prompts
Force
Action
Actions  [Force]
…
VAE Decoder
Cross Attention & FFN
Self Attention
AdaLN & Q K V
Wan VAE Encoder
Action Decoder
Cross Attention & FFN
Joint Attention
AdaLN & Q K V
Soft prompt
Action encoder
Figure 2: Architecture of InternW0.
Modality-specific video and action experts are coupled
through an observation-conditioned chunk K/V editor. The editor adapts cached video context to
chunk-level observations, allowing it to be reused across action updates. A frozen Wan VAE and
DINOv3 encoder provide visual representations, while precomputed language embeddings con-
dition both experts. Domain-specific interfaces and soft prompts support heterogeneous embodi-
ments; force and tactile channels are introduced optionally during contact-aware post-training.
As shown in Figure 2, video frames are encoded by a frozen Wan VAE (Wan et al., 2025), while
current chunk-level visual observations are encoded by a frozen DINOv3 encoder (Siméoni et al.,
2026). Language instructions are represented by precomputed text embeddings. Trainable projec-
tions connect these representations to the corresponding experts, and proprioceptive states provide
chunk-specific conditioning for action generation. During post-training on contact-rich tasks, force
and tactile histories can be introduced as additional action-side observations without modifying
the shared video-prediction backbone.
Asynchronous Duplex Inference.
Future visual prediction provides anticipatory context, but its
computational cost makes regenerating a plan for every control update impractical. Meanwhile,
action generation must incorporate observations that arrive after a plan was produced. InternW0
separates these timescales: the video expert updates its prediction on a slower schedule, while the
action expert generates short chunks from the latest available plan and current sensory feedback.
After an initial plan is generated, action inference can continue while the next video prediction is
computed.
This coordination is implemented through a cached video context and a layerwise chunk K/V editor.
Following the observation-guided video-context routing design of AHA-WAM (Cai et al., 2026a),
the latest chunk-level observation is used to adapt the cached predictive context before it is con-
sumed by the action expert. For each action chunk n, the chunk-aligned RGB observation is first
encoded by DINOv3 and projected into a visual context. A learned query encoder summarizes this
context into a small set of routing queries.
5

Let (Kℓ, Vℓ) denote the valid video context keys and values at layer ℓ. The editor first lets the
observation queries q retrieve task-relevant information from the predictive video representation,
Rℓ,n = Attn

Wq
ℓqn, Kℓ, Vℓ

.
(2)
It then routes the retrieved information back to the original video-token positions,
Dℓ,n = Attn (Kℓ, Rℓ,n, Rℓ,n) ,
(3)
from which a lightweight projection predicts residual corrections (∆Kℓ,n, ∆Vℓ,n). The chunk-specific
video context is
eKℓ,n = Kℓ+ σ(γℓ)∆Kℓ,n,
eVℓ,n = Vℓ+ σ(γℓ)∆Vℓ,n,
(4)
where γℓis a learned gate. The final residual projection is zero-initialized, so the editor begins as
an identity mapping and learns to introduce observation-dependent corrections progressively.
The underlying video context remains unchanged: each action chunk constructs its own edited
view from the same predictive context using its newly observed visual state. Consequently, the
model can preserve longer-horizon predictive structure while adapting its local control context as
execution deviates from the previously imagined future.
Soft Prompts for Cross-embodiment Pretraining.
Sharing interaction knowledge across robots
requires retaining the differences in their action semantics and sensing configurations. Inspired
by X-VLA’s design of embodiment-specific prompts for cross-embodiment learning (Zheng et al.,
2026), InternW0 associates each training domain d with learned soft prompts P(d), prepended to
every action chunk. Domains distinguish data sources and embodiment configurations, and need
not correspond one-to-one to robot morphologies. The prompts condition shared expert compu-
tation on these differences, while domain-specific input and output projections adapt action and
sensor representations to the shared token space. Per-dimension validity masks distinguish miss-
ing channels from valid zero values: unavailable inputs are zeroed before token processing and
excluded from the corresponding loss. Together, these interfaces allow joint training without re-
quiring identical control or sensing semantics across robots.
Mixed Supervision for Prediction and Control.
Action-free egocentric videos provide observa-
tions of physical interactions, while action-labeled robot trajectories connect those interactions to
executable control. InternW0 trains the same video expert on both sources, whereas robot trajecto-
ries additionally supervise the action expert. Real-robot and simulated trajectories follow the same
action-supervised pathway, while egocentric videos contribute predictive supervision without re-
quiring robot action annotations.
Both experts use continuous flow matching. For a clean target x0, representing either video latents
or a continuous action chunk, Gaussian noise ϵ, and normalized flow time τ ∈[0, 1], the noisy
input and target velocity are
xτ = (1 −τ)x0 + τϵ,
v∗= ϵ −x0.
(5)
Flow times are sampled using the shifted continuous flow-matching schedule, with one flow time
sampled for each video example and independent flow times for individual action chunks. Flow
times are sampled per video example and per action chunk. Video attention uses a first-frame-
causal mask: the observed first latent frame cannot attend to future video tokens, whereas future
latent frames can attend to the observed frame and interact bidirectionally with one another. The
first frame remains clean as a temporal anchor and is excluded from the video prediction loss.
The pretraining objective combines the video and action flow-matching losses, Lvideo and Ljoint,
respectively, over samples ξ drawn from the mixed training distribution Dmix:
Lpre = Eξ∼Dmix

λvLvideo(ξ) + λa Iact(ξ)Ljoint(ξ)

,
(6)
where λv and λa balance their contributions, and Iact indicates whether action supervision is avail-
able. Both terms are scheduler-weighted, masked squared velocity errors. Temporal padding, the
clean current video frame, and unavailable action dimensions are excluded from the corresponding
losses. Egocentric samples therefore contribute video supervision without requiring action labels.
6

IDM Conditioning.
During action-supervised training, InternW0 maintains a second video
stream that provides predictive conditioning for the action expert. This independent-denoising con-
dition stream shares all VideoDiT parameters with the primary video flow-matching stream but
uses an independently sampled noise realization and flow time.
For each training example, the condition stream is kept clean with probability 0.5; otherwise, it
is perturbed using an independently sampled shifted flow time. In both cases, the first latent
frame remains clean. The primary video stream is supervised by the video flow-matching ob-
jective, whereas the condition stream does not receive a separate reconstruction loss. Instead, its
layerwise K/V features are consumed by the observation-conditioned editor and subsequently by
the action expert.
Importantly, the condition K/V features are not detached from the video network. Gradients from
the action flow-matching objective therefore propagate through the K/V editor into the shared
video expert. Predictive representation learning is consequently shaped by both future-video mod-
eling and its usefulness for action generation, rather than by visual reconstruction alone.
At inference time, the ground-truth condition stream is replaced by a generated video context.
Its layerwise K/V features are cached and exposed to the action expert through the same context
interface.
Contact-Aware Post-training with Joint Action–Contact Prediction.
The hetero
