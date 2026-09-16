# GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation

paper_id: semanticscholar:935a8586ea05ae1d59efaca83f537025b6111822
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

Training manipulation policies typically relies on three
data sources—simulation, human videos, and real teleopera-
tion—each presenting a distinct trade-off. While simulation
provides a perfectly aligned action space for the robot, it
often suffers from many sim-to-real gaps. Human videos
offer the benefit of photo-realistic scenes and real physics,
but lack temporally aligned robot actions and operate in a
mismatched action space. Teleoperation successfully aligns
both perception and actions, yet its high cost and difficulty
to scale are significant limitations.
To resolve these trade-offs, we introduce GSWorld, a
closed-loop, photo-realistic simulation suite that couples 3D
Gaussian Splatting (3DGS) with physics to narrow both
the visual and action-space gaps for manipulation. “Closed-
loop” here means the same environment can be used to
train, evaluate, diagnose failures, and relabel, enabling rapid
iteration: policies perceive photo-realistic renderings while
issuing controls in the robot’s native space, all inside a
arXiv:2510.20813v1  [cs.RO]  23 Oct 2025

simulator that mirrors real scenes closely enough to support
zero-shot transfer and efficient adaptation. We demonstrate
that GSWorld enables a range of downstream applications
and, in particular, effective sim-to-real transfer for imitation
learning, reinforcement learning, and DAgger-style data col-
lection.
This closed-loop capability is powered by a bidirectional
pipeline that ensures tight alignment between the physical
world and its digital twin. In the real-to-sim direction, our
pipeline reconstructs a metric-accurate digital twin from
short multi-view captures, sets an absolute scale with ArUco
markers, and aligns the robot URDF to the scene via surface
fitting (e.g., ICP). We then attach collision meshes and ma-
terial properties to produce a versatile GSDF asset. The sim-
to-real direction is the reverse: policies trained in GSWorld
deploy on hardware without interface translation because
their control and observation spaces match the robot’s native
APIs. Policies trained with both sim and real data can
be deployed in the sim to evaluate, detect failures, and
gather DAgger corrections, thus closing the iteration loop.
Achieving this sim-to-real alignment requires that the scene’s
geometry, camera properties, and action semantics remain
consistent across the real-to-sim divide. We measure the
degree of this correspondence through a suite of metrics
evaluating the visual, geometric, and functional similarity
between the two worlds.
With photo-realistic perception and native action-space
control in one loop, GSWorld supports zero-shot sim-to-
real for both visual imitation learning and visual RL, while
exploiting scalable parallelism to accelerate data genera-
tion and training. Its closed-loop DAgger workflow lets
practitioners reproduce on-robot failures inside the digital
twin, step through them frame-by-frame, and collect targeted
corrective labels with far less teleoperation overhead. Finally,
GSWorld provides reproducible visual benchmarking: shared
GSDF assets, fixed camera intrinsics/extrinsics, consistent
lighting/materials, and standardized action semantics en-
able apples-to-apples comparisons across robots, scenes, and
tasks—so improvements reflect algorithmic progress rather
than environmental variance.
Existing GS-based simulators either target single-setup
photo-realistic rendering, provide engine-tied pipelines with-
out a portable asset standard, or limit the reproducible
cross-embodiment benchmarking and deployment-oriented
on-policy data collection [5, 27, 38, 45]. While GSWorld
delivers an effective real-to-sim-to-real workflow that unifies
photo-realistic 3DGS with contact-accurate physics, enabling
scalable cross-embodiment benchmarking, zero-shot imita-
tion and reinforcement learning, and automated high-quality
DAgger data collection for continual deployment-time im-
provement.
In summary, our contributions are:
• A solid real-to-sim-to-real pipeline. Our robust real-to-sim-
to-real pipeline accurately aligns the simulation with the
real environment, enabling a wide range of subsequent
applications.
• Simulation Data Collection and Visual Imitation Learn-
ing (IL). GSWorld supports multiple sim data collection

## Method

trained with GSWorld data can be directly deployed to the
reconstructed real-world scenes.
• Visual RL. GSWorld is designed to utilize parallel environ-
ments in the simulation to train RL policies. We provide
an analysis to show that GSWorld reduces RL sim2real
visual gaps.
• Closed-loop DAgger Learning with Visual Benchmarking.
GSWorld shows reliable policy evaluation results in corre-
lation with real-world deployments, contributing to using
DAgger to iteratively improve real-world policies.
