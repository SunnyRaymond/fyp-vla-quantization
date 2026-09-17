# Grounding LLMs For Robot Task Planning Using Closed-loop State Feedback

paper_id: semanticscholar:0ebad08c67e9bb775bd0bd7a0c5acf7555679c2d
tier: T2
source_used: pdf_arxiv_pymupdf
warning: none

## Intro

LLMs, trained on corpora of internet-sourced text,
have demonstrated capabilities akin to artificial
general intelligence (Bubeck et al., 2023). The
inherent world knowledge of LLMs, combined
with their in-context learning ability is paving
a new direction in robotic task planning. Prior
work showed that LLMs can generate step-by-
step instructions for complex tasks without any
re-training or model parameter updates (Huang
et al., 2022, 2023; Sun et al., 2023; Liang et al.,
2023; Yao et al., 2023; Ahn et al., 2022; Singh
et al., 2023; Song et al., 2023; Kambhampati et al.,
2024). Despite promising results in diverse robotic
tasks, grounding LLMs in a given environment is
still an open problem. Consider the task - “Make
me a coffee.” LLMs can decompose this prob-
lem into a sequence of steps– ‘1. Walk to fridge,’
‘2. Grab milk,’ and so on, through to ‘9. Serve
1
arXiv:2402.08546v3  [cs.RO]  20 Nov 2025

cup of coffee.’ Yet, these steps are not entirely
executable in a real-world environment with phys-
ical constraints. Additional steps like ‘Switch on
microwave’ or ‘Open microwave doors,’ are essen-
tial for task completion. Moreover, limitations like
absence of milk or water should not hinder task
execution. Robots must adapt to the environment,
refining plans towards successful task completion.
Thus, grounding LLMs in real-world scenes
is essential. Incorporating environmental feed-
back into task planning allows error resolution
in real-time, enhancing robot robustness and
utility (Huang et al., 2023). In this paper, we
introduce a novel planning algorithm that aims
at mitigating two issues in existing methods: i)

## Method

work, with clear distinction of planning, feedback
and execution components to avoid using expert
defined heuristics and ii) Our method can inte-
grate feedback from simulators, controllers or
human intervention to guide an LLM planner for
robust task execution, enhancing autonomy. Our
contributions are:
1. A novel planning algorithm that uses a Two-
LLM
Agentic
framework
(Brain-LLM
and
Body-LLM) to derive executable actions from
natural language instructions, leveraging a
closed-loop state feedback mechanism for error
resolution (Figure 1).
2. Improving task-oriented success rate by 15%
average over existing state-of-the-art tech-
niques in the VirtualHome Embodied Control
environment (using a dataset of 80 tasks).
BrainBody-LLM on average completes 81% of
all goal conditions for a given task.
3. Deployment and testing of our LLM based
planner on the Franka Research 3 robotic arm,
in 7 tasks of varied difficulties using a real-
istic physics simulator along with real robot
experiments.
Simulator/
Controller
Feedback
& Error
Messages
“Eat chips on 
the sofa”
Walk to kitchen
Find the chips
Grab the chips
Walk to the living room
Find the sofa
Sit on the sofa
Eat the chips
High-level Plan
Brain-LLM
Real World
Knowledge
Body-LLM
<char0> [walk] <kitchen>
<char0> [find] <chips>
<char0> [grab] <chips> 
<char0> [walk] <livingroom>
<char0> [find] <sofa> 
<char0> [sit] <sofa>
<pass>
Low-level Plan
Fig. 1 Illustration of how two LLMs work together in the
proposed algorithm: The Brain-LLM splits the given task,
‘Eat chips on the sofa’, into sequential steps using its real-
world knowledge. The Body-LLM takes these steps one-by-
one and determines executable actions. In instances where
a corresponding action is not found in the environment, as
demonstrated in the final step of this example, the Body-
LLM outputs a <pass> token.
