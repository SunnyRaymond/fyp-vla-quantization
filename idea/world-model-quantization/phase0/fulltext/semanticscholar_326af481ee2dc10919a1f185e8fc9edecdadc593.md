# Active Inference with Dynamic Planning and Information Gain in Continuous Space by Inferring Low-Dimensional Latent States

paper_id: semanticscholar:326af481ee2dc10919a1f185e8fc9edecdadc593
tier: T2
source_used: failed
warning: fetch failed across all paths; intro filled with abstract; method empty

## Intro

Active inference offers a unified framework in which agents can exhibit both goal-directed and epistemic behaviors. However, implementing policy search in high-dimensional continuous action spaces presents challenges in terms of scalability and stability. Our previously proposed model, T-GLean, addressed this issue by enabling efficient goal-directed planning through low-dimensional latent space search, further reduced by conditioning on prior habituated behavior. However, the lack of an epistemic term in minimizing expected free energy limited the agent’s ability to engage in information-seeking behavior that can be critical for attaining preferred outcomes. In this study, we present EFE-GLean, an extended version of T-GLean that overcomes this limitation by integrating epistemic value into the planning process. EFE-GLean generates goal-directed policies by inferring low-dimensional future posterior trajectories while maximizing expected information gain. Simulation experiments using an extended T-maze task—implemented in both discrete and continuous domains—demonstrate that the agent can successfully achieve its goals by exploiting hidden environmental information. Furthermore, we show that the agent is capable of adapting to abrupt environmental changes by dynamically revising plans through simultaneous minimization of past variational free energy and future expected free energy. Finally, analytical evaluations detail the underlying mechanisms and computational properties of the model.

## Method


